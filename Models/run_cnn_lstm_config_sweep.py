"""
Config sweep for the two-stage EOL CNN-LSTM.

Two sweep modes (choose with --mode):

  grid     (default) — cross product of dQ/dV VOLTAGE windows × input (cycle)
             windows. For every (cycle window, voltage window) cell run all
             configs over all seeds.

  voltage  — dQ/dV voltage-window ablation only (input/cycle window held fixed).

  cycles   — the original CYCLE-window sweep (6 cycle-index windows), using every
             dQ/dV window together.

Knobs varied per config:
  CNN  : cnn_channels (base), cnn_dropout
  LSTM : lstm_hidden_size, num_layers, rnn_dropout (auto when layers>1)
  Head : head_dropout
  Train: epochs, batch_size, lr, weight_decay, lambda_eol, loss, scheduler

The model is built by monkeypatching CNNLSTMEOLPredictor / train_model into
CNN_LSTM_EOL.py via runpy init_globals (config is baked into the patched classes,
not passed via env vars); the dQ/dV voltage-window selection still flows through
the EOL_DQDV_WINDOW env var, which the model reads in its __main__.

Seeds/flags come from .env (EOL_SEED_LIST, EOL_SHOW_PLOTS, EOL_SAVE_ARTIFACTS).

Usage:
    python run_cnn_lstm_config_sweep.py                             # grid (default)
    python run_cnn_lstm_config_sweep.py --mode voltage --windows V0 V4 all
    python run_cnn_lstm_config_sweep.py --mode grid --input-windows 3 --windows V5 all
    python run_cnn_lstm_config_sweep.py --mode cycles --input-windows 0 3
"""

import argparse
import json
import os
import re
import runpy
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from statistics import mean, pstdev


# ── path setup: repo root (for featureExtrcation) + Models dir (for the base
#    module import inside _run_one) importable regardless of launch style ──────
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = str(_HERE.parent)
for _p in (_REPO_ROOT, str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from featureExtrcation.processDatasets import DQDV_V_WINDOWS


# ── load .env ────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

_raw_seeds = os.environ.get("EOL_SEED_LIST", "17,42,89,123,256")
SEEDS = [int(s) for s in _raw_seeds.split(",") if s.strip().isdigit()]

# ── cycle windows (used by --mode cycles; fixed default for --mode voltage) ───
CYCLE_WINDOWS = [
    list(range(1, 51)),                                                          # W0: first 50
    list(range(1, 101)),                                                         # W1: first 100
    list(range(1, 251, 10)),                                                     # W2: 1-250 stride-10
    list(range(10, 41)) + list(range(180, 201)),                                 # W3: early+late (original)
    list(range(1, 51)) + list(range(100, 151)) + list(range(200, 251)),          # W4: three bands
    [1, 10, 50, 100, 150, 200, 250],                                             # W5: sparse landmarks
]
WINDOW_LABELS = [
    "W0: first 50 cycles",
    "W1: first 100 cycles",
    "W2: 1-250 stride-10 (25 pts)",
    "W3: early(10-40)+late(180-200) [original]",
    "W4: three bands 1-50,100-150,200-250",
    "W5: sparse landmarks [1,10,50,100,150,200,250]",
]

# Cycle window held fixed while sweeping dQ/dV voltage windows (--mode voltage).
_raw_cycles = os.environ.get("EOL_CYCLES_TO_USE", "")
FIXED_CYCLES = [int(c) for c in _raw_cycles.split(",") if c.strip().isdigit()] \
    or [1, 10, 50, 100, 150, 200, 250]

# ── CNN-LSTM configs ──────────────────────────────────────────────────────────
# cnn_channels : base CNN filter count; encoder doubles to cnn_channels*2
# lstm_hidden  : bidirectional LSTM hidden size (output = lstm_hidden*2)
# num_layers   : LSTM stacked layers (recurrent dropout=0.3 when >1)
# cnn_dropout  : dropout after each conv block
# head_dropout : dropout in the MLP prediction head
CONFIGS = {
    "A": {
        "label": "A — Baseline (cnn=64, lstm=128, layers=1)",
        "model": dict(cnn_channels=64,  lstm_hidden=128, num_layers=1, cnn_dropout=0.2, head_dropout=0.3),
        "train": dict(epochs=300, batch_size=4, lr=3e-4, weight_decay=5e-3, lambda_eol=1.5),
        "loss": "smooth_l1", "scheduler": "cosine",
    },
    "B": {
        "label": "B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2)",
        "model": dict(cnn_channels=128, lstm_hidden=128, num_layers=2, cnn_dropout=0.2, head_dropout=0.3),
        "train": dict(epochs=300, batch_size=8, lr=2e-4, weight_decay=5e-3, lambda_eol=1.5),
        "loss": "smooth_l1", "scheduler": "cosine",
    },
    "C": {
        "label": "C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1)",
        "model": dict(cnn_channels=32,  lstm_hidden=256, num_layers=1, cnn_dropout=0.1, head_dropout=0.2),
        "train": dict(epochs=300, batch_size=4, lr=3e-4, weight_decay=1e-2, lambda_eol=1.5),
        "loss": "smooth_l1", "scheduler": "cosine",
    },
    "D": {
        "label": "D — OneCycleLR (cnn=64, lstm=128, layers=2)",
        "model": dict(cnn_channels=64,  lstm_hidden=128, num_layers=2, cnn_dropout=0.2, head_dropout=0.3),
        "train": dict(epochs=300, batch_size=4, lr=5e-4, weight_decay=5e-3, lambda_eol=1.0),
        "loss": "smooth_l1", "scheduler": "onecycle",
    },
    "E": {
        "label": "E — L1 Loss + Small (cnn=32, lstm=64, layers=2)",
        "model": dict(cnn_channels=32,  lstm_hidden=64,  num_layers=2, cnn_dropout=0.3, head_dropout=0.4),
        "train": dict(epochs=300, batch_size=4, lr=1e-4, weight_decay=2e-2, lambda_eol=2.0),
        "loss": "l1", "scheduler": "cosine",
    },
}

METRICS = ("acc", "mae", "mae_pct", "rmse", "rmse_pct", "r2")
SPLITS = ("Train", "Val", "Test")


# ── window-run builders ───────────────────────────────────────────────────────
def _voltage_units(volt_keys=None):
    units = [(f"V{w}", str(w), f"{lo:g}-{hi:g}V") for w, (lo, hi) in enumerate(DQDV_V_WINDOWS)]
    units.append(("all", "all", "allV"))
    if volt_keys:
        units = [u for u in units if u[0] in set(volt_keys)]
    return units


def build_voltage_runs(volt_keys=None):
    runs = []
    for vkey, vwin, vlabel in _voltage_units(volt_keys):
        runs.append({
            "key": vkey, "label": f"{vkey}: {vlabel}",
            "cycles": FIXED_CYCLES, "dqdv_window": vwin,
            "input_key": "Wfix", "volt_key": vkey,
        })
    return runs


def build_cycle_runs(input_idxs=None):
    idxs = input_idxs if input_idxs is not None else range(len(CYCLE_WINDOWS))
    return [
        {"key": f"W{wi}", "label": WINDOW_LABELS[wi],
         "cycles": CYCLE_WINDOWS[wi], "dqdv_window": "all",
         "input_key": f"W{wi}", "volt_key": "all"}
        for wi in idxs
    ]


def build_grid_runs(input_idxs=None, volt_keys=None):
    """Cross product: every input (cycle) window × every dQ/dV voltage window."""
    idxs = input_idxs if input_idxs is not None else range(len(CYCLE_WINDOWS))
    runs = []
    for wi in idxs:
        for vkey, vwin, vlabel in _voltage_units(volt_keys):
            runs.append({
                "key": f"W{wi}_{vkey}",
                "label": f"W{wi} × {vkey} ({vlabel})",
                "cycles": CYCLE_WINDOWS[wi], "dqdv_window": vwin,
                "input_key": f"W{wi}", "volt_key": vkey,
            })
    return runs


# ── output parsers ────────────────────────────────────────────────────────────
def _parse_accuracy(text, label):
    m = re.search(rf"\[Stage 2\] {label} accuracy: ([0-9.]+)%", text)
    if not m:
        raise ValueError(f"Could not find '{label} accuracy' in output")
    return float(m.group(1))


def _parse_metrics(text, label):
    m = re.search(
        rf"\[Stage 2\] {label} metrics: MAE=([0-9eE+\-.]+), MAE%=([0-9eE+\-.]+), "
        rf"RMSE=([0-9eE+\-.]+), RMSE%=([0-9eE+\-.]+), R2=([0-9eE+\-.]+)",
        text,
    )
    if not m:
        raise ValueError(f"Could not find '{label} metrics' in output")
    return {
        "mae":      float(m.group(1)),
        "mae_pct":  float(m.group(2)),
        "rmse":     float(m.group(3)),
        "rmse_pct": float(m.group(4)),
        "r2":       float(m.group(5)),
    }


def _summarize(vals):
    if not vals:
        return float("nan"), float("nan")
    return mean(vals), pstdev(vals) if len(vals) > 1 else 0.0


# ── per-run executor (patched model + train fn injected via runpy) ────────────
def _run_one(seed: int, cfg: dict, cycles: list, dqdv_window="all") -> dict:
    import torch
    import torch.nn as nn
    import numpy as np
    import CNN_LSTM_EOL as _base_mod

    model_cfg  = cfg["model"]
    train_cfg  = cfg["train"]
    loss_type  = cfg["loss"]
    sched_type = cfg["scheduler"]

    cnn_ch      = model_cfg["cnn_channels"]
    lstm_hidden = model_cfg["lstm_hidden"]
    num_layers  = model_cfg["num_layers"]
    cnn_drop    = model_cfg["cnn_dropout"]
    head_drop   = model_cfg["head_dropout"]

    class PatchedCNNLSTM(_base_mod.CNNLSTMEOLPredictor):
        def __init__(self, input_size, **_ignored):
            nn.Module.__init__(self)
            self.cnn_encoder = nn.Sequential(
                nn.Conv1d(input_size,   cnn_ch,     kernel_size=3, padding=1),
                nn.BatchNorm1d(cnn_ch),
                nn.ReLU(),
                nn.Dropout(cnn_drop),
                nn.Conv1d(cnn_ch,       cnn_ch * 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(cnn_ch * 2),
                nn.ReLU(),
                nn.Dropout(cnn_drop),
                nn.Conv1d(cnn_ch * 2,   cnn_ch * 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(cnn_ch * 2),
                nn.ReLU(),
                nn.Dropout(cnn_drop),
            )
            self.lstm = nn.LSTM(
                input_size=cnn_ch * 2,
                hidden_size=lstm_hidden,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.3 if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(lstm_hidden * 2, 256),
                nn.ReLU(),
                nn.Dropout(head_drop),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(head_drop),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(max(0.0, head_drop - 0.1)),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            x_cnn = x.transpose(1, 2)
            encoded = self.cnn_encoder(x_cnn)
            encoded = encoded.transpose(1, 2)
            lstm_out, _ = self.lstm(encoded)
            attn_weights = torch.softmax(torch.mean(lstm_out, dim=2), dim=1)
            context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
            return self.head(context).squeeze(-1)

    def patched_train_model(X_train, y_train, X_val, y_val,
                            _epochs=None, _batch_size=None, _lr=None,
                            _weight_decay=None, _lambda_eol=None):
        from torch.utils.data import DataLoader, TensorDataset
        del _epochs, _batch_size, _lr, _weight_decay, _lambda_eol

        ep  = train_cfg["epochs"]
        bs  = train_cfg["batch_size"]
        _lr = train_cfg["lr"]
        wd  = train_cfg["weight_decay"]
        lam = train_cfg["lambda_eol"]
        pat = 80

        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
        else:
            DEVICE = torch.device("cpu")

        X_train = X_train.float(); X_val = X_val.float()
        y_train = y_train.float().squeeze(-1)
        y_val   = y_val.float().squeeze(-1)

        y_mean = y_train.mean().item()
        y_std  = y_train.std().item() + 1e-8
        y_train_n = (y_train - y_mean) / y_std
        y_val_n   = (y_val   - y_mean) / y_std

        train_loader = DataLoader(TensorDataset(X_train, y_train_n), batch_size=bs, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_val,   y_val_n),   batch_size=bs)

        model = PatchedCNNLSTM(input_size=X_train.size(-1)).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=_lr, weight_decay=wd)

        if sched_type == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=_lr, steps_per_epoch=len(train_loader), epochs=ep)
            sched_per_batch = True
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
            sched_per_batch = False

        loss_fn = nn.L1Loss() if loss_type == "l1" else nn.SmoothL1Loss()

        best_val_loss, best_state, no_improve = float("inf"), None, 0
        history = {"train_mae": [], "val_mae": []}

        print("Epoch |   TrLoss |   TrAcc |   VaLoss |   VaAcc |       LR")
        print("-" * 58)

        def _acc(p, t):
            p, t = np.asarray(p, float), np.asarray(t, float)
            return float(np.mean(np.abs(p - t) / np.clip(np.abs(t), 1e-8, None) <= 0.10))

        for epoch in range(ep):
            model.train()
            tr_loss_sum = tr_n = 0
            tr_preds, tr_tgts = [], []
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = loss_fn(pred, yb) * lam
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched_per_batch: sched.step()
                n_b = yb.size(0)
                tr_loss_sum += loss.item() * n_b; tr_n += n_b
                tr_preds.append((pred.detach().cpu().numpy() * y_std) + y_mean)
                tr_tgts.append((yb.detach().cpu().numpy() * y_std) + y_mean)

            tr_preds = np.concatenate(tr_preds); tr_tgts = np.concatenate(tr_tgts)

            model.eval()
            va_loss_sum = va_n = 0
            va_preds, va_tgts = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(xb)
                    n_b = yb.size(0)
                    va_loss_sum += (loss_fn(pred, yb) * lam).item() * n_b; va_n += n_b
                    va_preds.append((pred.cpu().numpy() * y_std) + y_mean)
                    va_tgts.append((yb.cpu().numpy() * y_std) + y_mean)

            va_cat = np.concatenate(va_preds)
            tr_acc = _acc(tr_preds, tr_tgts)
            va_acc = _acc(va_cat, np.concatenate(va_tgts))
            history["train_mae"].append(float(np.mean(np.abs(tr_preds - tr_tgts))))
            history["val_mae"].append(float(np.mean(np.abs(va_cat - np.concatenate(va_tgts)))))

            if not sched_per_batch: sched.step()
            tr_loss = tr_loss_sum / max(1, tr_n)
            va_loss = va_loss_sum / max(1, va_n)

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            cur_lr = sched.get_last_lr()[0] if not sched_per_batch else _lr
            if (epoch + 1) % 20 == 0 or (epoch + 1) == ep:
                print(f"{epoch+1:5d} | {tr_loss:8.4f} | {tr_acc:7.4f} | "
                      f"{va_loss:8.4f} | {va_acc:7.4f} | {cur_lr:8.2e}")

            if no_improve >= pat:
                print(f"Early stopping at epoch {epoch+1}"); break

        model.load_state_dict(best_state)
        return model, (y_mean, y_std), history, val_loader, best_val_loss

    prev = {k: os.environ.get(k) for k in
            ("EOL_SHOW_PLOTS", "EOL_SAVE_ARTIFACTS", "EOL_SEED", "EOL_CYCLES_TO_USE",
             "EOL_DQDV_WINDOW")}
    os.environ["EOL_SHOW_PLOTS"]     = "0"
    os.environ["EOL_SAVE_ARTIFACTS"] = "0"
    os.environ["EOL_SEED"]           = str(seed)
    os.environ["EOL_CYCLES_TO_USE"]  = ",".join(map(str, cycles))
    os.environ["EOL_DQDV_WINDOW"]    = str(dqdv_window)

    buf = StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(buf), redirect_stderr(buf):
                runpy.run_path(
                    str(Path(__file__).parent / "CNN_LSTM_EOL.py"),
                    init_globals={
                        "CNNLSTMEOLPredictor": PatchedCNNLSTM,
                        "train_model":         patched_train_model,
                        "__name__":            "__main__",
                    },
                    run_name="__main__",
                )
    finally:
        for key, val in prev.items():
            if val is None: os.environ.pop(key, None)
            else:           os.environ[key] = val

    out = buf.getvalue()
    result = {}
    for split in SPLITS:
        result[split] = {"acc": _parse_accuracy(out, split), **_parse_metrics(out, split)}
    return result


# ── sweep ─────────────────────────────────────────────────────────────────────
def run_sweep(config_keys, runs, out_path, plot=True):
    """runs: list of window-run units (see build_*_runs). Runs every config ×
    seed for each unit and writes a keyed summary to out_path."""
    all_results = {}   # keyed by f"{unit_key}_{cfg_key}"

    for unit in runs:
        wkey, wlabel = unit["key"], unit["label"]
        cycles, dqdv_window = unit["cycles"], unit["dqdv_window"]
        print(f"\n{'#'*72}")
        print(f"WINDOW {wkey}: {wlabel}   (cycles: {len(cycles)} pts | dQ/dV window: {dqdv_window})")
        print(f"{'#'*72}")

        for cfg_key in config_keys:
            cfg    = CONFIGS[cfg_key]
            run_id = f"{wkey}_{cfg_key}"
            print(f"\n{'='*70}")
            print(f"  {run_id} | {wlabel} | {cfg['label']}")
            print(f"  model : {cfg['model']}")
            print(f"  train : {cfg['train']}")
            print(f"  loss  : {cfg['loss']}   scheduler: {cfg['scheduler']}")
            print(f"{'='*70}")

            store = {split: {m: [] for m in METRICS} for split in SPLITS}
            run_log = []

            for run_i, seed in enumerate(SEEDS, 1):
                print(f"\n  Run {run_i}/{len(SEEDS)} seed={seed} window={wkey}", flush=True)
                try:
                    res = _run_one(seed, cfg, cycles, dqdv_window=dqdv_window)
                except Exception as exc:
                    print(f"  ERROR run {run_i}: {exc}")
                    continue

                row = {"seed": seed}
                for split in SPLITS:
                    for m in METRICS:
                        store[split][m].append(res[split][m])
                    row[split] = res[split]
                run_log.append(row)

                for split in SPLITS:
                    r = res[split]
                    print(f"  {split:<5} | Acc={r['acc']:.2f}%  MAE={r['mae']:.1f}"
                          f"  RMSE={r['rmse']:.1f}  R²={r['r2']:.4f}")

            print(f"\n{'─'*70}")
            print(f"  {run_id} SUMMARY  ({len(run_log)}/{len(SEEDS)} runs OK)")
            print(f"  {'Split':<5}  {'Acc%':>12}  {'MAE':>12}  {'RMSE':>12}  {'R²':>12}")
            print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
            summary = {}
            for split in SPLITS:
                s = {}
                for m in METRICS:
                    mu, sd = _summarize(store[split][m])
                    s[m] = {"mean": round(mu, 4), "std": round(sd, 4)}
                summary[split] = s
                print(
                    f"  {split:<5}  "
                    f"{s['acc']['mean']:>6.2f}±{s['acc']['std']:<5.2f}  "
                    f"{s['mae']['mean']:>6.1f}±{s['mae']['std']:<5.1f}  "
                    f"{s['rmse']['mean']:>6.1f}±{s['rmse']['std']:<5.1f}  "
                    f"{s['r2']['mean']:>6.3f}±{s['r2']['std']:<5.3f}"
                )

            all_results[run_id] = {
                "window_key":   wkey,
                "window_label": wlabel,
                "input_key":    unit.get("input_key", wkey),
                "volt_key":     unit.get("volt_key", "all"),
                "dqdv_window":  dqdv_window,
                "n_cycles":     len(cycles),
                "config_key":   cfg_key,
                "label":        cfg["label"],
                "config":       cfg,
                "runs":         run_log,
                "summary":      summary,
            }

        # save after each window so partial results are not lost
        Path(out_path).write_text(json.dumps(all_results, indent=2))
        print(f"\n  [checkpoint] saved -> {out_path}")

    _print_global_table(all_results)
    Path(out_path).write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results saved -> {out_path}")
    if plot:
        _plot_comparison(all_results, out_path)
    return all_results


def _print_global_table(all_results):
    print(f"\n\n{'#'*72}")
    print("GLOBAL COMPARISON — Test set (mean ± std, all windows × configs)")
    print(f"{'#'*72}")
    hdr = f"{'ID':<10} {'Window':<44} {'Config':<38} {'Acc%':>12} {'MAE':>10} {'RMSE':>10} {'R²':>8}"
    print(hdr)
    print("─" * len(hdr))
    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["summary"]["Test"]["mae"]["mean"])
    for run_id, r in ranked:
        s = r["summary"]["Test"]
        print(
            f"{run_id:<10} "
            f"{r['window_label'][:43]:<44} "
            f"{r['label'][:37]:<38} "
            f"{s['acc']['mean']:>6.2f}±{s['acc']['std']:<5.2f} "
            f"{s['mae']['mean']:>6.1f}±{s['mae']['std']:<4.1f} "
            f"{s['rmse']['mean']:>6.1f}±{s['rmse']['std']:<4.1f} "
            f"{s['r2']['mean']:>5.3f}±{s['r2']['std']:<5.3f}"
        )


def _axis_sort_key(k):
    """Sort 'W0','V3','all',... numerically with 'all' last."""
    if k == "all":
        return (1, 0)
    m = re.match(r"[A-Za-z]+(\d+)", k)
    return (0, int(m.group(1)) if m else 0)


def _plot_comparison(all_results, out_path):
    """Choose a plot by result shape: a 2-D (input × voltage) grid -> heatmap;
    otherwise a grouped bar chart (one bar per config)."""
    if not all_results:
        return
    n_inputs = len({r["input_key"] for r in all_results.values()})
    n_volts = len({r["volt_key"] for r in all_results.values()})
    if n_inputs > 1 and n_volts > 1:
        _plot_grid_heatmap(all_results, out_path)
    else:
        _plot_bar(all_results, out_path)


def _plot_grid_heatmap(all_results, out_path):
    """Heatmap of best-config (min mean) Test MAE per (input window, voltage window)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[plot skipped] {exc}")
        return

    input_keys = sorted({r["input_key"] for r in all_results.values()}, key=_axis_sort_key)
    volt_keys = sorted({r["volt_key"] for r in all_results.values()}, key=_axis_sort_key)
    mat = np.full((len(input_keys), len(volt_keys)), np.nan)
    best_cfg = {}
    for i, ik in enumerate(input_keys):
        for j, vk in enumerate(volt_keys):
            cells = [r for r in all_results.values()
                     if r["input_key"] == ik and r["volt_key"] == vk]
            if cells:
                best = min(cells, key=lambda r: r["summary"]["Test"]["mae"]["mean"])
                mat[i, j] = best["summary"]["Test"]["mae"]["mean"]
                best_cfg[(i, j)] = best["config_key"]

    fig, ax = plt.subplots(figsize=(1.1 * len(volt_keys) + 3, 0.7 * len(input_keys) + 2.5))
    im = ax.imshow(mat, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(volt_keys))); ax.set_xticklabels(volt_keys, rotation=45, ha="right")
    ax.set_yticks(range(len(input_keys))); ax.set_yticklabels(input_keys)
    ax.set_xlabel("dQ/dV voltage window")
    ax.set_ylabel("input (cycle) window")
    ax.set_title("CNN-LSTM: best-config Test MAE per (input × voltage) window")
    vmid = np.nanmean(mat) if np.isfinite(mat).any() else 0.0
    for i in range(len(input_keys)):
        for j in range(len(volt_keys)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}\n{best_cfg[(i, j)]}",
                        ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > vmid else "black")
    fig.colorbar(im, ax=ax, label="Test MAE (best config, mean over seeds)")
    fig.tight_layout()
    png_path = str(Path(out_path).with_suffix(".png"))
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    print(f"Saved grid heatmap -> {png_path}")
    if os.environ.get("EOL_SHOW_PLOTS", "1") == "1":
        plt.show()
    else:
        plt.close(fig)


def _plot_bar(all_results, out_path):
    """Grouped bar chart: mean Test MAE per window, one bar per config."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[plot skipped] {exc}")
        return
    if not all_results:
        return

    win_order, seen = [], set()
    for r in all_results.values():
        if r["window_key"] not in seen:
            win_order.append(r["window_key"]); seen.add(r["window_key"])
    cfg_order = sorted({r["config_key"] for r in all_results.values()})

    x = np.arange(len(win_order))
    width = 0.8 / max(1, len(cfg_order))
    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(win_order) * len(cfg_order)), 5))
    for j, cfg_key in enumerate(cfg_order):
        means, stds = [], []
        for wk in win_order:
            r = all_results.get(f"{wk}_{cfg_key}")
            means.append(r["summary"]["Test"]["mae"]["mean"] if r else np.nan)
            stds.append(r["summary"]["Test"]["mae"]["std"] if r else 0.0)
        ax.bar(x + j * width, means, width, yerr=stds, capsize=3, label=f"Config {cfg_key}")

    ax.set_xticks(x + width * (len(cfg_order) - 1) / 2)
    ax.set_xticklabels(win_order, rotation=45, ha="right")
    ax.set_ylabel("Test MAE (cycles), mean ± std over seeds")
    ax.set_xlabel("window")
    ax.set_title("CNN-LSTM sweep: Test MAE per window × config")
    ax.legend(title="Config", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png_path = str(Path(out_path).with_suffix(".png"))
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    print(f"Saved comparison plot -> {png_path}")
    if os.environ.get("EOL_SHOW_PLOTS", "1") == "1":
        plt.show()
    else:
        plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _voltage_keys = [f"V{w}" for w in range(len(DQDV_V_WINDOWS))] + ["all"]

    parser = argparse.ArgumentParser(
        description="CNN-LSTM sweep: window (grid / voltage / cycle) × configs × seeds"
    )
    parser.add_argument(
        "--mode", choices=["grid", "voltage", "cycles"], default="grid",
        help="grid = voltage × input (cycle) windows cross product (default); "
             "voltage = dQ/dV windows only (input fixed); cycles = cycle windows only (dQ/dV=all)",
    )
    parser.add_argument(
        "--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS),
        metavar="CFG", help="Configs to run, e.g. A B C  (default: all)",
    )
    parser.add_argument(
        "--windows", nargs="+", default=None, metavar="V",
        help=f"Voltage windows to include (grid/voltage modes): {_voltage_keys}. Default: all.",
    )
    parser.add_argument(
        "--input-windows", nargs="+", type=int, default=None, metavar="N",
        choices=list(range(len(CYCLE_WINDOWS))),
        help=f"Input (cycle) window indices 0..{len(CYCLE_WINDOWS)-1} (grid/cycles modes). Default: all.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip the comparison chart.",
    )
    parser.add_argument(
        "--out", default=None,
        help="JSON output path (default depends on --mode).",
    )
    args = parser.parse_args()

    if args.mode == "grid":
        runs = build_grid_runs(input_idxs=args.input_windows, volt_keys=args.windows)
        out_path = args.out or str(Path(__file__).parent / "cnn_lstm_grid_ablation_results.json")
    elif args.mode == "voltage":
        runs = build_voltage_runs(volt_keys=args.windows)
        out_path = args.out or str(Path(__file__).parent / "cnn_lstm_window_ablation_results.json")
    else:  # cycles
        runs = build_cycle_runs(input_idxs=args.input_windows)
        out_path = args.out or str(Path(__file__).parent / "cnn_lstm_sweep_results.json")

    if not runs:
        raise SystemExit("No windows selected — check --windows / --input-windows / --mode.")

    print(f"[mode={args.mode}] {len(runs)} window-units × {len(args.configs)} configs "
          f"× {len(SEEDS)} seeds = {len(runs) * len(args.configs) * len(SEEDS)} runs")
    run_sweep(args.configs, runs, out_path, plot=not args.no_plot)
