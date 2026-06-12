"""
Config sweep: 6 cycle windows × 5 CNN-LSTM configs × 10 seeds.
Seeds and flags are read from .env (EOL_SEED_LIST, EOL_SHOW_PLOTS, EOL_SAVE_ARTIFACTS).

Knobs varied per config:
  CNN  : cnn_channels (base), cnn_dropout
  LSTM : lstm_hidden_size, num_layers, rnn_dropout (auto when layers>1)
  Head : head_dropout
  Train: epochs, batch_size, lr, weight_decay, lambda_eol, loss, scheduler

Usage:
    python run_cnn_lstm_config_sweep.py
    python run_cnn_lstm_config_sweep.py --configs A B --windows 0 3
    python run_cnn_lstm_config_sweep.py --out results.json
"""

import argparse
import json
import os
import re
import runpy
import warnings
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from statistics import mean, pstdev

# ── load .env ────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

_raw_seeds = os.environ.get("EOL_SEED_LIST", "17,42,89,123,256,404,777,1024,2024,3141")
SEEDS = [int(s) for s in _raw_seeds.split(",") if s.strip().isdigit()]

# ── cycle windows ─────────────────────────────────────────────────────────────
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


# ── per-run executor ──────────────────────────────────────────────────────────
def _run_one(seed: int, cfg: dict, cycles: list) -> dict:
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
            ("EOL_SHOW_PLOTS", "EOL_SAVE_ARTIFACTS", "EOL_SEED", "EOL_CYCLES_TO_USE")}
    os.environ["EOL_SHOW_PLOTS"]     = "0"
    os.environ["EOL_SAVE_ARTIFACTS"] = "0"
    os.environ["EOL_SEED"]           = str(seed)
    os.environ["EOL_CYCLES_TO_USE"]  = ",".join(map(str, cycles))

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
    for split in ("Train", "Val", "Test"):
        result[split] = {"acc": _parse_accuracy(out, split), **_parse_metrics(out, split)}
    return result


# ── sweep ─────────────────────────────────────────────────────────────────────
def run_sweep(config_keys: list, window_indices: list, out_path: str):
    all_results = {}

    for wi in window_indices:
        cycles = CYCLE_WINDOWS[wi]
        wlabel = WINDOW_LABELS[wi]
        print(f"\n{'#'*72}")
        print(f"WINDOW {wi}: {wlabel}  ({len(cycles)} cycle points)")
        print(f"{'#'*72}")

        for cfg_key in config_keys:
            cfg    = CONFIGS[cfg_key]
            run_id = f"W{wi}_{cfg_key}"
            print(f"\n{'='*70}")
            print(f"  {run_id} | {cfg['label']}")
            print(f"  model : {cfg['model']}")
            print(f"  train : {cfg['train']}")
            print(f"  loss  : {cfg['loss']}   scheduler: {cfg['scheduler']}")
            print(f"{'='*70}")

            store = {
                split: {m: [] for m in ("acc", "mae", "mae_pct", "rmse", "rmse_pct", "r2")}
                for split in ("Train", "Val", "Test")
            }
            run_log = []

            for run_i, seed in enumerate(SEEDS, 1):
                print(f"\n  Run {run_i}/{len(SEEDS)} seed={seed}", flush=True)
                try:
                    res = _run_one(seed, cfg, cycles)
                except Exception as exc:
                    print(f"  ERROR run {run_i}: {exc}")
                    continue

                row = {"seed": seed}
                for split in ("Train", "Val", "Test"):
                    for m in ("acc", "mae", "mae_pct", "rmse", "rmse_pct", "r2"):
                        store[split][m].append(res[split][m])
                    row[split] = res[split]
                run_log.append(row)

                print(
                    f"  Train | Acc={res['Train']['acc']:.2f}%  MAE={res['Train']['mae']:.1f}"
                    f"  RMSE={res['Train']['rmse']:.1f}  R²={res['Train']['r2']:.4f}"
                )
                print(
                    f"  Val   | Acc={res['Val']['acc']:.2f}%  MAE={res['Val']['mae']:.1f}"
                    f"  RMSE={res['Val']['rmse']:.1f}  R²={res['Val']['r2']:.4f}"
                )
                print(
                    f"  Test  | Acc={res['Test']['acc']:.2f}%  MAE={res['Test']['mae']:.1f}"
                    f"  RMSE={res['Test']['rmse']:.1f}  R²={res['Test']['r2']:.4f}"
                )

            print(f"\n{'─'*70}")
            print(f"  {run_id} SUMMARY  ({len(run_log)}/{len(SEEDS)} runs OK)")
            print(f"  {'Split':<5}  {'Acc%':>12}  {'MAE':>12}  {'RMSE':>12}  {'R²':>12}")
            print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")
            summary = {}
            for split in ("Train", "Val", "Test"):
                s = {}
                for m in ("acc", "mae", "mae_pct", "rmse", "rmse_pct", "r2"):
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
                "window_idx":   wi,
                "window_label": wlabel,
                "n_cycles":     len(cycles),
                "config_key":   cfg_key,
                "label":        cfg["label"],
                "config":       cfg,
                "runs":         run_log,
                "summary":      summary,
            }

        Path(out_path).write_text(json.dumps(all_results, indent=2))
        print(f"\n  [checkpoint] saved -> {out_path}")

    print(f"\n\n{'#'*72}")
    print("GLOBAL COMPARISON — Test set (mean ± std, all windows × configs)")
    print(f"{'#'*72}")
    hdr = f"{'ID':<8} {'Window':<44} {'Config':<38} {'Acc%':>12} {'MAE':>10} {'RMSE':>10} {'R²':>8}"
    print(hdr)
    print("─" * len(hdr))

    ranked = sorted(
        all_results.items(),
        key=lambda kv: kv[1]["summary"]["Test"]["mae"]["mean"]
    )
    for run_id, r in ranked:
        s = r["summary"]["Test"]
        print(
            f"{run_id:<8} "
            f"{r['window_label'][:43]:<44} "
            f"{r['label'][:37]:<38} "
            f"{s['acc']['mean']:>6.2f}±{s['acc']['std']:<5.2f} "
            f"{s['mae']['mean']:>6.1f}±{s['mae']['std']:<4.1f} "
            f"{s['rmse']['mean']:>6.1f}±{s['rmse']['std']:<4.1f} "
            f"{s['r2']['mean']:>5.3f}±{s['r2']['std']:<5.3f}"
        )

    print(f"\nFull results saved -> {out_path}")
    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNN-LSTM sweep: cycle windows × configs × seeds"
    )
    parser.add_argument(
        "--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS),
        metavar="CFG", help="Configs to run, e.g. A B C  (default: all)"
    )
    parser.add_argument(
        "--windows", nargs="+", type=int,
        choices=list(range(len(CYCLE_WINDOWS))),
        default=list(range(len(CYCLE_WINDOWS))),
        metavar="N", help="Window indices to run, e.g. 0 3  (default: all)"
    )
    parser.add_argument(
        "--out", default="cnn_lstm_sweep_results.json",
        help="JSON output path (default: cnn_lstm_sweep_results.json)"
    )
    args = parser.parse_args()
    run_sweep(args.configs, args.windows, args.out)
