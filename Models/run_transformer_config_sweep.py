"""
Config sweep: 6 cycle windows × 5 transformer configs × 10 seeds.
Seeds and flags are read from .env (EOL_SEED_LIST, EOL_SHOW_PLOTS, EOL_SAVE_ARTIFACTS).

Usage:
    python run_transformer_config_sweep.py
    python run_transformer_config_sweep.py --configs A B --windows 0 3
    python run_transformer_config_sweep.py --out results.json
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

# ── transformer configs ───────────────────────────────────────────────────────
CONFIGS = {
    "A": {
        "label": "A — Baseline (current)",
        "model": dict(d_model=64,  nhead=4, num_layers=2, dim_ff=128, dropout=0.3),
        "train": dict(epochs=300, batch_size=4, lr=3e-4, weight_decay=5e-3,  lambda_eol=1.5),
        "loss": "smooth_l1", "scheduler": "cosine",
    },
    "B": {
        "label": "B — Deeper + Wider",
        "model": dict(d_model=128, nhead=8, num_layers=3, dim_ff=256, dropout=0.2),
        "train": dict(epochs=300, batch_size=8, lr=2e-4, weight_decay=5e-3,  lambda_eol=1.5),
        "loss": "smooth_l1", "scheduler": "cosine",
    },
    "C": {
        "label": "C — Regularized Small",
        "model": dict(d_model=64,  nhead=4, num_layers=2, dim_ff=128, dropout=0.4),
        "train": dict(epochs=300, batch_size=8, lr=3e-4, weight_decay=2e-2,  lambda_eol=1.5),
        "loss": "smooth_l1", "scheduler": "cosine",
    },
    "D": {
        "label": "D — OneCycleLR",
        "model": dict(d_model=64,  nhead=4, num_layers=2, dim_ff=128, dropout=0.3),
        "train": dict(epochs=300, batch_size=4, lr=5e-4, weight_decay=5e-3,  lambda_eol=1.0),
        "loss": "smooth_l1", "scheduler": "onecycle",
    },
    "E": {
        "label": "E — L1 Loss + Larger",
        "model": dict(d_model=96,  nhead=4, num_layers=3, dim_ff=192, dropout=0.25),
        "train": dict(epochs=300, batch_size=4, lr=1e-4, weight_decay=1e-2,  lambda_eol=2.0),
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
        "mae":     float(m.group(1)),
        "mae_pct": float(m.group(2)),
        "rmse":    float(m.group(3)),
        "rmse_pct":float(m.group(4)),
        "r2":      float(m.group(5)),
    }


def _summarize(vals):
    if not vals:
        return float("nan"), float("nan")
    return mean(vals), pstdev(vals) if len(vals) > 1 else 0.0


# ── per-run executor ──────────────────────────────────────────────────────────
def _run_one(seed: int, cfg: dict, cycles: list) -> dict:
    model_cfg  = cfg["model"]
    train_cfg  = cfg["train"]

    env_keys = (
        "EOL_SHOW_PLOTS", "EOL_SAVE_ARTIFACTS", "EOL_SEED", "EOL_CYCLES_TO_USE",
        "EOL_D_MODEL", "EOL_NHEAD", "EOL_NUM_LAYERS", "EOL_DIM_FF", "EOL_DROPOUT",
        "EOL_EPOCHS", "EOL_BATCH_SIZE", "EOL_LR", "EOL_WEIGHT_DECAY", "EOL_LAMBDA_EOL",
        "EOL_LOSS_TYPE", "EOL_SCHEDULER",
    )
    prev = {k: os.environ.get(k) for k in env_keys}

    os.environ["EOL_SHOW_PLOTS"]     = "0"
    os.environ["EOL_SAVE_ARTIFACTS"] = "0"
    os.environ["EOL_SEED"]           = str(seed)
    os.environ["EOL_CYCLES_TO_USE"]  = ",".join(map(str, cycles))
    os.environ["EOL_D_MODEL"]        = str(model_cfg["d_model"])
    os.environ["EOL_NHEAD"]          = str(model_cfg["nhead"])
    os.environ["EOL_NUM_LAYERS"]     = str(model_cfg["num_layers"])
    os.environ["EOL_DIM_FF"]         = str(model_cfg["dim_ff"])
    os.environ["EOL_DROPOUT"]        = str(model_cfg["dropout"])
    os.environ["EOL_EPOCHS"]         = str(train_cfg["epochs"])
    os.environ["EOL_BATCH_SIZE"]     = str(train_cfg["batch_size"])
    os.environ["EOL_LR"]             = str(train_cfg["lr"])
    os.environ["EOL_WEIGHT_DECAY"]   = str(train_cfg["weight_decay"])
    os.environ["EOL_LAMBDA_EOL"]     = str(train_cfg["lambda_eol"])
    os.environ["EOL_LOSS_TYPE"]      = cfg["loss"]
    os.environ["EOL_SCHEDULER"]      = cfg["scheduler"]

    buf = StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(buf), redirect_stderr(buf):
                runpy.run_path(
                    str(Path(__file__).parent / "MIT_transformer_EOL.py"),
                    init_globals={"__name__": "__main__"},
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
    all_results = {}   # keyed by f"W{wi}_{cfg_key}"

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

            # per-combo summary
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

        # save after each window so partial results are not lost
        Path(out_path).write_text(json.dumps(all_results, indent=2))
        print(f"\n  [checkpoint] saved -> {out_path}")

    # ── global comparison table ───────────────────────────────────────────────
    print(f"\n\n{'#'*72}")
    print("GLOBAL COMPARISON — Test set (mean ± std, all windows × configs)")
    print(f"{'#'*72}")
    hdr = f"{'ID':<8} {'Window':<44} {'Config':<28} {'Acc%':>12} {'MAE':>10} {'RMSE':>10} {'R²':>8}"
    print(hdr)
    print("─" * len(hdr))

    # sort by Test MAE mean ascending
    ranked = sorted(
        all_results.items(),
        key=lambda kv: kv[1]["summary"]["Test"]["mae"]["mean"]
    )
    for run_id, r in ranked:
        s = r["summary"]["Test"]
        print(
            f"{run_id:<8} "
            f"{r['window_label'][:43]:<44} "
            f"{r['label'][:27]:<28} "
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
        description="Transformer sweep: cycle windows × configs × seeds"
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
        "--out", default="transformer_sweep_results.json",
        help="JSON output path (default: transformer_sweep_results.json)"
    )
    args = parser.parse_args()
    run_sweep(args.configs, args.windows, args.out)
