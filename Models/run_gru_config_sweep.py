"""
Config sweep for the two-stage EOL GRU.

Two sweep modes (choose with --mode):

  voltage  (default) — dQ/dV VOLTAGE-window ablation.
             For each dQ/dV voltage window (+ an "all" baseline) run all configs
             over all seeds, holding the cycle window fixed. Shows which voltage
             band carries the most EOL-predictive dQ/dV signal.

  cycles   — the original CYCLE-window sweep (6 cycle-index windows), using every
             dQ/dV window together.

Both modes: configs × seeds per window, Train/Val/Test summarised as mean ± std.
Seeds/flags come from .env (EOL_SEED_LIST, EOL_SHOW_PLOTS, EOL_SAVE_ARTIFACTS).

Usage:
    python run_gru_config_sweep.py                      # voltage ablation, all windows/configs
    python run_gru_config_sweep.py --configs A B --windows V0 V4 all
    python run_gru_config_sweep.py --mode cycles --windows 0 3
    python run_gru_config_sweep.py --out results.json
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


# ── make repo root importable (for featureExtrcation + the runpy'd model) ─────
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from featureExtrcation.processDatasets import DQDV_V_WINDOWS  # noqa: E402


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

# Cycle window held fixed while sweeping dQ/dV voltage windows.
_raw_cycles = os.environ.get("EOL_CYCLES_TO_USE", "")
FIXED_CYCLES = [int(c) for c in _raw_cycles.split(",") if c.strip().isdigit()] \
    or [1, 10, 50, 100, 150, 200, 250]

# ── GRU configs ───────────────────────────────────────────────────────────────
CONFIGS = {
    "A": {
        "label": "A — Baseline (hidden=128, layers=1)",
        "model": dict(hidden_size=128, num_layers=1, head_dropout=0.3),
        "train": dict(epochs=300, batch_size=4, lr=3e-4, weight_decay=5e-3, lambda_eol=1.5),
        "loss": "smooth_l1", "scheduler": "cosine",
    },
    "B": {
        "label": "B — Deeper (hidden=128, layers=2)",
        "model": dict(hidden_size=128, num_layers=2, head_dropout=0.3),
        "train": dict(epochs=300, batch_size=8, lr=2e-4, weight_decay=5e-3, lambda_eol=1.5),
        "loss": "smooth_l1", "scheduler": "cosine",
    },
    "C": {
        "label": "C — Wider (hidden=256, layers=1)",
        "model": dict(hidden_size=256, num_layers=1, head_dropout=0.2),
        "train": dict(epochs=300, batch_size=4, lr=3e-4, weight_decay=1e-2, lambda_eol=1.5),
        "loss": "smooth_l1", "scheduler": "cosine",
    },
    "D": {
        "label": "D — OneCycleLR (hidden=128, layers=2)",
        "model": dict(hidden_size=128, num_layers=2, head_dropout=0.3),
        "train": dict(epochs=300, batch_size=4, lr=5e-4, weight_decay=5e-3, lambda_eol=1.0),
        "loss": "smooth_l1", "scheduler": "onecycle",
    },
    "E": {
        "label": "E — L1 Loss + Small (hidden=64, layers=2)",
        "model": dict(hidden_size=64, num_layers=2, head_dropout=0.4),
        "train": dict(epochs=300, batch_size=4, lr=1e-4, weight_decay=2e-2, lambda_eol=2.0),
        "loss": "l1", "scheduler": "cosine",
    },
}

METRICS = ("acc", "mae", "mae_pct", "rmse", "rmse_pct", "r2")
SPLITS = ("Train", "Val", "Test")


# ── window-run builders ───────────────────────────────────────────────────────
# A "run unit" = {key, label, cycles, dqdv_window, input_key, volt_key}; the sweep
# runs every config × seed for each unit. dqdv_window drives EOL_DQDV_WINDOW
# ('all' | 'none' | index). input_key / volt_key identify the two ablation axes
# (cycle window / voltage window) so results can be laid out as a grid.
def _voltage_units(volt_keys=None):
    """(volt_key, dqdv_window, short_label) for each dQ/dV window + 'all'."""
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


# ── per-run executor ──────────────────────────────────────────────────────────
def _run_one(seed: int, cfg: dict, cycles: list, dqdv_window="all") -> dict:
    model_cfg  = cfg["model"]
    train_cfg  = cfg["train"]

    env_keys = (
        "EOL_SHOW_PLOTS", "EOL_SAVE_ARTIFACTS", "EOL_SEED", "EOL_CYCLES_TO_USE",
        "EOL_HIDDEN_SIZE", "EOL_NUM_LAYERS", "EOL_HEAD_DROPOUT",
        "EOL_EPOCHS", "EOL_BATCH_SIZE", "EOL_LR", "EOL_WEIGHT_DECAY", "EOL_LAMBDA_EOL",
        "EOL_LOSS", "EOL_SCHEDULER", "EOL_DQDV_WINDOW",
    )
    prev = {k: os.environ.get(k) for k in env_keys}

    os.environ["EOL_SHOW_PLOTS"]     = "0"
    os.environ["EOL_SAVE_ARTIFACTS"] = "0"
    os.environ["EOL_SEED"]           = str(seed)
    os.environ["EOL_CYCLES_TO_USE"]  = ",".join(map(str, cycles))
    os.environ["EOL_HIDDEN_SIZE"]    = str(model_cfg["hidden_size"])
    os.environ["EOL_NUM_LAYERS"]     = str(model_cfg["num_layers"])
    os.environ["EOL_HEAD_DROPOUT"]   = str(model_cfg["head_dropout"])
    os.environ["EOL_EPOCHS"]         = str(train_cfg["epochs"])
    os.environ["EOL_BATCH_SIZE"]     = str(train_cfg["batch_size"])
    os.environ["EOL_LR"]             = str(train_cfg["lr"])
    os.environ["EOL_WEIGHT_DECAY"]   = str(train_cfg["weight_decay"])
    os.environ["EOL_LAMBDA_EOL"]     = str(train_cfg["lambda_eol"])
    os.environ["EOL_LOSS"]           = cfg["loss"]
    os.environ["EOL_SCHEDULER"]      = cfg["scheduler"]
    os.environ["EOL_DQDV_WINDOW"]    = str(dqdv_window)

    buf = StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with redirect_stdout(buf), redirect_stderr(buf):
                runpy.run_path(
                    str(Path(__file__).parent / "GRU_seq2seq.py"),
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
    hdr = f"{'ID':<10} {'Window':<44} {'Config':<34} {'Acc%':>12} {'MAE':>10} {'RMSE':>10} {'R²':>8}"
    print(hdr)
    print("─" * len(hdr))
    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["summary"]["Test"]["mae"]["mean"])
    for run_id, r in ranked:
        s = r["summary"]["Test"]
        print(
            f"{run_id:<10} "
            f"{r['window_label'][:43]:<44} "
            f"{r['label'][:33]:<34} "
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
    ax.set_title("GRU: best-config Test MAE per (input × voltage) window")
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
    ax.set_title("GRU sweep: Test MAE per window × config")
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
        description="GRU sweep: window (voltage or cycle) × configs × seeds"
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
        out_path = args.out or str(Path(__file__).parent / "gru_grid_ablation_results.json")
    elif args.mode == "voltage":
        runs = build_voltage_runs(volt_keys=args.windows)
        out_path = args.out or str(Path(__file__).parent / "gru_window_ablation_results.json")
    else:  # cycles
        runs = build_cycle_runs(input_idxs=args.input_windows)
        out_path = args.out or str(Path(__file__).parent / "gru_sweep_results.json")

    if not runs:
        raise SystemExit("No windows selected — check --windows / --input-windows / --mode.")

    print(f"[mode={args.mode}] {len(runs)} window-units × {len(args.configs)} configs "
          f"× {len(SEEDS)} seeds = {len(runs) * len(args.configs) * len(SEEDS)} runs")
    run_sweep(args.configs, runs, out_path, plot=not args.no_plot)
