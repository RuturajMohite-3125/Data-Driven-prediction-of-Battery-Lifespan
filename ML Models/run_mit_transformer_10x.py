import re
import os
import runpy
import random
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from statistics import mean, pstdev
import warnings


RUNS = 10


def parse_accuracy(text, label):
    pattern = rf"\[Stage 2\] {label} accuracy: ([0-9.]+)%"
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"Could not find '{label} accuracy' in output")
    return float(m.group(1))


def parse_metrics(text, label):
    pattern = rf"\[Stage 2\] {label} metrics: MAE=([0-9eE+\-.]+), MAE%=([0-9eE+\-.]+), RMSE=([0-9eE+\-.]+), RMSE%=([0-9eE+\-.]+), R2=([0-9eE+\-.]+)"
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"Could not find '{label} metrics' in output")
    return {
        "mae": float(m.group(1)),
        "mae_pct": float(m.group(2)),
        "rmse": float(m.group(3)),
        "rmse_pct": float(m.group(4)),
        "r2": float(m.group(5)),
    }


def summarize(values):
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def main():
    metrics_store = {
        "Train": {"mae": [], "mae_pct": [], "rmse": [], "rmse_pct": [], "r2": [], "acc": []},
        "Val":   {"mae": [], "mae_pct": [], "rmse": [], "rmse_pct": [], "r2": [], "acc": []},
        "Test":  {"mae": [], "mae_pct": [], "rmse": [], "rmse_pct": [], "r2": [], "acc": []},
    }

    for run_idx in range(1, RUNS + 1):
        prev_show = os.environ.get("EOL_SHOW_PLOTS")
        prev_save = os.environ.get("EOL_SAVE_ARTIFACTS")
        prev_seed = os.environ.get("EOL_SEED")
        run_seed = random.randint(1, 2_147_483_647)
        os.environ["EOL_SHOW_PLOTS"] = "0"
        os.environ["EOL_SAVE_ARTIFACTS"] = "0"
        os.environ["EOL_SEED"] = str(run_seed)

        buf = StringIO()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with redirect_stdout(buf), redirect_stderr(buf):
                    runpy.run_path("MIT_transformer_EOL.py", run_name="__main__")
        finally:
            if prev_show is None:
                os.environ.pop("EOL_SHOW_PLOTS", None)
            else:
                os.environ["EOL_SHOW_PLOTS"] = prev_show
            if prev_save is None:
                os.environ.pop("EOL_SAVE_ARTIFACTS", None)
            else:
                os.environ["EOL_SAVE_ARTIFACTS"] = prev_save
            if prev_seed is None:
                os.environ.pop("EOL_SEED", None)
            else:
                os.environ["EOL_SEED"] = prev_seed

        output = buf.getvalue()

        # parse
        t_acc = parse_accuracy(output, "Train")
        v_acc = parse_accuracy(output, "Val")
        te_acc = parse_accuracy(output, "Test")

        t_m = parse_metrics(output, "Train")
        v_m = parse_metrics(output, "Val")
        te_m = parse_metrics(output, "Test")

        # store
        metrics_store["Train"]["acc"].append(t_acc)
        metrics_store["Train"]["mae"].append(t_m["mae"]) if isinstance(t_m, dict) else metrics_store["Train"]["mae"].append(t_m[0])
        metrics_store["Train"]["mae_pct"].append(t_m["mae_pct"]) if isinstance(t_m, dict) else metrics_store["Train"]["mae_pct"].append(t_m[1])
        metrics_store["Train"]["rmse"].append(t_m["rmse"]) if isinstance(t_m, dict) else metrics_store["Train"]["rmse"].append(t_m[2])
        metrics_store["Train"]["rmse_pct"].append(t_m["rmse_pct"]) if isinstance(t_m, dict) else metrics_store["Train"]["rmse_pct"].append(t_m[3])
        metrics_store["Train"]["r2"].append(t_m["r2"]) if isinstance(t_m, dict) else metrics_store["Train"]["r2"].append(t_m[4])

        metrics_store["Val"]["acc"].append(v_acc)
        metrics_store["Val"]["mae"].append(v_m["mae"]) if isinstance(v_m, dict) else metrics_store["Val"]["mae"].append(v_m[0])
        metrics_store["Val"]["mae_pct"].append(v_m["mae_pct"]) if isinstance(v_m, dict) else metrics_store["Val"]["mae_pct"].append(v_m[1])
        metrics_store["Val"]["rmse"].append(v_m["rmse"]) if isinstance(v_m, dict) else metrics_store["Val"]["rmse"].append(v_m[2])
        metrics_store["Val"]["rmse_pct"].append(v_m["rmse_pct"]) if isinstance(v_m, dict) else metrics_store["Val"]["rmse_pct"].append(v_m[3])
        metrics_store["Val"]["r2"].append(v_m["r2"]) if isinstance(v_m, dict) else metrics_store["Val"]["r2"].append(v_m[4])

        metrics_store["Test"]["acc"].append(te_acc)
        metrics_store["Test"]["mae"].append(te_m["mae"]) if isinstance(te_m, dict) else metrics_store["Test"]["mae"].append(te_m[0])
        metrics_store["Test"]["mae_pct"].append(te_m["mae_pct"]) if isinstance(te_m, dict) else metrics_store["Test"]["mae_pct"].append(te_m[1])
        metrics_store["Test"]["rmse"].append(te_m["rmse"]) if isinstance(te_m, dict) else metrics_store["Test"]["rmse"].append(te_m[2])
        metrics_store["Test"]["rmse_pct"].append(te_m["rmse_pct"]) if isinstance(te_m, dict) else metrics_store["Test"]["rmse_pct"].append(te_m[3])
        metrics_store["Test"]["r2"].append(te_m["r2"]) if isinstance(te_m, dict) else metrics_store["Test"]["r2"].append(te_m[4])

        print(
            f"Run {run_idx:2d} seed={run_seed}: "
            f"Train(acc={t_acc:.2f}%, MAE={metrics_store['Train']['mae'][-1]:.2f}, RMSE={metrics_store['Train']['rmse'][-1]:.2f}, R2={metrics_store['Train']['r2'][-1]:.4f}) | "
            f"Val(acc={v_acc:.2f}%, MAE={metrics_store['Val']['mae'][-1]:.2f}, RMSE={metrics_store['Val']['rmse'][-1]:.2f}, R2={metrics_store['Val']['r2'][-1]:.4f}) | "
            f"Test(acc={te_acc:.2f}%, MAE={metrics_store['Test']['mae'][-1]:.2f}, RMSE={metrics_store['Test']['rmse'][-1]:.2f}, R2={metrics_store['Test']['r2'][-1]:.4f})"
        )

    # Summaries
    print("\n=== Summary over {} runs ===".format(RUNS))

    for split in ("Train", "Val", "Test"):
        acc_mean, acc_std = summarize(metrics_store[split]["acc"])
        mae_mean, mae_std = summarize(metrics_store[split]["mae"])
        rmse_mean, rmse_std = summarize(metrics_store[split]["rmse"])
        r2_mean, r2_std = summarize(metrics_store[split]["r2"])

        # Print in requested format: mean ± std
        print(f"{split} MAE: {mae_mean:.1f} ± {mae_std:.1f} cycles")
        print(f"{split} RMSE: {rmse_mean:.1f} ± {rmse_std:.1f} cycles")
        print(f"{split} R²: {r2_mean:.3f} ± {r2_std:.3f}")
        print(f"Accuracy : {acc_mean:.2f} ± {acc_std:.2f}%\n")


if __name__ == "__main__":
    main()
