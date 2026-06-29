#!/bin/bash
#SBATCH --job-name=cnn_gru_sweep
#SBATCH --output=logs/cnn_gru_sweep_%j.out
#SBATCH --error=logs/cnn_gru_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=CHANGE_ME   # run: sinfo -o "%P %a %l" to see available partitions
# #SBATCH --gres=gpu:1          # uncomment if the partition has GPUs

# ── activate environment ──────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_DIR/.venv/bin/activate"

# ── create log directory ──────────────────────────────────────────────────────
mkdir -p "$REPO_DIR/Models/logs"

# ── optional: override seeds or output path via env vars ─────────────────────
# export EOL_SEED_LIST="17,42,89"
# OUT="$REPO_DIR/Results/cnn_gru_sweep_results.json"

OUT="${OUT:-$REPO_DIR/Results/cnn_gru_sweep_results.json}"
mkdir -p "$(dirname "$OUT")"

echo "===== CNN-GRU Config Sweep ====="
echo "Start : $(date)"
echo "Host  : $(hostname)"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Out   : $OUT"
echo "================================"

cd "$REPO_DIR/Models"
python run_cnn_gru_config_sweep.py --out "$OUT"

echo "===== Done : $(date) ====="
