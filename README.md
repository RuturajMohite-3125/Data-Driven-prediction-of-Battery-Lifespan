# Data-Driven Prediction of Battery Lifespan

A Master's thesis project that predicts the **End-of-Life (EOL)** cycle count of lithium-ion battery cells from early-cycle measurement data using deep learning. Five model architectures (Transformer, GRU, LSTM, CNN-GRU, CNN-LSTM) are trained and evaluated on the MIT battery dataset via a systematic hyperparameter sweep across 6 cycle windows and 5 configurations.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Feature Extraction](#feature-extraction)
  - [Two-Stage Prediction Pipeline](#two-stage-prediction-pipeline)
  - [Cycle Windows](#cycle-windows)
  - [Model Architectures](#model-architectures)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1 — Preprocess the Dataset](#step-1--preprocess-the-dataset)
  - [Step 2 — Train a Model](#step-2--train-a-model)
  - [Step 3 — Run a Config Sweep](#step-3--run-a-config-sweep)
  - [Step 4 — Re-evaluate a Saved Checkpoint](#step-4--re-evaluate-a-saved-checkpoint)
- [Environment Variables](#environment-variables)
- [Saved Artifacts](#saved-artifacts)
- [License](#license)

---

## Problem Statement

Predicting how long a lithium-ion battery will last before reaching end-of-life is critical for electric vehicles, grid storage, and consumer electronics. This project learns to predict the total cycle life of a battery cell using only measurements from its first 50–250 charge/discharge cycles, long before the cell actually degrades to failure.

**EOL is defined as the cycle at which the discharge capacity permanently drops below 80% of the nominal capacity.**

---

## Dataset

The project uses the **MIT/Stanford/Toyota LFP battery dataset** (Severson et al., 2019), commonly referred to as the MATR dataset. It consists of lithium iron phosphate (LFP) pouch cells cycled under various fast-charging protocols until end-of-life.

- ~180+ cells across 4 batches (`b0`, `b1`, `b2`, `b3`)
- Raw data format: per-cell `.pkl` files with per-cycle voltage, current, and capacity traces
- EOL range: ~150–2000 cycles depending on the charging protocol

**Dataset split** (fixed, defined in [Models/cell_split.json](Models/cell_split.json)):

| Split      | # Cells |
|------------|---------|
| Training   | ~150    |
| Validation | 18      |
| Test       | 18      |

> The raw dataset is not included in this repository. Place the per-cell `.pkl` files in a directory and set `RAW_PKL_PATH` in each model script accordingly (default: `/Users/ruturaj/Master-Thesis/Dataset/MIT`) or alternatively use generate feature cache `processed_hust_MIT_cache.pkl`.

---

## Project Structure

```
.
├── featureExtrcation/
│   ├── __init__.py
│   └── processDatasets.py                # Raw pkl → per-cycle feature cache (HUSTDataProcessor)
│
├── Models/
│   ├── MIT_transformer_EOL.py            # Transformer — train + evaluate
│   ├── GRU_seq2seq.py                    # GRU — train + evaluate
│   ├── LSTM_seq2seq.py                   # LSTM — train + evaluate
│   ├── CNN_GRU_EOL.py                    # CNN feature extractor + GRU — train + evaluate
│   ├── CNN_LSTM_EOL.py                   # CNN feature extractor + LSTM — train + 
│   │
│   ├── run_transformer_config_sweep.py   # Sweep runner — Transformer
│   ├── run_gru_config_sweep.py           # Sweep runner — GRU
│   ├── run_lstm_config_sweep.py          # Sweep runner — LSTM
│   ├── run_cnn_gru_config_sweep.py       # Sweep runner — CNN-GRU
│   ├── run_cnn_lstm_config_sweep.py      # Sweep runner — CNN-LSTM
│   ├── run_cnn_gru_sweep.sh              # SLURM batch wrapper for the CNN-GRU sweep
│   │
│   ├── cell_split.json                   # Fixed train/val/test cell assignments
│   ├── processed_hust_MIT_cache.pkl      # Cached feature extraction output (loaded by every model script)
│   ├── SWEEP_CONFIGS.md                  # Full sweep configuration documentation
│   │
│   └── aging_classifier_*.json           # Stage-1 XGBoost classifiers, one per architecture
│       ├── aging_classifier_50_xgb.json          # Transformer pipeline
│       ├── aging_classifier_gru_50_xgb.json       # GRU pipeline
│       ├── aging_classifier_lstm_50_xgb.json      # LSTM pipeline
│       ├── aging_classifier_cnn_gru_50_xgb.json   # CNN-GRU pipeline
│       └── aging_classifier_cnn_lstm_50_xgb.json  # CNN-LSTM pipeline
│
├── Results/
│   ├── global_sweep_report.md      # Full cross-model sweep results table (all 5 architectures)
│   ├── gru_sweep_report.md         # GRU-specific sweep results
│   ├── lstm_sweep_report.md        # LSTM-specific sweep results
│   ├── transformer_sweep_report.md # Transformer-specific sweep results
│   ├── cnn_gru_sweep_report.md     # CNN-GRU-specific sweep results
│   ├── cnn_lstm_sweep_report.md    # CNN-LSTM-specific sweep results
│   └── *_sweep_results.json        # Raw per-seed sweep output backing each *_report.md above
│
├── best_eol_transformer_50_cycle_window.pt   # Best Transformer checkpoint (repo root)
├── best_eol_gru_50_cycle_window.pt           # Best GRU checkpoint
├── best_eol_lstm_50_cycle_window.pt          # Best LSTM checkpoint
├── best_eol_cnn_gru_50_cycle_window.pt       # Best CNN-GRU checkpoint
├── best_eol_cnn_lstm_50_cycle_window.pt      # Best CNN-LSTM checkpoint
│
├── requirements.txt
└── LICENSE
```

> Checkpoints (`*.pt`) are git-ignored — they're regenerated locally whenever a model script is run with `EOL_SAVE_ARTIFACTS=1` (the default). Only `best_eol_transformer_50_cycle_window.pt` and `Models/aging_classifier_50_xgb.json` are committed as reference artifacts; the rest are working-tree outputs.

---

## Methodology

### Feature Extraction

[featureExtrcation/processDatasets.py](featureExtrcation/processDatasets.py) processes the raw per-cell `.pkl` files into a structured feature cache. For each cycle of each cell, **44 scalar features** are extracted:

| Feature Group | # | Features |
|---|---|---|
| **dQ/dV per voltage window** (Δ vs cycle 10 baseline) | **24** | `(max, min, var)` of dQ/dV in each of **8 voltage windows** spanning **2.0–3.6 V in 0.2 V steps** (`dQdV_max_2-2.2`, `dQdV_min_2-2.2`, `dQdV_var_2-2.2`, … `dQdV_var_3.4-3.6`) |
| Log-scale std (charge phase) | 3 | `log_std_I_c`, `log_std_Q_c`, `log_std_V_c` |
| Log-scale std (discharge phase) | 3 | `log_std_I_d`, `log_std_Q_d`, `log_std_V_d` |
| Current range | 4 | `min_I_c`, `max_I_c`, `min_I_d`, `max_I_d` |
| Voltage range | 4 | `min_V_c`, `max_V_c`, `min_V_d`, `max_V_d` |
| Capacity + shape | 2 | `max_Q_d`, `kurtosis_V_d` |
| Efficiency + voltage drop | 2 | `coulombic_eff`, `v_drop_start` |
| Cycle timing | 2 | `charge_time`, `discharge_time` |

**Voltage-window dQ/dV features.** Rather than a single global `(max, min, var)` summary of the dQ/dV curve, the discharge voltage range `2.0–3.6 V` is split into 8 consecutive `0.2 V` windows (`DQDV_V_RANGE` / `DQDV_WINDOW_SIZE` in the script). Within each window the dQ/dV segment is smoothed with a moving-average kernel (width 10) and reduced to its `(max, min, var)`. Each statistic is stored as a **delta relative to the cycle-10 baseline** so that per-window shifts in the differential-capacity signature — which localise degradation modes to specific voltage plateaus — are captured directly. Empty windows contribute `(0, 0, 0)`.

Each cell is stored in the cache as:

```python
{
    'cell_name': str,
    'features': np.ndarray,   # shape [n_cycles, 44]
    'soh':      np.ndarray,   # discharge capacity per cycle (Ah)
    'soh_traj': np.ndarray,   # normalised SoH, padded to 1500 cycles (-1 sentinel beyond EOL)
    'eol':      int,          # cycle index where capacity < 88% of peak
    'num_cycles': int,
}
```

The 24 dQ/dV columns occupy the first feature positions, so a single voltage window can be isolated at model-input time via `EOL_DQDV_WINDOW` (see [Environment Variables](#environment-variables)).

### Two-Stage Prediction Pipeline

All five model families share the same two-stage pipeline:

```
Raw cycles  →  Feature extraction  →  [Stage 1] XGBoost aging classifier
                                                        ↓
                                         Class probabilities (Fast / Normal / Slow)
                                                        ↓
                                    [Stage 2] Deep learning EOL regressor
                                                        ↓
                                         Predicted EOL (cycles)
                                                        ↓
                                    [Calibration] Class-mean blending
```

**Stage 1 — XGBoost Aging Classifier**

Cells are divided into three aging-speed classes using the 33rd/66th percentiles of training EOL values:
- **Fast**: EOL in the bottom third
- **Normal**: EOL in the middle third
- **Slow**: EOL in the top third

An XGBoost multi-class classifier is trained on scalar features (mean + slope across cycles). The predicted class probabilities are appended to each timestep of the input sequence before Stage 2.

**Stage 2 — Deep Learning Regressor**

The model receives a sequence of per-cycle features (up to 47 dimensions — the 44 extracted features plus the 3 appended class probabilities; fewer when a single dQ/dV voltage window is selected via `EOL_DQDV_WINDOW`) and produces a single scalar EOL prediction. Training uses Z-score normalisation of targets and early stopping with 80-epoch patience.


### Cycle Windows

Six cycle sampling windows are evaluated for each model architecture:

| ID | Definition | # Points | Purpose |
|----|-----------|----------|---------|
| **W0** | Cycles 1–50 | 50 | Early-life only |
| **W1** | Cycles 1–100 | 100 | Extended early-life |
| **W2** | Cycles 1–250, stride 10 | 25 | Sparse uniform coverage |
| **W3** | Cycles 10–40 + 180–200 | 51 | Dense early + late snapshots |
| **W4** | Cycles 1–50, 100–150, 200–250 | 150 | Three balanced dense bands |
| **W5** | {1, 10, 50, 100, 150, 200, 250} | 7 | Extreme sparsity (landmark points) |

### Model Architectures

#### Transformer (`MIT_transformer_EOL.py`)
Encoder-only transformer with sinusoidal positional encoding and a CLS token for pooling. The CLS token output is passed through an MLP head to predict EOL.

- Hyperparameters: `d_model`, `nhead`, `num_layers`, `dim_ff`, `dropout`
- Default: `d_model=64`, `nhead=4`, `num_layers=2`, `dim_ff=128`

#### GRU (`GRU_seq2seq.py`)
Bidirectional GRU with soft attention pooling over all timesteps. The attention-weighted context vector is passed to a 3-layer MLP head.

- Architecture: BiGRU → Attention → Linear(256→128) → ReLU → Dropout → Linear(128→64) → ReLU → Linear(64→1)
- Default: `hidden_size=128`, `num_layers=1`, `head_dropout=0.3`

#### LSTM (`LSTM_seq2seq.py`)
Structurally identical to the GRU model but uses LSTM cells. LSTM has ~33% more parameters per layer due to the extra cell-state gate.

#### CNN-GRU (`CNN_GRU_EOL.py`)
1D convolutional layers extract local temporal patterns from the cycle sequence, followed by a bidirectional GRU for global sequence modeling.

- Hyperparameters: `cnn_channels`, `gru_hidden`, `num_layers`, `cnn_dropout`, `head_dropout`

#### CNN-LSTM (`CNN_LSTM_EOL.py`)
Same as CNN-GRU but with LSTM cells replacing the GRU.

---

## Results

Best results from the cross-model sweep (mean ± std over 10 random seeds). **Accuracy** = fraction of test cells with predicted EOL within ±10% of true EOL. Sweep now covers five architectures — Transformer, GRU, LSTM, CNN-GRU, and CNN-LSTM — across all 6 cycle windows and 5 configs (150 model × window × config combinations total).

| Rank | Model | Window | Config | Accuracy (%) | MAE (cycles) | RMSE (cycles) | R² |
|------|-------|--------|--------|-------------|--------------|--------------|-----|
| 1 | **GRU** | W5: 7 landmarks | E — L1 + Small (hidden=64, layers=2) | **100.0 ± 0.0** | 32.2 ± 2.4 | 41.9 ± 3.1 | 0.961 ± 0.006 |
| 2 | LSTM | W5: 7 landmarks | B — Deeper (hidden=128, layers=2) | 99.4 ± 1.8 | **28.5 ± 2.3** | 37.9 ± 3.0 | 0.968 ± 0.005 |
| 3 | GRU | W5: 7 landmarks | A — Baseline (hidden=128, layers=1) | 98.9 ± 2.3 | 28.9 ± 3.0 | 40.2 ± 5.2 | 0.964 ± 0.009 |
| 4 | LSTM | W5: 7 landmarks | A — Baseline (hidden=128, layers=1) | 98.9 ± 2.3 | 27.7 ± 2.4 | 38.9 ± 3.8 | 0.967 ± 0.006 |
| 5 | GRU | W5: 7 landmarks | B — Deeper (hidden=128, layers=2) | 98.3 ± 2.7 | 29.1 ± 3.0 | 40.6 ± 4.4 | 0.964 ± 0.008 |
| 6 | LSTM | W5: 7 landmarks | C — Wider (hidden=256, layers=1) | 97.2 ± 3.9 | 30.6 ± 3.5 | 43.3 ± 4.5 | 0.958 ± 0.009 |

Full results for all 150 model × window × config combinations are in [Results/global_sweep_report.md](Results/global_sweep_report.md), with per-architecture breakdowns in [Results/gru_sweep_report.md](Results/gru_sweep_report.md), [Results/lstm_sweep_report.md](Results/lstm_sweep_report.md), [Results/transformer_sweep_report.md](Results/transformer_sweep_report.md), [Results/cnn_gru_sweep_report.md](Results/cnn_gru_sweep_report.md), and [Results/cnn_lstm_sweep_report.md](Results/cnn_lstm_sweep_report.md).

### Best Config per Model Class

Top-performing window/config combination for each of the five architectures (mean ± std over 10 seeds):

| Model | Window | Config | Accuracy (%) | MAE (cycles) | RMSE (cycles) | R² |
|-------|--------|--------|-------------|--------------|--------------|-----|
| **GRU** | W5: 7 landmarks | E — L1 + Small (hidden=64, layers=2) | **100.0 ± 0.0** | 32.2 ± 2.4 | 41.9 ± 3.1 | 0.961 ± 0.006 |
| **LSTM** | W5: 7 landmarks | B — Deeper (hidden=128, layers=2) | 99.4 ± 1.8 | **28.5 ± 2.3** | **37.9 ± 3.0** | **0.968 ± 0.005** |
| **Transformer** | W4: three bands | A — Baseline (d_model=64, nhead=4, layers=2) | 92.2 ± 3.9 | 34.4 ± 4.8 | 46.1 ± 7.9 | 0.952 ± 0.017 |
| **CNN-LSTM** | W3: early+late | A — Baseline (cnn=64, lstm=128, layers=1) | 88.9 ± 8.3 | 38.9 ± 10.2 | 56.7 ± 17.2 | 0.924 ± 0.049 |
| **CNN-GRU** | W5: 7 landmarks | E — L1 + Small (cnn=32, gru=64, layers=2) | 86.7 ± 6.5 | 46.7 ± 6.5 | 63.3 ± 10.5 | 0.910 ± 0.029 |

Recurrent models without CNN front-ends (GRU, LSTM) dominate; adding a CNN feature extractor (CNN-GRU, CNN-LSTM) hurts accuracy on this dataset rather than helping it.

**Key findings:**

- The **GRU and LSTM on the sparse landmark window (W5: cycles 1, 10, 50, 100, 150, 200, 250)** are now the top performers, both reaching ≥98% accuracy within ±10% EOL with R² ≈ 0.96–0.97 — GRU/LSTM overtook the previously-best W2 (stride-10) window after the latest config retune.
- Recurrent models (GRU, LSTM) clearly outperform their CNN-augmented counterparts: best CNN-GRU tops out at 86.7% (W5) and best CNN-LSTM at 88.9% (W3), both well behind plain GRU/LSTM.
- **W5 and W4 consistently outperform W0 and W1** for every architecture, confirming that even a handful of well-chosen cycles beyond cycle 100 carry more degradation signal than a dense early-life-only window.
- Early-life-only windows (W0: first 50 cycles) top out around 73–81% accuracy across architectures — usable, but well short of what late-cycle landmarks provide.
- The near-perfect W5 accuracy (100.0 ± 0.0 for GRU) is on a held-out test set of only 18 cells; treat it as an upper bound rather than a generalization guarantee, and see it as a candidate for a leakage/overfitting sanity check before quoting it as a final result.

---

## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone <repo-url>
cd Data-Driven-prediction-of-Battery-Lifespan

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**

| Package | Version |
|---------|---------|
| torch | 2.12.0 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| xgboost | 2.0.3 |
| scipy | 1.17.1 |
| matplotlib | 3.11.0 |
| tqdm | 4.68.2 |

---

## Usage

All commands below are run from the **repository root** (with the virtualenv activated) so that the `featureExtrcation` package is importable and the relative artifact paths resolve correctly.

### Step 1 — Preprocess the Dataset

```bash
python featureExtrcation/processDatasets.py
```

This reads raw per-cell `.pkl` files from `RAW_PKL_PATH` (hard-coded at the bottom of the script — edit it to point at your local dataset copy), extracts 44 per-cycle features for every cell, and writes the cache to `processed_hust_MIT_cache.pkl` **in the current working directory** (the repo root, if run as above). Subsequent runs of this script load from that cache and skip reprocessing.

Every model script (`Models/*.py`) instead expects the cache at `Models/processed_hust_MIT_cache.pkl`. Move or copy it there once after generating it:

```bash
mv processed_hust_MIT_cache.pkl Models/
```

A pre-generated cache is already committed at [Models/processed_hust_MIT_cache.pkl](Models/processed_hust_MIT_cache.pkl), so this step can be skipped entirely if you just want to train/evaluate models against the existing feature set.

### Step 2 — Train a Model

Each model script can be run directly. The cache and cell split are loaded automatically.

```bash
# Transformer
python Models/MIT_transformer_EOL.py

# GRU (best single-run model)
python Models/GRU_seq2seq.py

# LSTM
python Models/LSTM_seq2seq.py

# CNN-GRU
python Models/CNN_GRU_EOL.py

# CNN-LSTM
python Models/CNN_LSTM_EOL.py
```

On completion each script:
1. Prints per-epoch training progress and final metrics (MAE, RMSE, R², accuracy)
2. Displays a 4-panel evaluation plot (predicted vs true, error histogram, residuals, CDF)
3. Saves the best model checkpoint and XGBoost classifier to disk (controlled by `EOL_SAVE_ARTIFACTS`)

### Step 3 — Run a Config Sweep

The sweep runners execute all 5 configs × 10 seeds for one model family and write a JSON results file plus a markdown report.

```bash
# GRU sweep (W5 window — 7 landmark cycles)
python Models/run_gru_config_sweep.py

# LSTM sweep
python Models/run_lstm_config_sweep.py

# Transformer sweep
python Models/run_transformer_config_sweep.py

# CNN-GRU sweep
python Models/run_cnn_gru_config_sweep.py

# CNN-LSTM sweep
python Models/run_cnn_lstm_config_sweep.py
```

The active cycle window is controlled by the `EOL_CYCLES_TO_USE` environment variable (see below).

To run the CNN-GRU sweep on a SLURM cluster instead of locally, use the batch wrapper (edit `#SBATCH --partition` first):

```bash
sbatch Models/run_cnn_gru_sweep.sh
```

---

## Environment Variables

All model scripts and sweep runners are fully configurable via environment variables — no source edits needed.

| Variable | Default | Description |
|---|---|---|
| `EOL_CYCLES_TO_USE` | model-specific | Comma-separated cycle indices to use as input (e.g. `1,10,50,100,150,200,250`) |
| `EOL_DQDV_WINDOW` | `all` | Which dQ/dV voltage window(s) feed the model: `all` (every window's 24 features), `none` (drop dQ/dV, keep only the 20 non-dQ/dV features), or an integer `0–7` (that single window's 3 stats + the non-dQ/dV features) |
| `EOL_SEED` | `42` | Global random seed |
| `EOL_SHOW_PLOTS` | `1` | Set to `0` to suppress matplotlib windows |
| `EOL_SAVE_ARTIFACTS` | `1` | Set to `0` to skip saving model checkpoints |
| `EOL_HIDDEN_SIZE` | `128` | GRU/LSTM hidden size |
| `EOL_NUM_LAYERS` | `1` | Number of recurrent layers |
| `EOL_HEAD_DROPOUT` | `0.3` | Dropout in the MLP prediction head |
| `EOL_EPOCHS` | `300` | Maximum training epochs |
| `EOL_BATCH_SIZE` | `4` | Mini-batch size |
| `EOL_LR` | `3e-4` | Initial learning rate |
| `EOL_WEIGHT_DECAY` | `5e-3` | AdamW weight decay |
| `EOL_LAMBDA_EOL` | `1.5` | Loss scale factor |
| `EOL_LOSS` | `smooth_l1` | Loss function: `smooth_l1` or `l1` |
| `EOL_SCHEDULER` | `cosine` | LR scheduler: `cosine` or `onecycle` |

**Example — run GRU with the W5 landmark window and no plots:**

```bash
EOL_CYCLES_TO_USE=1,10,50,100,150,200,250 EOL_SHOW_PLOTS=0 python Models/GRU_seq2seq.py
```

**Example — train on a single dQ/dV voltage window (e.g. window 3 = 2.6–2.8 V):**

```bash
EOL_DQDV_WINDOW=3 python Models/GRU_seq2seq.py
```

**Example — reproduce the best GRU result (W5 landmarks, Config E):**

```bash
EOL_CYCLES_TO_USE=1,10,50,100,150,200,250 \
  EOL_HIDDEN_SIZE=64 EOL_NUM_LAYERS=2 EOL_HEAD_DROPOUT=0.4 \
  EOL_LR=1e-4 EOL_WEIGHT_DECAY=0.02 EOL_LAMBDA_EOL=2.0 \
  EOL_LOSS=l1 EOL_SCHEDULER=cosine \
  python Models/GRU_seq2seq.py
```

---

## Saved Artifacts

Pre-trained checkpoints for the best single-seed runs are included in the repository root, each paired with a Stage-1 XGBoost classifier under `Models/`:

| Model | Checkpoint (repo root) | Stage-1 classifier (`Models/`) |
|---|---|---|
| Transformer | `best_eol_transformer_50_cycle_window.pt` | `aging_classifier_50_xgb.json` |
| GRU | `best_eol_gru_50_cycle_window.pt` | `aging_classifier_gru_50_xgb.json` |
| LSTM | `best_eol_lstm_50_cycle_window.pt` | `aging_classifier_lstm_50_xgb.json` |
| CNN-GRU | `best_eol_cnn_gru_50_cycle_window.pt` | `aging_classifier_cnn_gru_50_xgb.json` |
| CNN-LSTM | `best_eol_cnn_lstm_50_cycle_window.pt` | `aging_classifier_cnn_lstm_50_xgb.json` |

Each `.pt` checkpoint contains the full model state dict, feature normalisation statistics, target normalisation statistics, tertile thresholds, class-mean EOL priors, and the calibration blend coefficient α. Checkpoints are regenerated automatically by the corresponding model script in [Step 2](#step-2--train-a-model) whenever `EOL_SAVE_ARTIFACTS=1` (the default); only `best_eol_transformer_50_cycle_window.pt` is committed to the repo as a reference artifact.

---

## License

[MIT License](LICENSE)
