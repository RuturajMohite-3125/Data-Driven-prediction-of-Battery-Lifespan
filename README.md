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
│   └── processDatasets.py          # Raw pkl → per-cycle feature cache
│
├── Models/
│   ├── MIT_transformer_EOL.py            # Transformer
│   ├── GRU_seq2seq.py                    # GRU 
│   ├── LSTM_seq2seq.py                   # LSTM
│   ├── CNN_GRU_EOL.py                    # CNN feature extractor + GRU
│   ├── CNN_LSTM_EOL.py                   # CNN feature extractor + LSTM
│   ├── run_transformer_config_sweep.py   # Sweep runner — Transformer
│   ├── run_gru_config_sweep.py           # Sweep runner — GRU
│   ├── run_lstm_config_sweep.py          # Sweep runner — LSTM
│   ├── run_cnn_gru_config_sweep.py       # Sweep runner — CNN-GRU
│   ├── run_cnn_lstm_config_sweep.py      # Sweep runner — CNN-LSTM
│   ├── cell_split.json                   # Fixed train/val/test cell assignments
│   ├── processed_hust_MIT_cache.pkl      # Cached feature extraction output
│   └── SWEEP_CONFIGS.md            # Full sweep configuration documentation
│
├── Results/
│   ├── global_sweep_report.md      # Full cross-model sweep results table
│   ├── gru_sweep_report.md         # GRU-specific sweep results
│   ├── lstm_sweep_report.md        # LSTM-specific sweep results
│   ├── transformer_sweep_report.md # Transformer-specific sweep results
│   └── cnn_gru_sweep_results.json  # Raw CNN-GRU sweep output --> to be rerun
│
├── requirements.txt
└── LICENSE
```

---

## Methodology

### Feature Extraction

[featureExtrcation/processDatasets.py](featureExtrcation/processDatasets.py) processes the raw per-cell `.pkl` files into a structured feature cache. For each cycle of each cell, **18 scalar features** are extracted:

| Feature Group | Features |
|---|---|
| dQ/dV statistics (vs cycle 10 baseline) | `dQdV_min_delta`, `dQdV_var_delta` |
| Log-scale std (charge phase) | `log_std_I_c`, `log_std_Q_c`, `log_std_V_c` |
| Log-scale std (discharge phase) | `log_std_I_d`, `log_std_Q_d`, `log_std_V_d` |
| Current range (charge) | `min_I_c`, `max_I_c` |
| Voltage range (charge) | `min_V_c`, `max_V_c` |
| Current range (discharge) | `min_I_d`, `max_I_d` |
| Voltage range (discharge) | `min_V_d`, `max_V_d` |
| Capacity + shape | `max_Q_d`, `kurtosis_V_d` |

The differential capacity curve dQ/dV is smoothed using a Savitzky-Golay filter (window=31, order=3). Delta features are computed relative to cycle 10 to capture degradation signatures.

Each cell is stored in the cache as:

```python
{
    'cell_name': str,
    'features': np.ndarray,   # shape [n_cycles, 18]
    'soh':      np.ndarray,   # discharge capacity per cycle (Ah)
    'soh_traj': np.ndarray,   # normalised SoH, padded to 1500 cycles (-1 sentinel beyond EOL)
    'eol':      int,          # cycle index where capacity < 88% of peak
    'num_cycles': int,
}
```

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

The model receives a sequence of per-cycle features (19–21 dimensions, depending on the appended class probabilities) and produces a single scalar EOL prediction. Training uses Z-score normalisation of targets and early stopping with 80-epoch patience.


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

Best results from the cross-model sweep (mean ± std over 10 random seeds). **Accuracy** = fraction of test cells with predicted EOL within ±10% of true EOL.

| Rank | Model | Window | Config | Accuracy (%) | MAE (cycles) | RMSE (cycles) | R² |
|------|-------|--------|--------|-------------|--------------|--------------|-----|
| 1 | **GRU** | W2: stride-10, 1–250 | A — Baseline (hidden=128, layers=1) | **85.0 ± 3.6** | **34.3 ± 3.1** | 44.4 ± 4.4 | 0.956 ± 0.008 |
| 2 | GRU | W5: 7 landmarks | E — L1 + Small (hidden=64, layers=2) | 83.9 ± 7.2 | 35.0 ± 2.0 | 51.7 ± 3.6 | 0.941 ± 0.008 |
| 3 | LSTM | W5: 7 landmarks | E — L1 + Small (hidden=64, layers=2) | 83.9 ± 6.8 | 36.0 ± 2.8 | 48.1 ± 4.2 | 0.949 ± 0.009 |
| 4 | GRU | W2: stride-10, 1–250 | C — Wider (hidden=256, layers=1) | 82.2 ± 6.0 | **32.5 ± 3.4** | 44.7 ± 3.9 | 0.956 ± 0.008 |
| 5 | LSTM | W2: stride-10, 1–250 | E — L1 + Small (hidden=64, layers=2) | 82.2 ± 5.4 | 35.4 ± 2.9 | 46.0 ± 3.6 | 0.953 ± 0.008 |
| 6 | Transformer | W5: 7 landmarks | A — Baseline | 81.7 ± 7.5 | 35.0 ± 5.8 | 45.3 ± 8.2 | 0.954 ± 0.016 |

Full results for all 97 model × window × config combinations are in [Results/global_sweep_report.md](Results/global_sweep_report.md).

**Key findings:**

- The **GRU with sparse uniform sampling (W2, stride-10)** is the top performer, reaching 85% accuracy within ±10% EOL, with MAE of ~34 cycles and R² ≈ 0.956.
- **Window W2 and W5 consistently outperform W0 and W1**, demonstrating that even sparse observations beyond cycle 100 significantly improve prediction over early-life-only windows.
- All three architectures (GRU, LSTM, Transformer) converge to similar accuracy on the best windows, with GRU having a slight edge in stability (lower std across seeds).
- Early-life-only windows (W0: first 50 cycles) achieve ~62–65% accuracy, confirming that mid-life measurements carry substantial degradation information.

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

All scripts are run from the **repository root** so that the `featureExtrcation` package is importable.

### Step 1 — Preprocess the Dataset

```bash
python featureExtrcation/processDatasets.py
```

This reads raw `.pkl` files from `RAW_PKL_PATH` (edit the path at the bottom of the script), extracts 18 per-cycle features for every cell, and writes the result to `Models/processed_hust_MIT_cache.pkl`. Subsequent runs load from cache and skip reprocessing.

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

---

## Environment Variables

All model scripts and sweep runners are fully configurable via environment variables — no source edits needed.

| Variable | Default | Description |
|---|---|---|
| `EOL_CYCLES_TO_USE` | model-specific | Comma-separated cycle indices to use as input (e.g. `1,10,50,100,150,200,250`) |
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

**Example — reproduce the best GRU result (W2, stride-10):**

```bash
EOL_CYCLES_TO_USE=$(python -c "print(','.join(str(i) for i in range(1,251,10)))") \
  EOL_HIDDEN_SIZE=128 EOL_NUM_LAYERS=1 EOL_LOSS=smooth_l1 EOL_SCHEDULER=cosine \
  python Models/GRU_seq2seq.py
```

---

## Saved Artifacts

Pre-trained checkpoints for the best single-seed runs are included in the repository root:

| File | Description |
|---|---|
| `best_eol_gru_50_cycle_window.pt` | Best GRU model (W5 window, Config A) |
| `best_eol_lstm_50_cycle_window.pt` | Best LSTM model (W5 window) |
| `best_eol_transformer_50_cycle_window.pt` | Best Transformer model (W5 window) |
| `Models/aging_classifier_gru_50_xgb.json` | XGBoost Stage-1 classifier for GRU pipeline |
| `Models/aging_classifier_lstm_50_xgb.json` | XGBoost Stage-1 classifier for LSTM pipeline |
| `Models/aging_classifier_50_xgb.json` | XGBoost Stage-1 classifier for Transformer pipeline |

Each `.pt` checkpoint contains the full model state dict, feature normalisation statistics, target normalisation statistics, tertile thresholds, class-mean EOL priors, and the calibration blend coefficient α.

---

## License

[MIT License](LICENSE)
