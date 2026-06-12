# Sweep Configuration Reference

Config sweeps for three model families — **Transformer**, **GRU**, and **LSTM** — each run across **6 cycle windows × 5 configs × 10 seeds**.

---

## Cycle Windows (shared across all three sweeps)

| ID | Definition | # Points | Rationale |
|----|-----------|----------|-----------|
| W0 | cycles 1–50 | 50 | Early-life only; tests whether degradation signal is visible before capacity fade begins |
| W1 | cycles 1–100 | 100 | Extended early-life; captures the initial plateau and the onset of the knee |
| W2 | cycles 1–250, stride 10 | 25 | Sparse but uniform coverage across the full observed life; mimics coarse sampling |
| W3 | cycles 10–40 + 180–200 | 51 | Original working window: dense early + late snapshots, nothing in between |
| W4 | cycles 1–50, 100–150, 200–250 | 150 | Three dense bands at early, mid, and late life; balanced temporal coverage |
| W5 | {1, 10, 50, 100, 150, 200, 250} | 7 | Extreme sparsity; landmark points only — stress-tests whether models can generalise from very few observations |

---

## Transformer Configs (`run_transformer_config_sweep.py`)

The transformer uses `d_model`, `nhead`, `num_layers`, `dim_ff`, and `dropout` as model knobs, plus training hyperparameters and choice of loss function and LR scheduler.

### Config A — Baseline
```
d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.3
epochs=300, batch_size=4, lr=3e-4, weight_decay=5e-3, lambda_eol=1.5
loss=smooth_l1, scheduler=cosine
```
**Why:** This mirrors the architecture used in initial experiments that achieved the reference RMSE. It serves as the anchor point — all other configs are compared against it. The small `d_model=64` keeps training fast and avoids overfitting on the small battery dataset (~100 cells). `SmoothL1` is robust to the occasional outlier cell. Cosine annealing provides smooth LR decay without manual tuning.

---

### Config B — Deeper + Wider
```
d_model=128, nhead=8, num_layers=3, dim_ff=256, dropout=0.2
epochs=300, batch_size=8, lr=2e-4, weight_decay=5e-3, lambda_eol=1.5
loss=smooth_l1, scheduler=cosine
```
**Why:** Doubling `d_model` and adding an extra encoder layer tests whether the model is capacity-limited. `nhead=8` matches the larger `d_model` (keeping head dimension at 16). Dropout is slightly reduced because the larger model has more built-in regularisation from depth. Batch size 8 improves gradient stability at the larger scale. LR is lowered proportionally to avoid overshooting wider loss surfaces.

---

### Config C — Regularized Small
```
d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.4
epochs=300, batch_size=8, lr=3e-4, weight_decay=2e-2, lambda_eol=1.5
loss=smooth_l1, scheduler=cosine
```
**Why:** Same architecture as A but with stronger regularisation (`dropout=0.4`, `weight_decay=2e-2` vs `5e-3`). The battery dataset is small and the feature sequences are highly correlated cycle-to-cycle, making overfitting a real risk. This config tests whether the baseline is already under-regularised. Larger batch size reduces gradient noise without changing the model.

---

### Config D — OneCycleLR
```
d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.3
epochs=300, batch_size=4, lr=5e-4, weight_decay=5e-3, lambda_eol=1.0
loss=smooth_l1, scheduler=onecycle
```
**Why:** Identical model to A but swaps the LR schedule. OneCycleLR uses a warm-up phase followed by aggressive annealing — often converges faster and to sharper minima than cosine on small datasets. Peak LR is raised to `5e-4` since OneCycleLR expects a higher ceiling. `lambda_eol` is reduced to 1.0 (no amplification) to isolate the scheduler's effect from loss scaling.

---

### Config E — L1 Loss + Larger
```
d_model=96, nhead=4, num_layers=3, dim_ff=192, dropout=0.25
epochs=300, batch_size=4, lr=1e-4, weight_decay=1e-2, lambda_eol=2.0
loss=l1, scheduler=cosine
```
**Why:** Switches to pure L1 loss, which is fully median-seeking and gives zero gradient at exact predictions — potentially beneficial when a subset of cells are systematic outliers. `lambda_eol=2.0` compensates for L1's naturally smaller gradient magnitude. The model is slightly larger than A (`d_model=96`) but not as large as B, exploring the intermediate regime. Low LR (`1e-4`) prevents oscillation that pure L1 can cause early in training.

---

## GRU Configs (`run_gru_config_sweep.py`)

The GRU uses a bidirectional GRU with attention pooling. Model knobs are `hidden_size`, `num_layers`, and `head_dropout` (dropout in the MLP prediction head). RNN recurrent dropout is automatically `0.3` when `num_layers > 1`, matching the original implementation.

### Config A — Baseline
```
hidden_size=128, num_layers=1, head_dropout=0.3
epochs=300, batch_size=4, lr=3e-4, weight_decay=5e-3, lambda_eol=1.5
loss=smooth_l1, scheduler=cosine
```
**Why:** Reproduces the original GRU training setup as the reference point. A single bidirectional GRU layer with hidden size 128 gives 256-dimensional context vectors after concatenation — sufficient for the feature dimensionality of the battery dataset. No recurrent dropout with one layer keeps training stable.

---

### Config B — Deeper
```
hidden_size=128, num_layers=2, head_dropout=0.3
epochs=300, batch_size=8, lr=2e-4, weight_decay=5e-3, lambda_eol=1.5
loss=smooth_l1, scheduler=cosine
```
**Why:** Adding a second GRU layer tests whether the model benefits from hierarchical sequence abstractions. The second layer introduces recurrent dropout (`0.3`) automatically. Batch size is doubled and LR reduced to compensate for the increased gradient complexity from the deeper recurrence. This is the most natural extension of the baseline for sequence models.

---

### Config C — Wider
```
hidden_size=256, num_layers=1, head_dropout=0.2
epochs=300, batch_size=4, lr=3e-4, weight_decay=1e-2, lambda_eol=1.5
loss=smooth_l1, scheduler=cosine
```
**Why:** Doubles hidden size instead of depth. For short sequences (as few as 7 points in W5), extra width may be more useful than extra recurrent depth. The bidirectional output is now 512-dimensional, giving the attention head more expressive power. `head_dropout=0.2` is reduced because the wider representation is already more distributed. `weight_decay` is raised slightly to prevent the larger hidden state from memorising.

---

### Config D — OneCycleLR
```
hidden_size=128, num_layers=2, head_dropout=0.3
epochs=300, batch_size=4, lr=5e-4, weight_decay=5e-3, lambda_eol=1.0
loss=smooth_l1, scheduler=onecycle
```
**Why:** Deeper model (same as B) paired with OneCycleLR to test whether aggressive LR cycling helps escape poor local minima that deeper GRUs can get stuck in. The warm-up phase is particularly useful here because multi-layer GRUs are sensitive to initialisation. `lambda_eol=1.0` removes loss amplification to isolate the scheduler effect.

---

### Config E — L1 Loss + Small
```
hidden_size=64, num_layers=2, head_dropout=0.4
epochs=300, batch_size=4, lr=1e-4, weight_decay=2e-2, lambda_eol=2.0
loss=l1, scheduler=cosine
```
**Why:** A heavily regularised small model. `hidden_size=64` halves the parameter count, testing whether a compact GRU generalises better on the small dataset. High `head_dropout=0.4` and `weight_decay=2e-2` push strong regularisation. L1 loss with `lambda_eol=2.0` keeps gradient magnitudes comparable to SmoothL1 configs. This config is expected to underfit on longer windows (W1, W4) but may excel on sparse windows (W5) where overfitting is the dominant failure mode.

---

## LSTM Configs (`run_lstm_config_sweep.py`)

The LSTM architecture is structurally identical to the GRU (bidirectional + attention pooling + MLP head) but uses LSTM cells instead of GRU cells. LSTM has ~33% more parameters per layer due to the extra cell-state gate, making it slightly more expressive but also slower to train. The configs mirror the GRU sweep exactly to enable a direct apples-to-apples comparison.

### Config A — Baseline
```
hidden_size=128, num_layers=1, head_dropout=0.3
epochs=300, batch_size=4, lr=3e-4, weight_decay=5e-3, lambda_eol=1.5
loss=smooth_l1, scheduler=cosine
```
**Why:** Reproduces the original LSTM training setup. The extra cell state in LSTM allows longer-range memory than GRU, which may matter for windows spanning many cycles (W1, W2, W4). Using the same hyperparameters as the GRU baseline makes the GRU vs LSTM comparison clean.

---

### Config B — Deeper
```
hidden_size=128, num_layers=2, head_dropout=0.3
epochs=300, batch_size=8, lr=2e-4, weight_decay=5e-3, lambda_eol=1.5
loss=smooth_l1, scheduler=cosine
```
**Why:** Stacking two LSTM layers tests hierarchical temporal abstraction. LSTM's cell state makes deep stacking more stable than GRU in theory, so this config is of particular interest. Larger batch and lower LR mirror the GRU-B rationale: deeper recurrent networks need steadier gradient estimates.

---

### Config C — Wider
```
hidden_size=256, num_layers=1, head_dropout=0.2
epochs=300, batch_size=4, lr=3e-4, weight_decay=1e-2, lambda_eol=1.5
loss=smooth_l1, scheduler=cosine
```
**Why:** For LSTM, a wider single layer is a larger parameter increase than for GRU (four gates vs three), so this config explores whether the extra capacity is beneficial or harmful on the small dataset. `head_dropout=0.2` and slightly increased `weight_decay` provide compensating regularisation.

---

### Config D — OneCycleLR
```
hidden_size=128, num_layers=2, head_dropout=0.3
epochs=300, batch_size=4, lr=5e-4, weight_decay=5e-3, lambda_eol=1.0
loss=smooth_l1, scheduler=onecycle
```
**Why:** Same motivation as GRU-D. Multi-layer LSTMs are particularly prone to vanishing gradients in early training; the OneCycleLR warm-up phase can alleviate this. Comparing with GRU-D isolates whether the LSTM cell state or the scheduler is driving any observed improvement.

---

### Config E — L1 Loss + Small
```
hidden_size=64, num_layers=2, head_dropout=0.4
epochs=300, batch_size=4, lr=1e-4, weight_decay=2e-2, lambda_eol=2.0
loss=l1, scheduler=cosine
```
**Why:** Same rationale as GRU-E. A small, heavily regularised LSTM. Because LSTM has more parameters per unit of `hidden_size` than GRU, `hidden_size=64` here represents a smaller model in absolute terms than GRU-E. This makes it the most compact model tested across all three architectures, useful for understanding the minimum viable capacity for this task.

---

## Cross-Model Design Philosophy

The configs were designed with three goals in mind:

1. **Isolate one variable at a time.** Each config changes a small number of knobs relative to the baseline (A), so that differences in results can be attributed to specific design choices rather than confounded hyperparameter changes.

2. **Cover the regularisation axis.** Configs C and E represent two different regularisation strategies (wider+mild vs small+aggressive). On a ~100-cell dataset, overfitting is the dominant failure mode, so understanding where the regularisation sweet spot lies is critical.

3. **Enable direct architecture comparison.** Transformer, GRU, and LSTM use identical training hyperparameters in configs A–E, identical cycle windows, and identical seeds. This means the sweep results can be directly compared across architectures without controlling for training differences.
