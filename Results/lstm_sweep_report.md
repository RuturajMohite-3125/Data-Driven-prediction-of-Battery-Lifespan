# LSTM Sweep Results

## Configurations

### Config A: A — Baseline (hidden=128, layers=1)

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 128 |
| `num_layers` | 1 |
| `head_dropout` | 0.3 |
| `epochs` | 300 |
| `batch_size` | 4 |
| `lr` | 0.0003 |
| `weight_decay` | 0.005 |
| `lambda_eol` | 1.5 |
| `loss` | smooth_l1 |
| `scheduler` | cosine |

### Config B: B — Deeper (hidden=128, layers=2)

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 128 |
| `num_layers` | 2 |
| `head_dropout` | 0.3 |
| `epochs` | 300 |
| `batch_size` | 8 |
| `lr` | 0.0002 |
| `weight_decay` | 0.005 |
| `lambda_eol` | 1.5 |
| `loss` | smooth_l1 |
| `scheduler` | cosine |

### Config C: C — Wider (hidden=256, layers=1)

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 256 |
| `num_layers` | 1 |
| `head_dropout` | 0.2 |
| `epochs` | 300 |
| `batch_size` | 4 |
| `lr` | 0.0003 |
| `weight_decay` | 0.01 |
| `lambda_eol` | 1.5 |
| `loss` | smooth_l1 |
| `scheduler` | cosine |

### Config D: D — OneCycleLR (hidden=128, layers=2)

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 128 |
| `num_layers` | 2 |
| `head_dropout` | 0.3 |
| `epochs` | 300 |
| `batch_size` | 4 |
| `lr` | 0.0005 |
| `weight_decay` | 0.005 |
| `lambda_eol` | 1.0 |
| `loss` | smooth_l1 |
| `scheduler` | onecycle |

### Config E: E — L1 Loss + Small (hidden=64, layers=2)

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 64 |
| `num_layers` | 2 |
| `head_dropout` | 0.4 |
| `epochs` | 300 |
| `batch_size` | 4 |
| `lr` | 0.0001 |
| `weight_decay` | 0.02 |
| `lambda_eol` | 2.0 |
| `loss` | l1 |
| `scheduler` | cosine |

## Results — Test Set (mean ± std across seeds)

| ID | Window | Config | Acc% | MAE | RMSE | R² |
|----|--------|--------|------|-----|------|-----|
| W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (hidden=64, layers=2) | 83.9±6.8 | 36.0±2.8 | 48.1±4.2 | 0.949±0.009 |
| W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (hidden=64, layers=2) | 82.2±5.4 | 35.4±2.9 | 46.0±3.6 | 0.953±0.008 |
| W2_C | W2: 1-250 stride-10 (25 pts) | C — Wider (hidden=256, layers=1) | 79.4±6.1 | 34.5±3.8 | 47.1±3.6 | 0.951±0.008 |
| W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (hidden=128, layers=1) | 78.3±7.2 | 34.7±4.0 | 46.6±6.6 | 0.951±0.013 |
| W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (hidden=128, layers=1) | 77.8±5.0 | 35.2±2.9 | 46.3±3.8 | 0.953±0.008 |
| W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (hidden=128, layers=1) | 77.8±6.1 | 35.3±3.8 | 47.7±5.1 | 0.950±0.011 |
| W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper (hidden=128, layers=2) | 75.6±6.7 | 35.9±2.2 | 46.2±2.3 | 0.953±0.005 |
| W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper (hidden=128, layers=2) | 73.9±7.5 | 37.9±1.9 | 47.7±2.8 | 0.950±0.006 |
| W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (hidden=128, layers=2) | 73.9±7.5 | 38.0±5.8 | 49.4±6.4 | 0.946±0.014 |
| W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (hidden=64, layers=2) | 73.9±5.6 | 35.7±3.7 | 48.9±4.0 | 0.947±0.008 |
| W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper (hidden=128, layers=2) | 73.3±3.3 | 34.0±1.4 | 47.4±2.9 | 0.950±0.006 |
| W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (hidden=128, layers=2) | 72.2±7.9 | 39.0±5.6 | 53.0±5.8 | 0.938±0.014 |
| W3_C | W3: early(10-40)+late(180-200) [original] | C — Wider (hidden=256, layers=1) | 71.1±6.5 | 40.4±3.8 | 52.2±5.2 | 0.940±0.012 |
| W4_C | W4: three bands 1-50,100-150,200-250 | C — Wider (hidden=256, layers=1) | 71.1±6.5 | 39.8±5.6 | 52.3±5.7 | 0.939±0.014 |
| W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper (hidden=128, layers=2) | 70.0±5.7 | 40.3±2.6 | 51.3±2.5 | 0.942±0.006 |
| W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (hidden=128, layers=2) | 69.4±6.2 | 40.1±4.6 | 51.8±3.9 | 0.941±0.009 |
| W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (hidden=128, layers=2) | 68.9±8.7 | 42.4±8.2 | 53.5±10.0 | 0.935±0.024 |
| W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Wider (hidden=256, layers=1) | 68.3±3.6 | 43.5±3.3 | 58.3±4.2 | 0.925±0.011 |
| W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (hidden=128, layers=1) | 67.8±7.8 | 43.2±4.7 | 55.1±5.9 | 0.933±0.014 |
| W0_D | W0: first 50 cycles | D — OneCycleLR (hidden=128, layers=2) | 63.3±7.5 | 55.2±4.6 | 72.5±5.4 | 0.884±0.017 |
| W1_B | W1: first 100 cycles | B — Deeper (hidden=128, layers=2) | 62.2±4.8 | 58.7±4.2 | 76.7±5.2 | 0.870±0.018 |
| W0_B | W0: first 50 cycles | B — Deeper (hidden=128, layers=2) | 62.2±8.2 | 55.9±5.4 | 71.5±6.7 | 0.887±0.022 |
| W0_A | W0: first 50 cycles | A — Baseline (hidden=128, layers=1) | 62.2±8.2 | 56.3±5.6 | 72.4±7.4 | 0.884±0.023 |
| W1_A | W1: first 100 cycles | A — Baseline (hidden=128, layers=1) | 61.7±8.0 | 56.0±5.3 | 72.3±5.9 | 0.885±0.019 |
| W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (hidden=64, layers=2) | 60.0±4.8 | 44.1±4.9 | 58.2±7.0 | 0.925±0.019 |
| W1_C | W1: first 100 cycles | C — Wider (hidden=256, layers=1) | 59.4±6.1 | 61.1±4.7 | 78.4±7.8 | 0.864±0.029 |
| W1_E | W1: first 100 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 58.3±5.7 | 57.9±4.6 | 74.2±6.7 | 0.878±0.022 |
| W1_D | W1: first 100 cycles | D — OneCycleLR (hidden=128, layers=2) | 58.3±5.1 | 59.8±4.5 | 77.4±4.8 | 0.868±0.017 |
| W0_E | W0: first 50 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 57.8±6.2 | 56.8±3.3 | 76.0±4.3 | 0.873±0.015 |
| W0_C | W0: first 50 cycles | C — Wider (hidden=256, layers=1) | 55.0±8.8 | 59.0±6.0 | 74.2±6.5 | 0.879±0.022 |

## Per-Window Breakdown

### W0: first 50 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W0_D | D — OneCycleLR (hidden=128, layers=2) | 63.3±7.5 | 55.2±4.6 | 72.5±5.4 | 0.884±0.017 |
| W0_B | B — Deeper (hidden=128, layers=2) | 62.2±8.2 | 55.9±5.4 | 71.5±6.7 | 0.887±0.022 |
| W0_A | A — Baseline (hidden=128, layers=1) | 62.2±8.2 | 56.3±5.6 | 72.4±7.4 | 0.884±0.023 |
| W0_E | E — L1 Loss + Small (hidden=64, layers=2) | 57.8±6.2 | 56.8±3.3 | 76.0±4.3 | 0.873±0.015 |
| W0_C | C — Wider (hidden=256, layers=1) | 55.0±8.8 | 59.0±6.0 | 74.2±6.5 | 0.879±0.022 |

### W1: first 100 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W1_B | B — Deeper (hidden=128, layers=2) | 62.2±4.8 | 58.7±4.2 | 76.7±5.2 | 0.870±0.018 |
| W1_A | A — Baseline (hidden=128, layers=1) | 61.7±8.0 | 56.0±5.3 | 72.3±5.9 | 0.885±0.019 |
| W1_C | C — Wider (hidden=256, layers=1) | 59.4±6.1 | 61.1±4.7 | 78.4±7.8 | 0.864±0.029 |
| W1_E | E — L1 Loss + Small (hidden=64, layers=2) | 58.3±5.7 | 57.9±4.6 | 74.2±6.7 | 0.878±0.022 |
| W1_D | D — OneCycleLR (hidden=128, layers=2) | 58.3±5.1 | 59.8±4.5 | 77.4±4.8 | 0.868±0.017 |

### W2: 1-250 stride-10 (25 pts)

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W2_E | E — L1 Loss + Small (hidden=64, layers=2) | 82.2±5.4 | 35.4±2.9 | 46.0±3.6 | 0.953±0.008 |
| W2_C | C — Wider (hidden=256, layers=1) | 79.4±6.1 | 34.5±3.8 | 47.1±3.6 | 0.951±0.008 |
| W2_A | A — Baseline (hidden=128, layers=1) | 78.3±7.2 | 34.7±4.0 | 46.6±6.6 | 0.951±0.013 |
| W2_B | B — Deeper (hidden=128, layers=2) | 73.9±7.5 | 37.9±1.9 | 47.7±2.8 | 0.950±0.006 |
| W2_D | D — OneCycleLR (hidden=128, layers=2) | 73.9±7.5 | 38.0±5.8 | 49.4±6.4 | 0.946±0.014 |

### W3: early(10-40)+late(180-200) [original]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W3_C | C — Wider (hidden=256, layers=1) | 71.1±6.5 | 40.4±3.8 | 52.2±5.2 | 0.940±0.012 |
| W3_B | B — Deeper (hidden=128, layers=2) | 70.0±5.7 | 40.3±2.6 | 51.3±2.5 | 0.942±0.006 |
| W3_D | D — OneCycleLR (hidden=128, layers=2) | 68.9±8.7 | 42.4±8.2 | 53.5±10.0 | 0.935±0.024 |
| W3_A | A — Baseline (hidden=128, layers=1) | 67.8±7.8 | 43.2±4.7 | 55.1±5.9 | 0.933±0.014 |
| W3_E | E — L1 Loss + Small (hidden=64, layers=2) | 60.0±4.8 | 44.1±4.9 | 58.2±7.0 | 0.925±0.019 |

### W4: three bands 1-50,100-150,200-250

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W4_A | A — Baseline (hidden=128, layers=1) | 77.8±6.1 | 35.3±3.8 | 47.7±5.1 | 0.950±0.011 |
| W4_E | E — L1 Loss + Small (hidden=64, layers=2) | 73.9±5.6 | 35.7±3.7 | 48.9±4.0 | 0.947±0.008 |
| W4_B | B — Deeper (hidden=128, layers=2) | 73.3±3.3 | 34.0±1.4 | 47.4±2.9 | 0.950±0.006 |
| W4_D | D — OneCycleLR (hidden=128, layers=2) | 72.2±7.9 | 39.0±5.6 | 53.0±5.8 | 0.938±0.014 |
| W4_C | C — Wider (hidden=256, layers=1) | 71.1±6.5 | 39.8±5.6 | 52.3±5.7 | 0.939±0.014 |

### W5: sparse landmarks [1,10,50,100,150,200,250]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W5_E | E — L1 Loss + Small (hidden=64, layers=2) | 83.9±6.8 | 36.0±2.8 | 48.1±4.2 | 0.949±0.009 |
| W5_A | A — Baseline (hidden=128, layers=1) | 77.8±5.0 | 35.2±2.9 | 46.3±3.8 | 0.953±0.008 |
| W5_B | B — Deeper (hidden=128, layers=2) | 75.6±6.7 | 35.9±2.2 | 46.2±2.3 | 0.953±0.005 |
| W5_D | D — OneCycleLR (hidden=128, layers=2) | 69.4±6.2 | 40.1±4.6 | 51.8±3.9 | 0.941±0.009 |
| W5_C | C — Wider (hidden=256, layers=1) | 68.3±3.6 | 43.5±3.3 | 58.3±4.2 | 0.925±0.011 |
