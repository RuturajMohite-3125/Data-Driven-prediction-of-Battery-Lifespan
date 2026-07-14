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
| W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper (hidden=128, layers=2) | 99.4±1.8 | 28.5±2.3 | 37.9±3.0 | 0.968±0.005 |
| W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (hidden=128, layers=1) | 98.9±2.3 | 27.7±2.4 | 38.9±3.8 | 0.967±0.006 |
| W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Wider (hidden=256, layers=1) | 97.2±3.9 | 30.6±3.5 | 43.3±4.5 | 0.958±0.009 |
| W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (hidden=64, layers=2) | 97.2±2.9 | 31.8±2.2 | 39.9±2.8 | 0.965±0.005 |
| W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (hidden=128, layers=2) | 93.9±3.2 | 32.0±5.2 | 46.3±4.8 | 0.952±0.010 |
| W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (hidden=128, layers=1) | 93.3±4.4 | 32.5±2.2 | 45.6±3.5 | 0.954±0.007 |
| W4_C | W4: three bands 1-50,100-150,200-250 | C — Wider (hidden=256, layers=1) | 92.2±7.9 | 35.1±3.4 | 48.4±6.5 | 0.948±0.014 |
| W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (hidden=64, layers=2) | 92.2±2.9 | 32.1±2.6 | 43.5±4.3 | 0.958±0.009 |
| W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper (hidden=128, layers=2) | 91.1±2.9 | 28.4±1.7 | 43.8±3.1 | 0.958±0.006 |
| W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (hidden=128, layers=1) | 90.0±7.8 | 35.5±2.5 | 47.4±3.6 | 0.950±0.008 |
| W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (hidden=128, layers=1) | 89.4±4.1 | 32.1±4.1 | 46.7±5.1 | 0.952±0.010 |
| W2_C | W2: 1-250 stride-10 (25 pts) | C — Wider (hidden=256, layers=1) | 89.4±6.1 | 33.4±3.1 | 46.7±4.6 | 0.952±0.010 |
| W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper (hidden=128, layers=2) | 88.9±4.5 | 33.7±4.7 | 49.1±6.4 | 0.946±0.014 |
| W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (hidden=128, layers=2) | 87.8±5.1 | 32.5±5.0 | 48.7±4.9 | 0.948±0.011 |
| W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (hidden=64, layers=2) | 86.7±2.9 | 37.1±2.2 | 53.1±2.7 | 0.938±0.006 |
| W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper (hidden=128, layers=2) | 86.7±4.7 | 35.4±3.1 | 48.8±4.8 | 0.947±0.010 |
| W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (hidden=64, layers=2) | 86.1±4.7 | 35.3±2.5 | 47.6±3.1 | 0.950±0.007 |
| W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (hidden=128, layers=2) | 85.0±8.3 | 38.5±5.4 | 54.6±5.1 | 0.934±0.013 |
| W3_C | W3: early(10-40)+late(180-200) [original] | C — Wider (hidden=256, layers=1) | 83.9±3.2 | 37.6±3.9 | 50.7±5.9 | 0.943±0.013 |
| W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (hidden=128, layers=2) | 83.9±6.1 | 40.0±4.0 | 52.8±3.9 | 0.939±0.009 |
| W0_C | W0: first 50 cycles | C — Wider (hidden=256, layers=1) | 81.1±6.0 | 56.3±8.7 | 72.0±11.0 | 0.884±0.038 |
| W0_E | W0: first 50 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 80.0±5.4 | 54.7±2.8 | 74.0±3.9 | 0.880±0.013 |
| W1_A | W1: first 100 cycles | A — Baseline (hidden=128, layers=1) | 80.0±6.0 | 56.7±7.2 | 72.2±9.6 | 0.884±0.033 |
| W1_E | W1: first 100 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 78.9±4.4 | 54.8±3.2 | 69.1±3.9 | 0.895±0.012 |
| W0_D | W0: first 50 cycles | D — OneCycleLR (hidden=128, layers=2) | 77.2±7.6 | 58.6±8.4 | 74.2±10.0 | 0.877±0.032 |
| W1_D | W1: first 100 cycles | D — OneCycleLR (hidden=128, layers=2) | 77.2±7.6 | 58.1±7.7 | 72.4±10.1 | 0.883±0.035 |
| W0_A | W0: first 50 cycles | A — Baseline (hidden=128, layers=1) | 76.7±5.1 | 57.7±6.3 | 73.8±6.2 | 0.880±0.020 |
| W1_C | W1: first 100 cycles | C — Wider (hidden=256, layers=1) | 76.7±6.3 | 56.9±6.6 | 72.3±7.3 | 0.884±0.023 |
| W0_B | W0: first 50 cycles | B — Deeper (hidden=128, layers=2) | 76.7±5.7 | 58.6±3.9 | 77.4±4.5 | 0.868±0.015 |
| W1_B | W1: first 100 cycles | B — Deeper (hidden=128, layers=2) | 75.6±6.0 | 57.6±4.9 | 72.3±5.5 | 0.885±0.018 |

## Per-Window Breakdown

### W0: first 50 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W0_C | C — Wider (hidden=256, layers=1) | 81.1±6.0 | 56.3±8.7 | 72.0±11.0 | 0.884±0.038 |
| W0_E | E — L1 Loss + Small (hidden=64, layers=2) | 80.0±5.4 | 54.7±2.8 | 74.0±3.9 | 0.880±0.013 |
| W0_D | D — OneCycleLR (hidden=128, layers=2) | 77.2±7.6 | 58.6±8.4 | 74.2±10.0 | 0.877±0.032 |
| W0_A | A — Baseline (hidden=128, layers=1) | 76.7±5.1 | 57.7±6.3 | 73.8±6.2 | 0.880±0.020 |
| W0_B | B — Deeper (hidden=128, layers=2) | 76.7±5.7 | 58.6±3.9 | 77.4±4.5 | 0.868±0.015 |

### W1: first 100 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W1_A | A — Baseline (hidden=128, layers=1) | 80.0±6.0 | 56.7±7.2 | 72.2±9.6 | 0.884±0.033 |
| W1_E | E — L1 Loss + Small (hidden=64, layers=2) | 78.9±4.4 | 54.8±3.2 | 69.1±3.9 | 0.895±0.012 |
| W1_D | D — OneCycleLR (hidden=128, layers=2) | 77.2±7.6 | 58.1±7.7 | 72.4±10.1 | 0.883±0.035 |
| W1_C | C — Wider (hidden=256, layers=1) | 76.7±6.3 | 56.9±6.6 | 72.3±7.3 | 0.884±0.023 |
| W1_B | B — Deeper (hidden=128, layers=2) | 75.6±6.0 | 57.6±4.9 | 72.3±5.5 | 0.885±0.018 |

### W2: 1-250 stride-10 (25 pts)

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W2_E | E — L1 Loss + Small (hidden=64, layers=2) | 92.2±2.9 | 32.1±2.6 | 43.5±4.3 | 0.958±0.009 |
| W2_B | B — Deeper (hidden=128, layers=2) | 91.1±2.9 | 28.4±1.7 | 43.8±3.1 | 0.958±0.006 |
| W2_A | A — Baseline (hidden=128, layers=1) | 89.4±4.1 | 32.1±4.1 | 46.7±5.1 | 0.952±0.010 |
| W2_C | C — Wider (hidden=256, layers=1) | 89.4±6.1 | 33.4±3.1 | 46.7±4.6 | 0.952±0.010 |
| W2_D | D — OneCycleLR (hidden=128, layers=2) | 87.8±5.1 | 32.5±5.0 | 48.7±4.9 | 0.948±0.011 |

### W3: early(10-40)+late(180-200) [original]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W3_A | A — Baseline (hidden=128, layers=1) | 90.0±7.8 | 35.5±2.5 | 47.4±3.6 | 0.950±0.008 |
| W3_B | B — Deeper (hidden=128, layers=2) | 86.7±4.7 | 35.4±3.1 | 48.8±4.8 | 0.947±0.010 |
| W3_E | E — L1 Loss + Small (hidden=64, layers=2) | 86.1±4.7 | 35.3±2.5 | 47.6±3.1 | 0.950±0.007 |
| W3_C | C — Wider (hidden=256, layers=1) | 83.9±3.2 | 37.6±3.9 | 50.7±5.9 | 0.943±0.013 |
| W3_D | D — OneCycleLR (hidden=128, layers=2) | 83.9±6.1 | 40.0±4.0 | 52.8±3.9 | 0.939±0.009 |

### W4: three bands 1-50,100-150,200-250

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W4_A | A — Baseline (hidden=128, layers=1) | 93.3±4.4 | 32.5±2.2 | 45.6±3.5 | 0.954±0.007 |
| W4_C | C — Wider (hidden=256, layers=1) | 92.2±7.9 | 35.1±3.4 | 48.4±6.5 | 0.948±0.014 |
| W4_B | B — Deeper (hidden=128, layers=2) | 88.9±4.5 | 33.7±4.7 | 49.1±6.4 | 0.946±0.014 |
| W4_E | E — L1 Loss + Small (hidden=64, layers=2) | 86.7±2.9 | 37.1±2.2 | 53.1±2.7 | 0.938±0.006 |
| W4_D | D — OneCycleLR (hidden=128, layers=2) | 85.0±8.3 | 38.5±5.4 | 54.6±5.1 | 0.934±0.013 |

### W5: sparse landmarks [1,10,50,100,150,200,250]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W5_B | B — Deeper (hidden=128, layers=2) | 99.4±1.8 | 28.5±2.3 | 37.9±3.0 | 0.968±0.005 |
| W5_A | A — Baseline (hidden=128, layers=1) | 98.9±2.3 | 27.7±2.4 | 38.9±3.8 | 0.967±0.006 |
| W5_C | C — Wider (hidden=256, layers=1) | 97.2±3.9 | 30.6±3.5 | 43.3±4.5 | 0.958±0.009 |
| W5_E | E — L1 Loss + Small (hidden=64, layers=2) | 97.2±2.9 | 31.8±2.2 | 39.9±2.8 | 0.965±0.005 |
| W5_D | D — OneCycleLR (hidden=128, layers=2) | 93.9±3.2 | 32.0±5.2 | 46.3±4.8 | 0.952±0.010 |
