# GRU Sweep Results

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
| W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (hidden=64, layers=2) | 100.0±0.0 | 32.2±2.4 | 41.9±3.1 | 0.961±0.006 |
| W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (hidden=128, layers=1) | 98.9±2.3 | 28.9±3.0 | 40.2±5.2 | 0.964±0.009 |
| W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper (hidden=128, layers=2) | 98.3±2.7 | 29.1±3.0 | 40.6±4.4 | 0.964±0.008 |
| W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Wider (hidden=256, layers=1) | 97.2±2.9 | 31.3±1.8 | 42.1±2.1 | 0.961±0.004 |
| W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (hidden=128, layers=2) | 94.4±2.6 | 33.1±4.3 | 50.0±7.4 | 0.944±0.016 |
| W4_C | W4: three bands 1-50,100-150,200-250 | C — Wider (hidden=256, layers=1) | 92.8±5.9 | 31.9±2.0 | 44.0±2.9 | 0.957±0.006 |
| W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (hidden=64, layers=2) | 92.8±2.7 | 31.5±2.2 | 42.6±3.2 | 0.960±0.006 |
| W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (hidden=128, layers=1) | 92.2±5.4 | 31.5±4.2 | 45.2±6.4 | 0.954±0.012 |
| W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (hidden=128, layers=1) | 91.1±3.9 | 32.6±4.0 | 43.4±4.6 | 0.958±0.009 |
| W2_C | W2: 1-250 stride-10 (25 pts) | C — Wider (hidden=256, layers=1) | 90.6±5.9 | 31.3±3.7 | 42.6±4.0 | 0.960±0.008 |
| W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper (hidden=128, layers=2) | 90.0±3.5 | 37.1±3.4 | 56.7±5.5 | 0.929±0.014 |
| W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (hidden=64, layers=2) | 88.9±4.5 | 34.2±2.4 | 50.0±4.5 | 0.945±0.010 |
| W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper (hidden=128, layers=2) | 88.9±5.2 | 33.4±3.1 | 44.1±5.1 | 0.957±0.010 |
| W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper (hidden=128, layers=2) | 88.3±4.9 | 31.6±2.9 | 45.9±6.3 | 0.953±0.013 |
| W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (hidden=128, layers=2) | 86.1±7.1 | 40.4±4.3 | 60.3±6.1 | 0.920±0.016 |
| W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (hidden=64, layers=2) | 85.0±2.7 | 37.1±3.4 | 48.6±5.1 | 0.948±0.011 |
| W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (hidden=128, layers=1) | 85.0±3.7 | 36.2±3.2 | 47.7±4.8 | 0.950±0.010 |
| W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (hidden=128, layers=2) | 84.4±4.4 | 41.2±5.0 | 55.3±6.9 | 0.932±0.018 |
| W3_C | W3: early(10-40)+late(180-200) [original] | C — Wider (hidden=256, layers=1) | 84.4±3.5 | 36.0±3.0 | 48.7±6.6 | 0.947±0.015 |
| W1_C | W1: first 100 cycles | C — Wider (hidden=256, layers=1) | 82.8±5.5 | 58.1±6.3 | 74.7±7.2 | 0.877±0.024 |
| W1_A | W1: first 100 cycles | A — Baseline (hidden=128, layers=1) | 82.2±5.1 | 54.5±4.2 | 72.5±5.0 | 0.884±0.016 |
| W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (hidden=128, layers=2) | 81.7±5.9 | 37.7±5.6 | 54.2±6.0 | 0.935±0.014 |
| W1_E | W1: first 100 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 80.6±5.4 | 54.8±3.0 | 70.8±3.4 | 0.890±0.011 |
| W0_E | W0: first 50 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 80.0±4.7 | 58.4±3.2 | 77.3±2.6 | 0.869±0.009 |
| W0_B | W0: first 50 cycles | B — Deeper (hidden=128, layers=2) | 80.0±6.5 | 60.5±4.0 | 78.7±4.1 | 0.864±0.014 |
| W1_B | W1: first 100 cycles | B — Deeper (hidden=128, layers=2) | 80.0±5.4 | 60.1±6.2 | 76.6±6.7 | 0.870±0.022 |
| W1_D | W1: first 100 cycles | D — OneCycleLR (hidden=128, layers=2) | 77.2±8.5 | 61.5±8.9 | 78.5±10.5 | 0.863±0.036 |
| W0_A | W0: first 50 cycles | A — Baseline (hidden=128, layers=1) | 76.1±8.3 | 60.1±8.4 | 76.2±10.5 | 0.870±0.037 |
| W0_D | W0: first 50 cycles | D — OneCycleLR (hidden=128, layers=2) | 75.0±7.1 | 61.8±5.9 | 77.1±8.2 | 0.868±0.028 |
| W0_C | W0: first 50 cycles | C — Wider (hidden=256, layers=1) | 74.4±6.5 | 64.4±7.9 | 83.8±10.5 | 0.844±0.041 |

## Per-Window Breakdown

### W0: first 50 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W0_E | E — L1 Loss + Small (hidden=64, layers=2) | 80.0±4.7 | 58.4±3.2 | 77.3±2.6 | 0.869±0.009 |
| W0_B | B — Deeper (hidden=128, layers=2) | 80.0±6.5 | 60.5±4.0 | 78.7±4.1 | 0.864±0.014 |
| W0_A | A — Baseline (hidden=128, layers=1) | 76.1±8.3 | 60.1±8.4 | 76.2±10.5 | 0.870±0.037 |
| W0_D | D — OneCycleLR (hidden=128, layers=2) | 75.0±7.1 | 61.8±5.9 | 77.1±8.2 | 0.868±0.028 |
| W0_C | C — Wider (hidden=256, layers=1) | 74.4±6.5 | 64.4±7.9 | 83.8±10.5 | 0.844±0.041 |

### W1: first 100 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W1_C | C — Wider (hidden=256, layers=1) | 82.8±5.5 | 58.1±6.3 | 74.7±7.2 | 0.877±0.024 |
| W1_A | A — Baseline (hidden=128, layers=1) | 82.2±5.1 | 54.5±4.2 | 72.5±5.0 | 0.884±0.016 |
| W1_E | E — L1 Loss + Small (hidden=64, layers=2) | 80.6±5.4 | 54.8±3.0 | 70.8±3.4 | 0.890±0.011 |
| W1_B | B — Deeper (hidden=128, layers=2) | 80.0±5.4 | 60.1±6.2 | 76.6±6.7 | 0.870±0.022 |
| W1_D | D — OneCycleLR (hidden=128, layers=2) | 77.2±8.5 | 61.5±8.9 | 78.5±10.5 | 0.863±0.036 |

### W2: 1-250 stride-10 (25 pts)

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W2_E | E — L1 Loss + Small (hidden=64, layers=2) | 92.8±2.7 | 31.5±2.2 | 42.6±3.2 | 0.960±0.006 |
| W2_A | A — Baseline (hidden=128, layers=1) | 91.1±3.9 | 32.6±4.0 | 43.4±4.6 | 0.958±0.009 |
| W2_C | C — Wider (hidden=256, layers=1) | 90.6±5.9 | 31.3±3.7 | 42.6±4.0 | 0.960±0.008 |
| W2_B | B — Deeper (hidden=128, layers=2) | 88.3±4.9 | 31.6±2.9 | 45.9±6.3 | 0.953±0.013 |
| W2_D | D — OneCycleLR (hidden=128, layers=2) | 81.7±5.9 | 37.7±5.6 | 54.2±6.0 | 0.935±0.014 |

### W3: early(10-40)+late(180-200) [original]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W3_B | B — Deeper (hidden=128, layers=2) | 88.9±5.2 | 33.4±3.1 | 44.1±5.1 | 0.957±0.010 |
| W3_E | E — L1 Loss + Small (hidden=64, layers=2) | 85.0±2.7 | 37.1±3.4 | 48.6±5.1 | 0.948±0.011 |
| W3_A | A — Baseline (hidden=128, layers=1) | 85.0±3.7 | 36.2±3.2 | 47.7±4.8 | 0.950±0.010 |
| W3_D | D — OneCycleLR (hidden=128, layers=2) | 84.4±4.4 | 41.2±5.0 | 55.3±6.9 | 0.932±0.018 |
| W3_C | C — Wider (hidden=256, layers=1) | 84.4±3.5 | 36.0±3.0 | 48.7±6.6 | 0.947±0.015 |

### W4: three bands 1-50,100-150,200-250

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W4_C | C — Wider (hidden=256, layers=1) | 92.8±5.9 | 31.9±2.0 | 44.0±2.9 | 0.957±0.006 |
| W4_A | A — Baseline (hidden=128, layers=1) | 92.2±5.4 | 31.5±4.2 | 45.2±6.4 | 0.954±0.012 |
| W4_B | B — Deeper (hidden=128, layers=2) | 90.0±3.5 | 37.1±3.4 | 56.7±5.5 | 0.929±0.014 |
| W4_E | E — L1 Loss + Small (hidden=64, layers=2) | 88.9±4.5 | 34.2±2.4 | 50.0±4.5 | 0.945±0.010 |
| W4_D | D — OneCycleLR (hidden=128, layers=2) | 86.1±7.1 | 40.4±4.3 | 60.3±6.1 | 0.920±0.016 |

### W5: sparse landmarks [1,10,50,100,150,200,250]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W5_E | E — L1 Loss + Small (hidden=64, layers=2) | 100.0±0.0 | 32.2±2.4 | 41.9±3.1 | 0.961±0.006 |
| W5_A | A — Baseline (hidden=128, layers=1) | 98.9±2.3 | 28.9±3.0 | 40.2±5.2 | 0.964±0.009 |
| W5_B | B — Deeper (hidden=128, layers=2) | 98.3±2.7 | 29.1±3.0 | 40.6±4.4 | 0.964±0.008 |
| W5_C | C — Wider (hidden=256, layers=1) | 97.2±2.9 | 31.3±1.8 | 42.1±2.1 | 0.961±0.004 |
| W5_D | D — OneCycleLR (hidden=128, layers=2) | 94.4±2.6 | 33.1±4.3 | 50.0±7.4 | 0.944±0.016 |
