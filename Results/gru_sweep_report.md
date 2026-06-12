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
| W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (hidden=128, layers=1) | 85.0±3.6 | 34.3±3.1 | 44.4±4.4 | 0.956±0.008 |
| W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (hidden=64, layers=2) | 83.9±7.2 | 35.0±2.0 | 51.7±3.6 | 0.941±0.008 |
| W2_C | W2: 1-250 stride-10 (25 pts) | C — Wider (hidden=256, layers=1) | 82.2±6.0 | 32.5±3.4 | 44.7±3.9 | 0.956±0.008 |
| W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (hidden=128, layers=1) | 80.6±7.1 | 37.8±4.4 | 47.2±4.3 | 0.951±0.009 |
| W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper (hidden=128, layers=2) | 77.8±9.6 | 37.8±3.2 | 48.1±2.8 | 0.949±0.006 |
| W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Wider (hidden=256, layers=1) | 77.8±5.0 | 37.8±4.5 | 50.7±4.6 | 0.943±0.010 |
| W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (hidden=64, layers=2) | 77.8±5.0 | 38.0±1.9 | 50.9±4.0 | 0.943±0.009 |
| W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (hidden=128, layers=2) | 76.7±4.2 | 40.4±4.2 | 54.7±3.8 | 0.934±0.009 |
| W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper (hidden=128, layers=2) | 76.7±6.5 | 38.4±2.5 | 47.8±2.7 | 0.950±0.006 |
| W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (hidden=64, layers=2) | 75.6±5.1 | 40.4±1.6 | 57.8±2.9 | 0.927±0.007 |
| W4_C | W4: three bands 1-50,100-150,200-250 | C — Wider (hidden=256, layers=1) | 75.6±7.1 | 38.5±3.5 | 51.6±4.0 | 0.941±0.009 |
| W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper (hidden=128, layers=2) | 72.8±5.2 | 40.4±1.8 | 56.0±2.2 | 0.931±0.006 |
| W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (hidden=128, layers=2) | 72.8±7.2 | 40.4±5.0 | 52.8±4.3 | 0.938±0.010 |
| W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (hidden=128, layers=2) | 71.7±5.2 | 43.0±3.7 | 61.1±5.5 | 0.918±0.015 |
| W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (hidden=128, layers=1) | 71.7±7.2 | 39.7±3.9 | 53.2±3.6 | 0.938±0.009 |
| W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (hidden=128, layers=1) | 70.6±5.0 | 41.4±4.5 | 52.4±5.9 | 0.939±0.014 |
| W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper (hidden=128, layers=2) | 69.4±7.1 | 43.3±2.6 | 55.4±3.9 | 0.933±0.009 |
| W3_C | W3: early(10-40)+late(180-200) [original] | C — Wider (hidden=256, layers=1) | 68.9±7.1 | 44.6±4.3 | 56.8±6.4 | 0.928±0.016 |
| W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (hidden=128, layers=2) | 67.8±11.1 | 44.8±5.5 | 57.1±6.5 | 0.928±0.016 |
| W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (hidden=64, layers=2) | 67.2±6.3 | 46.2±1.4 | 61.1±1.9 | 0.918±0.005 |
| W0_D | W0: first 50 cycles | D — OneCycleLR (hidden=128, layers=2) | 66.7±5.0 | 57.5±7.7 | 72.6±9.1 | 0.883±0.030 |
| W1_C | W1: first 100 cycles | C — Wider (hidden=256, layers=1) | 63.9±3.7 | 57.8±4.3 | 76.3±6.1 | 0.872±0.020 |
| W0_A | W0: first 50 cycles | A — Baseline (hidden=128, layers=1) | 63.9±6.7 | 55.5±3.7 | 72.1±3.6 | 0.886±0.012 |
| W0_B | W0: first 50 cycles | B — Deeper (hidden=128, layers=2) | 62.8±5.6 | 54.8±3.4 | 72.2±4.0 | 0.886±0.012 |
| W1_A | W1: first 100 cycles | A — Baseline (hidden=128, layers=1) | 61.7±3.9 | 59.1±2.2 | 76.4±3.4 | 0.872±0.011 |
| W1_E | W1: first 100 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 61.7±3.9 | 52.1±4.9 | 69.3±5.8 | 0.894±0.018 |
| W0_C | W0: first 50 cycles | C — Wider (hidden=256, layers=1) | 61.7±11.5 | 61.4±12.7 | 78.8±18.3 | 0.856±0.078 |
| W1_B | W1: first 100 cycles | B — Deeper (hidden=128, layers=2) | 60.6±5.2 | 58.7±4.5 | 75.1±5.4 | 0.876±0.018 |
| W1_D | W1: first 100 cycles | D — OneCycleLR (hidden=128, layers=2) | 59.4±4.3 | 60.8±6.5 | 78.4±6.7 | 0.864±0.024 |
| W0_E | W0: first 50 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 54.4±7.8 | 57.2±3.2 | 76.6±2.7 | 0.871±0.009 |

## Per-Window Breakdown

### W0: first 50 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W0_D | D — OneCycleLR (hidden=128, layers=2) | 66.7±5.0 | 57.5±7.7 | 72.6±9.1 | 0.883±0.030 |
| W0_A | A — Baseline (hidden=128, layers=1) | 63.9±6.7 | 55.5±3.7 | 72.1±3.6 | 0.886±0.012 |
| W0_B | B — Deeper (hidden=128, layers=2) | 62.8±5.6 | 54.8±3.4 | 72.2±4.0 | 0.886±0.012 |
| W0_C | C — Wider (hidden=256, layers=1) | 61.7±11.5 | 61.4±12.7 | 78.8±18.3 | 0.856±0.078 |
| W0_E | E — L1 Loss + Small (hidden=64, layers=2) | 54.4±7.8 | 57.2±3.2 | 76.6±2.7 | 0.871±0.009 |

### W1: first 100 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W1_C | C — Wider (hidden=256, layers=1) | 63.9±3.7 | 57.8±4.3 | 76.3±6.1 | 0.872±0.020 |
| W1_A | A — Baseline (hidden=128, layers=1) | 61.7±3.9 | 59.1±2.2 | 76.4±3.4 | 0.872±0.011 |
| W1_E | E — L1 Loss + Small (hidden=64, layers=2) | 61.7±3.9 | 52.1±4.9 | 69.3±5.8 | 0.894±0.018 |
| W1_B | B — Deeper (hidden=128, layers=2) | 60.6±5.2 | 58.7±4.5 | 75.1±5.4 | 0.876±0.018 |
| W1_D | D — OneCycleLR (hidden=128, layers=2) | 59.4±4.3 | 60.8±6.5 | 78.4±6.7 | 0.864±0.024 |

### W2: 1-250 stride-10 (25 pts)

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W2_A | A — Baseline (hidden=128, layers=1) | 85.0±3.6 | 34.3±3.1 | 44.4±4.4 | 0.956±0.008 |
| W2_C | C — Wider (hidden=256, layers=1) | 82.2±6.0 | 32.5±3.4 | 44.7±3.9 | 0.956±0.008 |
| W2_E | E — L1 Loss + Small (hidden=64, layers=2) | 77.8±5.0 | 38.0±1.9 | 50.9±4.0 | 0.943±0.009 |
| W2_B | B — Deeper (hidden=128, layers=2) | 76.7±6.5 | 38.4±2.5 | 47.8±2.7 | 0.950±0.006 |
| W2_D | D — OneCycleLR (hidden=128, layers=2) | 72.8±7.2 | 40.4±5.0 | 52.8±4.3 | 0.938±0.010 |

### W3: early(10-40)+late(180-200) [original]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W3_A | A — Baseline (hidden=128, layers=1) | 70.6±5.0 | 41.4±4.5 | 52.4±5.9 | 0.939±0.014 |
| W3_B | B — Deeper (hidden=128, layers=2) | 69.4±7.1 | 43.3±2.6 | 55.4±3.9 | 0.933±0.009 |
| W3_C | C — Wider (hidden=256, layers=1) | 68.9±7.1 | 44.6±4.3 | 56.8±6.4 | 0.928±0.016 |
| W3_D | D — OneCycleLR (hidden=128, layers=2) | 67.8±11.1 | 44.8±5.5 | 57.1±6.5 | 0.928±0.016 |
| W3_E | E — L1 Loss + Small (hidden=64, layers=2) | 67.2±6.3 | 46.2±1.4 | 61.1±1.9 | 0.918±0.005 |

### W4: three bands 1-50,100-150,200-250

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W4_E | E — L1 Loss + Small (hidden=64, layers=2) | 75.6±5.1 | 40.4±1.6 | 57.8±2.9 | 0.927±0.007 |
| W4_C | C — Wider (hidden=256, layers=1) | 75.6±7.1 | 38.5±3.5 | 51.6±4.0 | 0.941±0.009 |
| W4_B | B — Deeper (hidden=128, layers=2) | 72.8±5.2 | 40.4±1.8 | 56.0±2.2 | 0.931±0.006 |
| W4_D | D — OneCycleLR (hidden=128, layers=2) | 71.7±5.2 | 43.0±3.7 | 61.1±5.5 | 0.918±0.015 |
| W4_A | A — Baseline (hidden=128, layers=1) | 71.7±7.2 | 39.7±3.9 | 53.2±3.6 | 0.938±0.009 |

### W5: sparse landmarks [1,10,50,100,150,200,250]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W5_E | E — L1 Loss + Small (hidden=64, layers=2) | 83.9±7.2 | 35.0±2.0 | 51.7±3.6 | 0.941±0.008 |
| W5_A | A — Baseline (hidden=128, layers=1) | 80.6±7.1 | 37.8±4.4 | 47.2±4.3 | 0.951±0.009 |
| W5_B | B — Deeper (hidden=128, layers=2) | 77.8±9.6 | 37.8±3.2 | 48.1±2.8 | 0.949±0.006 |
| W5_C | C — Wider (hidden=256, layers=1) | 77.8±5.0 | 37.8±4.5 | 50.7±4.6 | 0.943±0.010 |
| W5_D | D — OneCycleLR (hidden=128, layers=2) | 76.7±4.2 | 40.4±4.2 | 54.7±3.8 | 0.934±0.009 |
