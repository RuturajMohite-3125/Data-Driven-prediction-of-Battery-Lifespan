# Transformer Sweep Results

## Configurations

### Config A: A — Baseline (current)

| Parameter | Value |
|-----------|-------|
| `d_model` | 64 |
| `nhead` | 4 |
| `num_layers` | 2 |
| `dim_ff` | 128 |
| `dropout` | 0.3 |
| `epochs` | 300 |
| `batch_size` | 4 |
| `lr` | 0.0003 |
| `weight_decay` | 0.005 |
| `lambda_eol` | 1.5 |
| `loss` | smooth_l1 |
| `scheduler` | cosine |

### Config B: B — Deeper + Wider

| Parameter | Value |
|-----------|-------|
| `d_model` | 128 |
| `nhead` | 8 |
| `num_layers` | 3 |
| `dim_ff` | 256 |
| `dropout` | 0.2 |
| `epochs` | 300 |
| `batch_size` | 8 |
| `lr` | 0.0002 |
| `weight_decay` | 0.005 |
| `lambda_eol` | 1.5 |
| `loss` | smooth_l1 |
| `scheduler` | cosine |

### Config C: C — Regularized Small

| Parameter | Value |
|-----------|-------|
| `d_model` | 64 |
| `nhead` | 4 |
| `num_layers` | 2 |
| `dim_ff` | 128 |
| `dropout` | 0.4 |
| `epochs` | 300 |
| `batch_size` | 8 |
| `lr` | 0.0003 |
| `weight_decay` | 0.02 |
| `lambda_eol` | 1.5 |
| `loss` | smooth_l1 |
| `scheduler` | cosine |

### Config D: D — OneCycleLR

| Parameter | Value |
|-----------|-------|
| `d_model` | 64 |
| `nhead` | 4 |
| `num_layers` | 2 |
| `dim_ff` | 128 |
| `dropout` | 0.3 |
| `epochs` | 300 |
| `batch_size` | 4 |
| `lr` | 0.0005 |
| `weight_decay` | 0.005 |
| `lambda_eol` | 1.0 |
| `loss` | smooth_l1 |
| `scheduler` | onecycle |

### Config E: E — L1 Loss + Larger

| Parameter | Value |
|-----------|-------|
| `d_model` | 96 |
| `nhead` | 4 |
| `num_layers` | 3 |
| `dim_ff` | 192 |
| `dropout` | 0.25 |
| `epochs` | 300 |
| `batch_size` | 4 |
| `lr` | 0.0001 |
| `weight_decay` | 0.01 |
| `lambda_eol` | 2.0 |
| `loss` | l1 |
| `scheduler` | cosine |

## Results — Test Set (mean ± std across seeds)

| ID | Window | Config | Acc% | MAE | RMSE | R² |
|----|--------|--------|------|-----|------|-----|
| W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (current) | 81.7±7.5 | 35.0±5.8 | 45.3±8.2 | 0.954±0.016 |
| W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper + Wider | 78.9±8.5 | 35.7±4.5 | 49.6±7.6 | 0.945±0.017 |
| W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper + Wider | 77.2±9.8 | 37.6±5.4 | 52.5±7.6 | 0.938±0.017 |
| W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR | 77.2±8.4 | 37.9±6.6 | 50.7±9.6 | 0.942±0.020 |
| W2_C | W2: 1-250 stride-10 (25 pts) | C — Regularized Small | 76.7±7.4 | 38.3±5.4 | 51.7±4.9 | 0.941±0.011 |
| W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (current) | 76.1±5.6 | 39.5±4.4 | 56.0±5.7 | 0.931±0.014 |
| W4_C | W4: three bands 1-50,100-150,200-250 | C — Regularized Small | 76.1±6.1 | 42.9±3.6 | 58.3±6.0 | 0.925±0.015 |
| W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Larger | 76.1±5.0 | 39.6±3.8 | 55.6±5.1 | 0.932±0.013 |
| W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Larger | 75.6±5.7 | 39.7±3.7 | 55.3±8.8 | 0.931±0.023 |
| W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Regularized Small | 75.0±8.3 | 37.3±5.2 | 49.5±7.6 | 0.945±0.017 |
| W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR | 73.9±7.0 | 40.8±4.9 | 56.9±7.7 | 0.928±0.018 |
| W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR | 73.9±9.0 | 43.2±4.8 | 59.6±6.5 | 0.921±0.017 |
| W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper + Wider | 73.3±10.5 | 43.6±6.9 | 55.4±8.4 | 0.931±0.021 |
| W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Larger | 73.3±4.8 | 37.8±4.0 | 50.4±4.2 | 0.944±0.009 |
| W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR | 72.8±6.8 | 41.0±5.3 | 54.6±7.5 | 0.933±0.018 |
| W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (current) | 71.1±5.4 | 42.2±5.0 | 57.0±7.5 | 0.928±0.019 |
| W3_C | W3: early(10-40)+late(180-200) [original] | C — Regularized Small | 70.6±8.3 | 44.2±5.9 | 59.8±7.1 | 0.921±0.019 |
| W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Larger | 70.0±8.7 | 43.4±4.9 | 58.7±7.4 | 0.923±0.018 |
| W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper + Wider | 68.9±7.9 | 44.4±6.7 | 57.6±7.9 | 0.926±0.021 |
| W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (current) | 66.1±6.8 | 44.7±4.4 | 59.6±6.6 | 0.921±0.017 |
| W0_E | W0: first 50 cycles | E — L1 Loss + Larger | 62.2±8.2 | 59.8±12.6 | 81.8±16.1 | 0.848±0.062 |
| W1_E | W1: first 100 cycles | E — L1 Loss + Larger | 61.1±6.1 | 59.7±7.5 | 81.8±9.4 | 0.851±0.033 |
| W1_B | W1: first 100 cycles | B — Deeper + Wider | 58.9±6.2 | 64.0±6.1 | 84.6±12.2 | 0.840±0.047 |
| W0_C | W0: first 50 cycles | C — Regularized Small | 56.7±9.9 | 66.4±11.5 | 88.2±18.0 | 0.822±0.072 |
| W1_D | W1: first 100 cycles | D — OneCycleLR | 56.1±4.6 | 64.7±9.7 | 85.7±12.0 | 0.836±0.044 |
| W1_C | W1: first 100 cycles | C — Regularized Small | 55.6±6.1 | 64.8±6.8 | 88.9±10.4 | 0.824±0.040 |
| W0_A | W0: first 50 cycles | A — Baseline (current) | 55.0±5.2 | 67.7±11.2 | 89.0±15.6 | 0.821±0.063 |
| W0_D | W0: first 50 cycles | D — OneCycleLR | 53.9±6.6 | 68.1±7.5 | 87.3±9.0 | 0.831±0.035 |
| W1_A | W1: first 100 cycles | A — Baseline (current) | 52.8±5.1 | 64.9±6.9 | 85.1±9.8 | 0.839±0.038 |
| W0_B | W0: first 50 cycles | B — Deeper + Wider | 50.6±5.2 | 71.0±7.5 | 91.5±8.4 | 0.815±0.034 |

## Per-Window Breakdown

### W0: first 50 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W0_E | E — L1 Loss + Larger | 62.2±8.2 | 59.8±12.6 | 81.8±16.1 | 0.848±0.062 |
| W0_C | C — Regularized Small | 56.7±9.9 | 66.4±11.5 | 88.2±18.0 | 0.822±0.072 |
| W0_A | A — Baseline (current) | 55.0±5.2 | 67.7±11.2 | 89.0±15.6 | 0.821±0.063 |
| W0_D | D — OneCycleLR | 53.9±6.6 | 68.1±7.5 | 87.3±9.0 | 0.831±0.035 |
| W0_B | B — Deeper + Wider | 50.6±5.2 | 71.0±7.5 | 91.5±8.4 | 0.815±0.034 |

### W1: first 100 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W1_E | E — L1 Loss + Larger | 61.1±6.1 | 59.7±7.5 | 81.8±9.4 | 0.851±0.033 |
| W1_B | B — Deeper + Wider | 58.9±6.2 | 64.0±6.1 | 84.6±12.2 | 0.840±0.047 |
| W1_D | D — OneCycleLR | 56.1±4.6 | 64.7±9.7 | 85.7±12.0 | 0.836±0.044 |
| W1_C | C — Regularized Small | 55.6±6.1 | 64.8±6.8 | 88.9±10.4 | 0.824±0.040 |
| W1_A | A — Baseline (current) | 52.8±5.1 | 64.9±6.9 | 85.1±9.8 | 0.839±0.038 |

### W2: 1-250 stride-10 (25 pts)

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W2_B | B — Deeper + Wider | 77.2±9.8 | 37.6±5.4 | 52.5±7.6 | 0.938±0.017 |
| W2_C | C — Regularized Small | 76.7±7.4 | 38.3±5.4 | 51.7±4.9 | 0.941±0.011 |
| W2_A | A — Baseline (current) | 76.1±5.6 | 39.5±4.4 | 56.0±5.7 | 0.931±0.014 |
| W2_E | E — L1 Loss + Larger | 75.6±5.7 | 39.7±3.7 | 55.3±8.8 | 0.931±0.023 |
| W2_D | D — OneCycleLR | 73.9±7.0 | 40.8±4.9 | 56.9±7.7 | 0.928±0.018 |

### W3: early(10-40)+late(180-200) [original]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W3_D | D — OneCycleLR | 72.8±6.8 | 41.0±5.3 | 54.6±7.5 | 0.933±0.018 |
| W3_A | A — Baseline (current) | 71.1±5.4 | 42.2±5.0 | 57.0±7.5 | 0.928±0.019 |
| W3_C | C — Regularized Small | 70.6±8.3 | 44.2±5.9 | 59.8±7.1 | 0.921±0.019 |
| W3_E | E — L1 Loss + Larger | 70.0±8.7 | 43.4±4.9 | 58.7±7.4 | 0.923±0.018 |
| W3_B | B — Deeper + Wider | 68.9±7.9 | 44.4±6.7 | 57.6±7.9 | 0.926±0.021 |

### W4: three bands 1-50,100-150,200-250

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W4_C | C — Regularized Small | 76.1±6.1 | 42.9±3.6 | 58.3±6.0 | 0.925±0.015 |
| W4_E | E — L1 Loss + Larger | 76.1±5.0 | 39.6±3.8 | 55.6±5.1 | 0.932±0.013 |
| W4_D | D — OneCycleLR | 73.9±9.0 | 43.2±4.8 | 59.6±6.5 | 0.921±0.017 |
| W4_B | B — Deeper + Wider | 73.3±10.5 | 43.6±6.9 | 55.4±8.4 | 0.931±0.021 |
| W4_A | A — Baseline (current) | 66.1±6.8 | 44.7±4.4 | 59.6±6.6 | 0.921±0.017 |

### W5: sparse landmarks [1,10,50,100,150,200,250]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W5_A | A — Baseline (current) | 81.7±7.5 | 35.0±5.8 | 45.3±8.2 | 0.954±0.016 |
| W5_B | B — Deeper + Wider | 78.9±8.5 | 35.7±4.5 | 49.6±7.6 | 0.945±0.017 |
| W5_D | D — OneCycleLR | 77.2±8.4 | 37.9±6.6 | 50.7±9.6 | 0.942±0.020 |
| W5_C | C — Regularized Small | 75.0±8.3 | 37.3±5.2 | 49.5±7.6 | 0.945±0.017 |
| W5_E | E — L1 Loss + Larger | 73.3±4.8 | 37.8±4.0 | 50.4±4.2 | 0.944±0.009 |
