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
| W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (current) | 92.2±3.9 | 34.4±4.8 | 46.1±7.9 | 0.952±0.017 |
| W4_C | W4: three bands 1-50,100-150,200-250 | C — Regularized Small | 91.1±4.7 | 37.6±5.9 | 50.3±9.1 | 0.943±0.021 |
| W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR | 90.6±6.4 | 40.3±5.0 | 57.3±11.5 | 0.926±0.033 |
| W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR | 89.4±4.1 | 38.3±4.0 | 53.1±8.7 | 0.937±0.020 |
| W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Larger | 88.9±3.7 | 34.6±4.7 | 47.5±6.8 | 0.950±0.014 |
| W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper + Wider | 88.9±4.5 | 37.0±6.2 | 50.9±8.1 | 0.942±0.016 |
| W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (current) | 88.9±4.5 | 38.1±4.9 | 57.1±12.5 | 0.925±0.036 |
| W2_C | W2: 1-250 stride-10 (25 pts) | C — Regularized Small | 87.8±4.4 | 40.3±6.4 | 55.9±10.4 | 0.929±0.025 |
| W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR | 86.7±4.7 | 41.6±6.8 | 60.8±13.4 | 0.915±0.039 |
| W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Regularized Small | 86.7±4.7 | 41.5±4.8 | 60.3±10.0 | 0.918±0.028 |
| W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (current) | 85.6±5.4 | 42.2±5.9 | 60.6±8.7 | 0.918±0.026 |
| W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper + Wider | 85.6±7.5 | 42.1±5.3 | 58.8±8.9 | 0.923±0.022 |
| W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Larger | 85.6±6.5 | 41.6±5.6 | 60.6±12.9 | 0.916±0.035 |
| W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR | 84.4±5.1 | 44.6±7.8 | 62.1±12.0 | 0.913±0.033 |
| W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Larger | 84.4±6.8 | 43.3±6.3 | 60.5±14.4 | 0.916±0.041 |
| W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper + Wider | 84.4±5.7 | 44.7±6.7 | 60.7±8.4 | 0.918±0.022 |
| W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (current) | 83.9±4.1 | 46.4±6.3 | 63.7±8.0 | 0.910±0.023 |
| W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper + Wider | 82.8±4.1 | 44.6±5.8 | 63.1±9.7 | 0.911±0.027 |
| W3_C | W3: early(10-40)+late(180-200) [original] | C — Regularized Small | 82.2±4.4 | 50.5±5.7 | 69.7±7.4 | 0.892±0.024 |
| W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Larger | 80.0±9.5 | 50.3±7.0 | 71.4±10.8 | 0.886±0.033 |
| W0_E | W0: first 50 cycles | E — L1 Loss + Larger | 74.4±6.5 | 63.4±7.4 | 86.1±10.4 | 0.835±0.039 |
| W0_D | W0: first 50 cycles | D — OneCycleLR | 72.2±5.9 | 67.7±7.2 | 90.1±10.8 | 0.820±0.042 |
| W1_D | W1: first 100 cycles | D — OneCycleLR | 71.1±5.7 | 66.5±6.3 | 83.9±7.4 | 0.845±0.027 |
| W0_C | W0: first 50 cycles | C — Regularized Small | 70.6±6.4 | 68.1±5.3 | 91.6±6.7 | 0.815±0.027 |
| W1_C | W1: first 100 cycles | C — Regularized Small | 70.6±5.9 | 66.6±5.6 | 88.0±11.4 | 0.828±0.047 |
| W0_B | W0: first 50 cycles | B — Deeper + Wider | 68.9±6.0 | 71.6±6.3 | 97.2±11.9 | 0.790±0.052 |
| W0_A | W0: first 50 cycles | A — Baseline (current) | 68.9±7.5 | 68.5±6.9 | 90.5±9.6 | 0.819±0.040 |
| W1_E | W1: first 100 cycles | E — L1 Loss + Larger | 68.9±7.5 | 63.6±10.3 | 83.6±13.1 | 0.843±0.050 |
| W1_B | W1: first 100 cycles | B — Deeper + Wider | 68.3±6.4 | 70.5±10.6 | 90.9±13.8 | 0.815±0.056 |
| W1_A | W1: first 100 cycles | A — Baseline (current) | 67.2±6.7 | 67.8±6.1 | 88.3±9.2 | 0.827±0.033 |

## Per-Window Breakdown

### W0: first 50 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W0_E | E — L1 Loss + Larger | 74.4±6.5 | 63.4±7.4 | 86.1±10.4 | 0.835±0.039 |
| W0_D | D — OneCycleLR | 72.2±5.9 | 67.7±7.2 | 90.1±10.8 | 0.820±0.042 |
| W0_C | C — Regularized Small | 70.6±6.4 | 68.1±5.3 | 91.6±6.7 | 0.815±0.027 |
| W0_B | B — Deeper + Wider | 68.9±6.0 | 71.6±6.3 | 97.2±11.9 | 0.790±0.052 |
| W0_A | A — Baseline (current) | 68.9±7.5 | 68.5±6.9 | 90.5±9.6 | 0.819±0.040 |

### W1: first 100 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W1_D | D — OneCycleLR | 71.1±5.7 | 66.5±6.3 | 83.9±7.4 | 0.845±0.027 |
| W1_C | C — Regularized Small | 70.6±5.9 | 66.6±5.6 | 88.0±11.4 | 0.828±0.047 |
| W1_E | E — L1 Loss + Larger | 68.9±7.5 | 63.6±10.3 | 83.6±13.1 | 0.843±0.050 |
| W1_B | B — Deeper + Wider | 68.3±6.4 | 70.5±10.6 | 90.9±13.8 | 0.815±0.056 |
| W1_A | A — Baseline (current) | 67.2±6.7 | 67.8±6.1 | 88.3±9.2 | 0.827±0.033 |

### W2: 1-250 stride-10 (25 pts)

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W2_C | C — Regularized Small | 87.8±4.4 | 40.3±6.4 | 55.9±10.4 | 0.929±0.025 |
| W2_D | D — OneCycleLR | 86.7±4.7 | 41.6±6.8 | 60.8±13.4 | 0.915±0.039 |
| W2_A | A — Baseline (current) | 85.6±5.4 | 42.2±5.9 | 60.6±8.7 | 0.918±0.026 |
| W2_E | E — L1 Loss + Larger | 84.4±6.8 | 43.3±6.3 | 60.5±14.4 | 0.916±0.041 |
| W2_B | B — Deeper + Wider | 82.8±4.1 | 44.6±5.8 | 63.1±9.7 | 0.911±0.027 |

### W3: early(10-40)+late(180-200) [original]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W3_D | D — OneCycleLR | 84.4±5.1 | 44.6±7.8 | 62.1±12.0 | 0.913±0.033 |
| W3_B | B — Deeper + Wider | 84.4±5.7 | 44.7±6.7 | 60.7±8.4 | 0.918±0.022 |
| W3_A | A — Baseline (current) | 83.9±4.1 | 46.4±6.3 | 63.7±8.0 | 0.910±0.023 |
| W3_C | C — Regularized Small | 82.2±4.4 | 50.5±5.7 | 69.7±7.4 | 0.892±0.024 |
| W3_E | E — L1 Loss + Larger | 80.0±9.5 | 50.3±7.0 | 71.4±10.8 | 0.886±0.033 |

### W4: three bands 1-50,100-150,200-250

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W4_A | A — Baseline (current) | 92.2±3.9 | 34.4±4.8 | 46.1±7.9 | 0.952±0.017 |
| W4_C | C — Regularized Small | 91.1±4.7 | 37.6±5.9 | 50.3±9.1 | 0.943±0.021 |
| W4_D | D — OneCycleLR | 89.4±4.1 | 38.3±4.0 | 53.1±8.7 | 0.937±0.020 |
| W4_E | E — L1 Loss + Larger | 88.9±3.7 | 34.6±4.7 | 47.5±6.8 | 0.950±0.014 |
| W4_B | B — Deeper + Wider | 88.9±4.5 | 37.0±6.2 | 50.9±8.1 | 0.942±0.016 |

### W5: sparse landmarks [1,10,50,100,150,200,250]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W5_D | D — OneCycleLR | 90.6±6.4 | 40.3±5.0 | 57.3±11.5 | 0.926±0.033 |
| W5_A | A — Baseline (current) | 88.9±4.5 | 38.1±4.9 | 57.1±12.5 | 0.925±0.036 |
| W5_C | C — Regularized Small | 86.7±4.7 | 41.5±4.8 | 60.3±10.0 | 0.918±0.028 |
| W5_B | B — Deeper + Wider | 85.6±7.5 | 42.1±5.3 | 58.8±8.9 | 0.923±0.022 |
| W5_E | E — L1 Loss + Larger | 85.6±6.5 | 41.6±5.6 | 60.6±12.9 | 0.916±0.035 |
