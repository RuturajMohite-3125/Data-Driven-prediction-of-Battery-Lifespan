# CNN-GRU Sweep Results

## Configurations

### Config A: A — Baseline (cnn=64, gru=128, layers=1)

| Parameter | Value |
|-----------|-------|
| `cnn_channels` | 64 |
| `gru_hidden` | 128 |
| `num_layers` | 1 |
| `cnn_dropout` | 0.2 |
| `head_dropout` | 0.3 |
| `epochs` | 300 |
| `batch_size` | 4 |
| `lr` | 0.0003 |
| `weight_decay` | 0.005 |
| `lambda_eol` | 1.5 |
| `loss` | smooth_l1 |
| `scheduler` | cosine |

### Config B: B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2)

| Parameter | Value |
|-----------|-------|
| `cnn_channels` | 128 |
| `gru_hidden` | 128 |
| `num_layers` | 2 |
| `cnn_dropout` | 0.2 |
| `head_dropout` | 0.3 |
| `epochs` | 300 |
| `batch_size` | 8 |
| `lr` | 0.0002 |
| `weight_decay` | 0.005 |
| `lambda_eol` | 1.5 |
| `loss` | smooth_l1 |
| `scheduler` | cosine |

### Config C: C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1)

| Parameter | Value |
|-----------|-------|
| `cnn_channels` | 32 |
| `gru_hidden` | 256 |
| `num_layers` | 1 |
| `cnn_dropout` | 0.1 |
| `head_dropout` | 0.2 |
| `epochs` | 300 |
| `batch_size` | 4 |
| `lr` | 0.0003 |
| `weight_decay` | 0.01 |
| `lambda_eol` | 1.5 |
| `loss` | smooth_l1 |
| `scheduler` | cosine |

### Config D: D — OneCycleLR (cnn=64, gru=128, layers=2)

| Parameter | Value |
|-----------|-------|
| `cnn_channels` | 64 |
| `gru_hidden` | 128 |
| `num_layers` | 2 |
| `cnn_dropout` | 0.2 |
| `head_dropout` | 0.3 |
| `epochs` | 300 |
| `batch_size` | 4 |
| `lr` | 0.0005 |
| `weight_decay` | 0.005 |
| `lambda_eol` | 1.0 |
| `loss` | smooth_l1 |
| `scheduler` | onecycle |

### Config E: E — L1 Loss + Small (cnn=32, gru=64, layers=2)

| Parameter | Value |
|-----------|-------|
| `cnn_channels` | 32 |
| `gru_hidden` | 64 |
| `num_layers` | 2 |
| `cnn_dropout` | 0.3 |
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
| W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 86.7±6.5 | 46.7±6.5 | 63.3±10.5 | 0.910±0.029 |
| W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (cnn=64, gru=128, layers=1) | 84.4±8.2 | 43.2±10.4 | 59.7±12.9 | 0.919±0.035 |
| W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 84.4±11.4 | 52.7±7.4 | 63.2±6.8 | 0.911±0.018 |
| W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 83.3±6.4 | 47.1±3.4 | 59.3±5.1 | 0.922±0.013 |
| W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (cnn=64, gru=128, layers=2) | 82.8±6.7 | 52.4±6.2 | 65.9±8.9 | 0.903±0.026 |
| W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (cnn=64, gru=128, layers=2) | 82.2±8.2 | 51.6±8.5 | 66.4±8.6 | 0.902±0.025 |
| W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (cnn=64, gru=128, layers=1) | 81.7±5.9 | 48.1±10.1 | 68.9±11.1 | 0.893±0.034 |
| W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (cnn=64, gru=128, layers=1) | 81.7±5.3 | 45.5±6.6 | 57.1±7.8 | 0.927±0.019 |
| W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 81.7±5.3 | 50.2±5.5 | 64.1±8.4 | 0.909±0.025 |
| W3_B | W3: early(10-40)+late(180-200) [original] | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 81.7±3.7 | 43.4±4.6 | 56.3±4.7 | 0.930±0.012 |
| W3_C | W3: early(10-40)+late(180-200) [original] | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 81.7±5.9 | 46.5±8.1 | 58.9±12.3 | 0.921±0.033 |
| W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (cnn=64, gru=128, layers=1) | 81.1±8.8 | 54.1±11.7 | 68.0±14.9 | 0.894±0.051 |
| W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (cnn=64, gru=128, layers=2) | 80.6±7.5 | 49.3±8.1 | 64.0±8.5 | 0.909±0.024 |
| W4_C | W4: three bands 1-50,100-150,200-250 | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 80.6±6.5 | 45.2±9.1 | 60.6±10.4 | 0.917±0.028 |
| W2_C | W2: 1-250 stride-10 (25 pts) | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 80.0±6.0 | 48.6±8.9 | 61.8±8.5 | 0.915±0.024 |
| W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (cnn=64, gru=128, layers=2) | 79.4±7.0 | 53.3±9.8 | 74.5±8.0 | 0.877±0.026 |
| W2_B | W2: 1-250 stride-10 (25 pts) | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 78.9±3.5 | 48.0±6.3 | 63.8±6.5 | 0.910±0.018 |
| W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 78.9±6.8 | 51.1±6.8 | 66.3±8.0 | 0.902±0.023 |
| W4_B | W4: three bands 1-50,100-150,200-250 | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 78.3±5.5 | 46.6±6.9 | 64.2±7.4 | 0.909±0.021 |
| W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 76.7±8.6 | 53.7±7.2 | 67.9±10.6 | 0.897±0.032 |
| W0_D | W0: first 50 cycles | D — OneCycleLR (cnn=64, gru=128, layers=2) | 73.3±9.7 | 71.2±17.1 | 94.0±25.6 | 0.793±0.123 |
| W1_B | W1: first 100 cycles | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 72.8±5.5 | 66.3±6.9 | 89.9±10.4 | 0.821±0.039 |
| W1_E | W1: first 100 cycles | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 72.8±5.5 | 65.5±9.1 | 86.6±12.6 | 0.832±0.050 |
| W1_C | W1: first 100 cycles | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 72.2±5.9 | 63.0±6.2 | 84.4±8.6 | 0.842±0.031 |
| W0_A | W0: first 50 cycles | A — Baseline (cnn=64, gru=128, layers=1) | 72.2±6.9 | 65.2±11.3 | 93.4±16.1 | 0.804±0.072 |
| W0_B | W0: first 50 cycles | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 71.7±8.0 | 63.4±13.6 | 87.8±19.8 | 0.823±0.080 |
| W0_E | W0: first 50 cycles | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 71.7±9.2 | 69.8±10.5 | 91.6±15.6 | 0.811±0.065 |
| W1_A | W1: first 100 cycles | A — Baseline (cnn=64, gru=128, layers=1) | 71.7±6.1 | 70.4±7.1 | 101.5±15.2 | 0.770±0.066 |
| W0_C | W0: first 50 cycles | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 71.1±10.1 | 65.0±11.7 | 87.2±15.0 | 0.829±0.060 |
| W1_D | W1: first 100 cycles | D — OneCycleLR (cnn=64, gru=128, layers=2) | 70.0±5.4 | 76.3±13.3 | 103.8±25.0 | 0.751±0.119 |

## Per-Window Breakdown

### W0: first 50 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W0_D | D — OneCycleLR (cnn=64, gru=128, layers=2) | 73.3±9.7 | 71.2±17.1 | 94.0±25.6 | 0.793±0.123 |
| W0_A | A — Baseline (cnn=64, gru=128, layers=1) | 72.2±6.9 | 65.2±11.3 | 93.4±16.1 | 0.804±0.072 |
| W0_B | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 71.7±8.0 | 63.4±13.6 | 87.8±19.8 | 0.823±0.080 |
| W0_E | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 71.7±9.2 | 69.8±10.5 | 91.6±15.6 | 0.811±0.065 |
| W0_C | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 71.1±10.1 | 65.0±11.7 | 87.2±15.0 | 0.829±0.060 |

### W1: first 100 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W1_B | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 72.8±5.5 | 66.3±6.9 | 89.9±10.4 | 0.821±0.039 |
| W1_E | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 72.8±5.5 | 65.5±9.1 | 86.6±12.6 | 0.832±0.050 |
| W1_C | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 72.2±5.9 | 63.0±6.2 | 84.4±8.6 | 0.842±0.031 |
| W1_A | A — Baseline (cnn=64, gru=128, layers=1) | 71.7±6.1 | 70.4±7.1 | 101.5±15.2 | 0.770±0.066 |
| W1_D | D — OneCycleLR (cnn=64, gru=128, layers=2) | 70.0±5.4 | 76.3±13.3 | 103.8±25.0 | 0.751±0.119 |

### W2: 1-250 stride-10 (25 pts)

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W2_A | A — Baseline (cnn=64, gru=128, layers=1) | 81.7±5.3 | 45.5±6.6 | 57.1±7.8 | 0.927±0.019 |
| W2_E | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 81.7±5.3 | 50.2±5.5 | 64.1±8.4 | 0.909±0.025 |
| W2_D | D — OneCycleLR (cnn=64, gru=128, layers=2) | 80.6±7.5 | 49.3±8.1 | 64.0±8.5 | 0.909±0.024 |
| W2_C | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 80.0±6.0 | 48.6±8.9 | 61.8±8.5 | 0.915±0.024 |
| W2_B | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 78.9±3.5 | 48.0±6.3 | 63.8±6.5 | 0.910±0.018 |

### W3: early(10-40)+late(180-200) [original]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W3_A | A — Baseline (cnn=64, gru=128, layers=1) | 84.4±8.2 | 43.2±10.4 | 59.7±12.9 | 0.919±0.035 |
| W3_D | D — OneCycleLR (cnn=64, gru=128, layers=2) | 82.2±8.2 | 51.6±8.5 | 66.4±8.6 | 0.902±0.025 |
| W3_B | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 81.7±3.7 | 43.4±4.6 | 56.3±4.7 | 0.930±0.012 |
| W3_C | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 81.7±5.9 | 46.5±8.1 | 58.9±12.3 | 0.921±0.033 |
| W3_E | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 76.7±8.6 | 53.7±7.2 | 67.9±10.6 | 0.897±0.032 |

### W4: three bands 1-50,100-150,200-250

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W4_A | A — Baseline (cnn=64, gru=128, layers=1) | 81.7±5.9 | 48.1±10.1 | 68.9±11.1 | 0.893±0.034 |
| W4_C | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 80.6±6.5 | 45.2±9.1 | 60.6±10.4 | 0.917±0.028 |
| W4_D | D — OneCycleLR (cnn=64, gru=128, layers=2) | 79.4±7.0 | 53.3±9.8 | 74.5±8.0 | 0.877±0.026 |
| W4_E | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 78.9±6.8 | 51.1±6.8 | 66.3±8.0 | 0.902±0.023 |
| W4_B | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 78.3±5.5 | 46.6±6.9 | 64.2±7.4 | 0.909±0.021 |

### W5: sparse landmarks [1,10,50,100,150,200,250]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W5_E | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 86.7±6.5 | 46.7±6.5 | 63.3±10.5 | 0.910±0.029 |
| W5_C | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 84.4±11.4 | 52.7±7.4 | 63.2±6.8 | 0.911±0.018 |
| W5_B | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 83.3±6.4 | 47.1±3.4 | 59.3±5.1 | 0.922±0.013 |
| W5_D | D — OneCycleLR (cnn=64, gru=128, layers=2) | 82.8±6.7 | 52.4±6.2 | 65.9±8.9 | 0.903±0.026 |
| W5_A | A — Baseline (cnn=64, gru=128, layers=1) | 81.1±8.8 | 54.1±11.7 | 68.0±14.9 | 0.894±0.051 |
