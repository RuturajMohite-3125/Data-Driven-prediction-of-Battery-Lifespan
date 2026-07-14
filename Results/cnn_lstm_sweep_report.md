# CNN-LSTM Sweep Results

## Configurations

### Config A: A — Baseline (cnn=64, lstm=128, layers=1)

| Parameter | Value |
|-----------|-------|
| `cnn_channels` | 64 |
| `lstm_hidden` | 128 |
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

### Config B: B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2)

| Parameter | Value |
|-----------|-------|
| `cnn_channels` | 128 |
| `lstm_hidden` | 128 |
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

### Config C: C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1)

| Parameter | Value |
|-----------|-------|
| `cnn_channels` | 32 |
| `lstm_hidden` | 256 |
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

### Config D: D — OneCycleLR (cnn=64, lstm=128, layers=2)

| Parameter | Value |
|-----------|-------|
| `cnn_channels` | 64 |
| `lstm_hidden` | 128 |
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

### Config E: E — L1 Loss + Small (cnn=32, lstm=64, layers=2)

| Parameter | Value |
|-----------|-------|
| `cnn_channels` | 32 |
| `lstm_hidden` | 64 |
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
| W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (cnn=64, lstm=128, layers=1) | 88.9±8.3 | 38.9±10.2 | 56.7±17.2 | 0.924±0.049 |
| W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 88.9±8.3 | 38.7±11.2 | 52.9±15.6 | 0.934±0.042 |
| W3_B | W3: early(10-40)+late(180-200) [original] | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 88.3±6.1 | 38.9±7.1 | 53.6±10.2 | 0.935±0.025 |
| W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 87.2±8.3 | 41.3±9.6 | 55.5±13.8 | 0.929±0.037 |
| W2_C | W2: 1-250 stride-10 (25 pts) | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 85.6±4.7 | 45.9±7.8 | 62.0±8.7 | 0.914±0.023 |
| W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (cnn=64, lstm=128, layers=1) | 85.0±4.6 | 44.7±7.3 | 62.7±8.6 | 0.912±0.024 |
| W2_B | W2: 1-250 stride-10 (25 pts) | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 85.0±7.4 | 46.1±4.7 | 62.7±8.6 | 0.912±0.024 |
| W3_C | W3: early(10-40)+late(180-200) [original] | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 85.0±6.4 | 42.9±9.7 | 59.4±14.3 | 0.919±0.041 |
| W4_C | W4: three bands 1-50,100-150,200-250 | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 85.0±4.6 | 42.8±6.6 | 62.2±6.0 | 0.915±0.017 |
| W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 84.4±8.2 | 45.8±7.8 | 60.2±8.6 | 0.919±0.022 |
| W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (cnn=64, lstm=128, layers=1) | 83.3±7.4 | 47.8±9.8 | 62.2±9.9 | 0.913±0.027 |
| W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 83.3±7.4 | 45.9±9.9 | 63.5±8.3 | 0.910±0.024 |
| W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 82.8±6.1 | 46.9±8.0 | 62.9±5.5 | 0.913±0.016 |
| W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 82.2±2.3 | 47.3±4.8 | 62.1±6.9 | 0.914±0.019 |
| W4_B | W4: three bands 1-50,100-150,200-250 | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 80.6±2.9 | 46.9±8.0 | 64.7±5.5 | 0.908±0.015 |
| W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 78.3±8.0 | 51.0±6.7 | 63.9±7.5 | 0.910±0.021 |
| W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 78.3±8.5 | 50.6±7.2 | 68.1±17.1 | 0.892±0.064 |
| W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 77.2±5.5 | 53.5±6.8 | 66.7±8.8 | 0.901±0.028 |
| W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 76.7±8.6 | 51.0±6.7 | 63.9±6.4 | 0.910±0.018 |
| W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (cnn=64, lstm=128, layers=1) | 76.1±8.3 | 52.0±4.7 | 65.7±5.5 | 0.905±0.017 |
| W0_E | W0: first 50 cycles | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 74.4±6.5 | 72.8±12.9 | 100.7±19.5 | 0.770±0.089 |
| W1_C | W1: first 100 cycles | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 71.7±7.1 | 75.5±12.1 | 104.9±24.4 | 0.747±0.115 |
| W0_B | W0: first 50 cycles | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 71.1±5.7 | 69.8±8.2 | 97.5±12.8 | 0.788±0.055 |
| W1_B | W1: first 100 cycles | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 71.1±5.7 | 78.9±15.8 | 111.1±25.5 | 0.716±0.131 |
| W0_C | W0: first 50 cycles | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 71.1±6.3 | 68.4±10.1 | 95.8±13.3 | 0.795±0.055 |
| W1_A | W1: first 100 cycles | A — Baseline (cnn=64, lstm=128, layers=1) | 71.1±8.2 | 79.5±12.3 | 116.5±19.7 | 0.695±0.104 |
| W0_D | W0: first 50 cycles | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 70.6±7.0 | 71.8±11.2 | 99.2±20.3 | 0.776±0.090 |
| W1_E | W1: first 100 cycles | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 70.6±7.4 | 78.3±10.0 | 107.6±22.7 | 0.736±0.114 |
| W1_D | W1: first 100 cycles | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 70.6±5.9 | 77.8±13.3 | 106.3±23.3 | 0.742±0.114 |
| W0_A | W0: first 50 cycles | A — Baseline (cnn=64, lstm=128, layers=1) | 68.9±7.9 | 72.6±10.5 | 100.5±14.2 | 0.775±0.063 |

## Per-Window Breakdown

### W0: first 50 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W0_E | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 74.4±6.5 | 72.8±12.9 | 100.7±19.5 | 0.770±0.089 |
| W0_B | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 71.1±5.7 | 69.8±8.2 | 97.5±12.8 | 0.788±0.055 |
| W0_C | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 71.1±6.3 | 68.4±10.1 | 95.8±13.3 | 0.795±0.055 |
| W0_D | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 70.6±7.0 | 71.8±11.2 | 99.2±20.3 | 0.776±0.090 |
| W0_A | A — Baseline (cnn=64, lstm=128, layers=1) | 68.9±7.9 | 72.6±10.5 | 100.5±14.2 | 0.775±0.063 |

### W1: first 100 cycles

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W1_C | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 71.7±7.1 | 75.5±12.1 | 104.9±24.4 | 0.747±0.115 |
| W1_B | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 71.1±5.7 | 78.9±15.8 | 111.1±25.5 | 0.716±0.131 |
| W1_A | A — Baseline (cnn=64, lstm=128, layers=1) | 71.1±8.2 | 79.5±12.3 | 116.5±19.7 | 0.695±0.104 |
| W1_E | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 70.6±7.4 | 78.3±10.0 | 107.6±22.7 | 0.736±0.114 |
| W1_D | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 70.6±5.9 | 77.8±13.3 | 106.3±23.3 | 0.742±0.114 |

### W2: 1-250 stride-10 (25 pts)

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W2_C | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 85.6±4.7 | 45.9±7.8 | 62.0±8.7 | 0.914±0.023 |
| W2_B | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 85.0±7.4 | 46.1±4.7 | 62.7±8.6 | 0.912±0.024 |
| W2_E | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 84.4±8.2 | 45.8±7.8 | 60.2±8.6 | 0.919±0.022 |
| W2_A | A — Baseline (cnn=64, lstm=128, layers=1) | 83.3±7.4 | 47.8±9.8 | 62.2±9.9 | 0.913±0.027 |
| W2_D | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 82.2±2.3 | 47.3±4.8 | 62.1±6.9 | 0.914±0.019 |

### W3: early(10-40)+late(180-200) [original]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W3_A | A — Baseline (cnn=64, lstm=128, layers=1) | 88.9±8.3 | 38.9±10.2 | 56.7±17.2 | 0.924±0.049 |
| W3_E | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 88.9±8.3 | 38.7±11.2 | 52.9±15.6 | 0.934±0.042 |
| W3_B | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 88.3±6.1 | 38.9±7.1 | 53.6±10.2 | 0.935±0.025 |
| W3_D | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 87.2±8.3 | 41.3±9.6 | 55.5±13.8 | 0.929±0.037 |
| W3_C | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 85.0±6.4 | 42.9±9.7 | 59.4±14.3 | 0.919±0.041 |

### W4: three bands 1-50,100-150,200-250

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W4_A | A — Baseline (cnn=64, lstm=128, layers=1) | 85.0±4.6 | 44.7±7.3 | 62.7±8.6 | 0.912±0.024 |
| W4_C | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 85.0±4.6 | 42.8±6.6 | 62.2±6.0 | 0.915±0.017 |
| W4_E | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 83.3±7.4 | 45.9±9.9 | 63.5±8.3 | 0.910±0.024 |
| W4_D | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 82.8±6.1 | 46.9±8.0 | 62.9±5.5 | 0.913±0.016 |
| W4_B | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 80.6±2.9 | 46.9±8.0 | 64.7±5.5 | 0.908±0.015 |

### W5: sparse landmarks [1,10,50,100,150,200,250]

| ID | Config | Acc% | MAE | RMSE | R² |
|----|--------|------|-----|------|-----|
| W5_C | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 78.3±8.0 | 51.0±6.7 | 63.9±7.5 | 0.910±0.021 |
| W5_B | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 78.3±8.5 | 50.6±7.2 | 68.1±17.1 | 0.892±0.064 |
| W5_E | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 77.2±5.5 | 53.5±6.8 | 66.7±8.8 | 0.901±0.028 |
| W5_D | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 76.7±8.6 | 51.0±6.7 | 63.9±6.4 | 0.910±0.018 |
| W5_A | A — Baseline (cnn=64, lstm=128, layers=1) | 76.1±8.3 | 52.0±4.7 | 65.7±5.5 | 0.905±0.017 |
