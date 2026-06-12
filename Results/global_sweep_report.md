# Global Sweep Comparison — All Models

Comparison across Transformer, LSTM, and GRU on test set (mean ± std across seeds).

| Model | ID | Window | Config | Acc% | MAE | RMSE | R² |
|-------|----|--------|--------|------|-----|------|-----|
| GRU | W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (hidden=128, layers=1) | 85.0±3.6 | 34.3±3.1 | 44.4±4.4 | 0.956±0.008 |
| GRU | W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (hidden=64, layers=2) | 83.9±7.2 | 35.0±2.0 | 51.7±3.6 | 0.941±0.008 |
| LSTM | W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (hidden=64, layers=2) | 83.9±6.8 | 36.0±2.8 | 48.1±4.2 | 0.949±0.009 |
| GRU | W2_C | W2: 1-250 stride-10 (25 pts) | C — Wider (hidden=256, layers=1) | 82.2±6.0 | 32.5±3.4 | 44.7±3.9 | 0.956±0.008 |
| LSTM | W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (hidden=64, layers=2) | 82.2±5.4 | 35.4±2.9 | 46.0±3.6 | 0.953±0.008 |
| Transformer | W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (current) | 81.7±7.5 | 35.0±5.8 | 45.3±8.2 | 0.954±0.016 |
| GRU | W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (hidden=128, layers=1) | 80.6±7.1 | 37.8±4.4 | 47.2±4.3 | 0.951±0.009 |
| LSTM | W2_C | W2: 1-250 stride-10 (25 pts) | C — Wider (hidden=256, layers=1) | 79.4±6.1 | 34.5±3.8 | 47.1±3.6 | 0.951±0.008 |
| Transformer | W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper + Wider | 78.9±8.5 | 35.7±4.5 | 49.6±7.6 | 0.945±0.017 |
| LSTM | W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (hidden=128, layers=1) | 78.3±7.2 | 34.7±4.0 | 46.6±6.6 | 0.951±0.013 |
| GRU | W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper (hidden=128, layers=2) | 77.8±9.6 | 37.8±3.2 | 48.1±2.8 | 0.949±0.006 |
| LSTM | W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (hidden=128, layers=1) | 77.8±5.0 | 35.2±2.9 | 46.3±3.8 | 0.953±0.008 |
| GRU | W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Wider (hidden=256, layers=1) | 77.8±5.0 | 37.8±4.5 | 50.7±4.6 | 0.943±0.010 |
| GRU | W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (hidden=64, layers=2) | 77.8±5.0 | 38.0±1.9 | 50.9±4.0 | 0.943±0.009 |
| LSTM | W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (hidden=128, layers=1) | 77.8±6.1 | 35.3±3.8 | 47.7±5.1 | 0.950±0.011 |
| Transformer | W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper + Wider | 77.2±9.8 | 37.6±5.4 | 52.5±7.6 | 0.938±0.017 |
| Transformer | W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR | 77.2±8.4 | 37.9±6.6 | 50.7±9.6 | 0.942±0.020 |
| GRU | W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (hidden=128, layers=2) | 76.7±4.2 | 40.4±4.2 | 54.7±3.8 | 0.934±0.009 |
| Transformer | W2_C | W2: 1-250 stride-10 (25 pts) | C — Regularized Small | 76.7±7.4 | 38.3±5.4 | 51.7±4.9 | 0.941±0.011 |
| GRU | W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper (hidden=128, layers=2) | 76.7±6.5 | 38.4±2.5 | 47.8±2.7 | 0.950±0.006 |
| Transformer | W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (current) | 76.1±5.6 | 39.5±4.4 | 56.0±5.7 | 0.931±0.014 |
| Transformer | W4_C | W4: three bands 1-50,100-150,200-250 | C — Regularized Small | 76.1±6.1 | 42.9±3.6 | 58.3±6.0 | 0.925±0.015 |
| Transformer | W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Larger | 76.1±5.0 | 39.6±3.8 | 55.6±5.1 | 0.932±0.013 |
| GRU | W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (hidden=64, layers=2) | 75.6±5.1 | 40.4±1.6 | 57.8±2.9 | 0.927±0.007 |
| LSTM | W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper (hidden=128, layers=2) | 75.6±6.7 | 35.9±2.2 | 46.2±2.3 | 0.953±0.005 |
| Transformer | W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Larger | 75.6±5.7 | 39.7±3.7 | 55.3±8.8 | 0.931±0.023 |
| GRU | W4_C | W4: three bands 1-50,100-150,200-250 | C — Wider (hidden=256, layers=1) | 75.6±7.1 | 38.5±3.5 | 51.6±4.0 | 0.941±0.009 |
| Transformer | W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Regularized Small | 75.0±8.3 | 37.3±5.2 | 49.5±7.6 | 0.945±0.017 |
| Transformer | W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR | 73.9±7.0 | 40.8±4.9 | 56.9±7.7 | 0.928±0.018 |
| Transformer | W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR | 73.9±9.0 | 43.2±4.8 | 59.6±6.5 | 0.921±0.017 |
| LSTM | W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper (hidden=128, layers=2) | 73.9±7.5 | 37.9±1.9 | 47.7±2.8 | 0.950±0.006 |
| LSTM | W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (hidden=128, layers=2) | 73.9±7.5 | 38.0±5.8 | 49.4±6.4 | 0.946±0.014 |
| LSTM | W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (hidden=64, layers=2) | 73.9±5.6 | 35.7±3.7 | 48.9±4.0 | 0.947±0.008 |
| Transformer | W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper + Wider | 73.3±10.5 | 43.6±6.9 | 55.4±8.4 | 0.931±0.021 |
| Transformer | W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Larger | 73.3±4.8 | 37.8±4.0 | 50.4±4.2 | 0.944±0.009 |
| LSTM | W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper (hidden=128, layers=2) | 73.3±3.3 | 34.0±1.4 | 47.4±2.9 | 0.950±0.006 |
| GRU | W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper (hidden=128, layers=2) | 72.8±5.2 | 40.4±1.8 | 56.0±2.2 | 0.931±0.006 |
| GRU | W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (hidden=128, layers=2) | 72.8±7.2 | 40.4±5.0 | 52.8±4.3 | 0.938±0.010 |
| Transformer | W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR | 72.8±6.8 | 41.0±5.3 | 54.6±7.5 | 0.933±0.018 |
| LSTM | W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (hidden=128, layers=2) | 72.2±7.9 | 39.0±5.6 | 53.0±5.8 | 0.938±0.014 |
| GRU | W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (hidden=128, layers=2) | 71.7±5.2 | 43.0±3.7 | 61.1±5.5 | 0.918±0.015 |
| GRU | W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (hidden=128, layers=1) | 71.7±7.2 | 39.7±3.9 | 53.2±3.6 | 0.938±0.009 |
| LSTM | W3_C | W3: early(10-40)+late(180-200) [original] | C — Wider (hidden=256, layers=1) | 71.1±6.5 | 40.4±3.8 | 52.2±5.2 | 0.940±0.012 |
| LSTM | W4_C | W4: three bands 1-50,100-150,200-250 | C — Wider (hidden=256, layers=1) | 71.1±6.5 | 39.8±5.6 | 52.3±5.7 | 0.939±0.014 |
| Transformer | W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (current) | 71.1±5.4 | 42.2±5.0 | 57.0±7.5 | 0.928±0.019 |
| GRU | W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (hidden=128, layers=1) | 70.6±5.0 | 41.4±4.5 | 52.4±5.9 | 0.939±0.014 |
| Transformer | W3_C | W3: early(10-40)+late(180-200) [original] | C — Regularized Small | 70.6±8.3 | 44.2±5.9 | 59.8±7.1 | 0.921±0.019 |
| LSTM | W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper (hidden=128, layers=2) | 70.0±5.7 | 40.3±2.6 | 51.3±2.5 | 0.942±0.006 |
| Transformer | W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Larger | 70.0±8.7 | 43.4±4.9 | 58.7±7.4 | 0.923±0.018 |
| LSTM | W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (hidden=128, layers=2) | 69.4±6.2 | 40.1±4.6 | 51.8±3.9 | 0.941±0.009 |
| GRU | W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper (hidden=128, layers=2) | 69.4±7.1 | 43.3±2.6 | 55.4±3.9 | 0.933±0.009 |
| Transformer | W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper + Wider | 68.9±7.9 | 44.4±6.7 | 57.6±7.9 | 0.926±0.021 |
| GRU | W3_C | W3: early(10-40)+late(180-200) [original] | C — Wider (hidden=256, layers=1) | 68.9±7.1 | 44.6±4.3 | 56.8±6.4 | 0.928±0.016 |
| LSTM | W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (hidden=128, layers=2) | 68.9±8.7 | 42.4±8.2 | 53.5±10.0 | 0.935±0.024 |
| LSTM | W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Wider (hidden=256, layers=1) | 68.3±3.6 | 43.5±3.3 | 58.3±4.2 | 0.925±0.011 |
| LSTM | W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (hidden=128, layers=1) | 67.8±7.8 | 43.2±4.7 | 55.1±5.9 | 0.933±0.014 |
| GRU | W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (hidden=128, layers=2) | 67.8±11.1 | 44.8±5.5 | 57.1±6.5 | 0.928±0.016 |
| GRU | W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (hidden=64, layers=2) | 67.2±6.3 | 46.2±1.4 | 61.1±1.9 | 0.918±0.005 |
| GRU | W0_D | W0: first 50 cycles | D — OneCycleLR (hidden=128, layers=2) | 66.7±5.0 | 57.5±7.7 | 72.6±9.1 | 0.883±0.030 |
| Transformer | W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (current) | 66.1±6.8 | 44.7±4.4 | 59.6±6.6 | 0.921±0.017 |
| GRU | W1_C | W1: first 100 cycles | C — Wider (hidden=256, layers=1) | 63.9±3.7 | 57.8±4.3 | 76.3±6.1 | 0.872±0.020 |
| GRU | W0_A | W0: first 50 cycles | A — Baseline (hidden=128, layers=1) | 63.9±6.7 | 55.5±3.7 | 72.1±3.6 | 0.886±0.012 |
| LSTM | W0_D | W0: first 50 cycles | D — OneCycleLR (hidden=128, layers=2) | 63.3±7.5 | 55.2±4.6 | 72.5±5.4 | 0.884±0.017 |
| GRU | W0_B | W0: first 50 cycles | B — Deeper (hidden=128, layers=2) | 62.8±5.6 | 54.8±3.4 | 72.2±4.0 | 0.886±0.012 |
| LSTM | W1_B | W1: first 100 cycles | B — Deeper (hidden=128, layers=2) | 62.2±4.8 | 58.7±4.2 | 76.7±5.2 | 0.870±0.018 |
| Transformer | W0_E | W0: first 50 cycles | E — L1 Loss + Larger | 62.2±8.2 | 59.8±12.6 | 81.8±16.1 | 0.848±0.062 |
| LSTM | W0_B | W0: first 50 cycles | B — Deeper (hidden=128, layers=2) | 62.2±8.2 | 55.9±5.4 | 71.5±6.7 | 0.887±0.022 |
| LSTM | W0_A | W0: first 50 cycles | A — Baseline (hidden=128, layers=1) | 62.2±8.2 | 56.3±5.6 | 72.4±7.4 | 0.884±0.023 |
| LSTM | W1_A | W1: first 100 cycles | A — Baseline (hidden=128, layers=1) | 61.7±8.0 | 56.0±5.3 | 72.3±5.9 | 0.885±0.019 |
| GRU | W1_A | W1: first 100 cycles | A — Baseline (hidden=128, layers=1) | 61.7±3.9 | 59.1±2.2 | 76.4±3.4 | 0.872±0.011 |
| GRU | W1_E | W1: first 100 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 61.7±3.9 | 52.1±4.9 | 69.3±5.8 | 0.894±0.018 |
| GRU | W0_C | W0: first 50 cycles | C — Wider (hidden=256, layers=1) | 61.7±11.5 | 61.4±12.7 | 78.8±18.3 | 0.856±0.078 |
| Transformer | W1_E | W1: first 100 cycles | E — L1 Loss + Larger | 61.1±6.1 | 59.7±7.5 | 81.8±9.4 | 0.851±0.033 |
| GRU | W1_B | W1: first 100 cycles | B — Deeper (hidden=128, layers=2) | 60.6±5.2 | 58.7±4.5 | 75.1±5.4 | 0.876±0.018 |
| LSTM | W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (hidden=64, layers=2) | 60.0±4.8 | 44.1±4.9 | 58.2±7.0 | 0.925±0.019 |
| LSTM | W1_C | W1: first 100 cycles | C — Wider (hidden=256, layers=1) | 59.4±6.1 | 61.1±4.7 | 78.4±7.8 | 0.864±0.029 |
| GRU | W1_D | W1: first 100 cycles | D — OneCycleLR (hidden=128, layers=2) | 59.4±4.3 | 60.8±6.5 | 78.4±6.7 | 0.864±0.024 |
| Transformer | W1_B | W1: first 100 cycles | B — Deeper + Wider | 58.9±6.2 | 64.0±6.1 | 84.6±12.2 | 0.840±0.047 |
| LSTM | W1_E | W1: first 100 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 58.3±5.7 | 57.9±4.6 | 74.2±6.7 | 0.878±0.022 |
| LSTM | W1_D | W1: first 100 cycles | D — OneCycleLR (hidden=128, layers=2) | 58.3±5.1 | 59.8±4.5 | 77.4±4.8 | 0.868±0.017 |
| LSTM | W0_E | W0: first 50 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 57.8±6.2 | 56.8±3.3 | 76.0±4.3 | 0.873±0.015 |
| Transformer | W0_C | W0: first 50 cycles | C — Regularized Small | 56.7±9.9 | 66.4±11.5 | 88.2±18.0 | 0.822±0.072 |
| Transformer | W1_D | W1: first 100 cycles | D — OneCycleLR | 56.1±4.6 | 64.7±9.7 | 85.7±12.0 | 0.836±0.044 |
| Transformer | W1_C | W1: first 100 cycles | C — Regularized Small | 55.6±6.1 | 64.8±6.8 | 88.9±10.4 | 0.824±0.040 |
| LSTM | W0_C | W0: first 50 cycles | C — Wider (hidden=256, layers=1) | 55.0±8.8 | 59.0±6.0 | 74.2±6.5 | 0.879±0.022 |
| Transformer | W0_A | W0: first 50 cycles | A — Baseline (current) | 55.0±5.2 | 67.7±11.2 | 89.0±15.6 | 0.821±0.063 |
| GRU | W0_E | W0: first 50 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 54.4±7.8 | 57.2±3.2 | 76.6±2.7 | 0.871±0.009 |
| Transformer | W0_D | W0: first 50 cycles | D — OneCycleLR | 53.9±6.6 | 68.1±7.5 | 87.3±9.0 | 0.831±0.035 |
| Transformer | W1_A | W1: first 100 cycles | A — Baseline (current) | 52.8±5.1 | 64.9±6.9 | 85.1±9.8 | 0.839±0.038 |
| Transformer | W0_B | W0: first 50 cycles | B — Deeper + Wider | 50.6±5.2 | 71.0±7.5 | 91.5±8.4 | 0.815±0.034 |
