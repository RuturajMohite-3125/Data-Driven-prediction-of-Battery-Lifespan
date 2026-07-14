# Global Sweep Comparison — All Models

Comparison across Transformer, GRU, LSTM, CNN-GRU, and CNN-LSTM on test set (mean ± std across seeds).

| Model | ID | Window | Config | Acc% | MAE | RMSE | R² |
|-------|----|--------|--------|------|-----|------|-----|
| GRU | W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (hidden=64, layers=2) | 100.0±0.0 | 32.2±2.4 | 41.9±3.1 | 0.961±0.006 |
| LSTM | W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper (hidden=128, layers=2) | 99.4±1.8 | 28.5±2.3 | 37.9±3.0 | 0.968±0.005 |
| GRU | W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (hidden=128, layers=1) | 98.9±2.3 | 28.9±3.0 | 40.2±5.2 | 0.964±0.009 |
| LSTM | W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (hidden=128, layers=1) | 98.9±2.3 | 27.7±2.4 | 38.9±3.8 | 0.967±0.006 |
| GRU | W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper (hidden=128, layers=2) | 98.3±2.7 | 29.1±3.0 | 40.6±4.4 | 0.964±0.008 |
| LSTM | W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Wider (hidden=256, layers=1) | 97.2±3.9 | 30.6±3.5 | 43.3±4.5 | 0.958±0.009 |
| GRU | W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Wider (hidden=256, layers=1) | 97.2±2.9 | 31.3±1.8 | 42.1±2.1 | 0.961±0.004 |
| LSTM | W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (hidden=64, layers=2) | 97.2±2.9 | 31.8±2.2 | 39.9±2.8 | 0.965±0.005 |
| GRU | W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (hidden=128, layers=2) | 94.4±2.6 | 33.1±4.3 | 50.0±7.4 | 0.944±0.016 |
| LSTM | W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (hidden=128, layers=2) | 93.9±3.2 | 32.0±5.2 | 46.3±4.8 | 0.952±0.010 |
| LSTM | W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (hidden=128, layers=1) | 93.3±4.4 | 32.5±2.2 | 45.6±3.5 | 0.954±0.007 |
| GRU | W4_C | W4: three bands 1-50,100-150,200-250 | C — Wider (hidden=256, layers=1) | 92.8±5.9 | 31.9±2.0 | 44.0±2.9 | 0.957±0.006 |
| GRU | W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (hidden=64, layers=2) | 92.8±2.7 | 31.5±2.2 | 42.6±3.2 | 0.960±0.006 |
| LSTM | W4_C | W4: three bands 1-50,100-150,200-250 | C — Wider (hidden=256, layers=1) | 92.2±7.9 | 35.1±3.4 | 48.4±6.5 | 0.948±0.014 |
| GRU | W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (hidden=128, layers=1) | 92.2±5.4 | 31.5±4.2 | 45.2±6.4 | 0.954±0.012 |
| LSTM | W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (hidden=64, layers=2) | 92.2±2.9 | 32.1±2.6 | 43.5±4.3 | 0.958±0.009 |
| Transformer | W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (current) | 92.2±3.9 | 34.4±4.8 | 46.1±7.9 | 0.952±0.017 |
| Transformer | W4_C | W4: three bands 1-50,100-150,200-250 | C — Regularized Small | 91.1±4.7 | 37.6±5.9 | 50.3±9.1 | 0.943±0.021 |
| LSTM | W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper (hidden=128, layers=2) | 91.1±2.9 | 28.4±1.7 | 43.8±3.1 | 0.958±0.006 |
| GRU | W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (hidden=128, layers=1) | 91.1±3.9 | 32.6±4.0 | 43.4±4.6 | 0.958±0.009 |
| Transformer | W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR | 90.6±6.4 | 40.3±5.0 | 57.3±11.5 | 0.926±0.033 |
| GRU | W2_C | W2: 1-250 stride-10 (25 pts) | C — Wider (hidden=256, layers=1) | 90.6±5.9 | 31.3±3.7 | 42.6±4.0 | 0.960±0.008 |
| GRU | W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper (hidden=128, layers=2) | 90.0±3.5 | 37.1±3.4 | 56.7±5.5 | 0.929±0.014 |
| LSTM | W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (hidden=128, layers=1) | 90.0±7.8 | 35.5±2.5 | 47.4±3.6 | 0.950±0.008 |
| Transformer | W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR | 89.4±4.1 | 38.3±4.0 | 53.1±8.7 | 0.937±0.020 |
| LSTM | W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (hidden=128, layers=1) | 89.4±4.1 | 32.1±4.1 | 46.7±5.1 | 0.952±0.010 |
| LSTM | W2_C | W2: 1-250 stride-10 (25 pts) | C — Wider (hidden=256, layers=1) | 89.4±6.1 | 33.4±3.1 | 46.7±4.6 | 0.952±0.010 |
| Transformer | W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Larger | 88.9±3.7 | 34.6±4.7 | 47.5±6.8 | 0.950±0.014 |
| CNN-LSTM | W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (cnn=64, lstm=128, layers=1) | 88.9±8.3 | 38.9±10.2 | 56.7±17.2 | 0.924±0.049 |
| CNN-LSTM | W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 88.9±8.3 | 38.7±11.2 | 52.9±15.6 | 0.934±0.042 |
| Transformer | W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper + Wider | 88.9±4.5 | 37.0±6.2 | 50.9±8.1 | 0.942±0.016 |
| Transformer | W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (current) | 88.9±4.5 | 38.1±4.9 | 57.1±12.5 | 0.925±0.036 |
| GRU | W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (hidden=64, layers=2) | 88.9±4.5 | 34.2±2.4 | 50.0±4.5 | 0.945±0.010 |
| LSTM | W4_B | W4: three bands 1-50,100-150,200-250 | B — Deeper (hidden=128, layers=2) | 88.9±4.5 | 33.7±4.7 | 49.1±6.4 | 0.946±0.014 |
| GRU | W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper (hidden=128, layers=2) | 88.9±5.2 | 33.4±3.1 | 44.1±5.1 | 0.957±0.010 |
| GRU | W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper (hidden=128, layers=2) | 88.3±4.9 | 31.6±2.9 | 45.9±6.3 | 0.953±0.013 |
| CNN-LSTM | W3_B | W3: early(10-40)+late(180-200) [original] | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 88.3±6.1 | 38.9±7.1 | 53.6±10.2 | 0.935±0.025 |
| LSTM | W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (hidden=128, layers=2) | 87.8±5.1 | 32.5±5.0 | 48.7±4.9 | 0.948±0.011 |
| Transformer | W2_C | W2: 1-250 stride-10 (25 pts) | C — Regularized Small | 87.8±4.4 | 40.3±6.4 | 55.9±10.4 | 0.929±0.025 |
| CNN-LSTM | W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 87.2±8.3 | 41.3±9.6 | 55.5±13.8 | 0.929±0.037 |
| Transformer | W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR | 86.7±4.7 | 41.6±6.8 | 60.8±13.4 | 0.915±0.039 |
| Transformer | W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Regularized Small | 86.7±4.7 | 41.5±4.8 | 60.3±10.0 | 0.918±0.028 |
| LSTM | W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (hidden=64, layers=2) | 86.7±2.9 | 37.1±2.2 | 53.1±2.7 | 0.938±0.006 |
| CNN-GRU | W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 86.7±6.5 | 46.7±6.5 | 63.3±10.5 | 0.910±0.029 |
| LSTM | W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper (hidden=128, layers=2) | 86.7±4.7 | 35.4±3.1 | 48.8±4.8 | 0.947±0.010 |
| GRU | W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (hidden=128, layers=2) | 86.1±7.1 | 40.4±4.3 | 60.3±6.1 | 0.920±0.016 |
| LSTM | W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (hidden=64, layers=2) | 86.1±4.7 | 35.3±2.5 | 47.6±3.1 | 0.950±0.007 |
| CNN-LSTM | W2_C | W2: 1-250 stride-10 (25 pts) | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 85.6±4.7 | 45.9±7.8 | 62.0±8.7 | 0.914±0.023 |
| Transformer | W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (current) | 85.6±5.4 | 42.2±5.9 | 60.6±8.7 | 0.918±0.026 |
| Transformer | W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Deeper + Wider | 85.6±7.5 | 42.1±5.3 | 58.8±8.9 | 0.923±0.022 |
| Transformer | W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Larger | 85.6±6.5 | 41.6±5.6 | 60.6±12.9 | 0.916±0.035 |
| CNN-LSTM | W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (cnn=64, lstm=128, layers=1) | 85.0±4.6 | 44.7±7.3 | 62.7±8.6 | 0.912±0.024 |
| LSTM | W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (hidden=128, layers=2) | 85.0±8.3 | 38.5±5.4 | 54.6±5.1 | 0.934±0.013 |
| CNN-LSTM | W2_B | W2: 1-250 stride-10 (25 pts) | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 85.0±7.4 | 46.1±4.7 | 62.7±8.6 | 0.912±0.024 |
| CNN-LSTM | W3_C | W3: early(10-40)+late(180-200) [original] | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 85.0±6.4 | 42.9±9.7 | 59.4±14.3 | 0.919±0.041 |
| GRU | W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (hidden=64, layers=2) | 85.0±2.7 | 37.1±3.4 | 48.6±5.1 | 0.948±0.011 |
| CNN-LSTM | W4_C | W4: three bands 1-50,100-150,200-250 | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 85.0±4.6 | 42.8±6.6 | 62.2±6.0 | 0.915±0.017 |
| GRU | W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (hidden=128, layers=1) | 85.0±3.7 | 36.2±3.2 | 47.7±4.8 | 0.950±0.010 |
| Transformer | W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR | 84.4±5.1 | 44.6±7.8 | 62.1±12.0 | 0.913±0.033 |
| Transformer | W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Larger | 84.4±6.8 | 43.3±6.3 | 60.5±14.4 | 0.916±0.041 |
| Transformer | W3_B | W3: early(10-40)+late(180-200) [original] | B — Deeper + Wider | 84.4±5.7 | 44.7±6.7 | 60.7±8.4 | 0.918±0.022 |
| GRU | W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (hidden=128, layers=2) | 84.4±4.4 | 41.2±5.0 | 55.3±6.9 | 0.932±0.018 |
| GRU | W3_C | W3: early(10-40)+late(180-200) [original] | C — Wider (hidden=256, layers=1) | 84.4±3.5 | 36.0±3.0 | 48.7±6.6 | 0.947±0.015 |
| CNN-GRU | W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (cnn=64, gru=128, layers=1) | 84.4±8.2 | 43.2±10.4 | 59.7±12.9 | 0.919±0.035 |
| CNN-LSTM | W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 84.4±8.2 | 45.8±7.8 | 60.2±8.6 | 0.919±0.022 |
| CNN-GRU | W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 84.4±11.4 | 52.7±7.4 | 63.2±6.8 | 0.911±0.018 |
| Transformer | W3_A | W3: early(10-40)+late(180-200) [original] | A — Baseline (current) | 83.9±4.1 | 46.4±6.3 | 63.7±8.0 | 0.910±0.023 |
| LSTM | W3_C | W3: early(10-40)+late(180-200) [original] | C — Wider (hidden=256, layers=1) | 83.9±3.2 | 37.6±3.9 | 50.7±5.9 | 0.943±0.013 |
| LSTM | W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (hidden=128, layers=2) | 83.9±6.1 | 40.0±4.0 | 52.8±3.9 | 0.939±0.009 |
| CNN-GRU | W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 83.3±6.4 | 47.1±3.4 | 59.3±5.1 | 0.922±0.013 |
| CNN-LSTM | W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (cnn=64, lstm=128, layers=1) | 83.3±7.4 | 47.8±9.8 | 62.2±9.9 | 0.913±0.027 |
| CNN-LSTM | W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 83.3±7.4 | 45.9±9.9 | 63.5±8.3 | 0.910±0.024 |
| GRU | W1_C | W1: first 100 cycles | C — Wider (hidden=256, layers=1) | 82.8±5.5 | 58.1±6.3 | 74.7±7.2 | 0.877±0.024 |
| CNN-LSTM | W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 82.8±6.1 | 46.9±8.0 | 62.9±5.5 | 0.913±0.016 |
| Transformer | W2_B | W2: 1-250 stride-10 (25 pts) | B — Deeper + Wider | 82.8±4.1 | 44.6±5.8 | 63.1±9.7 | 0.911±0.027 |
| CNN-GRU | W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (cnn=64, gru=128, layers=2) | 82.8±6.7 | 52.4±6.2 | 65.9±8.9 | 0.903±0.026 |
| GRU | W1_A | W1: first 100 cycles | A — Baseline (hidden=128, layers=1) | 82.2±5.1 | 54.5±4.2 | 72.5±5.0 | 0.884±0.016 |
| CNN-GRU | W3_D | W3: early(10-40)+late(180-200) [original] | D — OneCycleLR (cnn=64, gru=128, layers=2) | 82.2±8.2 | 51.6±8.5 | 66.4±8.6 | 0.902±0.025 |
| Transformer | W3_C | W3: early(10-40)+late(180-200) [original] | C — Regularized Small | 82.2±4.4 | 50.5±5.7 | 69.7±7.4 | 0.892±0.024 |
| CNN-LSTM | W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 82.2±2.3 | 47.3±4.8 | 62.1±6.9 | 0.914±0.019 |
| GRU | W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (hidden=128, layers=2) | 81.7±5.9 | 37.7±5.6 | 54.2±6.0 | 0.935±0.014 |
| CNN-GRU | W4_A | W4: three bands 1-50,100-150,200-250 | A — Baseline (cnn=64, gru=128, layers=1) | 81.7±5.9 | 48.1±10.1 | 68.9±11.1 | 0.893±0.034 |
| CNN-GRU | W2_A | W2: 1-250 stride-10 (25 pts) | A — Baseline (cnn=64, gru=128, layers=1) | 81.7±5.3 | 45.5±6.6 | 57.1±7.8 | 0.927±0.019 |
| CNN-GRU | W2_E | W2: 1-250 stride-10 (25 pts) | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 81.7±5.3 | 50.2±5.5 | 64.1±8.4 | 0.909±0.025 |
| CNN-GRU | W3_B | W3: early(10-40)+late(180-200) [original] | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 81.7±3.7 | 43.4±4.6 | 56.3±4.7 | 0.930±0.012 |
| CNN-GRU | W3_C | W3: early(10-40)+late(180-200) [original] | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 81.7±5.9 | 46.5±8.1 | 58.9±12.3 | 0.921±0.033 |
| LSTM | W0_C | W0: first 50 cycles | C — Wider (hidden=256, layers=1) | 81.1±6.0 | 56.3±8.7 | 72.0±11.0 | 0.884±0.038 |
| CNN-GRU | W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (cnn=64, gru=128, layers=1) | 81.1±8.8 | 54.1±11.7 | 68.0±14.9 | 0.894±0.051 |
| CNN-GRU | W2_D | W2: 1-250 stride-10 (25 pts) | D — OneCycleLR (cnn=64, gru=128, layers=2) | 80.6±7.5 | 49.3±8.1 | 64.0±8.5 | 0.909±0.024 |
| CNN-LSTM | W4_B | W4: three bands 1-50,100-150,200-250 | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 80.6±2.9 | 46.9±8.0 | 64.7±5.5 | 0.908±0.015 |
| GRU | W1_E | W1: first 100 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 80.6±5.4 | 54.8±3.0 | 70.8±3.4 | 0.890±0.011 |
| CNN-GRU | W4_C | W4: three bands 1-50,100-150,200-250 | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 80.6±6.5 | 45.2±9.1 | 60.6±10.4 | 0.917±0.028 |
| LSTM | W0_E | W0: first 50 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 80.0±5.4 | 54.7±2.8 | 74.0±3.9 | 0.880±0.013 |
| Transformer | W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Larger | 80.0±9.5 | 50.3±7.0 | 71.4±10.8 | 0.886±0.033 |
| GRU | W0_E | W0: first 50 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 80.0±4.7 | 58.4±3.2 | 77.3±2.6 | 0.869±0.009 |
| CNN-GRU | W2_C | W2: 1-250 stride-10 (25 pts) | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 80.0±6.0 | 48.6±8.9 | 61.8±8.5 | 0.915±0.024 |
| GRU | W0_B | W0: first 50 cycles | B — Deeper (hidden=128, layers=2) | 80.0±6.5 | 60.5±4.0 | 78.7±4.1 | 0.864±0.014 |
| GRU | W1_B | W1: first 100 cycles | B — Deeper (hidden=128, layers=2) | 80.0±5.4 | 60.1±6.2 | 76.6±6.7 | 0.870±0.022 |
| LSTM | W1_A | W1: first 100 cycles | A — Baseline (hidden=128, layers=1) | 80.0±6.0 | 56.7±7.2 | 72.2±9.6 | 0.884±0.033 |
| CNN-GRU | W4_D | W4: three bands 1-50,100-150,200-250 | D — OneCycleLR (cnn=64, gru=128, layers=2) | 79.4±7.0 | 53.3±9.8 | 74.5±8.0 | 0.877±0.026 |
| CNN-GRU | W2_B | W2: 1-250 stride-10 (25 pts) | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 78.9±3.5 | 48.0±6.3 | 63.8±6.5 | 0.910±0.018 |
| LSTM | W1_E | W1: first 100 cycles | E — L1 Loss + Small (hidden=64, layers=2) | 78.9±4.4 | 54.8±3.2 | 69.1±3.9 | 0.895±0.012 |
| CNN-GRU | W4_E | W4: three bands 1-50,100-150,200-250 | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 78.9±6.8 | 51.1±6.8 | 66.3±8.0 | 0.902±0.023 |
| CNN-LSTM | W5_C | W5: sparse landmarks [1,10,50,100,150,200,250] | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 78.3±8.0 | 51.0±6.7 | 63.9±7.5 | 0.910±0.021 |
| CNN-GRU | W4_B | W4: three bands 1-50,100-150,200-250 | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 78.3±5.5 | 46.6±6.9 | 64.2±7.4 | 0.909±0.021 |
| CNN-LSTM | W5_B | W5: sparse landmarks [1,10,50,100,150,200,250] | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 78.3±8.5 | 50.6±7.2 | 68.1±17.1 | 0.892±0.064 |
| GRU | W1_D | W1: first 100 cycles | D — OneCycleLR (hidden=128, layers=2) | 77.2±8.5 | 61.5±8.9 | 78.5±10.5 | 0.863±0.036 |
| LSTM | W0_D | W0: first 50 cycles | D — OneCycleLR (hidden=128, layers=2) | 77.2±7.6 | 58.6±8.4 | 74.2±10.0 | 0.877±0.032 |
| LSTM | W1_D | W1: first 100 cycles | D — OneCycleLR (hidden=128, layers=2) | 77.2±7.6 | 58.1±7.7 | 72.4±10.1 | 0.883±0.035 |
| CNN-LSTM | W5_E | W5: sparse landmarks [1,10,50,100,150,200,250] | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 77.2±5.5 | 53.5±6.8 | 66.7±8.8 | 0.901±0.028 |
| LSTM | W0_A | W0: first 50 cycles | A — Baseline (hidden=128, layers=1) | 76.7±5.1 | 57.7±6.3 | 73.8±6.2 | 0.880±0.020 |
| LSTM | W1_C | W1: first 100 cycles | C — Wider (hidden=256, layers=1) | 76.7±6.3 | 56.9±6.6 | 72.3±7.3 | 0.884±0.023 |
| LSTM | W0_B | W0: first 50 cycles | B — Deeper (hidden=128, layers=2) | 76.7±5.7 | 58.6±3.9 | 77.4±4.5 | 0.868±0.015 |
| CNN-GRU | W3_E | W3: early(10-40)+late(180-200) [original] | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 76.7±8.6 | 53.7±7.2 | 67.9±10.6 | 0.897±0.032 |
| CNN-LSTM | W5_D | W5: sparse landmarks [1,10,50,100,150,200,250] | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 76.7±8.6 | 51.0±6.7 | 63.9±6.4 | 0.910±0.018 |
| GRU | W0_A | W0: first 50 cycles | A — Baseline (hidden=128, layers=1) | 76.1±8.3 | 60.1±8.4 | 76.2±10.5 | 0.870±0.037 |
| CNN-LSTM | W5_A | W5: sparse landmarks [1,10,50,100,150,200,250] | A — Baseline (cnn=64, lstm=128, layers=1) | 76.1±8.3 | 52.0±4.7 | 65.7±5.5 | 0.905±0.017 |
| LSTM | W1_B | W1: first 100 cycles | B — Deeper (hidden=128, layers=2) | 75.6±6.0 | 57.6±4.9 | 72.3±5.5 | 0.885±0.018 |
| GRU | W0_D | W0: first 50 cycles | D — OneCycleLR (hidden=128, layers=2) | 75.0±7.1 | 61.8±5.9 | 77.1±8.2 | 0.868±0.028 |
| Transformer | W0_E | W0: first 50 cycles | E — L1 Loss + Larger | 74.4±6.5 | 63.4±7.4 | 86.1±10.4 | 0.835±0.039 |
| CNN-LSTM | W0_E | W0: first 50 cycles | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 74.4±6.5 | 72.8±12.9 | 100.7±19.5 | 0.770±0.089 |
| GRU | W0_C | W0: first 50 cycles | C — Wider (hidden=256, layers=1) | 74.4±6.5 | 64.4±7.9 | 83.8±10.5 | 0.844±0.041 |
| CNN-GRU | W0_D | W0: first 50 cycles | D — OneCycleLR (cnn=64, gru=128, layers=2) | 73.3±9.7 | 71.2±17.1 | 94.0±25.6 | 0.793±0.123 |
| CNN-GRU | W1_B | W1: first 100 cycles | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 72.8±5.5 | 66.3±6.9 | 89.9±10.4 | 0.821±0.039 |
| CNN-GRU | W1_E | W1: first 100 cycles | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 72.8±5.5 | 65.5±9.1 | 86.6±12.6 | 0.832±0.050 |
| Transformer | W0_D | W0: first 50 cycles | D — OneCycleLR | 72.2±5.9 | 67.7±7.2 | 90.1±10.8 | 0.820±0.042 |
| CNN-GRU | W1_C | W1: first 100 cycles | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 72.2±5.9 | 63.0±6.2 | 84.4±8.6 | 0.842±0.031 |
| CNN-GRU | W0_A | W0: first 50 cycles | A — Baseline (cnn=64, gru=128, layers=1) | 72.2±6.9 | 65.2±11.3 | 93.4±16.1 | 0.804±0.072 |
| CNN-GRU | W0_B | W0: first 50 cycles | B — Wider CNN + Deeper GRU (cnn=128, gru=128, layers=2) | 71.7±8.0 | 63.4±13.6 | 87.8±19.8 | 0.823±0.080 |
| CNN-GRU | W0_E | W0: first 50 cycles | E — L1 Loss + Small (cnn=32, gru=64, layers=2) | 71.7±9.2 | 69.8±10.5 | 91.6±15.6 | 0.811±0.065 |
| CNN-GRU | W1_A | W1: first 100 cycles | A — Baseline (cnn=64, gru=128, layers=1) | 71.7±6.1 | 70.4±7.1 | 101.5±15.2 | 0.770±0.066 |
| CNN-LSTM | W1_C | W1: first 100 cycles | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 71.7±7.1 | 75.5±12.1 | 104.9±24.4 | 0.747±0.115 |
| Transformer | W1_D | W1: first 100 cycles | D — OneCycleLR | 71.1±5.7 | 66.5±6.3 | 83.9±7.4 | 0.845±0.027 |
| CNN-GRU | W0_C | W0: first 50 cycles | C — Narrow CNN + Wide GRU (cnn=32, gru=256, layers=1) | 71.1±10.1 | 65.0±11.7 | 87.2±15.0 | 0.829±0.060 |
| CNN-LSTM | W0_B | W0: first 50 cycles | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 71.1±5.7 | 69.8±8.2 | 97.5±12.8 | 0.788±0.055 |
| CNN-LSTM | W1_B | W1: first 100 cycles | B — Wider CNN + Deeper LSTM (cnn=128, lstm=128, layers=2) | 71.1±5.7 | 78.9±15.8 | 111.1±25.5 | 0.716±0.131 |
| CNN-LSTM | W0_C | W0: first 50 cycles | C — Narrow CNN + Wide LSTM (cnn=32, lstm=256, layers=1) | 71.1±6.3 | 68.4±10.1 | 95.8±13.3 | 0.795±0.055 |
| CNN-LSTM | W1_A | W1: first 100 cycles | A — Baseline (cnn=64, lstm=128, layers=1) | 71.1±8.2 | 79.5±12.3 | 116.5±19.7 | 0.695±0.104 |
| Transformer | W0_C | W0: first 50 cycles | C — Regularized Small | 70.6±6.4 | 68.1±5.3 | 91.6±6.7 | 0.815±0.027 |
| Transformer | W1_C | W1: first 100 cycles | C — Regularized Small | 70.6±5.9 | 66.6±5.6 | 88.0±11.4 | 0.828±0.047 |
| CNN-LSTM | W0_D | W0: first 50 cycles | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 70.6±7.0 | 71.8±11.2 | 99.2±20.3 | 0.776±0.090 |
| CNN-LSTM | W1_E | W1: first 100 cycles | E — L1 Loss + Small (cnn=32, lstm=64, layers=2) | 70.6±7.4 | 78.3±10.0 | 107.6±22.7 | 0.736±0.114 |
| CNN-LSTM | W1_D | W1: first 100 cycles | D — OneCycleLR (cnn=64, lstm=128, layers=2) | 70.6±5.9 | 77.8±13.3 | 106.3±23.3 | 0.742±0.114 |
| CNN-GRU | W1_D | W1: first 100 cycles | D — OneCycleLR (cnn=64, gru=128, layers=2) | 70.0±5.4 | 76.3±13.3 | 103.8±25.0 | 0.751±0.119 |
| Transformer | W0_B | W0: first 50 cycles | B — Deeper + Wider | 68.9±6.0 | 71.6±6.3 | 97.2±11.9 | 0.790±0.052 |
| CNN-LSTM | W0_A | W0: first 50 cycles | A — Baseline (cnn=64, lstm=128, layers=1) | 68.9±7.9 | 72.6±10.5 | 100.5±14.2 | 0.775±0.063 |
| Transformer | W0_A | W0: first 50 cycles | A — Baseline (current) | 68.9±7.5 | 68.5±6.9 | 90.5±9.6 | 0.819±0.040 |
| Transformer | W1_E | W1: first 100 cycles | E — L1 Loss + Larger | 68.9±7.5 | 63.6±10.3 | 83.6±13.1 | 0.843±0.050 |
| Transformer | W1_B | W1: first 100 cycles | B — Deeper + Wider | 68.3±6.4 | 70.5±10.6 | 90.9±13.8 | 0.815±0.056 |
| Transformer | W1_A | W1: first 100 cycles | A — Baseline (current) | 67.2±6.7 | 67.8±6.1 | 88.3±9.2 | 0.827±0.033 |
