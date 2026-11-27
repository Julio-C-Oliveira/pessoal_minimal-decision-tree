# Fixed-Point Q16.16 vs Float32 Benchmark Report (Phase 8 — Delivery 8.5)

Direct A/B parity evaluation across all 8 flagship datasets of `main_benchmark/data/` comparing **IEEE 754 `float32`** vs **Zero-Libm `Q16.16` Fixed-Point (`-DHT_USE_FIXED_POINT`)** using identical tuned hyperparameters on both **High Tier (`Tier 256` / `Tier 40`)** and **MCU `Tier 4`**.

## High Tier (Tier 256 / Tier 40)

### Classification Datasets (4)

| Dataset | Float32 Holdout Acc | Q16.16 Holdout Acc | Delta Acc (pp) | Float32 Preq Acc | Q16.16 Preq Acc | Float32 Nodes | Q16.16 Nodes | Float32 RAM (KB) | Q16.16 RAM (KB) | Float32 us/s | Q16.16 us/s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | 99.22% | 99.15% | -0.07 pp | 97.41% | 97.41% | 45 | 45 | 468.2 | 468.2 | 0.49 | 1.33 |
| **Electricity** | 64.01% | 63.34% | -0.66 pp | 64.07% | 63.94% | 127 | 123 | 474.8 | 474.8 | 2.60 | 7.72 |
| **Forest Covertype** | 68.92% | 67.83% | -1.09 pp | 60.25% | 60.61% | 113 | 119 | 453.0 | 453.0 | 7.04 | 23.14 |
| **Airlines** | 75.76% | 76.00% | +0.24 pp | 73.92% | 74.09% | 45 | 41 | 468.2 | 468.2 | 1.95 | 4.82 |

### Regression Datasets (4)

| Dataset | Float32 MAE | Q16.16 MAE | Delta MAE | Float32 RMSE | Q16.16 RMSE | Float32 R² | Q16.16 R² | Float32 Nodes | Q16.16 Nodes | Float32 us/s | Q16.16 us/s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | 1.5513 | 1.8120 | +0.2607 | 1.9664 | 2.2851 | 0.8444 | 0.7899 | 51 | 49 | 2.79 | 7.19 |
| **Bikes** | 18.8594 | 17.6582 | -1.2012 | 29.0653 | 29.1697 | 0.8502 | 0.8491 | 93 | 93 | 1.43 | 2.10 |
| **Appliances Energy** | 26.8277 | 25.4207 | -1.4070 | 33.9524 | 31.8414 | 0.7938 | 0.8186 | 105 | 119 | 10.19 | 16.09 |
| **NYC Taxi** | 277.1476 | 296.6093 | +19.4617 | 369.6050 | 396.7166 | 0.4417 | 0.3568 | 75 | 79 | 5.17 | 9.37 |

## MCU Tier 4 (4 Active Leaves / 64 Nodes)

### Classification Datasets (4)

| Dataset | Float32 Holdout Acc | Q16.16 Holdout Acc | Delta Acc (pp) | Float32 Preq Acc | Q16.16 Preq Acc | Float32 Nodes | Q16.16 Nodes | Float32 RAM (KB) | Q16.16 RAM (KB) | Float32 us/s | Q16.16 us/s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | 97.30% | 97.57% | +0.27 pp | 94.99% | 95.14% | 21 | 21 | 14.1 | 14.1 | 0.72 | 2.32 |
| **Electricity** | 64.18% | 64.13% | -0.05 pp | 64.69% | 64.60% | 19 | 21 | 14.1 | 14.1 | 2.03 | 5.89 |
| **Forest Covertype** | 64.19% | 64.44% | +0.25 pp | 58.93% | 59.24% | 47 | 49 | 43.7 | 43.7 | 3.21 | 11.20 |
| **Airlines** | 77.19% | 77.19% | +0.00 pp | 76.14% | 76.16% | 33 | 33 | 7.5 | 7.5 | 0.36 | 0.90 |

### Regression Datasets (4)

| Dataset | Float32 MAE | Q16.16 MAE | Delta MAE | Float32 RMSE | Q16.16 RMSE | Float32 R² | Q16.16 R² | Float32 Nodes | Q16.16 Nodes | Float32 us/s | Q16.16 us/s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | 2.3536 | 2.3655 | +0.0119 | 2.9588 | 2.9732 | 0.6477 | 0.6443 | 35 | 25 | 1.14 | 3.13 |
| **Bikes** | 44.6479 | 47.8470 | +3.1991 | 58.8770 | 65.7574 | 0.3852 | 0.2332 | 25 | 21 | 1.05 | 2.65 |
| **Appliances Energy** | 36.2031 | 33.6763 | -2.5268 | 46.6150 | 43.5503 | 0.6112 | 0.6607 | 33 | 29 | 3.01 | 4.56 |
| **NYC Taxi** | 348.9519 | 354.2735 | +5.3215 | 453.9079 | 465.8703 | 0.1579 | 0.1130 | 35 | 25 | 0.93 | 2.39 |

