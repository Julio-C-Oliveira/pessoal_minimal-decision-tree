# Relatório Consolidado de Otimização e Sensibilidade de Hiperparâmetros

Suíte automatizada de exploração de hiperparâmetros comparando **C11 libht**, **River** e **Scikit-Learn** em 8 datasets emblemáticos.

---

## Dataset: SEA (Classification)

- **Descrição:** SEA (Synthetic, 3 continuous, 2 classes)
- **Total de Experimentos (Trials):** 73

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=400`, `delta=0.01`, `tau=0.01`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 95.13% | 94.39% | 19.6 KB | 0.07 µs |
| #2 | `grace_period=100`, `delta=0.01`, `tau=0.1`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 94.83% | 94.16% | 19.6 KB | 0.09 µs |
| #3 | `grace_period=50`, `delta=0.0001`, `tau=0.01`, `tier=4`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 93.75% | 92.40% | 26.1 KB | 0.54 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=400`, `delta=0.001`, `tau=0.15`, `tier=8`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.12% | 97.84% | 45.3 KB | 0.86 µs |
| #2 | `grace_period=50`, `delta=0.0001`, `tau=0.1`, `tier=8`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 97.83% | 97.50% | 38.8 KB | 0.09 µs |
| #3 | `grace_period=50`, `delta=0.01`, `tau=0.1`, `tier=8`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 97.60% | 97.23% | 38.8 KB | 0.08 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=0.01`, `tau=0.15`, `tier=16`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 98.70% | 98.51% | 83.7 KB | 0.44 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.01`, `tau=0.05`, `tier=64`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 99.02% | 98.87% | 307.5 KB | 0.32 µs |
| #2 | `grace_period=200`, `delta=1e-07`, `tau=0.05`, `tier=64`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 94.43% | 93.68% | 307.5 KB | 0.30 µs |

#### C11 libht (Tier 128)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=400`, `delta=0.001`, `tau=0.15`, `tier=128`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 98.97% | 98.81% | 621.2 KB | 0.37 µs |
| #2 | `grace_period=100`, `delta=0.0001`, `tau=0.01`, `tier=128`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 93.80% | 92.47% | 614.6 KB | 0.24 µs |

#### River

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `leaf_prediction=nba`, `split_criterion=info_gain`, `splitter=gaussian` | 99.03% | 98.89% | 166.85 µs |
| #2 | `grace_period=200`, `delta=0.001`, `tau=0.05`, `prep_scaler=standard`, `leaf_prediction=nba`, `split_criterion=info_gain`, `splitter=gaussian` | 98.58% | 98.37% | 163.79 µs |
| #3 | `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `leaf_prediction=nba`, `split_criterion=info_gain`, `splitter=gaussian` | 98.27% | 98.01% | 64.23 µs |

#### Sklearn

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=None`, `criterion=entropy` | 98.58% | 98.37% | 1.36 µs |
| #2 | `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=log2`, `prep_scaler=None`, `criterion=gini` | 98.32% | 98.07% | 3.81 µs |
| #3 | `max_depth=None`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=gini` | 98.13% | 97.85% | 4.21 µs |


---

## Dataset: Electricity (Classification)

- **Descrição:** Electricity / Elec2 (Real, 8 continuous, 2 classes)
- **Total de Experimentos (Trials):** 73

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.001`, `tau=0.01`, `tier=4`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 63.62% | 63.57% | 26.1 KB | 0.32 µs |
| #2 | `grace_period=100`, `delta=0.01`, `tau=0.1`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 63.21% | 62.03% | 19.6 KB | 0.17 µs |
| #3 | `grace_period=50`, `delta=0.0001`, `tau=0.01`, `tier=4`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 61.67% | 61.43% | 26.1 KB | 0.96 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.01`, `tau=0.1`, `tier=8`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 64.21% | 63.79% | 38.8 KB | 0.18 µs |
| #2 | `grace_period=50`, `delta=0.0001`, `tau=0.1`, `tier=8`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 64.09% | 63.31% | 38.8 KB | 0.25 µs |
| #3 | `grace_period=200`, `delta=1e-07`, `tau=0.15`, `tier=8`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 64.23% | 63.28% | 38.8 KB | 0.34 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=0.01`, `tau=0.15`, `tier=16`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 63.51% | 62.62% | 83.7 KB | 0.49 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=200`, `delta=1e-07`, `tau=0.05`, `tier=64`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 64.31% | 63.44% | 307.5 KB | 0.38 µs |
| #2 | `grace_period=50`, `delta=0.01`, `tau=0.05`, `tier=64`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 54.30% | 54.03% | 307.5 KB | 0.37 µs |

#### C11 libht (Tier 128)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=0.0001`, `tau=0.01`, `tier=128`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 60.34% | 59.76% | 614.6 KB | 0.29 µs |
| #2 | `grace_period=400`, `delta=0.001`, `tau=0.15`, `tier=128`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 52.21% | 49.43% | 621.2 KB | 0.39 µs |

#### River

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `leaf_prediction=nba`, `split_criterion=gini`, `splitter=histogram` | 63.93% | 63.76% | 708.11 µs |
| #2 | `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `leaf_prediction=mc`, `split_criterion=gini`, `splitter=gaussian` | 64.18% | 63.65% | 213.60 µs |
| #3 | `grace_period=200`, `delta=0.0001`, `tau=0.05`, `prep_scaler=standard`, `leaf_prediction=mc`, `split_criterion=info_gain`, `splitter=gaussian` | 64.34% | 63.58% | 164.58 µs |

#### Sklearn

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=4`, `min_samples_split=50`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=gini` | 63.83% | 63.52% | 4.44 µs |
| #2 | `max_depth=6`, `min_samples_split=50`, `min_samples_leaf=100`, `max_features=None`, `prep_scaler=None`, `criterion=entropy` | 63.95% | 63.38% | 7.04 µs |
| #3 | `max_depth=4`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=None`, `criterion=entropy` | 64.26% | 63.34% | 5.68 µs |


---

## Dataset: Forest Covertype (Classification)

- **Descrição:** Forest Covertype (Real, 54 features, 7 classes)
- **Total de Experimentos (Trials):** 69

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.0001`, `tau=0.01`, `tier=4`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 52.56% | 51.23% | 96.3 KB | 7.11 µs |
| #2 | `grace_period=100`, `delta=0.01`, `tau=0.1`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 53.73% | 44.51% | 70.1 KB | 0.84 µs |
| #3 | `grace_period=400`, `delta=0.01`, `tau=0.01`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 22.99% | 5.34% | 70.1 KB | 0.50 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.01`, `tau=0.1`, `tier=8`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 53.00% | 43.97% | 139.0 KB | 1.92 µs |
| #2 | `grace_period=400`, `delta=0.001`, `tau=0.15`, `tier=8`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 39.67% | 36.71% | 165.2 KB | 8.04 µs |
| #3 | `grace_period=200`, `delta=1e-07`, `tau=0.15`, `tier=8`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 42.84% | 30.20% | 139.0 KB | 1.12 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=0.01`, `tau=0.15`, `tier=16`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 66.55% | 61.75% | 303.1 KB | 4.17 µs |

#### River

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `leaf_prediction=mc`, `split_criterion=gini`, `splitter=gaussian` | 62.95% | 59.30% | 1161.92 µs |
| #2 | `grace_period=200`, `delta=0.001`, `tau=0.05`, `prep_scaler=standard`, `leaf_prediction=mc`, `split_criterion=gini`, `splitter=histogram` | 60.36% | 54.49% | 2248.05 µs |
| #3 | `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=None`, `leaf_prediction=nb`, `split_criterion=gini`, `splitter=gaussian` | 56.47% | 52.32% | 1165.39 µs |

#### Sklearn

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=None`, `criterion=entropy` | 82.23% | 80.25% | 14.69 µs |
| #2 | `max_depth=None`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=gini` | 78.76% | 76.71% | 23.24 µs |
| #3 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=100`, `max_features=None`, `prep_scaler=standard`, `criterion=gini` | 74.35% | 71.87% | 15.30 µs |


---

## Dataset: Airlines (Classification)

- **Descrição:** Airlines Delay (Real, 7 features, 2 classes)
- **Total de Experimentos (Trials):** 73

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.0001`, `tau=0.01`, `tier=4`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 77.32% | 51.85% | 26.1 KB | 1.05 µs |
| #2 | `grace_period=100`, `delta=0.01`, `tau=0.1`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 76.25% | 48.99% | 19.6 KB | 0.12 µs |
| #3 | `grace_period=400`, `delta=0.01`, `tau=0.01`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 76.39% | 43.31% | 19.6 KB | 0.10 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=200`, `delta=1e-07`, `tau=0.15`, `tier=8`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 77.17% | 57.57% | 38.8 KB | 0.55 µs |
| #2 | `grace_period=400`, `delta=0.001`, `tau=0.15`, `tier=8`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 74.69% | 56.79% | 45.3 KB | 1.41 µs |
| #3 | `grace_period=50`, `delta=0.01`, `tau=0.1`, `tier=8`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 76.56% | 52.64% | 38.8 KB | 0.18 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=0.01`, `tau=0.15`, `tier=16`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 75.85% | 55.72% | 83.7 KB | 0.69 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.01`, `tau=0.05`, `tier=64`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 75.15% | 56.24% | 307.5 KB | 0.65 µs |
| #2 | `grace_period=200`, `delta=1e-07`, `tau=0.05`, `tier=64`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 76.39% | 43.31% | 307.5 KB | 0.34 µs |

#### C11 libht (Tier 128)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=400`, `delta=0.001`, `tau=0.15`, `tier=128`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 75.00% | 55.72% | 621.2 KB | 0.40 µs |
| #2 | `grace_period=100`, `delta=0.0001`, `tau=0.01`, `tier=128`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 77.29% | 51.71% | 614.6 KB | 0.39 µs |

#### River

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `leaf_prediction=nba`, `split_criterion=info_gain`, `splitter=histogram` | 75.25% | 57.62% | 498.80 µs |
| #2 | `grace_period=200`, `delta=0.001`, `tau=0.05`, `prep_scaler=standard`, `leaf_prediction=mc`, `split_criterion=gini`, `splitter=histogram` | 77.20% | 56.86% | 389.11 µs |
| #3 | `grace_period=100`, `delta=0.001`, `tau=0.15`, `prep_scaler=None`, `leaf_prediction=nb`, `split_criterion=gini`, `splitter=gaussian` | 73.51% | 56.67% | 194.69 µs |

#### Sklearn

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=8`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=entropy` | 76.84% | 57.50% | 6.41 µs |
| #2 | `max_depth=6`, `min_samples_split=50`, `min_samples_leaf=100`, `max_features=None`, `prep_scaler=None`, `criterion=entropy` | 77.07% | 57.20% | 4.19 µs |
| #3 | `max_depth=8`, `min_samples_split=500`, `min_samples_leaf=20`, `max_features=log2`, `prep_scaler=None`, `criterion=gini` | 76.93% | 57.15% | 1.42 µs |


---

## Dataset: Friedman #1 (Regression)

- **Descrição:** Friedman #1 (Synthetic, 10 continuous, nonlinear target)
- **Total de Experimentos (Trials):** 71

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=400`, `delta=0.01`, `tau=0.15`, `tier=4`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False` | 2.4721 | 0.6144 | 18.1 KB | 0.79 µs |
| #2 | `grace_period=200`, `delta=1e-07`, `tau=0.15`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=False` | 2.5018 | 0.5993 | 11.6 KB | 0.25 µs |
| #3 | `grace_period=400`, `delta=0.01`, `tau=0.01`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=True` | 2.5544 | 0.5922 | 11.6 KB | 0.58 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=0.0001`, `tau=0.1`, `tier=8`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 2.3584 | 0.6470 | 77.3 KB | 0.88 µs |
| #2 | `grace_period=100`, `delta=0.0001`, `tau=0.15`, `tier=8`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 2.4341 | 0.6202 | 38.8 KB | 0.28 µs |
| #3 | `grace_period=50`, `delta=0.01`, `tau=0.1`, `tier=8`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 2.5484 | 0.5854 | 70.8 KB | 0.35 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=1e-07`, `tau=0.05`, `tier=16`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=True` | 2.2733 | 0.6715 | 83.7 KB | 0.59 µs |
| #2 | `grace_period=100`, `delta=1e-07`, `tau=0.15`, `tier=16`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=True` | 2.2921 | 0.6588 | 45.2 KB | 0.72 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=200`, `delta=1e-07`, `tau=0.05`, `tier=64`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=True` | 2.2189 | 0.6889 | 179.5 KB | 0.66 µs |

#### River

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `leaf_prediction=adaptive`, `splitter=ebst` | 1.5319 | 0.8458 | 2297.69 µs |
| #2 | `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `leaf_prediction=model`, `splitter=ebst` | 1.5302 | 0.8456 | 2340.25 µs |
| #3 | `grace_period=50`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `leaf_prediction=adaptive`, `splitter=tebst` | 1.5405 | 0.8421 | 923.27 µs |

#### Sklearn

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=None`, `criterion=squared_error` | 1.5741 | 0.8403 | 5.40 µs |
| #2 | `max_depth=None`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=absolute_error` | 1.7234 | 0.8096 | 30.84 µs |
| #3 | `max_depth=8`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=squared_error` | 1.8119 | 0.7923 | 7.91 µs |


---

## Dataset: Bikes (Regression)

- **Descrição:** Bikes (Real, 12 features, demand count)
- **Total de Experimentos (Trials):** 71

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=200`, `delta=1e-07`, `tau=0.15`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=False` | 45.5890 | 0.3676 | 11.6 KB | 0.19 µs |
| #2 | `grace_period=50`, `delta=1e-07`, `tau=0.1`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=False` | 53.4181 | 0.0651 | 19.6 KB | 0.24 µs |
| #3 | `grace_period=400`, `delta=0.01`, `tau=0.15`, `tier=4`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False` | 52.0919 | -0.0297 | 18.1 KB | 0.58 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=0.0001`, `tau=0.1`, `tier=8`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 41.9636 | 0.3916 | 77.3 KB | 1.01 µs |
| #2 | `grace_period=100`, `delta=0.0001`, `tau=0.15`, `tier=8`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 40.9528 | 0.3605 | 38.8 KB | 0.45 µs |
| #3 | `grace_period=50`, `delta=0.01`, `tau=0.1`, `tier=8`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 47.7232 | 0.0354 | 70.8 KB | 0.53 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=1e-07`, `tau=0.15`, `tier=16`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=True` | 35.3454 | 0.4697 | 45.2 KB | 1.06 µs |
| #2 | `grace_period=100`, `delta=1e-07`, `tau=0.05`, `tier=16`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=True` | 41.9876 | 0.2125 | 83.7 KB | 0.71 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=200`, `delta=1e-07`, `tau=0.05`, `tier=64`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=True` | 46.5301 | 0.0829 | 179.5 KB | 0.73 µs |

#### River

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=200`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `leaf_prediction=mean`, `splitter=ebst` | 24.7664 | 0.7080 | 1887.21 µs |
| #2 | `grace_period=50`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `leaf_prediction=adaptive`, `splitter=tebst` | 27.6792 | 0.6361 | 683.80 µs |
| #3 | `grace_period=100`, `delta=0.001`, `tau=0.15`, `prep_scaler=None`, `leaf_prediction=model`, `splitter=ebst` | 25.9531 | 0.6302 | 580.61 µs |

#### Sklearn

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=None`, `criterion=squared_error` | 12.2486 | 0.9325 | 5.30 µs |
| #2 | `max_depth=8`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=squared_error` | 13.9591 | 0.9132 | 6.04 µs |
| #3 | `max_depth=None`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=absolute_error` | 12.9996 | 0.9110 | 25.41 µs |


---

## Dataset: Appliances Energy (Regression)

- **Descrição:** Appliances Energy (Real, 28 features, building Wh)
- **Total de Experimentos (Trials):** 71

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=400`, `delta=0.01`, `tau=0.01`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=True` | 34.5141 | 0.6556 | 46.1 KB | 0.63 µs |
| #2 | `grace_period=50`, `delta=1e-07`, `tau=0.1`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=False` | 35.3776 | 0.6380 | 70.1 KB | 0.47 µs |
| #3 | `grace_period=400`, `delta=0.01`, `tau=0.15`, `tier=4`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False` | 40.3720 | 0.5206 | 72.3 KB | 0.82 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=0.0001`, `tau=0.1`, `tier=8`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 36.2987 | 0.6143 | 293.2 KB | 1.14 µs |
| #2 | `grace_period=50`, `delta=0.01`, `tau=0.1`, `tier=8`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 37.2893 | 0.5777 | 267.0 KB | 0.69 µs |
| #3 | `grace_period=100`, `delta=0.0001`, `tau=0.15`, `tier=8`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 122725433804880.9844 | -32867301187577822259970048.0000 | 139.0 KB | 0.58 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=1e-07`, `tau=0.15`, `tier=16`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=True` | 29.1035 | 0.7521 | 180.9 KB | 1.40 µs |
| #2 | `grace_period=100`, `delta=1e-07`, `tau=0.05`, `tier=16`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=True` | 30.7646 | 0.7281 | 303.1 KB | 1.59 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=200`, `delta=1e-07`, `tau=0.05`, `tier=64`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=True` | 27.2551 | 0.7867 | 720.3 KB | 1.22 µs |

#### River

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `leaf_prediction=mean`, `splitter=tebst` | 26.2688 | 0.8039 | 2418.59 µs |
| #2 | `grace_period=100`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `leaf_prediction=adaptive`, `splitter=tebst` | 26.3976 | 0.8032 | 2257.49 µs |
| #3 | `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `leaf_prediction=adaptive`, `splitter=ebst` | 26.3170 | 0.8032 | 4914.85 µs |

#### Sklearn

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=8`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=squared_error` | 25.0968 | 0.8242 | 19.20 µs |
| #2 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=100`, `max_features=None`, `prep_scaler=standard`, `criterion=absolute_error` | 25.2563 | 0.8205 | 72.23 µs |
| #3 | `max_depth=None`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=absolute_error` | 25.8709 | 0.8128 | 69.95 µs |


---

## Dataset: NYC Taxi (Regression)

- **Descrição:** NYC Taxi (Real, 7 features, trip duration seconds)
- **Total de Experimentos (Trials):** 71

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=400`, `delta=0.01`, `tau=0.01`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=True` | 351.2015 | 0.1344 | 11.6 KB | 0.52 µs |
| #2 | `grace_period=400`, `delta=0.01`, `tau=0.15`, `tier=4`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False` | 352.9003 | 0.1214 | 18.1 KB | 0.84 µs |
| #3 | `grace_period=50`, `delta=1e-07`, `tau=0.1`, `tier=4`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=False` | 355.0487 | 0.1181 | 19.6 KB | 0.21 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.01`, `tau=0.1`, `tier=8`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 343.2246 | 0.1756 | 70.8 KB | 0.39 µs |
| #2 | `grace_period=100`, `delta=0.0001`, `tau=0.1`, `tier=8`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 355.3801 | 0.1120 | 77.3 KB | 1.02 µs |
| #3 | `grace_period=100`, `delta=0.01`, `tau=0.01`, `tier=8`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=False` | 19872461804744.0469 | -20080218810726825852928.0000 | 22.8 KB | 0.23 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=1e-07`, `tau=0.15`, `tier=16`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=True` | 341.2690 | 0.1948 | 45.2 KB | 1.59 µs |
| #2 | `grace_period=100`, `delta=1e-07`, `tau=0.05`, `tier=16`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=True` | 346.9789 | 0.1490 | 83.7 KB | 1.18 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `grace_period=200`, `delta=1e-07`, `tau=0.05`, `tier=64`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=True` | 325.6622 | 0.2505 | 179.5 KB | 1.15 µs |

#### River

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `leaf_prediction=model`, `splitter=ebst` | 309.0727 | 0.3240 | 1251.16 µs |
| #2 | `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `leaf_prediction=adaptive`, `splitter=ebst` | 308.3840 | 0.3227 | 1230.30 µs |
| #3 | `grace_period=100`, `delta=0.001`, `tau=0.01`, `prep_scaler=None`, `leaf_prediction=mean`, `splitter=ebst` | 315.2634 | 0.3070 | 765.52 µs |

#### Sklearn

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=None`, `criterion=squared_error` | 305.0938 | 0.3461 | 5.16 µs |
| #2 | `max_depth=None`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=absolute_error` | 300.0886 | 0.3335 | 34.87 µs |
| #3 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=100`, `max_features=None`, `prep_scaler=standard`, `criterion=absolute_error` | 313.8903 | 0.2787 | 20.00 µs |


---

## 2. Síntese e Diretrizes de Engenharia

1. **Análise por Tier:** O ranking estratificado revela que configurações ideais para 256 folhas frequentemente diferem daqueles necessários para evitar subajuste em 4 ou 8 folhas.
2. **Eficiência:** O uso de latência como desempate garante que configurações computacionalmente custosas que entreguem ganhos marginais de acurácia sejam penalizadas.
