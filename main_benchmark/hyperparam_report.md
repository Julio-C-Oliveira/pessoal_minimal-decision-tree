# Relatório Consolidado de Otimização e Sensibilidade de Hiperparâmetros

Suíte automatizada de exploração de hiperparâmetros comparando **C11 libht**, **River** e **Scikit-Learn** em 8 datasets emblemáticos.

---

## Dataset: SEA (Classification)

- **Descrição:** SEA (Synthetic, 3 continuous, 2 classes)
- **Total de Experimentos (Trials):** 468

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 97.30% | 96.88% | 26.1 KB | 1.32 µs |
| #2 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 97.10% | 96.62% | 19.6 KB | 0.66 µs |
| #3 | `tier=4`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 96.67% | 96.10% | 19.6 KB | 0.26 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.02% | 98.87% | 38.8 KB | 0.45 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 98.57% | 98.35% | 38.8 KB | 0.48 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 98.42% | 98.17% | 45.3 KB | 0.97 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 77.2 KB | 0.99 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 98.92% | 98.75% | 83.7 KB | 0.55 µs |
| #3 | `tier=16`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.25% | 97.99% | 77.2 KB | 0.81 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 307.5 KB | 0.60 µs |
| #2 | `tier=64`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 99.15% | 99.02% | 314.1 KB | 0.59 µs |
| #3 | `tier=64`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.38% | 98.14% | 314.1 KB | 0.84 µs |

#### C11 libht (Tier 128)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 614.6 KB | 0.72 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 98.30% | 98.03% | 621.2 KB | 1.15 µs |
| #3 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.20% | 97.94% | 614.6 KB | 1.18 µs |

#### C11 libht (Tier 256)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 979.4 KB | 1.17 µs |
| #2 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 98.98% | 98.83% | 985.9 KB | 0.50 µs |
| #3 | `tier=256`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 98.57% | 98.36% | 985.9 KB | 2.00 µs |

#### River

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `leaf_prediction=nba`, `split_criterion=info_gain`, `splitter=gaussian` | 99.03% | 98.89% | 173.79 µs |
| #2 | `grace_period=100`, `delta=0.001`, `tau=0.15`, `prep_scaler=None`, `leaf_prediction=nba`, `split_criterion=info_gain`, `splitter=gaussian` | 98.95% | 98.80% | 185.40 µs |
| #3 | `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `leaf_prediction=nba`, `split_criterion=info_gain`, `splitter=gaussian` | 98.88% | 98.71% | 208.29 µs |

#### Sklearn

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=None`, `criterion=entropy` | 99.18% | 99.06% | 10.57 µs |
| #2 | `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=standard`, `criterion=gini` | 99.18% | 99.06% | 3.07 µs |
| #3 | `max_depth=16`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=None`, `criterion=gini` | 99.18% | 99.06% | 5.77 µs |


---

## Dataset: Electricity (Classification)

- **Descrição:** Electricity / Elec2 (Real, 8 continuous, 2 classes)
- **Total de Experimentos (Trials):** 468

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.18% | 63.92% | 26.1 KB | 2.71 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.33% | 63.77% | 26.1 KB | 2.31 µs |
| #3 | `tier=4`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.32% | 63.76% | 26.1 KB | 2.00 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.37% | 63.96% | 38.8 KB | 0.42 µs |
| #2 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.29% | 63.94% | 45.3 KB | 1.09 µs |
| #3 | `tier=8`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.44% | 63.75% | 38.8 KB | 0.49 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.26% | 63.86% | 77.2 KB | 1.92 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 83.7 KB | 2.49 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 63.77% | 63.67% | 83.7 KB | 0.97 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 314.1 KB | 1.96 µs |
| #2 | `tier=64`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 63.78% | 63.69% | 307.5 KB | 0.29 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.45% | 63.61% | 307.5 KB | 0.29 µs |

#### C11 libht (Tier 128)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 621.2 KB | 1.73 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.45% | 63.61% | 614.6 KB | 0.17 µs |
| #3 | `tier=128`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.30% | 63.55% | 621.2 KB | 1.05 µs |

#### C11 libht (Tier 256)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.30% | 63.81% | 985.9 KB | 2.72 µs |
| #2 | `tier=256`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 985.9 KB | 2.21 µs |
| #3 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.32% | 63.76% | 985.9 KB | 1.07 µs |

#### River

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=200`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `leaf_prediction=mc`, `split_criterion=gini`, `splitter=histogram` | 63.93% | 63.76% | 492.77 µs |
| #2 | `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `leaf_prediction=mc`, `split_criterion=gini`, `splitter=histogram` | 63.93% | 63.76% | 618.12 µs |
| #3 | `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `leaf_prediction=nba`, `split_criterion=gini`, `splitter=histogram` | 63.93% | 63.76% | 917.67 µs |

#### Sklearn

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=6`, `min_samples_split=500`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=entropy` | 63.90% | 63.52% | 9.92 µs |
| #2 | `max_depth=4`, `min_samples_split=50`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=None`, `criterion=gini` | 63.83% | 63.52% | 4.81 µs |
| #3 | `max_depth=4`, `min_samples_split=50`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=gini` | 63.83% | 63.52% | 4.92 µs |


---

## Dataset: Forest Covertype (Classification)

- **Descrição:** Forest Covertype (Real, 54 features, 7 classes)
- **Total de Experimentos (Trials):** 311

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.19% | 62.04% | 70.1 KB | 3.48 µs |
| #2 | `tier=4`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 60.83% | 58.52% | 96.3 KB | 2.27 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 62.95% | 58.09% | 70.1 KB | 0.56 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 67.28% | 65.21% | 139.0 KB | 4.19 µs |
| #2 | `tier=8`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.65% | 60.36% | 139.0 KB | 0.67 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 63.60% | 60.33% | 139.0 KB | 2.60 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 68.49% | 66.49% | 276.9 KB | 5.45 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 66.31% | 62.59% | 303.1 KB | 6.49 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 65.12% | 61.37% | 276.9 KB | 0.85 µs |

#### C11 libht (Tier 40)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=40`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 68.92% | 66.86% | 713.8 KB | 6.29 µs |
| #2 | `tier=40`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 66.96% | 63.30% | 740.0 KB | 10.95 µs |
| #3 | `tier=40`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 65.12% | 61.37% | 713.8 KB | 0.86 µs |

#### River

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `leaf_prediction=nba`, `split_criterion=gini`, `splitter=gaussian` | 66.57% | 65.11% | 2303.88 µs |
| #2 | `grace_period=100`, `delta=0.0001`, `tau=0.15`, `prep_scaler=None`, `leaf_prediction=mc`, `split_criterion=gini`, `splitter=gaussian` | 65.95% | 62.52% | 839.46 µs |
| #3 | `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=None`, `leaf_prediction=nba`, `split_criterion=gini`, `splitter=gaussian` | 66.15% | 62.06% | 2118.78 µs |

#### Sklearn

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=standard`, `criterion=gini` | 87.40% | 85.81% | 32.49 µs |
| #2 | `max_depth=16`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=None`, `criterion=gini` | 87.19% | 85.62% | 14.85 µs |
| #3 | `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=None`, `criterion=entropy` | 86.43% | 84.88% | 44.92 µs |


---

## Dataset: Airlines (Classification)

- **Descrição:** Airlines Delay (Real, 7 features, 2 classes)
- **Total de Experimentos (Trials):** 468

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 77.19% | 56.64% | 19.6 KB | 0.40 µs |
| #2 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 77.17% | 54.91% | 26.1 KB | 1.85 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 77.03% | 54.30% | 19.6 KB | 0.23 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 76.09% | 58.33% | 45.3 KB | 2.15 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.97% | 56.80% | 38.8 KB | 0.46 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 75.92% | 55.45% | 45.3 KB | 1.33 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 72.67% | 57.12% | 83.7 KB | 2.02 µs |
| #2 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 72.55% | 57.05% | 83.7 KB | 2.06 µs |
| #3 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.87% | 56.72% | 77.2 KB | 1.52 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 74.81% | 57.20% | 314.1 KB | 2.85 µs |
| #2 | `tier=64`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 71.28% | 56.88% | 314.1 KB | 2.50 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 76.11% | 56.69% | 307.5 KB | 1.91 µs |

#### C11 libht (Tier 128)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=200`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 76.99% | 56.90% | 614.6 KB | 1.07 µs |
| #2 | `tier=128`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 73.31% | 56.19% | 614.6 KB | 1.71 µs |
| #3 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 70.45% | 55.99% | 614.6 KB | 2.13 µs |

#### C11 libht (Tier 256)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 75.76% | 56.86% | 979.4 KB | 1.14 µs |
| #2 | `tier=256`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.69% | 55.59% | 979.4 KB | 0.37 µs |
| #3 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 76.83% | 55.47% | 985.9 KB | 0.84 µs |

#### River

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=1e-07`, `tau=0.05`, `prep_scaler=None`, `leaf_prediction=nb`, `split_criterion=gini`, `splitter=histogram` | 68.37% | 59.65% | 605.13 µs |
| #2 | `grace_period=50`, `delta=0.0001`, `tau=0.05`, `prep_scaler=None`, `leaf_prediction=mc`, `split_criterion=gini`, `splitter=gaussian` | 75.75% | 59.47% | 140.81 µs |
| #3 | `grace_period=200`, `delta=0.01`, `tau=0.05`, `prep_scaler=standard`, `leaf_prediction=nba`, `split_criterion=info_gain`, `splitter=histogram` | 76.87% | 58.90% | 644.12 µs |

#### Sklearn

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=20`, `max_features=None`, `prep_scaler=standard`, `criterion=entropy` | 73.31% | 57.59% | 15.31 µs |
| #2 | `max_depth=None`, `min_samples_split=50`, `min_samples_leaf=5`, `max_features=log2`, `prep_scaler=None`, `criterion=gini` | 73.68% | 57.56% | 3.80 µs |
| #3 | `max_depth=None`, `min_samples_split=50`, `min_samples_leaf=5`, `max_features=sqrt`, `prep_scaler=standard`, `criterion=gini` | 73.68% | 57.56% | 4.11 µs |


---

## Dataset: Friedman #1 (Regression)

- **Descrição:** Friedman #1 (Synthetic, 10 continuous, nonlinear target)
- **Total de Experimentos (Trials):** 461

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.3536 | 0.6477 | 26.1 KB | 0.57 µs |
| #2 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 2.3876 | 0.6322 | 26.1 KB | 1.31 µs |
| #3 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.4125 | 0.6280 | 19.6 KB | 1.20 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.1537 | 0.7013 | 45.3 KB | 1.85 µs |
| #2 | `tier=8`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 2.1618 | 0.6909 | 45.3 KB | 0.99 µs |
| #3 | `tier=8`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 2.2440 | 0.6739 | 38.8 KB | 0.92 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.9341 | 0.7506 | 147.7 KB | 4.75 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.9860 | 0.7423 | 83.7 KB | 1.51 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 1.9556 | 0.7413 | 83.7 KB | 2.10 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 314.1 KB | 1.25 µs |
| #2 | `tier=64`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 1.6348 | 0.8288 | 563.5 KB | 6.41 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.6547 | 0.8205 | 570.1 KB | 6.90 µs |

#### C11 libht (Tier 128)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 1.5215 | 0.8462 | 621.2 KB | 3.30 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True` | 1.5600 | 0.8432 | 621.2 KB | 3.53 µs |
| #3 | `tier=128`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 621.2 KB | 1.69 µs |

#### C11 libht (Tier 256)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 1.5513 | 0.8444 | 723.4 KB | 2.51 µs |
| #2 | `tier=256`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 985.9 KB | 1.93 µs |
| #3 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.5816 | 0.8361 | 922.0 KB | 5.32 µs |

#### River

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=standard`, `leaf_prediction=model`, `splitter=ebst` | 1.4331 | 0.8647 | 2196.74 µs |
| #2 | `grace_period=200`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `leaf_prediction=model`, `splitter=tebst` | 1.4995 | 0.8565 | 597.51 µs |
| #3 | `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `leaf_prediction=adaptive`, `splitter=tebst` | 1.5039 | 0.8547 | 506.92 µs |

#### Sklearn

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=5`, `max_features=None`, `prep_scaler=None`, `criterion=squared_error` | 1.5375 | 0.8464 | 19.05 µs |
| #2 | `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=20`, `max_features=None`, `prep_scaler=None`, `criterion=squared_error` | 1.5657 | 0.8423 | 12.51 µs |
| #3 | `max_depth=16`, `min_samples_split=10`, `min_samples_leaf=20`, `max_features=None`, `prep_scaler=None`, `criterion=squared_error` | 1.5655 | 0.8423 | 14.53 µs |


---

## Dataset: Bikes (Regression)

- **Descrição:** Bikes (Real, 12 features, demand count)
- **Total de Experimentos (Trials):** 461

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=False` | 44.6479 | 0.3852 | 26.1 KB | 1.53 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 44.5617 | 0.3847 | 18.1 KB | 0.96 µs |
| #3 | `tier=4`, `grace_period=200`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 45.2019 | 0.3763 | 26.1 KB | 0.67 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 40.4648 | 0.4080 | 77.3 KB | 0.56 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True` | 37.9817 | 0.3948 | 38.8 KB | 1.13 µs |
| #3 | `tier=8`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 39.6668 | 0.3932 | 22.8 KB | 0.96 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 34.2102 | 0.6398 | 83.7 KB | 1.96 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 29.3316 | 0.6199 | 45.2 KB | 0.67 µs |
| #3 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 30.3082 | 0.6119 | 141.2 KB | 4.60 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2040 | 0.7062 | 307.5 KB | 4.20 µs |
| #2 | `tier=64`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.0234 | 0.6731 | 179.5 KB | 1.07 µs |
| #3 | `tier=64`, `grace_period=100`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 30.7080 | 0.6126 | 307.5 KB | 3.55 µs |

#### C11 libht (Tier 128)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2040 | 0.7062 | 614.6 KB | 4.12 µs |
| #2 | `tier=128`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.0234 | 0.6731 | 358.6 KB | 1.05 µs |
| #3 | `tier=128`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 27.7748 | 0.6509 | 904.4 KB | 2.62 µs |

#### C11 libht (Tier 256)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=True` | 18.8594 | 0.8502 | 979.4 KB | 1.67 µs |
| #2 | `tier=256`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=False` | 21.6795 | 0.7746 | 716.9 KB | 1.18 µs |
| #3 | `tier=256`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=True` | 24.2433 | 0.7393 | 915.5 KB | 2.33 µs |

#### River

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=None`, `leaf_prediction=mean`, `splitter=ebst` | 18.9138 | 0.8228 | 1629.05 µs |
| #2 | `grace_period=100`, `delta=0.0001`, `tau=0.15`, `prep_scaler=None`, `leaf_prediction=mean`, `splitter=ebst` | 19.9020 | 0.8002 | 876.41 µs |
| #3 | `grace_period=100`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `leaf_prediction=mean`, `splitter=ebst` | 22.6233 | 0.7888 | 1285.42 µs |

#### Sklearn

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=16`, `min_samples_split=10`, `min_samples_leaf=20`, `max_features=None`, `prep_scaler=None`, `criterion=squared_error` | 11.8925 | 0.9360 | 8.72 µs |
| #2 | `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=20`, `max_features=None`, `prep_scaler=None`, `criterion=squared_error` | 11.8925 | 0.9360 | 9.81 µs |
| #3 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=20`, `max_features=None`, `prep_scaler=standard`, `criterion=squared_error` | 12.0834 | 0.9338 | 8.23 µs |


---

## Dataset: Appliances Energy (Regression)

- **Descrição:** Appliances Energy (Real, 28 features, building Wh)
- **Total de Experimentos (Trials):** 314

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 32.9710 | 0.6832 | 70.1 KB | 1.75 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 34.0290 | 0.6650 | 46.1 KB | 2.03 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 34.4312 | 0.6594 | 134.1 KB | 4.16 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=0.0001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 29.7630 | 0.7401 | 91.0 KB | 2.76 µs |
| #2 | `tier=8`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 30.5630 | 0.7312 | 139.0 KB | 3.23 µs |
| #3 | `tier=8`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 30.5785 | 0.7311 | 139.0 KB | 2.29 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 27.8188 | 0.7786 | 532.9 KB | 10.09 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 27.9501 | 0.7770 | 276.9 KB | 4.80 µs |
| #3 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=False` | 28.4250 | 0.7710 | 559.1 KB | 5.89 µs |

#### C11 libht (Tier 40)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=40`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2261 | 0.8055 | 713.8 KB | 6.37 µs |
| #2 | `tier=40`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.6769 | 0.7963 | 473.8 KB | 1.17 µs |
| #3 | `tier=40`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 27.6225 | 0.7817 | 832.1 KB | 12.77 µs |

#### River

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=100`, `delta=0.001`, `tau=0.15`, `prep_scaler=None`, `leaf_prediction=mean`, `splitter=tebst` | 26.1249 | 0.8059 | 3106.15 µs |
| #2 | `grace_period=100`, `delta=1e-07`, `tau=0.01`, `prep_scaler=None`, `leaf_prediction=mean`, `splitter=ebst` | 26.2581 | 0.8040 | 7279.64 µs |
| #3 | `grace_period=100`, `delta=1e-07`, `tau=0.01`, `prep_scaler=None`, `leaf_prediction=adaptive`, `splitter=ebst` | 26.2581 | 0.8040 | 7459.21 µs |

#### Sklearn

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=8`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=squared_error` | 25.0968 | 0.8242 | 38.03 µs |
| #2 | `max_depth=12`, `min_samples_split=100`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=squared_error` | 25.0574 | 0.8239 | 30.16 µs |
| #3 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=50`, `max_features=None`, `prep_scaler=standard`, `criterion=squared_error` | 25.1394 | 0.8228 | 23.29 µs |


---

## Dataset: NYC Taxi (Regression)

- **Descrição:** NYC Taxi (Real, 7 features, trip duration seconds)
- **Total de Experimentos (Trials):** 461

### 1. Melhores Modelos por Tier (C11) e Competidores (Top 3)

#### C11 libht (Tier 4)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 348.9520 | 0.1579 | 35.6 KB | 0.68 µs |
| #2 | `tier=4`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 351.1585 | 0.1504 | 19.6 KB | 0.40 µs |
| #3 | `tier=4`, `grace_period=100`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 351.3464 | 0.1375 | 42.1 KB | 1.01 µs |

#### C11 libht (Tier 8)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=100`, `delta=0.0001`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 339.2442 | 0.2054 | 29.3 KB | 0.67 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 341.0358 | 0.1823 | 70.8 KB | 1.35 µs |
| #3 | `tier=8`, `grace_period=200`, `delta=0.0001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 339.2015 | 0.1761 | 22.8 KB | 1.06 µs |

#### C11 libht (Tier 16)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 317.5174 | 0.2909 | 83.7 KB | 1.89 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 319.2773 | 0.2730 | 141.2 KB | 3.58 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 323.0949 | 0.2643 | 83.7 KB | 1.27 µs |

#### C11 libht (Tier 64)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=400`, `delta=0.0001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 284.4319 | 0.4078 | 179.5 KB | 1.60 µs |
| #2 | `tier=64`, `grace_period=100`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 299.4966 | 0.3577 | 307.5 KB | 2.18 µs |
| #3 | `tier=64`, `grace_period=50`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 299.9648 | 0.3477 | 563.5 KB | 3.58 µs |

#### C11 libht (Tier 128)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 300.8008 | 0.3425 | 614.6 KB | 3.50 µs |
| #2 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=False` | 305.9346 | 0.3361 | 897.9 KB | 1.17 µs |
| #3 | `tier=128`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 310.3592 | 0.3236 | 621.2 KB | 1.94 µs |

#### C11 libht (Tier 256)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 277.1476 | 0.4417 | 922.0 KB | 3.98 µs |
| #2 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 286.9681 | 0.3946 | 723.4 KB | 1.31 µs |
| #3 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 302.8055 | 0.3517 | 723.4 KB | 1.18 µs |

#### River

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `leaf_prediction=adaptive`, `splitter=tebst` | 287.2391 | 0.4118 | 424.76 µs |
| #2 | `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `leaf_prediction=model`, `splitter=tebst` | 292.0904 | 0.3926 | 379.78 µs |
| #3 | `grace_period=100`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `leaf_prediction=model`, `splitter=ebst` | 300.9913 | 0.3616 | 1178.54 µs |

#### Sklearn

| Rank | Configuração | MAE | R² | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: |
| #1 | `max_depth=None`, `min_samples_split=10`, `min_samples_leaf=20`, `max_features=None`, `prep_scaler=None`, `criterion=squared_error` | 291.6292 | 0.3880 | 8.82 µs |
| #2 | `max_depth=16`, `min_samples_split=10`, `min_samples_leaf=20`, `max_features=None`, `prep_scaler=None`, `criterion=squared_error` | 300.2324 | 0.3643 | 14.59 µs |
| #3 | `max_depth=16`, `min_samples_split=50`, `min_samples_leaf=20`, `max_features=None`, `prep_scaler=standard`, `criterion=squared_error` | 301.0283 | 0.3622 | 13.98 µs |


---

## 2. Síntese e Diretrizes de Engenharia

1. **Análise por Tier:** O ranking estratificado revela que configurações ideais para 256 folhas frequentemente diferem daqueles necessários para evitar subajuste em 4 ou 8 folhas.
2. **Eficiência:** O uso de latência como desempate garante que configurações computacionalmente custosas que entreguem ganhos marginais de acurácia sejam penalizadas.
