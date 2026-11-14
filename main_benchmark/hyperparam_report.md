# Relatório Consolidado de Otimização e Sensibilidade de Hiperparâmetros

Suíte automatizada de exploração de hiperparâmetros comparando **Kansho**, **River** e **Scikit-Learn** em 8 datasets emblemáticos.

---

## Dataset: SEA (Classification)

- **Descrição:** SEA (Synthetic, 3 continuous, 2 classes)
- **Total de Experimentos (Trials):** 528

### 1. Melhores Modelos por Tier (Kansho) e Competidores (Top 3)

#### Kansho (Tier 1: 1 nó ativo, 1 utilizado, 16 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=1`, `grace_period=50`, `delta=0.001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.87% | 93.83% | 2.1 KB | 1.40 µs |
| #2 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.73% | 93.66% | 8.7 KB | 10.55 µs |
| #3 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 94.73% | 93.66% | 8.7 KB | 3.69 µs |

#### Kansho (Tier 2: 2 nós ativos, 1 utilizado, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=0.001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.87% | 93.83% | 2.7 KB | 7.42 µs |
| #2 | `tier=2`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.73% | 93.66% | 9.2 KB | 8.58 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 94.73% | 93.66% | 9.2 KB | 4.64 µs |

#### Kansho (Tier 4: 4 nós ativos, 21 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 97.30% | 96.88% | 10.3 KB | 4.80 µs |
| #2 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 97.10% | 96.62% | 3.8 KB | 1.62 µs |
| #3 | `tier=4`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 96.67% | 96.10% | 3.8 KB | 0.16 µs |

#### Kansho (Tier 8: 8 nós ativos, 43 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.02% | 98.87% | 6.7 KB | 0.28 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 98.57% | 98.35% | 6.7 KB | 3.98 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 98.42% | 98.17% | 13.3 KB | 0.99 µs |

#### Kansho (Tier 16: 16 nós ativos, 45 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 13.2 KB | 2.38 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 98.92% | 98.75% | 19.8 KB | 1.97 µs |
| #3 | `tier=16`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.25% | 97.99% | 13.2 KB | 2.36 µs |

#### Kansho (Tier 64: 64 nós ativos, 45 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 52.2 KB | 0.68 µs |
| #2 | `tier=64`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 99.15% | 99.02% | 58.8 KB | 1.94 µs |
| #3 | `tier=64`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.38% | 98.14% | 58.8 KB | 1.58 µs |

#### Kansho (Tier 128: 128 nós ativos, 45 utilizados, 2048 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 104.2 KB | 1.43 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 98.30% | 98.03% | 110.8 KB | 3.88 µs |
| #3 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.20% | 97.94% | 104.2 KB | 4.68 µs |

#### Kansho (Tier 256: 256 nós ativos, 45 utilizados, 4096 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 208.2 KB | 0.46 µs |
| #2 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 98.98% | 98.83% | 214.8 KB | 1.19 µs |
| #3 | `tier=256`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 98.57% | 98.36% | 214.8 KB | 0.60 µs |

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
- **Total de Experimentos (Trials):** 528

### 1. Melhores Modelos por Tier (Kansho) e Competidores (Top 3)

#### Kansho (Tier 1: 1 nó ativo, 3 utilizados, 16 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.33% | 63.77% | 8.7 KB | 2.86 µs |
| #2 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.32% | 63.76% | 8.7 KB | 2.14 µs |
| #3 | `tier=1`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 63.78% | 63.69% | 2.1 KB | 0.57 µs |

#### Kansho (Tier 2: 2 nós ativos, 11 utilizados, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.18% | 63.92% | 9.2 KB | 2.17 µs |
| #2 | `tier=2`, `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.00% | 63.82% | 2.7 KB | 0.75 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.33% | 63.77% | 9.2 KB | 3.62 µs |

#### Kansho (Tier 4: 4 nós ativos, 19 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.18% | 63.92% | 11.6 KB | 3.02 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.33% | 63.77% | 11.6 KB | 3.54 µs |
| #3 | `tier=4`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.32% | 63.76% | 11.6 KB | 3.94 µs |

#### Kansho (Tier 8: 8 nós ativos, 79 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.37% | 63.96% | 9.8 KB | 0.29 µs |
| #2 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.29% | 63.94% | 16.4 KB | 1.85 µs |
| #3 | `tier=8`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.44% | 63.75% | 9.8 KB | 0.54 µs |

#### Kansho (Tier 16: 16 nós ativos, 59 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.26% | 63.86% | 19.5 KB | 3.01 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 26.0 KB | 4.15 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 63.77% | 63.67% | 26.0 KB | 1.35 µs |

#### Kansho (Tier 64: 64 nós ativos, 41 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 83.8 KB | 3.93 µs |
| #2 | `tier=64`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 63.78% | 63.69% | 77.2 KB | 0.38 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.45% | 63.61% | 77.2 KB | 0.43 µs |

#### Kansho (Tier 128: 128 nós ativos, 41 utilizados, 2048 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 160.8 KB | 3.74 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.45% | 63.61% | 154.2 KB | 0.39 µs |
| #3 | `tier=128`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.30% | 63.55% | 160.8 KB | 2.07 µs |

#### Kansho (Tier 256: 256 nós ativos, 127 utilizados, 4096 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.30% | 63.81% | 314.8 KB | 2.62 µs |
| #2 | `tier=256`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 314.8 KB | 3.84 µs |
| #3 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.32% | 63.76% | 314.8 KB | 1.95 µs |

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
- **Total de Experimentos (Trials):** 371

### 1. Melhores Modelos por Tier (Kansho) e Competidores (Top 3)

#### Kansho (Tier 1: 1 nó ativo, 15 utilizados, 16 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=1`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 58.68% | 52.66% | 11.5 KB | 22.97 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 54.73% | 52.36% | 37.7 KB | 3.97 µs |
| #3 | `tier=1`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 54.71% | 52.02% | 11.5 KB | 0.97 µs |

#### Kansho (Tier 2: 2 nós ativos, 27 utilizados, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 61.43% | 58.51% | 18.9 KB | 8.20 µs |
| #2 | `tier=2`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 57.73% | 54.25% | 18.9 KB | 0.91 µs |
| #3 | `tier=2`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 56.64% | 53.98% | 45.1 KB | 7.42 µs |

#### Kansho (Tier 4: 4 nós ativos, 47 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.19% | 62.04% | 37.5 KB | 7.41 µs |
| #2 | `tier=4`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 60.83% | 58.52% | 63.7 KB | 3.55 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 62.95% | 58.09% | 37.5 KB | 0.99 µs |

#### Kansho (Tier 8: 8 nós ativos, 77 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 67.28% | 65.21% | 74.5 KB | 7.70 µs |
| #2 | `tier=8`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.65% | 60.36% | 74.5 KB | 1.29 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 63.60% | 60.33% | 74.5 KB | 5.34 µs |

#### Kansho (Tier 16: 16 nós ativos, 99 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 68.49% | 66.49% | 148.6 KB | 10.94 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 66.31% | 62.59% | 174.8 KB | 14.32 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 65.12% | 61.37% | 148.6 KB | 1.19 µs |

#### Kansho (Tier 40: 40 nós ativos, 113 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=40`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 68.92% | 66.86% | 390.5 KB | 10.37 µs |
| #2 | `tier=40`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 66.96% | 63.30% | 416.7 KB | 9.84 µs |
| #3 | `tier=40`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 65.12% | 61.37% | 390.5 KB | 0.95 µs |

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
- **Total de Experimentos (Trials):** 528

### 1. Melhores Modelos por Tier (Kansho) e Competidores (Top 3)

#### Kansho (Tier 1: 1 nó ativo, 1 utilizado, 16 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=1`, `grace_period=200`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.32% | 51.85% | 8.7 KB | 1.20 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.29% | 51.71% | 2.1 KB | 1.20 µs |
| #3 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 77.09% | 49.93% | 8.7 KB | 5.49 µs |

#### Kansho (Tier 2: 2 nós ativos, 7 utilizados, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 77.17% | 54.91% | 9.2 KB | 3.76 µs |
| #2 | `tier=2`, `grace_period=200`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.32% | 51.85% | 9.2 KB | 1.05 µs |
| #3 | `tier=2`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.29% | 51.71% | 2.7 KB | 1.53 µs |

#### Kansho (Tier 4: 4 nós ativos, 33 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 77.19% | 56.64% | 4.7 KB | 0.73 µs |
| #2 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 77.17% | 54.91% | 11.3 KB | 4.01 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 77.03% | 54.30% | 4.7 KB | 0.23 µs |

#### Kansho (Tier 8: 8 nós ativos, 39 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 76.09% | 58.33% | 15.8 KB | 3.16 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.97% | 56.80% | 9.2 KB | 1.11 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 75.92% | 55.45% | 15.8 KB | 2.02 µs |

#### Kansho (Tier 16: 16 nós ativos, 67 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 72.67% | 57.12% | 24.8 KB | 4.72 µs |
| #2 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 72.55% | 57.05% | 24.8 KB | 3.19 µs |
| #3 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.87% | 56.72% | 18.2 KB | 1.16 µs |

#### Kansho (Tier 64: 64 nós ativos, 43 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 74.81% | 57.20% | 78.8 KB | 3.76 µs |
| #2 | `tier=64`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 71.28% | 56.88% | 78.8 KB | 6.40 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 76.11% | 56.69% | 72.2 KB | 4.45 µs |

#### Kansho (Tier 128: 128 nós ativos, 25 utilizados, 2048 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=200`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 76.99% | 56.90% | 144.2 KB | 1.76 µs |
| #2 | `tier=128`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 73.31% | 56.19% | 144.2 KB | 4.00 µs |
| #3 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 70.45% | 55.99% | 144.2 KB | 4.09 µs |

#### Kansho (Tier 256: 256 nós ativos, 45 utilizados, 4096 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 75.76% | 56.86% | 288.2 KB | 1.71 µs |
| #2 | `tier=256`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.69% | 55.59% | 288.2 KB | 1.12 µs |
| #3 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 76.83% | 55.47% | 294.8 KB | 1.45 µs |

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
- **Total de Experimentos (Trials):** 521

### 1. Melhores Modelos por Tier (Kansho) e Competidores (Top 3)

#### Kansho (Tier 1: 1 nó ativo, 11 utilizados, 16 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 3.2307 | 0.3607 | 3.0 KB | 0.38 µs |
| #2 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 3.2352 | 0.3603 | 9.0 KB | 0.43 µs |
| #3 | `tier=1`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 3.2352 | 0.3568 | 11.6 KB | 0.63 µs |

#### Kansho (Tier 2: 2 nós ativos, 19 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 2.8418 | 0.5009 | 18.2 KB | 1.82 µs |
| #2 | `tier=2`, `grace_period=100`, `delta=0.0001`, `tau=0.15`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 2.8464 | 0.4917 | 18.2 KB | 3.09 µs |
| #3 | `tier=2`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 2.9126 | 0.4780 | 11.6 KB | 2.00 µs |

#### Kansho (Tier 4: 4 nós ativos, 35 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.3536 | 0.6477 | 19.4 KB | 1.90 µs |
| #2 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 2.3876 | 0.6322 | 19.4 KB | 1.14 µs |
| #3 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.4125 | 0.6280 | 12.9 KB | 2.35 µs |

#### Kansho (Tier 8: 8 nós ativos, 45 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.1537 | 0.7013 | 31.9 KB | 1.83 µs |
| #2 | `tier=8`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 2.1618 | 0.6909 | 31.9 KB | 2.14 µs |
| #3 | `tier=8`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 2.2440 | 0.6739 | 25.3 KB | 1.89 µs |

#### Kansho (Tier 16: 16 nós ativos, 63 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.9341 | 0.7506 | 96.8 KB | 5.99 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.9860 | 0.7423 | 56.8 KB | 4.09 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 1.9556 | 0.7413 | 56.8 KB | 2.53 µs |

#### Kansho (Tier 64: 64 nós ativos, 67 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 206.5 KB | 1.80 µs |
| #2 | `tier=64`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 1.6348 | 0.8288 | 359.9 KB | 13.46 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.6547 | 0.8205 | 366.5 KB | 8.91 µs |

#### Kansho (Tier 128: 128 nós ativos, 125 utilizados, 2048 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 1.5215 | 0.8462 | 406.0 KB | 2.42 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True` | 1.5600 | 0.8432 | 406.0 KB | 8.59 µs |
| #3 | `tier=128`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 406.0 KB | 4.06 µs |

#### Kansho (Tier 256: 256 nós ativos, 51 utilizados, 4096 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 1.5513 | 0.8444 | 485.0 KB | 3.21 µs |
| #2 | `tier=256`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 642.9 KB | 2.08 µs |
| #3 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.5816 | 0.8361 | 591.1 KB | 9.13 µs |

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
- **Total de Experimentos (Trials):** 521

### 1. Melhores Modelos por Tier (Kansho) e Competidores (Top 3)

#### Kansho (Tier 1: 1 nó ativo, 15 utilizados, 16 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=1`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 65.1895 | -0.2303 | 9.6 KB | 0.63 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 65.2157 | -0.2330 | 15.6 KB | 1.42 µs |
| #3 | `tier=1`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 65.2479 | -0.2340 | 9.0 KB | 0.93 µs |

#### Kansho (Tier 2: 2 nós ativos, 25 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False` | 41.7599 | 0.1977 | 4.6 KB | 0.42 µs |
| #2 | `tier=2`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 50.6979 | 0.1085 | 14.2 KB | 0.87 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=False` | 51.2777 | -0.0250 | 11.2 KB | 0.91 µs |

#### Kansho (Tier 4: 4 nós ativos, 25 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=False` | 44.6479 | 0.3852 | 21.5 KB | 0.91 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 44.5617 | 0.3847 | 15.5 KB | 1.43 µs |
| #3 | `tier=4`, `grace_period=200`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 45.2019 | 0.3763 | 21.5 KB | 0.91 µs |

#### Kansho (Tier 8: 8 nós ativos, 63 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 40.4648 | 0.4080 | 60.0 KB | 1.11 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True` | 37.9817 | 0.3948 | 29.4 KB | 1.68 µs |
| #3 | `tier=8`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 39.6668 | 0.3932 | 17.4 KB | 0.79 µs |

#### Kansho (Tier 16: 16 nós ativos, 43 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 34.2102 | 0.6398 | 65.0 KB | 2.34 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 29.3316 | 0.6199 | 34.4 KB | 0.78 µs |
| #3 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 30.3082 | 0.6119 | 106.4 KB | 6.76 µs |

#### Kansho (Tier 64: 64 nós ativos, 61 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2040 | 0.7062 | 232.4 KB | 5.32 µs |
| #2 | `tier=64`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.0234 | 0.6731 | 136.4 KB | 1.40 µs |
| #3 | `tier=64`, `grace_period=100`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 30.7080 | 0.6126 | 232.4 KB | 6.25 µs |

#### Kansho (Tier 128: 128 nós ativos, 61 utilizados, 2048 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2040 | 0.7062 | 464.4 KB | 4.97 µs |
| #2 | `tier=128`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.0234 | 0.6731 | 272.4 KB | 1.03 µs |
| #3 | `tier=128`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 27.7748 | 0.6509 | 682.7 KB | 2.81 µs |

#### Kansho (Tier 256: 256 nós ativos, 93 utilizados, 4096 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=True` | 18.8594 | 0.8502 | 739.9 KB | 1.65 µs |
| #2 | `tier=256`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=False` | 21.6795 | 0.7746 | 544.4 KB | 1.19 µs |
| #3 | `tier=256`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=True` | 24.2433 | 0.7393 | 689.4 KB | 2.60 µs |

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
- **Total de Experimentos (Trials):** 374

### 1. Melhores Modelos por Tier (Kansho) e Competidores (Top 3)

#### Kansho (Tier 1: 1 nó ativo, 15 utilizados, 16 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 49.6079 | 0.3264 | 34.2 KB | 1.37 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 50.1659 | 0.2881 | 34.2 KB | 5.97 µs |
| #3 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 50.3539 | 0.2860 | 12.2 KB | 3.84 µs |

#### Kansho (Tier 2: 2 nós ativos, 15 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 35.6992 | 0.6302 | 61.2 KB | 8.66 µs |
| #2 | `tier=2`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 39.3623 | 0.5414 | 19.0 KB | 7.11 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 42.9380 | 0.4458 | 35.0 KB | 8.37 µs |

#### Kansho (Tier 4: 4 nós ativos, 29 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 32.9710 | 0.6832 | 32.9 KB | 9.54 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 35.6041 | 0.6337 | 18.9 KB | 7.26 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 34.4312 | 0.6594 | 60.9 KB | 9.63 µs |

#### Kansho (Tier 8: 8 nós ativos, 43 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=0.0001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 30.4721 | 0.7279 | 36.6 KB | 15.03 µs |
| #2 | `tier=8`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 30.5630 | 0.7312 | 64.6 KB | 25.80 µs |
| #3 | `tier=8`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 30.5785 | 0.7311 | 64.6 KB | 9.20 µs |

#### Kansho (Tier 16: 16 nós ativos, 67 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 27.8188 | 0.7786 | 240.1 KB | 62.51 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 27.9501 | 0.7770 | 128.1 KB | 19.66 µs |
| #3 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=False` | 28.4250 | 0.7710 | 266.3 KB | 18.10 µs |

#### Kansho (Tier 40: 40 nós ativos, 125 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=40`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2261 | 0.8055 | 338.1 KB | 27.82 µs |
| #2 | `tier=40`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.7276 | 0.7957 | 198.1 KB | 8.10 µs |
| #3 | `tier=40`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 27.6225 | 0.7817 | 374.6 KB | 42.11 µs |

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
- **Total de Experimentos (Trials):** 521

### 1. Melhores Modelos por Tier (Kansho) e Competidores (Top 3)

#### Kansho (Tier 1: 1 nó ativo, 11 utilizados, 16 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=1`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 365.4457 | 0.0782 | 5.0 KB | 0.41 µs |
| #2 | `tier=1`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 371.0965 | 0.0600 | 9.6 KB | 0.41 µs |
| #3 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 371.7788 | 0.0532 | 9.0 KB | 0.56 µs |

#### Kansho (Tier 2: 2 nós ativos, 25 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 360.5960 | 0.1045 | 9.6 KB | 1.96 µs |
| #2 | `tier=2`, `grace_period=200`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 361.0352 | 0.0955 | 9.6 KB | 0.74 µs |
| #3 | `tier=2`, `grace_period=100`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 363.4571 | 0.0842 | 16.2 KB | 2.56 µs |

#### Kansho (Tier 4: 4 nós ativos, 35 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 348.9520 | 0.1579 | 16.8 KB | 0.91 µs |
| #2 | `tier=4`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 351.1585 | 0.1504 | 9.8 KB | 0.53 µs |
| #3 | `tier=4`, `grace_period=100`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 351.3464 | 0.1375 | 23.4 KB | 1.89 µs |

#### Kansho (Tier 8: 8 nós ativos, 51 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=100`, `delta=0.0001`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 339.2442 | 0.2054 | 18.8 KB | 0.71 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 341.0358 | 0.1823 | 33.2 KB | 1.13 µs |
| #3 | `tier=8`, `grace_period=200`, `delta=0.0001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 339.2015 | 0.1761 | 12.2 KB | 1.72 µs |

#### Kansho (Tier 16: 16 nós ativos, 39 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 317.5174 | 0.2909 | 44.6 KB | 1.39 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 319.2773 | 0.2730 | 66.1 KB | 6.88 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 323.0949 | 0.2643 | 44.6 KB | 1.35 µs |

#### Kansho (Tier 64: 64 nós ativos, 37 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=400`, `delta=0.0001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 284.4319 | 0.4078 | 95.1 KB | 1.91 µs |
| #2 | `tier=64`, `grace_period=100`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 299.4966 | 0.3577 | 151.1 KB | 3.34 µs |
| #3 | `tier=64`, `grace_period=50`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 299.9648 | 0.3477 | 263.1 KB | 4.38 µs |

#### Kansho (Tier 128: 128 nós ativos, 127 utilizados, 2048 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 300.8008 | 0.3425 | 301.9 KB | 3.56 µs |
| #2 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=False` | 305.9346 | 0.3361 | 419.2 KB | 2.14 µs |
| #3 | `tier=128`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 310.3592 | 0.3236 | 308.5 KB | 1.99 µs |

#### Kansho (Tier 256: 256 nós ativos, 75 utilizados, 4096 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 277.1476 | 0.4417 | 433.9 KB | 5.28 µs |
| #2 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 286.9681 | 0.3946 | 386.0 KB | 2.13 µs |
| #3 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 302.8055 | 0.3517 | 386.0 KB | 1.32 µs |

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
