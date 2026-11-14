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
| #1 | `tier=1`, `grace_period=50`, `delta=0.001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.87% | 93.83% | 2.3 KB | 2.68 µs |
| #2 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.73% | 93.66% | 8.8 KB | 1.42 µs |
| #3 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 94.73% | 93.66% | 8.8 KB | 2.46 µs |

#### Kansho (Tier 2: 2 nós ativos, 1 utilizado, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=0.001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.87% | 93.83% | 4.1 KB | 1.49 µs |
| #2 | `tier=2`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.73% | 93.66% | 10.6 KB | 2.25 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 94.73% | 93.66% | 10.6 KB | 1.86 µs |

#### Kansho (Tier 4: 4 nós ativos, 21 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 97.30% | 96.88% | 14.3 KB | 1.46 µs |
| #2 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 97.10% | 96.62% | 7.7 KB | 0.55 µs |
| #3 | `tier=4`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 96.67% | 96.10% | 7.7 KB | 0.16 µs |

#### Kansho (Tier 8: 8 nós ativos, 43 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.02% | 98.87% | 15.0 KB | 0.37 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 98.57% | 98.35% | 15.0 KB | 0.88 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 98.42% | 98.17% | 21.6 KB | 1.09 µs |

#### Kansho (Tier 16: 16 nós ativos, 45 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 29.6 KB | 0.40 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 98.92% | 98.75% | 36.2 KB | 2.44 µs |
| #3 | `tier=16`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.25% | 97.99% | 29.6 KB | 0.60 µs |

#### Kansho (Tier 64: 64 nós ativos, 45 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 117.4 KB | 0.49 µs |
| #2 | `tier=64`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 99.15% | 99.02% | 124.0 KB | 1.49 µs |
| #3 | `tier=64`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.38% | 98.14% | 124.0 KB | 1.90 µs |

#### Kansho (Tier 128: 128 nós ativos, 45 utilizados, 2048 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 234.4 KB | 2.93 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 98.30% | 98.03% | 241.0 KB | 1.83 µs |
| #3 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.20% | 97.94% | 234.4 KB | 2.71 µs |

#### Kansho (Tier 256: 256 nós ativos, 45 utilizados, 4096 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.22% | 99.10% | 468.4 KB | 1.24 µs |
| #2 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 98.98% | 98.83% | 475.0 KB | 2.19 µs |
| #3 | `tier=256`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 98.57% | 98.36% | 475.0 KB | 1.06 µs |

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
| #1 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.33% | 63.77% | 8.8 KB | 2.55 µs |
| #2 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.32% | 63.76% | 8.8 KB | 2.20 µs |
| #3 | `tier=1`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 63.78% | 63.69% | 2.3 KB | 0.53 µs |

#### Kansho (Tier 2: 2 nós ativos, 11 utilizados, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.18% | 63.92% | 10.6 KB | 2.16 µs |
| #2 | `tier=2`, `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.00% | 63.82% | 4.1 KB | 0.87 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.33% | 63.77% | 10.6 KB | 2.42 µs |

#### Kansho (Tier 4: 4 nós ativos, 19 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.18% | 63.92% | 14.3 KB | 3.55 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.33% | 63.77% | 14.3 KB | 2.36 µs |
| #3 | `tier=4`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.32% | 63.76% | 14.3 KB | 2.46 µs |

#### Kansho (Tier 8: 8 nós ativos, 79 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.37% | 63.96% | 15.0 KB | 0.38 µs |
| #2 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.29% | 63.94% | 21.6 KB | 1.15 µs |
| #3 | `tier=8`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.44% | 63.75% | 15.0 KB | 0.65 µs |

#### Kansho (Tier 16: 16 nós ativos, 59 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.26% | 63.86% | 29.6 KB | 2.16 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 36.2 KB | 2.35 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 63.77% | 63.67% | 36.2 KB | 1.38 µs |

#### Kansho (Tier 64: 64 nós ativos, 41 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 124.0 KB | 3.05 µs |
| #2 | `tier=64`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 63.78% | 63.69% | 117.4 KB | 0.45 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.45% | 63.61% | 117.4 KB | 0.49 µs |

#### Kansho (Tier 128: 128 nós ativos, 41 utilizados, 2048 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 241.0 KB | 2.82 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.45% | 63.61% | 234.4 KB | 0.41 µs |
| #3 | `tier=128`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.30% | 63.55% | 241.0 KB | 0.99 µs |

#### Kansho (Tier 256: 256 nós ativos, 127 utilizados, 4096 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.30% | 63.81% | 475.0 KB | 2.98 µs |
| #2 | `tier=256`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 475.0 KB | 3.05 µs |
| #3 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.32% | 63.76% | 475.0 KB | 2.02 µs |

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
| #1 | `tier=1`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 58.68% | 52.66% | 12.2 KB | 13.48 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 54.73% | 52.36% | 38.4 KB | 2.79 µs |
| #3 | `tier=1`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 54.71% | 52.02% | 12.2 KB | 0.39 µs |

#### Kansho (Tier 2: 2 nós ativos, 27 utilizados, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 61.43% | 58.51% | 22.8 KB | 5.50 µs |
| #2 | `tier=2`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 57.73% | 54.25% | 22.8 KB | 0.65 µs |
| #3 | `tier=2`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 56.64% | 53.98% | 49.0 KB | 1.97 µs |

#### Kansho (Tier 4: 4 nós ativos, 47 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.19% | 62.04% | 44.5 KB | 3.23 µs |
| #2 | `tier=4`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 60.83% | 58.52% | 70.7 KB | 1.99 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 62.95% | 58.09% | 44.5 KB | 0.97 µs |

#### Kansho (Tier 8: 8 nós ativos, 77 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 67.28% | 65.21% | 87.8 KB | 11.76 µs |
| #2 | `tier=8`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.65% | 60.36% | 87.8 KB | 0.75 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 63.60% | 60.33% | 87.8 KB | 2.18 µs |

#### Kansho (Tier 16: 16 nós ativos, 99 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 68.49% | 66.49% | 174.4 KB | 7.41 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 66.31% | 62.59% | 200.6 KB | 12.07 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 65.12% | 61.37% | 174.4 KB | 1.03 µs |

#### Kansho (Tier 40: 40 nós ativos, 113 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=40`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 68.92% | 66.86% | 453.8 KB | 7.36 µs |
| #2 | `tier=40`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 66.96% | 63.30% | 480.0 KB | 13.32 µs |
| #3 | `tier=40`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 65.12% | 61.37% | 453.8 KB | 1.01 µs |

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
| #1 | `tier=1`, `grace_period=200`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.32% | 51.85% | 8.8 KB | 1.14 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.29% | 51.71% | 2.3 KB | 1.12 µs |
| #3 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 77.09% | 49.93% | 8.8 KB | 3.46 µs |

#### Kansho (Tier 2: 2 nós ativos, 7 utilizados, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 77.17% | 54.91% | 10.6 KB | 3.15 µs |
| #2 | `tier=2`, `grace_period=200`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.32% | 51.85% | 10.6 KB | 2.10 µs |
| #3 | `tier=2`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.29% | 51.71% | 4.1 KB | 1.13 µs |

#### Kansho (Tier 4: 4 nós ativos, 33 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 77.19% | 56.64% | 7.7 KB | 0.49 µs |
| #2 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 77.17% | 54.91% | 14.3 KB | 5.31 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 77.03% | 54.30% | 7.7 KB | 0.44 µs |

#### Kansho (Tier 8: 8 nós ativos, 39 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 76.09% | 58.33% | 21.6 KB | 2.88 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.97% | 56.80% | 15.0 KB | 1.06 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 75.92% | 55.45% | 21.6 KB | 1.34 µs |

#### Kansho (Tier 16: 16 nós ativos, 67 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 72.67% | 57.12% | 36.2 KB | 2.29 µs |
| #2 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 72.55% | 57.05% | 36.2 KB | 3.39 µs |
| #3 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.87% | 56.72% | 29.6 KB | 0.65 µs |

#### Kansho (Tier 64: 64 nós ativos, 43 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 74.81% | 57.20% | 124.0 KB | 3.25 µs |
| #2 | `tier=64`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 71.28% | 56.88% | 124.0 KB | 2.78 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 76.11% | 56.69% | 117.4 KB | 3.56 µs |

#### Kansho (Tier 128: 128 nós ativos, 25 utilizados, 2048 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=200`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 76.99% | 56.90% | 234.4 KB | 0.74 µs |
| #2 | `tier=128`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 73.31% | 56.19% | 234.4 KB | 1.89 µs |
| #3 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 70.45% | 55.99% | 234.4 KB | 1.94 µs |

#### Kansho (Tier 256: 256 nós ativos, 45 utilizados, 4096 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 75.76% | 56.86% | 468.4 KB | 2.87 µs |
| #2 | `tier=256`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.69% | 55.59% | 468.4 KB | 1.97 µs |
| #3 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 76.83% | 55.47% | 475.0 KB | 1.45 µs |

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
| #1 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 3.2307 | 0.3607 | 3.0 KB | 0.97 µs |
| #2 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 3.2352 | 0.3603 | 9.0 KB | 0.91 µs |
| #3 | `tier=1`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 3.2352 | 0.3568 | 11.6 KB | 1.11 µs |

#### Kansho (Tier 2: 2 nós ativos, 19 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 2.8418 | 0.5009 | 24.2 KB | 4.17 µs |
| #2 | `tier=2`, `grace_period=100`, `delta=0.0001`, `tau=0.15`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 2.8464 | 0.4917 | 24.2 KB | 3.03 µs |
| #3 | `tier=2`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 2.9126 | 0.4780 | 17.7 KB | 1.20 µs |

#### Kansho (Tier 4: 4 nós ativos, 35 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.3536 | 0.6477 | 25.5 KB | 1.51 µs |
| #2 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 2.3876 | 0.6322 | 25.5 KB | 0.89 µs |
| #3 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.4125 | 0.6280 | 19.0 KB | 3.09 µs |

#### Kansho (Tier 8: 8 nós ativos, 45 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.1537 | 0.7013 | 44.1 KB | 4.65 µs |
| #2 | `tier=8`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 2.1618 | 0.6909 | 44.1 KB | 4.54 µs |
| #3 | `tier=8`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 2.2440 | 0.6739 | 37.5 KB | 2.12 µs |

#### Kansho (Tier 16: 16 nós ativos, 63 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.9341 | 0.7506 | 145.2 KB | 8.85 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.9860 | 0.7423 | 81.2 KB | 3.88 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 1.9556 | 0.7413 | 81.2 KB | 2.33 µs |

#### Kansho (Tier 64: 64 nós ativos, 67 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 304.0 KB | 3.73 µs |
| #2 | `tier=64`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 1.6348 | 0.8288 | 553.4 KB | 12.58 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.6547 | 0.8205 | 560.0 KB | 11.86 µs |

#### Kansho (Tier 128: 128 nós ativos, 125 utilizados, 2048 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 1.5215 | 0.8462 | 601.0 KB | 2.56 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True` | 1.5600 | 0.8432 | 601.0 KB | 7.09 µs |
| #3 | `tier=128`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 601.0 KB | 5.83 µs |

#### Kansho (Tier 256: 256 nós ativos, 51 utilizados, 4096 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 1.5513 | 0.8444 | 683.0 KB | 2.92 µs |
| #2 | `tier=256`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 953.6 KB | 3.82 µs |
| #3 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.5816 | 0.8361 | 905.6 KB | 7.76 µs |

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
| #1 | `tier=1`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 65.1895 | -0.2303 | 9.6 KB | 0.67 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 65.2157 | -0.2330 | 15.6 KB | 0.90 µs |
| #3 | `tier=1`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 65.2479 | -0.2340 | 9.0 KB | 0.54 µs |

#### Kansho (Tier 2: 2 nós ativos, 25 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False` | 41.7599 | 0.1977 | 5.7 KB | 0.37 µs |
| #2 | `tier=2`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 50.6979 | 0.1085 | 16.2 KB | 0.83 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=False` | 51.2777 | -0.0250 | 12.2 KB | 0.77 µs |

#### Kansho (Tier 4: 4 nós ativos, 25 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=False` | 44.6479 | 0.3852 | 25.5 KB | 0.86 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 44.5617 | 0.3847 | 17.5 KB | 0.75 µs |
| #3 | `tier=4`, `grace_period=200`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 45.2019 | 0.3763 | 25.5 KB | 0.94 µs |

#### Kansho (Tier 8: 8 nós ativos, 63 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 40.4648 | 0.4080 | 76.1 KB | 1.02 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True` | 37.9817 | 0.3948 | 37.5 KB | 1.52 µs |
| #3 | `tier=8`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 39.6668 | 0.3932 | 21.5 KB | 1.16 µs |

#### Kansho (Tier 16: 16 nós ativos, 43 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 34.2102 | 0.6398 | 81.2 KB | 2.08 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 29.3316 | 0.6199 | 42.6 KB | 0.77 µs |
| #3 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 30.3082 | 0.6119 | 138.6 KB | 7.01 µs |

#### Kansho (Tier 64: 64 nós ativos, 61 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2040 | 0.7062 | 297.4 KB | 4.95 µs |
| #2 | `tier=64`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.0234 | 0.6731 | 169.4 KB | 1.63 µs |
| #3 | `tier=64`, `grace_period=100`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 30.7080 | 0.6126 | 297.4 KB | 4.41 µs |

#### Kansho (Tier 128: 128 nós ativos, 61 utilizados, 2048 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2040 | 0.7062 | 594.4 KB | 5.63 µs |
| #2 | `tier=128`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.0234 | 0.6731 | 338.4 KB | 1.21 µs |
| #3 | `tier=128`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 27.7748 | 0.6509 | 888.3 KB | 2.77 µs |

#### Kansho (Tier 256: 256 nós ativos, 93 utilizados, 4096 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=True` | 18.8594 | 0.8502 | 947.1 KB | 1.44 µs |
| #2 | `tier=256`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=False` | 21.6795 | 0.7746 | 676.4 KB | 1.20 µs |
| #3 | `tier=256`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=True` | 24.2433 | 0.7393 | 899.0 KB | 2.30 µs |

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
| #1 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 49.6079 | 0.3264 | 34.2 KB | 1.71 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 50.1659 | 0.2881 | 34.2 KB | 1.57 µs |
| #3 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 50.3539 | 0.2860 | 12.2 KB | 1.05 µs |

#### Kansho (Tier 2: 2 nós ativos, 15 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 35.6992 | 0.6302 | 93.5 KB | 5.10 µs |
| #2 | `tier=2`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 39.3623 | 0.5414 | 35.3 KB | 1.76 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 42.9380 | 0.4458 | 67.3 KB | 1.31 µs |

#### Kansho (Tier 4: 4 nós ativos, 29 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 32.9710 | 0.6832 | 69.5 KB | 5.50 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 34.0290 | 0.6650 | 45.5 KB | 1.93 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 34.4312 | 0.6594 | 133.5 KB | 3.19 µs |

#### Kansho (Tier 8: 8 nós ativos, 43 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=0.0001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 29.7630 | 0.7401 | 89.8 KB | 2.87 µs |
| #2 | `tier=8`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 30.5630 | 0.7312 | 137.8 KB | 5.96 µs |
| #3 | `tier=8`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 30.5785 | 0.7311 | 137.8 KB | 1.66 µs |

#### Kansho (Tier 16: 16 nós ativos, 67 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 27.8188 | 0.7786 | 530.4 KB | 35.00 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 27.9501 | 0.7770 | 274.4 KB | 8.72 µs |
| #3 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=False` | 28.4250 | 0.7710 | 556.6 KB | 13.52 µs |

#### Kansho (Tier 40: 40 nós ativos, 125 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=40`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2261 | 0.8055 | 703.8 KB | 25.91 µs |
| #2 | `tier=40`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.6769 | 0.7963 | 463.8 KB | 2.68 µs |
| #3 | `tier=40`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 27.6225 | 0.7817 | 828.1 KB | 22.13 µs |

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
| #1 | `tier=1`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 365.4457 | 0.0782 | 5.0 KB | 0.65 µs |
| #2 | `tier=1`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 371.0965 | 0.0600 | 9.6 KB | 0.49 µs |
| #3 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 371.7788 | 0.0532 | 9.0 KB | 0.88 µs |

#### Kansho (Tier 2: 2 nós ativos, 25 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 360.5960 | 0.1045 | 17.7 KB | 2.58 µs |
| #2 | `tier=2`, `grace_period=200`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 361.0352 | 0.0955 | 17.7 KB | 1.56 µs |
| #3 | `tier=2`, `grace_period=100`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 363.4571 | 0.0842 | 24.2 KB | 3.61 µs |

#### Kansho (Tier 4: 4 nós ativos, 35 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 348.9520 | 0.1579 | 35.0 KB | 1.02 µs |
| #2 | `tier=4`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 351.1585 | 0.1504 | 19.0 KB | 0.68 µs |
| #3 | `tier=4`, `grace_period=100`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 351.3464 | 0.1375 | 41.5 KB | 1.77 µs |

#### Kansho (Tier 8: 8 nós ativos, 51 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=100`, `delta=0.0001`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 339.2442 | 0.2054 | 28.1 KB | 0.65 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 341.0358 | 0.1823 | 69.5 KB | 3.05 µs |
| #3 | `tier=8`, `grace_period=200`, `delta=0.0001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 339.2015 | 0.1761 | 21.5 KB | 0.96 µs |

#### Kansho (Tier 16: 16 nós ativos, 39 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 317.5174 | 0.2909 | 81.2 KB | 1.31 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 319.2773 | 0.2730 | 138.6 KB | 6.08 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 323.0949 | 0.2643 | 81.2 KB | 1.97 µs |

#### Kansho (Tier 64: 64 nós ativos, 37 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=400`, `delta=0.0001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 284.4319 | 0.4078 | 169.4 KB | 2.50 µs |
| #2 | `tier=64`, `grace_period=100`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 299.4966 | 0.3577 | 297.4 KB | 5.01 µs |
| #3 | `tier=64`, `grace_period=50`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 299.9648 | 0.3477 | 553.4 KB | 4.64 µs |

#### Kansho (Tier 128: 128 nós ativos, 127 utilizados, 2048 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 300.8008 | 0.3425 | 594.4 KB | 4.02 µs |
| #2 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=False` | 305.9346 | 0.3361 | 881.7 KB | 2.59 µs |
| #3 | `tier=128`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 310.3592 | 0.3236 | 601.0 KB | 1.50 µs |

#### Kansho (Tier 256: 256 nós ativos, 75 utilizados, 4096 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 277.1476 | 0.4417 | 905.6 KB | 8.27 µs |
| #2 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 286.9681 | 0.3946 | 683.0 KB | 3.46 µs |
| #3 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 302.8055 | 0.3517 | 683.0 KB | 2.01 µs |

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
