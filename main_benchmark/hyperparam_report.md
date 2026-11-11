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
| #1 | `tier=1`, `grace_period=50`, `delta=0.001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.87% | 93.83% | 4.8 KB | 0.77 µs |
| #2 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.73% | 93.66% | 11.3 KB | 0.96 µs |
| #3 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 94.73% | 93.66% | 11.3 KB | 0.96 µs |

#### Kansho (Tier 2: 2 nós ativos, 1 utilizado, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=0.001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.87% | 93.83% | 9.2 KB | 0.85 µs |
| #2 | `tier=2`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 94.73% | 93.66% | 15.7 KB | 0.70 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 94.73% | 93.66% | 15.7 KB | 0.94 µs |

#### Kansho (Tier 4: 4 nós ativos, 21 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 97.30% | 96.88% | 24.5 KB | 0.34 µs |
| #2 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 97.33% | 96.91% | 18.0 KB | 0.23 µs |
| #3 | `tier=4`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=mc` | 96.20% | 95.53% | 18.0 KB | 0.12 µs |

#### Kansho (Tier 8: 8 nós ativos, 43 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 98.82% | 98.64% | 35.5 KB | 0.22 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 98.58% | 98.37% | 35.5 KB | 0.17 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 98.42% | 98.17% | 42.1 KB | 0.29 µs |

#### Kansho (Tier 16: 16 nós ativos, 45 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.08% | 98.95% | 70.6 KB | 0.54 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 98.95% | 98.79% | 77.2 KB | 1.24 µs |
| #3 | `tier=16`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.25% | 97.99% | 70.6 KB | 0.74 µs |

#### Kansho (Tier 64: 64 nós ativos, 45 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.10% | 98.97% | 281.4 KB | 0.68 µs |
| #2 | `tier=64`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 99.12% | 98.99% | 288.0 KB | 0.81 µs |
| #3 | `tier=64`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.38% | 98.14% | 288.0 KB | 1.13 µs |

#### Kansho (Tier 128: 128 nós ativos, 45 utilizados, 2048 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.10% | 98.97% | 562.4 KB | 0.77 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 98.30% | 98.03% | 569.0 KB | 1.50 µs |
| #3 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 98.20% | 97.94% | 562.4 KB | 1.12 µs |

#### Kansho (Tier 256: 256 nós ativos, 45 utilizados, 4096 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=100`, `delta=0.01`, `tau=0.1`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 99.10% | 98.97% | 896.1 KB | 0.71 µs |
| #2 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 98.98% | 98.83% | 902.6 KB | 1.26 µs |
| #3 | `tier=256`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 98.57% | 98.36% | 902.6 KB | 0.56 µs |

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
| #1 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.33% | 63.77% | 11.3 KB | 1.65 µs |
| #2 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.32% | 63.76% | 11.3 KB | 1.48 µs |
| #3 | `tier=1`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 63.78% | 63.69% | 4.8 KB | 0.53 µs |

#### Kansho (Tier 2: 2 nós ativos, 11 utilizados, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.18% | 63.92% | 15.7 KB | 1.34 µs |
| #2 | `tier=2`, `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.00% | 63.82% | 9.2 KB | 0.53 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.33% | 63.77% | 15.7 KB | 2.67 µs |

#### Kansho (Tier 4: 4 nós ativos, 19 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.18% | 63.92% | 24.5 KB | 1.67 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.33% | 63.77% | 24.5 KB | 2.20 µs |
| #3 | `tier=4`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.32% | 63.76% | 24.5 KB | 1.96 µs |

#### Kansho (Tier 8: 8 nós ativos, 79 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.37% | 63.50% | 35.5 KB | 0.30 µs |
| #2 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.29% | 63.94% | 42.1 KB | 1.03 µs |
| #3 | `tier=8`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.44% | 63.75% | 35.5 KB | 0.63 µs |

#### Kansho (Tier 16: 16 nós ativos, 59 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.26% | 63.86% | 70.6 KB | 0.78 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 77.2 KB | 1.21 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 63.77% | 63.67% | 77.2 KB | 0.69 µs |

#### Kansho (Tier 64: 64 nós ativos, 41 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 288.0 KB | 1.28 µs |
| #2 | `tier=64`, `grace_period=200`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 63.78% | 63.69% | 281.4 KB | 0.19 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.18% | 63.66% | 281.4 KB | 0.20 µs |

#### Kansho (Tier 128: 128 nós ativos, 41 utilizados, 2048 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 569.0 KB | 1.46 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.18% | 63.66% | 562.4 KB | 0.23 µs |
| #3 | `tier=128`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 64.30% | 63.55% | 569.0 KB | 0.73 µs |

#### Kansho (Tier 256: 256 nós ativos, 127 utilizados, 4096 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 64.30% | 63.81% | 902.6 KB | 1.58 µs |
| #2 | `tier=256`, `grace_period=200`, `delta=0.0001`, `tau=0.05`, `split_criterion=info_gain`, `leaf_prediction=nba`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 64.20% | 63.77% | 902.6 KB | 1.28 µs |
| #3 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 64.32% | 63.76% | 902.6 KB | 0.91 µs |

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
| #1 | `tier=1`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 52.15% | 45.82% | 18.0 KB | 1.73 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 54.73% | 52.36% | 44.2 KB | 1.25 µs |
| #3 | `tier=1`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 54.71% | 52.02% | 18.0 KB | 0.33 µs |

#### Kansho (Tier 2: 2 nós ativos, 27 utilizados, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 53.81% | 49.23% | 34.8 KB | 1.20 µs |
| #2 | `tier=2`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 57.73% | 54.25% | 34.8 KB | 0.20 µs |
| #3 | `tier=2`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 56.64% | 53.98% | 61.0 KB | 0.80 µs |

#### Kansho (Tier 4: 4 nós ativos, 47 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 60.35% | 55.80% | 68.5 KB | 1.61 µs |
| #2 | `tier=4`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 60.67% | 58.20% | 94.7 KB | 0.79 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 62.95% | 58.09% | 68.5 KB | 0.36 µs |

#### Kansho (Tier 8: 8 nós ativos, 77 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 63.07% | 57.94% | 135.8 KB | 2.91 µs |
| #2 | `tier=8`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 64.64% | 60.41% | 135.8 KB | 0.44 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 59.83% | 56.03% | 135.8 KB | 1.49 µs |

#### Kansho (Tier 16: 16 nós ativos, 99 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 63.25% | 58.54% | 270.4 KB | 4.53 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 67.61% | 64.41% | 296.6 KB | 4.46 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 65.08% | 61.60% | 270.4 KB | 0.74 µs |

#### Kansho (Tier 40: 40 nós ativos, 113 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=40`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 63.95% | 59.88% | 687.8 KB | 3.89 µs |
| #2 | `tier=40`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=mc` | 66.96% | 63.30% | 714.0 KB | 2.63 µs |
| #3 | `tier=40`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 65.08% | 61.60% | 687.8 KB | 0.40 µs |

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
| #1 | `tier=1`, `grace_period=200`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.32% | 51.85% | 11.3 KB | 1.29 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.29% | 51.71% | 4.8 KB | 1.44 µs |
| #3 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 77.09% | 49.93% | 11.3 KB | 2.29 µs |

#### Kansho (Tier 2: 2 nós ativos, 7 utilizados, 32 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 77.17% | 54.91% | 15.7 KB | 1.74 µs |
| #2 | `tier=2`, `grace_period=200`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.32% | 51.85% | 15.7 KB | 1.03 µs |
| #3 | `tier=2`, `grace_period=400`, `delta=0.0001`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nba` | 77.29% | 51.71% | 9.2 KB | 1.01 µs |

#### Kansho (Tier 4: 4 nós ativos, 33 utilizados, 64 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 77.19% | 56.64% | 18.0 KB | 0.46 µs |
| #2 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 77.17% | 54.91% | 24.5 KB | 2.06 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=mc`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 77.03% | 54.30% | 18.0 KB | 0.33 µs |

#### Kansho (Tier 8: 8 nós ativos, 39 utilizados, 128 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 76.09% | 58.33% | 42.1 KB | 1.80 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.97% | 56.80% | 35.5 KB | 1.53 µs |
| #3 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 75.92% | 55.45% | 42.1 KB | 1.00 µs |

#### Kansho (Tier 16: 16 nós ativos, 67 utilizados, 256 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 72.67% | 57.12% | 77.2 KB | 1.00 µs |
| #2 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 72.55% | 57.05% | 77.2 KB | 1.13 µs |
| #3 | `tier=16`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.87% | 56.72% | 70.6 KB | 0.30 µs |

#### Kansho (Tier 64: 64 nós ativos, 43 utilizados, 1024 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 74.81% | 57.20% | 288.0 KB | 0.95 µs |
| #2 | `tier=64`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 71.28% | 56.88% | 288.0 KB | 1.43 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 76.11% | 56.69% | 281.4 KB | 1.00 µs |

#### Kansho (Tier 128: 128 nós ativos, 25 utilizados, 2048 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=200`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 76.99% | 56.90% | 562.4 KB | 0.62 µs |
| #2 | `tier=128`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=gini`, `leaf_prediction=nb` | 73.31% | 56.19% | 562.4 KB | 0.90 µs |
| #3 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 70.45% | 55.99% | 562.4 KB | 1.39 µs |

#### Kansho (Tier 256: 256 nós ativos, 45 utilizados, 4096 nós no max)

| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nb` | 75.76% | 56.86% | 896.1 KB | 1.10 µs |
| #2 | `tier=256`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `split_criterion=gini`, `leaf_prediction=nba`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 76.69% | 55.59% | 896.1 KB | 0.47 µs |
| #3 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `split_criterion=info_gain`, `leaf_prediction=nba` | 76.87% | 55.50% | 902.6 KB | 0.69 µs |

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
| #1 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 3.2307 | 0.3607 | 2.8 KB | 0.14 µs |
| #2 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 3.2352 | 0.3603 | 8.8 KB | 0.17 µs |
| #3 | `tier=1`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 3.2352 | 0.3568 | 11.3 KB | 0.43 µs |

#### Kansho (Tier 2: 2 nós ativos, 19 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 2.8418 | 0.5009 | 23.7 KB | 0.79 µs |
| #2 | `tier=2`, `grace_period=100`, `delta=0.0001`, `tau=0.15`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False` | 2.8464 | 0.4917 | 23.7 KB | 1.02 µs |
| #3 | `tier=2`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 2.9126 | 0.4780 | 17.2 KB | 0.53 µs |

#### Kansho (Tier 4: 4 nós ativos, 35 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.3536 | 0.6477 | 24.5 KB | 0.65 µs |
| #2 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 2.3876 | 0.6322 | 24.5 KB | 0.70 µs |
| #3 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.4125 | 0.6280 | 18.0 KB | 0.86 µs |

#### Kansho (Tier 8: 8 nós ativos, 45 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 2.1537 | 0.7013 | 42.1 KB | 1.02 µs |
| #2 | `tier=8`, `grace_period=400`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 2.1618 | 0.6909 | 42.1 KB | 1.03 µs |
| #3 | `tier=8`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 2.2440 | 0.6739 | 35.5 KB | 0.67 µs |

#### Kansho (Tier 16: 16 nós ativos, 63 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=0.001`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.9341 | 0.7506 | 141.2 KB | 3.10 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.9860 | 0.7423 | 77.2 KB | 1.40 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 1.9556 | 0.7413 | 77.2 KB | 1.36 µs |

#### Kansho (Tier 64: 64 nós ativos, 67 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 288.0 KB | 1.67 µs |
| #2 | `tier=64`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 1.6348 | 0.8288 | 537.4 KB | 4.63 µs |
| #3 | `tier=64`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.6547 | 0.8205 | 544.0 KB | 3.98 µs |

#### Kansho (Tier 128: 128 nós ativos, 125 utilizados, 2048 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 1.5215 | 0.8462 | 569.0 KB | 1.46 µs |
| #2 | `tier=128`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True` | 1.5600 | 0.8432 | 569.0 KB | 2.67 µs |
| #3 | `tier=128`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 569.0 KB | 1.35 µs |

#### Kansho (Tier 256: 256 nós ativos, 51 utilizados, 4096 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 1.5513 | 0.8444 | 619.0 KB | 1.10 µs |
| #2 | `tier=256`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 1.5644 | 0.8415 | 902.6 KB | 0.80 µs |
| #3 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 1.5816 | 0.8361 | 879.6 KB | 2.46 µs |

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
| #1 | `tier=1`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 65.1895 | -0.2303 | 9.3 KB | 0.45 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 65.2157 | -0.2330 | 15.3 KB | 1.31 µs |
| #3 | `tier=1`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 65.2479 | -0.2340 | 8.8 KB | 0.30 µs |

#### Kansho (Tier 2: 2 nós ativos, 25 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False` | 41.7599 | 0.1977 | 5.2 KB | 0.34 µs |
| #2 | `tier=2`, `grace_period=100`, `delta=1e-07`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 50.6979 | 0.1085 | 15.7 KB | 0.89 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=False` | 51.2777 | -0.0250 | 11.7 KB | 0.97 µs |

#### Kansho (Tier 4: 4 nós ativos, 25 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=False` | 44.6479 | 0.3852 | 24.5 KB | 0.37 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 44.5617 | 0.3847 | 16.5 KB | 0.56 µs |
| #3 | `tier=4`, `grace_period=200`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 45.2019 | 0.3763 | 24.5 KB | 0.65 µs |

#### Kansho (Tier 8: 8 nós ativos, 63 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 40.4648 | 0.4080 | 74.1 KB | 0.43 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True` | 37.9817 | 0.3948 | 35.5 KB | 0.88 µs |
| #3 | `tier=8`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 39.6668 | 0.3932 | 19.5 KB | 0.45 µs |

#### Kansho (Tier 16: 16 nós ativos, 43 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 34.2102 | 0.6398 | 77.2 KB | 1.34 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 29.3316 | 0.6199 | 38.6 KB | 0.94 µs |
| #3 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 30.3082 | 0.6119 | 134.6 KB | 3.67 µs |

#### Kansho (Tier 64: 64 nós ativos, 61 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2040 | 0.7062 | 281.4 KB | 3.05 µs |
| #2 | `tier=64`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.0234 | 0.6731 | 153.4 KB | 1.04 µs |
| #3 | `tier=64`, `grace_period=100`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 30.7080 | 0.6126 | 281.4 KB | 2.68 µs |

#### Kansho (Tier 128: 128 nós ativos, 61 utilizados, 2048 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2040 | 0.7062 | 562.4 KB | 2.74 µs |
| #2 | `tier=128`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.0234 | 0.6731 | 306.4 KB | 0.57 µs |
| #3 | `tier=128`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 27.7748 | 0.6509 | 862.8 KB | 1.32 µs |

#### Kansho (Tier 256: 256 nós ativos, 93 utilizados, 4096 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=16`, `leaf_normalize=True` | 18.8594 | 0.8502 | 896.1 KB | 0.94 µs |
| #2 | `tier=256`, `grace_period=50`, `delta=0.001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=False` | 21.6795 | 0.7746 | 612.4 KB | 0.75 µs |
| #3 | `tier=256`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=True` | 24.2433 | 0.7393 | 873.0 KB | 1.10 µs |

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
| #1 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 49.6079 | 0.3264 | 34.0 KB | 0.99 µs |
| #2 | `tier=1`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 50.1659 | 0.2881 | 34.0 KB | 0.72 µs |
| #3 | `tier=1`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 50.3539 | 0.2860 | 12.0 KB | 0.38 µs |

#### Kansho (Tier 2: 2 nós ativos, 15 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=400`, `delta=0.0001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 35.6992 | 0.6302 | 93.0 KB | 1.75 µs |
| #2 | `tier=2`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 39.3623 | 0.5414 | 34.8 KB | 0.75 µs |
| #3 | `tier=2`, `grace_period=200`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 42.9380 | 0.4458 | 66.8 KB | 0.70 µs |

#### Kansho (Tier 4: 4 nós ativos, 29 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 32.9710 | 0.6832 | 68.5 KB | 1.15 µs |
| #2 | `tier=4`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 34.0290 | 0.6650 | 44.5 KB | 1.02 µs |
| #3 | `tier=4`, `grace_period=400`, `delta=0.01`, `tau=0.01`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 34.4312 | 0.6594 | 132.5 KB | 1.32 µs |

#### Kansho (Tier 8: 8 nós ativos, 43 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=200`, `delta=0.0001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 29.7630 | 0.7401 | 87.8 KB | 2.36 µs |
| #2 | `tier=8`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 30.5630 | 0.7312 | 135.8 KB | 2.10 µs |
| #3 | `tier=8`, `grace_period=100`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 30.5785 | 0.7311 | 135.8 KB | 1.06 µs |

#### Kansho (Tier 16: 16 nós ativos, 67 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 27.8188 | 0.7786 | 526.4 KB | 6.80 µs |
| #2 | `tier=16`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 27.9501 | 0.7770 | 270.4 KB | 2.90 µs |
| #3 | `tier=16`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=False` | 28.4250 | 0.7710 | 552.6 KB | 2.73 µs |

#### Kansho (Tier 40: 40 nós ativos, 125 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=40`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.2261 | 0.8055 | 687.8 KB | 4.09 µs |
| #2 | `tier=40`, `grace_period=200`, `delta=0.01`, `tau=0.1`, `leaf_prediction=mean`, `reg_num_slots=8`, `leaf_normalize=False`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 26.6769 | 0.7963 | 447.8 KB | 1.05 µs |
| #3 | `tier=40`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 27.6225 | 0.7817 | 821.9 KB | 7.05 µs |

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
| #1 | `tier=1`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 365.4457 | 0.0782 | 4.8 KB | 0.09 µs |
| #2 | `tier=1`, `grace_period=200`, `delta=1e-07`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 371.0965 | 0.0600 | 9.3 KB | 0.22 µs |
| #3 | `tier=1`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 371.7788 | 0.0532 | 8.8 KB | 0.23 µs |

#### Kansho (Tier 2: 2 nós ativos, 25 utilizados, 32 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=2`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 360.5960 | 0.1045 | 17.2 KB | 0.44 µs |
| #2 | `tier=2`, `grace_period=200`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=mean`, `reg_num_slots=32`, `leaf_normalize=False` | 361.0352 | 0.0955 | 17.2 KB | 0.35 µs |
| #3 | `tier=2`, `grace_period=100`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 363.4571 | 0.0842 | 23.7 KB | 0.78 µs |

#### Kansho (Tier 4: 4 nós ativos, 35 utilizados, 64 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=4`, `grace_period=50`, `delta=1e-07`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 348.9520 | 0.1579 | 34.0 KB | 1.25 µs |
| #2 | `tier=4`, `grace_period=50`, `delta=0.01`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 351.1585 | 0.1504 | 18.0 KB | 0.58 µs |
| #3 | `tier=4`, `grace_period=100`, `delta=0.01`, `tau=0.01`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 351.3464 | 0.1375 | 40.5 KB | 1.90 µs |

#### Kansho (Tier 8: 8 nós ativos, 51 utilizados, 128 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=8`, `grace_period=100`, `delta=0.0001`, `tau=0.05`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 339.2442 | 0.2054 | 26.1 KB | 1.21 µs |
| #2 | `tier=8`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 341.0358 | 0.1823 | 67.5 KB | 1.61 µs |
| #3 | `tier=8`, `grace_period=200`, `delta=0.0001`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 339.2015 | 0.1761 | 19.5 KB | 1.77 µs |

#### Kansho (Tier 16: 16 nós ativos, 39 utilizados, 256 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=16`, `grace_period=400`, `delta=1e-07`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False` | 317.5174 | 0.2909 | 77.2 KB | 1.92 µs |
| #2 | `tier=16`, `grace_period=200`, `delta=0.0001`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 319.2773 | 0.2730 | 134.6 KB | 3.58 µs |
| #3 | `tier=16`, `grace_period=400`, `delta=0.001`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=False` | 323.0949 | 0.2643 | 77.2 KB | 1.88 µs |

#### Kansho (Tier 64: 64 nós ativos, 37 utilizados, 1024 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=64`, `grace_period=400`, `delta=0.0001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 284.4319 | 0.4078 | 153.4 KB | 1.83 µs |
| #2 | `tier=64`, `grace_period=100`, `delta=0.001`, `tau=0.05`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=True` | 299.4966 | 0.3577 | 281.4 KB | 3.43 µs |
| #3 | `tier=64`, `grace_period=50`, `delta=1e-07`, `tau=0.15`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=True` | 299.9648 | 0.3477 | 537.4 KB | 4.79 µs |

#### Kansho (Tier 128: 128 nós ativos, 127 utilizados, 2048 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=128`, `grace_period=100`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=adaptive`, `reg_num_slots=16`, `leaf_normalize=True`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False` | 300.8008 | 0.3425 | 562.4 KB | 1.14 µs |
| #2 | `tier=128`, `grace_period=50`, `delta=0.01`, `tau=0.1`, `prep_scaler=None`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=32`, `leaf_normalize=False` | 305.9346 | 0.3361 | 856.2 KB | 0.65 µs |
| #3 | `tier=128`, `grace_period=200`, `delta=1e-07`, `tau=0.01`, `leaf_prediction=model`, `reg_num_slots=16`, `leaf_normalize=False`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False` | 310.3592 | 0.3236 | 569.0 KB | 0.82 µs |

#### Kansho (Tier 256: 256 nós ativos, 75 utilizados, 4096 nós no max)

| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `tier=256`, `grace_period=200`, `delta=0.01`, `tau=0.15`, `prep_scaler=standard`, `obs_splitter=histogram`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=32`, `leaf_normalize=True` | 277.1476 | 0.4417 | 879.6 KB | 1.73 µs |
| #2 | `tier=256`, `grace_period=400`, `delta=0.001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=model`, `reg_num_slots=8`, `leaf_normalize=True` | 286.9681 | 0.3946 | 619.0 KB | 0.80 µs |
| #3 | `tier=256`, `grace_period=100`, `delta=0.0001`, `tau=0.1`, `prep_scaler=standard`, `obs_splitter=gaussian`, `filter_hampel=False`, `leaf_prediction=adaptive`, `reg_num_slots=8`, `leaf_normalize=False` | 302.8055 | 0.3517 | 619.0 KB | 0.50 µs |

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
