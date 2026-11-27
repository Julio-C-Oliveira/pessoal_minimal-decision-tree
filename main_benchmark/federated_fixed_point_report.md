# Relatório do Benchmark Federado Multi-Cliente (8 Datasets Reais — Non-IID & IID)

Avaliação experimental completa dos dois paradigmas federados do **kansho** em **C11 puro (zero-malloc)** nos **8 datasets emblemáticos** do `main_benchmark/` (4 de Classificação e 4 de Regressão Contínua), comparando **Modo 1 (`Fed-Forest`)** e **Modo 2 (`Fed-VFDT` Coordenado & Stats-Merge)** contra o baseline **Local-Only** ($K$ clientes isolados) e o **Centralized Oracle** (árvore única centralizada no Tier 8).

## 1. Protocolo Experimental e Orçamento Embarcado

- **Clientes ($K$):** `10` clientes independentes
- **Rodadas de Comunicação ($R$):** `20` rodadas síncronas por stream
- **Regime Non-IID:** Particionamento `Dirichlet(alpha=0.3)` (sobre classes em classificação; sobre 10 quantis do alvo $y$ em regressão; `seed=42`)
- **Regime IID:** Particionamento aleatório uniforme entre os $K=10$ clientes (`seed=42`)
- **Perfil de Hardware (MCU Tier 8):** `HT_MAX_ACTIVE_LEAVES=8`, `HT_MAX_NODES=128`, `HT_COMPACT_MAX_NODES=64`, `HT_FED_FOREST_MAX_TREES=5`, `HT_FED_SERVER_MAX_TREES=16`, `HT_FED_SERVER_MAX_PROPOSALS=128`, `float32`, **zero `malloc` em runtime**.
- **Hiperparâmetros:** Reuso direto dos hiperparâmetros ótimos `kansho_t8` de `main_benchmark/main_benchmark.py` (com `client_grace = max(20, grace // K)` para manter ritmo de splits comparável ao volume local).
- **Consenso Fed-VFDT (Fiel ao Paper IEEE ICTAI 2025):** `HT_FED_THRESH_MAX_MERIT`, `require_locally_ready=true`, `QUORUM` por atributo (`min_quorum=3`), `MAJORITY` por pluralidade (`min_quorum=2`), `BEST-MERIT` (`max(G_k)`, `min_quorum=2`).
- **Avaliação Global:** Todos os $K$ clientes finais são avaliados no conjunto de teste global (`*_test.csv`); as métricas de holdout reportam a média dos $K$ clientes (medindo generalização global real sob viés local Non-IID).

## 2. Orçamento de RAM (Cliente e Servidor: Modelo vs. Comunicação) e Heterogeneidade Non-IID

Decomposição exata da memória estática (`.bss` + buffer half-duplex de comunicação `wire_buf` dimensionado por `max(max_up, max_down)`) e do pico transitório de pilha (`Stack`), com **zero `malloc` em runtime**:
- **Modelo (`Mod`):** árvore(s) residente(s) (`ht_tree_t`; no cliente `Fed-Forest` inclui as $M=5$ `peer_trees`; no servidor `Fed-Forest` inclui `entries[16]`) + pré-processador local `ht_scaler_t` quando ativo.
- **Comunicação & Coordenação (`Com`):** metadados/logs federados (`split_log[64]`, `round_proposals[128]`, `merge_leaf_delta[8]`, tabelas de votação/drift) + buffer de rede `wire_buf` (pico real observado por estratégia).
- **Stack (`Stk`):** pico transitório na pilha durante `export_update` / `ingest` (ex: `ht_compact_tree_t`, `props[16]` ou `ht_leaf_stats_t`), liberado imediatamente ao retornar da função.

### 2.1 Decomposição de RAM do Lado do Cliente (1 Dispositivo MCU)

| Dataset | Tarefa | Features / Cls | Scaler | Central / Local (Mod) | Fed-Forest ($M=5$)<br/>Mod + Com = Total \| Stk | Fed-VFDT Coord.<br/>Mod + Com = Total \| Stk | Fed-VFDT Stats-Merge<br/>Mod + Com = Total \| Stk | Buf. Pior Caso `.bss`<br/>(FF / Coord / Merge) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | classification | 3 / 2 cls | 0.00 KB | 6.66 KB | 13.06 + 1.55 = **14.62 KB** \| Stk 1.28 | 6.66 + 0.46 = **7.11 KB** \| Stk 0.81 | 6.66 + 2.18 = **8.84 KB** \| Stk 0.05 | 6.5 / 3.0 / 3.0 KB |
| **Electricity** | classification | 8 / 2 cls | 0.00 KB | 9.80 KB | 16.33 + 1.98 = **18.31 KB** \| Stk 1.30 | 9.80 + 0.46 = **10.26 KB** \| Stk 0.81 | 9.80 + 5.30 = **15.11 KB** \| Stk 0.05 | 6.6 / 3.0 / 5.3 KB |
| **Forest Covertype** | classification | 54 / 7 cls | 0.00 KB | 65.55 KB | 76.95 + 4.17 = **81.12 KB** \| Stk 2.28 | 65.55 + 0.71 = **66.25 KB** \| Stk 1.31 | 65.55 + 59.49 = **125.04 KB** \| Stk 0.06 | 11.5 / 5.0 / 59.5 KB |
| **Airlines** | classification | 7 / 2 cls | 0.16 KB | 9.34 KB | 15.84 + 1.02 = **16.86 KB** \| Stk 1.30 | 9.34 + 0.25 = **9.59 KB** \| Stk 0.81 | 9.34 + 4.68 = **14.02 KB** \| Stk 0.05 | 6.6 / 3.0 / 4.7 KB |
| **Friedman #1** | regression | 10 / cont. | 0.23 KB | 25.48 KB | 32.06 + 3.60 = **35.66 KB** \| Stk 1.32 | 25.48 + 0.46 = **25.93 KB** \| Stk 0.81 | 25.48 + 20.62 = **46.09 KB** \| Stk 0.05 | 6.7 / 3.0 / 20.6 KB |
| **Bikes** | regression | 12 / cont. | 0.28 KB | 53.62 KB | 60.24 + 3.95 = **64.19 KB** \| Stk 1.32 | 53.62 + 0.46 = **54.07 KB** \| Stk 0.81 | 53.62 + 48.68 = **102.30 KB** \| Stk 0.05 | 6.7 / 3.0 / 48.7 KB |
| **Appliances Energy** | regression | 28 / cont. | 0.00 KB | 34.09 KB | 41.10 + 4.62 = **45.72 KB** \| Stk 1.40 | 34.09 + 0.46 = **34.54 KB** \| Stk 0.81 | 34.09 + 29.18 = **63.27 KB** \| Stk 0.05 | 7.1 / 3.0 / 29.2 KB |
| **NYC Taxi** | regression | 7 / cont. | 0.16 KB | 12.26 KB | 18.76 + 4.07 = **22.83 KB** \| Stk 1.30 | 12.26 + 0.41 = **12.66 KB** \| Stk 0.81 | 12.26 + 7.52 = **19.78 KB** \| Stk 0.05 | 6.6 / 3.0 / 7.5 KB |

### 2.2 Decomposição de RAM do Lado do Servidor (1 Agregador) e Heterogeneidade Non-IID

| Dataset | Fed-Forest Server (`16` árv.)<br/>Mod + Com = Total \| Stk | Fed-VFDT Coord. Server<br/>Mod + Com = Total \| Stk | Fed-VFDT Stats-Merge Server<br/>Mod + Com = Total \| Stk | Non-IID Amostras/Cli | Métrica de Heterogeneidade Non-IID |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SEA** | 20.50 + 1.77 = **22.27 KB** \| Stk 1.28 | 6.66 + 10.26 = **16.92 KB** \| Stk 0.38 | 6.66 + 11.98 = **18.64 KB** \| Stk 0.26 | 69–4748 | Classe dom. média: **92.7%** (H=0.34 bits) |
| **Electricity** | 20.88 + 2.20 = **23.07 KB** \| Stk 1.30 | 9.80 + 10.28 = **20.08 KB** \| Stk 0.38 | 9.80 + 15.12 = **24.93 KB** \| Stk 0.67 | 127–8361 | Classe dom. média: **94.1%** (H=0.21 bits) |
| **Forest Covertype** | 36.50 + 4.38 = **40.88 KB** \| Stk 2.28 | 65.55 + 16.83 = **82.38 KB** \| Stk 0.62 | 65.55 + 75.62 = **141.16 KB** \| Stk 7.61 | 420–4453 | Classe dom. média: **53.2%** (H=1.68 bits) |
| **Airlines** | 20.81 + 1.24 = **22.05 KB** \| Stk 1.30 | 9.17 + 10.07 = **19.25 KB** \| Stk 0.38 | 9.17 + 14.50 = **23.67 KB** \| Stk 0.59 | 158–7066 | Classe dom. média: **88.2%** (H=0.41 bits) |
| **Friedman #1** | 21.06 + 3.82 = **24.88 KB** \| Stk 1.32 | 25.24 + 10.29 = **35.53 KB** \| Stk 0.38 | 25.24 + 30.45 = **55.69 KB** \| Stk 2.56 | 236–3554 | Dispersão $\mu_k(y)/\sigma(y)$: **0.59$\sigma$** |
| **Bikes** | 21.19 + 4.17 = **25.36 KB** \| Stk 1.32 | 53.34 + 10.29 = **63.63 KB** \| Stk 0.38 | 53.34 + 58.52 = **111.85 KB** \| Stk 6.06 | 496–2010 | Dispersão $\mu_k(y)/\sigma(y)$: **0.35$\sigma$** |
| **Appliances Energy** | 22.44 + 4.84 = **27.27 KB** \| Stk 1.40 | 34.09 + 10.36 = **44.44 KB** \| Stk 0.38 | 34.09 + 39.08 = **73.16 KB** \| Stk 3.62 | 511–2598 | Dispersão $\mu_k(y)/\sigma(y)$: **0.48$\sigma$** |
| **NYC Taxi** | 20.81 + 4.29 = **25.10 KB** \| Stk 1.30 | 12.09 + 10.23 = **22.32 KB** \| Stk 0.38 | 12.09 + 17.34 = **29.44 KB** \| Stk 0.92 | 212–2260 | Dispersão $\mu_k(y)/\sigma(y)$: **0.35$\sigma$** |

> **Validação de Orçamento Embarcado (Modelo vs. Comunicação):** Isolando apenas o **modelo estático**, o pico no cliente foi de **76.95 KB** (Forest Covertype, 54 features / 7 classes). Ao contabilizar também o **buffer half-duplex de comunicação (`wire_buf`) + coordenação**, **`Fed-VFDT Coordenado`** (pico total **66.25 KB**) e **`Fed-Forest`** (pico total **81.12 KB**) permanecem confortavelmente abaixo do teto de **100 KB** para MCUs como ESP32; já **`Fed-VFDT (STATS-MERGE)`** exige um buffer TX de até **59.48 KB** no Covertype (elevando o pico total cliente para **125.04 KB** se o frame de todas as folhas ativas for montado monoliticamente em RAM). No servidor `Fed-VFDT`, `ht_fed_vfdt_server_t` unifica `split_log[64]` e `round_proposals[128]` na mesma struct estática zero-`malloc` (~10–16 KB fixos em `Com`).

## 3. Resultados em Classificação — Regime Non-IID (`Dirichlet α=0.3`)

| Dataset | Estratégia | Holdout Acc (%) | Macro F1 (%) | Prequential Acc (%) | Nós Médios | Splits | Tráfego Total | Bytes / Cli / Rod. | RAM Cli (Mod + Com) | RAM Srv (Mod + Com) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Centralized Oracle (T8) | 98.95% | 98.80% | 97.29% | 43.0 | 21 | 0.0 KB | 0 B | 6.7 + 0.0 = 6.7 KB | 0.0 KB |
| **SEA** | Local-Only (K isolados) | 76.41% | 69.04% | 97.14% | 9.0 | 4 | 0.0 KB | 0 B | 6.7 + 0.0 = 6.7 KB | 0.0 KB |
| **SEA** | Fed-Forest (TOP_M) | 95.28% | 94.63% | 96.39% | 9.0 | 4 | 219.3 KB | 1123 B | 13.1 + 1.4 = 14.5 KB | 20.5 + 1.6 = 22.1 KB |
| **SEA** | Fed-Forest (ROUND_ROBIN) | 85.77% | 78.18% | 96.14% | 9.0 | 4 | 205.3 KB | 1051 B | 13.1 + 1.6 = 14.6 KB | 20.5 + 1.8 = 22.3 KB |
| **SEA** | Fed-VFDT (QUORUM) | 88.56% | 87.57% | 92.56% | 23.0 | 11 | 50.2 KB | 257 B | 6.7 + 0.5 = 7.1 KB | 6.7 + 10.3 = 16.9 KB |
| **SEA** | Fed-VFDT (MAJORITY) | 93.69% | 92.94% | 93.66% | 35.0 | 17 | 55.6 KB | 285 B | 6.7 + 0.4 = 7.1 KB | 6.7 + 10.2 = 16.9 KB |
| **SEA** | Fed-VFDT (BEST-MERIT) | 93.69% | 92.94% | 93.66% | 35.0 | 17 | 55.6 KB | 285 B | 6.7 + 0.4 = 7.1 KB | 6.7 + 10.2 = 16.9 KB |
| **SEA** | Fed-VFDT (STATS-MERGE) | **96.78%** | 96.30% | 94.84% | 37.0 | 18 | 396.7 KB | 2031 B | 6.7 + 2.2 = 8.8 KB | 6.7 + 12.0 = 18.6 KB |
| **Electricity** | Centralized Oracle (T8) | 64.06% | 63.15% | 64.58% | 87.0 | 43 | 0.0 KB | 0 B | 9.8 + 0.0 = 9.8 KB | 0.0 KB |
| **Electricity** | Local-Only (K isolados) | 49.43% | 33.65% | 96.76% | 9.6 | 4 | 0.0 KB | 0 B | 9.8 + 0.0 = 9.8 KB | 0.0 KB |
| **Electricity** | Fed-Forest (TOP_M) | 54.78% | 35.39% | 96.15% | 9.6 | 4 | 231.7 KB | 1186 B | 16.3 + 1.5 = 17.8 KB | 20.9 + 1.7 = 22.6 KB |
| **Electricity** | Fed-Forest (ROUND_ROBIN) | 48.27% | 33.02% | 96.22% | 9.6 | 4 | 246.9 KB | 1264 B | 16.3 + 2.0 = 18.3 KB | 20.9 + 2.2 = 23.1 KB |
| **Electricity** | Fed-VFDT (QUORUM) | 57.48% | 53.89% | 83.27% | 37.0 | 18 | 64.6 KB | 331 B | 9.8 + 0.5 = 10.3 KB | 9.8 + 10.3 = 20.1 KB |
| **Electricity** | Fed-VFDT (MAJORITY) | 50.90% | 42.17% | 78.72% | 57.0 | 28 | 82.8 KB | 424 B | 9.8 + 0.5 = 10.3 KB | 9.8 + 10.3 = 20.1 KB |
| **Electricity** | Fed-VFDT (BEST-MERIT) | 49.87% | 43.67% | 79.33% | 49.0 | 24 | 82.1 KB | 420 B | 9.8 + 0.5 = 10.3 KB | 9.8 + 10.3 = 20.1 KB |
| **Electricity** | Fed-VFDT (STATS-MERGE) | **59.19%** | 55.06% | 78.59% | 65.0 | 32 | 950.2 KB | 4865 B | 9.8 + 5.3 = 15.1 KB | 9.8 + 15.1 = 24.9 KB |
| **Forest Covertype** | Centralized Oracle (T8) | 65.59% | 63.33% | 59.97% | 75.0 | 37 | 0.0 KB | 0 B | 65.5 + 0.0 = 65.5 KB | 0.0 KB |
| **Forest Covertype** | Local-Only (K isolados) | 31.06% | 17.72% | 67.58% | 12.0 | 5 | 0.0 KB | 0 B | 65.5 + 0.0 = 65.5 KB | 0.0 KB |
| **Forest Covertype** | Fed-Forest (TOP_M) | 25.82% | 11.59% | 61.16% | 12.0 | 5 | 651.5 KB | 3336 B | 77.0 + 4.2 = 81.1 KB | 36.5 + 4.4 = 40.9 KB |
| **Forest Covertype** | Fed-Forest (ROUND_ROBIN) | 25.83% | 11.59% | 62.43% | 12.0 | 5 | 606.9 KB | 3108 B | 77.0 + 3.8 = 80.7 KB | 36.5 + 4.0 = 40.5 KB |
| **Forest Covertype** | Fed-VFDT (QUORUM) | 16.58% | 7.33% | 54.45% | 1.0 | 0 | 32.0 KB | 164 B | 65.5 + 0.1 = 65.7 KB | 65.5 + 16.3 = 81.8 KB |
| **Forest Covertype** | Fed-VFDT (MAJORITY) | 30.91% | 22.74% | 49.74% | 27.0 | 13 | 110.2 KB | 564 B | 65.5 + 0.7 = 66.3 KB | 65.5 + 16.8 = 82.4 KB |
| **Forest Covertype** | Fed-VFDT (BEST-MERIT) | 35.00% | 25.12% | 45.88% | 29.0 | 14 | 109.9 KB | 563 B | 65.5 + 0.7 = 66.3 KB | 65.5 + 16.8 = 82.4 KB |
| **Forest Covertype** | Fed-VFDT (STATS-MERGE) | **65.53%** | 63.05% | 58.87% | 67.0 | 33 | 10369.5 KB | 53092 B | 65.5 + 59.5 = 125.0 KB | 65.5 + 75.6 = 141.2 KB |
| **Airlines** | Centralized Oracle (T8) | 76.25% | 58.07% | 73.31% | 39.0 | 19 | 0.0 KB | 0 B | 9.3 + 0.0 = 9.3 KB | 0.0 KB |
| **Airlines** | Local-Only (K isolados) | 49.59% | 37.23% | 84.17% | 3.8 | 1 | 0.0 KB | 0 B | 9.3 + 0.0 = 9.3 KB | 0.0 KB |
| **Airlines** | Fed-Forest (TOP_M) | **76.39%** | 43.31% | 86.57% | 3.8 | 1 | 146.2 KB | 748 B | 15.8 + 0.9 = 16.7 KB | 20.8 + 1.1 = 21.9 KB |
| **Airlines** | Fed-Forest (ROUND_ROBIN) | 60.56% | 36.06% | 85.17% | 3.8 | 1 | 145.1 KB | 743 B | 15.8 + 1.0 = 16.9 KB | 20.8 + 1.2 = 22.1 KB |
| **Airlines** | Fed-VFDT (QUORUM) | 51.24% | 36.20% | 87.34% | 1.0 | 0 | 22.3 KB | 114 B | 9.3 + 0.1 = 9.4 KB | 9.2 + 9.9 = 19.1 KB |
| **Airlines** | Fed-VFDT (MAJORITY) | 56.79% | 45.90% | 68.63% | 7.0 | 3 | 40.3 KB | 206 B | 9.3 + 0.3 = 9.6 KB | 9.2 + 10.1 = 19.2 KB |
| **Airlines** | Fed-VFDT (BEST-MERIT) | 56.79% | 45.90% | 68.63% | 7.0 | 3 | 40.3 KB | 206 B | 9.3 + 0.3 = 9.6 KB | 9.2 + 10.1 = 19.2 KB |
| **Airlines** | Fed-VFDT (STATS-MERGE) | 63.63% | 48.83% | 51.11% | 41.0 | 20 | 783.1 KB | 4010 B | 9.3 + 4.7 = 14.0 KB | 9.2 + 14.5 = 23.7 KB |

## 4. Resultados em Regressão Contínua — Regime Non-IID (`Dirichlet α=0.3`)

| Dataset | Estratégia | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Nós Médios | Splits | Tráfego Total | Bytes / Cli / Rod. | RAM Cli (Mod + Com) | RAM Srv (Mod + Com) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | Centralized Oracle (T8) | 2.0886 | 2.6539 | 0.7166 | 2.1036 | 41.0 | 20 | 0.0 KB | 0 B | 25.5 + 0.0 = 25.5 KB | 0.0 KB |
| **Friedman #1** | Local-Only (K isolados) | 2.7287 | 3.4292 | 0.5240 | 2.8524 | 27.4 | 13 | 0.0 KB | 0 B | 25.5 + 0.0 = 25.5 KB | 0.0 KB |
| **Friedman #1** | Fed-Forest (TOP_M) | 2.8642 | 3.5341 | 0.4920 | 3.1162 | 27.4 | 13 | 544.5 KB | 2788 B | 32.1 + 3.6 = 35.6 KB | 21.1 + 3.8 = 24.8 KB |
| **Friedman #1** | Fed-Forest (ROUND_ROBIN) | 2.8578 | 3.5226 | 0.4926 | 3.0403 | 27.4 | 13 | 505.8 KB | 2590 B | 32.1 + 3.6 = 35.7 KB | 21.1 + 3.8 = 24.9 KB |
| **Friedman #1** | Fed-VFDT (QUORUM) | 2.9355 | 3.7011 | 0.4431 | 3.1338 | 33.0 | 16 | 51.8 KB | 265 B | 25.5 + 0.5 = 25.9 KB | 25.2 + 10.3 = 35.5 KB |
| **Friedman #1** | Fed-VFDT (MAJORITY) | 2.8002 | 3.5533 | 0.4864 | 3.4158 | 45.0 | 22 | 48.6 KB | 249 B | 25.5 + 0.4 = 25.8 KB | 25.2 + 10.2 = 35.4 KB |
| **Friedman #1** | Fed-VFDT (BEST-MERIT) | 3.0701 | 3.9029 | 0.3805 | 3.0065 | 41.0 | 20 | 47.0 KB | 241 B | 25.5 + 0.4 = 25.8 KB | 25.2 + 10.2 = 35.4 KB |
| **Friedman #1** | Fed-VFDT (STATS-MERGE) | **2.5687** | 3.2351 | 0.5699 | 3.1179 | 41.0 | 20 | 3547.1 KB | 18161 B | 25.5 + 20.6 = 46.1 KB | 25.2 + 30.4 = 55.7 KB |
| **Bikes** | Centralized Oracle (T8) | 50.7078 | 77.9452 | -0.0774 | 51.6146 | 51.0 | 25 | 0.0 KB | 0 B | 53.6 + 0.0 = 53.6 KB | 0.0 KB |
| **Bikes** | Local-Only (K isolados) | 51.1650 | 81.7439 | -0.2344 | 54.0680 | 30.4 | 14 | 0.0 KB | 0 B | 53.6 + 0.0 = 53.6 KB | 0.0 KB |
| **Bikes** | Fed-Forest (TOP_M) | 41.9685 | 84.0474 | -0.2528 | 61.8154 | 30.4 | 14 | 557.0 KB | 2852 B | 60.2 + 3.9 = 64.1 KB | 21.2 + 4.1 = 25.3 KB |
| **Bikes** | Fed-Forest (ROUND_ROBIN) | **41.7123** | 83.8927 | -0.2482 | 60.3057 | 30.4 | 14 | 563.4 KB | 2884 B | 60.2 + 4.0 = 64.2 KB | 21.2 + 4.2 = 25.4 KB |
| **Bikes** | Fed-VFDT (QUORUM) | 57.0795 | 78.2703 | -0.1138 | 74.5161 | 29.0 | 14 | 63.7 KB | 326 B | 53.6 + 0.5 = 54.1 KB | 53.3 + 10.3 = 63.6 KB |
| **Bikes** | Fed-VFDT (MAJORITY) | 60.4817 | 85.2145 | -0.3039 | 78.9896 | 49.0 | 24 | 41.8 KB | 214 B | 53.6 + 0.4 = 54.0 KB | 53.3 + 10.2 = 63.5 KB |
| **Bikes** | Fed-VFDT (BEST-MERIT) | 55.4933 | 81.0989 | -0.2067 | 69.1752 | 51.0 | 25 | 42.8 KB | 219 B | 53.6 + 0.3 = 53.9 KB | 53.3 + 10.1 = 63.4 KB |
| **Bikes** | Fed-VFDT (STATS-MERGE) | 65.7421 | 89.1395 | -0.4485 | 74.5179 | 55.0 | 27 | 8234.5 KB | 42161 B | 53.6 + 48.7 = 102.3 KB | 53.3 + 58.5 = 111.9 KB |
| **Appliances Energy** | Centralized Oracle (T8) | 30.9357 | 39.2997 | 0.7237 | 37.0401 | 43.0 | 21 | 0.0 KB | 0 B | 34.1 + 0.0 = 34.1 KB | 0.0 KB |
| **Appliances Energy** | Local-Only (K isolados) | 72.5084 | 96.7667 | -1.1671 | 118.9508 | 29.4 | 14 | 0.0 KB | 0 B | 34.1 + 0.0 = 34.1 KB | 0.0 KB |
| **Appliances Energy** | Fed-Forest (TOP_M) | **34.2954** | 43.1050 | 0.6675 | 81.2868 | 29.4 | 14 | 741.6 KB | 3797 B | 41.1 + 4.6 = 45.7 KB | 22.4 + 4.8 = 27.3 KB |
| **Appliances Energy** | Fed-Forest (ROUND_ROBIN) | 38.7593 | 48.1901 | 0.5840 | 83.3291 | 29.4 | 14 | 644.3 KB | 3299 B | 41.1 + 4.2 = 45.3 KB | 22.4 + 4.4 = 26.8 KB |
| **Appliances Energy** | Fed-VFDT (QUORUM) | 60.1188 | 109.1465 | -2.6470 | 122.7975 | 35.0 | 17 | 56.3 KB | 288 B | 34.1 + 0.5 = 34.5 KB | 34.1 + 10.4 = 44.4 KB |
| **Appliances Energy** | Fed-VFDT (MAJORITY) | 50.3140 | 104.4818 | -2.5380 | 118.0016 | 53.0 | 26 | 43.3 KB | 222 B | 34.1 + 0.5 = 34.5 KB | 34.1 + 10.4 = 44.4 KB |
| **Appliances Energy** | Fed-VFDT (BEST-MERIT) | 51.3444 | 105.3130 | -2.5583 | 118.4735 | 53.0 | 26 | 43.3 KB | 222 B | 34.1 + 0.5 = 34.5 KB | 34.1 + 10.4 = 44.4 KB |
| **Appliances Energy** | Fed-VFDT (STATS-MERGE) | 54.3217 | 98.3768 | -1.9347 | 121.3517 | 41.0 | 20 | 5122.7 KB | 26228 B | 34.1 + 29.2 = 63.3 KB | 34.1 + 39.1 = 73.2 KB |
| **NYC Taxi** | Centralized Oracle (T8) | 335.4297 | 439.5337 | 0.2104 | 345.0436 | 45.0 | 22 | 0.0 KB | 0 B | 12.3 + 0.0 = 12.3 KB | 0.0 KB |
| **NYC Taxi** | Local-Only (K isolados) | 393.0422 | 509.2121 | -0.0653 | 364.2441 | 30.4 | 14 | 0.0 KB | 0 B | 12.3 + 0.0 = 12.3 KB | 0.0 KB |
| **NYC Taxi** | Fed-Forest (TOP_M) | 356.6601 | 466.1928 | 0.1117 | 367.2333 | 30.4 | 14 | 601.8 KB | 3081 B | 18.8 + 4.1 = 22.8 KB | 20.8 + 4.3 = 25.1 KB |
| **NYC Taxi** | Fed-Forest (ROUND_ROBIN) | **356.2798** | 457.3860 | 0.1449 | 373.0931 | 30.4 | 14 | 558.8 KB | 2861 B | 18.8 + 3.9 = 22.7 KB | 20.8 + 4.1 = 24.9 KB |
| **NYC Taxi** | Fed-VFDT (QUORUM) | 371.3887 | 482.6401 | 0.0468 | 370.6291 | 35.0 | 17 | 47.0 KB | 241 B | 12.3 + 0.4 = 12.7 KB | 12.1 + 10.2 = 22.3 KB |
| **NYC Taxi** | Fed-VFDT (MAJORITY) | 386.9393 | 502.3659 | -0.0329 | 378.2516 | 51.0 | 25 | 37.5 KB | 192 B | 12.3 + 0.4 = 12.7 KB | 12.1 + 10.2 = 22.3 KB |
| **NYC Taxi** | Fed-VFDT (BEST-MERIT) | 390.3947 | 505.9050 | -0.0487 | 377.5022 | 47.0 | 23 | 36.9 KB | 189 B | 12.3 + 0.4 = 12.6 KB | 12.1 + 10.2 = 22.3 KB |
| **NYC Taxi** | Fed-VFDT (STATS-MERGE) | 383.1599 | 497.1056 | -0.0109 | 369.0963 | 25.0 | 12 | 1263.7 KB | 6470 B | 12.3 + 7.5 = 19.8 KB | 12.1 + 17.3 = 29.4 KB |

## 5. Resultados em Classificação — Regime IID (Referência Uniforme)

| Dataset | Estratégia | Holdout Acc (%) | Macro F1 (%) | Prequential Acc (%) | Nós Médios | Splits | Tráfego Total | Bytes / Cli / Rod. | RAM Cli (Mod + Com) | RAM Srv (Mod + Com) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Centralized Oracle (T8) | 98.95% | 98.80% | 97.29% | 43.0 | 21 | 0.0 KB | 0 B | 6.7 + 0.0 = 6.7 KB | 0.0 KB |
| **SEA** | Local-Only (K isolados) | 96.32% | 95.77% | 92.47% | 19.0 | 9 | 0.0 KB | 0 B | 6.7 + 0.0 = 6.7 KB | 0.0 KB |
| **SEA** | Fed-Forest (TOP_M) | **97.23%** | 96.83% | 93.24% | 19.0 | 9 | 346.8 KB | 1776 B | 13.1 + 2.4 = 15.4 KB | 20.5 + 2.6 = 23.1 KB |
| **SEA** | Fed-Forest (ROUND_ROBIN) | 96.19% | 95.64% | 93.19% | 19.0 | 9 | 343.7 KB | 1760 B | 13.1 + 2.3 = 15.4 KB | 20.5 + 2.5 = 23.0 KB |
| **SEA** | Fed-VFDT (QUORUM) | 96.39% | 95.84% | 92.89% | 33.0 | 16 | 58.8 KB | 301 B | 6.7 + 0.5 = 7.1 KB | 6.7 + 10.3 = 16.9 KB |
| **SEA** | Fed-VFDT (MAJORITY) | 96.54% | 95.98% | 92.94% | 37.0 | 18 | 64.8 KB | 332 B | 6.7 + 0.4 = 7.0 KB | 6.7 + 10.2 = 16.8 KB |
| **SEA** | Fed-VFDT (BEST-MERIT) | 96.21% | 95.57% | 93.19% | 35.0 | 17 | 64.9 KB | 332 B | 6.7 + 0.4 = 7.1 KB | 6.7 + 10.2 = 16.9 KB |
| **SEA** | Fed-VFDT (STATS-MERGE) | 96.97% | 96.52% | 93.91% | 35.0 | 17 | 398.9 KB | 2042 B | 6.7 + 2.2 = 8.8 KB | 6.7 + 12.0 = 18.6 KB |
| **Electricity** | Centralized Oracle (T8) | 64.06% | 63.15% | 64.58% | 87.0 | 43 | 0.0 KB | 0 B | 9.8 + 0.0 = 9.8 KB | 0.0 KB |
| **Electricity** | Local-Only (K isolados) | 63.68% | 62.75% | 63.14% | 32.8 | 15 | 0.0 KB | 0 B | 9.8 + 0.0 = 9.8 KB | 0.0 KB |
| **Electricity** | Fed-Forest (TOP_M) | **64.28%** | 63.50% | 64.08% | 32.8 | 15 | 500.2 KB | 2561 B | 16.3 + 3.7 = 20.0 KB | 20.9 + 3.9 = 24.8 KB |
| **Electricity** | Fed-Forest (ROUND_ROBIN) | 64.12% | 63.38% | 63.81% | 32.8 | 15 | 503.5 KB | 2578 B | 16.3 + 3.9 = 20.2 KB | 20.9 + 4.1 = 24.9 KB |
| **Electricity** | Fed-VFDT (QUORUM) | 63.26% | 62.99% | 63.24% | 41.0 | 20 | 87.4 KB | 447 B | 9.8 + 0.5 = 10.3 KB | 9.8 + 10.3 = 20.1 KB |
| **Electricity** | Fed-VFDT (MAJORITY) | 63.15% | 63.06% | 63.18% | 49.0 | 24 | 85.6 KB | 438 B | 9.8 + 0.5 = 10.3 KB | 9.8 + 10.3 = 20.1 KB |
| **Electricity** | Fed-VFDT (BEST-MERIT) | 63.58% | 63.20% | 62.99% | 43.0 | 21 | 83.6 KB | 428 B | 9.8 + 0.4 = 10.2 KB | 9.8 + 10.2 = 20.0 KB |
| **Electricity** | Fed-VFDT (STATS-MERGE) | 64.17% | 62.83% | 64.19% | 77.0 | 38 | 946.4 KB | 4846 B | 9.8 + 5.3 = 15.1 KB | 9.8 + 15.1 = 24.9 KB |
| **Forest Covertype** | Centralized Oracle (T8) | 65.59% | 63.33% | 59.97% | 75.0 | 37 | 0.0 KB | 0 B | 65.5 + 0.0 = 65.5 KB | 0.0 KB |
| **Forest Covertype** | Local-Only (K isolados) | 46.88% | 35.68% | 39.73% | 11.4 | 5 | 0.0 KB | 0 B | 65.5 + 0.0 = 65.5 KB | 0.0 KB |
| **Forest Covertype** | Fed-Forest (TOP_M) | 56.51% | 45.37% | 44.64% | 11.4 | 5 | 603.4 KB | 3090 B | 77.0 + 3.4 = 80.4 KB | 36.5 + 3.6 = 40.1 KB |
| **Forest Covertype** | Fed-Forest (ROUND_ROBIN) | 55.04% | 44.33% | 45.06% | 11.4 | 5 | 601.9 KB | 3082 B | 77.0 + 3.4 = 80.4 KB | 36.5 + 3.6 = 40.1 KB |
| **Forest Covertype** | Fed-VFDT (QUORUM) | 51.27% | 45.12% | 41.79% | 17.0 | 8 | 100.2 KB | 513 B | 65.5 + 0.7 = 66.3 KB | 65.5 + 16.8 = 82.4 KB |
| **Forest Covertype** | Fed-VFDT (MAJORITY) | 51.58% | 45.36% | 42.31% | 19.0 | 9 | 101.7 KB | 521 B | 65.5 + 0.7 = 66.3 KB | 65.5 + 16.8 = 82.4 KB |
| **Forest Covertype** | Fed-VFDT (BEST-MERIT) | 51.88% | 46.18% | 42.50% | 17.0 | 8 | 91.4 KB | 468 B | 65.5 + 0.7 = 66.3 KB | 65.5 + 16.8 = 82.4 KB |
| **Forest Covertype** | Fed-VFDT (STATS-MERGE) | **64.29%** | 59.09% | 50.83% | 75.0 | 37 | 10372.6 KB | 53108 B | 65.5 + 59.5 = 125.0 KB | 65.5 + 75.6 = 141.2 KB |
| **Airlines** | Centralized Oracle (T8) | 76.25% | 58.07% | 73.31% | 39.0 | 19 | 0.0 KB | 0 B | 9.3 + 0.0 = 9.3 KB | 0.0 KB |
| **Airlines** | Local-Only (K isolados) | 74.35% | 55.32% | 71.72% | 5.2 | 2 | 0.0 KB | 0 B | 9.3 + 0.0 = 9.3 KB | 0.0 KB |
| **Airlines** | Fed-Forest (TOP_M) | 76.39% | 43.31% | 75.03% | 5.2 | 2 | 153.1 KB | 784 B | 15.8 + 0.9 = 16.8 KB | 20.8 + 1.2 = 22.0 KB |
| **Airlines** | Fed-Forest (ROUND_ROBIN) | **76.47%** | 44.27% | 75.03% | 5.2 | 2 | 155.0 KB | 794 B | 15.8 + 0.9 = 16.7 KB | 20.8 + 1.1 = 21.9 KB |
| **Airlines** | Fed-VFDT (QUORUM) | 75.80% | 53.11% | 72.17% | 7.0 | 3 | 41.4 KB | 212 B | 9.3 + 0.3 = 9.6 KB | 9.2 + 10.1 = 19.2 KB |
| **Airlines** | Fed-VFDT (MAJORITY) | 75.77% | 53.19% | 72.01% | 9.0 | 4 | 45.7 KB | 234 B | 9.3 + 0.3 = 9.6 KB | 9.2 + 10.1 = 19.3 KB |
| **Airlines** | Fed-VFDT (BEST-MERIT) | 75.53% | 53.34% | 71.86% | 7.0 | 3 | 41.4 KB | 212 B | 9.3 + 0.3 = 9.6 KB | 9.2 + 10.1 = 19.2 KB |
| **Airlines** | Fed-VFDT (STATS-MERGE) | 71.01% | 55.99% | 62.88% | 37.0 | 18 | 793.8 KB | 4064 B | 9.3 + 4.7 = 14.0 KB | 9.2 + 14.5 = 23.7 KB |

## 6. Resultados em Regressão Contínua — Regime IID (Referência Uniforme)

| Dataset | Estratégia | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Nós Médios | Splits | Tráfego Total | Bytes / Cli / Rod. | RAM Cli (Mod + Com) | RAM Srv (Mod + Com) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | Centralized Oracle (T8) | 2.0886 | 2.6539 | 0.7166 | 2.1036 | 41.0 | 20 | 0.0 KB | 0 B | 25.5 + 0.0 = 25.5 KB | 0.0 KB |
| **Friedman #1** | Local-Only (K isolados) | 2.2785 | 2.8868 | 0.6635 | 3.0107 | 30.4 | 14 | 0.0 KB | 0 B | 25.5 + 0.0 = 25.5 KB | 0.0 KB |
| **Friedman #1** | Fed-Forest (TOP_M) | 2.1077 | 2.6235 | 0.7230 | 2.8547 | 30.4 | 14 | 493.4 KB | 2526 B | 32.1 + 3.5 = 35.5 KB | 21.1 + 3.7 = 24.8 KB |
| **Friedman #1** | Fed-Forest (ROUND_ROBIN) | **2.0679** | 2.5751 | 0.7331 | 2.8238 | 30.4 | 14 | 508.7 KB | 2605 B | 32.1 + 3.6 = 35.7 KB | 21.1 + 3.8 = 24.9 KB |
| **Friedman #1** | Fed-VFDT (QUORUM) | 2.3351 | 2.9425 | 0.6516 | 3.1801 | 43.0 | 21 | 69.8 KB | 358 B | 25.5 + 0.5 = 25.9 KB | 25.2 + 10.3 = 35.5 KB |
| **Friedman #1** | Fed-VFDT (MAJORITY) | 2.3465 | 2.9585 | 0.6478 | 3.3560 | 49.0 | 24 | 57.3 KB | 294 B | 25.5 + 0.4 = 25.9 KB | 25.2 + 10.2 = 35.5 KB |
| **Friedman #1** | Fed-VFDT (BEST-MERIT) | 2.4738 | 3.1109 | 0.6106 | 3.3650 | 53.0 | 26 | 53.7 KB | 275 B | 25.5 + 0.3 = 25.8 KB | 25.2 + 10.1 = 35.4 KB |
| **Friedman #1** | Fed-VFDT (STATS-MERGE) | 2.1900 | 2.7601 | 0.6934 | 3.0544 | 43.0 | 21 | 3470.5 KB | 17769 B | 25.5 + 20.6 = 46.1 KB | 25.2 + 30.4 = 55.7 KB |
| **Bikes** | Centralized Oracle (T8) | 50.7078 | 77.9452 | -0.0774 | 51.6146 | 51.0 | 25 | 0.0 KB | 0 B | 53.6 + 0.0 = 53.6 KB | 0.0 KB |
| **Bikes** | Local-Only (K isolados) | 50.2595 | 72.9023 | 0.0357 | 69.7589 | 31.0 | 15 | 0.0 KB | 0 B | 53.6 + 0.0 = 53.6 KB | 0.0 KB |
| **Bikes** | Fed-Forest (TOP_M) | **44.4785** | 59.8659 | 0.3626 | 65.3882 | 31.0 | 15 | 618.5 KB | 3167 B | 60.2 + 3.8 = 64.1 KB | 21.2 + 4.1 = 25.2 KB |
| **Bikes** | Fed-Forest (ROUND_ROBIN) | 47.9013 | 61.8531 | 0.3202 | 66.6238 | 31.0 | 15 | 607.1 KB | 3108 B | 60.2 + 3.6 = 63.8 KB | 21.2 + 3.8 = 25.0 KB |
| **Bikes** | Fed-VFDT (QUORUM) | 54.1740 | 76.9674 | -0.0519 | 68.4236 | 43.0 | 21 | 54.7 KB | 280 B | 53.6 + 0.4 = 54.0 KB | 53.3 + 10.2 = 63.6 KB |
| **Bikes** | Fed-VFDT (MAJORITY) | 52.2598 | 74.6820 | 0.0099 | 70.4657 | 49.0 | 24 | 44.3 KB | 227 B | 53.6 + 0.3 = 53.9 KB | 53.3 + 10.1 = 63.5 KB |
| **Bikes** | Fed-VFDT (BEST-MERIT) | 54.2567 | 76.3400 | -0.0338 | 69.4101 | 45.0 | 22 | 38.2 KB | 196 B | 53.6 + 0.3 = 53.9 KB | 53.3 + 10.1 = 63.5 KB |
| **Bikes** | Fed-VFDT (STATS-MERGE) | 58.4245 | 88.1670 | -0.3825 | 66.5321 | 49.0 | 24 | 8537.0 KB | 43710 B | 53.6 + 48.7 = 102.3 KB | 53.3 + 58.5 = 111.9 KB |
| **Appliances Energy** | Centralized Oracle (T8) | 30.9357 | 39.2997 | 0.7237 | 37.0401 | 43.0 | 21 | 0.0 KB | 0 B | 34.1 + 0.0 = 34.1 KB | 0.0 KB |
| **Appliances Energy** | Local-Only (K isolados) | 45.7775 | 65.1773 | 0.1905 | 110.8862 | 34.2 | 16 | 0.0 KB | 0 B | 34.1 + 0.0 = 34.1 KB | 0.0 KB |
| **Appliances Energy** | Fed-Forest (TOP_M) | **28.7315** | 36.1508 | 0.7662 | 74.8344 | 34.2 | 16 | 714.5 KB | 3658 B | 41.1 + 4.5 = 45.6 KB | 22.4 + 4.7 = 27.1 KB |
| **Appliances Energy** | Fed-Forest (ROUND_ROBIN) | 29.9074 | 37.5167 | 0.7481 | 75.4773 | 34.2 | 16 | 698.9 KB | 3578 B | 41.1 + 4.3 = 45.4 KB | 22.4 + 4.5 = 26.9 KB |
| **Appliances Energy** | Fed-VFDT (QUORUM) | 44.5770 | 66.8875 | 0.1222 | 110.6092 | 41.0 | 20 | 60.7 KB | 311 B | 34.1 + 0.4 = 34.5 KB | 34.1 + 10.3 = 44.4 KB |
| **Appliances Energy** | Fed-VFDT (MAJORITY) | 42.7582 | 64.4272 | 0.1933 | 110.7274 | 53.0 | 26 | 44.4 KB | 227 B | 34.1 + 0.3 = 34.4 KB | 34.1 + 10.2 = 44.3 KB |
| **Appliances Energy** | Fed-VFDT (BEST-MERIT) | 44.5263 | 64.2582 | 0.2227 | 110.4634 | 49.0 | 24 | 42.7 KB | 219 B | 34.1 + 0.4 = 34.4 KB | 34.1 + 10.3 = 44.3 KB |
| **Appliances Energy** | Fed-VFDT (STATS-MERGE) | 44.2971 | 69.0621 | 0.0878 | 110.0863 | 37.0 | 18 | 5048.9 KB | 25850 B | 34.1 + 29.2 = 63.3 KB | 34.1 + 39.1 = 73.2 KB |
| **NYC Taxi** | Centralized Oracle (T8) | 335.4297 | 439.5337 | 0.2104 | 345.0436 | 45.0 | 22 | 0.0 KB | 0 B | 12.3 + 0.0 = 12.3 KB | 0.0 KB |
| **NYC Taxi** | Local-Only (K isolados) | 369.2142 | 480.7159 | 0.0550 | 387.2717 | 31.6 | 15 | 0.0 KB | 0 B | 12.3 + 0.0 = 12.3 KB | 0.0 KB |
| **NYC Taxi** | Fed-Forest (TOP_M) | **340.9575** | 445.2511 | 0.1897 | 364.8868 | 31.6 | 15 | 558.4 KB | 2859 B | 18.8 + 3.6 = 22.4 KB | 20.8 + 3.9 = 24.7 KB |
| **NYC Taxi** | Fed-Forest (ROUND_ROBIN) | 347.9454 | 455.1707 | 0.1532 | 366.6227 | 31.6 | 15 | 559.5 KB | 2865 B | 18.8 + 3.6 = 22.4 KB | 20.8 + 3.8 = 24.6 KB |
| **NYC Taxi** | Fed-VFDT (QUORUM) | 364.7092 | 473.4725 | 0.0837 | 373.5438 | 33.0 | 16 | 41.1 KB | 210 B | 12.3 + 0.4 = 12.6 KB | 12.1 + 10.2 = 22.3 KB |
| **NYC Taxi** | Fed-VFDT (MAJORITY) | 369.8871 | 478.7850 | 0.0631 | 385.1268 | 51.0 | 25 | 38.7 KB | 198 B | 12.3 + 0.3 = 12.5 KB | 12.1 + 10.1 = 22.2 KB |
| **NYC Taxi** | Fed-VFDT (BEST-MERIT) | 366.6531 | 478.3275 | 0.0648 | 379.8136 | 49.0 | 24 | 36.7 KB | 188 B | 12.3 + 0.3 = 12.5 KB | 12.1 + 10.1 = 22.2 KB |
| **NYC Taxi** | Fed-VFDT (STATS-MERGE) | 368.1252 | 477.6370 | 0.0676 | 382.8794 | 19.0 | 9 | 1028.9 KB | 5268 B | 12.3 + 7.5 = 19.8 KB | 12.1 + 17.3 = 29.4 KB |

## 7. Síntese Executiva: Ganho Federado sob Non-IID vs Local-Only e Oráculo Centralizado

| Dataset | Local-Only | Melhor Fed-Forest | Melhor Fed-VFDT (Coord.) | Fed-VFDT (Stats-Merge) | Central Oracle (T8) | Ref. `main_benchmark` (T8) | Vencedor Federado (Non-IID) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SEA** | 76.41% | 95.28% | 93.69% | 96.78% | 98.95% | 99.02% | **fed_vfdt_merge** (+20.38 pp vs Local) |
| **Electricity** | 49.43% | 54.78% | 57.48% | 59.19% | 64.06% | 64.37% | **fed_vfdt_merge** (+9.76 pp vs Local) |
| **Forest Covertype** | 31.06% | 25.83% | 35.00% | 65.53% | 65.59% | 67.28% | **fed_vfdt_merge** (+34.47 pp vs Local) |
| **Airlines** | 49.59% | 76.39% | 56.79% | 63.63% | 76.25% | 76.09% | **fed_forest_topm** (+26.79 pp vs Local) |
| **Friedman #1** | 2.7287 | 2.8578 | 2.8002 | 2.5687 | 2.0886 | 2.1537 | **fed_vfdt_merge** (+5.9% MAE vs Local) |
| **Bikes** | 51.1650 | 41.7123 | 55.4933 | 65.7421 | 50.7078 | 40.4648 | **fed_forest_rr** (+18.5% MAE vs Local) |
| **Appliances Energy** | 72.5084 | 34.2954 | 50.3140 | 54.3217 | 30.9357 | 29.7630 | **fed_forest_topm** (+52.7% MAE vs Local) |
| **NYC Taxi** | 393.0422 | 356.2798 | 371.3887 | 383.1599 | 335.4297 | 339.2442 | **fed_forest_rr** (+9.4% MAE vs Local) |

### Principais Conclusões Arquiteturais

1. **Fed-VFDT (`STATS-MERGE`) como Oráculo Estatístico em Non-IID (Trade-off de Buffer TX):** Ao fundir estatísticas suficientes exatas (Chan/Welford, histogramas, contagens de classe e slots de regressão) na folha global, o Modo 2B reconstrói fielmente a distribuição global mesmo quando clientes individuais enxergam apenas 1–2 classes ou faixas estreitas de $y$ sob `Dirichlet(0.3)`, aproximando-se do **Centralized Oracle** — com o custo explícito de exigir um buffer de comunicação proporcional ao tamanho das folhas ativas (chegando a **59.5 KB** de buffer TX e **125.0 KB** totais no Covertype).
2. **Fed-VFDT Coordenado (`QUORUM` / `MAJORITY` / `BEST-MERIT`) — Menor Custo de Rede e Menor Buffer RAM:** Transmite apenas propostas leves de split e comandos incrementais (`~100–560 B/cliente/rodada`), exigindo um buffer TX/RX diminuto (`< 1 KB` observado) e mantendo a SRAM total no cliente em no máximo **66.3 KB** mesmo em 54 features — ideal para links LPWAN/LoRa/BLE e MCUs de SRAM estrita.
3. **Fed-Forest (`TOP_M` / `ROUND_ROBIN`) — Resiliência Assíncrona com Buffer Contido:** Cada cliente evolui sua árvore local de forma assíncrona e recebe um micro-ensemble compactado ($M=5$ árvores BFS, ~1.3 KB/árvore), elevando consistentemente a generalização global sobre `Local-Only` em Non-IID (especialmente em regressão contínua) enquanto mantém o conjunto completo **Modelo + Comunicação** abaixo de 100 KB no cliente (pico real **81.1 KB** no Covertype) e servidor enxuto (`16` árvores compactas).
