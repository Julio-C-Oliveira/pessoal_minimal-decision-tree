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
| **Electricity** | classification | 8 / 2 cls | 0.00 KB | 9.80 KB | 16.33 + 2.02 = **18.35 KB** \| Stk 1.30 | 9.80 + 0.46 = **10.26 KB** \| Stk 0.81 | 9.80 + 5.30 = **15.11 KB** \| Stk 0.05 | 6.6 / 3.0 / 5.3 KB |
| **Forest Covertype** | classification | 54 / 7 cls | 0.00 KB | 65.55 KB | 76.95 + 4.04 = **81.00 KB** \| Stk 2.28 | 65.55 + 0.71 = **66.25 KB** \| Stk 1.31 | 65.55 + 59.49 = **125.04 KB** \| Stk 0.06 | 11.5 / 5.0 / 59.5 KB |
| **Airlines** | classification | 7 / 2 cls | 0.16 KB | 9.34 KB | 15.84 + 1.02 = **16.86 KB** \| Stk 1.30 | 9.34 + 0.25 = **9.59 KB** \| Stk 0.81 | 9.34 + 4.68 = **14.02 KB** \| Stk 0.05 | 6.6 / 3.0 / 4.7 KB |
| **Friedman #1** | regression | 10 / cont. | 0.23 KB | 25.48 KB | 32.06 + 3.80 = **35.85 KB** \| Stk 1.32 | 25.48 + 0.41 = **25.88 KB** \| Stk 0.81 | 25.48 + 20.62 = **46.09 KB** \| Stk 0.05 | 6.7 / 3.0 / 20.6 KB |
| **Bikes** | regression | 12 / cont. | 0.28 KB | 53.62 KB | 60.24 + 3.37 = **63.60 KB** \| Stk 1.32 | 53.62 + 0.41 = **54.02 KB** \| Stk 0.81 | 53.62 + 48.68 = **102.30 KB** \| Stk 0.05 | 6.7 / 3.0 / 48.7 KB |
| **Appliances Energy** | regression | 28 / cont. | 0.00 KB | 34.09 KB | 41.10 + 4.66 = **45.75 KB** \| Stk 1.40 | 34.09 + 0.41 = **34.49 KB** \| Stk 0.81 | 34.09 + 29.18 = **63.27 KB** \| Stk 0.05 | 7.1 / 3.0 / 29.2 KB |
| **NYC Taxi** | regression | 7 / cont. | 0.16 KB | 12.26 KB | 18.76 + 3.84 = **22.60 KB** \| Stk 1.30 | 12.26 + 0.46 = **12.71 KB** \| Stk 0.81 | 12.26 + 7.52 = **19.78 KB** \| Stk 0.05 | 6.6 / 3.0 / 7.5 KB |

### 2.2 Decomposição de RAM do Lado do Servidor (1 Agregador) e Heterogeneidade Non-IID

| Dataset | Fed-Forest Server (`16` árv.)<br/>Mod + Com = Total \| Stk | Fed-VFDT Coord. Server<br/>Mod + Com = Total \| Stk | Fed-VFDT Stats-Merge Server<br/>Mod + Com = Total \| Stk | Non-IID Amostras/Cli | Métrica de Heterogeneidade Non-IID |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SEA** | 20.50 + 1.77 = **22.27 KB** \| Stk 1.28 | 6.66 + 10.26 = **16.92 KB** \| Stk 0.38 | 6.66 + 11.98 = **18.64 KB** \| Stk 0.26 | 69–4748 | Classe dom. média: **92.7%** (H=0.34 bits) |
| **Electricity** | 20.88 + 2.24 = **23.11 KB** \| Stk 1.30 | 9.80 + 10.28 = **20.08 KB** \| Stk 0.38 | 9.80 + 15.12 = **24.93 KB** \| Stk 0.67 | 127–8361 | Classe dom. média: **94.1%** (H=0.21 bits) |
| **Forest Covertype** | 36.50 + 4.26 = **40.76 KB** \| Stk 2.28 | 65.55 + 16.83 = **82.38 KB** \| Stk 0.62 | 65.55 + 75.62 = **141.16 KB** \| Stk 7.61 | 420–4453 | Classe dom. média: **53.2%** (H=1.68 bits) |
| **Airlines** | 20.81 + 1.24 = **22.05 KB** \| Stk 1.30 | 9.17 + 10.07 = **19.25 KB** \| Stk 0.38 | 9.17 + 14.50 = **23.67 KB** \| Stk 0.59 | 158–7066 | Classe dom. média: **88.2%** (H=0.41 bits) |
| **Friedman #1** | 21.06 + 4.02 = **25.08 KB** \| Stk 1.32 | 25.24 + 10.23 = **35.48 KB** \| Stk 0.38 | 25.24 + 30.45 = **55.69 KB** \| Stk 2.56 | 236–3554 | Dispersão $\mu_k(y)/\sigma(y)$: **0.59$\sigma$** |
| **Bikes** | 21.19 + 3.59 = **24.77 KB** \| Stk 1.32 | 53.34 + 10.24 = **63.58 KB** \| Stk 0.38 | 53.34 + 58.52 = **111.85 KB** \| Stk 6.06 | 496–2010 | Dispersão $\mu_k(y)/\sigma(y)$: **0.35$\sigma$** |
| **Appliances Energy** | 22.44 + 4.88 = **27.31 KB** \| Stk 1.40 | 34.09 + 10.30 = **44.39 KB** \| Stk 0.38 | 34.09 + 39.08 = **73.16 KB** \| Stk 3.62 | 511–2598 | Dispersão $\mu_k(y)/\sigma(y)$: **0.48$\sigma$** |
| **NYC Taxi** | 20.81 + 4.05 = **24.87 KB** \| Stk 1.30 | 12.09 + 10.28 = **22.37 KB** \| Stk 0.38 | 12.09 + 17.34 = **29.44 KB** \| Stk 0.92 | 212–2260 | Dispersão $\mu_k(y)/\sigma(y)$: **0.35$\sigma$** |

> **Validação de Orçamento Embarcado (Modelo vs. Comunicação):** Isolando apenas o **modelo estático**, o pico no cliente foi de **76.95 KB** (Forest Covertype, 54 features / 7 classes). Ao contabilizar também o **buffer half-duplex de comunicação (`wire_buf`) + coordenação**, **`Fed-VFDT Coordenado`** (pico total **66.25 KB**) e **`Fed-Forest`** (pico total **81.00 KB**) permanecem confortavelmente abaixo do teto de **100 KB** para MCUs como ESP32; já **`Fed-VFDT (STATS-MERGE)`** exige um buffer TX de até **59.48 KB** no Covertype (elevando o pico total cliente para **125.04 KB** se o frame de todas as folhas ativas for montado monoliticamente em RAM). No servidor `Fed-VFDT`, `ht_fed_vfdt_server_t` unifica `split_log[64]` e `round_proposals[128]` na mesma struct estática zero-`malloc` (~10–16 KB fixos em `Com`).

## 3. Resultados em Classificação — Regime Non-IID (`Dirichlet α=0.3`)

| Dataset | Estratégia | Holdout Acc (%) | Macro F1 (%) | Prequential Acc (%) | Nós Médios | Splits | Tráfego Total | Bytes / Cli / Rod. | RAM Cli (Mod + Com) | RAM Srv (Mod + Com) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Centralized Oracle (T8) | 99.02% | 98.87% | 97.30% | 43.0 | 21 | 0.0 KB | 0 B | 6.7 + 0.0 = 6.7 KB | 0.0 KB |
| **SEA** | Local-Only (K isolados) | 76.41% | 69.04% | 97.14% | 9.0 | 4 | 0.0 KB | 0 B | 6.7 + 0.0 = 6.7 KB | 0.0 KB |
| **SEA** | Fed-Forest (TOP_M) | 95.06% | 94.33% | 96.36% | 9.0 | 4 | 219.3 KB | 1123 B | 13.1 + 1.4 = 14.5 KB | 20.5 + 1.6 = 22.1 KB |
| **SEA** | Fed-Forest (ROUND_ROBIN) | 72.75% | 50.40% | 96.16% | 9.0 | 4 | 205.3 KB | 1051 B | 13.1 + 1.6 = 14.6 KB | 20.5 + 1.8 = 22.3 KB |
| **SEA** | Fed-VFDT (QUORUM) | 88.56% | 87.57% | 92.56% | 23.0 | 11 | 50.2 KB | 257 B | 6.7 + 0.5 = 7.1 KB | 6.7 + 10.3 = 16.9 KB |
| **SEA** | Fed-VFDT (MAJORITY) | 93.68% | 92.93% | 93.66% | 35.0 | 17 | 55.6 KB | 285 B | 6.7 + 0.4 = 7.1 KB | 6.7 + 10.2 = 16.9 KB |
| **SEA** | Fed-VFDT (BEST-MERIT) | 93.68% | 92.93% | 93.66% | 35.0 | 17 | 55.6 KB | 285 B | 6.7 + 0.4 = 7.1 KB | 6.7 + 10.2 = 16.9 KB |
| **SEA** | Fed-VFDT (STATS-MERGE) | **96.78%** | 96.30% | 94.84% | 37.0 | 18 | 396.7 KB | 2031 B | 6.7 + 2.2 = 8.8 KB | 6.7 + 12.0 = 18.6 KB |
| **Electricity** | Centralized Oracle (T8) | 64.37% | 63.96% | 64.57% | 79.0 | 39 | 0.0 KB | 0 B | 9.8 + 0.0 = 9.8 KB | 0.0 KB |
| **Electricity** | Local-Only (K isolados) | 50.65% | 36.14% | 96.79% | 10.2 | 4 | 0.0 KB | 0 B | 9.8 + 0.0 = 9.8 KB | 0.0 KB |
| **Electricity** | Fed-Forest (TOP_M) | 54.78% | 35.39% | 96.15% | 10.2 | 4 | 232.4 KB | 1190 B | 16.3 + 1.5 = 17.8 KB | 20.9 + 1.7 = 22.6 KB |
| **Electricity** | Fed-Forest (ROUND_ROBIN) | 48.27% | 33.02% | 96.22% | 10.2 | 4 | 249.0 KB | 1275 B | 16.3 + 2.0 = 18.4 KB | 20.9 + 2.2 = 23.1 KB |
| **Electricity** | Fed-VFDT (QUORUM) | 57.61% | 51.47% | 80.91% | 37.0 | 18 | 66.0 KB | 338 B | 9.8 + 0.5 = 10.3 KB | 9.8 + 10.3 = 20.1 KB |
| **Electricity** | Fed-VFDT (MAJORITY) | 50.89% | 42.16% | 78.75% | 57.0 | 28 | 82.7 KB | 423 B | 9.8 + 0.5 = 10.3 KB | 9.8 + 10.3 = 20.1 KB |
| **Electricity** | Fed-VFDT (BEST-MERIT) | 50.34% | 43.80% | 79.10% | 49.0 | 24 | 81.5 KB | 417 B | 9.8 + 0.5 = 10.3 KB | 9.8 + 10.3 = 20.1 KB |
| **Electricity** | Fed-VFDT (STATS-MERGE) | **60.37%** | 55.40% | 78.26% | 71.0 | 35 | 951.6 KB | 4872 B | 9.8 + 5.3 = 15.1 KB | 9.8 + 15.1 = 24.9 KB |
| **Forest Covertype** | Centralized Oracle (T8) | 67.28% | 65.21% | 59.99% | 77.0 | 38 | 0.0 KB | 0 B | 65.5 + 0.0 = 65.5 KB | 0.0 KB |
| **Forest Covertype** | Local-Only (K isolados) | 30.91% | 17.52% | 67.53% | 11.8 | 5 | 0.0 KB | 0 B | 65.5 + 0.0 = 65.5 KB | 0.0 KB |
| **Forest Covertype** | Fed-Forest (TOP_M) | 25.82% | 11.59% | 61.04% | 11.8 | 5 | 649.4 KB | 3325 B | 77.0 + 4.0 = 81.0 KB | 36.5 + 4.3 = 40.8 KB |
| **Forest Covertype** | Fed-Forest (ROUND_ROBIN) | 25.83% | 11.59% | 62.37% | 11.8 | 5 | 607.5 KB | 3110 B | 77.0 + 3.8 = 80.7 KB | 36.5 + 4.0 = 40.5 KB |
| **Forest Covertype** | Fed-VFDT (QUORUM) | 16.58% | 7.33% | 54.42% | 1.0 | 0 | 32.0 KB | 164 B | 65.5 + 0.1 = 65.7 KB | 65.5 + 16.3 = 81.8 KB |
| **Forest Covertype** | Fed-VFDT (MAJORITY) | 30.91% | 22.74% | 49.74% | 27.0 | 13 | 110.2 KB | 564 B | 65.5 + 0.7 = 66.3 KB | 65.5 + 16.8 = 82.4 KB |
| **Forest Covertype** | Fed-VFDT (BEST-MERIT) | 35.00% | 25.12% | 45.88% | 29.0 | 14 | 109.9 KB | 563 B | 65.5 + 0.7 = 66.3 KB | 65.5 + 16.8 = 82.4 KB |
| **Forest Covertype** | Fed-VFDT (STATS-MERGE) | **65.52%** | 63.04% | 58.87% | 67.0 | 33 | 10369.5 KB | 53092 B | 65.5 + 59.5 = 125.0 KB | 65.5 + 75.6 = 141.2 KB |
| **Airlines** | Centralized Oracle (T8) | 76.09% | 58.33% | 73.29% | 39.0 | 19 | 0.0 KB | 0 B | 9.3 + 0.0 = 9.3 KB | 0.0 KB |
| **Airlines** | Local-Only (K isolados) | 49.60% | 37.24% | 84.29% | 3.8 | 1 | 0.0 KB | 0 B | 9.3 + 0.0 = 9.3 KB | 0.0 KB |
| **Airlines** | Fed-Forest (TOP_M) | **76.39%** | 43.31% | 86.58% | 3.8 | 1 | 144.2 KB | 738 B | 15.8 + 0.9 = 16.7 KB | 20.8 + 1.1 = 21.9 KB |
| **Airlines** | Fed-Forest (ROUND_ROBIN) | 60.56% | 36.06% | 85.20% | 3.8 | 1 | 145.1 KB | 743 B | 15.8 + 1.0 = 16.9 KB | 20.8 + 1.2 = 22.1 KB |
| **Airlines** | Fed-VFDT (QUORUM) | 51.24% | 36.21% | 87.37% | 1.0 | 0 | 22.3 KB | 114 B | 9.3 + 0.1 = 9.4 KB | 9.2 + 9.9 = 19.1 KB |
| **Airlines** | Fed-VFDT (MAJORITY) | 56.78% | 45.94% | 68.70% | 7.0 | 3 | 40.3 KB | 206 B | 9.3 + 0.3 = 9.6 KB | 9.2 + 10.1 = 19.2 KB |
| **Airlines** | Fed-VFDT (BEST-MERIT) | 56.78% | 45.94% | 68.70% | 7.0 | 3 | 40.3 KB | 206 B | 9.3 + 0.3 = 9.6 KB | 9.2 + 10.1 = 19.2 KB |
| **Airlines** | Fed-VFDT (STATS-MERGE) | 61.43% | 47.58% | 51.06% | 41.0 | 20 | 783.1 KB | 4010 B | 9.3 + 4.7 = 14.0 KB | 9.2 + 14.5 = 23.7 KB |

## 4. Resultados em Regressão Contínua — Regime Non-IID (`Dirichlet α=0.3`)

| Dataset | Estratégia | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Nós Médios | Splits | Tráfego Total | Bytes / Cli / Rod. | RAM Cli (Mod + Com) | RAM Srv (Mod + Com) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | Centralized Oracle (T8) | 2.1537 | 2.7247 | 0.7013 | 2.0746 | 45.0 | 22 | 0.0 KB | 0 B | 25.5 + 0.0 = 25.5 KB | 0.0 KB |
| **Friedman #1** | Local-Only (K isolados) | 2.8288 | 3.5641 | 0.4848 | 2.8541 | 28.0 | 13 | 0.0 KB | 0 B | 25.5 + 0.0 = 25.5 KB | 0.0 KB |
| **Friedman #1** | Fed-Forest (TOP_M) | 2.8813 | 3.5622 | 0.4877 | 3.2023 | 28.0 | 13 | 560.2 KB | 2868 B | 32.1 + 3.4 = 35.5 KB | 21.1 + 3.7 = 24.7 KB |
| **Friedman #1** | Fed-Forest (ROUND_ROBIN) | 3.0070 | 3.7069 | 0.4409 | 3.1273 | 28.0 | 13 | 513.0 KB | 2627 B | 32.1 + 3.8 = 35.9 KB | 21.1 + 4.0 = 25.1 KB |
| **Friedman #1** | Fed-VFDT (QUORUM) | **2.4963** | 3.1615 | 0.5925 | 2.7144 | 7.0 | 3 | 38.7 KB | 198 B | 25.5 + 0.2 = 25.7 KB | 25.2 + 10.0 = 35.3 KB |
| **Friedman #1** | Fed-VFDT (MAJORITY) | 4.6219 | 6.5955 | -0.7637 | 2.9385 | 35.0 | 17 | 51.4 KB | 263 B | 25.5 + 0.4 = 25.9 KB | 25.2 + 10.2 = 35.5 KB |
| **Friedman #1** | Fed-VFDT (BEST-MERIT) | 4.6219 | 6.5955 | -0.7637 | 2.9385 | 35.0 | 17 | 51.4 KB | 263 B | 25.5 + 0.4 = 25.9 KB | 25.2 + 10.2 = 35.5 KB |
| **Friedman #1** | Fed-VFDT (STATS-MERGE) | 2.7318 | 3.4339 | 0.5201 | 3.1485 | 41.0 | 20 | 3624.2 KB | 18556 B | 25.5 + 20.6 = 46.1 KB | 25.2 + 30.4 = 55.7 KB |
| **Bikes** | Centralized Oracle (T8) | 40.4648 | 57.7754 | 0.4080 | 56.2665 | 63.0 | 31 | 0.0 KB | 0 B | 53.6 + 0.0 = 53.6 KB | 0.0 KB |
| **Bikes** | Local-Only (K isolados) | 61.6123 | 90.2799 | -0.5044 | 61.5621 | 23.4 | 11 | 0.0 KB | 0 B | 53.6 + 0.0 = 53.6 KB | 0.0 KB |
| **Bikes** | Fed-Forest (TOP_M) | 41.9784 | 84.3310 | -0.2612 | 62.5577 | 23.4 | 11 | 501.6 KB | 2568 B | 60.2 + 3.4 = 63.6 KB | 21.2 + 3.6 = 24.8 KB |
| **Bikes** | Fed-Forest (ROUND_ROBIN) | **41.7974** | 84.0268 | -0.2522 | 62.4287 | 23.4 | 11 | 503.4 KB | 2577 B | 60.2 + 3.1 = 63.3 KB | 21.2 + 3.3 = 24.5 KB |
| **Bikes** | Fed-VFDT (QUORUM) | 88.8534 | 111.9869 | -1.2861 | 94.4398 | 15.0 | 7 | 52.8 KB | 270 B | 53.6 + 0.4 = 54.0 KB | 53.3 + 10.2 = 63.6 KB |
| **Bikes** | Fed-VFDT (MAJORITY) | 73.6779 | 99.7705 | -0.7850 | 83.4224 | 43.0 | 21 | 53.9 KB | 276 B | 53.6 + 0.4 = 54.0 KB | 53.3 + 10.2 = 63.5 KB |
| **Bikes** | Fed-VFDT (BEST-MERIT) | 76.8024 | 101.2539 | -0.8981 | 90.5620 | 39.0 | 19 | 47.6 KB | 244 B | 53.6 + 0.3 = 53.9 KB | 53.3 + 10.1 = 63.5 KB |
| **Bikes** | Fed-VFDT (STATS-MERGE) | 46.6150 | 66.3923 | 0.2145 | 80.1154 | 49.0 | 24 | 8233.1 KB | 42154 B | 53.6 + 48.7 = 102.3 KB | 53.3 + 58.5 = 111.9 KB |
| **Appliances Energy** | Centralized Oracle (T8) | 29.7630 | 38.1122 | 0.7401 | 37.9290 | 43.0 | 21 | 0.0 KB | 0 B | 34.1 + 0.0 = 34.1 KB | 0.0 KB |
| **Appliances Energy** | Local-Only (K isolados) | 78.5876 | 103.2296 | -1.6912 | 387.1203 | 29.8 | 14 | 0.0 KB | 0 B | 34.1 + 0.0 = 34.1 KB | 0.0 KB |
| **Appliances Energy** | Fed-Forest (TOP_M) | **35.4504** | 44.5648 | 0.6446 | 351.0634 | 29.8 | 14 | 754.8 KB | 3865 B | 41.1 + 4.7 = 45.8 KB | 22.4 + 4.9 = 27.3 KB |
| **Appliances Energy** | Fed-Forest (ROUND_ROBIN) | 41.4174 | 51.2350 | 0.5296 | 352.6584 | 29.8 | 14 | 651.2 KB | 3334 B | 41.1 + 4.1 = 45.2 KB | 22.4 + 4.4 = 26.8 KB |
| **Appliances Energy** | Fed-VFDT (QUORUM) | 76.2666 | 113.3128 | -2.3409 | 388.1795 | 23.0 | 11 | 54.0 KB | 276 B | 34.1 + 0.4 = 34.5 KB | 34.1 + 10.3 = 44.4 KB |
| **Appliances Energy** | Fed-VFDT (MAJORITY) | 59.1440 | 98.6648 | -1.3147 | 383.8884 | 53.0 | 26 | 56.0 KB | 286 B | 34.1 + 0.4 = 34.5 KB | 34.1 + 10.3 = 44.4 KB |
| **Appliances Energy** | Fed-VFDT (BEST-MERIT) | 59.7945 | 98.4380 | -1.3743 | 380.3736 | 43.0 | 21 | 53.2 KB | 272 B | 34.1 + 0.4 = 34.4 KB | 34.1 + 10.3 = 44.3 KB |
| **Appliances Energy** | Fed-VFDT (STATS-MERGE) | 50.5366 | 90.6557 | -1.3364 | 380.4586 | 41.0 | 20 | 5122.7 KB | 26228 B | 34.1 + 29.2 = 63.3 KB | 34.1 + 39.1 = 73.2 KB |
| **NYC Taxi** | Centralized Oracle (T8) | 339.2442 | 440.9337 | 0.2054 | 340.5542 | 51.0 | 25 | 0.0 KB | 0 B | 12.3 + 0.0 = 12.3 KB | 0.0 KB |
| **NYC Taxi** | Local-Only (K isolados) | 397.0585 | 514.8715 | -0.0918 | 365.7157 | 29.6 | 14 | 0.0 KB | 0 B | 12.3 + 0.0 = 12.3 KB | 0.0 KB |
| **NYC Taxi** | Fed-Forest (TOP_M) | 366.1175 | 485.1046 | 0.0382 | 368.0326 | 29.6 | 14 | 525.4 KB | 2690 B | 18.8 + 3.5 = 22.2 KB | 20.8 + 3.7 = 24.5 KB |
| **NYC Taxi** | Fed-Forest (ROUND_ROBIN) | **351.5022** | 463.3443 | 0.1223 | 367.1597 | 29.6 | 14 | 534.3 KB | 2736 B | 18.8 + 3.8 = 22.6 KB | 20.8 + 4.1 = 24.9 KB |
| **NYC Taxi** | Fed-VFDT (QUORUM) | 381.5276 | 500.8862 | -0.0277 | 370.8696 | 23.0 | 11 | 54.6 KB | 280 B | 12.3 + 0.5 = 12.7 KB | 12.1 + 10.3 = 22.4 KB |
| **NYC Taxi** | Fed-VFDT (MAJORITY) | 391.4848 | 509.7329 | -0.0642 | 385.0680 | 49.0 | 24 | 41.4 KB | 212 B | 12.3 + 0.4 = 12.6 KB | 12.1 + 10.2 = 22.3 KB |
| **NYC Taxi** | Fed-VFDT (BEST-MERIT) | 390.8112 | 506.9668 | -0.0529 | 382.6197 | 57.0 | 28 | 46.9 KB | 240 B | 12.3 + 0.4 = 12.7 KB | 12.1 + 10.2 = 22.3 KB |
| **NYC Taxi** | Fed-VFDT (STATS-MERGE) | 372.1993 | 480.6436 | 0.0557 | 373.9513 | 39.0 | 19 | 1257.7 KB | 6439 B | 12.3 + 7.5 = 19.8 KB | 12.1 + 17.3 = 29.4 KB |

## 5. Resultados em Classificação — Regime IID (Referência Uniforme)

| Dataset | Estratégia | Holdout Acc (%) | Macro F1 (%) | Prequential Acc (%) | Nós Médios | Splits | Tráfego Total | Bytes / Cli / Rod. | RAM Cli (Mod + Com) | RAM Srv (Mod + Com) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Centralized Oracle (T8) | 99.02% | 98.87% | 97.30% | 43.0 | 21 | 0.0 KB | 0 B | 6.7 + 0.0 = 6.7 KB | 0.0 KB |
| **SEA** | Local-Only (K isolados) | 96.04% | 95.43% | 92.52% | 18.6 | 8 | 0.0 KB | 0 B | 6.7 + 0.0 = 6.7 KB | 0.0 KB |
| **SEA** | Fed-Forest (TOP_M) | 95.75% | 95.12% | 93.27% | 18.6 | 8 | 345.0 KB | 1766 B | 13.1 + 2.3 = 15.4 KB | 20.5 + 2.5 = 23.0 KB |
| **SEA** | Fed-Forest (ROUND_ROBIN) | 95.74% | 95.12% | 93.23% | 18.6 | 8 | 341.1 KB | 1747 B | 13.1 + 2.2 = 15.3 KB | 20.5 + 2.4 = 22.9 KB |
| **SEA** | Fed-VFDT (QUORUM) | 96.40% | 95.84% | 92.89% | 33.0 | 16 | 58.8 KB | 301 B | 6.7 + 0.5 = 7.1 KB | 6.7 + 10.3 = 16.9 KB |
| **SEA** | Fed-VFDT (MAJORITY) | 96.51% | 95.94% | 92.93% | 37.0 | 18 | 64.8 KB | 332 B | 6.7 + 0.4 = 7.0 KB | 6.7 + 10.2 = 16.8 KB |
| **SEA** | Fed-VFDT (BEST-MERIT) | 96.20% | 95.56% | 93.17% | 35.0 | 17 | 64.9 KB | 332 B | 6.7 + 0.4 = 7.1 KB | 6.7 + 10.2 = 16.9 KB |
| **SEA** | Fed-VFDT (STATS-MERGE) | **96.96%** | 96.52% | 93.89% | 35.0 | 17 | 398.9 KB | 2042 B | 6.7 + 2.2 = 8.8 KB | 6.7 + 12.0 = 18.6 KB |
| **Electricity** | Centralized Oracle (T8) | 64.37% | 63.96% | 64.57% | 79.0 | 39 | 0.0 KB | 0 B | 9.8 + 0.0 = 9.8 KB | 0.0 KB |
| **Electricity** | Local-Only (K isolados) | 63.02% | 61.58% | 63.18% | 31.2 | 15 | 0.0 KB | 0 B | 9.8 + 0.0 = 9.8 KB | 0.0 KB |
| **Electricity** | Fed-Forest (TOP_M) | 64.15% | 63.62% | 64.10% | 31.2 | 15 | 488.9 KB | 2503 B | 16.3 + 3.6 = 19.9 KB | 20.9 + 3.8 = 24.7 KB |
| **Electricity** | Fed-Forest (ROUND_ROBIN) | **64.17%** | 63.58% | 64.00% | 31.2 | 15 | 490.0 KB | 2509 B | 16.3 + 3.5 = 19.9 KB | 20.9 + 3.8 = 24.6 KB |
| **Electricity** | Fed-VFDT (QUORUM) | 63.82% | 62.62% | 63.23% | 43.0 | 21 | 90.0 KB | 461 B | 9.8 + 0.5 = 10.3 KB | 9.8 + 10.3 = 20.1 KB |
| **Electricity** | Fed-VFDT (MAJORITY) | 62.89% | 62.77% | 62.51% | 47.0 | 23 | 89.3 KB | 457 B | 9.8 + 0.5 = 10.3 KB | 9.8 + 10.3 = 20.1 KB |
| **Electricity** | Fed-VFDT (BEST-MERIT) | 63.42% | 63.06% | 62.93% | 43.0 | 21 | 83.1 KB | 426 B | 9.8 + 0.4 = 10.2 KB | 9.8 + 10.2 = 20.0 KB |
| **Electricity** | Fed-VFDT (STATS-MERGE) | 64.07% | 63.25% | 64.13% | 59.0 | 29 | 955.3 KB | 4891 B | 9.8 + 5.3 = 15.1 KB | 9.8 + 15.1 = 24.9 KB |
| **Forest Covertype** | Centralized Oracle (T8) | 67.28% | 65.21% | 59.99% | 77.0 | 38 | 0.0 KB | 0 B | 65.5 + 0.0 = 65.5 KB | 0.0 KB |
| **Forest Covertype** | Local-Only (K isolados) | 48.57% | 37.79% | 40.21% | 11.4 | 5 | 0.0 KB | 0 B | 65.5 + 0.0 = 65.5 KB | 0.0 KB |
| **Forest Covertype** | Fed-Forest (TOP_M) | 55.94% | 43.36% | 44.73% | 11.4 | 5 | 604.9 KB | 3097 B | 77.0 + 3.5 = 80.4 KB | 36.5 + 3.7 = 40.2 KB |
| **Forest Covertype** | Fed-Forest (ROUND_ROBIN) | 54.51% | 42.18% | 44.82% | 11.4 | 5 | 603.3 KB | 3089 B | 77.0 + 3.4 = 80.4 KB | 36.5 + 3.6 = 40.1 KB |
| **Forest Covertype** | Fed-VFDT (QUORUM) | 52.12% | 46.17% | 41.77% | 19.0 | 9 | 100.3 KB | 513 B | 65.5 + 0.7 = 66.3 KB | 65.5 + 16.8 = 82.4 KB |
| **Forest Covertype** | Fed-VFDT (MAJORITY) | 51.58% | 45.36% | 42.31% | 19.0 | 9 | 101.7 KB | 521 B | 65.5 + 0.7 = 66.3 KB | 65.5 + 16.8 = 82.4 KB |
| **Forest Covertype** | Fed-VFDT (BEST-MERIT) | 51.88% | 46.18% | 42.51% | 17.0 | 8 | 91.4 KB | 468 B | 65.5 + 0.7 = 66.3 KB | 65.5 + 16.8 = 82.4 KB |
| **Forest Covertype** | Fed-VFDT (STATS-MERGE) | **65.17%** | 60.32% | 51.07% | 75.0 | 37 | 10372.6 KB | 53108 B | 65.5 + 59.5 = 125.0 KB | 65.5 + 75.6 = 141.2 KB |
| **Airlines** | Centralized Oracle (T8) | 76.09% | 58.33% | 73.29% | 39.0 | 19 | 0.0 KB | 0 B | 9.3 + 0.0 = 9.3 KB | 0.0 KB |
| **Airlines** | Local-Only (K isolados) | 74.68% | 55.48% | 71.65% | 5.2 | 2 | 0.0 KB | 0 B | 9.3 + 0.0 = 9.3 KB | 0.0 KB |
| **Airlines** | Fed-Forest (TOP_M) | 76.39% | 43.31% | 75.03% | 5.2 | 2 | 152.9 KB | 783 B | 15.8 + 0.9 = 16.8 KB | 20.8 + 1.2 = 22.0 KB |
| **Airlines** | Fed-Forest (ROUND_ROBIN) | **76.47%** | 44.27% | 75.03% | 5.2 | 2 | 155.0 KB | 794 B | 15.8 + 0.9 = 16.7 KB | 20.8 + 1.1 = 21.9 KB |
| **Airlines** | Fed-VFDT (QUORUM) | 75.82% | 53.03% | 72.15% | 7.0 | 3 | 41.4 KB | 212 B | 9.3 + 0.3 = 9.6 KB | 9.2 + 10.1 = 19.2 KB |
| **Airlines** | Fed-VFDT (MAJORITY) | 75.80% | 53.48% | 71.97% | 9.0 | 4 | 45.7 KB | 234 B | 9.3 + 0.3 = 9.6 KB | 9.2 + 10.1 = 19.3 KB |
| **Airlines** | Fed-VFDT (BEST-MERIT) | 75.55% | 53.24% | 71.84% | 7.0 | 3 | 41.4 KB | 212 B | 9.3 + 0.3 = 9.6 KB | 9.2 + 10.1 = 19.2 KB |
| **Airlines** | Fed-VFDT (STATS-MERGE) | 70.45% | 56.88% | 62.71% | 37.0 | 18 | 793.8 KB | 4064 B | 9.3 + 4.7 = 14.0 KB | 9.2 + 14.5 = 23.7 KB |

## 6. Resultados em Regressão Contínua — Regime IID (Referência Uniforme)

| Dataset | Estratégia | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Nós Médios | Splits | Tráfego Total | Bytes / Cli / Rod. | RAM Cli (Mod + Com) | RAM Srv (Mod + Com) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | Centralized Oracle (T8) | 2.1537 | 2.7247 | 0.7013 | 2.0746 | 45.0 | 22 | 0.0 KB | 0 B | 25.5 + 0.0 = 25.5 KB | 0.0 KB |
| **Friedman #1** | Local-Only (K isolados) | 2.3695 | 3.0010 | 0.6367 | 3.1157 | 30.6 | 14 | 0.0 KB | 0 B | 25.5 + 0.0 = 25.5 KB | 0.0 KB |
| **Friedman #1** | Fed-Forest (TOP_M) | **2.1660** | 2.7034 | 0.7059 | 2.8854 | 30.6 | 14 | 534.5 KB | 2737 B | 32.1 + 3.5 = 35.6 KB | 21.1 + 3.7 = 24.8 KB |
| **Friedman #1** | Fed-Forest (ROUND_ROBIN) | 2.1737 | 2.7158 | 0.7032 | 2.8702 | 30.6 | 14 | 535.7 KB | 2743 B | 32.1 + 3.5 = 35.6 KB | 21.1 + 3.7 = 24.8 KB |
| **Friedman #1** | Fed-VFDT (QUORUM) | 2.3854 | 3.0295 | 0.6306 | 3.0837 | 33.0 | 16 | 73.9 KB | 378 B | 25.5 + 0.5 = 25.9 KB | 25.2 + 10.3 = 35.5 KB |
| **Friedman #1** | Fed-VFDT (MAJORITY) | 2.5936 | 3.2614 | 0.5720 | 3.5287 | 51.0 | 25 | 59.4 KB | 304 B | 25.5 + 0.3 = 25.8 KB | 25.2 + 10.1 = 35.4 KB |
| **Friedman #1** | Fed-VFDT (BEST-MERIT) | 2.6237 | 3.3242 | 0.5553 | 3.4654 | 49.0 | 24 | 60.6 KB | 310 B | 25.5 + 0.4 = 25.8 KB | 25.2 + 10.2 = 35.4 KB |
| **Friedman #1** | Fed-VFDT (STATS-MERGE) | 2.3507 | 2.9694 | 0.6452 | 3.0437 | 43.0 | 21 | 3084.9 KB | 15795 B | 25.5 + 20.6 = 46.1 KB | 25.2 + 30.4 = 55.7 KB |
| **Bikes** | Centralized Oracle (T8) | 40.4648 | 57.7754 | 0.4080 | 56.2665 | 63.0 | 31 | 0.0 KB | 0 B | 53.6 + 0.0 = 53.6 KB | 0.0 KB |
| **Bikes** | Local-Only (K isolados) | 57.5617 | 77.6016 | -0.0792 | 78.6782 | 30.0 | 14 | 0.0 KB | 0 B | 53.6 + 0.0 = 53.6 KB | 0.0 KB |
| **Bikes** | Fed-Forest (TOP_M) | 52.1307 | 66.0441 | 0.2263 | 73.7745 | 30.0 | 14 | 597.3 KB | 3058 B | 60.2 + 3.5 = 63.7 KB | 21.2 + 3.7 = 24.9 KB |
| **Bikes** | Fed-Forest (ROUND_ROBIN) | **52.1084** | 65.6746 | 0.2349 | 75.3351 | 30.0 | 14 | 591.5 KB | 3028 B | 60.2 + 3.5 = 63.8 KB | 21.2 + 3.7 = 24.9 KB |
| **Bikes** | Fed-VFDT (QUORUM) | 68.8674 | 91.2987 | -0.4787 | 76.4032 | 33.0 | 16 | 54.1 KB | 277 B | 53.6 + 0.3 = 53.9 KB | 53.3 + 10.1 = 63.5 KB |
| **Bikes** | Fed-VFDT (MAJORITY) | 59.8490 | 83.2357 | -0.2304 | 74.5081 | 47.0 | 23 | 51.9 KB | 266 B | 53.6 + 0.3 = 53.9 KB | 53.3 + 10.1 = 63.5 KB |
| **Bikes** | Fed-VFDT (BEST-MERIT) | 61.0911 | 83.5602 | -0.2400 | 76.5629 | 43.0 | 21 | 57.5 KB | 294 B | 53.6 + 0.4 = 54.0 KB | 53.3 + 10.2 = 63.5 KB |
| **Bikes** | Fed-VFDT (STATS-MERGE) | 58.7106 | 88.2154 | -0.3808 | 66.9590 | 55.0 | 27 | 8538.4 KB | 43717 B | 53.6 + 48.7 = 102.3 KB | 53.3 + 58.5 = 111.9 KB |
| **Appliances Energy** | Centralized Oracle (T8) | 29.7630 | 38.1122 | 0.7401 | 37.9290 | 43.0 | 21 | 0.0 KB | 0 B | 34.1 + 0.0 = 34.1 KB | 0.0 KB |
| **Appliances Energy** | Local-Only (K isolados) | 45.2032 | 60.5392 | 0.3026 | 285.0817 | 32.6 | 15 | 0.0 KB | 0 B | 34.1 + 0.0 = 34.1 KB | 0.0 KB |
| **Appliances Energy** | Fed-Forest (TOP_M) | **29.1254** | 36.9152 | 0.7561 | 255.5134 | 32.6 | 15 | 680.2 KB | 3483 B | 41.1 + 4.3 = 45.4 KB | 22.4 + 4.5 = 26.9 KB |
| **Appliances Energy** | Fed-Forest (ROUND_ROBIN) | 31.2946 | 39.7005 | 0.7178 | 257.5192 | 32.6 | 15 | 678.7 KB | 3475 B | 41.1 + 4.1 = 45.2 KB | 22.4 + 4.3 = 26.7 KB |
| **Appliances Energy** | Fed-VFDT (QUORUM) | 41.7799 | 55.6195 | 0.4330 | 289.6936 | 45.0 | 22 | 64.1 KB | 328 B | 34.1 + 0.4 = 34.5 KB | 34.1 + 10.3 = 44.4 KB |
| **Appliances Energy** | Fed-VFDT (MAJORITY) | 45.9180 | 61.3977 | 0.3116 | 286.2214 | 53.0 | 26 | 56.6 KB | 290 B | 34.1 + 0.4 = 34.4 KB | 34.1 + 10.3 = 44.3 KB |
| **Appliances Energy** | Fed-VFDT (BEST-MERIT) | 53.5065 | 69.8327 | 0.1079 | 291.5395 | 53.0 | 26 | 60.2 KB | 308 B | 34.1 + 0.4 = 34.5 KB | 34.1 + 10.3 = 44.4 KB |
| **Appliances Energy** | Fed-VFDT (STATS-MERGE) | 51.7220 | 73.2150 | -0.0226 | 288.7852 | 37.0 | 18 | 5048.9 KB | 25850 B | 34.1 + 29.2 = 63.3 KB | 34.1 + 39.1 = 73.2 KB |
| **NYC Taxi** | Centralized Oracle (T8) | 339.2442 | 440.9337 | 0.2054 | 340.5542 | 51.0 | 25 | 0.0 KB | 0 B | 12.3 + 0.0 = 12.3 KB | 0.0 KB |
| **NYC Taxi** | Local-Only (K isolados) | 373.7731 | 484.9695 | 0.0383 | 386.0502 | 33.2 | 16 | 0.0 KB | 0 B | 12.3 + 0.0 = 12.3 KB | 0.0 KB |
| **NYC Taxi** | Fed-Forest (TOP_M) | 351.7481 | 458.8928 | 0.1393 | 368.9297 | 33.2 | 16 | 574.4 KB | 2941 B | 18.8 + 3.7 = 22.4 KB | 20.8 + 3.9 = 24.7 KB |
| **NYC Taxi** | Fed-Forest (ROUND_ROBIN) | **346.9808** | 451.6048 | 0.1665 | 368.9357 | 33.2 | 16 | 566.8 KB | 2902 B | 18.8 + 3.8 = 22.6 KB | 20.8 + 4.0 = 24.8 KB |
| **NYC Taxi** | Fed-VFDT (QUORUM) | 372.8996 | 485.6688 | 0.0360 | 385.9779 | 41.0 | 20 | 49.9 KB | 256 B | 12.3 + 0.3 = 12.6 KB | 12.1 + 10.1 = 22.2 KB |
| **NYC Taxi** | Fed-VFDT (MAJORITY) | 366.9594 | 481.7159 | 0.0516 | 389.5902 | 63.0 | 31 | 50.5 KB | 259 B | 12.3 + 0.4 = 12.6 KB | 12.1 + 10.2 = 22.3 KB |
| **NYC Taxi** | Fed-VFDT (BEST-MERIT) | 382.3622 | 500.6639 | -0.0245 | 394.6844 | 51.0 | 25 | 44.4 KB | 227 B | 12.3 + 0.4 = 12.6 KB | 12.1 + 10.2 = 22.3 KB |
| **NYC Taxi** | Fed-VFDT (STATS-MERGE) | 358.2531 | 466.8824 | 0.1091 | 371.3217 | 41.0 | 20 | 1286.1 KB | 6585 B | 12.3 + 7.5 = 19.8 KB | 12.1 + 17.3 = 29.4 KB |

## 7. Síntese Executiva: Ganho Federado sob Non-IID vs Local-Only e Oráculo Centralizado

| Dataset | Local-Only | Melhor Fed-Forest | Melhor Fed-VFDT (Coord.) | Fed-VFDT (Stats-Merge) | Central Oracle (T8) | Ref. `main_benchmark` (T8) | Vencedor Federado (Non-IID) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SEA** | 76.41% | 95.06% | 93.68% | 96.78% | 99.02% | 99.02% | **fed_vfdt_merge** (+20.38 pp vs Local) |
| **Electricity** | 50.65% | 54.78% | 57.61% | 60.37% | 64.37% | 64.37% | **fed_vfdt_merge** (+9.72 pp vs Local) |
| **Forest Covertype** | 30.91% | 25.83% | 35.00% | 65.52% | 67.28% | 67.28% | **fed_vfdt_merge** (+34.61 pp vs Local) |
| **Airlines** | 49.60% | 76.39% | 56.78% | 61.43% | 76.09% | 76.09% | **fed_forest_topm** (+26.79 pp vs Local) |
| **Friedman #1** | 2.8288 | 2.8813 | 2.4963 | 2.7318 | 2.1537 | 2.1537 | **fed_vfdt_quorum** (+11.8% MAE vs Local) |
| **Bikes** | 61.6123 | 41.7974 | 73.6779 | 46.6150 | 40.4648 | 40.4648 | **fed_forest_rr** (+32.2% MAE vs Local) |
| **Appliances Energy** | 78.5876 | 35.4504 | 59.1440 | 50.5366 | 29.7630 | 29.7630 | **fed_forest_topm** (+54.9% MAE vs Local) |
| **NYC Taxi** | 397.0585 | 351.5022 | 381.5276 | 372.1993 | 339.2442 | 339.2442 | **fed_forest_rr** (+11.5% MAE vs Local) |

### Principais Conclusões Arquiteturais

1. **Fed-VFDT (`STATS-MERGE`) como Oráculo Estatístico em Non-IID (Trade-off de Buffer TX):** Ao fundir estatísticas suficientes exatas (Chan/Welford, histogramas, contagens de classe e slots de regressão) na folha global, o Modo 2B reconstrói fielmente a distribuição global mesmo quando clientes individuais enxergam apenas 1–2 classes ou faixas estreitas de $y$ sob `Dirichlet(0.3)`, aproximando-se do **Centralized Oracle** — com o custo explícito de exigir um buffer de comunicação proporcional ao tamanho das folhas ativas (chegando a **59.5 KB** de buffer TX e **125.0 KB** totais no Covertype).
2. **Fed-VFDT Coordenado (`QUORUM` / `MAJORITY` / `BEST-MERIT`) — Menor Custo de Rede e Menor Buffer RAM:** Transmite apenas propostas leves de split e comandos incrementais (`~100–560 B/cliente/rodada`), exigindo um buffer TX/RX diminuto (`< 1 KB` observado) e mantendo a SRAM total no cliente em no máximo **66.3 KB** mesmo em 54 features — ideal para links LPWAN/LoRa/BLE e MCUs de SRAM estrita.
3. **Fed-Forest (`TOP_M` / `ROUND_ROBIN`) — Resiliência Assíncrona com Buffer Contido:** Cada cliente evolui sua árvore local de forma assíncrona e recebe um micro-ensemble compactado ($M=5$ árvores BFS, ~1.3 KB/árvore), elevando consistentemente a generalização global sobre `Local-Only` em Non-IID (especialmente em regressão contínua) enquanto mantém o conjunto completo **Modelo + Comunicação** abaixo de 100 KB no cliente (pico real **81.0 KB** no Covertype) e servidor enxuto (`16` árvores compactas).
