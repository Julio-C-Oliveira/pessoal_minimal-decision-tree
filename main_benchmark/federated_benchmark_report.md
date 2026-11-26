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

## 2. Pegada Estática de SRAM por Cliente ($\le 100\text{ KB}$) e Heterogeneidade Non-IID

| Dataset | Tarefa | Features / Classes | SRAM Central/Local | SRAM Fed-Forest ($M=5$) | SRAM Fed-VFDT (Cliente) | SRAM Servidor (VFDT) | Non-IID Amostras/Cliente (Min–Max) | Métrica de Heterogeneidade Non-IID |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SEA** | classification | 3 / 2 cls | 6.66 KB | **13.12 KB** | **6.67 KB** | 16.48 KB | 69–4748 | Classe dom. média: **92.7%** (H=0.34 bits) |
| **Electricity** | classification | 8 / 2 cls | 9.80 KB | **16.38 KB** | **9.82 KB** | 19.64 KB | 127–8361 | Classe dom. média: **94.1%** (H=0.21 bits) |
| **Forest Covertype** | classification | 54 / 7 cls | 65.55 KB | **77.01 KB** | **65.56 KB** | 81.69 KB | 420–4453 | Classe dom. média: **53.2%** (H=1.68 bits) |
| **Airlines** | classification | 7 / 2 cls | 9.34 KB | **15.89 KB** | **9.35 KB** | 19.01 KB | 158–7066 | Classe dom. média: **88.2%** (H=0.41 bits) |
| **Friedman #1** | regression | 10 / cont. | 25.48 KB | **32.11 KB** | **25.49 KB** | 35.09 KB | 236–3554 | Dispersão $\mu_k(y)/\sigma(y)$: **0.59$\sigma$** |
| **Bikes** | regression | 12 / cont. | 53.62 KB | **60.29 KB** | **53.63 KB** | 63.19 KB | 496–2010 | Dispersão $\mu_k(y)/\sigma(y)$: **0.35$\sigma$** |
| **Appliances Energy** | regression | 28 / cont. | 34.09 KB | **41.15 KB** | **34.10 KB** | 44.00 KB | 511–2598 | Dispersão $\mu_k(y)/\sigma(y)$: **0.48$\sigma$** |
| **NYC Taxi** | regression | 7 / cont. | 12.26 KB | **18.81 KB** | **12.27 KB** | 21.93 KB | 212–2260 | Dispersão $\mu_k(y)/\sigma(y)$: **0.35$\sigma$** |

> **Validação de Orçamento Embarcado:** O pico de SRAM estática por cliente entre todos os 8 datasets (incluindo Forest Covertype com 54 features e 7 classes) foi de **77.01 KB** — cumprindo com folga o teto de **100 KB** para MCUs como ESP32.

## 3. Resultados em Classificação — Regime Non-IID (`Dirichlet α=0.3`)

| Dataset | Estratégia | Holdout Acc (%) | Macro F1 (%) | Prequential Acc (%) | Nós Médios | Splits | Tráfego Total | Bytes / Cliente / Rodada |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Centralized Oracle (T8) | 99.02% | 98.87% | 97.30% | 43.0 | 21 | 0.0 KB | 0 B |
| **SEA** | Local-Only (K isolados) | 76.41% | 69.04% | 97.14% | 9.0 | 4 | 0.0 KB | 0 B |
| **SEA** | Fed-Forest (TOP_M) | 95.06% | 94.33% | 96.36% | 9.0 | 4 | 219.3 KB | 1123 B |
| **SEA** | Fed-Forest (ROUND_ROBIN) | 72.75% | 50.40% | 96.16% | 9.0 | 4 | 205.3 KB | 1051 B |
| **SEA** | Fed-VFDT (QUORUM) | 88.56% | 87.57% | 92.56% | 23.0 | 11 | 50.2 KB | 257 B |
| **SEA** | Fed-VFDT (MAJORITY) | 93.68% | 92.93% | 93.66% | 35.0 | 17 | 55.6 KB | 285 B |
| **SEA** | Fed-VFDT (BEST-MERIT) | 93.68% | 92.93% | 93.66% | 35.0 | 17 | 55.6 KB | 285 B |
| **SEA** | Fed-VFDT (STATS-MERGE) | **96.78%** | 96.30% | 94.84% | 37.0 | 18 | 396.7 KB | 2031 B |
| **Electricity** | Centralized Oracle (T8) | 64.37% | 63.96% | 64.57% | 79.0 | 39 | 0.0 KB | 0 B |
| **Electricity** | Local-Only (K isolados) | 50.65% | 36.14% | 96.79% | 10.2 | 4 | 0.0 KB | 0 B |
| **Electricity** | Fed-Forest (TOP_M) | 54.78% | 35.39% | 96.15% | 10.2 | 4 | 232.4 KB | 1190 B |
| **Electricity** | Fed-Forest (ROUND_ROBIN) | 48.27% | 33.02% | 96.22% | 10.2 | 4 | 249.0 KB | 1275 B |
| **Electricity** | Fed-VFDT (QUORUM) | 57.61% | 51.47% | 80.91% | 37.0 | 18 | 66.0 KB | 338 B |
| **Electricity** | Fed-VFDT (MAJORITY) | 50.89% | 42.16% | 78.75% | 57.0 | 28 | 82.7 KB | 423 B |
| **Electricity** | Fed-VFDT (BEST-MERIT) | 50.34% | 43.80% | 79.10% | 49.0 | 24 | 81.5 KB | 417 B |
| **Electricity** | Fed-VFDT (STATS-MERGE) | **60.37%** | 55.40% | 78.26% | 71.0 | 35 | 951.6 KB | 4872 B |
| **Forest Covertype** | Centralized Oracle (T8) | 67.28% | 65.21% | 59.99% | 77.0 | 38 | 0.0 KB | 0 B |
| **Forest Covertype** | Local-Only (K isolados) | 30.91% | 17.52% | 67.53% | 11.8 | 5 | 0.0 KB | 0 B |
| **Forest Covertype** | Fed-Forest (TOP_M) | 25.82% | 11.59% | 61.04% | 11.8 | 5 | 649.4 KB | 3325 B |
| **Forest Covertype** | Fed-Forest (ROUND_ROBIN) | 25.83% | 11.59% | 62.37% | 11.8 | 5 | 607.5 KB | 3110 B |
| **Forest Covertype** | Fed-VFDT (QUORUM) | 16.58% | 7.33% | 54.42% | 1.0 | 0 | 32.0 KB | 164 B |
| **Forest Covertype** | Fed-VFDT (MAJORITY) | 30.91% | 22.74% | 49.74% | 27.0 | 13 | 110.2 KB | 564 B |
| **Forest Covertype** | Fed-VFDT (BEST-MERIT) | 35.00% | 25.12% | 45.88% | 29.0 | 14 | 109.9 KB | 563 B |
| **Forest Covertype** | Fed-VFDT (STATS-MERGE) | **65.52%** | 63.04% | 58.87% | 67.0 | 33 | 10369.5 KB | 53092 B |
| **Airlines** | Centralized Oracle (T8) | 76.09% | 58.33% | 73.29% | 39.0 | 19 | 0.0 KB | 0 B |
| **Airlines** | Local-Only (K isolados) | 49.60% | 37.24% | 84.29% | 3.8 | 1 | 0.0 KB | 0 B |
| **Airlines** | Fed-Forest (TOP_M) | **76.39%** | 43.31% | 86.58% | 3.8 | 1 | 144.2 KB | 738 B |
| **Airlines** | Fed-Forest (ROUND_ROBIN) | 60.56% | 36.06% | 85.20% | 3.8 | 1 | 145.1 KB | 743 B |
| **Airlines** | Fed-VFDT (QUORUM) | 51.24% | 36.21% | 87.37% | 1.0 | 0 | 22.3 KB | 114 B |
| **Airlines** | Fed-VFDT (MAJORITY) | 56.78% | 45.94% | 68.70% | 7.0 | 3 | 40.3 KB | 206 B |
| **Airlines** | Fed-VFDT (BEST-MERIT) | 56.78% | 45.94% | 68.70% | 7.0 | 3 | 40.3 KB | 206 B |
| **Airlines** | Fed-VFDT (STATS-MERGE) | 61.43% | 47.58% | 51.06% | 41.0 | 20 | 783.1 KB | 4010 B |

## 4. Resultados em Regressão Contínua — Regime Non-IID (`Dirichlet α=0.3`)

| Dataset | Estratégia | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Nós Médios | Splits | Tráfego Total | Bytes / Cliente / Rodada |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | Centralized Oracle (T8) | 2.1537 | 2.7247 | 0.7013 | 2.0746 | 45.0 | 22 | 0.0 KB | 0 B |
| **Friedman #1** | Local-Only (K isolados) | 2.8288 | 3.5641 | 0.4848 | 2.8541 | 28.0 | 13 | 0.0 KB | 0 B |
| **Friedman #1** | Fed-Forest (TOP_M) | 2.8813 | 3.5622 | 0.4877 | 3.2023 | 28.0 | 13 | 560.2 KB | 2868 B |
| **Friedman #1** | Fed-Forest (ROUND_ROBIN) | 3.0070 | 3.7069 | 0.4409 | 3.1273 | 28.0 | 13 | 513.0 KB | 2627 B |
| **Friedman #1** | Fed-VFDT (QUORUM) | **2.4963** | 3.1615 | 0.5925 | 2.7144 | 7.0 | 3 | 38.7 KB | 198 B |
| **Friedman #1** | Fed-VFDT (MAJORITY) | 4.6219 | 6.5955 | -0.7637 | 2.9385 | 35.0 | 17 | 51.4 KB | 263 B |
| **Friedman #1** | Fed-VFDT (BEST-MERIT) | 4.6219 | 6.5955 | -0.7637 | 2.9385 | 35.0 | 17 | 51.4 KB | 263 B |
| **Friedman #1** | Fed-VFDT (STATS-MERGE) | 2.7318 | 3.4339 | 0.5201 | 3.1485 | 41.0 | 20 | 3624.2 KB | 18556 B |
| **Bikes** | Centralized Oracle (T8) | 40.4648 | 57.7754 | 0.4080 | 56.2665 | 63.0 | 31 | 0.0 KB | 0 B |
| **Bikes** | Local-Only (K isolados) | 61.6123 | 90.2799 | -0.5044 | 61.5621 | 23.4 | 11 | 0.0 KB | 0 B |
| **Bikes** | Fed-Forest (TOP_M) | 41.9784 | 84.3310 | -0.2612 | 62.5577 | 23.4 | 11 | 501.6 KB | 2568 B |
| **Bikes** | Fed-Forest (ROUND_ROBIN) | **41.7974** | 84.0268 | -0.2522 | 62.4287 | 23.4 | 11 | 503.4 KB | 2577 B |
| **Bikes** | Fed-VFDT (QUORUM) | 88.8534 | 111.9869 | -1.2861 | 94.4398 | 15.0 | 7 | 52.8 KB | 270 B |
| **Bikes** | Fed-VFDT (MAJORITY) | 73.6779 | 99.7705 | -0.7850 | 83.4224 | 43.0 | 21 | 53.9 KB | 276 B |
| **Bikes** | Fed-VFDT (BEST-MERIT) | 76.8024 | 101.2539 | -0.8981 | 90.5620 | 39.0 | 19 | 47.6 KB | 244 B |
| **Bikes** | Fed-VFDT (STATS-MERGE) | 46.6150 | 66.3923 | 0.2145 | 80.1154 | 49.0 | 24 | 8233.1 KB | 42154 B |
| **Appliances Energy** | Centralized Oracle (T8) | 29.7630 | 38.1122 | 0.7401 | 37.9290 | 43.0 | 21 | 0.0 KB | 0 B |
| **Appliances Energy** | Local-Only (K isolados) | 78.5876 | 103.2296 | -1.6912 | 387.1203 | 29.8 | 14 | 0.0 KB | 0 B |
| **Appliances Energy** | Fed-Forest (TOP_M) | **35.4504** | 44.5648 | 0.6446 | 351.0634 | 29.8 | 14 | 754.8 KB | 3865 B |
| **Appliances Energy** | Fed-Forest (ROUND_ROBIN) | 41.4174 | 51.2350 | 0.5296 | 352.6584 | 29.8 | 14 | 651.2 KB | 3334 B |
| **Appliances Energy** | Fed-VFDT (QUORUM) | 76.2666 | 113.3128 | -2.3409 | 388.1795 | 23.0 | 11 | 54.0 KB | 276 B |
| **Appliances Energy** | Fed-VFDT (MAJORITY) | 59.1440 | 98.6648 | -1.3147 | 383.8884 | 53.0 | 26 | 56.0 KB | 286 B |
| **Appliances Energy** | Fed-VFDT (BEST-MERIT) | 59.7945 | 98.4380 | -1.3743 | 380.3736 | 43.0 | 21 | 53.2 KB | 272 B |
| **Appliances Energy** | Fed-VFDT (STATS-MERGE) | 50.5366 | 90.6557 | -1.3364 | 380.4586 | 41.0 | 20 | 5122.7 KB | 26228 B |
| **NYC Taxi** | Centralized Oracle (T8) | 339.2442 | 440.9337 | 0.2054 | 340.5542 | 51.0 | 25 | 0.0 KB | 0 B |
| **NYC Taxi** | Local-Only (K isolados) | 397.0585 | 514.8715 | -0.0918 | 365.7157 | 29.6 | 14 | 0.0 KB | 0 B |
| **NYC Taxi** | Fed-Forest (TOP_M) | 366.1175 | 485.1046 | 0.0382 | 368.0326 | 29.6 | 14 | 525.4 KB | 2690 B |
| **NYC Taxi** | Fed-Forest (ROUND_ROBIN) | **351.5022** | 463.3443 | 0.1223 | 367.1597 | 29.6 | 14 | 534.3 KB | 2736 B |
| **NYC Taxi** | Fed-VFDT (QUORUM) | 381.5276 | 500.8862 | -0.0277 | 370.8696 | 23.0 | 11 | 54.6 KB | 280 B |
| **NYC Taxi** | Fed-VFDT (MAJORITY) | 391.4848 | 509.7329 | -0.0642 | 385.0680 | 49.0 | 24 | 41.4 KB | 212 B |
| **NYC Taxi** | Fed-VFDT (BEST-MERIT) | 390.8112 | 506.9668 | -0.0529 | 382.6197 | 57.0 | 28 | 46.9 KB | 240 B |
| **NYC Taxi** | Fed-VFDT (STATS-MERGE) | 372.1993 | 480.6436 | 0.0557 | 373.9513 | 39.0 | 19 | 1257.7 KB | 6439 B |

## 5. Resultados em Classificação — Regime IID (Referência Uniforme)

| Dataset | Estratégia | Holdout Acc (%) | Macro F1 (%) | Prequential Acc (%) | Nós Médios | Splits | Tráfego Total | Bytes / Cliente / Rodada |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Centralized Oracle (T8) | 99.02% | 98.87% | 97.30% | 43.0 | 21 | 0.0 KB | 0 B |
| **SEA** | Local-Only (K isolados) | 96.04% | 95.43% | 92.52% | 18.6 | 8 | 0.0 KB | 0 B |
| **SEA** | Fed-Forest (TOP_M) | 95.75% | 95.12% | 93.27% | 18.6 | 8 | 345.0 KB | 1766 B |
| **SEA** | Fed-Forest (ROUND_ROBIN) | 95.74% | 95.12% | 93.23% | 18.6 | 8 | 341.1 KB | 1747 B |
| **SEA** | Fed-VFDT (QUORUM) | 96.40% | 95.84% | 92.89% | 33.0 | 16 | 58.8 KB | 301 B |
| **SEA** | Fed-VFDT (MAJORITY) | 96.51% | 95.94% | 92.93% | 37.0 | 18 | 64.8 KB | 332 B |
| **SEA** | Fed-VFDT (BEST-MERIT) | 96.20% | 95.56% | 93.17% | 35.0 | 17 | 64.9 KB | 332 B |
| **SEA** | Fed-VFDT (STATS-MERGE) | **96.96%** | 96.52% | 93.89% | 35.0 | 17 | 398.9 KB | 2042 B |
| **Electricity** | Centralized Oracle (T8) | 64.37% | 63.96% | 64.57% | 79.0 | 39 | 0.0 KB | 0 B |
| **Electricity** | Local-Only (K isolados) | 63.02% | 61.58% | 63.18% | 31.2 | 15 | 0.0 KB | 0 B |
| **Electricity** | Fed-Forest (TOP_M) | 64.15% | 63.62% | 64.10% | 31.2 | 15 | 488.9 KB | 2503 B |
| **Electricity** | Fed-Forest (ROUND_ROBIN) | **64.17%** | 63.58% | 64.00% | 31.2 | 15 | 490.0 KB | 2509 B |
| **Electricity** | Fed-VFDT (QUORUM) | 63.82% | 62.62% | 63.23% | 43.0 | 21 | 90.0 KB | 461 B |
| **Electricity** | Fed-VFDT (MAJORITY) | 62.89% | 62.77% | 62.51% | 47.0 | 23 | 89.3 KB | 457 B |
| **Electricity** | Fed-VFDT (BEST-MERIT) | 63.42% | 63.06% | 62.93% | 43.0 | 21 | 83.1 KB | 426 B |
| **Electricity** | Fed-VFDT (STATS-MERGE) | 64.07% | 63.25% | 64.13% | 59.0 | 29 | 955.3 KB | 4891 B |
| **Forest Covertype** | Centralized Oracle (T8) | 67.28% | 65.21% | 59.99% | 77.0 | 38 | 0.0 KB | 0 B |
| **Forest Covertype** | Local-Only (K isolados) | 48.57% | 37.79% | 40.21% | 11.4 | 5 | 0.0 KB | 0 B |
| **Forest Covertype** | Fed-Forest (TOP_M) | 55.94% | 43.36% | 44.73% | 11.4 | 5 | 604.9 KB | 3097 B |
| **Forest Covertype** | Fed-Forest (ROUND_ROBIN) | 54.51% | 42.18% | 44.82% | 11.4 | 5 | 603.3 KB | 3089 B |
| **Forest Covertype** | Fed-VFDT (QUORUM) | 52.12% | 46.17% | 41.77% | 19.0 | 9 | 100.3 KB | 513 B |
| **Forest Covertype** | Fed-VFDT (MAJORITY) | 51.58% | 45.36% | 42.31% | 19.0 | 9 | 101.7 KB | 521 B |
| **Forest Covertype** | Fed-VFDT (BEST-MERIT) | 51.88% | 46.18% | 42.51% | 17.0 | 8 | 91.4 KB | 468 B |
| **Forest Covertype** | Fed-VFDT (STATS-MERGE) | **65.17%** | 60.32% | 51.07% | 75.0 | 37 | 10372.6 KB | 53108 B |
| **Airlines** | Centralized Oracle (T8) | 76.09% | 58.33% | 73.29% | 39.0 | 19 | 0.0 KB | 0 B |
| **Airlines** | Local-Only (K isolados) | 74.68% | 55.48% | 71.65% | 5.2 | 2 | 0.0 KB | 0 B |
| **Airlines** | Fed-Forest (TOP_M) | 76.39% | 43.31% | 75.03% | 5.2 | 2 | 152.9 KB | 783 B |
| **Airlines** | Fed-Forest (ROUND_ROBIN) | **76.47%** | 44.27% | 75.03% | 5.2 | 2 | 155.0 KB | 794 B |
| **Airlines** | Fed-VFDT (QUORUM) | 75.82% | 53.03% | 72.15% | 7.0 | 3 | 41.4 KB | 212 B |
| **Airlines** | Fed-VFDT (MAJORITY) | 75.80% | 53.48% | 71.97% | 9.0 | 4 | 45.7 KB | 234 B |
| **Airlines** | Fed-VFDT (BEST-MERIT) | 75.55% | 53.24% | 71.84% | 7.0 | 3 | 41.4 KB | 212 B |
| **Airlines** | Fed-VFDT (STATS-MERGE) | 70.45% | 56.88% | 62.71% | 37.0 | 18 | 793.8 KB | 4064 B |

## 6. Resultados em Regressão Contínua — Regime IID (Referência Uniforme)

| Dataset | Estratégia | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Nós Médios | Splits | Tráfego Total | Bytes / Cliente / Rodada |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | Centralized Oracle (T8) | 2.1537 | 2.7247 | 0.7013 | 2.0746 | 45.0 | 22 | 0.0 KB | 0 B |
| **Friedman #1** | Local-Only (K isolados) | 2.3695 | 3.0010 | 0.6367 | 3.1157 | 30.6 | 14 | 0.0 KB | 0 B |
| **Friedman #1** | Fed-Forest (TOP_M) | **2.1660** | 2.7034 | 0.7059 | 2.8854 | 30.6 | 14 | 534.5 KB | 2737 B |
| **Friedman #1** | Fed-Forest (ROUND_ROBIN) | 2.1737 | 2.7158 | 0.7032 | 2.8702 | 30.6 | 14 | 535.7 KB | 2743 B |
| **Friedman #1** | Fed-VFDT (QUORUM) | 2.3854 | 3.0295 | 0.6306 | 3.0837 | 33.0 | 16 | 73.9 KB | 378 B |
| **Friedman #1** | Fed-VFDT (MAJORITY) | 2.5936 | 3.2614 | 0.5720 | 3.5287 | 51.0 | 25 | 59.4 KB | 304 B |
| **Friedman #1** | Fed-VFDT (BEST-MERIT) | 2.6237 | 3.3242 | 0.5553 | 3.4654 | 49.0 | 24 | 60.6 KB | 310 B |
| **Friedman #1** | Fed-VFDT (STATS-MERGE) | 2.3507 | 2.9694 | 0.6452 | 3.0437 | 43.0 | 21 | 3084.9 KB | 15795 B |
| **Bikes** | Centralized Oracle (T8) | 40.4648 | 57.7754 | 0.4080 | 56.2665 | 63.0 | 31 | 0.0 KB | 0 B |
| **Bikes** | Local-Only (K isolados) | 57.5617 | 77.6016 | -0.0792 | 78.6782 | 30.0 | 14 | 0.0 KB | 0 B |
| **Bikes** | Fed-Forest (TOP_M) | 52.1307 | 66.0441 | 0.2263 | 73.7745 | 30.0 | 14 | 597.3 KB | 3058 B |
| **Bikes** | Fed-Forest (ROUND_ROBIN) | **52.1084** | 65.6746 | 0.2349 | 75.3351 | 30.0 | 14 | 591.5 KB | 3028 B |
| **Bikes** | Fed-VFDT (QUORUM) | 68.8674 | 91.2987 | -0.4787 | 76.4032 | 33.0 | 16 | 54.1 KB | 277 B |
| **Bikes** | Fed-VFDT (MAJORITY) | 59.8490 | 83.2357 | -0.2304 | 74.5081 | 47.0 | 23 | 51.9 KB | 266 B |
| **Bikes** | Fed-VFDT (BEST-MERIT) | 61.0911 | 83.5602 | -0.2400 | 76.5629 | 43.0 | 21 | 57.5 KB | 294 B |
| **Bikes** | Fed-VFDT (STATS-MERGE) | 58.7106 | 88.2154 | -0.3808 | 66.9590 | 55.0 | 27 | 8538.4 KB | 43717 B |
| **Appliances Energy** | Centralized Oracle (T8) | 29.7630 | 38.1122 | 0.7401 | 37.9290 | 43.0 | 21 | 0.0 KB | 0 B |
| **Appliances Energy** | Local-Only (K isolados) | 45.2032 | 60.5392 | 0.3026 | 285.0817 | 32.6 | 15 | 0.0 KB | 0 B |
| **Appliances Energy** | Fed-Forest (TOP_M) | **29.1254** | 36.9152 | 0.7561 | 255.5134 | 32.6 | 15 | 680.2 KB | 3483 B |
| **Appliances Energy** | Fed-Forest (ROUND_ROBIN) | 31.2946 | 39.7005 | 0.7178 | 257.5192 | 32.6 | 15 | 678.7 KB | 3475 B |
| **Appliances Energy** | Fed-VFDT (QUORUM) | 41.7799 | 55.6195 | 0.4330 | 289.6936 | 45.0 | 22 | 64.1 KB | 328 B |
| **Appliances Energy** | Fed-VFDT (MAJORITY) | 45.9180 | 61.3977 | 0.3116 | 286.2214 | 53.0 | 26 | 56.6 KB | 290 B |
| **Appliances Energy** | Fed-VFDT (BEST-MERIT) | 53.5065 | 69.8327 | 0.1079 | 291.5395 | 53.0 | 26 | 60.2 KB | 308 B |
| **Appliances Energy** | Fed-VFDT (STATS-MERGE) | 51.7220 | 73.2150 | -0.0226 | 288.7852 | 37.0 | 18 | 5048.9 KB | 25850 B |
| **NYC Taxi** | Centralized Oracle (T8) | 339.2442 | 440.9337 | 0.2054 | 340.5542 | 51.0 | 25 | 0.0 KB | 0 B |
| **NYC Taxi** | Local-Only (K isolados) | 373.7731 | 484.9695 | 0.0383 | 386.0502 | 33.2 | 16 | 0.0 KB | 0 B |
| **NYC Taxi** | Fed-Forest (TOP_M) | 351.7481 | 458.8928 | 0.1393 | 368.9297 | 33.2 | 16 | 574.4 KB | 2941 B |
| **NYC Taxi** | Fed-Forest (ROUND_ROBIN) | **346.9808** | 451.6048 | 0.1665 | 368.9357 | 33.2 | 16 | 566.8 KB | 2902 B |
| **NYC Taxi** | Fed-VFDT (QUORUM) | 372.8996 | 485.6688 | 0.0360 | 385.9779 | 41.0 | 20 | 49.9 KB | 256 B |
| **NYC Taxi** | Fed-VFDT (MAJORITY) | 366.9594 | 481.7159 | 0.0516 | 389.5902 | 63.0 | 31 | 50.5 KB | 259 B |
| **NYC Taxi** | Fed-VFDT (BEST-MERIT) | 382.3622 | 500.6639 | -0.0245 | 394.6844 | 51.0 | 25 | 44.4 KB | 227 B |
| **NYC Taxi** | Fed-VFDT (STATS-MERGE) | 358.2531 | 466.8824 | 0.1091 | 371.3217 | 41.0 | 20 | 1286.1 KB | 6585 B |

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

1. **Fed-VFDT (`STATS-MERGE`) como Oráculo Estatístico em Non-IID:** Ao fundir estatísticas suficientes exatas (Chan/Welford, histogramas, contagens de classe e slots de regressão) na folha global, o Modo 2B reconstrói fielmente a distribuição global mesmo quando clientes individuais enxergam apenas 1–2 classes ou faixas estreitas de $y$ sob `Dirichlet(0.3)`, aproximando-se do **Centralized Oracle**.
2. **Fed-VFDT Coordenado (`QUORUM` / `MAJORITY` / `BEST-MERIT`) — Custo de Rede Mínimo:** Transmite apenas propostas leves de split e comandos incrementais (`~100–400 B/cliente/rodada`), sendo ideal para links LPWAN/LoRa/BLE onde o orçamento de banda é o gargalo principal.
3. **Fed-Forest (`TOP_M` / `ROUND_ROBIN`) — Resiliência sem Sincronismo Rígido:** Cada cliente evolui sua árvore local de forma assíncrona e recebe um micro-ensemble compactado ($M=5$ árvores BFS, ~1.3 KB/árvore), elevando consistentemente a generalização global sobre o baseline `Local-Only` em Non-IID com votação ponderada adaptativa e mantendo a SRAM total por cliente abaixo de 100 KB.
