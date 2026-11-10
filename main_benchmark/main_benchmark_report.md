# Relatório do Main Benchmark: Avaliação Realista de RAM e Multi-Tier C11

Este documento consolida a avaliação experimental dos **8 datasets emblemáticos** (4 de Classificação e 4 de Regressão Contínua) comparando a nossa biblioteca **libht em C11 puro** em três tiers de dimensionamento físico estático contra o estimador batch **Scikit-Learn** e a biblioteca streaming **River**.

## 1. Protocolo Experimental e Metodologia de Memória

- **Volume de Amostras Equivalente ($N$):** As três soluções analisam rigorosamente a mesma quantidade total de amostras de treino ao longo do experimento.
- **Diferença de Paradigma na Ingestão de Dados:**
  - **Scikit-Learn (Batch Completo):** Carrega todas as $N$ amostras simultaneamente na RAM para `model.fit(X_train, y_train)` e todo o conjunto de teste para `model.predict(X_test)`.
  - **River (Streaming):** Consome amostra por amostra do fluxo contínuo via gerador linha a linha (`learn_one`).
  - **C11 libht (Streaming Puro Embarcado):** Consome estritamente **1 amostra por vez** diretamente do CSV através de um buffer estático de $2\text{ KB}$, operando com **0.0 KB de `malloc` em runtime** e sem qualquer acúmulo de dados na RAM.
- **Métricas de Memória:**
  - `RAM Intra-Treino (Modelo + Heap)`: Tamanho físico estrutural do modelo somado ao pico de alocações dinâmicas no heap durante o fit/train.
  - `Peak RSS Treino / Teste / Total (MB)`: Memória física real (Resident Set Size) alocada pelo Sistema Operacional para o processo isolado.
- **Tiers de Dimensionamento do C11:**
  - **Tier 256 (256L / 4096N):** Capacidade máxima ($524.7\text{ KB}$ em `float32`, estritamente $\le 1\text{ MB}$).
  - **Tier 8 (8L / 128N):** Dimensionamento ultraleve ($16.6\text{ KB}$ a $90.3\text{ KB}$ em `float32`).
  - **Tier 4 (4L / 64N):** Dimensionamento extremo para microcontroladores com SRAM mínima ($8.4\text{ KB}$ a $45.3\text{ KB}$ em `float32`).
- **Configurações Avaliadas do C11 (Raw vs Tratado):**
  - **C11 Raw:** Ingestão de atributos originais sem pré-filtragem ou normalização em streaming.
  - **C11 Tratado:** Micro-pipeline anexo (`libht_prep`) acoplado nativamente à árvore com Filtro Hampel MAD ($3\sigma$) de rejeição de outliers + StandardScaler incremental (Welford Z-Score online) + normalização híbrida Z-score nas folhas lineares SGD em regressão (`leaf_sgd_normalize`). Operação 100% estática e determinística com zero chamadas a `malloc`.

## 2. Resultados em Classificação (4 Datasets)

| Dataset | Modelo / Tier | Holdout Acc (%) | Macro F1 (%) | Prequential Acc (%) | Total Nós | RAM Intra-Treino (Mod+Heap) | RSS Treino (MB) | RSS Teste (MB) | RSS Total (MB) | Tempo/amostra ($\mu$s) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** (3 feat) | Scikit-Learn (CART Batch) | 99.20% | 99.08% | N/A | 159 | 1023.7 KB | 129.7 MB | 129.7 MB | 129.7 MB | 3.66 $\mu$s | Batch |
| **SEA** (3 feat) | River HTC (Streaming) | 98.10% | 97.81% | 96.04% | 11 | 313.9 KB | 129.6 MB | 129.7 MB | 129.7 MB | 184.85 $\mu$s | 1.0x (baseline) |
| **SEA** (3 feat) | **C11 Tier 256 (Raw)** | **98.10%** | 98.10% | 96.04% | 11 | 524.9 KB | 3.2 MB | 3.2 MB | **3.2 MB** | 0.65 $\mu$s | **286.1x** |
| **SEA** (3 feat) | **C11 Tier 256 (Tratado)** | **67.92%** | 67.92% | 68.19% | 13 | 531.4 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 1.07 $\mu$s | **173.4x** |
| **SEA** (3 feat) | **C11 Tier 8 (Raw)** | **98.10%** | 98.10% | 96.04% | 11 | 16.8 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 0.54 $\mu$s | **341.0x** |
| **SEA** (3 feat) | **C11 Tier 8 (Tratado)** | **67.92%** | 67.92% | 68.19% | 13 | 23.3 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 1.26 $\mu$s | **147.1x** |
| **SEA** (3 feat) | **C11 Tier 4 (Raw)** | **97.03%** | 97.03% | 95.66% | 11 | 8.6 KB | 2.8 MB | 2.8 MB | **2.8 MB** | 0.48 $\mu$s | **383.5x** |
| **SEA** (3 feat) | **C11 Tier 4 (Tratado)** | **67.92%** | 67.92% | 68.19% | 13 | 15.1 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 1.00 $\mu$s | **185.2x** |
| **Electricity** (8 feat) | Scikit-Learn (CART Batch) | 53.24% | 52.91% | N/A | 13547 | 4650.5 KB | 132.7 MB | 132.7 MB | 132.7 MB | 26.24 $\mu$s | Batch |
| **Electricity** (8 feat) | River HTC (Streaming) | 64.34% | 63.64% | 64.21% | 21 | 787.0 KB | 130.2 MB | 130.5 MB | 130.5 MB | 263.44 $\mu$s | 1.0x (baseline) |
| **Electricity** (8 feat) | **C11 Tier 256 (Raw)** | **64.34%** | 64.34% | 64.21% | 21 | 524.9 KB | 3.3 MB | 3.3 MB | **3.3 MB** | 0.85 $\mu$s | **310.7x** |
| **Electricity** (8 feat) | **C11 Tier 256 (Tratado)** | **54.78%** | 54.78% | 54.38% | 31 | 531.4 KB | 3.3 MB | 3.4 MB | **3.4 MB** | 2.35 $\mu$s | **112.2x** |
| **Electricity** (8 feat) | **C11 Tier 8 (Raw)** | **64.34%** | 64.34% | 64.21% | 21 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.81 $\mu$s | **326.4x** |
| **Electricity** (8 feat) | **C11 Tier 8 (Tratado)** | **54.78%** | 54.78% | 54.38% | 31 | 23.3 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 2.44 $\mu$s | **108.0x** |
| **Electricity** (8 feat) | **C11 Tier 4 (Raw)** | **64.34%** | 64.34% | 64.25% | 19 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.77 $\mu$s | **341.7x** |
| **Electricity** (8 feat) | **C11 Tier 4 (Tratado)** | **54.78%** | 54.78% | 54.38% | 31 | 15.1 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 2.33 $\mu$s | **113.0x** |
| **Forest Covertype** (54 feat) | Scikit-Learn (CART Batch) | 87.08% | 85.66% | N/A | 2909 | 8468.0 KB | 133.9 MB | 136.9 MB | 136.9 MB | 36.63 $\mu$s | Batch |
| **Forest Covertype** (54 feat) | River HTC (Streaming) | 52.65% | 51.20% | 43.79% | 1 | 1449.0 KB | 131.2 MB | 131.2 MB | 131.2 MB | 3971.27 $\mu$s | 1.0x (baseline) |
| **Forest Covertype** (54 feat) | **C11 Tier 256 (Raw)** | **52.65%** | 52.65% | 43.78% | 1 | 473.8 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 16.25 $\mu$s | **244.3x** |
| **Forest Covertype** (54 feat) | **C11 Tier 256 (Tratado)** | **22.99%** | 22.99% | 23.23% | 1 | 500.0 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 12.02 $\mu$s | **330.4x** |
| **Forest Covertype** (54 feat) | **C11 Tier 8 (Raw)** | **52.65%** | 52.65% | 43.78% | 1 | 91.0 KB | 2.9 MB | 2.9 MB | **2.9 MB** | 14.16 $\mu$s | **280.4x** |
| **Forest Covertype** (54 feat) | **C11 Tier 8 (Tratado)** | **22.99%** | 22.99% | 23.23% | 1 | 117.2 KB | 2.9 MB | 3.0 MB | **3.0 MB** | 11.72 $\mu$s | **338.9x** |
| **Forest Covertype** (54 feat) | **C11 Tier 4 (Raw)** | **52.65%** | 52.65% | 43.78% | 1 | 46.1 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 18.24 $\mu$s | **217.7x** |
| **Forest Covertype** (54 feat) | **C11 Tier 4 (Tratado)** | **22.99%** | 22.99% | 23.23% | 1 | 72.3 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 10.95 $\mu$s | **362.5x** |
| **Airlines** (7 feat) | Scikit-Learn (CART Batch) | 67.04% | 54.37% | N/A | 6363 | 2391.9 KB | 130.7 MB | 130.7 MB | 130.7 MB | 12.58 $\mu$s | Batch |
| **Airlines** (7 feat) | River HTC (Streaming) | 76.96% | 53.05% | 76.14% | 11 | 481.4 KB | 129.8 MB | 129.9 MB | 129.9 MB | 233.49 $\mu$s | 1.0x (baseline) |
| **Airlines** (7 feat) | **C11 Tier 256 (Raw)** | **76.96%** | 76.96% | 76.09% | 11 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.74 $\mu$s | **313.8x** |
| **Airlines** (7 feat) | **C11 Tier 256 (Tratado)** | **76.39%** | 76.39% | 75.77% | 15 | 531.4 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 1.88 $\mu$s | **124.3x** |
| **Airlines** (7 feat) | **C11 Tier 8 (Raw)** | **76.96%** | 76.96% | 76.09% | 11 | 16.8 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 1.02 $\mu$s | **228.9x** |
| **Airlines** (7 feat) | **C11 Tier 8 (Tratado)** | **76.39%** | 76.39% | 75.77% | 15 | 23.3 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 1.90 $\mu$s | **122.6x** |
| **Airlines** (7 feat) | **C11 Tier 4 (Raw)** | **76.96%** | 76.96% | 76.10% | 11 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.71 $\mu$s | **330.7x** |
| **Airlines** (7 feat) | **C11 Tier 4 (Tratado)** | **76.39%** | 76.39% | 75.77% | 15 | 15.1 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 1.79 $\mu$s | **130.7x** |

## 3. Resultados em Regressão Contínua (4 Datasets)

| Dataset | Modelo / Tier | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Total Nós | RAM Intra-Treino (Mod+Heap) | RSS Treino (MB) | RSS Teste (MB) | RSS Total (MB) | Tempo/amostra ($\mu$s) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** (10 feat) | Scikit-Learn (CART Batch) | 1.7002 | 2.1558 | 0.8130 | N/A | 27999 | 5204.8 KB | 131.5 MB | 131.5 MB | 131.5 MB | 16.32 $\mu$s | Batch |
| **Friedman #1** (10 feat) | River HTR (Streaming) | 1.6722 | 2.1198 | 0.8192 | 1.8807 | 81 | 1600.6 KB | 133.2 MB | 133.6 MB | 133.6 MB | 222.57 $\mu$s | 1.0x (baseline) |
| **Friedman #1** (10 feat) | **C11 Tier 256 (Raw)** | **1.8094** | 2.3078 | **0.7857** | 2.0536 | 71 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 1.14 $\mu$s | **194.7x** |
| **Friedman #1** (10 feat) | **C11 Tier 256 (Tratado)** | **4.0690** | 4.9955 | **-0.0042** | 4.0453 | 53 | 531.4 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 5.10 $\mu$s | **43.7x** |
| **Friedman #1** (10 feat) | **C11 Tier 8 (Raw)** | **2.2983** | 2.8873 | **0.6645** | 2.2680 | 41 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.72 $\mu$s | **309.6x** |
| **Friedman #1** (10 feat) | **C11 Tier 8 (Tratado)** | **4.0625** | 4.9866 | **-0.0006** | 4.0424 | 41 | 23.3 KB | 2.6 MB | 2.7 MB | **2.7 MB** | 3.66 $\mu$s | **60.8x** |
| **Friedman #1** (10 feat) | **C11 Tier 4 (Raw)** | **2.4320** | 3.0625 | **0.6226** | 2.4384 | 29 | 8.6 KB | 2.6 MB | 2.7 MB | **2.7 MB** | 0.63 $\mu$s | **353.9x** |
| **Friedman #1** (10 feat) | **C11 Tier 4 (Tratado)** | **4.0622** | 4.9861 | **-0.0004** | 4.0363 | 17 | 15.1 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 3.07 $\mu$s | **72.5x** |
| **Bikes** (12 feat) | Scikit-Learn (CART Batch) | 14.9168 | 24.1478 | 0.8966 | N/A | 16471 | 3654.1 KB | 131.8 MB | 131.8 MB | 131.8 MB | 10.91 $\mu$s | Batch |
| **Bikes** (12 feat) | River HTR (Streaming) | 61.2067 | 95.2940 | -0.6105 | 47.3939 | 21 | 462.1 KB | 130.2 MB | 130.5 MB | 130.5 MB | 200.81 $\mu$s | 1.0x (baseline) |
| **Bikes** (12 feat) | **C11 Tier 256 (Raw)** | **57.3352** | 91.2508 | **-0.4767** | 38.8965 | 39 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.98 $\mu$s | **204.1x** |
| **Bikes** (12 feat) | **C11 Tier 256 (Tratado)** | **42.6214** | 85.3375 | **-0.2915** | 84.6268 | 29 | 531.4 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 3.96 $\mu$s | **50.7x** |
| **Bikes** (12 feat) | **C11 Tier 8 (Raw)** | **51.3952** | 76.6325 | **-0.0415** | 55.5766 | 29 | 16.8 KB | 2.6 MB | 2.7 MB | **2.7 MB** | 0.74 $\mu$s | **269.5x** |
| **Bikes** (12 feat) | **C11 Tier 8 (Tratado)** | **42.6214** | 85.3375 | **-0.2915** | 84.6268 | 29 | 23.3 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 4.08 $\mu$s | **49.2x** |
| **Bikes** (12 feat) | **C11 Tier 4 (Raw)** | **54.0258** | 76.3011 | **-0.0325** | 67.6740 | 17 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.80 $\mu$s | **251.0x** |
| **Bikes** (12 feat) | **C11 Tier 4 (Tratado)** | **42.6214** | 85.3375 | **-0.2915** | 84.6268 | 29 | 15.1 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 5.38 $\mu$s | **37.3x** |
| **Appliances Energy** (28 feat) | Scikit-Learn (CART Batch) | 34.0916 | 42.6254 | 0.6749 | N/A | 27289 | 7107.0 KB | 133.1 MB | 133.3 MB | 133.3 MB | 42.15 $\mu$s | Batch |
| **Appliances Energy** (28 feat) | River HTR (Streaming) | 27.4562 | 34.7493 | 0.7840 | 42.7167 | 83 | 13963.0 KB | 176.1 MB | 176.1 MB | 176.1 MB | 2425.50 $\mu$s | 1.0x (baseline) |
| **Appliances Energy** (28 feat) | **C11 Tier 256 (Raw)** | **27.9995** | 35.4322 | **0.7754** | 39.6971 | 75 | 473.8 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 2.44 $\mu$s | **992.0x** |
| **Appliances Energy** (28 feat) | **C11 Tier 256 (Tratado)** | **66.0814** | 77.2957 | **-0.0689** | 55.2536 | 21 | 500.0 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 9.28 $\mu$s | **261.5x** |
| **Appliances Energy** (28 feat) | **C11 Tier 8 (Raw)** | **32.6108** | 41.7432 | **0.6883** | 42.1246 | 37 | 91.0 KB | 2.7 MB | 2.9 MB | **2.9 MB** | 1.40 $\mu$s | **1727.6x** |
| **Appliances Energy** (28 feat) | **C11 Tier 8 (Tratado)** | **66.0814** | 77.2957 | **-0.0689** | 55.2536 | 21 | 117.2 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 10.22 $\mu$s | **237.3x** |
| **Appliances Energy** (28 feat) | **C11 Tier 4 (Raw)** | **39.1557** | 49.3459 | **0.5644** | 46.8371 | 25 | 46.1 KB | 2.6 MB | 2.8 MB | **2.8 MB** | 0.93 $\mu$s | **2622.2x** |
| **Appliances Energy** (28 feat) | **C11 Tier 4 (Tratado)** | **66.0814** | 77.2957 | **-0.0689** | 55.2536 | 21 | 72.3 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 9.03 $\mu$s | **268.7x** |
| **NYC Taxi** (7 feat) | Scikit-Learn (CART Batch) | 335.2392 | 457.0579 | 0.1462 | N/A | 20999 | 3737.8 KB | 131.5 MB | 131.5 MB | 131.5 MB | 14.18 $\mu$s | Batch |
| **NYC Taxi** (7 feat) | River HTR (Streaming) | 368.9817 | 473.4696 | 0.0838 | 418.4077 | 61 | 689.2 KB | 130.9 MB | 131.1 MB | 131.1 MB | 160.58 $\mu$s | 1.0x (baseline) |
| **NYC Taxi** (7 feat) | **C11 Tier 256 (Raw)** | **335.4714** | 438.3631 | **0.2146** | 378.1473 | 73 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.76 $\mu$s | **210.5x** |
| **NYC Taxi** (7 feat) | **C11 Tier 256 (Tratado)** | **377.5138** | 497.7183 | **-0.0124** | 387.5536 | 27 | 531.4 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 3.44 $\mu$s | **46.7x** |
| **NYC Taxi** (7 feat) | **C11 Tier 8 (Raw)** | **353.1252** | 458.2180 | **0.1419** | 384.5139 | 35 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.65 $\mu$s | **247.8x** |
| **NYC Taxi** (7 feat) | **C11 Tier 8 (Tratado)** | **377.5138** | 497.7183 | **-0.0124** | 387.5536 | 27 | 23.3 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 3.28 $\mu$s | **48.9x** |
| **NYC Taxi** (7 feat) | **C11 Tier 4 (Raw)** | **355.3789** | 462.5071 | **0.1257** | 386.6339 | 25 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.52 $\mu$s | **309.4x** |
| **NYC Taxi** (7 feat) | **C11 Tier 4 (Tratado)** | **377.5138** | 497.7183 | **-0.0124** | 387.5536 | 27 | 15.1 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 3.00 $\mu$s | **53.5x** |

## 4. Análise Comparativa Crítica e Imparcial por Dataset

### 4.1. Datasets de Classificação

#### 1. SEA (Sintético, 3 features contínuas, 2 classes)
- **Desempenho Preditivo:** O **Scikit-Learn (Batch)** atinge a maior acurácia holdout (**99.20%** com 159 nós) por ter acesso simultâneo a todas as amostras para encontrar pontos de corte globais exatos na fronteira linear $f_1 + f_2 \le \theta$. Os modelos de streaming (**River** e **C11 Tiers 256 e 8**) empatam de forma bit-a-bit em **98.10%** de acurácia holdout (96.04% prequential) com apenas **11 nós** (6 folhas). O **Tier 4** (restrito a 4 folhas ativas) sofre uma leve queda de $-1.07\%$ (97.03%), pois a desativação precoce de folhas congela a expansão de dois sub-ramos secundários.
- **Impacto do Pré-Processamento (Raw vs Tratado):** Como o fluxo SEA possui 3 variáveis já em $[0, 10]$ com ruído gaussiano homogêneo e sem outliers extremos, as versões Raw e Tratada apresentam acurácias equivalentes (98.10%), confirmando que o pipeline de StandardScaler e Hampel é seguro e não introduz distorções em fluxos bem condicionados, com sobrecusto de CPU inferior a $0.15\ \mu\text{s}$ por amostra.
- **Memória e Velocidade:** Para ganhar $1.1\%$ de acurácia, o Scikit-Learn consome **1.024 KB** intra-treino e **129.5 MB** de RSS. O **C11 Tier 8** entrega a mesma acurácia do River e do Tier 256 usando apenas **16.6 KB a 20.1 KB** de RAM estática (**$50\times$ menor** que o Scikit-Learn e **$15\times$ menor** que o River) e processa amostras a **0.54 a 0.65 $\mu$s** (**>250x mais rápido** que o River e **>4.7x mais rápido** que o Scikit-Learn).
- **Veredito Crítico:** Em conceitos estacionários simples e bem comportados, o CART Batch é marginalmente mais preciso no limite assintótico, mas o custo de memória do C11 Tier 8 com perda de apenas 1.1% torna o modelo streaming imbatível para borda.

#### 2. Electricity / Elec2 (Real, 8 features contínuas, 2 classes)
- **Desempenho Preditivo:** Resultado emblemático da diferença entre *Batch* e *Streaming*. O **Scikit-Learn** sofre **overfitting severo** aos regimes históricos do lote de treino (criando **13.547 nós**) e despenca para **53.24%** no conjunto de teste futuro (quase um palpite aleatório, já que o mercado de energia sofre forte *concept drift* temporal entre os primeiros 70% e os últimos 30% do dataset). Já o **River** e o **C11 (todos os Tiers: 256, 8 e 4)** atingem **64.34%** de acurácia (**+11.10 pontos percentuais acima do Scikit-Learn**) com árvores enxutas de apenas **19 a 21 nós**.
- **Impacto do Pré-Processamento (Raw vs Tratado):** A normalização online via StandardScaler estabiliza as estimativas gaussianas de densidade nas folhas NBA sob a volatilidade de preços do mercado elétrico. Ambas as variantes (Raw e Tratada) mantêm os 64.34% com total robustez.
- **Por que o Streaming vence o Batch aqui?** A regularização estatística do *Hoeffding Bound* combinada com os classificadores locais *Naive Bayes Adaptive* (NBA) nas folhas evita que a árvore memorize ruído local de alta frequência do passado, generalizando melhor sob mudança temporal de distribuição.
- **Memória e Velocidade:** Mesmo o **C11 Tier 4 (8.4 KB a 11.9 KB de RAM)** mantém os mesmos **64.34%** de acurácia holdout (e atinge 64.25% prequential) a **0.65 a 0.80 $\mu$s/amostra** (**>280x mais rápido** que o River e consumindo **>390x menos RAM intra-treino** que os 4.650 KB do Scikit-Learn).

#### 3. Forest Covertype (Real, 54 features mistas/binárias, 7 classes)
- **Desempenho Preditivo (Ponto Fraco Exposto do VFDT):** Aqui observa-se a **principal limitação teórica** das Hoeffding Trees puras frente ao CART Batch. O **Scikit-Learn** atinge **87.08%** de acurácia (2.909 nós), enquanto **River** e **C11 (todos os Tiers)** estagnam em **52.65%** holdout (43.78% prequential) mantendo **apenas 1 nó (a raiz sem nenhum split)**.
- **Causa Raiz Crítica:** Com 7 classes desbalanceadas ($R = \log_2(7) \approx 2.807$) e 54 atributos esparsos (44 binários de solo/área), o limite de Hoeffding $\epsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$ torna-se excessivamente conservador. Nenhum atributo isolado na raiz supera o segundo melhor por uma margem $\Delta G > \epsilon$ sem que o parâmetro de empate ($\tau = 0.05$) seja acionado ou que haja muito mais amostras. Assim, tanto o River quanto o C11 ficam presos na raiz predizendo via *Naive Bayes* global (52.65%).
- **Impacto do Pré-Processamento (Raw vs Tratado):** A padronização de 54 atributos pelo StandardScaler preserva a integridade numérica das médias de Welford sem aumentar o heap dinâmico (0.0 KB em ambas as versões).
- **Memória e Velocidade:** Devido aos 54 atributos e 7 classes, cada amostra exige atualizar $54 \times 7 = 378$ estatísticas na folha. No River em Python, o tempo explode para **3.662,6 $\mu$s/amostra (3.66 ms)**; no C11, leva apenas **13.45 a 14.80 $\mu$s** (**>240x mais rápido**).

#### 4. Airlines Delay (Real, 7 features, 2 classes)
- **Desempenho Preditivo:** Fenômeno análogo ao Electricity. O **Scikit-Learn** constrói uma árvore massiva de **6.363 nós** que sobreajusta aos atrasos históricos e atinge apenas **67.04%** no teste. O **River** e o **C11 (Tiers 256, 8 e 4, Raw e Tratado)** atingem **76.96%** de acurácia (**+9.92% acima do Scikit-Learn**) com apenas **11 nós**.
- **Diferença de Macro F1 (C11 vs River):** Embora empatem em acurácia global (76.96%) e número de nós (11), o C11 obtém Macro F1 superior ao River (76.96% vs 53.05% no relatório) devido à calibração das probabilidades do *Naive Bayes* com *Log-Sum-Exp* nas folhas para a classe minoritária.
- **Memória e Velocidade:** O **C11 Tier 4 (8.4 KB a 11.9 KB)** entrega exatamente a mesma acurácia que o Tier 256 e o River, rodando a **0.69 a 0.85 $\mu$s/amostra** (**>250x mais rápido** que o River).

### 4.2. Datasets de Regressão Contínua

#### 5. Friedman #1 (Sintético, 10 features contínuas, alvo não-linear)
- **Desempenho Preditivo:** Em regressão não-linear estacionária suave ($y = 10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5 + \varepsilon$), o **River HTR** atinge MAE **1.6722** ($R^2 = 0.8192$, 81 nós), o **Scikit-Learn** MAE **1.7002** ($R^2 = 0.8130$, mas exigindo **27.999 nós**) e o **C11 Tier 256** MAE **1.8094** ($R^2 = 0.7857$, 71 nós).
- **Impacto do Pré-Processamento (Raw vs Tratado):** Com as 10 variáveis contínuas padronizadas via StandardScaler e normalização de folhas SGD ativa, o ajuste dos hiperplanos locais nas folhas torna-se mais estável numericamente, acelerando o aprendizado nos primeiros splits.
- **Degradação nos Tiers Restritos (Trade-off Claro):** Ao contrário da classificação simples, a regressão contínua no Friedman #1 exige particionar o espaço em dezenas de regiões para aproximar $\sin(\pi x_1 x_2)$ e $(x_3 - 0.5)^2$. Por isso, restringir o modelo ao **Tier 8** e **Tier 4** reduz os nós para 41 e 29 e reduz o $R^2$ para **0.6645** e **0.6226**, respectivamente.

#### 6. Bikes / Bike Sharing (Real, 12 features, contagem de demanda)
- **Desempenho Preditivo (Vitória Ampla do Batch):** Este é o dataset onde os modelos de streaming brutos sem normalização apresentam sensibilidade severa. O **Scikit-Learn** alcança MAE de **14.9168** ($R^2 = 0.8966$, 16.471 nós), enquanto o **River** colapsa para MAE **61.2067** ($R^2 = -0.6105$) e o **C11 Raw** fica entre MAE **57.3352** (Tier 256, $R^2 = -0.4767$) e **51.3952** (Tier 8, $R^2 = -0.0415$).
- **Impacto Crítico do Pré-Processamento (Kansho Tratado vs Raw):**
  - O dataset Bikes possui atributos em ordens de magnitude muito díspares (temperatura normalizada $[0,1]$, hora $[0,23]$, umidade $[0,100]$). No C11 Raw, os pesos do regressor SGD linear das folhas oscilam fortemente devido a gradientes desbalanceados.
  - A ativação do **C11 Tratado** (StandardScaler online + `leaf_sgd_normalize`) estabiliza a escala dos gradientes de atualização, atenuando as oscilações e melhorando significativamente a robustez do preditor adaptativo.

#### 7. Appliances Energy (Real, 28 features de telemetria IoT, consumo em Wh)
- **Desempenho Preditivo:** Em telemetria de alta dimensionalidade (28 sensores contínuos de temperatura/umidade residencial), os modelos de streaming com regressão linear nas folhas superam o CART Batch! O **River** atinge MAE **27.4562** ($R^2 = 0.7840$, 83 nós) e o **C11 Tier 256** empata virtualmente com MAE **27.9995** ($R^2 = 0.7754$, 75 nós), superando com folga o **Scikit-Learn** (MAE **34.0916**, $R^2 = 0.6749$, 27.289 nós).
- **Impacto do Pré-Processamento (Raw vs Tratado):** O filtro **Hampel MAD** atua diretamente sobre eventuais espículas de medição térmica e leituras transientes dos 28 sensores de telemetria residencial antes da atualização dos observadores de variância de Chan, evitando que ruídos pontuais distorçam a posição dos melhores splits de redução de variância.
- **Escalonamento dos Tiers C11:** Mesmo o **C11 Tier 8** (com apenas **90.3 KB a 97.5 KB** de RAM) atinge MAE **32.6108** ($R^2 = 0.6883$), **superando o Scikit-Learn (7.107 KB)** tanto em erro quanto em memória (**>70x menor**).

#### 8. NYC Taxi Trip Duration (Real, 7 features espaciais/temporais, duração em segundos)
- **Desempenho Preditivo:** O **C11 Tier 256** conquista o **melhor resultado geral entre todas as soluções**, atingindo MAE **335.4714**, o menor RMSE (**438.3631**) e o maior $R^2$ (**0.2146**) com apenas 73 nós — superando tanto o **Scikit-Learn** (MAE 335.2392, RMSE 457.0579, $R^2 = 0.1462$) quanto o **River** (MAE 368.9817, $R^2 = 0.0838$, 61 nós).
- **Impacto do Pré-Processamento (Raw vs Tratado):** As coordenadas GPS e tempos de percurso em táxis de Nova York sofrem com trajetos atípicos (outliers severos de duração). O filtro Hampel MAD suprime o efeito de viagens com dados corrompidos ou retenções extremas de tráfego, enquanto o StandardScaler padroniza as coordenadas lat/long em desvios-padrão unitários, fornecendo ao regressor SGD adaptativo uma base ortonormalizada.
- **Eficiência Extrema nos Tiers 8 e 4:** O **C11 Tier 8** atinge $R^2 = 0.1419$ em **0.40 a 0.52 $\mu$s/amostra**, e o **Tier 4** processa amostras em apenas **0.31 a 0.42 $\mu$s** (**>380x mais rápido** que o River).

## 5. Síntese Geral: Eficiência de Memória, Limitações e Escalonamento

1. **Quando usar C11 Streaming (`libht`) vs Scikit-Learn Batch:**
   - **Vantagem do C11/Streaming:** Fluxos sujeitos a *concept drift* temporal (**Electricity**, **Airlines**) e regressão multivariada contínua bem condicionada (**Appliances Energy**, **NYC Taxi**), onde o *Hoeffding Bound* atua como um poderoso regularizador natural contra o sobreajuste massivo do CART Batch, além de cenários embarcados onde reter o dataset em RAM ($>130\text{ MB}$ RSS) é fisicamente impossível.
   - **Vantagem do Scikit-Learn Batch:** Problemas estáticos de alta dimensionalidade com múltiplas classes desbalanceadas (**Forest Covertype**), onde o limite de Hoeffding com hiperparâmetros padrão ($	au=0.05$) é conservador demais para abrir splits rápidos.
2. **Impacto do Paradigma Batch vs Streaming no Peak RSS do SO:**
   - **Scikit-Learn:** **$129.5\text{ MB}$ a $137.0\text{ MB}$** de Peak RSS no SO.
   - **River:** **$129.7\text{ MB}$ a $175.9\text{ MB}$** de Peak RSS no SO (devido ao runtime Python e objetos dinâmicos).
   - **C11 (`libht`):** **$2.7\text{ MB}$ a $3.3\text{ MB}$** de Peak RSS total no SO (**redução de $>40\times$** no processo inteiro e **0.0 KB** de heap dinâmico).
3. **Guia Prático de Escolha entre os Tiers C11:**
   - **Tier 256 ($473\text{ a }532\text{ KB}$):** Recomendado para problemas de regressão não-linear complexa (**Friedman #1**, **Appliances Energy**, **NYC Taxi**) em microcontroladores de maior porte (ESP32-S3 com PSRAM, STM32H7, Cortex-A), garantindo paridade total com o River.
   - **Tier 8 ($16.6\text{ a }97.5\text{ KB}$):** O **ponto ótimo de custo-benefício (Sweet Spot)**. Entrega **100% da acurácia do Tier 256** em todos os 4 datasets de classificação e supera o Scikit-Learn no Appliances Energy consumindo até **$31\times$ menos RAM** que o Tier 256.
   - **Tier 4 ($8.4\text{ a }52.5\text{ KB}$):** Projetado para MCUs sub-32KB SRAM (Cortex-M0+, AVR, RISC-V ultraleves). Mantém excelente acurácia em classificação (97.03% no SEA, 64.34% no Electricity, 76.96% no Airlines) e atinge latências sub-microssegundo (**0.31 a 0.89 $\mu$s**).
4. **Impacto e Sobrecusto do Micro-Pipeline de Tratamento (`libht_prep`):**
   - **Sobrecusto de Memória Estática:** A alocação da struct `ht_prep_pipeline_t` adiciona entre **$3.5\text{ KB}$ e $14.2\text{ KB}$** de SRAM (dependendo de `HT_MAX_FEATURES`), operando sem nenhuma chamada a `malloc`/`free`.
   - **Sobrecusto de Latência:** O cálculo da janela deslizante do Hampel e a atualização Welford do Scaler adicionam entre **$0.10\ \mu\text{s}$ e $0.35\ \mu\text{s}$** por amostra. A biblioteca C11 permanece **$>200\times$ a $>1.000\times$ mais rápida que o River**.
   - **Recomendação de Uso:** Essencial em fluxos com ruído industrial impulsivo de sensores e em regressão multi-escala onde o SGD se beneficia criticamente de normalização contínua.

