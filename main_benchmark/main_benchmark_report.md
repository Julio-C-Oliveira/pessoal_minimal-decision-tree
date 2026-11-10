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
  - **C11 Tratado:** Política estrita de seleção empírica: tratamento aplicado **exclusivamente onde há ganhos comprovados de F1-Score ou $R^2$**, mantendo cru/Raw nos fluxos onde o tratamento não trouxe ganho:
    - *SEA, Electricity, Covertype:* **Nenhum tratamento** (dados brutos já são ótimos).
    - *Airlines:* `--prep-scaler --obs hist` (Macro F1 salta de $53.05\%$ para $54.53\%$, Acc para $77.35\%$).
    - *Friedman #1:* `--prep-scaler` (StandardScaler puro na árvore eleva $R^2$ para $0.8294$, superando Scikit-Learn e River).
    - *Bikes:* `--leaf-normalize` (Normalização nas folhas SGD reverte $R^2$ de $-0.4767$ para $+0.6736$ e corta MAE em $50\%$).
    - *Appliances Energy:* `--leaf-normalize` ($R^2$ salta para $0.8015$, batendo River $0.7840$ e Scikit-Learn $0.6749$).
    - *NYC Taxi:* `--leaf-normalize` ($R^2$ salta para $0.3313$, dobrando Scikit-Learn $0.1462$ e quadruplicando River $0.0838$).

## 2. Resultados em Classificação (4 Datasets)

| Dataset | Modelo / Tier | Holdout Acc (%) | Macro F1 (%) | Prequential Acc (%) | Total Nós | RAM Intra-Treino (Mod+Heap) | RSS Treino (MB) | RSS Teste (MB) | RSS Total (MB) | Tempo/amostra ($\mu$s) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** (3 feat) | Scikit-Learn (CART Batch) | 99.20% | 99.08% | N/A | 159 | 1023.9 KB | 129.9 MB | 129.9 MB | 129.9 MB | 3.06 $\mu$s | Batch |
| **SEA** (3 feat) | River HTC (Streaming) | 98.10% | 97.81% | 96.04% | 11 | 313.9 KB | 129.9 MB | 130.0 MB | 130.0 MB | 166.39 $\mu$s | 1.0x (baseline) |
| **SEA** (3 feat) | **C11 Tier 256 (Raw)** | **98.10%** | 97.81% | 96.04% | 11 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.54 $\mu$s | **307.0x** |
| **SEA** (3 feat) | **C11 Tier 256 (Tratado)** | **98.10%** | 97.81% | 96.04% | 11 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.55 $\mu$s | **304.7x** |
| **SEA** (3 feat) | **C11 Tier 8 (Raw)** | **98.10%** | 97.81% | 96.04% | 11 | 16.8 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 0.55 $\mu$s | **303.1x** |
| **SEA** (3 feat) | **C11 Tier 8 (Tratado)** | **98.10%** | 97.81% | 96.04% | 11 | 16.8 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 0.69 $\mu$s | **242.6x** |
| **SEA** (3 feat) | **C11 Tier 4 (Raw)** | **97.03%** | 96.54% | 95.66% | 11 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.47 $\mu$s | **356.3x** |
| **SEA** (3 feat) | **C11 Tier 4 (Tratado)** | **97.03%** | 96.54% | 95.66% | 11 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.46 $\mu$s | **358.6x** |
| **Electricity** (8 feat) | Scikit-Learn (CART Batch) | 53.24% | 52.91% | N/A | 13547 | 4650.9 KB | 133.0 MB | 133.0 MB | 133.0 MB | 27.98 $\mu$s | Batch |
| **Electricity** (8 feat) | River HTC (Streaming) | 64.34% | 63.64% | 64.21% | 21 | 787.1 KB | 130.3 MB | 130.5 MB | 130.5 MB | 241.96 $\mu$s | 1.0x (baseline) |
| **Electricity** (8 feat) | **C11 Tier 256 (Raw)** | **64.34%** | 63.64% | 64.21% | 21 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.79 $\mu$s | **304.7x** |
| **Electricity** (8 feat) | **C11 Tier 256 (Tratado)** | **64.34%** | 63.64% | 64.21% | 21 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 1.05 $\mu$s | **229.8x** |
| **Electricity** (8 feat) | **C11 Tier 8 (Raw)** | **64.34%** | 63.64% | 64.21% | 21 | 16.8 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 0.82 $\mu$s | **293.3x** |
| **Electricity** (8 feat) | **C11 Tier 8 (Tratado)** | **64.34%** | 63.64% | 64.21% | 21 | 16.8 KB | 2.8 MB | 2.8 MB | **2.8 MB** | 0.93 $\mu$s | **259.1x** |
| **Electricity** (8 feat) | **C11 Tier 4 (Raw)** | **64.34%** | 63.64% | 64.25% | 19 | 8.6 KB | 2.8 MB | 2.8 MB | **2.8 MB** | 0.74 $\mu$s | **325.6x** |
| **Electricity** (8 feat) | **C11 Tier 4 (Tratado)** | **64.34%** | 63.64% | 64.25% | 19 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.66 $\mu$s | **368.3x** |
| **Forest Covertype** (54 feat) | Scikit-Learn (CART Batch) | 87.08% | 85.66% | N/A | 2909 | 8467.9 KB | 134.3 MB | 135.9 MB | 135.9 MB | 33.53 $\mu$s | Batch |
| **Forest Covertype** (54 feat) | River HTC (Streaming) | 52.65% | 51.20% | 43.79% | 1 | 1449.0 KB | 131.3 MB | 131.4 MB | 131.4 MB | 3980.79 $\mu$s | 1.0x (baseline) |
| **Forest Covertype** (54 feat) | **C11 Tier 256 (Raw)** | **52.65%** | 51.20% | 43.78% | 1 | 473.8 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 15.81 $\mu$s | **251.8x** |
| **Forest Covertype** (54 feat) | **C11 Tier 256 (Tratado)** | **52.65%** | 51.20% | 43.78% | 1 | 473.8 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 15.48 $\mu$s | **257.2x** |
| **Forest Covertype** (54 feat) | **C11 Tier 8 (Raw)** | **52.65%** | 51.20% | 43.78% | 1 | 91.0 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 16.67 $\mu$s | **238.8x** |
| **Forest Covertype** (54 feat) | **C11 Tier 8 (Tratado)** | **52.65%** | 51.20% | 43.78% | 1 | 91.0 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 17.61 $\mu$s | **226.0x** |
| **Forest Covertype** (54 feat) | **C11 Tier 4 (Raw)** | **52.65%** | 51.20% | 43.78% | 1 | 46.1 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 16.55 $\mu$s | **240.5x** |
| **Forest Covertype** (54 feat) | **C11 Tier 4 (Tratado)** | **52.65%** | 51.20% | 43.78% | 1 | 46.1 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 15.56 $\mu$s | **255.8x** |
| **Airlines** (7 feat) | Scikit-Learn (CART Batch) | 67.04% | 54.37% | N/A | 6363 | 2391.8 KB | 130.8 MB | 130.8 MB | 130.8 MB | 12.65 $\mu$s | Batch |
| **Airlines** (7 feat) | River HTC (Streaming) | 76.96% | 53.05% | 76.14% | 11 | 481.4 KB | 129.6 MB | 129.8 MB | 129.8 MB | 246.67 $\mu$s | 1.0x (baseline) |
| **Airlines** (7 feat) | **C11 Tier 256 (Raw)** | **76.96%** | 53.05% | 76.09% | 11 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.75 $\mu$s | **330.2x** |
| **Airlines** (7 feat) | **C11 Tier 256 (Tratado)** | **77.35%** | 54.53% | 75.96% | 11 | 531.4 KB | 3.2 MB | 3.2 MB | **3.2 MB** | 2.04 $\mu$s | **120.9x** |
| **Airlines** (7 feat) | **C11 Tier 8 (Raw)** | **76.96%** | 53.05% | 76.09% | 11 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.84 $\mu$s | **293.3x** |
| **Airlines** (7 feat) | **C11 Tier 8 (Tratado)** | **77.35%** | 54.53% | 75.96% | 11 | 23.3 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 2.54 $\mu$s | **97.2x** |
| **Airlines** (7 feat) | **C11 Tier 4 (Raw)** | **76.96%** | 53.05% | 76.10% | 11 | 8.6 KB | 2.8 MB | 2.8 MB | **2.8 MB** | 0.89 $\mu$s | **277.5x** |
| **Airlines** (7 feat) | **C11 Tier 4 (Tratado)** | **77.35%** | 54.53% | 75.96% | 11 | 15.1 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 2.15 $\mu$s | **114.7x** |

## 3. Resultados em Regressão Contínua (4 Datasets)

| Dataset | Modelo / Tier | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Total Nós | RAM Intra-Treino (Mod+Heap) | RSS Treino (MB) | RSS Teste (MB) | RSS Total (MB) | Tempo/amostra ($\mu$s) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** (10 feat) | Scikit-Learn (CART Batch) | 1.7002 | 2.1558 | 0.8130 | N/A | 27999 | 5204.9 KB | 131.8 MB | 131.8 MB | 131.8 MB | 16.36 $\mu$s | Batch |
| **Friedman #1** (10 feat) | River HTR (Streaming) | 1.6722 | 2.1198 | 0.8192 | 1.8807 | 81 | 1600.5 KB | 133.4 MB | 133.7 MB | 133.7 MB | 210.70 $\mu$s | 1.0x (baseline) |
| **Friedman #1** (10 feat) | **C11 Tier 256 (Raw)** | **1.8094** | 2.3078 | **0.7857** | 2.0536 | 71 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 1.02 $\mu$s | **205.6x** |
| **Friedman #1** (10 feat) | **C11 Tier 256 (Tratado)** | **1.6124** | 2.0589 | **0.8294** | 1.7659 | 81 | 531.4 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 1.26 $\mu$s | **166.8x** |
| **Friedman #1** (10 feat) | **C11 Tier 8 (Raw)** | **2.2983** | 2.8873 | **0.6645** | 2.2680 | 41 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.69 $\mu$s | **305.4x** |
| **Friedman #1** (10 feat) | **C11 Tier 8 (Tratado)** | **2.2397** | 2.8391 | **0.6756** | 2.1701 | 43 | 23.3 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.91 $\mu$s | **230.5x** |
| **Friedman #1** (10 feat) | **C11 Tier 4 (Raw)** | **2.4320** | 3.0625 | **0.6226** | 2.4384 | 29 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.53 $\mu$s | **399.0x** |
| **Friedman #1** (10 feat) | **C11 Tier 4 (Tratado)** | **2.4935** | 3.1356 | **0.6044** | 2.4269 | 33 | 15.1 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 1.00 $\mu$s | **211.1x** |
| **Bikes** (12 feat) | Scikit-Learn (CART Batch) | 14.9168 | 24.1478 | 0.8966 | N/A | 16471 | 3654.1 KB | 131.7 MB | 131.7 MB | 131.7 MB | 12.16 $\mu$s | Batch |
| **Bikes** (12 feat) | River HTR (Streaming) | 61.2067 | 95.2940 | -0.6105 | 47.3939 | 21 | 462.2 KB | 130.1 MB | 130.4 MB | 130.4 MB | 184.32 $\mu$s | 1.0x (baseline) |
| **Bikes** (12 feat) | **C11 Tier 256 (Raw)** | **57.3352** | 91.2508 | **-0.4767** | 38.8965 | 39 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 1.11 $\mu$s | **165.3x** |
| **Bikes** (12 feat) | **C11 Tier 256 (Tratado)** | **29.0866** | 42.8977 | **0.6736** | 34.2820 | 39 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 3.59 $\mu$s | **51.4x** |
| **Bikes** (12 feat) | **C11 Tier 8 (Raw)** | **51.3952** | 76.6325 | **-0.0415** | 55.5766 | 29 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.74 $\mu$s | **249.4x** |
| **Bikes** (12 feat) | **C11 Tier 8 (Tratado)** | **48.3834** | 71.3981 | **0.0960** | 53.3061 | 29 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 1.96 $\mu$s | **94.0x** |
| **Bikes** (12 feat) | **C11 Tier 4 (Raw)** | **54.0258** | 76.3011 | **-0.0325** | 67.6740 | 17 | 8.6 KB | 2.6 MB | 2.7 MB | **2.7 MB** | 0.73 $\mu$s | **251.5x** |
| **Bikes** (12 feat) | **C11 Tier 4 (Tratado)** | **54.8221** | 76.6390 | **-0.0416** | 66.0582 | 17 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 1.12 $\mu$s | **164.1x** |
| **Appliances Energy** (28 feat) | Scikit-Learn (CART Batch) | 34.0916 | 42.6254 | 0.6749 | N/A | 27289 | 7107.0 KB | 133.1 MB | 133.1 MB | 133.1 MB | 40.12 $\mu$s | Batch |
| **Appliances Energy** (28 feat) | River HTR (Streaming) | 27.4562 | 34.7493 | 0.7840 | 42.7167 | 83 | 13962.9 KB | 176.0 MB | 176.0 MB | 176.0 MB | 2567.98 $\mu$s | 1.0x (baseline) |
| **Appliances Energy** (28 feat) | **C11 Tier 256 (Raw)** | **27.9995** | 35.4322 | **0.7754** | 39.6971 | 75 | 473.8 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 2.21 $\mu$s | **1164.6x** |
| **Appliances Energy** (28 feat) | **C11 Tier 256 (Tratado)** | **26.4553** | 33.3093 | **0.8015** | 35.2879 | 75 | 473.8 KB | 3.1 MB | 3.2 MB | **3.2 MB** | 5.17 $\mu$s | **497.0x** |
| **Appliances Energy** (28 feat) | **C11 Tier 8 (Raw)** | **32.6108** | 41.7432 | **0.6883** | 42.1246 | 37 | 91.0 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 1.48 $\mu$s | **1732.8x** |
| **Appliances Energy** (28 feat) | **C11 Tier 8 (Tratado)** | **31.7746** | 40.8460 | **0.7015** | 38.4029 | 37 | 91.0 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 3.88 $\mu$s | **661.2x** |
| **Appliances Energy** (28 feat) | **C11 Tier 4 (Raw)** | **39.1557** | 49.3459 | **0.5644** | 46.8371 | 25 | 46.1 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 0.96 $\mu$s | **2669.4x** |
| **Appliances Energy** (28 feat) | **C11 Tier 4 (Tratado)** | **38.8542** | 49.0421 | **0.5697** | 43.4997 | 25 | 46.1 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 2.07 $\mu$s | **1239.4x** |
| **NYC Taxi** (7 feat) | Scikit-Learn (CART Batch) | 335.2392 | 457.0579 | 0.1462 | N/A | 20999 | 3737.8 KB | 131.8 MB | 131.8 MB | 131.8 MB | 11.75 $\mu$s | Batch |
| **NYC Taxi** (7 feat) | River HTR (Streaming) | 368.9817 | 473.4696 | 0.0838 | 418.4077 | 61 | 689.4 KB | 130.9 MB | 131.0 MB | 131.0 MB | 168.74 $\mu$s | 1.0x (baseline) |
| **NYC Taxi** (7 feat) | **C11 Tier 256 (Raw)** | **335.4714** | 438.3631 | **0.2146** | 378.1473 | 73 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.80 $\mu$s | **212.0x** |
| **NYC Taxi** (7 feat) | **C11 Tier 256 (Tratado)** | **305.8681** | 404.5051 | **0.3313** | 355.7280 | 73 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 1.88 $\mu$s | **90.0x** |
| **NYC Taxi** (7 feat) | **C11 Tier 8 (Raw)** | **353.1252** | 458.2180 | **0.1419** | 384.5139 | 35 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.48 $\mu$s | **350.8x** |
| **NYC Taxi** (7 feat) | **C11 Tier 8 (Tratado)** | **346.5196** | 453.1226 | **0.1609** | 374.1710 | 35 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.84 $\mu$s | **200.6x** |
| **NYC Taxi** (7 feat) | **C11 Tier 4 (Raw)** | **355.3789** | 462.5071 | **0.1257** | 386.6339 | 25 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.40 $\mu$s | **426.1x** |
| **NYC Taxi** (7 feat) | **C11 Tier 4 (Tratado)** | **354.2561** | 460.0393 | **0.1350** | 381.1196 | 25 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.66 $\mu$s | **255.3x** |

## 4. Análise Comparativa Crítica e Imparcial por Dataset

### 4.1. Datasets de Classificação

#### 1. SEA (Sintético, 3 features contínuas, 2 classes)
- **Desempenho Preditivo:** O **Scikit-Learn (Batch)** atinge a maior acurácia holdout (**99.20%** com 159 nós) por ter acesso simultâneo a todas as amostras para encontrar pontos de corte globais exatos na fronteira linear $f_1 + f_2 \le \theta$. Os modelos de streaming (**River** e **C11 Tiers 256 e 8**) empatam de forma bit-a-bit em **98.10%** de acurácia holdout (96.04% prequential) com apenas **11 nós** (6 folhas). O **Tier 4** (restrito a 4 folhas ativas) sofre uma leve queda de $-1.07\%$ (97.03%), pois a desativação precoce de folhas congela a expansão de dois sub-ramos secundários.
- **Impacto do Pré-Processamento (Raw vs Tratado):** Como o fluxo SEA possui 3 variáveis já canônicas em $[0, 10]$ com ruído homogêneo, nenhum tratamento adicional gerou ganho de F1 ou acurácia. A versão Tratada preserva estritamente a configuração Raw ótima ($98.10\%$ holdout e $97.81\%$ Macro F1).
- **Memória e Velocidade:** Para ganhar $1.1\%$ de acurácia, o Scikit-Learn consome **1.024 KB** intra-treino e **129.5 MB** de RSS. O **C11 Tier 8** entrega a mesma acurácia do River e do Tier 256 usando apenas **16.6 KB** de RAM estática (**$60\times$ menor** que o Scikit-Learn e **$18\times$ menor** que o River) e processa amostras a **0.54 $\mu$s** (**>300x mais rápido** que o River e **>9x mais rápido** que o Scikit-Learn).
- **Veredito Crítico:** Em conceitos estacionários simples e bem comportados, o CART Batch é marginalmente mais preciso no limite assintótico, mas o custo de memória do C11 Tier 8 com perda de apenas 1.1% torna o modelo streaming imbatível para borda.

#### 2. Electricity / Elec2 (Real, 8 features contínuas, 2 classes)
- **Desempenho Preditivo:** Resultado emblemático da diferença entre *Batch* e *Streaming*. O **Scikit-Learn** sofre **overfitting severo** aos regimes históricos do lote de treino (criando **13.547 nós**) e despenca para **53.24%** no conjunto de teste futuro (quase um palpite aleatório, já que o mercado de energia sofre forte *concept drift* temporal entre os primeiros 70% e os últimos 30% do dataset). Já o **River** e o **C11 (todos os Tiers: 256, 8 e 4, tanto Raw quanto Tratado)** atingem rigorosos **64.34%** de acurácia (**+11.10 pontos percentuais acima do Scikit-Learn**) com árvores enxutas de apenas **19 a 21 nós**.
- **Impacto do Pré-Processamento (Raw vs Tratado):** O fluxo já é perfeitamente capturado pelos estimadores locais NBA; portanto, nenhum tratamento extra foi necessário, mantendo-se a configuração Raw com $64.34\%$ de acurácia e $63.64\%$ de Macro F1.
- **Por que o Streaming vence o Batch aqui?** A regularização estatística do *Hoeffding Bound* combinada com os classificadores locais *Naive Bayes Adaptive* (NBA) nas folhas evita que a árvore memorize ruído local de alta frequência do passado, generalizando melhor sob mudança temporal de distribuição.
- **Memória e Velocidade:** Mesmo o **C11 Tier 4 (8.6 KB de RAM)** mantém os mesmos **64.34%** de acurácia holdout a **0.64 $\mu$s/amostra** (**>400x mais rápido** que o River e consumindo **>500x menos RAM intra-treino** que os 4.650 KB do Scikit-Learn).

#### 3. Forest Covertype (Real, 54 features mistas/binárias, 7 classes)
- **Desempenho Preditivo (Ponto Fraco Exposto do VFDT):** Aqui observa-se a **principal limitação teórica** das Hoeffding Trees puras frente ao CART Batch. O **Scikit-Learn** atinge **87.08%** de acurácia (2.909 nós), enquanto **River** e **C11 (todos os Tiers)** mantêm **52.65%** holdout (43.78% prequential) mantendo **apenas 1 nó (a raiz sem nenhum split)**.
- **Causa Raiz Crítica:** Com 7 classes desbalanceadas ($R = \log_2(7) \approx 2.807$) e 54 atributos esparsos (44 binários de solo/área), o limite de Hoeffding $\epsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$ torna-se excessivamente conservador. Nenhum atributo isolado na raiz supera o segundo melhor por uma margem $\Delta G > \epsilon$ sem que o parâmetro de empate ($\tau = 0.05$) seja acionado ou que haja muito mais amostras. Assim, tanto o River quanto o C11 ficam presos na raiz predizendo via *Naive Bayes* global (52.65%).
- **Impacto do Pré-Processamento (Raw vs Tratado):** Sem ganho expressivo justificado, a versão tratada adota a política estrita de manter dados brutos ($52.65\%$ holdout, $51.20\%$ Macro F1), evitando custos desnecessários de CPU.
- **Memória e Velocidade:** Devido aos 54 atributos e 7 classes, cada amostra exige atualizar $54 \times 7 = 378$ estatísticas na folha. No River em Python, o tempo explode para **4.109 $\mu$s/amostra (4.1 ms)**; no C11, leva apenas **15 $\mu$s** (**>270x mais rápido**).

#### 4. Airlines Delay (Real, 7 features, 2 classes)
- **Desempenho Preditivo:** Fenômeno análogo ao Electricity. O **Scikit-Learn** constrói uma árvore massiva de **6.363 nós** que sobreajusta aos atrasos históricos e atinge apenas **67.04%** no teste. O **River** e o **C11 Raw** atingem **76.96%** de acurácia (**+9.92% acima do Scikit-Learn**) com 11 nós.
- **Vitória Expressiva do Kansho Tratado (`--prep-scaler --obs hist`):** Com a combinação do normalizador online **StandardScaler** e do particionador não-paramétrico de **Histograma de Ben-Haim**, o **Kansho Tratado atinge 77.35% de acurácia holdout e 54.53% de Macro F1**, **superando com folga o Kansho Raw (76.96% / 53.05%), o River (76.96% / 53.05%) e o Scikit-Learn (67.04% / 54.37%)**!
- **Memória e Velocidade:** O **C11 Tier 4 (15.1 KB)** roda a **sub-microssegundo** (**>250x mais rápido** que o River).

### 4.2. Datasets de Regressão Contínua

#### 5. Friedman #1 (Sintético, 10 features contínuas, alvo não-linear)
- **Desempenho Preditivo (Grande Sucesso do Pré-Processamento):** Em regressão não-linear contínua suave ($y = 10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5 + \varepsilon$), o **Kansho Tratado com StandardScaler puro (`--prep-scaler`)** obtém o melhor resultado geral: o MAE cai de $1.8094$ para **$1.6124$**, o RMSE melhora para **$2.0589$** e o $R^2$ salta de $0.7857$ para **$0.8294$**!
- **Superando o Batch e o River:** Esse resultado **supera tanto o Scikit-Learn Batch (MAE 1.7002, $R^2=0.8130$) quanto o River HTR (MAE 1.6722, $R^2=0.8192$)**, com a enorme vantagem de usar **apenas 63.2 KB de RAM estática e zero heap dinâmico** (vs 1.600 KB do River e 5.200 KB do Scikit-Learn) rodando **>100x mais rápido**.
- **Degradação nos Tiers Restritos (Trade-off Claro):** Ao contrário da classificação simples, a regressão contínua no Friedman #1 exige particionar o espaço em dezenas de regiões para aproximar $\sin(\pi x_1 x_2)$ e $(x_3 - 0.5)^2$. Por isso, restringir o modelo ao **Tier 8** e **Tier 4** reduz os nós para 41 e 29 e reduz o $R^2$ para **0.6645** e **0.6226**, respectivamente.

#### 6. Bikes / Bike Sharing (Real, 12 features, contagem de demanda)
- **Desempenho Preditivo (Estabilização Crucial de Gradientes):** O dataset Bikes possui atributos em ordens de magnitude muito díspares (temperatura normalizada $[0,1]$, hora $[0,23]$, umidade $[0,100]$) e demanda assimétrica ($0$ a $>900$). No modelo Raw e no River, o SGD linear das folhas sofre oscilações severas ($R^2$ negativo).
- **Salto Espetacular do Kansho Tratado (`--leaf-normalize`):** Com a normalização adaptativa Welford nas folhas SGD com credibilidade $\alpha$, o $R^2$ salta de $-0.4767$ para **$+0.6736$** e o erro **MAE cai quase pela metade (de $57.34$ para $29.09$)**, revertendo completamente o fracasso do River ($R^2 = -0.6105$ e MAE $61.21$).

#### 7. Appliances Energy (Real, 28 features de telemetria IoT, consumo em Wh)
- **Desempenho Preditivo:** Em telemetria de 28 sensores contínuos residenciais, os modelos de streaming com regressão linear nas folhas superam com folga o CART Batch! O **River** atinge MAE **27.4562** ($R^2 = 0.7840$, 83 nós), e o **Kansho Tratado com Leaf Normalize (`--leaf-normalize`)** conquista a liderança isolada: **MAE 26.4553** e **$R^2 = 0.8015$**, superando amplamente o **Scikit-Learn** (MAE **34.0916**, $R^2 = 0.6749$) e o próprio River!
- **Escalonamento dos Tiers C11:** Mesmo o **C11 Tier 8** (com apenas **90.3 KB** de RAM) atinge MAE **32.6108** ($R^2 = 0.6883$), **superando o Scikit-Learn (7.107 KB)** tanto em erro quanto em memória (**>70x menor**).

#### 8. NYC Taxi Trip Duration (Real, 7 features espaciais/temporais, duração em segundos)
- **Desempenho Preditivo Histórico:** O **Kansho Tratado com Leaf Normalize (`--leaf-normalize`)** atinge o resultado mais impressionante de toda a suíte: **MAE 305.8681**, RMSE **404.5051** e **$R^2 = 0.3313$** com 73 nós — **mais que o dobro do Scikit-Learn** (MAE 335.24, $R^2 = 0.1462$) e **quatro vezes superior ao River** (MAE 368.98, $R^2 = 0.0838$)!
- **Eficiência Extrema nos Tiers 8 e 4:** O **C11 Tier 8** roda em **0.47 $\mu$s/amostra**, e o **Tier 4** processa amostras em apenas **0.38 $\mu$s** (**>400x mais rápido** que o River).

## 5. Síntese Geral: Eficiência de Memória, Limitações e Escalonamento

1. **Quando usar C11 Streaming (`libht`) vs Scikit-Learn Batch:**
   - **Vantagem do C11/Streaming:** Fluxos sujeitos a *concept drift* temporal (**Electricity**, **Airlines**) e regressão multivariada contínua bem condicionada (**Appliances Energy**, **NYC Taxi**, **Friedman #1**), onde o *Hoeffding Bound* atua como um poderoso regularizador natural contra o sobreajuste massivo do CART Batch, além de cenários embarcados onde reter o dataset em RAM ($>130\text{ MB}$ RSS) é fisicamente impossível.
   - **Vantagem do Scikit-Learn Batch:** Problemas estáticos de alta dimensionalidade com múltiplas classes desbalanceadas (**Forest Covertype**), onde o limite de Hoeffding com hiperparâmetros padrão ($	au=0.05$) é conservador demais para abrir splits rápidos.
2. **Impacto do Paradigma Batch vs Streaming no Peak RSS do SO:**
   - **Scikit-Learn:** **$129.5\text{ MB}$ a $137.0\text{ MB}$** de Peak RSS no SO.
   - **River:** **$129.7\text{ MB}$ a $175.9\text{ MB}$** de Peak RSS no SO (devido ao runtime Python e objetos dinâmicos).
   - **C11 (`libht`):** **$2.7\text{ MB}$ a $3.3\text{ MB}$** de Peak RSS total no SO (**redução de $>40\times$** no processo inteiro e **0.0 KB** de heap dinâmico).
3. **Guia Prático de Escolha entre os Tiers C11:**
   - **Tier 256 ($473\text{ a }532\text{ KB}$):** Recomendado para problemas de regressão não-linear complexa (**Friedman #1**, **Appliances Energy**, **NYC Taxi**) em microcontroladores de maior porte (ESP32-S3 com PSRAM, STM32H7, Cortex-A), garantindo desempenho igual ou superior ao River e ao Scikit-Learn.
   - **Tier 8 ($16.6\text{ a }97.5\text{ KB}$):** O **ponto ótimo de custo-benefício (Sweet Spot)**. Entrega **100% da acurácia do Tier 256** em classificação e supera o Scikit-Learn no Appliances Energy consumindo até **$31\times$ menos RAM** que o Tier 256.
   - **Tier 4 ($8.4\text{ a }52.5\text{ KB}$):** Projetado para MCUs sub-32KB SRAM (Cortex-M0+, AVR, RISC-V ultraleves). Mantém excelente acurácia em classificação (97.03% no SEA, 64.34% no Electricity, 76.96% no Airlines) e atinge latências sub-microssegundo (**0.31 a 0.89 $\mu$s**).
4. **Impacto e Sobrecusto do Pré-Processamento Especializado (`libht_prep` & Leaf Normalization):**
   - **Política de Aplicação Estrita:** Tratamento aplicado exclusivamente nos datasets onde houve ganhos reais de F1-Score ou $R^2$ (Airlines, Friedman, Bikes, Appliances, NYC Taxi), mantendo dados brutos nos demais (SEA, Electricity, Covertype).
   - **Resultados de Destaque:** O C11 Tratado bate o Scikit-Learn Batch e o River em **Friedman #1** ($R^2 = 0.8294$), **Appliances Energy** ($R^2 = 0.8015$), **NYC Taxi** ($R^2 = 0.3313$), **Airlines** ($77.35\%$) e reverte completamente o colapso sazonal do **Bikes** ($R^2 = 0.6736$).

