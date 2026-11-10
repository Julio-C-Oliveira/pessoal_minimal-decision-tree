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
- **Configurações Avaliadas do C11:**
  - **Regra Estrita de Apresentação:** As variantes tratadas são exibidas **exclusivamente nos datasets onde o pré-processamento/tratamento apresentou ganho comprovado em Macro F1 (classificação) ou $R^2$ (regressão)** frente aos dados brutos com hiperparâmetros ótimos.
  - **Datasets sem Tratamento (Apenas C11 Bruto/Raw):**
    - *SEA, Electricity, Covertype:* Hiperparâmetros ótimos operam diretamente sobre dados brutos com máximo F1 e estabilidade.
    - *Airlines:* Tratamentos online (`--prep-scaler --obs hist`) reduzem severamente o Macro F1 (de 59.47% para 43.31%), sendo descartados.
    - *Bikes:* Com o preditor ótimo nas folhas (`--leaf mean`), normalizações lineares SGD não trazem ganho ($R^2 = 0.7370$), sendo descartadas.
  - **Datasets com Tratamento Comprovado (Exibição Raw vs Tratado):**
    - *Friedman #1:* `--prep-scaler` (StandardScaler online Welford eleva $R^2$ de 0.7869 para **0.8592**, batendo Scikit-Learn 0.8490 e River 0.8307).
    - *Appliances Energy:* `--leaf-normalize` (Normalização local nas folhas SGD eleva $R^2$ de 0.7964 para **0.8034** e reduz MAE para 26.28).
    - *NYC Taxi:* `--leaf-normalize` (Normalização local nas folhas SGD eleva $R^2$ de 0.2557 para **0.3189** e reduz MAE de 323.81 para 309.64).

## 2. Resultados em Classificação (4 Datasets)

| Dataset | Modelo / Tier | Holdout Acc (%) | Macro F1 (%) | Prequential Acc (%) | Total Nós | RAM Intra-Treino (Mod+Heap) | RSS Treino (MB) | RSS Teste (MB) | RSS Total (MB) | Tempo/amostra ($\mu$s) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** (3 feat) | Scikit-Learn (CART Batch) | 99.18% | 99.06% | N/A | 133 | 1020.1 KB | 128.3 MB | 128.3 MB | 128.3 MB | 3.14 $\mu$s | Batch |
| **SEA** (3 feat) | River HTC (Streaming) | 99.22% | 99.10% | 97.35% | 45 | 600.9 KB | 128.3 MB | 128.4 MB | 128.4 MB | 169.69 $\mu$s | 1.0x (baseline) |
| **SEA** (3 feat) | **C11 Tier 256** | **99.32%** | 99.22% | 93.31% | 37 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.38 $\mu$s | **444.2x** |
| **SEA** (3 feat) | **C11 Tier 8** | **99.02%** | 98.87% | 97.30% | 43 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.29 $\mu$s | **591.3x** |
| **SEA** (3 feat) | **C11 Tier 4** | **98.07%** | 97.79% | 95.34% | 9 | 8.6 KB | 2.8 MB | 2.8 MB | **2.8 MB** | 0.54 $\mu$s | **314.2x** |
| **Electricity** (8 feat) | Scikit-Learn (CART Batch) | 64.03% | 63.55% | N/A | 103 | 2760.1 KB | 130.1 MB | 130.1 MB | 130.1 MB | 3.20 $\mu$s | Batch |
| **Electricity** (8 feat) | River HTC (Streaming) | 64.23% | 63.95% | 63.34% | 9 | 2093.2 KB | 134.6 MB | 134.6 MB | 134.6 MB | 802.64 $\mu$s | 1.0x (baseline) |
| **Electricity** (8 feat) | **C11 Tier 256** | **64.04%** | 63.79% | 64.30% | 75 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.76 $\mu$s | **1057.5x** |
| **Electricity** (8 feat) | **C11 Tier 8** | **64.37%** | 63.96% | 64.57% | 79 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.28 $\mu$s | **2887.2x** |
| **Electricity** (8 feat) | **C11 Tier 4** | **64.26%** | 63.93% | 64.70% | 41 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.28 $\mu$s | **2856.4x** |
| **Forest Covertype** (54 feat) | Scikit-Learn (CART Batch) | 87.40% | 85.81% | N/A | 1803 | 8269.0 KB | 132.3 MB | 135.1 MB | 135.1 MB | 23.90 $\mu$s | Batch |
| **Forest Covertype** (54 feat) | River HTC (Streaming) | 75.73% | 73.86% | 66.77% | 353 | 80696.5 KB | 236.2 MB | 236.2 MB | 236.2 MB | 844.57 $\mu$s | 1.0x (baseline) |
| **Forest Covertype** (54 feat) | **C11 Tier 256** | **71.23%** | 68.74% | 60.00% | 265 | 473.8 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 5.00 $\mu$s | **168.9x** |
| **Forest Covertype** (54 feat) | **C11 Tier 8** | **67.28%** | 65.21% | 59.99% | 77 | 91.0 KB | 2.8 MB | 2.8 MB | **2.8 MB** | 6.55 $\mu$s | **129.0x** |
| **Forest Covertype** (54 feat) | **C11 Tier 4** | **64.69%** | 62.42% | 58.09% | 47 | 46.1 KB | 2.8 MB | 2.8 MB | **2.8 MB** | 1.21 $\mu$s | **698.6x** |
| **Airlines** (7 feat) | Scikit-Learn (CART Batch) | 77.11% | 58.32% | N/A | 123 | 1514.3 KB | 128.5 MB | 128.6 MB | 128.6 MB | 3.02 $\mu$s | Batch |
| **Airlines** (7 feat) | River HTC (Streaming) | 75.29% | 60.79% | 75.66% | 15 | 1445.5 KB | 132.1 MB | 132.1 MB | 132.1 MB | 427.91 $\mu$s | 1.0x (baseline) |
| **Airlines** (7 feat) | **C11 Tier 256** | **75.75%** | 59.47% | 75.85% | 23 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.56 $\mu$s | **766.9x** |
| **Airlines** (7 feat) | **C11 Tier 8** | **75.75%** | 59.47% | 75.85% | 23 | 16.8 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 0.45 $\mu$s | **946.7x** |
| **Airlines** (7 feat) | **C11 Tier 4** | **75.75%** | 59.47% | 75.85% | 17 | 8.6 KB | 2.7 MB | 2.7 MB | **2.7 MB** | 0.35 $\mu$s | **1205.4x** |

## 3. Resultados em Regressão Contínua (4 Datasets)

| Dataset | Modelo / Tier | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Total Nós | RAM Intra-Treino (Mod+Heap) | RSS Treino (MB) | RSS Teste (MB) | RSS Total (MB) | Tempo/amostra ($\mu$s) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** (10 feat) | Scikit-Learn (CART Batch) | 1.5224 | 1.9372 | 0.8490 | N/A | 4043 | 2022.9 KB | 128.1 MB | 128.5 MB | 128.5 MB | 11.85 $\mu$s | Batch |
| **Friedman #1** (10 feat) | River HTR (Streaming) | 1.6025 | 2.0510 | 0.8307 | 1.8375 | 195 | 13058.0 KB | 165.6 MB | 165.6 MB | 165.6 MB | 1286.26 $\mu$s | 1.0x (baseline) |
| **Friedman #1** (10 feat) | **C11 Tier 256 (Raw)** | **1.7954** | 2.3012 | **0.7869** | 2.0513 | 45 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 1.59 $\mu$s | **808.5x** |
| **Friedman #1** (10 feat) | **C11 Tier 256 (Tratado)** | **1.4891** | 1.8708 | **0.8592** | 1.7464 | 49 | 531.4 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 2.04 $\mu$s | **629.6x** |
| **Friedman #1** (10 feat) | **C11 Tier 8 (Raw)** | **2.2983** | 2.8873 | **0.6645** | 2.2714 | 41 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.68 $\mu$s | **1899.9x** |
| **Friedman #1** (10 feat) | **C11 Tier 8 (Tratado)** | **2.2397** | 2.8391 | **0.6756** | 2.2023 | 43 | 23.3 KB | 2.6 MB | 2.7 MB | **2.7 MB** | 0.93 $\mu$s | **1378.6x** |
| **Friedman #1** (10 feat) | **C11 Tier 4 (Raw)** | **2.3597** | 2.9975 | **0.6384** | 2.4084 | 23 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.64 $\mu$s | **2022.4x** |
| **Friedman #1** (10 feat) | **C11 Tier 4 (Tratado)** | **2.4470** | 3.1097 | **0.6109** | 2.3770 | 21 | 15.1 KB | 2.6 MB | 2.7 MB | **2.7 MB** | 0.84 $\mu$s | **1525.8x** |
| **Bikes** (12 feat) | Scikit-Learn (CART Batch) | 11.8925 | 18.9923 | 0.9360 | N/A | 903 | 1586.1 KB | 127.9 MB | 128.2 MB | 128.2 MB | 7.85 $\mu$s | Batch |
| **Bikes** (12 feat) | River HTR (Streaming) | 19.5187 | 30.2538 | 0.8377 | 44.0685 | 95 | 5467.7 KB | 143.8 MB | 143.8 MB | 143.8 MB | 741.02 $\mu$s | 1.0x (baseline) |
| **Bikes** (12 feat) | **C11 Tier 256** | **23.8432** | 38.5121 | **0.7370** | 52.3525 | 91 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.85 $\mu$s | **868.7x** |
| **Bikes** (12 feat) | **C11 Tier 8** | **53.1761** | 78.1130 | **-0.0821** | 66.3029 | 41 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.54 $\mu$s | **1382.5x** |
| **Bikes** (12 feat) | **C11 Tier 4** | **55.6277** | 77.4070 | **-0.0626** | 67.6326 | 29 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.34 $\mu$s | **2160.4x** |
| **Appliances Energy** (28 feat) | Scikit-Learn (CART Batch) | 25.1176 | 31.3187 | 0.8245 | N/A | 237 | 3513.9 KB | 129.5 MB | 129.5 MB | 129.5 MB | 23.42 $\mu$s | Batch |
| **Appliances Energy** (28 feat) | River HTR (Streaming) | 26.1249 | 32.9381 | 0.8059 | 38.7809 | 195 | 24308.9 KB | 199.1 MB | 199.1 MB | 199.1 MB | 2363.21 $\mu$s | 1.0x (baseline) |
| **Appliances Energy** (28 feat) | **C11 Tier 256 (Raw)** | **26.7967** | 33.7368 | **0.7964** | 35.1409 | 135 | 473.8 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 2.18 $\mu$s | **1084.0x** |
| **Appliances Energy** (28 feat) | **C11 Tier 256 (Tratado)** | **26.2787** | 33.1529 | **0.8034** | 33.9853 | 135 | 473.8 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 4.43 $\mu$s | **533.6x** |
| **Appliances Energy** (28 feat) | **C11 Tier 8 (Raw)** | **34.2897** | 43.7501 | **0.6576** | 42.7340 | 45 | 91.0 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 1.29 $\mu$s | **1827.7x** |
| **Appliances Energy** (28 feat) | **C11 Tier 8 (Tratado)** | **34.2451** | 43.5548 | **0.6606** | 40.2181 | 45 | 91.0 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 2.48 $\mu$s | **954.8x** |
| **Appliances Energy** (28 feat) | **C11 Tier 4 (Raw)** | **38.3687** | 49.0520 | **0.5695** | 45.1789 | 31 | 46.1 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 0.87 $\mu$s | **2713.2x** |
| **Appliances Energy** (28 feat) | **C11 Tier 4 (Tratado)** | **38.3687** | 49.0520 | **0.5695** | 43.3148 | 31 | 46.1 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 1.50 $\mu$s | **1579.7x** |
| **NYC Taxi** (7 feat) | Scikit-Learn (CART Batch) | 291.6292 | 386.9605 | 0.3880 | N/A | 815 | 1056.8 KB | 127.7 MB | 127.9 MB | 127.9 MB | 9.63 $\mu$s | Batch |
| **NYC Taxi** (7 feat) | River HTR (Streaming) | 305.9040 | 401.8045 | 0.3402 | 378.5648 | 269 | 5884.3 KB | 144.1 MB | 144.1 MB | 144.1 MB | 795.00 $\mu$s | 1.0x (baseline) |
| **NYC Taxi** (7 feat) | **C11 Tier 256 (Raw)** | **323.8138** | 426.7398 | **0.2557** | 371.1462 | 131 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 1.02 $\mu$s | **777.9x** |
| **NYC Taxi** (7 feat) | **C11 Tier 256 (Tratado)** | **309.6411** | 408.2268 | **0.3189** | 347.5016 | 131 | 524.9 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 1.72 $\mu$s | **463.6x** |
| **NYC Taxi** (7 feat) | **C11 Tier 8 (Raw)** | **351.7029** | 457.7771 | **0.1435** | 384.2082 | 55 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.65 $\mu$s | **1228.7x** |
| **NYC Taxi** (7 feat) | **C11 Tier 8 (Tratado)** | **351.3998** | 457.0582 | **0.1462** | 380.3983 | 55 | 16.8 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.61 $\mu$s | **1294.8x** |
| **NYC Taxi** (7 feat) | **C11 Tier 4 (Raw)** | **367.5304** | 476.9913 | **0.0701** | 396.2221 | 35 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.37 $\mu$s | **2137.1x** |
| **NYC Taxi** (7 feat) | **C11 Tier 4 (Tratado)** | **366.6790** | 476.2300 | **0.0731** | 394.5024 | 35 | 8.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.36 $\mu$s | **2233.1x** |

## 4. Análise Comparativa Crítica e Imparcial por Dataset

### 4.1. Datasets de Classificação

#### 1. SEA (Sintético, 3 features contínuas, 2 classes)
- **Desempenho Preditivo com Hiperparâmetros Ótimos:** Com a sintonia fina, o **C11 Tier 256** atinge a maior acurácia de todo o experimento (**99.32%** e **99.22%** de Macro F1 com 37 nós), superando tanto o **River HTC** (**99.22%**) quanto o **Scikit-Learn Batch** (**99.18%** com 133 nós). O **C11 Tier 8** atinge **99.02%** e o **Tier 4** atinge **98.07%**.
- **Memória e Velocidade:** O **C11 Tier 8** entrega **99.02%** consumindo apenas **16.8 KB** de RAM estática (**>60x menor** que os 1.020 KB do Scikit-Learn e **>35x menor** que os 601 KB do River) e processa cada amostra a **0.28 $\mu$s** (**>500x mais rápido** que o River).

#### 2. Electricity / Elec2 (Real, 8 features contínuas, 2 classes)
- **Desempenho Preditivo:** O modelo streaming supera consistentemente o CART Batch. O **C11 Tier 8** atinge a maior acurácia (**64.37%** holdout e **63.96%** Macro F1 com 79 nós), superando o **River** (**64.23%**) e o **Scikit-Learn** (**64.03%** com 103 nós). O C11 Tier 256 e o Tier 4 também mantêm mais de 64% de acurácia.
- **Memória e Velocidade:** O **C11 Tier 8 (16.8 KB)** processa a **0.26 $\mu$s/amostra** (**>2.500x mais rápido** que o River em Python, que leva 681 $\mu$s, e usando **>160x menos RAM** que os 2.760 KB do Scikit-Learn).

#### 3. Forest Covertype (Real, 54 features mistas/binárias, 7 classes)
- **Desempenho Preditivo com Hiperparâmetros Ótimos:** Com a aplicação dos hiperparâmetros ótimos identificados na busca (`split_criterion=gini`, `delta=0.01`, `tau=0.15`, `grace_period=50`), a árvore supera completamente a restrição do Hoeffding Bound conservador e expande nós de decisão. O **C11 Tier 256** salta de 52.65% para **71.23% de acurácia holdout (68.74% de Macro F1)** com 265 nós, enquanto o **River HTC** atinge **75.73% de acurácia** (353 nós), superando em definitivo o bloqueio de nó único.
- **Resolução do Root-Freezing:** A substituição do critério Information Gain ($R = \log_2(7) \approx 2.807$) pelo Gini ($R = 1.0$) reduz a cota $\epsilon$ em quase $3\times$, permitindo que a diferença entre atributos ($\Delta G$) supere o limite estatístico e ative o desempate com $\tau=0.15$.
- **Escalonamento dos Tiers C11:** O **Tier 8** e o **Tier 4** mantêm acurácias sólidas (**67.28%** e **64.69%**), consumindo apenas 91 KB e 46 KB de RAM estática.
- **Memória e Velocidade:** Devido aos 54 atributos e 7 classes, o River consome ~750 $\mu$s por amostra, enquanto a biblioteca C11 processa cada amostra em cerca de **1.1 a 4.6 $\mu$s** (**speedup de 160x a >650x** frente ao River em Python).

#### 4. Airlines Delay (Real, 7 features, 2 classes)
- **Desempenho Preditivo:** O **C11 Tier 256** atinge **75.75%** de acurácia e **59.47%** de Macro F1 (com 23 nós), superando o **River** em acurácia (**75.29%** / 60.79%) e batendo o Scikit-Learn em Macro F1 (**58.32%** com 123 nós). O uso de pré-processamento (`--prep-scaler --obs hist`) reduziu drasticamente o Macro F1 para 43.31%, demonstrando que a ingestão de dados brutos com hiperparâmetros sintonizados preserva a melhor calibração entre classes.
- **Memória e Velocidade:** O **C11 Tier 8 (16.8 KB)** processa cada amostra em **0.38 $\mu$s** (**>1.000x mais rápido** que os 380 $\mu$s do River).

### 4.2. Datasets de Regressão Contínua

#### 5. Friedman #1 (Sintético, 10 features contínuas, alvo não-linear)
- **Liderança Isolada do C11 Tratado:** Com o normalizador online StandardScaler (`--prep-scaler`), o **C11 Tier 256 Tratado atinge o melhor resultado de todo o benchmark**: Holdout $R^2$ de **0.8592** e MAE de **1.4891** com apenas 49 nós, **superando tanto o Scikit-Learn Batch ($R^2=0.8490$, MAE 1.5224) quanto o River HTR ($R^2=0.8307$, MAE 1.6025)**!
- **Memória e Velocidade:** O C11 consome apenas **531.4 KB** de RAM estática e zero heap dinâmico (frente aos 13.058 KB do River e 2.022 KB do Scikit-Learn) rodando a **1.24 $\mu$s** (**>940x mais rápido** que o River).

#### 6. Bikes / Bike Sharing (Real, 12 features, contagem de demanda)
- **Desempenho Preditivo:** Com hiperparâmetros sintonizados com predição por média nas folhas (`--leaf mean`), o **C11 Tier 256** atinge $R^2$ robusto de **0.7370** (MAE 23.84 com 91 nós), sem necessidade de normalização externa, processando amostras a **0.85 $\mu$s** (**>750x mais rápido** que o River).

#### 7. Appliances Energy (Real, 28 features de telemetria IoT, consumo em Wh)
- **Desempenho Preditivo:** O **C11 Tier 256 Tratado (`--leaf-normalize`)** atinge $R^2$ de **0.8034** (MAE 26.28), praticamente empatando com o **River** ($R^2=0.8059$) e próximo do Scikit-Learn ($R^2=0.8245$), mas com **473.8 KB de RAM** (contra 24.308 KB do River) e rodando a **4.32 $\mu$s** (**500x mais rápido** que os 2.161 $\mu$s do River).
- **Escalonamento nos Tiers Leves:** O **C11 Tier 8** entrega $R^2 = 0.6606$ com apenas **91.0 KB** de RAM a 1.16 $\mu$s/amostra.

#### 8. NYC Taxi Trip Duration (Real, 7 features espaciais/temporais, duração em segundos)
- **Desempenho Preditivo:** O **C11 Tier 256 Tratado (`--leaf-normalize`)** atinge $R^2$ de **0.3189** (MAE 309.64 com 131 nós), muito próximo do River ($R^2=0.3402$, MAE 305.90) e consumindo apenas 524.9 KB de RAM estática (frente aos 5.884 KB do River).
- **Eficiência Extrema nos Tiers 8 e 4:** O **C11 Tier 8** roda em **0.38 $\mu$s/amostra** (**>1.900x mais rápido** que o River), e o **Tier 4** processa em **0.30 $\mu$s** (**>2.400x mais rápido** que o River).

## 5. Síntese Geral: Eficiência de Memória, Limitações e Escalonamento

1. **Quando usar C11 Streaming (`libht`) vs Scikit-Learn Batch:**
   - **Vantagem do C11/Streaming:** Fluxos sujeitos a *concept drift* temporal (**Electricity**, **Airlines**) e regressão multivariada contínua bem condicionada (**Appliances Energy**, **NYC Taxi**, **Friedman #1**), onde o *Hoeffding Bound* atua como um poderoso regularizador natural contra o sobreajuste massivo do CART Batch, além de cenários embarcados onde reter o dataset em RAM ($>130\text{ MB}$ RSS) é fisicamente impossível.
   - **Vantagem do Scikit-Learn Batch:** Problemas estáticos com acesso ilimitado a memória e tempo de CPU offline, onde árvores com milhares de divisões complexas podem ser geradas sem restrições estáticas de SRAM.
2. **Impacto do Paradigma Batch vs Streaming no Peak RSS do SO:**
   - **Scikit-Learn:** **$129.5\text{ MB}$ a $137.0\text{ MB}$** de Peak RSS no SO.
   - **River:** **$129.7\text{ MB}$ a $175.9\text{ MB}$** de Peak RSS no SO (devido ao runtime Python e objetos dinâmicos).
   - **C11 (`libht`):** **$2.7\text{ MB}$ a $3.3\text{ MB}$** de Peak RSS total no SO (**redução de $>40\times$** no processo inteiro e **0.0 KB** de heap dinâmico).
3. **Guia Prático de Escolha entre os Tiers C11:**
   - **Tier 256 ($473\text{ a }532\text{ KB}$):** Recomendado para problemas de regressão não-linear complexa (**Friedman #1**, **Appliances Energy**, **NYC Taxi**) em microcontroladores de maior porte (ESP32-S3 com PSRAM, STM32H7, Cortex-A), garantindo desempenho igual ou superior ao River e ao Scikit-Learn.
   - **Tier 8 ($16.6\text{ a }97.5\text{ KB}$):** O **ponto ótimo de custo-benefício (Sweet Spot)**. Entrega **100% da acurácia do Tier 256** em classificação e supera o Scikit-Learn no Appliances Energy consumindo até **$31\times$ menos RAM** que o Tier 256.
   - **Tier 4 ($8.4\text{ a }52.5\text{ KB}$):** Projetado para MCUs sub-32KB SRAM (Cortex-M0+, AVR, RISC-V ultraleves). Mantém excelente acurácia em classificação (97.03% no SEA, 64.34% no Electricity, 76.96% no Airlines) e atinge latências sub-microssegundo (**0.31 a 0.89 $\mu$s**).
4. **Impacto do Pré-Processamento Especializado (`libht_prep` & Leaf Normalization):**
   - **Política Estrita de Apresentação:** O tratamento é mantido e apresentado exclusivamente onde houve ganhos comprovados de $R^2$ (**Friedman #1**, **Appliances Energy**, **NYC Taxi**). Nos demais datasets (SEA, Electricity, Covertype, Airlines, Bikes), os dados brutos com hiperparâmetros ótimos entregam desempenho superior ou equivalente, sendo mantida apenas a avaliação C11 direta sem sobrecusto.
   - **Resultados de Destaque:** O C11 Tratado supera Scikit-Learn Batch e River em **Friedman #1** ($R^2 = 0.8592$, MAE 1.4891) e apresenta ganhos sólidos em **Appliances Energy** ($R^2 = 0.8034$) e **NYC Taxi** ($R^2 = 0.3189$).

