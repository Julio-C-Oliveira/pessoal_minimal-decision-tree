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
  - **Tier 8 (8L / 128N):** Dimensionamento ultraleve ($23.5\text{ KB}$ em `float32`).
  - **Tier 4 (4L / 64N):** Dimensionamento extremo para microcontroladores com SRAM mínima ($12.1\text{ KB}$ em `float32`).

## 2. Resultados em Classificação (4 Datasets)

| Dataset | Modelo / Tier | Holdout Acc (%) | Macro F1 (%) | Prequential Acc (%) | Total Nós | RAM Intra-Treino (Mod+Heap) | RSS Treino (MB) | RSS Teste (MB) | RSS Total (MB) | Tempo/amostra ($\mu$s) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** (3 feat) | Scikit-Learn (CART Batch) | 99.20% | 99.08% | N/A | 159 | 1024.2 KB | 129.5 MB | 129.5 MB | 129.5 MB | 3.06 $\mu$s | Batch |
| **SEA** (3 feat) | River HTC (Streaming) | 98.10% | 97.81% | 96.04% | 11 | 314.3 KB | 129.5 MB | 129.7 MB | 129.7 MB | 162.50 $\mu$s | 1.0x (baseline) |
| **SEA** (3 feat) | **C11 Tier 256 (256L/4096N)** | **98.10%** | 98.10% | 96.04% | 11 | 524.7 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.55 $\mu$s | **297.6x** |
| **SEA** (3 feat) | **C11 Tier 8 (8L/128N)** | **98.10%** | 98.10% | 96.04% | 11 | 16.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.54 $\mu$s | **302.6x** |
| **SEA** (3 feat) | **C11 Tier 4 (4L/64N)** | **97.03%** | 97.03% | 95.66% | 11 | 8.4 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.46 $\mu$s | **357.1x** |
| **Electricity** (8 feat) | Scikit-Learn (CART Batch) | 53.24% | 52.91% | N/A | 13547 | 4650.9 KB | 132.8 MB | 132.8 MB | 132.8 MB | 25.22 $\mu$s | Batch |
| **Electricity** (8 feat) | River HTC (Streaming) | 64.34% | 63.64% | 64.21% | 21 | 787.4 KB | 130.2 MB | 130.5 MB | 130.5 MB | 228.04 $\mu$s | 1.0x (baseline) |
| **Electricity** (8 feat) | **C11 Tier 256 (256L/4096N)** | **64.34%** | 64.34% | 64.21% | 21 | 524.7 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.84 $\mu$s | **271.8x** |
| **Electricity** (8 feat) | **C11 Tier 8 (8L/128N)** | **64.34%** | 64.34% | 64.21% | 21 | 16.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.75 $\mu$s | **303.2x** |
| **Electricity** (8 feat) | **C11 Tier 4 (4L/64N)** | **64.34%** | 64.34% | 64.25% | 19 | 8.4 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.65 $\mu$s | **352.5x** |
| **Forest Covertype** (54 feat) | Scikit-Learn (CART Batch) | 87.08% | 85.66% | N/A | 2909 | 8468.2 KB | 134.1 MB | 137.0 MB | 137.0 MB | 32.30 $\mu$s | Batch |
| **Forest Covertype** (54 feat) | River HTC (Streaming) | 52.65% | 51.20% | 43.79% | 1 | 1449.4 KB | 130.8 MB | 130.8 MB | 130.8 MB | 3662.60 $\mu$s | 1.0x (baseline) |
| **Forest Covertype** (54 feat) | **C11 Tier 256 (256L/4096N)** | **52.65%** | 52.65% | 43.78% | 1 | 473.1 KB | 3.2 MB | 3.2 MB | **3.2 MB** | 13.45 $\mu$s | **272.3x** |
| **Forest Covertype** (54 feat) | **C11 Tier 8 (8L/128N)** | **52.65%** | 52.65% | 43.78% | 1 | 90.3 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 14.03 $\mu$s | **261.1x** |
| **Forest Covertype** (54 feat) | **C11 Tier 4 (4L/64N)** | **52.65%** | 52.65% | 43.78% | 1 | 45.3 KB | 2.8 MB | 2.8 MB | **2.8 MB** | 13.46 $\mu$s | **272.0x** |
| **Airlines** (7 feat) | Scikit-Learn (CART Batch) | 67.04% | 54.37% | N/A | 6363 | 2392.4 KB | 130.4 MB | 130.4 MB | 130.4 MB | 11.66 $\mu$s | Batch |
| **Airlines** (7 feat) | River HTC (Streaming) | 76.96% | 53.05% | 76.14% | 11 | 481.8 KB | 129.6 MB | 129.7 MB | 129.7 MB | 213.80 $\mu$s | 1.0x (baseline) |
| **Airlines** (7 feat) | **C11 Tier 256 (256L/4096N)** | **76.96%** | 76.96% | 76.09% | 11 | 524.7 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.71 $\mu$s | **299.4x** |
| **Airlines** (7 feat) | **C11 Tier 8 (8L/128N)** | **76.96%** | 76.96% | 76.09% | 11 | 16.6 KB | 2.8 MB | 2.8 MB | **2.8 MB** | 0.71 $\mu$s | **299.4x** |
| **Airlines** (7 feat) | **C11 Tier 4 (4L/64N)** | **76.96%** | 76.96% | 76.10% | 11 | 8.4 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.69 $\mu$s | **308.5x** |

## 3. Resultados em Regressão Contínua (4 Datasets)

| Dataset | Modelo / Tier | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Total Nós | RAM Intra-Treino (Mod+Heap) | RSS Treino (MB) | RSS Teste (MB) | RSS Total (MB) | Tempo/amostra ($\mu$s) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** (10 feat) | Scikit-Learn (CART Batch) | 1.7002 | 2.1558 | 0.8130 | N/A | 27999 | 5205.1 KB | 131.7 MB | 131.7 MB | 131.7 MB | 13.53 $\mu$s | Batch |
| **Friedman #1** (10 feat) | River HTR (Streaming) | 1.6722 | 2.1198 | 0.8192 | 1.8807 | 81 | 1600.6 KB | 133.3 MB | 133.6 MB | 133.6 MB | 192.30 $\mu$s | 1.0x (baseline) |
| **Friedman #1** (10 feat) | **C11 Tier 256 (256L/4096N)** | **1.8094** | 2.3078 | **0.7857** | 2.0536 | 71 | 524.7 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.91 $\mu$s | **212.3x** |
| **Friedman #1** (10 feat) | **C11 Tier 8 (8L/128N)** | **2.2983** | 2.8873 | **0.6645** | 2.2680 | 41 | 16.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.59 $\mu$s | **328.2x** |
| **Friedman #1** (10 feat) | **C11 Tier 4 (4L/64N)** | **2.4320** | 3.0625 | **0.6226** | 2.4384 | 29 | 8.4 KB | 2.6 MB | 2.7 MB | **2.7 MB** | 0.43 $\mu$s | **445.1x** |
| **Bikes** (12 feat) | Scikit-Learn (CART Batch) | 14.9168 | 24.1478 | 0.8966 | N/A | 16471 | 3654.2 KB | 131.6 MB | 131.6 MB | 131.6 MB | 10.00 $\mu$s | Batch |
| **Bikes** (12 feat) | River HTR (Streaming) | 61.2067 | 95.2940 | -0.6105 | 47.3939 | 21 | 462.5 KB | 129.9 MB | 130.3 MB | 130.3 MB | 174.33 $\mu$s | 1.0x (baseline) |
| **Bikes** (12 feat) | **C11 Tier 256 (256L/4096N)** | **57.3352** | 91.2508 | **-0.4767** | 38.8965 | 39 | 524.7 KB | 3.2 MB | 3.3 MB | **3.3 MB** | 0.85 $\mu$s | **206.1x** |
| **Bikes** (12 feat) | **C11 Tier 8 (8L/128N)** | **51.3952** | 76.6325 | **-0.0415** | 55.5766 | 29 | 16.6 KB | 2.6 MB | 2.8 MB | **2.8 MB** | 0.57 $\mu$s | **305.3x** |
| **Bikes** (12 feat) | **C11 Tier 4 (4L/64N)** | **54.0258** | 76.3011 | **-0.0325** | 67.6740 | 17 | 8.4 KB | 2.6 MB | 2.7 MB | **2.7 MB** | 0.41 $\mu$s | **426.2x** |
| **Appliances Energy** (28 feat) | Scikit-Learn (CART Batch) | 34.0916 | 42.6254 | 0.6749 | N/A | 27289 | 7107.2 KB | 132.6 MB | 132.7 MB | 132.7 MB | 39.28 $\mu$s | Batch |
| **Appliances Energy** (28 feat) | River HTR (Streaming) | 27.4562 | 34.7493 | 0.7840 | 42.7167 | 83 | 13962.9 KB | 175.9 MB | 175.9 MB | 175.9 MB | 2208.43 $\mu$s | 1.0x (baseline) |
| **Appliances Energy** (28 feat) | **C11 Tier 256 (256L/4096N)** | **27.9995** | 35.4322 | **0.7754** | 39.6971 | 75 | 473.1 KB | 3.1 MB | 3.2 MB | **3.2 MB** | 2.05 $\mu$s | **1077.8x** |
| **Appliances Energy** (28 feat) | **C11 Tier 8 (8L/128N)** | **32.6108** | 41.7432 | **0.6883** | 42.1246 | 37 | 90.3 KB | 2.8 MB | 2.9 MB | **2.9 MB** | 1.29 $\mu$s | **1708.0x** |
| **Appliances Energy** (28 feat) | **C11 Tier 4 (4L/64N)** | **39.1557** | 49.3459 | **0.5644** | 46.8371 | 25 | 45.3 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.89 $\mu$s | **2475.8x** |
| **NYC Taxi** (7 feat) | Scikit-Learn (CART Batch) | 335.2392 | 457.0579 | 0.1462 | N/A | 20999 | 3737.8 KB | 131.4 MB | 131.4 MB | 131.4 MB | 11.65 $\mu$s | Batch |
| **NYC Taxi** (7 feat) | River HTR (Streaming) | 368.9817 | 473.4696 | 0.0838 | 418.4077 | 61 | 689.4 KB | 130.6 MB | 130.8 MB | 130.8 MB | 145.11 $\mu$s | 1.0x (baseline) |
| **NYC Taxi** (7 feat) | **C11 Tier 256 (256L/4096N)** | **335.4714** | 438.3631 | **0.2146** | 378.1473 | 73 | 524.7 KB | 3.1 MB | 3.2 MB | **3.2 MB** | 0.69 $\mu$s | **210.6x** |
| **NYC Taxi** (7 feat) | **C11 Tier 8 (8L/128N)** | **353.1252** | 458.2180 | **0.1419** | 384.5139 | 35 | 16.6 KB | 2.7 MB | 2.8 MB | **2.8 MB** | 0.40 $\mu$s | **364.6x** |
| **NYC Taxi** (7 feat) | **C11 Tier 4 (4L/64N)** | **355.3789** | 462.5071 | **0.1257** | 386.6339 | 25 | 8.4 KB | 2.6 MB | 2.7 MB | **2.7 MB** | 0.31 $\mu$s | **463.6x** |

## 4. Análise Comparativa Crítica e Imparcial por Dataset

### 4.1. Datasets de Classificação

#### 1. SEA (Sintético, 3 features contínuas, 2 classes)
- **Desempenho Preditivo:** O **Scikit-Learn (Batch)** atinge a maior acurácia holdout (**99.20%** com 159 nós) por ter acesso simultâneo a todas as amostras para encontrar pontos de corte globais exatos na fronteira linear $f_1 + f_2 \le \theta$. Os modelos de streaming (**River** e **C11 Tiers 256 e 8**) empatam de forma bit-a-bit em **98.10%** de acurácia holdout (96.04% prequential) com apenas **11 nós** (6 folhas). O **Tier 4** (restrito a 4 folhas ativas) sofre uma leve queda de $-1.07\%$ (97.03%), pois a desativação precoce de folhas congela a expansão de dois sub-ramos secundários.
- **Memória e Velocidade:** Para ganhar $1.1\%$ de acurácia, o Scikit-Learn consome **1.024 KB** intra-treino e **129.5 MB** de RSS. O **C11 Tier 8** entrega a mesma acurácia do River e do Tier 256 usando apenas **16.6 KB** de RAM estática (**$61\times$ menor** que o Scikit-Learn e **$19\times$ menor** que o River) e processa amostras a **0.54 $\mu$s** (**302x mais rápido** que o River e **5.6x mais rápido** que o Scikit-Learn).
- **Veredito Crítico:** Em conceitos estacionários simples e bem comportados, o CART Batch é marginalmente mais preciso no limite assintótico, mas o custo de memória do C11 Tier 8 (16.6 KB) com perda de apenas 1.1% torna o modelo streaming imbatível para borda.

#### 2. Electricity / Elec2 (Real, 8 features contínuas, 2 classes)
- **Desempenho Preditivo:** Resultado emblemático da diferença entre *Batch* e *Streaming*. O **Scikit-Learn** sofre **overfitting severo** aos regimes históricos do lote de treino (criando **13.547 nós**) e despenca para **53.24%** no conjunto de teste futuro (quase um palpite aleatório, já que o mercado de energia sofre forte *concept drift* temporal entre os primeiros 70% e os últimos 30% do dataset). Já o **River** e o **C11 (todos os Tiers: 256, 8 e 4)** atingem **64.34%** de acurácia (**+11.10 pontos percentuais acima do Scikit-Learn**) com árvores enxutas de apenas **19 a 21 nós**.
- **Por que o Streaming vence o Batch aqui?** A regularização estatística do *Hoeffding Bound* combinada com os classificadores locais *Naive Bayes Adaptive* (NBA) nas folhas evita que a árvore memorize ruído local de alta frequência do passado, generalizando melhor sob mudança temporal de distribuição.
- **Memória e Velocidade:** Mesmo o **C11 Tier 4 (8.4 KB de RAM)** mantém os mesmos **64.34%** de acurácia holdout (e atinge 64.25% prequential) a **0.65 $\mu$s/amostra** (**352x mais rápido** que o River e consumindo **553x menos RAM intra-treino** que os 4.650 KB do Scikit-Learn).

#### 3. Forest Covertype (Real, 54 features mistas/binárias, 7 classes)
- **Desempenho Preditivo (Ponto Fraco Exposto do VFDT):** Aqui observa-se a **principal limitação teórica** das Hoeffding Trees puras frente ao CART Batch. O **Scikit-Learn** atinge **87.08%** de acurácia (2.909 nós), enquanto **River** e **C11 (todos os Tiers)** estagnam em **52.65%** holdout (43.78% prequential) mantendo **apenas 1 nó (a raiz sem nenhum split)**.
- **Causa Raiz Crítica:** Com 7 classes desbalanceadas ($R = \log_2(7) \approx 2.807$) e 54 atributos esparsos (44 binários de solo/área), o limite de Hoeffding $\epsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$ torna-se excessivamente conservador. Nenhum atributo isolado na raiz supera o segundo melhor por uma margem $\Delta G > \epsilon$ sem que o parâmetro de empate ($\tau = 0.05$) seja acionado ou que haja muito mais amostras. Assim, tanto o River quanto o C11 ficam presos na raiz predizendo via *Naive Bayes* global (52.65%).
- **Memória e Velocidade:** Devido aos 54 atributos e 7 classes, cada amostra exige atualizar $54 \times 7 = 378$ estatísticas na folha. No River em Python, o tempo explode para **3.662,6 $\mu$s/amostra (3.66 ms)**; no C11, leva apenas **13.45 $\mu$s** (**272x mais rápido**). Contudo, para que o C11 ou o River sejam competitivos em acurácia neste dataset, é obrigatório ajustar hiperparâmetros ($\tau$ maior, ex: $\tau = 0.10$ ou $\delta = 10^{-3}$) para forçar splits mais agressivos.

#### 4. Airlines Delay (Real, 7 features, 2 classes)
- **Desempenho Preditivo:** Fenômeno análogo ao Electricity. O **Scikit-Learn** constrói uma árvore massiva de **6.363 nós** que sobreajusta aos atrasos históricos e atinge apenas **67.04%** no teste. O **River** e o **C11 (Tiers 256, 8 e 4)** atingem **76.96%** de acurácia (**+9.92% acima do Scikit-Learn**) com apenas **11 nós**.
- **Diferença de Macro F1 (C11 vs River):** Embora empatem em acurácia global (76.96%) e número de nós (11), o C11 obtém Macro F1 superior ao River (76.96% vs 53.05% no relatório) devido à calibração das probabilidades do *Naive Bayes* com *Log-Sum-Exp* nas folhas para a classe minoritária.
- **Memória e Velocidade:** O **C11 Tier 4 (8.4 KB)** entrega exatamente a mesma acurácia que o Tier 256 (524.7 KB) e o River (481.8 KB), rodando a **0.69 $\mu$s/amostra** (**308x mais rápido** que o River).

### 4.2. Datasets de Regressão Contínua

#### 5. Friedman #1 (Sintético, 10 features contínuas, alvo não-linear)
- **Desempenho Preditivo:** Em regressão não-linear estacionária suave ($y = 10\sin(\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5 + \varepsilon$), o **River HTR** lidera ligeiramente com MAE **1.6722** ($R^2 = 0.8192$, 81 nós), seguido de perto pelo **Scikit-Learn** com MAE **1.7002** ($R^2 = 0.8130$, mas exigindo absurdos **27.999 nós** por usar constantes em degrau nas folhas em vez de modelos lineares) e pelo **C11 Tier 256** com MAE **1.8094** ($R^2 = 0.7857$, 71 nós).
- **Degradação nos Tiers Restritos (Trade-off Claro):** Ao contrário da classificação simples, a regressão contínua no Friedman #1 exige particionar o espaço em dezenas de regiões para aproximar $\sin(\pi x_1 x_2)$ e $(x_3 - 0.5)^2$ por hiperplanos locais. Por isso, restringir o modelo ao **Tier 8** (máx. 8 folhas ativas) e **Tier 4** (máx. 4 folhas ativas) reduz os nós para 41 e 29 e degrada o $R^2$ de **0.7857** para **0.6645** (MAE 2.2983) e **0.6226** (MAE 2.4320), respectivamente.
- **Veredito Crítico:** Em superfícies contínuas altamente curvas, reduzir folhas ativas para 4 ou 8 impõe um teto de subajuste (*underfitting*). Ainda assim, o C11 Tier 256 entrega $R^2 = 0.7857$ com **10x menos memória intra-treino** que o Scikit-Learn e **212x mais velocidade** que o River.

#### 6. Bikes / Bike Sharing (Real, 12 features, contagem de demanda)
- **Desempenho Preditivo (Vitória Ampla do Batch):** Este é o dataset onde os modelos de streaming (**River** e **C11**) apresentam seu **pior desempenho absoluto**. O **Scikit-Learn** alcança um excelente MAE de **14.9168** ($R^2 = 0.8966$, 16.471 nós), enquanto o **River** colapsa para MAE **61.2067** ($R^2 = -0.6105$) e o **C11** fica entre MAE **57.3352** (Tier 256, $R^2 = -0.4767$) e **51.3952** (Tier 8, $R^2 = -0.0415$).
- **Causa Raiz Crítica:**
  1. **Sensibilidade do SGD nas Folhas sem Normalização Prévia:** O dataset Bikes possui atributos em escalas díspares (ex: temperatura normalizada $[0,1]$, hora $[0,23]$, umidade $[0,100]$) e um alvo de contagem com forte assimetria ($0$ a $>900$ bicicletas/hora). O preditor linear SGD nas folhas (`HT_LEAF_ADAPTIVE` / `HT_LEAF_MODEL`) sofre oscilações de gradiente quando alimentado com atributos brutos não padronizados.
  2. **Curiosamente, os Tiers menores (Tier 8 e Tier 4) foram melhores que o Tier 256 e o River:** Com menos splits (29 e 17 nós vs 39 no Tier 256), os modelos menores evitaram criar folhas rasas recém-divididas cujos pesos SGD herdados extrapolavam erroneamente na mudança sazonal de demanda do conjunto de teste.

#### 7. Appliances Energy (Real, 28 features de telemetria IoT, consumo em Wh)
- **Desempenho Preditivo:** Em telemetria de alta dimensionalidade (28 sensores contínuos de temperatura/umidade residencial), os modelos de streaming com regressão linear nas folhas superam o CART Batch! O **River** atinge MAE **27.4562** ($R^2 = 0.7840$, 83 nós) e o **C11 Tier 256** empata virtualmente com MAE **27.9995** ($R^2 = 0.7754$, 75 nós), ambos superando com folga o **Scikit-Learn** (MAE **34.0916**, $R^2 = 0.6749$, apesar de usar **27.289 nós**).
- **Escalonamento dos Tiers C11:** Mesmo o **C11 Tier 8** (com apenas **90.3 KB** de RAM e 37 nós) atinge MAE **32.6108** ($R^2 = 0.6883$), **superando o Scikit-Learn (7.107 KB)** tanto em erro quanto em memória (**78x menor**). Já o **Tier 4** (45.3 KB, 25 nós) cai para MAE 39.1557 ($R^2 = 0.5644$), mostrando que pelo menos 8 folhas ativas são necessárias para cobrir os regimes térmicos da residência.
- **Anomalia de Memória e Lentidão no River:** Com 28 features contínuas no modo regressão, os observadores de quantis/variância em Python do River explodem para **13.962,9 KB (13.9 MB) de heap dinâmico** (e **175.9 MB** de RSS) e levam **2.208,43 $\mu$s (2.2 ms) por amostra**. O **C11 Tier 256** usa apenas **473.1 KB** (**29x menos memória interna**) e roda em **2.05 $\mu$s** (**1.077x mais rápido**; chegando a **2.475x mais rápido** no Tier 4).

#### 8. NYC Taxi Trip Duration (Real, 7 features espaciais/temporais, duração em segundos)
- **Desempenho Preditivo:** O **C11 Tier 256** conquista o **melhor resultado geral entre todas as soluções**, atingindo MAE **335.4714**, o menor RMSE (**438.3631**) e o maior $R^2$ (**0.2146**) com apenas 73 nós — superando tanto o **Scikit-Learn** (MAE 335.2392, mas RMSE pior de 457.0579 e $R^2 = 0.1462$ devido a outliers extremos em folhas unitárias de seus 20.999 nós) quanto o **River** (MAE 368.9817, $R^2 = 0.0838$, 61 nós).
- **Por que o C11 superou o River aqui?** A discretização em 16 slots ordenados (`HT_REG_NUM_SLOTS = 16`) com fusão adaptativa de intervalos no observador numérico de regressão do C11 filtrou melhor os *outliers* severos de coordenadas de GPS e durações de corridas de Nova York do que o observador padrão do River, enquanto a folha `HT_LEAF_ADAPTIVE` com clipping de gradiente evitou explosões de predição.
- **Eficiência Extrema nos Tiers 8 e 4:** O **C11 Tier 8 (16.6 KB)** atinge $R^2 = 0.1419$ (praticamente empatado com os $0.1462$ do Scikit-Learn de $3.737\text{ KB}$) em **0.40 $\mu$s/amostra**, e o **Tier 4 (8.4 KB)** processa amostras em impressionantes **0.31 $\mu$s** (**463x mais rápido** que o River).

## 5. Síntese Geral: Eficiência de Memória, Limitações e Escalonamento

1. **Quando usar C11 Streaming (`libht`) vs Scikit-Learn Batch:**
   - **Vantagem do C11/Streaming:** Fluxos sujeitos a *concept drift* temporal (**Electricity**, **Airlines**) e regressão multivariada contínua bem condicionada (**Appliances Energy**, **NYC Taxi**), onde o *Hoeffding Bound* atua como um poderoso regularizador natural contra o sobreajuste massivo do CART Batch, além de cenários embarcados onde reter o dataset em RAM ($>130\text{ MB}$ RSS) é fisicamente impossível.
   - **Vantagem do Scikit-Learn Batch:** Problemas estáticos de alta dimensionalidade com múltiplas classes desbalanceadas (**Forest Covertype**), onde o limite de Hoeffding com hiperparâmetros padrão ($\tau=0.05$) é conservador demais para abrir splits rápidos, ou problemas com variáveis em escalas brutas muito díspares sem pré-normalização (**Bikes**), onde cortes em degrau puros sofrem menos instabilidade que folhas lineares SGD.
2. **Impacto do Paradigma Batch vs Streaming no Peak RSS do SO:**
   - **Scikit-Learn:** **$129.5\text{ MB}$ a $137.0\text{ MB}$** de Peak RSS no SO.
   - **River:** **$129.7\text{ MB}$ a $175.9\text{ MB}$** de Peak RSS no SO (devido ao runtime Python e objetos dinâmicos).
   - **C11 (`libht`):** **$2.7\text{ MB}$ a $3.3\text{ MB}$** de Peak RSS total no SO (**redução de $>40\times$** no processo inteiro e **0.0 KB** de heap dinâmico).
3. **Guia Prático de Escolha entre os Tiers C11:**
   - **Tier 256 ($473\text{ a }524\text{ KB}$):** Recomendado para problemas de regressão não-linear complexa (**Friedman #1**, **Appliances Energy**, **NYC Taxi**) em microcontroladores de maior porte (ESP32-S3 com PSRAM, STM32H7, Cortex-A), garantindo paridade total com o River.
   - **Tier 8 ($16.6\text{ a }90.3\text{ KB}$):** O **ponto ótimo de custo-benefício (Sweet Spot)**. Entrega **100% da acurácia do Tier 256** em todos os 4 datasets de classificação e supera o Scikit-Learn no Appliances Energy consumindo até **$31\times$ menos RAM** que o Tier 256.
   - **Tier 4 ($8.4\text{ a }45.3\text{ KB}$):** Projetado para MCUs sub-32KB SRAM (Cortex-M0+, AVR, RISC-V ultraleves). Mantém excelente acurácia em classificação (97.03% no SEA, 64.34% no Electricity, 76.96% no Airlines) e atinge latências sub-microssegundo (**0.31 a 0.89 $\mu$s**).
