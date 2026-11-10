# kansho — ultralight embedded streaming decision trees in c11 (<= 1 mb ram, zero malloc)

> **kansho** (簡樹, do japonês *árvore simples*): implementação ultraleve de **Hoeffding Trees (`VFDT`, `HAT`, `VFDT-R` / `FIMT-DD` e `HAT-R`)** em **C11 puro**, projetada para **treino incremental (*online/streaming*) e inferência em tempo real** em microcontroladores e sistemas embarcados com forte restrição de memória (**$\le 1\text{ MB}$ de RAM**, escalável para **$\approx 11\text{ KB}$**, com **zero alocação dinâmica `malloc`/`free`**).

---

## 1. Variantes e Capacidades do Modelo

A biblioteca unifica **4 variantes de árvores de decisão em streaming** dentro da mesma estrutura `ht_tree_t` (sem custo adicional de RAM graças ao uso de `union` anônimas do C11):

| Variante do Modelo | Tarefa de ML | Inicialização de Parâmetros | Adaptação a *Concept Drift* | Critérios de Split Suportados | Predição nas Folhas |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **VFDT** (*Hoeffding Tree Classifier*) | Classificação Multiclasse | `ht_default_params()` | Não (Estacionário) | *Information Gain* (`HT_CRIT_INFO_GAIN`) ou *Gini* (`HT_CRIT_GINI`) | **MC** (*Majority Class*), **NB** (*Naive Bayes*) ou **NBA** (*Adaptive*) |
| **HAT** (*Hoeffding Adaptive Tree*) | Classificação Multiclasse | `ht_default_hat_params()` | **Sim** (ADWIN2 + Subárvores Alternativas + $\text{Poisson}(1)$) | *Information Gain* ou *Gini Impurity* | **MC**, **NB** ou **NBA** + Votação ponderada $1/\hat{e}_l^2$ |
| **VFDT-R** (*FIMT-DD Regressor*) | Regressão Contínua | `ht_default_reg_params()` | Não (Estacionário) | *Variance Reduction* (`HT_CRIT_VARIANCE_REDUCTION`, `ddof=1`) | **MEAN** (*Média*), **MODEL** (*Linear SGD*) ou **ADAPTIVE** (*fMSE*) |
| **HAT-R** (*Hoeffding Adaptive Regressor*) | Regressão Contínua | `ht_default_hat_reg_params()` | **Sim** (ADWIN2 + Teste Z Bilateral de Welch) | *Variance Reduction* (`ddof=1`) | **MEAN**, **MODEL** (SGD) ou **ADAPTIVE** + Média entre subárvores |

### Destaques Técnicos:
- **Aprendizado Incremental *Test-Then-Train*:** Aprende amostra por amostra (`ht_tree_learn_one` / `ht_tree_learn_one_reg`) em tempo $O(\text{profundidade} \times F)$, sem reter o dataset na memória.
- **Micro-Pipelines de Pré-Processamento Zero-Malloc (`libht_prep`, v0.2.0):** Módulos determinísticos para normalização online Z-Score/Min-Max (`ht_scaler_t`), filtro mediano de Hampel com janela deslizante MAD ($3\sigma$) para rejeição de ruído de sensores industriais (`ht_filter_hampel_t`), codificador Space-Saving (`ht_encoder_freq_t`) e quantizador Ben-Haim & Tom-Tov (`ht_quantizer_bh_t`), integráveis transparentemente via `ht_tree_set_prep_pipeline`.
- **Observadores Numéricos Paramétricos e Não-Paramétricos:** Welford Gaussiano (`HT_OBS_GAUSSIAN`) e Histograma Dinâmico de Ben-Haim & Tom-Tov (`HT_OBS_HISTOGRAM`) para distribuições multimodais.
- **Normalização SGD Híbrida nas Folhas:** Padronização online com fator de credibilidade $\alpha = \frac{N_{\text{slots}}}{N_{\text{slots}} + 10.0}$ para estabilização dos regressores lineares em variáveis multiescala com 0.0 KB de heap.
- **Atributos Mistos e Valores Ausentes (`NaN`):** Suporte nativo a colunas numéricas (`HT_FEAT_NUMERIC`) e categóricas (`HT_FEAT_CATEGORICAL`), roteando leituras ausentes (`NAN` ou categoria `< 0`) pelo filho mais populoso com custo zero de RAM.
- **Reciclagem Determinística (*Slab/Pool Allocators*):** Indexação 100% baseada em `uint16_t` (sem ponteiros) e desativação automática de folhas menos promissoras (*Promise Deactivation*) quando o pool atinge o limite.
- **Serialização Binária Direta (`ht_tree_save` / `ht_tree_load`):** Exporta checkpoints completos de treino ou imagens enxutas somente-inferência ($28\text{ B}$ por nó em `float32`) prontas para Flash/EEPROM.
- **Precisão Configurável (`ht_float_t`):** `float` (32 bits) por padrão para FPU de microcontroladores ou `double` (64 bits) via `-DHT_USE_DOUBLE`.

---

## 2. Quick Start

### 2.1 Compilação e Testes

O projeto utiliza exclusivamente a biblioteca padrão C11 e `<math.h>` (`-lm`):

```bash
# Compila a biblioteca estática (code/build/libht.a) e roda todos os testes unitários e E2E
make all

# Compila e executa os testes em precisão dupla (64-bit double)
make clean && make test CFLAGS_EXTRA="-DHT_USE_DOUBLE"

# Executa os benchmarks comparativos contra Python River e Scikit-Learn
make bench
make bench-drift
make bench-regression
make bench-triple
make bench-main
make bench-prep

# Executa a suíte de emulação ESP32 (Hardware-in-the-Loop via QEMU)
make bench-esp32
```

### 2.2 Exemplo Básico em C11 (`VFDT`)

```c
#include <stdio.h>
#include "ht_tree.h"

static ht_tree_t tree; /* Alocação estática em .bss (zero malloc/heap) */

int main(void) {
    ht_config_params_t params;
    ht_default_params(&params, /*n_features=*/4, /*n_classes=*/3);

    if (ht_tree_init(&tree, &params) != HT_OK) {
        return -1;
    }

    ht_value_t x[HT_MAX_FEATURES] = {0};
    x[0].num = 5.1f; x[1].num = 3.5f; x[2].num = 1.4f; x[3].num = 0.2f;

    /* 1. Inferência (Test) */
    uint8_t y_pred = ht_tree_predict(&tree, x);
    printf("Classe predita: %u\n", y_pred);

    /* 2. Treinamento incremental (Train) */
    ht_tree_learn_one(&tree, x, /*y_true=*/0, /*weight=*/1.0f);
    return 0;
}
```

> [!TIP]
> Para exemplos completos de **Concept Drift (`HAT`)**, **Regressão Contínua (`VFDT-R` / `HAT-R`)**, **Atributos Mistos / `NaN`** e **Serialização em Flash (`save`/`load`)**, acesse **[Exemplos e Receitas da API](wiki/Exemplos-e-Receitas-API.md)**.

---

## 3. Consumo de Memória e Desempenho Comparativo

A biblioteca foi avaliada empiricamente contra duas referências consolidadas da literatura: o **[`online-ml/river`](https://github.com/online-ml/river)** (padrão de referência para aprendizado em streaming contínuo em Python/Cython) e o **[`scikit-learn`](https://scikit-learn.org)** (baseline em lote / *batch* CART).

Esta seção reúne o consumo de memória (RAM e Flash) e os resultados de desempenho, na seguinte ordem: **3.1** footprint estático por perfil de hardware, **3.2** resumo do *Main Benchmark* (8 datasets), **3.3** e **3.4** benchmarks sintéticos de classificação e regressão, **3.5** avaliação no MNIST, e **3.6** emulação Hardware-in-the-Loop em ESP32 (Xtensa Dual-Core LX6 @ 240 MHz).

### 3.1 Resumo de Consumo de Memória (RAM & Flash)

O footprint de RAM de `ht_tree_t` é dimensionado estaticamente em tempo de compilação via flags `-D` (veja todos os detalhes em **[Referência de Parâmetros e Macros](wiki/Referencia-de-Parametros-e-Macros.md)** e **[Orçamento de Memória e Cross-Compilation](wiki/Orcamento-de-Memoria-e-Cross-Compilation.md)**):

| Perfil de Hardware Alvo | Flags de Compilação (`CFLAGS_EXTRA`) | Tamanho do Nó | Folha Ativa | RAM Total (`ht_tree_t`) | Flash (`.text` em `-Os`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Somente Inferência** | `-DHT_MAX_NODES=256 -DHT_MAX_ACTIVE_LEAVES=1 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2` | $24\text{ B}$ | $688\text{ B}$ | **$\approx 11,3\text{ KB}$** | **$\approx 6\text{–}8\text{ KB}$** |
| **MCU Ultra-Fraco (32–64 KB)** | `-DHT_MAX_NODES=128 -DHT_MAX_ACTIVE_LEAVES=16 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2 -DHT_MAX_CATEGORIES=4 -DHT_REG_NUM_SLOTS=4` | $24\text{ B}$ | $560\text{ B}$ | **$\approx 14,2\text{ KB}$** | **$\approx 16\text{–}28\text{ KB}$** |
| **MCU Compacto (64–128 KB)** | `-DHT_MAX_NODES=256 -DHT_MAX_ACTIVE_LEAVES=32 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2` | $24\text{ B}$ | $688\text{ B}$ | **$\approx 32,2\text{ KB}$** | **$\approx 16\text{–}28\text{ KB}$** |
| **MCU Intermediário (256–512 KB)** | `-DHT_MAX_NODES=1024 -DHT_MAX_ACTIVE_LEAVES=64 -DHT_MAX_FEATURES=12 -DHT_MAX_CLASSES=4` | $28\text{ B}$ | $1.024\text{ B}$ | **$\approx 110,3\text{ KB}$** | **$\approx 16\text{–}28\text{ KB}$** |
| **Padrão do Repositório ($\le 1\text{ MB}$)** | *(Padrão: `NODES=4096`, `LEAVES=256`, `FEATURES=16`, `CLASSES=4`)* | $28\text{ B}$ | $1.360\text{ B}$ | **$\approx 524,7\text{ KB}$** | **$\approx 22\text{–}28\text{ KB}$** |

---

### 3.2 Main Benchmark: Resumo dos Resultados em 8 Datasets

Suíte principal de desempenho ([`main_benchmark/`](main_benchmark/), relatório completo com análise crítica por dataset em [`main_benchmark_report.md`](main_benchmark/main_benchmark_report.md)), com **4 datasets de classificação** (SEA, Electricity, Forest Covertype, Airlines) e **4 de regressão** (Friedman #1, Bikes, Appliances Energy, NYC Taxi). Reproduza com `make bench-main`.

**Protocolo:** as três soluções treinam sobre o mesmo volume $N$ de amostras. O **Scikit-Learn** recebe todo o lote de treino de uma só vez (`fit()` batch) e o **River** e o **C11** consomem o fluxo **1 amostra por vez**. A memória é medida como o **Peak RSS real do processo** no SO (treino + teste), além da RAM intra-treino (modelo + heap). O C11 é avaliado em três tiers de capacidade estática:

| Tier | Capacidade | RAM do modelo (`float32`) |
| :--- | :---: | :---: |
| **Tier 256** | 256 folhas / 4.096 nós | $\approx 473\text{–}525\text{ KB}$ |
| **Tier 8** | 8 folhas / 128 nós | $\approx 16{,}6\text{–}90{,}3\text{ KB}$ |
| **Tier 4** | 4 folhas / 64 nós | $\approx 8{,}4\text{–}45{,}3\text{ KB}$ |

*(Os valores de 473 KB, 90,3 KB e 45,3 KB correspondem ao Forest Covertype e ao Appliances Energy, compilados com 64 atributos / 8 classes; os demais datasets usam o perfil padrão de 16 atributos / 4 classes.)*

#### 3.2.1 Resultados em Classificação (4 Datasets)

| Dataset | Modelo / Algoritmo | Acurácia Holdout (%) | Precisão Macro (%) | Recall Macro (%) | F1-Score Macro (%) | Acurácia Prequential (%) | Topologia (Total Nós / Folhas) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** (3 feat, 2 classes) | Scikit-Learn (CART Batch) | **99,20%** | **99,14%** | **99,02%** | **99,08%** | N/A | 159 nós |
| | River HTC (Streaming) | 98,10% | 98,05% | 97,58% | 97,81% | 96,04% | 11 nós (6 folhas) |
| | **C11 Tier 256 (256L/4096N)** | **98,10%** | **98,05%** | **97,58%** | **97,81%** | **96,04%** | 11 nós (6 folhas) |
| | **C11 Tier 8 (8L/128N)** | **98,10%** | **98,05%** | **97,58%** | **97,81%** | **96,04%** | 11 nós (6 folhas) |
| | **C11 Tier 4 (4L/64N)** | 97,03% | 97,10% | 96,22% | 96,65% | 95,66% | 11 nós (4 folhas) |
| **Electricity** (8 feat, 2 classes) | Scikit-Learn (CART Batch) | 53,24% | 53,02% | 52,80% | 52,91% | N/A | 13.547 nós |
| | River HTC (Streaming) | **64,34%** | 64,12% | 63,18% | 63,64% | 64,21% | 21 nós (11 folhas) |
| | **C11 Tier 256 (256L/4096N)** | **64,34%** | **64,40%** | **64,28%** | **64,34%** | 64,21% | 21 nós (11 folhas) |
| | **C11 Tier 8 (8L/128N)** | **64,34%** | **64,40%** | **64,28%** | **64,34%** | 64,21% | 21 nós (8 folhas) |
| | **C11 Tier 4 (4L/64N)** | **64,34%** | **64,40%** | **64,28%** | **64,34%** | **64,25%** | 19 nós (4 folhas) |
| **Forest Covertype** (54 feat, 7 classes) | Scikit-Learn (CART Batch) | **87,08%** | **86,42%** | **84,93%** | **85,66%** | N/A | 2.909 nós |
| | River HTC (Streaming) | 52,65% | 52,10% | 50,35% | 51,20% | 43,79% | 1 nós (1 folha) |
| | **C11 Tier 256 (256L/4096N)** | 52,65% | 52,80% | 52,50% | 52,65% | 43,78% | 1 nós (1 folha) |
| | **C11 Tier 8 (8L/128N)** | 52,65% | 52,80% | 52,50% | 52,65% | 43,78% | 1 nós (1 folha) |
| | **C11 Tier 4 (4L/64N)** | 52,65% | 52,80% | 52,50% | 52,65% | 43,78% | 1 nós (1 folha) |
| **Airlines Delay** (7 feat, 2 classes) | Scikit-Learn (CART Batch) | 67,04% | 65,20% | 52,90% | 54,37% | N/A | 6.363 nós |
| | River HTC (Streaming) | **76,96%** | 75,10% | 51,60% | 53,05% | 76,14% | 11 nós (6 folhas) |
| | **C11 Tier 256 (256L/4096N)** | **76,96%** | **76,80%** | **77,10%** | **76,96%** | 76,09% | 11 nós (6 folhas) |
| | **C11 Tier 8 (8L/128N)** | **76,96%** | **76,80%** | **77,10%** | **76,96%** | 76,09% | 11 nós (6 folhas) |
| | **C11 Tier 4 (4L/64N)** | **76,96%** | **76,80%** | **77,10%** | **76,96%** | **76,10%** | 11 nós (4 folhas) |

---

#### 3.2.2 Resultados em Regressão Contínua (4 Datasets)

| Dataset | Modelo / Algoritmo | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Topologia (Total Nós / Folhas) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** (10 feat, contínuo) | Scikit-Learn (CART Batch) | 1,7002 | 2,1558 | 0,8130 | N/A | 27.999 nós |
| | River HTR (Streaming) | **1,6722** | **2,1198** | **0,8192** | **1,8807** | 81 nós (41 folhas) |
| | **C11 Tier 256 (256L/4096N)** | 1,8094 | 2,3078 | 0,7857 | 2,0536 | 71 nós (36 folhas) |
| | **C11 Tier 8 (8L/128N)** | 2,2983 | 2,8873 | 0,6645 | 2,2680 | 41 nós (8 folhas) |
| | **C11 Tier 4 (4L/64N)** | 2,4320 | 3,0625 | 0,6226 | 2,4384 | 29 nós (4 folhas) |
| **Bikes** (12 feat, contínuo) | Scikit-Learn (CART Batch) | **14,9168** | **24,1478** | **0,8966** | N/A | 16.471 nós |
| | River HTR (Streaming) | 61,2067 | 95,2940 | -0,6105 | 47,3939 | 21 nós (11 folhas) |
| | **C11 Tier 256 (256L/4096N)** | 57,3352 | 91,2508 | -0,4767 | **38,8965** | 39 nós (20 folhas) |
| | **C11 Tier 8 (8L/128N)** | **51,3952** | **76,6325** | **-0,0415** | 55,5766 | 29 nós (8 folhas) |
| | **C11 Tier 4 (4L/64N)** | 54,0258 | 76,3011 | -0,0325 | 67,6740 | 17 nós (4 folhas) |
| **Appliances Energy** (28 feat, contínuo) | Scikit-Learn (CART Batch) | 34,0916 | 42,6254 | 0,6749 | N/A | 27.289 nós |
| | River HTR (Streaming) | **27,4562** | **34,7493** | **0,7840** | 42,7167 | 83 nós (42 folhas) |
| | **C11 Tier 256 (256L/4096N)** | 27,9995 | 35,4322 | 0,7754 | **39,6971** | 75 nós (38 folhas) |
| | **C11 Tier 8 (8L/128N)** | 32,6108 | 41,7432 | 0,6883 | 42,1246 | 37 nós (8 folhas) |
| | **C11 Tier 4 (4L/64N)** | 39,1557 | 49,3459 | 0,5644 | 46,8371 | 25 nós (4 folhas) |
| **NYC Taxi** (7 feat, contínuo) | Scikit-Learn (CART Batch) | 335,2392 | 457,0579 | 0,1462 | N/A | 20.999 nós |
| | River HTR (Streaming) | 368,9817 | 473,4696 | 0,0838 | 418,4077 | 61 nós (31 folhas) |
| | **C11 Tier 256 (256L/4096N)** | **335,4714** | **438,3631** | **0,2146** | **378,1473** | 73 nós (37 folhas) |
| | **C11 Tier 8 (8L/128N)** | 353,1252 | 458,2180 | 0,1419 | 384,5139 | 35 nós (8 folhas) |
| | **C11 Tier 4 (4L/64N)** | 355,3789 | 462,5071 | 0,1257 | 386,6339 | 25 nós (4 folhas) |

---

#### 3.2.3 Métricas de Performance, Latência e Recursos (8 Datasets)

> [!NOTE]
> **Entendendo as Métricas de Latência por Amostra:**
> - **Escopo:** Cada amostra em streaming é avaliada no ciclo completo **Test-Then-Train** (inferência da predição com `ht_tree_predict` seguida de treino online e atualização dos observadores de Welford com `ht_tree_learn_one`).
> - **Latência Média:** Média aritmética em microssegundos ($\mu\text{s}$) necessária para processar 1 amostra completa.
> - **Latência $p_{95}$ e $p_{99}$ (Latência de Cauda):** Percentis 95% e 99% da distribuição de tempo por amostra. No algoritmo VFDT, a maioria das amostras é absorvida em tempo constante $O(1)$ pelas folhas; contudo, a cada $N = \text{grace\_period} = 200$ amostras, a folha avalia o *Hoeffding Bound* e testa critérios de impureza/variância para todos os atributos, gerando um custo computacional momentâneo que é rigorosamente delimitado por essas métricas de cauda.

| Dataset | Modelo / Algoritmo | Latência Média | Latência $p_{95}$ | Latência $p_{99}$ | Throughput (Vazão) | Model Footprint | Consumo RAM (Modelo / SO Peak) | FLOPs / Amostra |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Scikit-Learn (Batch) | 3,06 µs | 4,20 µs | 6,50 µs | 326.797 am/s | 22,4 KB | 1.024 KB / 129,5 MB | $\approx 318$ FLOPs |
| | River HTC | 162,50 µs | 210,00 µs | 340,00 µs | 6.154 am/s | 10,0 KB | 314 KB / 129,7 MB | $\approx 92$ FLOPs |
| | **C11 Tier 256** | **0,55 µs** | 0,80 µs | 1,40 µs | **1.818.181 am/s** | 340 B (.bin) / 524,7 KB | **524,7 KB (.bss) / 3,3 MB** | **67 FLOPs** |
| | **C11 Tier 8** | **0,54 µs** | 0,78 µs | 1,35 µs | **1.851.851 am/s** | 340 B (.bin) / 16,6 KB | **16,6 KB (.bss) / 2,8 MB** | **67 FLOPs** |
| | **C11 Tier 4** | **0,46 µs** | 0,65 µs | 1,20 µs | **2.173.913 am/s** | 340 B (.bin) / 8,4 KB | **8,4 KB (.bss) / 2,8 MB** | **67 FLOPs** |
| **Electricity** | Scikit-Learn (Batch) | 25,22 µs | 32,50 µs | 48,10 µs | 39.651 am/s | 1.890 KB | 4.651 KB / 132,8 MB | $\approx 27.000$ FLOPs |
| | River HTC | 228,04 µs | 290,00 µs | 480,00 µs | 4.385 am/s | 18,2 KB | 787 KB / 130,5 MB | $\approx 210$ FLOPs |
| | **C11 Tier 256** | **0,84 µs** | 1,15 µs | 2,10 µs | **1.190.476 am/s** | 620 B (.bin) / 524,7 KB | **524,7 KB (.bss) / 3,3 MB** | **148 FLOPs** |
| | **C11 Tier 8** | **0,75 µs** | 1,02 µs | 1,95 µs | **1.333.333 am/s** | 620 B (.bin) / 16,6 KB | **16,6 KB (.bss) / 2,8 MB** | **148 FLOPs** |
| | **C11 Tier 4** | **0,65 µs** | 0,90 µs | 1,70 µs | **1.538.461 am/s** | 564 B (.bin) / 8,4 KB | **8,4 KB (.bss) / 2,8 MB** | **148 FLOPs** |
| **Covertype** | Scikit-Learn (Batch) | 32,30 µs | 45,00 µs | 68,00 µs | 30.959 am/s | 420 KB | 8.468 KB / 137,0 MB | $\approx 5.800$ FLOPs |
| | River HTC | 3.662,60 µs | 4.800,00 µs | 7.200,00 µs | 273 am/s | 1,6 KB | 1.449 KB / 130,8 MB | $\approx 3.200$ FLOPs |
| | **C11 Tier 256** | **13,45 µs** | 18,20 µs | 32,50 µs | **74.349 am/s** | 60 B (.bin) / 473,1 KB | **473,1 KB (.bss) / 3,2 MB** | **2.246 FLOPs** |
| | **C11 Tier 8** | **14,03 µs** | 19,10 µs | 33,80 µs | **71.275 am/s** | 60 B (.bin) / 90,3 KB | **90,3 KB (.bss) / 2,9 MB** | **2.246 FLOPs** |
| | **C11 Tier 4** | **13,46 µs** | 18,30 µs | 32,60 µs | **74.294 am/s** | 60 B (.bin) / 45,3 KB | **45,3 KB (.bss) / 2,8 MB** | **2.246 FLOPs** |
| **Airlines** | Scikit-Learn (Batch) | 11,66 µs | 16,00 µs | 24,00 µs | 85.763 am/s | 890 KB | 2.392 KB / 130,4 MB | $\approx 12.700$ FLOPs |
| | River HTC | 213,80 µs | 280,00 µs | 450,00 µs | 4.677 am/s | 9,8 KB | 482 KB / 129,7 MB | $\approx 185$ FLOPs |
| | **C11 Tier 256** | **0,71 µs** | 0,98 µs | 1,80 µs | **1.408.450 am/s** | 340 B (.bin) / 524,7 KB | **524,7 KB (.bss) / 3,3 MB** | **131 FLOPs** |
| | **C11 Tier 8** | **0,71 µs** | 0,98 µs | 1,80 µs | **1.408.450 am/s** | 340 B (.bin) / 16,6 KB | **16,6 KB (.bss) / 2,8 MB** | **131 FLOPs** |
| | **C11 Tier 4** | **0,69 µs** | 0,92 µs | 1,75 µs | **1.449.275 am/s** | 340 B (.bin) / 8,4 KB | **8,4 KB (.bss) / 2,8 MB** | **131 FLOPs** |
| **Friedman #1** | Scikit-Learn (Batch) | 13,53 µs | 18,20 µs | 27,00 µs | 73.909 am/s | 3.800 KB | 5.205 KB / 131,7 MB | $\approx 55.000$ FLOPs |
| | River HTR | 192,30 µs | 250,00 µs | 410,00 µs | 5.200 am/s | 218,6 KB | 1.601 KB / 133,6 MB | $\approx 180$ FLOPs |
| | **C11 Tier 256** | **0,91 µs** | 1,25 µs | 2,30 µs | **1.098.901 am/s** | 2.020 B (.bin) / 524,7 KB | **524,7 KB (.bss) / 3,3 MB** | **134 FLOPs** |
| | **C11 Tier 8** | **0,59 µs** | 0,82 µs | 1,50 µs | **1.694.915 am/s** | 1.180 B (.bin) / 16,6 KB | **16,6 KB (.bss) / 2,8 MB** | **134 FLOPs** |
| | **C11 Tier 4** | **0,43 µs** | 0,60 µs | 1,15 µs | **2.325.581 am/s** | 844 B (.bin) / 8,4 KB | **8,4 KB (.bss) / 2,7 MB** | **134 FLOPs** |
| **Bikes** | Scikit-Learn (Batch) | 10,00 µs | 14,00 µs | 21,00 µs | 100.000 am/s | 2.200 KB | 3.654 KB / 131,6 MB | $\approx 32.000$ FLOPs |
| | River HTR | 174,33 µs | 230,00 µs | 390,00 µs | 5.736 am/s | 48,0 KB | 463 KB / 130,3 MB | $\approx 210$ FLOPs |
| | **C11 Tier 256** | **0,85 µs** | 1,18 µs | 2,15 µs | **1.176.470 am/s** | 1.124 B (.bin) / 524,7 KB | **524,7 KB (.bss) / 3,3 MB** | **155 FLOPs** |
| | **C11 Tier 8** | **0,57 µs** | 0,80 µs | 1,45 µs | **1.754.385 am/s** | 844 B (.bin) / 16,6 KB | **16,6 KB (.bss) / 2,8 MB** | **155 FLOPs** |
| | **C11 Tier 4** | **0,41 µs** | 0,58 µs | 1,10 µs | **2.439.024 am/s** | 508 B (.bin) / 8,4 KB | **8,4 KB (.bss) / 2,7 MB** | **155 FLOPs** |
| **Appliances** | Scikit-Learn (Batch) | 39,28 µs | 52,00 µs | 78,00 µs | 25.458 am/s | 3.700 KB | 7.107 KB / 132,7 MB | $\approx 54.000$ FLOPs |
| | River HTR | 2.208,43 µs | 3.100,00 µs | 5.200,00 µs | 452 am/s | 320,0 KB | 13.963 KB / 175,9 MB | $\approx 450$ FLOPs |
| | **C11 Tier 256** | **2,05 µs** | 2,80 µs | 5,10 µs | **487.804 am/s** | 2.132 B (.bin) / 473,1 KB | **473,1 KB (.bss) / 3,2 MB** | **332 FLOPs** |
| | **C11 Tier 8** | **1,29 µs** | 1,75 µs | 3,20 µs | **775.193 am/s** | 1.068 B (.bin) / 90,3 KB | **90,3 KB (.bss) / 2,9 MB** | **332 FLOPs** |
| | **C11 Tier 4** | **0,89 µs** | 1,20 µs | 2,25 µs | **1.123.595 am/s** | 732 B (.bin) / 45,3 KB | **45,3 KB (.bss) / 2,8 MB** | **332 FLOPs** |
| **NYC Taxi** | Scikit-Learn (Batch) | 11,65 µs | 15,80 µs | 23,50 µs | 85.836 am/s | 2.800 KB | 3.738 KB / 131,4 MB | $\approx 41.000$ FLOPs |
| | River HTR | 145,11 µs | 195,00 µs | 320,00 µs | 6.891 am/s | 110,0 KB | 689 KB / 130,8 MB | $\approx 140$ FLOPs |
| | **C11 Tier 256** | **0,69 µs** | 0,95 µs | 1,75 µs | **1.449.275 am/s** | 2.076 B (.bin) / 524,7 KB | **524,7 KB (.bss) / 3,2 MB** | **101 FLOPs** |
| | **C11 Tier 8** | **0,40 µs** | 0,55 µs | 1,05 µs | **2.500.000 am/s** | 1.012 B (.bin) / 16,6 KB | **16,6 KB (.bss) / 2,8 MB** | **101 FLOPs** |
| | **C11 Tier 4** | **0,31 µs** | 0,44 µs | 0,85 µs | **3.225.806 am/s** | 732 B (.bin) / 8,4 KB | **8,4 KB (.bss) / 2,7 MB** | **101 FLOPs** |

> [!TIP]
> **Principais conclusões do Main Benchmark:**
> - **Streaming vence sob *concept drift* temporal:** em **Electricity** e **Airlines**, River e C11 superam o CART batch em $+11{,}1$ e $+9{,}9$ pontos percentuais (o CART sobreajusta ao passado, com 13.547 e 6.363 nós contra 11 a 21 nós do C11), e até o **Tier 4 (8,4 KB)** mantém a mesma acurácia do Tier 256.
> - **Tier 8 como ponto de equilíbrio:** reproduz a acurácia do Tier 256 em todos os 4 datasets de classificação com $\approx 31\times$ menos RAM de modelo (16,6 KB vs 524,7 KB) e, no **Appliances Energy**, supera o Scikit-Learn em erro com $\approx 78\times$ menos RAM intra-treino (90,3 KB vs 7.107 KB).
> - **Footprint real do processo:** o C11 ocupa **2,7 a 3,3 MB de Peak RSS** contra **$\approx 130$ a $176\text{ MB}$** de Scikit-Learn e River (redução de $\approx 39\times$ a $65\times$), com **0,0 KB de heap** em runtime e speedup de **$206\times$ a $2.476\times$** sobre o River. O tempo por amostra sobe para $\approx 13\text{–}14\,\mu\text{s}$ apenas no Forest Covertype (54 atributos × 7 classes).
> - **Regressão em RAM reduzida degrada de forma gradual:** no Friedman #1, o $R^2$ cai de 0,786 (Tier 256) para 0,665 (Tier 8) e 0,623 (Tier 4), um *trade-off* explícito entre memória e capacidade de aproximar superfícies não lineares.

> [!WARNING]
> **Onde o modelo de streaming perde (resultados desfavoráveis, reportados sem filtro):**
> - **Forest Covertype (7 classes, 54 atributos):** River e C11 estagnam em **52,65%** com **1 único nó** (nenhum split dentro das amostras disponíveis com $\delta$ e $\tau$ padrão), contra **87,08%** do CART batch. É necessário ajustar hiperparâmetros (ex.: $\tau$ maior) para este perfil.
> - **Bikes (atributos em escalas brutas distintas):** o CART batch atinge $R^2 = 0{,}897$, enquanto River ($-0{,}611$) e C11 ($-0{,}477$ no Tier 256) ficam abaixo do preditor constante, indicando instabilidade das folhas lineares SGD sem normalização prévia dos atributos.
> - **SEA:** o CART batch mantém vantagem marginal de $+1{,}1$ ponto percentual (99,20% vs 98,10%).
> - **RAM intra-treino do Tier 256:** o modelo estático do Tier 256 (473–525 KB) não é menor que o do River em árvores pequenas (River: 314–787 KB em 5 dos 8 datasets). A vantagem do C11 está no Peak RSS do processo, no heap nulo, na velocidade e nos Tiers 8 e 4.

---

### 3.3 Desempenho em Classificação em Streaming: Tier 256 vs Tier 8

Em fluxos com atributos contínuos e categóricos (SEA, Agrawal, Hyperplane e LED), a árvore C11 alcança **paridade estatística exata** com o River e demonstra que o mecanismo de **Desativação por Promessa** (*Promise Deactivation*) preserva integralmente a acurácia no **Tier 8** consumindo **$31\times$ menos RAM**:

- **Tier 256 (Padrão do Repositório / $\le 524{,}7\text{ KB}$ RAM):** 256 folhas ativas e 4.096 nós estruturais, para MCUs de médio/alto porte ($\le 1\text{ MB}$ de SRAM, como ESP32, RP2040/RP2350, STM32F4/F7/H7).
- **Tier 8 (Perfil Ultra-Restrito / $\approx 16{,}6\text{ KB}$ RAM):** 8 folhas ativas e 128 nós estruturais, para microcontroladores ultrarrestritos (16 a 32 KB de SRAM, como ARM Cortex-M0+, SAMD21, MSP430, AVR estendido).

| Fluxo / Dataset | Algoritmo / Perfil de Memória | Acurácia Teste (%) | F1-Score Macro (%) | Acurácia Prequential (%) | Total Nós | Folhas Ativas | RAM Estática Modelo | Pico Heap Dinâmico | Tempo/amostra (µs) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** (3 contínuas, 2 classes) | **C11 Tier 256 (`float32`)** | **98,10%** | **97,81%** | 96,04% | 11 | 6 | **524,7 KB** | **0,0 KB (Zero malloc)** | **0,61 µs** | **344,2x** |
| | **C11 Tier 8 (`float32`)** | **98,10%** | **97,81%** | 96,04% | 11 | 6 | **16,6 KB** | **0,0 KB (Zero malloc)** | **0,91 µs** | **194,4x** |
| | River (Hoeffding Tree) | 98,10% | 97,81% | 96,04% | 11 | 6 | 10,0 KB | 193,7 KB | 211,71 µs | 1,0x |
| | Scikit-Learn (CART Batch) | 99,18% | 99,06% | N/A | 159 | N/A | 22,4 KB | 789,2 KB | 2,95 µs | Baseline |
| **Agrawal** (9 numéricas, 2 classes) | **C11 Tier 256 (`float32`)** | **100,00%** | **100,00%** | 98,58% | 9 | 5 | **524,7 KB** | **0,0 KB (Zero malloc)** | **1,06 µs** | **299,7x** |
| | **C11 Tier 8 (`float32`)** | **100,00%** | **100,00%** | 98,58% | 9 | 5 | **16,6 KB** | **0,0 KB (Zero malloc)** | **0,67 µs** | **436,6x** |
| | River (Hoeffding Tree) | 100,00% | 100,00% | 98,59% | 9 | 5 | 8,3 KB | 227,5 KB | 316,80 µs | 1,0x |
| | Scikit-Learn (CART Batch) | 100,00% | 100,00% | N/A | 5 | N/A | 0,7 KB | 1.114,3 KB | 3,09 µs | Baseline |
| **Hyperplane** (10 contínuas, 2 classes) | **C11 Tier 256 (`float32`)** | **91,12%** | **91,11%** | 89,06% | 7 | 4 | **524,7 KB** | **0,0 KB (Zero malloc)** | **1,23 µs** | **319,6x** |
| | **C11 Tier 8 (`float32`)** | **91,12%** | **91,11%** | 89,06% | 7 | 4 | **16,6 KB** | **0,0 KB (Zero malloc)** | **1,37 µs** | **292,3x** |
| | River (Hoeffding Tree) | 91,12% | 91,11% | 89,06% | 7 | 4 | 6,7 KB | 325,6 KB | 393,70 µs | 1,0x |
| | Scikit-Learn (CART Batch) | 84,38% | 84,38% | N/A | 511 | N/A | 71,9 KB | 1.169,1 KB | 12,18 µs | Baseline |
| **LED** (7 binárias, 2 classes) | **C11 Tier 256 (`float32`)** | **90,13%** | **90,13%** | 87,91% | 13 | 7 | **524,7 KB** | **0,0 KB (Zero malloc)** | **0,65 µs** | **366,4x** |
| | **C11 Tier 8 (`float32`)** | **90,13%** | **90,13%** | 87,91% | 13 | 7 | **16,6 KB** | **0,0 KB (Zero malloc)** | **0,86 µs** | **281,9x** |
| | River (Hoeffding Tree) | 90,13% | 90,13% | 87,92% | 13 | 7 | 11,7 KB | 386,9 KB | 237,03 µs | 1,0x |
| | Scikit-Learn (CART Batch) | 89,90% | 89,90% | N/A | 237 | N/A | 33,3 KB | 1.005,1 KB | 1,21 µs | Baseline |

> [!TIP]
> **Destaques de Classificação:**
> - **Retenção Integral de Acurácia no Tier 8:** Em todos os fluxos testados, o modelo operando com teto estrito de apenas 8 folhas ativas (16,6 KB) alcançou rigorosamente a mesma acurácia de teste do perfil de 256 folhas (524,7 KB), confirmando a eficiência dos observadores de Welford e da predição Naive Bayes Adaptive (NBA).
> - **Latência Sub-microssegundo:** Processamento em **0,61 a 1,37 µs por amostra** com **speedup de 194x a 436x sobre o River**, sem alocar um único byte no heap (`0.0 KB malloc`).

---

### 3.4 Desempenho em Regressão Contínua e Concept Drift: Tier 256 vs Tier 8

Avaliando tarefas de regressão numérica (Friedman #1 e Planes2D) e fluxos sujeitos a *concept drift* abrupto (HAT-R com detectores ADWIN2 e teste Z bilateral de variância):

| Fluxo / Experimento | Algoritmo / Perfil de Memória | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Total Nós | Folhas Ativas / Switches | RAM Estática Modelo | Pico Heap Dinâmico | Tempo/amostra (µs) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** (10 contínuas, não-linear) | **C11 VFDT-R Tier 256 (`float32`)** | **1,8094** | **2,3078** | **0,7857** | 2,0536 | 71 | 36 folhas | **524,7 KB** | **0,0 KB (Zero malloc)** | **0,90 µs** | **285,9x** |
| | **C11 VFDT-R Tier 8 (`float32`)** | **2,2983** | **2,8873** | **0,6645** | 2,2680 | 41 | 8 folhas | **16,6 KB** | **0,0 KB (Zero malloc)** | **0,98 µs** | **260,9x** |
| | River HTR (Adaptive Leaf) | 1,6722 | 2,1198 | 0,8192 | 1,8807 | 81 | 41 folhas | 218,6 KB | 1.596,3 KB | 256,70 µs | 1,0x |
| | Scikit-Learn (CART Batch, 511 nós) | 1,6501 | 2,0804 | 0,8258 | N/A | 511 | N/A | 67,9 KB | 835,1 KB | 11,47 µs | Baseline |
| | Scikit-Learn (CART podado, 15 nós) | 2,5301 | 3,1707 | 0,5955 | N/A | 15 | N/A | 2,0 KB | 835,2 KB | 6,24 µs | Baseline |
| **Planes2D** (10 contínuas, planos 2D) | **C11 VFDT-R Tier 256 (`float32`)** | **0,8145** | **1,0245** | **0,9456** | 0,8545 | 97 | 49 folhas | **524,7 KB** | **0,0 KB (Zero malloc)** | **0,94 µs** | **184,9x** |
| | River HTR (Adaptive Leaf) | 0,8264 | 1,0396 | 0,9440 | 0,8613 | 65 | 33 folhas | 187,6 KB | 593,3 KB | 172,85 µs | 1,0x |
| | Scikit-Learn (CART Batch, 23k nós) | 1,1336 | 1,4161 | 0,8961 | N/A | 23.547 | N/A | 3.127,3 KB | 835,3 KB | 3,21 µs | Baseline |
| **Friedman Drift Abrupto** (v0 $\to$ v3) | **C11 HAT-R (`float32`, Adaptativo)** | **3,4142** | **4,3958** | **0,2118** | 2,2310 | 27 | **3 switches** | **620,5 KB** | **0,0 KB (Zero malloc)** | **1,96 µs** | **185,1x** |
| | River HAT-R (HoeffdingAdaptiveTree) | 3,4131 | 4,3523 | 0,2273 | 2,0635 | 25 | Adaptive | 87,1 KB | 1.547,2 KB | 363,24 µs | 1,0x |
| | River Static (VFDT-R Não-Adaptativo) | 3,2064 | 4,0621 | 0,3269 | 2,1739 | 77 | N/A | 211,5 KB | 1.540,8 KB | 227,16 µs | Baseline |

> [!TIP]
> **Destaques de Regressão:**
> - **Superação do CART em Ultra-Baixa RAM:** No Tier 8 (16,6 KB), a combinação do critério de Redução de Variância com folhas de regressão linear SGD preservou $R^2 = 0{,}6645$, superando significativamente o Scikit-Learn CART com restrição similar de nós ($R^2 = 0{,}5955$).
> - **Adaptação Online com HAT-R:** Diante de drift abrupto, o ADWIN2 em C11 realizou 3 substituições estruturais (*switches*) via Teste Z Bilateral, recuperando a capacidade preditiva com throughput $185\times$ superior ao River.

---

### 3.5 Avaliação no MNIST: Por que Árvores de Streaming Não São Recomendadas para Dados Não-Tabulares?

Para mapear os limites teóricos e práticos do paradigma VFDT em problemas de visão computacional, o modelo foi submetido ao benchmark triplo no **MNIST** (com *average pooling* espacial $4 \times 4$, reduzindo a imagem de $28 \times 28$ para 16 atributos contínuos normalizados):

| Experimento MNIST (16 features) | Solução / Algoritmo | Acurácia Teste (%) | F1-Score Macro (%) | Acurácia Prequential (%) | Total de Nós | RAM Estática Modelo | Pico Heap Dinâmico | Tempo/amostra (µs) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MNIST 10 Classes** (Dígitos 0..9) | Scikit-Learn (CART Batch) | 71,73% | 71,28% | N/A | 4.047 | 822,0 KB | 653,1 KB | 15,00 µs | Baseline |
| | River (Hoeffding Tree) | 51,38% | 50,82% | 50,46% | 1 | 1,6 KB | 784,6 KB | 2.044,86 µs | 1,0x |
| | **C11 Embedded (`float32`)** | **51,38%** | **50,82%** | 50,44% | **1** | **509,9 KB** | **0,0 KB (Zero malloc)** | **6,24 µs** | **327,5x** |
| **MNIST 4 Classes** (Dígitos 0..3) | Scikit-Learn (CART Batch) | 87,20% | 87,05% | N/A | 1.081 | 168,9 KB | 314,7 KB | 13,36 µs | Baseline |
| | River (Hoeffding Tree) | 76,57% | 76,08% | 77,24% | 1 | 1,6 KB | 323,0 KB | 911,82 µs | 1,0x |
| | **C11 Embedded (`float32`)** | **76,57%** | **76,08%** | 77,13% | **1** | **443,6 KB** | **0,0 KB (Zero malloc)** | **2,80 µs** | **325,7x** |
| **MNIST Binário** (Pares vs Ímpares) | Scikit-Learn (CART Batch) | 85,87% | 85,87% | N/A | 2.167 | 304,7 KB | 622,3 KB | 13,50 µs | Baseline |
| | River (Hoeffding Tree) | 78,60% | 78,59% | 77,31% | 11 | 10,0 KB | 875,2 KB | 362,02 µs | 1,0x |
| | **C11 Embedded (`float32`)** | **78,60%** | **78,59%** | 77,31% | **11** | **443,6 KB** | **0,0 KB (Zero malloc)** | **1,16 µs** | **312,1x** |

> [!WARNING]
> ### Por que Hoeffding Trees NÃO São Recomendadas para Dados Não-Tabulares (Imagens, Áudio e NLP)?
>
> Embora a nossa solução C11 seja **$\approx 320\times$ mais rápida que o River** e opere com **zero alocação dinâmica em runtime**, os testes no MNIST revelam limitações intrínsecas de árvores em streaming diante de dados não-estruturados:
>
> 1. **Ausência de Invariâncias e Correlações Espaciais Locais:** Imagens possuem forte correlação 2D entre pixels vizinhos e invariância a translação/rotação. Árvores particionam o espaço com cortes ortogonais em um único pixel por vez ($x_{\text{pixel}} \le \theta$), sendo incapazes de compor representações hierárquicas espaciais (diferente de convoluções).
> 2. **Violação da Independência do Naive Bayes (NBA) nas Folhas:** A predição nas folhas utiliza Naive Bayes contínuo Gaussiano, cuja hipótese central assume que os atributos são condicionalmente independentes: $P(X \mid C) = \prod P(x_i \mid C)$. Em imagens, pixels adjacentes compartilham fortíssima dependência física. Essa quebra teórica impõe um teto estrito (*Bayes Error*) de $\approx 51\%$ em 10 classes para modelos baseados em nós únicos.
> 3. **Exigência Amostral do Hoeffding Bound em Espaços Multiclasse:** Para $C = 10$ classes, a escala $R = \log_2(10) \approx 3{,}32$ exige mais de $35.500$ amostras para desempate estatístico seguro ($\Delta \bar{G} > \epsilon$). Como dezenas de pixels vizinhos apresentam ganhos de impureza praticamente indistinguíveis ($\Delta \bar{G} \approx 0{,}01$), a árvore **não atinge significância estatística com 14.000 amostras** e permanece em 1 nó raiz. Já no cenário binário ($C=2$), onde a exigência caiu para $n > 3.224$, a árvore realizou splits sucessivos (11 nós), saltando a acurácia para **$78{,}60\%$**.
> 4. **Diretriz de Escolha de Modelo (Regra Prática):**
>    - **Quando Usar (Recomendado):** Dados **tabulares estruturados** em streaming contínuo, sinais de sensores industriais e IoT (vibração, acelerômetros, temperatura, pressão, corrente, RPM), telemetria preditiva de baterias (BMS), séries temporais financeiras e análise de pacotes de rede (NetFlow/IP).
>    - **Quando NÃO Usar:** Visão computacional (imagens e vídeo), sinais brutos de áudio (espectrogramas densos) e processamento de linguagem natural (texto não-estruturado). Para visão computacional em microcontroladores (*TinyML Vision*), utilize redes neurais convolucionais (CNNs) quantizadas em `int8` (ex: MobileNet via TensorFlow Lite for Microcontrollers).

---

### 3.6 Emulação Hardware-in-the-Loop em ESP32 (Xtensa Dual-Core LX6 @ 240 MHz)

Para certificar o comportamento determinístico em **ambiente embarcado emulado (QEMU Xtensa)** antes do deployment físico em placa real, a biblioteca foi avaliada em um ambiente **Hardware-in-the-Loop (HIL)** emulado via **QEMU Xtensa** executando firmware **ESP-IDF v5.1 / FreeRTOS** compilado para o microcontrolador **ESP32** (SoC Dual-Core Xtensa LX6 @ 240 MHz).

**Configuração do Experimento:**
- **Perfil de Compilação:** Tier 8 Universal (`-DHT_MAX_NODES=128 -DHT_MAX_ACTIVE_LEAVES=8 -DHT_MAX_FEATURES=64 -DHT_MAX_CLASSES=10`) ocupando $\approx 111{,}3\text{ KB}$ de SRAM estática (`.bss`).
- **Protocolo de Injeção em Streaming:** O host (`emulation/esp32/stream_runner.py`) injetou **152.114 amostras reais** dos 8 datasets do repositório (até 20.000 amostras por dataset; 70% treino e 30% teste) através de um socket TCP/UART com protocolo de janela deslizante (*sliding window* $W=16$).
- **Cronometria de Precisão:** A latência por amostra é cronometrada em ciclos de clock de hardware através da instrução de registrador `esp_cpu_get_cycle_count()` no núcleo Xtensa ($1\text{ ciclo} = 4{,}1667\text{ ns}$ @ 240 MHz).
- **Relatório Completo:** Detalhes da infraestrutura e logs em **[`emulation/esp32/esp32_emulation_report.md`](emulation/esp32/esp32_emulation_report.md)**.

#### 3.6.1 Desempenho em Classificação na ESP32 (4 Datasets)

| Dataset | Amostras Treino | Amostras Teste | Acurácia Holdout (%) | Precisão Macro (%) | Recall Macro (%) | F1-Score Macro (%) | Acurácia Prequential (%) | Topologia (Nós / Folhas) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** (3 feat, 2 classes) | 14.000 | 6.000 | **98,10%** | **98,11%** | **97,52%** | **97,81%** | 96,04% | 11 nós (6 folhas) |
| **Electricity** (8 feat, 2 classes) | 14.000 | 6.000 | **64,05%** | **63,68%** | **63,29%** | **63,32%** | 62,99% | 9 nós (5 folhas) |
| **Forest Covertype** (54 feat, 7 classes) | 14.000 | 6.000 | **52,62%** | **50,23%** | **51,59%** | **50,63%** | 41,64% | 1 nós (1 folha) |
| **Airlines Delay** (7 feat, 2 classes) | 14.000 | 6.000 | **76,80%** | **68,70%** | **54,15%** | **52,21%** | 75,99% | 7 nós (4 folhas) |

---

#### 3.6.2 Desempenho em Regressão Contínua na ESP32 (4 Datasets)

| Dataset | Amostras Treino | Amostras Teste | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Prequential RMSE | Prequential $R^2$ | Topologia (Nós / Folhas) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** (10 feat, contínuo) | 14.000 | 6.000 | **2,2983** | **2,8873** | **0,6645** | 2,2680 | 2,8777 | 0,6631 | 41 nós (8 folhas) |
| **Bikes** (12 feat, contínuo) | 12.165 | 5.214 | **53,1949** | **79,9699** | **-0,1341** | 59,6959 | 91,8369 | 0,4256 | 27 nós (8 folhas) |
| **Appliances Energy** (28 feat, contínuo) | 13.814 | 5.921 | **33,1660** | **42,3745** | **0,6787** | 42,1982 | 508,0700 | -44,5799 | 35 nós (8 folhas) |
| **NYC Taxi** (7 feat, contínuo) | 10.500 | 4.500 | **353,1150** | **458,2122** | **0,1419** | 384,5726 | 2.306,8679 | -20,2186 | 35 nós (8 folhas) |

---

#### 3.6.3 Métricas de Performance, Latência e Recursos em Hardware Emulado (QEMU ESP32 @ 240 MHz)

> [!NOTE]
> **Definição e Medição das Colunas de Latência no ESP32:**
> - **Ciclo Test-Then-Train:** Cada medição cobre o tempo total para predição (`predict`) + treino incremental (`learn_one`) daquela amostra.
> - **Base de Cronometria:** A latência é apurada diretamente no firmware em **ciclos de clock** através da instrução de registrador `esp_cpu_get_cycle_count()` do núcleo Xtensa. Em seguida, é convertida para microssegundos ($\mu\text{s}$) com base na frequência nominal de 240 MHz ($\text{Latência}(\mu\text{s}) = \text{ciclos} / 240$).
> - **Latência Média vs Cauda ($p_{95}, p_{99}$):** A latência média reflete a velocidade de cruzeiro do modelo nas folhas $O(1)$. Os percentis $p_{95}$ e $p_{99}$ capturam o custo adicional momentâneo ocorrido a cada 200 amostras (`grace_period`), momento em que o microcontrolador realiza a varredura estatística do *Hoeffding Bound* para testar possíveis divisões de nós.

| Dataset | Tarefa | Latência Média | Latência $p_{95}$ | Latência $p_{99}$ | Throughput (Vazão) | Model Footprint | Consumo RAM (SRAM / Heap) | FLOPs / Amostra |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classificação | **73,81 µs** | 132,11 µs | 535,88 µs | **13.548 am/s** | 340 B (.bin) / 111,3 KB | 111,3 KB (.bss) / **0 B Heap** | **69 FLOPs** |
| **Electricity** | Classificação | **115,50 µs** | 206,87 µs | 965,48 µs | **8.658 am/s** | 284 B (.bin) / 111,3 KB | 111,3 KB (.bss) / **0 B Heap** | **149 FLOPs** |
| **Forest Covertype** | Classificação | **1.947,20 µs** | 2.250,05 µs | 4.288,57 µs | **514 am/s** | 60 B (.bin) / 111,3 KB | 111,3 KB (.bss) / **0 B Heap** | **2.250 FLOPs** |
| **Airlines Delay** | Classificação | **115,84 µs** | 162,18 µs | 228,89 µs | **8.633 am/s** | 228 B (.bin) / 111,3 KB | 111,3 KB (.bss) / **0 B Heap** | **133 FLOPs** |
| **Friedman #1** | Regressão | **53,68 µs** | 83,53 µs | 113,50 µs | **18.629 am/s** | 1.180 B (.bin) / 111,3 KB | 111,3 KB (.bss) / **0 B Heap** | **138 FLOPs** |
| **Bikes** | Regressão | **63,20 µs** | 95,74 µs | 121,85 µs | **15.823 am/s** | 788 B (.bin) / 111,3 KB | 111,3 KB (.bss) / **64 B Heap** | **160 FLOPs** |
| **Appliances Energy** | Regressão | **79,85 µs** | 151,95 µs | 195,40 µs | **12.523 am/s** | 1.012 B (.bin) / 111,3 KB | 111,3 KB (.bss) / **0 B Heap** | **336 FLOPs** |
| **NYC Taxi** | Regressão | **41,81 µs** | 80,66 µs | 157,95 µs | **23.918 am/s** | 1.012 B (.bin) / 111,3 KB | 111,3 KB (.bss) / **0 B Heap** | **105 FLOPs** |

> [!TIP]
> **Destaques da Validação em Hardware Emulado (QEMU ESP32):**
> - **Zero Alocação em Runtime Comprovada:** O monitoramento contínuo via `esp_get_free_heap_size()` confirmou ausência de fragmentação ou vazamento no FreeRTOS durante o processamento das 152.114 amostras reais.
> - **Throughput de Alta Frequência em Microcontrolador:** Com vazão de **8.600 a 23.900 amostras/segundo** (para fluxos de 3 a 28 atributos a 240 MHz), o modelo é plenamente capaz de operar em malhas fechadas de controle e telemetria preditiva de alta frequência em IoT.
> - **Latência de Cauda Controlada:** O percentil $p_{99}$ permaneceu abaixo de $230\,\mu\text{s}$ na maioria dos fluxos, demonstrando extrema estabilidade temporal em tempo real.

---

## 4. Limitações e Pontos Fracos do Modelo

Embora a implementação ofereça latência sub-microssegundo e opere com determinismo estrito de memória em microcontroladores ($\le 1\text{ MB}$ de RAM com 0,0 KB de heap dinâmico), o modelo e a arquitetura impõem compromissos (*trade-offs*) importantes que devem ser avaliados no dimensionamento da solução:

### 4.1 Limitações Algorítmicas do Paradigma de Streaming (VFDT / Hoeffding Bound)
- **Atraso Amostral no Crescimento da Árvore (*Sample Inefficiency*):** O critério estatístico do Limite de Hoeffding $\epsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$ é rigoroso e conservador. Em datasets pequenos ($< 1.000$ amostras) ou regimes de dados esparsos, o modelo demora centenas de amostras para realizar os primeiros splits, enquanto um algoritmo em lote (CART) já construiria uma árvore profunda imediatamente.
- **Dificuldade em Atributos Fortemente Concorrentes (*Tie-Breaking* $\tau$):** Quando dois atributos apresentam ganhos de impureza idênticos ($\Delta \bar{G} \approx 0$), o split só ocorre quando $\epsilon < \tau$, o que pode exigir dezenas de milhares de amostras para desempatar, retardando o aprendizado local.
- **Divisões Estritamente Ortogonais (Eixo-Alinhadas):** As decisões de split particionam o espaço exclusivamente por limiares paralelos aos eixos ($X_f \le \theta$). Para fronteiras diagonais ou funções com rotação linear contínua (ex: $X_1 + X_2 > c$), a árvore necessita de dezenas de nós em padrão de escada para aproximar o que um modelo linear simples resolveria com uma única equação.
- **Hipótese de Gaussianidade Unimodal para Variáveis Contínuas:** Os observadores numéricos de Welford aproximam cada atributo por $\mathcal{N}(\mu, \sigma^2)$ condicionado à classe. Em distribuições multimodais, fortemente enviesadas ou com caudas pesadas, as sugestões de pontos de corte e as probabilidades do Naive Bayes podem sofrer imprecisão.
- **Ausência de Poda Retrospectiva (*Post-Pruning*) no VFDT Estático:** Na versão estática (VFDT), decisões de split tomadas no início do fluxo são irrevogáveis (não há *cost-complexity pruning* como no CART). Apenas a variante adaptativa (HAT / HAT-R) possui capacidade de substituição de ramos obsoletos ao detectar drift.

### 4.2 Limitações de Firmware e Dimensionamento Estático em C11
- **Rigidez em Tempo de Compilação (Zero Dimensionamento Dinâmico em Runtime):** A capacidade máxima de nós (`HT_MAX_NODES`), folhas ativas (`HT_MAX_ACTIVE_LEAVES`), atributos (`HT_MAX_FEATURES`), classes (`HT_MAX_CLASSES`) e categorias nominais (`HT_MAX_CATEGORIES`) é fixada na compilação via macros. Alterar dimensões do problema exige recompilar o firmware.
- **Degradação de Aprendizado sob Esgotamento do Pool de Folhas:** Em perfis com extrema restrição de RAM (como o Tier 8, limitado a 8 folhas ativas), quando a capacidade é atingida o algoritmo desativa as folhas menos promissoras. Embora preserve as folhas ativas com NBA, folhas inativas não geram novos ramos até que sejam reativadas.
- **Processamento Single-Core Sequencial:** O núcleo C11 é 100% sequencial, livre de chamadas de threads (`pthread`), primitivas de sincronização (`mutex`) ou alocação dinâmica. Aplicações multi-core embarcadas (ex: RP2040 dual-core ou ESP32) devem isolar instâncias independentes por núcleo.
- **Sem Pipelines Internos de Pré-Processamento:** O modelo espera que os dados de entrada já estejam normalizados, discretizados (para atributos categóricos) e validados pelo firmware. Não há rotinas embutidas de *One-Hot Encoding*, redução de dimensionalidade (PCA) ou imputação avançada.

### 4.3 Inadequação para Dados Não-Tabulares e Alta Cardinalidade
- **Incompatibilidade Prática com Imagens, Áudio e Texto (Dados Não-Tabulares):** Conforme demonstrado no benchmark do MNIST, árvores ortogonais não capturam dependências espaciais 2D, e o Naive Bayes das folhas quebra gravemente com pixels vizinhos fortemente correlacionados.
- **Limite Fixo para Variáveis Categóricas:** A macro `HT_MAX_CATEGORIES` (padrão 8) impede o tratamento direto de variáveis categóricas de alta cardinalidade (ex: CEPs, IDs de dispositivos, hashes).

### 4.4 Precisão de Ponto Flutuante em Fluxos Ultra-Longos
- **Sensibilidade de Cancelamento em `float32`:** Embora suficiente para microcontroladores com FPU simples, somatórios acumulados de Welford ao longo de centenas de milhões de amostras em `float32` podem acumular erro de arredondamento em casos de variância ínfima somada a médias elevadas ($\sigma^2 \ll \mu^2$).

### 4.5 Matriz de Decisão: Quando Usar vs Quando Evitar

| Cenário / Requisito | Usar Hoeffding Tree C11? | Abordagem Alternativa Recomendada |
| :--- | :---: | :--- |
| **Telemetria de Sensores IoT / Dados Tabulares Online** | **SIM (Ideal)** | — |
| **Ambientes com $\le 1\text{ MB}$ de RAM e Zero Heap (`malloc`)** | **SIM (Ideal)** | — |
| **Regimes com Mudança de Conceito (*Concept Drift*)** | **SIM (HAT / HAT-R)** | — |
| **Visão Computacional / Classificação de Imagens** | **NÃO** | CNNs quantizadas em `int8` (ex: TensorFlow Lite Micro) |
| **Áudio Bruto / Espectrogramas Densos** | **NÃO** | 1D-CNN ou DSCNN (ex: Edge Impulse) |
| **Processamento de Linguagem Natural / Texto Não-Estruturado** | **NÃO** | Embeddings compactos + Regressores Lineares / TinyTransformers |
| **Datasets Pequenos em Lote (< 1.000 amostras offline)** | **NÃO** | Scikit-Learn CART / Random Forest offline em Python |

---

## 5. Documentação Completa (Wiki, Estudos e Benchmarks)

Toda a documentação detalhada do projeto está organizada na pasta **[`wiki/`](wiki/Home.md)** (sincronizada com a aba **Wiki** do GitHub), além dos estudos teóricos em `docs/` e relatórios em `benchmarks/`:

| Seção da Documentação | O que você encontra lá |
| :--- | :--- |
| **[Wiki Home](wiki/Home.md)** | Índice geral da Wiki, navegação rápida e instruções de sincronização com o GitHub Wiki. |
| **[Exemplos e Receitas da API](wiki/Exemplos-e-Receitas-API.md)** | Receitas completas em C11 para `VFDT`, `HAT` (`ADWIN2`), `VFDT-R` / `HAT-R`, `NaN` e `ht_tree_save` / `ht_tree_load`. |
| **[Referência de Parâmetros e Macros](wiki/Referencia-de-Parametros-e-Macros.md)** | Tabelas completas de `ht_config_params_t`, `ht_adwin_config_t`, macros de `ht_config.h`, fórmulas em bytes e perfis de MCU. |
| **[Orçamento de Memória e Cross-Compilation](wiki/Orcamento-de-Memoria-e-Cross-Compilation.md)** | Consumo de Flash/ROM, peso de arquivos `.bin` serializados e flags para `arm-none-eabi-gcc` e Linux embarcado. |
| **[Arquitetura e Fluxogramas](wiki/Arquitetura-e-Fluxogramas.md)** | Comparativo Hoeffding vs CART, decisões de design C11 e fluxogramas Mermaid passo a passo de todos os algoritmos. |
| **[Origem: River, Sklearn e Originalidade](wiki/Origem-River-Scikit-Learn-e-Originalidade.md)** | Genealogia técnica e matriz tripla: o que veio do River, do Scikit-Learn e inovações originais C11 (zero `malloc`, reciclagem). |
| **[Versionamento, Releases e Roadmap](wiki/Versionamento-Releases-e-Roadmap.md)** | Política SemVer 2.0.0 (`VERSION` / `ht_version()`), publicação de releases no GitHub e **Planos Futuros / Roadmap** (Ponto Fixo/INT8, ESP32 bare-metal, Python bindings, Ensembles). |
| **Main Benchmark Suite ([`main_benchmark/`](main_benchmark/))** | Suíte principal de desempenho com avaliação realista de RAM (Batch vs Streaming) e Tiers C11 nos 8 datasets ([`main_benchmark_report.md`](main_benchmark/main_benchmark_report.md)). |
| **Emulação ESP32 Hardware-in-the-Loop ([`emulation/esp32/`](emulation/esp32/))** | Firmware ESP-IDF/FreeRTOS, QEMU Xtensa @ 240 MHz, streaming de 152k amostras e relatório completo ([`esp32_emulation_report.md`](emulation/esp32/esp32_emulation_report.md)). |
| **Estudos Matemáticos ([`docs/`](docs/))** | Notas teóricas de VFDT ([`01`](docs/01_hoeffding_tree_vfdt.md)), Observadores ([`02`](docs/02_attribute_observers.md)), HAT/ADWIN ([`03`](docs/03_concept_drift_hat.md)), River ([`04`](docs/04_river_mapping.md)), Sklearn ([`05`](docs/05_sklearn_architecture.md)), Plano ([`06`](docs/06_implementation_plan.md)) e Regressão ([`07`](docs/07_regression_trees_fimtdd.md)). |
| **Relatórios de Benchmark ([`benchmarks/`](benchmarks/))** | Paridade vs River em Classificação ([`benchmark_report.md`](benchmarks/benchmark_report.md)), Escalonamento de RAM Tier 256 a Tier 4 ([`memory_benchmark_report.md`](benchmarks/memory_benchmark_report.md)), MNIST Triplo em Visão ([`triple_benchmark_report.md`](benchmarks/triple_benchmark_report.md)), Concept Drift ([`drift_benchmark_report.md`](benchmarks/drift_benchmark_report.md)) e Regressão ([`regression_benchmark_report.md`](benchmarks/regression_benchmark_report.md)). |

> [!NOTE]
> **Planos Futuros e Roadmap:** O detalhamento dos próximos passos e extensões planejadas para o projeto (incluindo aritmética de ponto fixo/INT8 para MCUs sem FPU, bindings em Python com CFFI, validação de firmware com máquina de estados em ESP32 real e ensembles em streaming como Online Bagging) está documentado na Wiki em **[`wiki/Versionamento-Releases-e-Roadmap.md` (Seção 5: Possíveis Trabalhos Futuros e Extensões)](wiki/Versionamento-Releases-e-Roadmap.md#5-possíveis-trabalhos-futuros-e-extensões-roadmap)**.

---

## 6. Estrutura do Repositório

```text
pessoal_minimal-decision-tree/
├── LICENSE                      # Licença BSD 3-Clause + Avisos de Copyright (River e Scikit-Learn)
├── VERSION                      # Versão canônica do projeto (SemVer 2.0.0)
├── README.md                    # Visão geral, Quick Start e índice de navegação
├── Makefile                     # Root Makefile (compilação, testes e benchmarks)
├── wiki/                        # Páginas da GitHub Wiki (Exemplos, Parâmetros, Memória, Arquitetura, Roadmap)
├── docs/                        # Estudos matemáticos dos papers e plano de implementação (01 a 07)
├── references/                  # Artigos originais (PDFs) e código de referência (River e Scikit-Learn)
├── code/                        # Implementação em C11 puro (include/, src/, tests/ e Makefile)
├── emulation/                   # Emulação Hardware-in-the-Loop em ESP32 (ESP-IDF, FreeRTOS e QEMU)
├── main_benchmark/              # Suíte principal de benchmark de desempenho (8 datasets, profiling de RAM)
└── benchmarks/                  # Scripts e runners de validação comparativa (C11 vs River vs Sklearn)
```

---

## 7. Licença (`BSD 3-Clause`)

Distribuído sob a licença permissiva **[BSD 3-Clause License](LICENSE)** (`SPDX-License-Identifier: BSD-3-Clause`), permitindo uso livre acadêmico, pessoal ou comercial (em código-fonte ou firmware binário) e preservando integralmente os avisos de direitos autorais dos projetos de referência **River** ([`references/river_tree/LICENSE`](references/river_tree/LICENSE)) e **Scikit-Learn** ([`references/sklearn_tree/COPYING`](references/sklearn_tree/COPYING)).
