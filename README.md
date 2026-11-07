# Minimal Decision Tree (Embedded Hoeffding Tree in C11)

Implementação ultraleve de **Hoeffding Trees (`VFDT`, `HAT`, `VFDT-R` / `FIMT-DD` e `HAT-R`)** em **C11 puro**, projetada para **treino incremental (*online/streaming*) e inferência em tempo real** em microcontroladores e sistemas embarcados com forte restrição de memória (**$\le 1\text{ MB}$ de RAM**, escalável para **$\approx 11\text{ KB}$**, com **zero alocação dinâmica `malloc`/`free`**).

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

## 3. Resumo de Consumo de Memória (RAM & Flash)

O footprint de RAM de `ht_tree_t` é dimensionado estaticamente em tempo de compilação via flags `-D` (veja todos os detalhes em **[Referência de Parâmetros e Macros](wiki/Referencia-de-Parametros-e-Macros.md)** e **[Orçamento de Memória e Cross-Compilation](wiki/Orcamento-de-Memoria-e-Cross-Compilation.md)**):

| Perfil de Hardware Alvo | Flags de Compilação (`CFLAGS_EXTRA`) | Tamanho do Nó | Folha Ativa | RAM Total (`ht_tree_t`) | Flash (`.text` em `-Os`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Somente Inferência** | `-DHT_MAX_NODES=256 -DHT_MAX_ACTIVE_LEAVES=1 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2` | $24\text{ B}$ | $688\text{ B}$ | **$\approx 11,3\text{ KB}$** | **$\approx 6\text{–}8\text{ KB}$** |
| **MCU Ultra-Fraco (32–64 KB)** | `-DHT_MAX_NODES=128 -DHT_MAX_ACTIVE_LEAVES=16 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2 -DHT_MAX_CATEGORIES=4 -DHT_REG_NUM_SLOTS=4` | $24\text{ B}$ | $560\text{ B}$ | **$\approx 14,2\text{ KB}$** | **$\approx 16\text{–}28\text{ KB}$** |
| **MCU Compacto (64–128 KB)** | `-DHT_MAX_NODES=256 -DHT_MAX_ACTIVE_LEAVES=32 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2` | $24\text{ B}$ | $688\text{ B}$ | **$\approx 32,2\text{ KB}$** | **$\approx 16\text{–}28\text{ KB}$** |
| **MCU Intermediário (256–512 KB)** | `-DHT_MAX_NODES=1024 -DHT_MAX_ACTIVE_LEAVES=64 -DHT_MAX_FEATURES=12 -DHT_MAX_CLASSES=4` | $28\text{ B}$ | $1.024\text{ B}$ | **$\approx 110,3\text{ KB}$** | **$\approx 16\text{–}28\text{ KB}$** |
| **Padrão do Repositório ($\le 1\text{ MB}$)** | *(Padrão: `NODES=4096`, `LEAVES=256`, `FEATURES=16`, `CLASSES=4`)* | $28\text{ B}$ | $1.360\text{ B}$ | **$\approx 524,7\text{ KB}$** | **$\approx 22\text{–}28\text{ KB}$** |

---

## 4. Desempenho e Benchmarks Comparativos

A biblioteca foi avaliada empiricamente contra duas referências consolidadas da literatura: o **[`online-ml/river`](https://github.com/online-ml/river)** (padrão de referência para aprendizado em streaming contínuo em Python/Cython) e o **[`scikit-learn`](https://scikit-learn.org)** (baseline em lote / *batch* CART).

Para ilustrar o comportamento em hardware embarcado, os testes destacam dois limites operacionais:
- **Tier 256 (Padrão do Repositório / $\le 524{,}7\text{ KB}$ RAM):** 256 folhas ativas e 4.096 nós estruturais, para MCUs de médio/alto porte ($\le 1\text{ MB}$ de SRAM, como ESP32, RP2040/RP2350, STM32F4/F7/H7).
- **Tier 8 (Perfil Ultra-Restrito / $\approx 16{,}6\text{ KB}$ RAM):** 8 folhas ativas e 128 nós estruturais, para microcontroladores ultrarrestritos (16 a 32 KB de SRAM, como ARM Cortex-M0+, SAMD21, MSP430, AVR estendido).

---

### 4.1 Desempenho em Classificação em Streaming: Tier 256 vs Tier 8

Em fluxos com atributos contínuos e categóricos (SEA, Agrawal, Hyperplane e LED), a árvore C11 alcança **paridade estatística exata** com o River e demonstra que o mecanismo de **Desativação por Promessa** (*Promise Deactivation*) preserva integralmente a acurácia no **Tier 8** consumindo **$31\times$ menos RAM**:

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

### 4.2 Desempenho em Regressão Contínua e Concept Drift: Tier 256 vs Tier 8

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

### 4.3 Avaliação no MNIST: Por que Árvores de Streaming Não São Recomendadas para Dados Não-Tabulares?

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

## 5. Documentação Completa (Wiki, Estudos e Benchmarks)

Toda a documentação detalhada do projeto está organizada na pasta **[`wiki/`](wiki/Home.md)** (sincronizada com a aba **Wiki** do GitHub), além dos estudos teóricos em `docs/` e relatórios em `benchmarks/`:

| Seção da Documentação | O que você encontra lá |
| :--- | :--- |
| **[Wiki Home](wiki/Home.md)** | Índice geral da Wiki, navegação rápida e instruções de sincronização com o GitHub Wiki. |
| **[Exemplos e Receitas da API](wiki/Exemplos-e-Receitas-API.md)** | Receitas completas em C11 para `VFDT`, `HAT` (`ADWIN2`), `VFDT-R` / `HAT-R`, `NaN` e `ht_tree_save` / `ht_tree_load`. |
| **[Referência de Parâmetros e Macros](wiki/Referencia-de-Parametros-e-Macros.md)** | Tabelas completas de `ht_config_params_t`, `ht_adwin_config_t`, macros de `ht_config.h`, fórmulas em bytes e perfis de MCU. |
| **[Orçamento de Memória e Cross-Compilation](wiki/Orcamento-de-Memoria-e-Cross-Compilation.md)** | Consumo de Flash/ROM, peso de arquivos `.bin` serializados e flags para `arm-none-eabi-gcc` e Linux embarcado. |
| **[Arquitetura e Fluxogramas](wiki/Arquitetura-e-Fluxogramas.md)** | Comparativo Hoeffding vs CART, decisões de design C11 e fluxogramas Mermaid passo a passo de todos os algoritmos. |
| **[Versionamento, Releases e Roadmap](wiki/Versionamento-Releases-e-Roadmap.md)** | Política SemVer 2.0.0 (`VERSION` / `ht_version()`), publicação de releases no GitHub e trabalhos futuros. |
| **Estudos Matemáticos ([`docs/`](docs/))** | Notas teóricas de VFDT ([`01`](docs/01_hoeffding_tree_vfdt.md)), Observadores ([`02`](docs/02_attribute_observers.md)), HAT/ADWIN ([`03`](docs/03_concept_drift_hat.md)), River ([`04`](docs/04_river_mapping.md)), Sklearn ([`05`](docs/05_sklearn_architecture.md)), Plano ([`06`](docs/06_implementation_plan.md)) e Regressão ([`07`](docs/07_regression_trees_fimtdd.md)). |
| **Relatórios de Benchmark ([`benchmarks/`](benchmarks/))** | Paridade vs River em Classificação ([`benchmark_report.md`](benchmarks/benchmark_report.md)), Escalonamento de RAM Tier 256 a Tier 4 ([`memory_benchmark_report.md`](benchmarks/memory_benchmark_report.md)), MNIST Triplo em Visão ([`triple_benchmark_report.md`](benchmarks/triple_benchmark_report.md)), Concept Drift ([`drift_benchmark_report.md`](benchmarks/drift_benchmark_report.md)) e Regressão ([`regression_benchmark_report.md`](benchmarks/regression_benchmark_report.md)). |

---

## 6. Estrutura do Repositório

```text
pessoal_minimal-decision-tree/
├── LICENSE                      # Licença BSD 3-Clause + Avisos de Copyright (River e Scikit-Learn)
├── VERSION                      # Versão canônica do projeto (SemVer 2.0.0)
├── README.md                    # Visão geral, Quick Start e índice de navegação
├── Makefile                     # Root Makefile (compilação, testes e benchmarks)
├── wiki/                        # Páginas da GitHub Wiki (Exemplos, Parâmetros, Memória, Arquitetura)
├── docs/                        # Estudos matemáticos dos papers e plano de implementação (01 a 07)
├── references/                  # Artigos originais (PDFs) e código de referência (River e Scikit-Learn)
├── code/                        # Implementação em C11 puro (include/, src/, tests/ e Makefile)
└── benchmarks/                  # Scripts e runners de validação comparativa (C11 vs River vs Sklearn)
```

---

## 7. Licença (`BSD 3-Clause`)

Distribuído sob a licença permissiva **[BSD 3-Clause License](LICENSE)** (`SPDX-License-Identifier: BSD-3-Clause`), permitindo uso livre acadêmico, pessoal ou comercial (em código-fonte ou firmware binário) e preservando integralmente os avisos de direitos autorais dos projetos de referência **River** ([`references/river_tree/LICENSE`](references/river_tree/LICENSE)) e **Scikit-Learn** ([`references/sklearn_tree/COPYING`](references/sklearn_tree/COPYING)).
