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
- **Aprendizado Federado Embarcado (`v0.5.0`, Zero-Malloc):** Suporte nativo em C11 estático para **Modo 1 (`Fed-Forest`, `ht_fed_forest`)** com micro-florestas compactadas por BFS (`ht_compact_tree`) + votação ponderada (`ht_voting`), e **Modo 2 (`Fed-VFDT`, `ht_fed_vfdt`)** unificando *Coordinated Splitting* (alinhado ao IEEE ICTAI 2025: `QUORUM`, `MAJORITY`, `BEST-MERIT`) e *Exact Sufficient Statistics Merge* (`Stats-Merge`) com framing binário `HFED` + CRC32 e recuperação de *stragglers* (`topology_version` / `split_log` / `FULL_SNAPSHOT`), mantendo $\le 100\text{ KB}$ de SRAM por cliente.
- **Micro-Pipelines de Pré-Processamento Zero-Malloc (`libht_prep`, v0.2.0):** Módulos determinísticos para normalização online Z-Score/Min-Max (`ht_scaler_t`), filtro mediano de Hampel com janela deslizante MAD ($3\sigma$) para rejeição de ruído de sensores industriais (`ht_filter_hampel_t`), codificador Space-Saving (`ht_encoder_freq_t`) e quantizador Ben-Haim & Tom-Tov (`ht_quantizer_t`), integráveis transparentemente via `ht_tree_set_prep_pipeline`.
- **Observadores Numéricos Paramétricos e Não-Paramétricos:** Welford Gaussiano (`HT_OBS_GAUSSIAN`) e Histograma Dinâmico de Ben-Haim & Tom-Tov (`HT_OBS_HISTOGRAM`) para distribuições multimodais.
- **Normalização SGD Híbrida nas Folhas:** Padronização online com fator de credibilidade $\alpha = \frac{N_{\text{slots}}}{N_{\text{slots}} + 10.0}$ para estabilização dos regressores lineares em variáveis multiescala com 0.0 KB de heap.
- **Atributos Mistos e Valores Ausentes (`NaN`):** Suporte nativo a colunas numéricas (`HT_FEAT_NUMERIC`) e categóricas (`HT_FEAT_CATEGORICAL`), roteando leituras ausentes (`NAN` ou categoria `< 0`) pelo filho mais populoso com custo zero de RAM.
- **Reciclagem Determinística (*Slab/Pool Allocators*):** Indexação 100% baseada em `uint16_t` (sem ponteiros) e desativação automática de folhas menos promissoras (*Promise Deactivation*) quando o pool atinge o limite.
- **Serialização Binária Direta (`ht_tree_save` / `ht_tree_load`):** Exporta checkpoints completos de treino ou imagens enxutas somente-inferência ($20\text{ B}$ por nó em `float32`) prontas para Flash/EEPROM.
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

# Executa os benchmarks e testes de Aprendizado Federado (in-process, 8 datasets e TCP POSIX multi-processo)
make bench-federated
make bench-main-federated
make bench-fed-net

# Executa a suíte de emulação ESP32 e ARM Cortex-M3 (QEMU)
make bench-esp32
bash emulation/cortex-m3/run_qemu.sh bench
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

## 3. Dimensionamento de Memória e Suítes de Benchmark

A biblioteca projeta o consumo de memória com determinismo estrito em tempo de compilação ($\le 1\text{ MB}$ de RAM, zero `malloc`/`free`) e é avaliada empiricamente contra duas referências consolidadas da literatura: o **[`online-ml/river`](https://github.com/online-ml/river)** (padrão de referência para streaming contínuo em Python/Cython) e o **[`scikit-learn`](https://scikit-learn.org)** (baseline em lote / *batch* CART).

Esta seção apresenta em **3.1** o dimensionamento físico estático de memória (RAM e Flash) por perfil de microcontrolador e, em **3.2**, o catálogo das suítes de benchmark com direcionamento direto para seus scripts executáveis, relatórios Markdown detalhados e comandos de reprodução.

### 3.1 Resumo de Consumo de Memória (RAM & Flash)

O footprint de RAM de `ht_tree_t` é dimensionado estaticamente em tempo de compilação via flags `-D` (veja todos os detalhes em **[Referência de Parâmetros e Macros](wiki/Referencia-de-Parametros-e-Macros.md)** e **[Orçamento de Memória e Cross-Compilation](wiki/Orcamento-de-Memoria-e-Cross-Compilation.md)**):

| Perfil de Hardware Alvo | Flags de Compilação (`CFLAGS_EXTRA`) | Tamanho do Nó | Folha Ativa | RAM Total (`ht_tree_t`) | Flash (`.text` em `-Os`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Somente Inferência** | `-DHT_MAX_NODES=256 -DHT_MAX_ACTIVE_LEAVES=1 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2` | $20\text{ B}$ | $688\text{ B}$ | **$\approx 9,9\text{ KB}$** | **$\approx 6\text{–}8\text{ KB}$** |
| **MCU Ultra-Fraco (32–64 KB)** | `-DHT_MAX_NODES=128 -DHT_MAX_ACTIVE_LEAVES=16 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2 -DHT_MAX_CATEGORIES=4 -DHT_REG_NUM_SLOTS=4` | $20\text{ B}$ | $560\text{ B}$ | **$\approx 13,5\text{ KB}$** | **$\approx 16\text{–}28\text{ KB}$** |
| **MCU Compacto (64–128 KB)** | `-DHT_MAX_NODES=256 -DHT_MAX_ACTIVE_LEAVES=32 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2` | $20\text{ B}$ | $688\text{ B}$ | **$\approx 30,8\text{ KB}$** | **$\approx 16\text{–}28\text{ KB}$** |
| **MCU Intermediário (256–512 KB)** | `-DHT_MAX_NODES=1024 -DHT_MAX_ACTIVE_LEAVES=64 -DHT_MAX_FEATURES=12 -DHT_MAX_CLASSES=4` | $20\text{ B}$ | $1.024\text{ B}$ | **$\approx 100,3\text{ KB}$** | **$\approx 16\text{–}28\text{ KB}$** |
| **Padrão do Repositório ($\le 1\text{ MB}$)** | *(Padrão: `NODES=4096`, `LEAVES=256`, `FEATURES=16`, `CLASSES=4`)* | $20\text{ B}$ | $1.360\text{ B}$ | **$\approx 484,4\text{ KB}$** *($468,2\text{ KB}$ c/ `-DHT_CLASSIF_ONLY`; até $\approx 6,7\text{ KB}$ no SEA Tier 8 c/ Dynamic Stride)* | **$\approx 22\text{–}28\text{ KB}$** |

---

### 3.2 Catálogo e Direcionamento das Suítes de Benchmark

Para evitar dados defasados no `README.md`, os resultados detalhados de acurácia, latência sub-microssegundo, vazão (*throughput*), topologia de nós e consumo real de recursos (Peak RSS e Heap) são mantidos diretamente nos relatórios gerados e arquivos de cada suíte especializada:

| Suíte de Validação | Diretório / Guia | Relatórios Gerados | Scripts Executores / Ferramentas | Como Reproduzir (`make`) |
| :--- | :--- | :--- | :--- | :--- |
| **Main Benchmark Suite** *(8 Datasets Reais e Sintéticos)* | [`main_benchmark/`](main_benchmark/README.md) | • [`main_benchmark_report.md`](main_benchmark/main_benchmark_report.md)<br/>• [`hyperparam_report.md`](main_benchmark/hyperparam_report.md) | • `main_benchmark.py`<br/>• `prepare_datasets.py`<br/>• `tune_hyperparams.py` | `make bench-main`<br/>`make bench-tune` |
| **Benchmark Federado Multi-Cliente** *(8 Datasets Non-IID & IID)* | [`main_benchmark/`](main_benchmark/README.md) | • [`federated_benchmark_report.md`](main_benchmark/federated_benchmark_report.md) | • `federated_benchmark.py`<br/>• `fed_c_runner.c` | `make bench-main-federated` |
| **Federado Sintético & TCP POSIX** *(Multi-Processo + Stragglers)* | [`benchmarks/`](benchmarks/README.md) | • [`docs/13_federated_learning.md`](docs/13_federated_learning.md) | • `fed_simulator.c`<br/>• `fed_net/fed_net_{server,client}.c`<br/>• `fed_net/run_fed_net_tests.py` | `make bench-federated`<br/>`make bench-fed-net` |
| **Paridade Algorítmica e Regressão** | [`benchmarks/`](benchmarks/README.md) | • [`benchmark_report.md`](benchmarks/benchmark_report.md)<br/>• [`regression_benchmark_report.md`](benchmarks/regression_benchmark_report.md) | • `compare_river.py`<br/>• `bench_c_runner.c`<br/>• `generate_streams.py` | `make bench`<br/>`make bench-regression` |
| **Concept Drift (`ADWIN2` / `HAT`)** | [`benchmarks/`](benchmarks/README.md) | • [`drift_benchmark_report.md`](benchmarks/drift_benchmark_report.md) | • `compare_river.py`<br/>• `generate_streams.py` | `make bench-drift` |
| **Escalonamento e Extrema Restrição de RAM** *(Tier 256 a Tier 4)* | [`benchmarks/`](benchmarks/README.md) | • [`memory_benchmark_report.md`](benchmarks/memory_benchmark_report.md) | • `compare_river.py`<br/>• `bench_c_runner.c` | `make bench-memory` |
| **Micro-Pipelines de Pré-Processamento** *(`libht_prep`)* | [`benchmarks/`](benchmarks/README.md) | • [`prep_benchmark_report.md`](benchmarks/prep_benchmark_report.md) | • `compare_prep.py` | `make bench-prep` |
| **Limites em Visão Computacional** *(MNIST Triplo)* | [`benchmarks/`](benchmarks/README.md) | • [`triple_benchmark_report.md`](benchmarks/triple_benchmark_report.md) | • `compare_triple.py`<br/>• `generate_mnist.py` | `make bench-triple` |
| **Emulação Hardware-in-the-Loop em ESP32 & ARM Cortex-M3** *(QEMU)* | [`emulation/esp32/`](emulation/esp32/esp32_emulation_report.md)<br/>[`emulation/cortex-m3/`](emulation/cortex-m3/cortex_m3_emulation_report.md) | • [`esp32_emulation_report.md`](emulation/esp32/esp32_emulation_report.md)<br/>• [`cortex_m3_emulation_report.md`](emulation/cortex-m3/cortex_m3_emulation_report.md) | • `stream_runner.py`<br/>• `run_qemu.sh`<br/>• Firmwares ESP-IDF & Bare-Metal | `make bench-esp32`<br/>`bash emulation/cortex-m3/run_qemu.sh bench` |

#### Descrição Detalhada das Suítes:

1. **Main Benchmark Suite ([`main_benchmark/`](main_benchmark/README.md)):**
   - **Escopo:** Avaliação em 8 fluxos de dados (4 de classificação: SEA, Electricity, Forest Covertype, Airlines; e 4 de regressão contínua: Friedman #1, Bikes, Appliances Energy, NYC Taxi) comparando C11 (`float32` e `float64`), River e Scikit-Learn (CART batch).
   - **Métricas:** Acurácia holdout e prequential, Macro F1, MAE/RMSE/$R^2$, latência média e de cauda ($p_{95}, p_{99}$), Throughput (amostras/segundo), contagem de FLOPs e consumo real de memória do SO (**Peak RSS** via `getrusage`).
   - **Onde Consultar:** Consulte o relatório analítico em [`main_benchmark/main_benchmark_report.md`](main_benchmark/main_benchmark_report.md) e o estudo de sensibilidade e curvas de Pareto em [`main_benchmark/hyperparam_report.md`](main_benchmark/hyperparam_report.md).

2. **Suíte Especializada de Paridade e Algoritmos ([`benchmarks/`](benchmarks/README.md)):**
   - **Escopo:** Validação matemática e empírica de componentes individuais contra o River e Scikit-Learn:
     - **Classificação Estacionária:** Paridade estatística exata nos fluxos sintéticos clássicos (SEA, Agrawal, Hyperplane, LED).
     - **Regressão Contínua (`VFDT-R` / `HAT-R`):** Redução de variância e folhas lineares com SGD em superfícies não-lineares (Friedman #1, Planes2D).
     - **Concept Drift:** Monitoramento de taxas de erro com `ADWIN2` e substituição estrutural de subárvores no `HAT`.
     - **Escalonamento em Extrema Restrição de RAM:** Avaliação do mecanismo de Desativação por Promessa (*Promise Deactivation*) comparando Tier 256 (524 KB), Tier 8 (16,6 KB) e Tier 4 (8,4 KB).
     - **Micro-Pipelines (`libht_prep`):** Filtro de Hampel MAD, quantizador Ben-Haim & Tom-Tov e escalonadores Z-Score/Min-Max zero-malloc.
     - **Limites Teóricos no MNIST:** Análise analítica da razão pela qual árvores ortogonais de streaming não são recomendadas para dados com dependência espacial 2D (imagens).
   - **Onde Consultar:** Consulte o catálogo detalhado em [`benchmarks/README.md`](benchmarks/README.md) e os relatórios específicos listados na tabela acima.

3. **Benchmark Federado Multi-Cliente & Transporte Real TCP ([`main_benchmark/`](main_benchmark/federated_benchmark_report.md) e [`benchmarks/fed_net/`](benchmarks/fed_net/)):**
   - **Escopo:** Avaliação dos Modos 1 (`Fed-Forest`) e 2 (`Fed-VFDT` Coordenado & `Stats-Merge`) nos 8 datasets reais ($K=10$ clientes, $R=20$ rodadas, Non-IID `Dirichlet α=0.3` & IID, pico de `77,01 KB` SRAM/cliente no Covertype) e validação de transporte em processos OS separados sobre TCP POSIX com fragmentação e *straggler catch-up* (`5/5` cenários em `0,33 s`).
4. **Emulação Hardware-in-the-Loop em ESP32 & ARM Cortex-M3 ([`emulation/esp32/`](emulation/esp32/esp32_emulation_report.md) e [`emulation/cortex-m3/`](emulation/cortex-m3/cortex_m3_emulation_report.md)):**
   - **Escopo:** Validação em firmware ESP-IDF/FreeRTOS (`qemu-system-xtensa` @ 240 MHz) e bare-metal ARM Cortex-M3 (`qemu-system-arm -M lm3s6965evb` @ 50 MHz, $\le 64\text{ KB}$ SRAM), incluindo compilação cruzada de todos os 11 módulos (`core/`, `ensemble/`, `federated/`) com zero warnings.
   - **Destaques:** Injeção contínua via socket TCP/UART, cronometria em ciclos de clock de hardware (`esp_cpu_get_cycle_count()` / `SysTick`) e comprovação de **0 B de alocação dinâmica no Heap**.

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
- **Escopo Leve de Pré-Processamento (`libht_prep`):** Embora a biblioteca inclua normalização online Z-Score/Min-Max, filtro Hampel MAD, codificador Space-Saving e quantizador Ben-Haim & Tom-Tov anexáveis via `ht_prep_pipeline_t`, ela não oferece transformações matriciais offline pesadas (PCA, *One-Hot Encoding* denso automático ou imputação multivariada).

### 4.3 Inadequação para Dados Não-Tabulares e Alta Cardinalidade
- **Incompatibilidade Prática com Imagens, Áudio e Texto (Dados Não-Tabulares):** Conforme demonstrado no benchmark do MNIST, árvores ortogonais não capturam dependências espaciais 2D, e o Naive Bayes das folhas quebra gravemente com pixels vizinhos fortemente correlacionados.
- **Limite Fixo para Variáveis Categóricas:** A macro `HT_MAX_CATEGORIES` (padrão `5`) impede o tratamento direto sem discretização prévia de variáveis categóricas de alta cardinalidade (ex: CEPs, IDs de dispositivos, hashes — mitigável via `ht_encoder_freq_t`).

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
| **[Exemplos e Receitas da API](wiki/Exemplos-e-Receitas-API.md)** | Receitas completas em C11 para `VFDT`, `HAT` (`ADWIN2`), `VFDT-R` / `HAT-R`, `NaN`, `ht_tree_save` / `ht_tree_load` e Aprendizado Federado. |
| **[Aprendizado Federado (`Fed-Forest` & `Fed-VFDT`)](wiki/Aprendizado-Federado.md)** | Guia completo da `v0.5.0`: Modo 1 (`Fed-Forest`), Modo 2 (`Fed-VFDT` Coordenado & `Stats-Merge`), protocolo binário `HFED` + CRC32 e resultados Non-IID. |
| **[Referência de Parâmetros e Macros](wiki/Referencia-de-Parametros-e-Macros.md)** | Tabelas completas de `ht_config_params_t`, `ht_adwin_config_t`, macros de `ht_config.h`, fórmulas em bytes e perfis de MCU. |
| **[Orçamento de Memória e Cross-Compilation](wiki/Orcamento-de-Memoria-e-Cross-Compilation.md)** | Consumo de Flash/ROM (`.text` em `arm-none-eabi-gcc` e ESP-IDF), peso de arquivos `.bin` serializados e flags de compilação cruzada. |
| **[Arquitetura e Fluxogramas](wiki/Arquitetura-e-Fluxogramas.md)** | Comparativo Hoeffding vs CART, mapa modular (`core/`, `ensemble/`, `federated/`) e fluxogramas Mermaid passo a passo de todos os algoritmos. |
| **[Origem: River, Sklearn e Originalidade](wiki/Origem-River-Scikit-Learn-e-Originalidade.md)** | Genealogia técnica e matriz tripla: o que veio do River, do Scikit-Learn, do *Fed-VFDT* (ICTAI 2025) e inovações originais C11. |
| **[Versionamento, Releases e Roadmap](wiki/Versionamento-Releases-e-Roadmap.md)** | Política SemVer 2.0.0 (`VERSION` / `ht_version()`), histórico até a **`v0.5.0`** e **Roadmap (`v0.6.0+`)** (Ponto Fixo `-DHT_USE_FIXED_POINT`, Micro-Ensembles `< 100 KB`, Welford-Adaptive SGD). |
| **Main & Federated Benchmarks ([`main_benchmark/`](main_benchmark/README.md))** | Suíte principal nos 8 datasets ([`main_benchmark_report.md`](main_benchmark/main_benchmark_report.md)), tuning ([`hyperparam_report.md`](main_benchmark/hyperparam_report.md)) e benchmark federado multi-cliente ([`federated_benchmark_report.md`](main_benchmark/federated_benchmark_report.md)). |
| **Emulação ESP32 & Cortex-M3 ([`emulation/`](emulation/esp32/esp32_emulation_report.md))** | Emulação Hardware-in-the-Loop em ESP32 Xtensa ([`esp32_emulation_report.md`](emulation/esp32/esp32_emulation_report.md)) e ARM Cortex-M3 bare-metal ([`cortex_m3_emulation_report.md`](emulation/cortex-m3/cortex_m3_emulation_report.md)). |
| **Estudos Matemáticos ([`docs/`](docs/))** | VFDT ([`01`](docs/01_hoeffding_tree_vfdt.md)), Observadores ([`02`](docs/02_attribute_observers.md)), HAT/ADWIN ([`03`](docs/03_concept_drift_hat.md)), River ([`04`](docs/04_river_mapping.md)), Sklearn ([`05`](docs/05_sklearn_architecture.md)), Plano ([`06`](docs/06_implementation_plan.md)), Regressão ([`07`](docs/07_regression_trees_fimtdd.md)), Pré-Proc ([`08`](docs/08_data_preprocessing_and_nonparametric.md)), RAM ([`09`](docs/09_extreme_ram_optimization.md)), Ponto Fixo ([`10`](docs/10_fixed_point_arithmetic.md)), Micro-Ensembles ([`11`](docs/11_embedded_micro_ensembles.md)), Welford-SGD ([`12`](docs/12_welford_adaptive_sgd.md)) e Aprendizado Federado ([`13`](docs/13_federated_learning.md)). |
| **Relatórios de Benchmark ([`benchmarks/`](benchmarks/README.md))** | Paridade vs River ([`benchmark_report.md`](benchmarks/benchmark_report.md)), Escalonamento de RAM ([`memory_benchmark_report.md`](benchmarks/memory_benchmark_report.md)), MNIST Triplo ([`triple_benchmark_report.md`](benchmarks/triple_benchmark_report.md)), Concept Drift ([`drift_benchmark_report.md`](benchmarks/drift_benchmark_report.md)), Regressão ([`regression_benchmark_report.md`](benchmarks/regression_benchmark_report.md)), Pré-Proc ([`prep_benchmark_report.md`](benchmarks/prep_benchmark_report.md)), Simulador Federado (`fed_simulator.c`) e Transporte TCP (`fed_net/`). |

> [!NOTE]
> **Planos Futuros e Roadmap (`v0.6.0+`):** Com a entrega da Fase 11 (**Aprendizado Federado `Fed-Forest` & `Fed-VFDT` na `v0.5.0`**), as próximas prioridades estratégicas para consolidar o **kansho** em edge/IoT — **(1) Aritmética de Ponto Fixo (`Q16.16` via `-DHT_USE_FIXED_POINT`) para MCUs sem FPU**, **(2) Micro-Ensembles de subespaços (`< 100 KB` SRAM)** e **(3) Welford-Adaptive SGD** — estão documentadas na Wiki em **[`wiki/Versionamento-Releases-e-Roadmap.md` (Seção 5: Possíveis Trabalhos Futuros e Extensões)](wiki/Versionamento-Releases-e-Roadmap.md#5-possíveis-trabalhos-futuros-e-extensões-roadmap)**.

---

## 6. Estrutura do Repositório

```text
pessoal_minimal-decision-tree/
├── LICENSE                      # Licença MPL 2.0 + Avisos de Copyright (River e Scikit-Learn)
├── VERSION                      # Versão canônica do projeto (SemVer 2.0.0 — v0.5.0)
├── README.md                    # Visão geral, Quick Start e índice de navegação
├── Makefile                     # Root Makefile (compilação, testes, emulação e benchmarks)
├── wiki/                        # Páginas da GitHub Wiki (Exemplos, Federado, Parâmetros, Memória, Arquitetura, Roadmap)
├── docs/                        # Estudos matemáticos dos papers, especificações e plano persistente (01 a 13)
├── references/                  # Artigos originais (PDFs) e código de referência (River e Scikit-Learn)
├── code/                        # Implementação C11 pura (include/, src/, tests/ em core/, ensemble/, federated/ e Makefile)
├── emulation/                   # Emulação Hardware-in-the-Loop (esp32/ FreeRTOS e cortex-m3/ Bare-Metal via QEMU)
├── main_benchmark/              # Suíte principal nos 8 datasets (centralizado, tuning e benchmark federado multi-cliente)
└── benchmarks/                  # Suíte de paridade (C11 vs River vs Sklearn), simulador federado e testes TCP multi-processo
```

---

## 7. Licença (`MPL 2.0`)

Distribuído sob a licença de código aberto **[Mozilla Public License 2.0 (MPL 2.0)](LICENSE)** (`SPDX-License-Identifier: MPL-2.0`), permitindo uso livre acadêmico, pessoal ou comercial (inclusive integração a obras maiores e firmwares proprietários, exigindo que modificações diretas nos arquivos cobertos do Kansho permaneçam sob MPL 2.0) e preservando integralmente os avisos de direitos autorais dos projetos de referência **River** ([`references/river_tree/LICENSE`](references/river_tree/LICENSE)) e **Scikit-Learn** ([`references/sklearn_tree/COPYING`](references/sklearn_tree/COPYING)).
