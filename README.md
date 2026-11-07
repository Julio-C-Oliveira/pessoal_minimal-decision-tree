# Minimal Decision Tree (Embedded Hoeffding Tree in C11)

Este repositório disponibiliza uma implementação ultraleve de **Hoeffding Trees (VFDT, HAT, VFDT-R / FIMT-DD e HAT-R)** em **C11 puro**, projetada para rodar **treino incremental (*online/streaming*) e inferência em tempo real** em sistemas embarcados e microcontroladores com forte restrição de memória (**orçamento estrito $\le 1\text{ MB}$ de RAM**, escalável para dezenas de KB, com **zero alocação dinâmica `malloc`/`free`**).

> [!TIP]
> **Fluxogramas Completos da Arquitetura:** Para visualizar o funcionamento interno passo a passo (roteamento iterativo, aprendizado nas folhas, avaliação do *Hoeffding Bound*, reciclagem de memória, detector ADWIN2 e substituição de subárvores no HAT/HAT-R), consulte o documento dedicado [**`arquitetura.md`**](file:///home/julio/Documentos/pessoal_minimal-decision-tree/arquitetura.md).

---

## 1. Possibilidades e Capacidades do Modelo

A biblioteca unifica **4 variantes de árvores de decisão em streaming** dentro da mesma estrutura `ht_tree_t` (sem custo adicional de RAM graças ao uso de `union` anônimas do C11):

| Variante do Modelo | Tarefa de ML | Função de Inicialização de Parâmetros | Adaptação a *Concept Drift* | Critérios de Split Suportados | Estratégias de Predição nas Folhas |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **VFDT** (*Hoeffding Tree Classifier*) | Classificação Multiclasse | `ht_default_params()` | Não (Fluxo estacionário) | *Information Gain* (`HT_CRIT_INFO_GAIN`) ou *Gini Impurity* (`HT_CRIT_GINI`) | **MC** (*Majority Class*), **NB** (*Naive Bayes*) ou **NBA** (*Naive Bayes Adaptive*) |
| **HAT** (*Hoeffding Adaptive Tree*) | Classificação Multiclasse | `ht_default_hat_params()` | **Sim** (ADWIN2 + Subárvores Alternativas + Bootstrap $\text{Poisson}(1)$) | *Information Gain* ou *Gini Impurity* | **MC**, **NB** ou **NBA** + Votação ponderada $1/\hat{e}_l^2$ entre subárvores |
| **VFDT-R** (*FIMT-DD Regressor*) | Regressão Contínua | `ht_default_reg_params()` | Não (Fluxo estacionário) | *Variance Reduction* (`HT_CRIT_VARIANCE_REDUCTION`) com correção de Bessel (`ddof=1`) | **MEAN** (*Média*), **MODEL** (*Regressão Linear Online via SGD*) ou **ADAPTIVE** (*Faded MSE*) |
| **HAT-R** (*Hoeffding Adaptive Regressor*) | Regressão Contínua | `ht_default_hat_reg_params()` | **Sim** (ADWIN2 c/ Erro Absoluto + Teste Z Bilateral de Welch) | *Variance Reduction* (`ddof=1`) | **MEAN**, **MODEL** (SGD) ou **ADAPTIVE** + Média entre subárvores ativas |

### Principais Recursos Nativos:
1. **Aprendizado Incremental Prequential (*Test-Then-Train*):** Aprende amostra por amostra (`ht_tree_learn_one` / `ht_tree_learn_one_reg`) em tempo $O(\text{profundidade} \times F)$, sem nunca reter o dataset na memória.
2. **Atributos Mistos (Numéricos e Categóricos):** Cada coluna do vetor de entrada `ht_value_t x[]` pode ser configurada individualmente como contínua (`HT_FEAT_NUMERIC`) ou nominal discreta (`HT_FEAT_CATEGORICAL`).
3. **Tolerância Nativa a Valores Ausentes (`NaN` / Categoria `< 0`):**
   - Na inferência e roteamento: se o atributo testado em um nó interno for `NAN` (numérico) ou `< 0` (categórico), a amostra segue automaticamente pelo **filho mais populoso** (`ht_most_common_child`).
   - No treino: os observadores dos demais atributos válidos continuam aprendendo normalmente, ignorando apenas a feature ausente.
4. **Gerenciamento Automático de Memória (*Promise Deactivation*):** Quando o pool de folhas ativas (`HT_MAX_ACTIVE_LEAVES`) enche, a árvore desativa automaticamente a folha menos promissora (liberando seus $1.360\text{ B}$ de observadores de volta ao pool), mas mantendo a folha capaz de predizer via *Majority Class* / *Target Mean*.
5. **Serialização Binária Direta (`ht_tree_save` / `ht_tree_load`):**
   - **Modo Checkpoint Completo (`include_training_stats = true`):** Salva toda a árvore e seus observadores para retomar o treinamento exatamente de onde parou.
   - **Modo Somente-Inferência (`include_training_stats = false`):** Descarta os blocos pesados de aprendizado (`ht_leaf_stats_t`), gerando uma imagem binária ultracompacta (apenas $28\text{ B}$ por nó em `float32`) ideal para gravação em Flash/EEPROM.
6. **Precisão Ponto Flutuante Configurável (`ht_float_t`):** Compila por padrão em `float` (32 bits) para máxima eficiência em microcontroladores com FPU (ARM Cortex-M4F/M7, ESP32, RP2350), ou em `double` (64 bits) passando `-DHT_USE_DOUBLE`.

---

## 2. Como Compilar e Usar o Modelo

### 2.1 Compilação da Biblioteca e Suíte de Testes

O projeto utiliza apenas a biblioteca padrão C11 e `<math.h>` (`-lm`):

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

Para integrar em seu próprio projeto C/C++ ou firmware embarcado, basta incluir [`code/include/`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/code/include) no caminho de headers (`-Icode/include`) e linkar com `code/build/libht.a -lm` (ou compilar diretamente os 5 arquivos `.c` de `code/src/`).

---

### 2.2 Exemplo 1: Classificação Incremental Padrão (VFDT)

```c
#include <stdio.h>
#include "ht_tree.h"

/* Aloca a árvore estaticamente (ex: em seção .bss fora da stack de microcontroladores) */
static ht_tree_t tree;

int main(void) {
    /* 1. Configura parâmetros padrão para 4 atributos e 3 classes */
    ht_config_params_t params;
    ht_default_params(&params, /*n_features=*/4, /*n_classes=*/3);

    /* Opcional: customizar hiperparâmetros */
    params.criterion       = HT_CRIT_INFO_GAIN; /* ou HT_CRIT_GINI */
    params.leaf_prediction = HT_LEAF_NBA;       /* MC, NB ou NBA (default) */
    params.grace_period    = 100;               /* Avalia split a cada 100 amostras na folha */
    params.delta           = 1e-6f;             /* Confiança do Hoeffding Bound (1 - delta) */

    /* 2. Inicializa a árvore */
    if (ht_tree_init(&tree, &params) != HT_OK) {
        return -1;
    }

    /* 3. Loop Prequential (Testa primeiro, treina depois) em streaming */
    ht_value_t x[HT_MAX_FEATURES] = {0};
    x[0].num = 5.1f;
    x[1].num = 3.5f;
    x[2].num = 1.4f;
    x[3].num = 0.2f;
    uint8_t y_true = 0;

    /* Predição de probabilidades por classe e classe vencedora */
    ht_float_t proba[HT_MAX_CLASSES];
    ht_tree_predict_proba(&tree, x, proba);
    uint8_t y_pred = ht_tree_predict(&tree, x);

    printf("Predicao: classe %u (P = %.2f%%)\n", y_pred, (double)(proba[y_pred] * 100.0f));

    /* Treinamento incremental com peso = 1.0 */
    ht_tree_learn_one(&tree, x, y_true, 1.0f);

    return 0;
}
```

---

### 2.3 Exemplo 2: Classificação Adaptativa a *Concept Drift* (HAT + ADWIN2)

Para fluxos não-estacionários onde a distribuição dos dados muda ao longo do tempo, ative o **HAT** fornecendo um pool estático de detectores ADWIN (`ht_adwin_pool_t`):

```c
#include "ht_tree.h"

static ht_tree_t       hat_tree;
static ht_adwin_pool_t adwin_pool;

void setup_and_run_hat(void) {
    /* 1. Inicializa o pool estático de detectores ADWIN */
    ht_adwin_pool_init(&adwin_pool);

    /* 2. Carrega os parâmetros padrão do HAT (ativa enable_hat e bootstrap_sampling) */
    ht_config_params_t params;
    ht_default_hat_params(&params, /*n_features=*/8, /*n_classes=*/2);
    params.adwin_delta            = 0.002f; /* Sensibilidade do detector ADWIN */
    params.drift_window_threshold = 300;    /* Amostras mínimas antes de trocar subárvore */

    /* 3. Inicializa a árvore conectada ao pool ADWIN */
    ht_tree_init_hat(&hat_tree, &params, &adwin_pool);

    /* O uso de ht_tree_predict() e ht_tree_learn_one() é exatamente o mesmo! */
    /* A detecção de drift, criação de subárvores alternativas e substituição (Switch/Prune)
       ocorrem de forma 100% automática dentro de ht_tree_learn_one(). */
}
```

---

### 2.4 Exemplo 3: Regressão Contínua Incremental (VFDT-R e HAT-R)

Para prever alvos contínuos ($y \in \mathbb{R}$) usando *Variance Reduction* e modelos lineares com SGD nas folhas:

```c
#include <stdio.h>
#include "ht_tree.h"

static ht_tree_t       reg_tree;
static ht_adwin_pool_t reg_adwin_pool;

void run_regression_example(void) {
    ht_config_params_t params;

    /* Para VFDT-R estacionária use: ht_default_reg_params(&params, 6);
       Para HAT-R adaptativa a drift use: ht_default_hat_reg_params(&params, 6); */
    ht_default_hat_reg_params(&params, /*n_features=*/6);

    params.leaf_prediction      = HT_LEAF_ADAPTIVE; /* Alterna entre Média e Modelo Linear SGD */
    params.learning_rate        = 0.01f;            /* Taxa de aprendizado inicial do SGD */
    params.l2_regularization    = 1e-4f;            /* Regularização L2 nos pesos lineares */
    params.model_selector_decay = 0.95f;            /* Fator de esquecimento do Faded MSE */

    ht_adwin_pool_init(&reg_adwin_pool);
    ht_tree_init_hat(&reg_tree, &params, &reg_adwin_pool);

    ht_value_t x[HT_MAX_FEATURES] = {0};
    x[0].num = 0.75f;
    x[1].num = -1.20f;
    ht_float_t y_target = 42.5f;

    /* Inferência contínua */
    ht_float_t y_hat = ht_tree_predict_reg(&reg_tree, x);

    /* Passo de aprendizado incremental de regressão */
    ht_tree_learn_one_reg(&reg_tree, x, y_target, 1.0f);
    (void)y_hat;
}
```

---

### 2.5 Exemplo 4: Atributos Mistos (Numéricos + Categóricos) e Valores Ausentes (`NaN`)

```c
#include <math.h>
#include "ht_tree.h"

void configure_mixed_features(ht_tree_t *tree) {
    ht_config_params_t params;
    ht_default_params(&params, /*n_features=*/3, /*n_classes=*/2);

    /* Define o schema por coluna: */
    params.feature_types[0] = HT_FEAT_NUMERIC;     /* Coluna 0: contínua (float) */
    params.feature_types[1] = HT_FEAT_CATEGORICAL; /* Coluna 1: nominal (0 .. HT_MAX_CATEGORIES-1) */
    params.feature_types[2] = HT_FEAT_NUMERIC;     /* Coluna 2: contínua (float) */

    ht_tree_init(tree, &params);

    /* Amostra com valor numérico ausente (NAN) na coluna 2 e categoria 3 na coluna 1 */
    ht_value_t x[HT_MAX_FEATURES] = {0};
    x[0].num = 12.3f;
    x[1].cat = 3;      /* Para indicar categoria ausente, passe -1 */
    x[2].num = NAN;    /* Tratado nativamente sem erro e sem custo extra de RAM */

    ht_tree_learn_one(tree, x, /*y=*/1, /*weight=*/1.0f);
}
```

---

### 2.6 Exemplo 5: Serialização Binária (`ht_tree_save` e `ht_tree_load`)

```c
#include <stddef.h>
#include <stdint.h>
#include "ht_tree.h"

static uint8_t flash_buffer[65536];

void save_and_restore_model(const ht_tree_t *trained_tree, ht_tree_t *loaded_tree) {
    size_t bytes_needed = 0;

    /* 1. Consulta o tamanho necessário (passando buffer = NULL)
       include_training_stats = false gera imagem enxuta somente-inferência! */
    ht_tree_save(trained_tree, NULL, 0, &bytes_needed, /*include_training_stats=*/false);

    /* 2. Serializa para o buffer se couber */
    if (bytes_needed <= sizeof(flash_buffer)) {
        size_t bytes_written = 0;
        ht_tree_save(trained_tree, flash_buffer, sizeof(flash_buffer), &bytes_written, false);

        /* 3. Carrega a árvore em qualquer instância a partir do buffer binário */
        ht_tree_load(loaded_tree, flash_buffer, bytes_written);
    }
}
```

---

## 3. Referência Completa de Parâmetros

### 3.1 Parâmetros de Configuração em Runtime (`ht_config_params_t`)

Todos os campos abaixo pertencem à struct [`ht_config_params_t`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/code/include/ht_tree.h#L44-L68) e são inicializados automaticamente pelas funções `ht_default_params()`, `ht_default_hat_params()`, `ht_default_reg_params()` ou `ht_default_hat_reg_params()`:

| Parâmetro (`ht_config_params_t`) | Tipo | Padrão (VFDT / HAT) | Padrão (VFDT-R / HAT-R) | Descrição e Efeito no Modelo |
| :--- | :--- | :---: | :---: | :--- |
| `task_type` | `ht_task_type_t` | `HT_TASK_CLASSIFICATION` | `HT_TASK_REGRESSION` | Define se a árvore opera em modo de Classificação Multiclasse ou Regressão Contínua. |
| `n_features` | `uint8_t` | *(Definido pelo usuário)* | *(Definido pelo usuário)* | Número de atributos de entrada no fluxo ($1 \dots \text{HT\_MAX\_FEATURES}$). |
| `n_classes` | `uint8_t` | *(Definido pelo usuário)* | `1` | Número de classes alvo na classificação ($2 \dots \text{HT\_MAX\_CLASSES}$). Ignorado na regressão. |
| `feature_types[i]` | `ht_feature_type_t` | `HT_FEAT_NUMERIC` | `HT_FEAT_NUMERIC` | Tipo de cada coluna $i$: `HT_FEAT_NUMERIC` (contínuo) ou `HT_FEAT_CATEGORICAL` (inteiro $0 \dots \text{HT\_MAX\_CATEGORIES}-1$). |
| `criterion` | `ht_criterion_type_t` | `HT_CRIT_INFO_GAIN` | `HT_CRIT_VARIANCE_REDUCTION` | Critério de impureza: `HT_CRIT_INFO_GAIN` (Entropia base 2), `HT_CRIT_GINI` (Impureza de Gini) ou `HT_CRIT_VARIANCE_REDUCTION` (Redução de Variância com `ddof=1`). |
| `leaf_prediction` | `ht_leaf_pred_t` | `HT_LEAF_NBA` | `HT_LEAF_ADAPTIVE` | **Classificação:** `HT_LEAF_MC` (Classe Majoritária), `HT_LEAF_NB` (Naive Bayes), `HT_LEAF_NBA` (Adaptativo entre MC e NB).<br/>**Regressão:** `HT_LEAF_MEAN` (Média), `HT_LEAF_MODEL` (Linear SGD), `HT_LEAF_ADAPTIVE` (Alterna entre Média e SGD via fMSE). |
| `grace_period` | `uint16_t` | `200` | `200` | Peso acumulado de amostras ($n_{\min}$) que uma folha ativa aguarda entre avaliações consecutivas de candidatos de split. Valores menores aceleram o crescimento da árvore ao custo de mais CPU. |
| `max_depth` | `uint16_t` | `64` | `64` | Profundidade máxima permitida para a árvore (raiz = `0`). Ao atingir `max_depth`, a folha é automaticamente desativada. |
| `delta` ($\delta$) | `ht_float_t` | `1e-7` | `1e-7` | Nível de significância do *Hoeffding Bound* ($1 - \delta$ é a confiança estatística do split). Deve estar em $(0, 1)$. Valores menores exigem mais evidência antes de dividir. |
| `tau` ($\tau$) | `ht_float_t` | `0.05` | `0.05` | Limiar de desempate (*tie-breaking*). Se dois atributos têm mérito muito parecido e o limite de Hoeffding cai abaixo de $\tau$ ($\epsilon < \tau$), o split é forçado no melhor candidato atual. |
| `min_branch_fraction` | `ht_float_t` | `0.01` (`1%`) | `0.01` (`1%`) | Fração mínima do peso total da folha que cada ramo filho (esquerdo e direito) deve receber para que um candidato de split seja considerado válido ($0 < f \le 0.5$). |
| `max_share_to_split` | `ht_float_t` | `0.99` (`99%`) | `0.99` | *(Classificação)* Se a classe majoritária na folha já representa mais que essa fração do peso total, a folha é considerada suficientemente pura e não tenta fazer split. |
| `nb_threshold` | `ht_float_t` | `0.0` | `0.0` | *(Classificação c/ `HT_LEAF_NB`)* Peso mínimo acumulado na folha antes de habilitar o uso do Naive Bayes (evita instabilidade com pouquíssimas amostras). |
| `merit_preprune` | `bool` | `true` | `true` | Quando `true`, inclui o candidato de "não dividir" (*null split*) na competição do Hoeffding Bound, evitando splits com ganho irrelevante. |
| `enable_hat` | `bool` | `false` (VFDT) / `true` (HAT) | `false` (VFDT-R) / `true` (HAT-R) | Ativa o monitoramento de *Concept Drift* via detectores ADWIN2 e cultivo de subárvores alternativas em cada nó interno. |
| `bootstrap_sampling` | `bool` | `false` (VFDT) / `true` (HAT) | `false` (VFDT-R) / `true` (HAT-R) | Quando `enable_hat = true`, multiplica o peso de cada amostra recebida por $k \sim \text{Poisson}(1)$, aumentando a diversidade estatística durante transições de drift. |
| `drift_window_threshold` | `uint16_t` | `300` | `300` | *(HAT / HAT-R)* Número mínimo de amostras observadas tanto no ramo principal quanto na subárvore alternativa antes de avaliar a substituição (`SWITCH`) ou poda (`PRUNE`). |
| `switch_significance` ($\alpha$) | `ht_float_t` | `0.05` | `0.05` | *(HAT / HAT-R)* Nível de significância estatística para substituir um ramo principal por sua subárvore alternativa (ou podá-la). |
| `adwin_delta` | `ht_float_t` | `0.002` | `0.002` | *(HAT / HAT-R)* Parâmetro de confiança $\delta_{\text{ADWIN}}$ dos detectores de drift alocados para os nós da árvore. |
| `learning_rate` ($\eta_0$) | `ht_float_t` | `0.01` | `0.01` | *(Regressão `MODEL` / `ADAPTIVE`)* Taxa de aprendizado base do modelo linear treinado via *Stochastic Gradient Descent* (SGD) nas folhas. |
| `l2_regularization` ($\lambda$) | `ht_float_t` | `1e-4` | `1e-4` | *(Regressão `MODEL` / `ADAPTIVE`)* Penalidade de regularização L2 e fator de decaimento da taxa de aprendizado ($\eta_t = \eta_0 / (1 + \lambda n)$). |
| `model_selector_decay` ($\alpha$) | `ht_float_t` | `0.95` | `0.95` | *(Regressão `ADAPTIVE`)* Fator de esquecimento exponencial do erro quadrático médio ponderado (*Faded MSE*: $\text{fmse}_t = \alpha \cdot \text{fmse}_{t-1} + e_t^2$). |
| `min_samples_split` | `ht_float_t` | `5.0` | `5.0` | *(Regressão)* Peso mínimo de amostras exigido em cada ramo filho (esquerdo e direito) para validar um candidato de split de regressão. |

---

### 3.2 Parâmetros do Detector ADWIN (`ht_adwin_config_t`)

O detector [`ht_adwin_t`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/code/include/ht_drift.h#L58-L75) também pode ser usado de forma independente (standalone) para monitorar qualquer fluxo de dados numéricos ou taxas de erro:

| Parâmetro (`ht_adwin_config_t`) | Tipo | Valor Padrão | Significado |
| :--- | :--- | :---: | :--- |
| `delta` | `ht_float_t` | `0.002` | Nível de significância para o teste de corte de janela $\epsilon_{\text{cut}}$. Valores maiores tornam o detector mais sensível a pequenas mudanças. |
| `clock` | `uint16_t` | `32` | Intervalo de passos entre verificações de corte da janela (reduz custo computacional amortizado para $O(1)$). |
| `min_window_length` | `uint16_t` | `5` | Tamanho mínimo de cada subjanela ($W_0$ e $W_1$) exigido ao testar uma hipótese de bipartição. |
| `grace_period` | `uint16_t` | `10` | Número mínimo de elementos totais na janela antes de habilitar os testes de detecção de drift. |
| `max_buckets` ($M$) | `uint8_t` | `5` | Número máximo de buckets de capacidade $2^k$ por nível antes de comprimir os 2 mais antigos em um bucket de capacidade $2^{k+1}$ no nível seguinte. |

---

### 3.3 Macros de Compilação para Dimensionamento de RAM (`ht_config.h`)

Você pode ajustar o consumo estático de RAM para caber exatamente no seu microcontrolador passando flags `-D` no compilador (ex: `CFLAGS_EXTRA="-DHT_MAX_NODES=1024 -DHT_MAX_ACTIVE_LEAVES=64"`):

| Macro de Compilação | Valor Padrão | Descrição e Impacto em Memória |
| :--- | :---: | :--- |
| `HT_USE_DOUBLE` | *(Indefinido)* | Se definido (`-DHT_USE_DOUBLE`), altera `ht_float_t` de `float` (32-bit) para `double` (64-bit). |
| `HT_MAX_FEATURES` | `16` | Número máximo de atributos suportados por amostra. Dimensiona o array `observers[HT_MAX_FEATURES]` dentro de cada folha ativa. |
| `HT_MAX_CLASSES` | `4` | Número máximo de classes na classificação. Dimensiona `class_counts[]` nos nós e o número de Gaussianas por atributo numérico. |
| `HT_MAX_CATEGORIES` | `5` | Cardinalidade máxima suportada para atributos categóricos nominais ($0 \dots \text{HT\_MAX\_CATEGORIES}-1$). |
| `HT_MAX_NODES` | `4096` | Capacidade total do `node_pool` (soma de nós internos + folhas ativas + folhas inativas). Cada `ht_node_t` custa apenas $28\text{ B}$ em `float32`. |
| `HT_MAX_ACTIVE_LEAVES` | `256` | Capacidade do `stats_pool` (número máximo de folhas ativas aprendendo simultaneamente). É o principal fator de consumo de RAM ($1.360\text{ B}$ por folha em `float32`). |
| `HT_GAUSSIAN_SPLITS` | `10` | Quantidade de thresholds candidatos avaliados entre `[min, max]` no observador Gaussiano de classificação. |
| `HT_REG_NUM_SLOTS` | `5` | Quantidade de slots ordenados de quantização por atributo numérico na regressão (mantém paridade de $80\text{ B}$ com o observador de classificação). |
| `HT_MAX_ADWIN_DETECTORS` | `128` | Quantidade de detectores ADWIN no `ht_adwin_pool_t` ($764\text{ B}$ por detector em `float32`). |
| `HT_ADWIN_LEVELS` | `14` | Quantidade de níveis logarítmicos do histograma exponencial ADWIN (suporta janelas de até $5 \times (2^{14}-1) \approx 81.915$ amostras). |

---

## 4. Por que Hoeffding Tree no lugar do CART?

Algoritmos batch tradicionais como o **CART** exigem que todo o dataset esteja carregado em RAM (ou seja re-escaneado múltiplas vezes) para ordenar atributos e encontrar pontos de corte. Já a **Hoeffding Tree (VFDT)** processa cada amostra $O(1)$ única vez (*stream learning*), mantendo apenas estatísticas suficientes nas folhas e usando a **Desigualdade de Hoeffding** para garantir matematicamente (com probabilidade $1 - \delta$) que o split escolhido com $n$ amostras é o mesmo que seria escolhido com infinitas amostras:

$$\epsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$$

---

## 5. Fontes e Referências no Repositório (`references/`)

Para embasar a matemática e a arquitetura em baixo nível, o repositório armazena cópias locais de referência em [`references/`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references):

1. **River (`references/river_tree/`):**
   - **Origem:** Repositório oficial [`online-ml/river`](https://github.com/online-ml/river) (branch `main`, commit `4768de0579f977315e65cd23ee658534c1d62c26`, licença *BSD-3-Clause*).
   - **Módulos extraídos:** `river/tree/` e `river/drift/`.
   - **Objetivo no projeto:** Referência algorítmica direta para a *Hoeffding Tree Classifier* ([`hoeffding_tree_classifier.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree_classifier.py)), folhas MC/NB/NBA ([`htc_nodes.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py)), aproximação Gaussiana ([`gaussian_splitter.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/gaussian_splitter.py)), critérios de impureza ([`gini.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/gini.py), [`info_gain.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/info_gain.py)), regressão incremental ([`hoeffding_tree_regressor.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree_regressor.py), [`hoeffding_adaptive_tree_regressor.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_adaptive_tree_regressor.py)) e *Concept Drift* ([`hoeffding_adaptive_tree_classifier.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_adaptive_tree_classifier.py), [`adwin.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/drift/adwin.py)).

2. **Scikit-Learn (`references/sklearn_tree/`):**
   - **Origem:** Repositório oficial [`scikit-learn/scikit-learn`](https://github.com/scikit-learn/scikit-learn) (branch `main`, commit `6e37063f7ab7ef7a7e257b9d44bc3ec6ee1b1bcd`, licença *BSD-3-Clause*).
   - **Módulo extraído:** `sklearn/tree/` (implementado em Cython/C).
   - **Objetivo no projeto:** Referência arquitetural de dados contíguos e baixo nível — especialmente o layout de array plano de structs `Node` sem ponteiros ([`_tree.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pxd), [`_utils.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_utils.pxd)) e a separação modular de `Criterion` ([`_criterion.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_criterion.pxd)) e `Splitter` ([`_splitter.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_splitter.pxd)).

3. **Artigos Científicos (`references/papers/`):**
   - PDFs dos artigos fundamentais de Domingos, Hulten, Spencer, Bifet, Gavaldà e Ikonomovska.

---

## 6. Decisões de Arquitetura e Design

1. **Escopo de ML (Classificação e Regressão Unificadas):**
   - **Classificação Multiclasse e Regressão Contínua Incremental** (*on-device training + inference* no estilo *prequential*: testa primeiro, treina depois).
   - **Predição nas Folhas:** Suporte a *Majority Class* (MC), *Naive Bayes* (NB) e *Naive Bayes Adaptive* (NBA) na classificação; e *Target Mean* (MEAN), *Linear Model via SGD* (MODEL) e *Adaptive fMSE* (ADAPTIVE) na regressão.
   - **Critérios de Split:** *Gini Impurity*, *Information Gain (Entropia)* e *Variance Reduction* selecionáveis por configuração.
   - **Topologia:** Splits **estritamente binários** para todos os tipos de atributos:
     - Numéricos: $x_i \le \text{threshold}$ vs $x_i > \text{threshold}$
     - Categóricos: $x_i == \text{categoria}$ vs $x_i \neq \text{categoria}$
     - *Vantagem:* Todos os nós da árvore possuem tamanho fixo (`left` e `right`), eliminando arrays de ponteiros dinâmicos.

2. **API de Entrada (`ht_value_t`) e Schema Fixo:**
   - Em vez de dicionários dinâmicos (como no Python/River), a árvore registra no `init` um schema fixo de tipos por coluna (`HT_FEAT_NUMERIC` ou `HT_FEAT_CATEGORICAL`).
   - Cada amostra é passada como um array contíguo de `union ht_value_t { ht_float_t num; int32_t cat; }`, garantindo zero overhead de parsing e excelente localidade de cache.

3. **Tratamento de Valores Ausentes (`NaN` / Categoria Desconhecida) com Custo Zero de RAM:**
   - **No Roteamento (Inferência/Treino):** Se o atributo de split de um nó interno estiver ausente (`isnan(x)` ou `cat < 0`), a amostra segue para o **filho mais populoso** comparando em tempo real o peso acumulado dos filhos `left` e `right` (zero bytes extras no nó).
   - **No Aprendizado da Folha:** Atualiza normalmente a estatística do alvo na folha e os observadores dos demais atributos válidos, ignorando apenas o observador do atributo ausente.

4. **Representação de Atributos ($O(1)$ por Atributo):**
   - **Numéricos (Classificação):** Aproximação Gaussiana incremental via **Algoritmo de Welford** (`count`, `mean`, `M2`, `min`, `max` por classe), avaliando $k$ pontos candidatos (10 bins entre `min` e `max`) na hora de testar o split.
   - **Numéricos (Regressão):** Quantização dinâmica em **5 slots ordenados** combinados via **Fórmulas Paralelas de Chan** (`ht_var_combine` / `ht_var_subtract`), ocupando exatamente os mesmos $80\text{ B}$ em `float32`.
   - **Categóricos Nominais:** Tabela de frequência ou estatísticas de variância limitada pela cardinalidade máxima configurada (`HT_MAX_CATEGORIES`).

5. **Polimorfismo Híbrido (Inspirado em `sklearn/tree` + Data-Oriented Design):**
   - **Nível da Árvore:** Polimorfismo via *function pointers* (`ht_criterion_vtable_t`) para o critério de impureza, já que há apenas 1 instância por árvore.
   - **Nível dos Nós e Observadores:** Uso direto de `enum` e `union` (sem *vtable* por folha ou atributo), economizando 4 a 8 bytes de ponteiro por observador e permitindo *inlining* agressivo pelo compilador C.

6. **Precisão Numérica Configurável (`ht_float_t`):**
   - Padrão em **`float` (32-bit)** para cortar o consumo de RAM das estatísticas pela metade e aproveitar a FPU nativa de microcontroladores (ARM Cortex-M4F/M7, ESP32, RP2350).
   - Flag de compilação `-DHT_USE_DOUBLE` para alternar para `double` (64-bit) em testes de paridade numérica com o River.

7. **Gerenciamento de Memória Determinístico e Indexação Livre de Ponteiros (`uint16_t`):**
   - **Índices em vez de Ponteiros:** Inspirado no `sklearn/tree/_tree.pxd`, todas as referências entre nós (`left_child`, `right_child`, `stats_idx`) usam índices inteiros (`uint16_t`, com `HT_INVALID_IDX = 0xFFFF`) em arrays contíguos. Isso economiza RAM em arquiteturas 32/64-bit e torna a estrutura **100% relocável para serialização binária direta**.
   - **Separação entre Topologia e Estado de Aprendizado:**
     - `ht_node_t` (**Nó da Árvore - Leve**): Guarda apenas índices dos filhos (`left`, `right`), índice do estado (`stats_idx`), condição de split (`feature_idx`, `threshold`/`cat_val`) e contadores de classe (`class_counts`) ou variância do alvo (`reg_stats`).
     - `ht_leaf_stats_t` (**Estado de Aprendizado da Folha - Pesado**): Guarda os observadores de atributos, acumuladores do NBA / fMSE e pesos lineares do SGD.
   - **Slab / Pool Allocator Estático:** Dois pools de blocos de tamanho fixo (`node_pool` e `stats_pool` com *free-list* em $O(1)$) dentro do buffer de RAM de $\le 1\text{ MB}$:
     - Quando uma folha faz um split, ela vira um nó interno e **devolve imediatamente seu `ht_leaf_stats_t` para o `stats_pool`**.
     - **Desativação de Folhas (VFDT Memory Management):** Quando o `stats_pool` atinge o limite, as folhas menos promissoras são **desativadas** — liberando seu `ht_leaf_stats_t` de volta ao pool, mas mantendo `class_counts` / `reg_stats` no `ht_node_t` para continuar fazendo predições.

---

## 7. Estimativa de Orçamento de Memória (Configuração Padrão)

Com os limites padrão ($F = 16$ atributos, $C = 4$ classes, `4096` nós e `256` folhas ativas):
- **Observador por Atributo (`ht_feature_obs_t`):** Exatamente **$80\text{ B}$** em `float32` ($160\text{ B}$ em `float64`) tanto para Classificação quanto para Regressão.
- **Estado de Aprendizado por Folha Ativa (`ht_leaf_stats_t`):**
  $$16 \times 80\text{ B} + \text{pesos SGD} + \text{metadados NBA/fMSE} = \mathbf{1.360\text{ B por folha ativa}} \text{ (em float32)}$$
- **Nó Estrutural (`ht_node_t` — Internos e Folhas Inativas):**
  $$\text{union alvo } (16\text{ B}) + \text{split } (4\text{ B}) + \text{índices } (3 \times 2\text{ B}) + \text{flags } (2\text{ B}) = \mathbf{28\text{ B por nó}} \text{ (em float32)}$$
- **Consumo Total de `sizeof(ht_tree_t)`:**
  - **Em `float` (32-bit):** $\approx \mathbf{537\text{ KB}}$ (metade do orçamento de 1 MB, suportando $256$ folhas ativas e $4.096$ nós).
  - **Em `double` (64-bit):** $\approx \mathbf{1.000\text{ KB}}$ (validado em tempo de compilação via `_Static_assert(sizeof(ht_tree_t) <= 1024 * 1024)`).

---

## 8. Estrutura do Repositório

```text
pessoal_minimal-decision-tree/
├── README.md                    # Guia de uso, parâmetros, arquitetura e visão geral
├── arquitetura.md               # Fluxogramas completos do algoritmo (Mermaid)
├── Makefile                     # Root Makefile (repassa comandos para code/)
├── references/                  # Materiais de referência (Artigos + Código fonte de base)
│   ├── papers/                  # PDFs dos artigos originais (VFDT, Hoeffding, Bifet, HAT)
│   ├── river_tree/              # Código extraído de online-ml/river (river/tree + river/drift)
│   └── sklearn_tree/            # Código extraído de scikit-learn/scikit-learn (sklearn/tree)
├── docs/                        # Notas de estudo dos papers e mapeamento arquitetural
│   ├── 01_hoeffding_tree_vfdt.md
│   ├── 02_attribute_observers.md
│   ├── 03_concept_drift_hat.md
│   ├── 04_river_mapping.md
│   ├── 05_sklearn_architecture.md
│   ├── 06_implementation_plan.md     # Plano de implementação persistente multi-chat/agente
│   └── 07_regression_trees_fimtdd.md # Estudo teórico de regressão incremental (FIMT-DD / HAT-R)
├── code/                        # Implementação em C11 e testes (organização limpa do código)
│   ├── Makefile                 # Build C11 puro (lib, testes, flags float/double)
│   ├── include/                 # Headers públicos e internos (C11)
│   │   ├── ht_config.h          # Tipos (ht_float_t, ht_value_t), limites e constantes
│   │   ├── ht_memory.h          # Slab/Pool allocator determinístico baseado em índices uint16_t
│   │   ├── ht_observer.h        # Observadores Gaussianos (Welford), Categóricos e Slots de Chan
│   │   ├── ht_criterion.h       # Vtable de Gini, Info Gain, Variance Reduction e Hoeffding Bound
│   │   ├── ht_tree.h            # API unificada (VFDT, HAT, VFDT-R, HAT-R, save/load)
│   │   └── ht_drift.h           # ADWIN2 com histograma exponencial fixo e estatísticas HAT/HAT-R
│   ├── src/                     # Implementação em C11 (zero dependências além de math.h/stdint.h)
│   │   ├── ht_memory.c
│   │   ├── ht_observer.c
│   │   ├── ht_criterion.c
│   │   ├── ht_tree.c
│   │   └── ht_drift.c
│   └── tests/                   # Suíte modular de testes em C (função -> integração -> fluxo total)
│       ├── test_harness.h       # Macros de asserção sem dependências (tolerância adaptativa float/double)
│       ├── test_memory.c        # Testes de unidade, stress e reciclagem do ht_memory
│       ├── test_observer.c      # Testes do Welford Gaussiano, PDF/CDF, Chan e observador categórico
│       ├── test_criterion.c     # Testes de Gini, Info Gain, VR, Hoeffding Bound e busca de split
│       ├── test_tree.c          # Testes de roteamento, folhas MC/NB/NBA e Mean/SGD/Adaptive, save/load
│       ├── test_drift.c         # Testes de ADWIN estático e subárvores alternativas (HAT e HAT-R)
│       ├── test_e2e_flow.c      # Testes do fluxo completo ponta a ponta acumulado
│       └── test_main.c          # Runner principal que orquestra todas as suítes
└── benchmarks/                  # Comparação prequential e holdout (C11 vs River vs Scikit-Learn)
    ├── generate_streams.py      # Gera datasets sintéticos de classificação, drift e regressão
    ├── generate_mnist.py        # Gera streams do dataset MNIST (binário, 4 classes e 10 classes)
    ├── compare_river.py         # Compara Acurácia/MAE/R2, Tempo e Memória (C11 vs River vs Sklearn)
    ├── compare_triple.py        # Benchmark triplo no MNIST com profiling de RAM e heap
    └── bench_c_runner.c         # Executável C que lê os streams e reporta métricas e uso de RAM
```

---

## 9. Plano de Implementação e Relatórios de Benchmark

- **Plano de Implementação Concluído (Fases 0 a 4):** [`docs/06_implementation_plan.md`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/docs/06_implementation_plan.md)
- **Relatórios de Validação e Paridade em `benchmarks/`:**
  - [`benchmarks/benchmark_report.md`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/benchmarks/benchmark_report.md) — Paridade de Classificação Estacionária (VFDT vs River vs Scikit-Learn).
  - [`benchmarks/triple_benchmark_report.md`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/benchmarks/triple_benchmark_report.md) — Benchmark Triplo no MNIST com Profiling de Memória.
  - [`benchmarks/drift_benchmark_report.md`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/benchmarks/drift_benchmark_report.md) — Adaptação a *Concept Drift* (HAT vs River HAT).
  - [`benchmarks/regression_benchmark_report.md`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/benchmarks/regression_benchmark_report.md) — Regressão Contínua e Drift (VFDT-R / HAT-R vs River HTR / HATR vs Scikit-Learn).

---

## 10. Possíveis Trabalhos Futuros e Extensões

Embora todas as fases planejadas (Fases 0 a 4, incluindo Classificação, Adaptação a Concept Drift com ADWIN2/HAT e Regressão Contínua FIMT-DD/HAT-R) estejam 100% concluídas com zero heap dinâmico em runtime e conformidade estrita com o teto de 1 MB de RAM, o projeto pode ser expandido nas seguintes frentes:

1. **Deploy e Cross-Compilation em Hardware Real (Bare-Metal / RTOS):**
   - Targets de compilação cruzada para arquiteturas embarcadas reais (`arm-none-eabi-gcc` para ARM Cortex-M4F / Cortex-M7, RISC-V e ESP-IDF para ESP32).
   - Validação direta em microcontroladores rodando FreeRTOS ou Zephyr OS, medindo ciclos de clock e consumo de energia por amostra.

2. **Quantização e Aritmética de Ponto Fixo (Fixed-Point / INT8 / Q16.16):**
   - Suporte a microcontroladores de baixíssimo custo desprovidos de FPU de ponto flutuante em hardware (ex: ARM Cortex-M0/M0+).
   - Substituição configurável de `ht_float_t` por inteiros escalados com aproximação de ponto fixo (CORDIC / look-up tables) para logaritmos e funções exponenciais.

3. **Bindings em Python (CFFI / ctypes / Cython):**
   - Pacote Python com interface compatível com Scikit-Learn e River (`learn_one`, `predict_one`, `predict_proba_one`), permitindo usar a biblioteca C11 diretamente em pipelines de Machine Learning em streaming com velocidade nativa ($110\times$ a $400\times$ mais rápido que Python puro).

4. **Ensembles Online Embarcados (Streaming Random Forests / Online Bagging):**
   - Implementação de algoritmos de ensemble em streaming (*Online Bagging*, *Leveraging Bagging* ou *Streaming Random Patches*).
   - Particionamento do pool estático de nós e detectores ADWIN entre múltiplas árvores leves ($M$ árvores) dentro do mesmo buffer estático de $\le 1\text{ MB}$.
