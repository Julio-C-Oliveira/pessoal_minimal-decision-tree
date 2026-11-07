# Exemplos e Receitas da API (C11)

Esta página reúne exemplos práticos e completos de uso da API pública ([`code/include/ht_tree.h`](../code/include/ht_tree.h)) cobrindo todas as 4 variantes de modelo (`VFDT`, `HAT`, `VFDT-R`, `HAT-R`), atributos mistos com valores ausentes (`NaN`) e serialização binária para Flash/EEPROM.

---

## 1. Compilação da Biblioteca e Suíte de Testes

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

Para integrar em seu próprio projeto C/C++ ou firmware embarcado, basta incluir `code/include/` no caminho de headers (`-Icode/include`) e linkar com `code/build/libht.a -lm` (ou compilar diretamente os 5 arquivos `.c` de `code/src/`).

---

## 2. Exemplo 1: Classificação Incremental Padrão (`VFDT`)

Fluxo *prequential* (*Test-Then-Train*) para classificação multiclasse estacionária usando o *Hoeffding Tree Classifier*:

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

## 3. Exemplo 2: Classificação Adaptativa a *Concept Drift* (`HAT` + `ADWIN2`)

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

## 4. Exemplo 3: Regressão Contínua Incremental (`VFDT-R` e `HAT-R`)

Para prever alvos contínuos ($y \in \mathbb{R}$) usando *Variance Reduction* (`FIMT-DD`) e modelos lineares com SGD nas folhas:

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

## 5. Exemplo 4: Atributos Mistos (Numéricos + Categóricos) e Valores Ausentes (`NaN`)

Cada coluna do vetor `ht_value_t x[]` pode ser definida como contínua (`HT_FEAT_NUMERIC`) ou nominal discreta (`HT_FEAT_CATEGORICAL`), com suporte nativo a leituras ausentes:

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

## 6. Exemplo 5: Serialização Binária (`ht_tree_save` e `ht_tree_load`)

Salve o modelo completo (`include_training_stats = true`) para checkpoint de treinamento, ou gere uma imagem ultracompacta somente-inferência (`include_training_stats = false`) descartando todos os blocos `ht_leaf_stats_t`:

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
