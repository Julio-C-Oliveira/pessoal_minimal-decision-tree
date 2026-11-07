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

## 4. Documentação Completa (Wiki, Estudos e Benchmarks)

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
| **Relatórios de Benchmark ([`benchmarks/`](benchmarks/))** | Paridade vs River e Scikit-Learn em Classificação ([`benchmark_report.md`](benchmarks/benchmark_report.md)), MNIST Triplo ([`triple_benchmark_report.md`](benchmarks/triple_benchmark_report.md)), Drift ([`drift_benchmark_report.md`](benchmarks/drift_benchmark_report.md)) e Regressão ([`regression_benchmark_report.md`](benchmarks/regression_benchmark_report.md)). |

---

## 5. Estrutura do Repositório

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

## 6. Licença (`BSD 3-Clause`)

Distribuído sob a licença permissiva **[BSD 3-Clause License](LICENSE)** (`SPDX-License-Identifier: BSD-3-Clause`), permitindo uso livre acadêmico, pessoal ou comercial (em código-fonte ou firmware binário) e preservando integralmente os avisos de direitos autorais dos projetos de referência **River** ([`references/river_tree/LICENSE`](references/river_tree/LICENSE)) e **Scikit-Learn** ([`references/sklearn_tree/COPYING`](references/sklearn_tree/COPYING)).
