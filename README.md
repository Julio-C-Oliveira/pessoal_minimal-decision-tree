# Minimal Decision Tree (Embedded Hoeffding Tree in C)

Este repositório é dedicado ao estudo teórico e desenvolvimento de uma implementação ultraleve de **Hoeffding Tree (VFDT / HAT)** em **C11 puro**, projetada para rodar **treino incremental (online) e inferência em tempo real** em dispositivos embarcados/microcontroladores com forte restrição de memória (**orçamento $\le$ 1 MB de RAM**, escalável para dezenas de KB).

## Por que Hoeffding Tree no lugar do CART?
Algoritmos batch tradicionais como o **CART** exigem que todo o dataset esteja carregado em RAM (ou seja re-escaneado múltiplas vezes) para ordenar atributos e encontrar pontos de corte. Já a **Hoeffding Tree (VFDT)** processa cada amostra $O(1)$ única vez (*stream learning*), mantendo apenas estatísticas suficientes nas folhas e usando a **Desigualdade de Hoeffding** para garantir matematicamente (com probabilidade $1 - \delta$) que o split escolhido com $n$ amostras é o mesmo que seria escolhido com infinitas amostras:

$$\epsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$$

---

## Fontes e Referências no Repositório (`references/`)

Para embasar a matemática e a arquitetura em baixo nível, o repositório armazena cópias locais de referência em [`references/`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references):

1. **River (`references/river_tree/`):**
   - **Origem:** Repositório oficial [`online-ml/river`](https://github.com/online-ml/river) (branch `main`, commit `4768de0579f977315e65cd23ee658534c1d62c26`, licença *BSD-3-Clause*).
   - **Módulos extraídos:** `river/tree/` e `river/drift/`.
   - **Objetivo no projeto:** Referência algorítmica direta para a *Hoeffding Tree Classifier* ([`hoeffding_tree_classifier.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree_classifier.py)), folhas MC/NB/NBA ([`htc_nodes.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py)), aproximação Gaussiana ([`gaussian_splitter.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/gaussian_splitter.py)), critérios de impureza ([`gini.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/gini.py), [`info_gain.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/info_gain.py)) e *Concept Drift* ([`hoeffding_adaptive_tree_classifier.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_adaptive_tree_classifier.py), [`adwin.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/drift/adwin.py)).

2. **Scikit-Learn (`references/sklearn_tree/`):**
   - **Origem:** Repositório oficial [`scikit-learn/scikit-learn`](https://github.com/scikit-learn/scikit-learn) (branch `main`, commit `6e37063f7ab7ef7a7e257b9d44bc3ec6ee1b1bcd`, licença *BSD-3-Clause*).
   - **Módulo extraído:** `sklearn/tree/` (implementado em Cython/C).
   - **Objetivo no projeto:** Referência arquitetural de dados contíguos e baixo nível — especialmente o layout de array plano de structs `Node` sem ponteiros ([`_tree.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pxd), [`_utils.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_utils.pxd)) e a separação modular de `Criterion` ([`_criterion.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_criterion.pxd)) e `Splitter` ([`_splitter.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_splitter.pxd)).

3. **Artigos Científicos (`references/papers/`):**
   - PDFs dos artigos fundamentais de Domingos, Hulten, Spencer, Bifet e Gavaldà.

---

## Decisões de Arquitetura e Design

1. **Escopo de ML:**
   - **Classificação Multiclasse Incremental** (*on-device training + inference* no estilo *prequential*: testa primeiro, treina depois).
   - **Predição nas Folhas:** Suporte a *Majority Class* (MC), *Naive Bayes* (NB) e *Naive Bayes Adaptive* (NBA) — aproveitando as estatísticas que as folhas ativas já armazenam para avaliação de splits com custo adicional quase zero de RAM.
   - **Critérios de Split:** *Gini Impurity* e *Information Gain (Entropia)* selecionáveis por configuração.
   - **Topologia:** Splits **estritamente binários** para todos os tipos de atributos:
     - Numéricos: $x_i \le \text{threshold}$ vs $x_i > \text{threshold}$
     - Categóricos: $x_i == \text{categoria}$ vs $x_i \neq \text{categoria}$
     - *Vantagem:* Todos os nós da árvore possuem tamanho fixo (`left` e `right`), eliminando arrays de ponteiros dinâmicos.

2. **API de Entrada (`ht_value_t`) e Schema Fixo:**
   - Em vez de dicionários dinâmicos (como no Python/River), a árvore registra no `init` um schema fixo de tipos por coluna (`HT_FEAT_NUMERIC` ou `HT_FEAT_CATEGORICAL`).
   - Cada amostra é passada como um array contíguo de `union ht_value_t { ht_float_t num; int32_t cat; }`, garantindo zero overhead de parsing e excelente localidade de cache.

3. **Tratamento de Valores Ausentes (`NaN` / Categoria Desconhecida) com Custo Zero de RAM:**
   - **No Roteamento (Inferência/Treino):** Se o atributo de split de um nó interno estiver ausente (`isnan(x)` ou `cat < 0`), a amostra segue para o **filho mais populoso** comparando em tempo real a soma de `class_counts` dos filhos `left` e `right` (zero bytes extras no nó).
   - **No Aprendizado da Folha:** Atualiza normalmente o contador da classe verdadeira na folha e os observadores dos demais atributos válidos, ignorando apenas o observador do atributo ausente.

4. **Representação de Atributos (O(1) por Atributo/Classe):**
   - **Numéricos:** Aproximação Gaussiana incremental via **Algoritmo de Welford** (`count`, `mean`, `M2`, `min`, `max` por classe), avaliando $k$ pontos candidatos (ex: 10 bins entre `min` e `max`) na hora de testar o split (`grace_period`).
   - **Categóricos Nominais:** Tabela de frequência de tamanho limitado pela cardinalidade máxima configurada (`MAX_CATEGORIES`).

5. **Polimorfismo Híbrido (Inspirado em `sklearn/tree` + Data-Oriented Design):**
   - **Nível da Árvore:** Polimorfismo via *function pointers* (`ht_criterion_vtable_t`) para o critério de impureza (Gini vs Info Gain), já que há apenas 1 instância por árvore.
   - **Nível dos Nós e Observadores:** Uso direto de `enum` e `union` (sem *vtable* por folha ou atributo), economizando 4 a 8 bytes de ponteiro por observador e permitindo *inlining* agressivo pelo compilador C.

6. **Precisão Numérica Configurável (`ht_float_t`):**
   - Padrão em **`float` (32-bit)** para cortar o consumo de RAM das estatísticas pela metade e aproveitar a FPU nativa de microcontroladores (ARM Cortex-M4F/M7, ESP32, RP2350).
   - Flag de compilação `-DHT_USE_DOUBLE` para alternar para `double` (64-bit) em testes de paridade numérica com o River.

7. **Gerenciamento de Memória Determinístico e Indexação Livre de Ponteiros (`uint16_t`):**
   - **Índices em vez de Ponteiros:** Inspirado no `sklearn/tree/_tree.pxd`, todas as referências entre nós (`left_child`, `right_child`, `stats_idx`) usam índices inteiros (`uint16_t`, com `HT_INVALID_IDX = 0xFFFF`) em arrays contíguos. Isso economiza RAM em arquiteturas 32/64-bit e torna a estrutura **100% relocável para serialização binária direta** (salvar/carregar da Flash/disco tanto em modo *checkpoint completo* quanto em modo *somente-inferência*).
   - **Separação entre Topologia e Estado de Aprendizado:**
     - `ht_node_t` (**Nó da Árvore - Leve**): Guarda apenas índices dos filhos (`left`, `right`), índice do estado (`stats_idx`), condição de split (`feature_idx`, `threshold`/`cat_value`) e contadores de classe (`class_counts[NUM_CLASSES]`).
     - `ht_leaf_stats_t` (**Estado de Aprendizado da Folha - Pesado**): Guarda os observadores de atributos (Gaussianas de Welford / contadores categóricos por classe) e o acumulador de acertos do Naive Bayes (para NBA).
   - **Slab / Pool Allocator Estático:** Dois pools de blocos de tamanho fixo (`node_pool` e `stats_pool` com *free-list* em $O(1)$) dentro do buffer de RAM de $\le 1\text{ MB}$:
     - Quando uma folha faz um split, ela vira um nó interno e **devolve imediatamente seu `ht_leaf_stats_t` para o `stats_pool`**.
     - **Desativação de Folhas (VFDT Memory Management):** Quando o `stats_pool` atinge o limite, as folhas menos promissoras ($p_l \times e_l$) são **desativadas** — liberando seu `ht_leaf_stats_t` de volta ao pool, mas mantendo `class_counts` no `ht_node_t` para continuar fazendo predições via *Majority Class*.

---

## Estimativa de Orçamento de Memória (Exemplo Prático)

Considere um problema com $F = 16$ atributos numéricos, $C = 4$ classes e `float` (32-bit):
- **Observador Gaussiano por (atributo, classe):** `mean`, `m2`, `min`, `max`, `count` = $5 \times 4\text{ B} = 20\text{ B}$.
- **Estado de Aprendizado por Folha Ativa (`ht_leaf_stats_t`):**
  $$16 \text{ features} \times 4 \text{ classes} \times 20\text{ B} + \text{metadados NBA} \approx 1.3\text{ KB por folha ativa}$$
- **Nó Estrutural (`ht_node_t` — Internos e Folhas Inativas):**
  $$\text{índices } (3 \times 2\text{ B}) + \text{flags/feature } (2\text{ B}) + \text{split } (4\text{ B}) + \text{contadores de classe } (4 \times 4\text{ B}) = 28\text{ B por nó}$$
- **Capacidade em apenas 512 KB de RAM:**
  - $256$ folhas ativas aprendendo simultaneamente ($\approx 332\text{ KB}$)
  - Até $4.096$ nós internos / folhas inativas ($\approx 115\text{ KB}$)
  - Folga ampla abaixo do teto de **1 MB**!

---

## Estrutura Planejada do Repositório

```text
pessoal_minimal-decision-tree/
├── README.md
├── Makefile                     # Build C11 puro (lib, testes, benchmarks, flags float/double)
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
│   └── 06_implementation_plan.md # Plano de implementação persistente multi-chat/agente
├── include/                     # Headers públicos e internos (C11)
│   ├── ht_config.h              # Tipos (ht_float_t, ht_value_t), limites e constantes
│   ├── ht_memory.h              # Slab/Pool allocator determinístico baseado em índices uint16_t
│   ├── ht_observer.h            # Observadores Gaussianos (Welford) e Categóricos
│   ├── ht_criterion.h           # Vtable e implementações de Gini Impurity, Info Gain e Hoeffding Bound
│   ├── ht_tree.h                # API da VFDT (init, predict, learn_one, serialize/deserialize)
│   └── ht_drift.h               # Fase 3: ADWIN com buckets fixos e HAT
├── src/                         # Implementação em C11 (zero dependências além de math.h/stdint.h)
│   ├── ht_memory.c
│   ├── ht_observer.c
│   ├── ht_criterion.c
│   ├── ht_tree.c
│   └── ht_drift.c
├── tests/                       # Suíte modular de testes em C (função -> integração -> fluxo total)
│   ├── test_harness.h           # Macros de asserção sem dependências (tolerância adaptativa float/double)
│   ├── test_memory.c            # Testes de unidade, stress e reciclagem do ht_memory
│   ├── test_observer.c          # Testes do Welford Gaussiano, PDF/CDF e observador categórico
│   ├── test_criterion.c         # Testes de Gini, Info Gain, Hoeffding Bound e busca de split
│   ├── test_tree.c              # Testes de roteamento, folhas MC/NB/NBA, splits, desativação e dump
│   ├── test_drift.c             # Testes de ADWIN estático e subárvores alternativas (HAT)
│   ├── test_e2e_flow.c          # Testes do fluxo completo ponta a ponta acumulado
│   └── test_main.c              # Runner principal que orquestra todas as suítes
└── benchmarks/                  # Comparação prequential (C vs River em Python)
    ├── generate_streams.py      # Gera datasets sintéticos (SEA, Agrawal, Hyperplane, LED)
    ├── compare_river.py         # Compara Acurácia Prequential, Tempo e Memória (River vs C)
    └── bench_c_runner.c         # Executável C que lê o stream e reporta métricas e uso de RAM
```

---

## Roadmap e Tarefas (TO-DO)

### Fase 0: Estudo Teórico e Mapeamento (`docs/`)
- [x] **Estudar sobre Hoeffding Tree (`references/papers/`):**
  - [x] *Mining High-Speed Data Streams* (Pedro Domingos & Geoff Hulten, 2000) — Derivação do limite de Hoeffding, algoritmo base, `grace_period` ($n_{\min}$) e *tie-breaking* ($\tau$).
  - [x] *Very Fast Decision Tree Learner* (Hulten, Spencer & Domingos, 2001) — Desativação/reativação de folhas menos promissoras ($p_l \cdot e_l$), descarte de atributos ruins e controle de memória.
  - [x] *Adaptive Strategies for Learning Classifiers from Data Streams* (Albert Bifet, 2009) — Tratamento de atributos numéricos (aproximação Gaussiana vs quantis/EBST) e categóricos, além de *Naive Bayes Adaptive* (NBA) nas folhas.
  - [x] *Learning from Time-Changing Data Streams with Adaptive Hoeffding Trees* (Albert Bifet & Ricard Gavaldà, 2009) — Detecção de *Concept Drift* com ADWIN em cada nó e treinamento de subárvores alternativas.
- [x] **Estudar a implementação do River (`references/river_tree/`):**
  - [x] [`hoeffding_tree_classifier.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree_classifier.py) (Fluxo de `learn_one`, `predict_proba_one`, tentativa de split e gestão de memória).
  - [x] [`nodes/branch.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/branch.py) e [`nodes/htc_nodes.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py) (Diferença entre folhas MC, NB e NBA).
  - [x] [`splitter/gaussian_splitter.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/gaussian_splitter.py) e [`utils.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/utils.py) (Como o River calcula os pontos de corte candidatos a partir da distribuição normal por classe).
  - [x] [`split_criterion/gini.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/gini.py) e [`split_criterion/info_gain.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/info_gain.py) (Cálculo do mérito, normalização do range $R$ e penalidade por fração mínima de ramo).
- [x] **Estudar a arquitetura do Scikit-Learn (`references/sklearn_tree/`):**
  - [x] [`_tree.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pxd) / [`_utils.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_utils.pxd) (Representação da árvore como array plano contíguo de structs `Node` indexado por inteiros e separação do buffer `value`).
  - [x] [`_criterion.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_criterion.pxd) / [`_criterion.pyx`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_criterion.pyx) (Design modular dos critérios de impureza e avaliação incremental de partições).
  - [x] [`_splitter.pxd`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_splitter.pxd) / [`_splitter.pyx`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_splitter.pyx) (Registro `SplitRecord` para retorno de candidatos a ponto de corte).

### Fase 1: Núcleo VFDT em C11 + Gestão de Memória (`<= 1MB`) + Testes Progressivos (`tests/`)
> **Protocolo de Testes em 3 Níveis:** Cada módulo implementado abaixo deve vir imediatamente acompanhado de seus testes de **Nível 1 (Função/Unidade)**, **Nível 2 (Integração com camadas anteriores)** e **Nível 3 (Fluxo Total Acumulado em `test_e2e_flow.c`)**, validados em `float` e `double` antes de avançar ao próximo item.

- [x] Configurar `Makefile`, `ht_config.h` e `tests/test_harness.h` (suporte a `float`/`double`, `ht_value_t`, `stdint.h`, compilação limpa com `-Wall -Wextra -Werror -std=c11`).
- [ ] Implementar `ht_memory` + `tests/test_memory.c`: Pool/Slab allocator estático baseado em índices `uint16_t` para `ht_node_t` e `ht_leaf_stats_t` (zero `malloc` em runtime, medição exata de bytes usados, stress test de reciclagem).
- [ ] Implementar `ht_observer` + `tests/test_observer.c`:
  - [ ] Observador numérico Gaussiano incremental (Welford) + avaliação de $k$ thresholds candidatos usando PDF/CDF normal.
  - [ ] Observador categórico nominal com busca do melhor split binário ($x_i == c$ vs $x_i \neq c$).
- [ ] Implementar `ht_criterion` + `tests/test_criterion.c`:
  - [ ] Vtable (`ht_criterion_vtable_t`) + implementações de *Gini Impurity* e *Information Gain (Entropia)*.
  - [ ] Cálculo do *Hoeffding Bound* ($\epsilon$) e verificação de split ($\Delta G > \epsilon$ ou $\epsilon < \tau$) integrado aos observadores.
- [ ] Implementar `ht_tree` (VFDT) + `tests/test_tree.c` + `tests/test_e2e_flow.c`:
  - [ ] Roteamento de amostras da raiz até a folha (incluindo fallback para filho mais populoso em caso de `NaN`).
  - [ ] Predição nas folhas via *Majority Class* (MC), *Naive Bayes* (NB) e *Naive Bayes Adaptive* (NBA).
  - [ ] Ciclo `learn_one`: atualização de estatísticas, checagem a cada `grace_period`, execução de split e reciclagem do `ht_leaf_stats_t`.
  - [ ] Gestão de memória VFDT: cálculo de folha menos promissora ($p_l \cdot e_l$) e desativação automática quando o `stats_pool` encher.
  - [ ] Serialização binária (`ht_save` / `ht_load` em modo completo ou somente-inferência) e validação do fluxo ponta a ponta completo.

### Fase 2: Validação Prequential e Benchmark contra o River (`benchmarks/`)
- [ ] Criar script Python com `river` para gerar streams clássicos (ex: *SEA*, *Agrawal*, *Hyperplane*) e comparar curva de acurácia *prequential*, número de nós, número de splits e consumo de RAM entre o River e nossa implementação em C.

### Fase 3: Concept Drift — Hoeffding Adaptive Tree (HAT)
- [ ] Implementar detector **ADWIN** em C com lista de buckets estática (alocada a partir de um pool dedicado) + testes unitários em `tests/test_drift.c`.
- [ ] Adicionar suporte a subárvores alternativas (`alternate_tree`) nos nós internos da árvore quando o ADWIN detectar mudança de distribuição, substituindo a subárvore antiga caso a alternativa atinja menor erro + teste de fluxo total com drift em `tests/test_e2e_flow.c`.
- [ ] Validar em streams com *Concept Drift* abrupto e gradual contra `river.tree.HoeffdingAdaptiveTreeClassifier`.
