# Scikit-Learn Decision Tree (`references/sklearn_tree/`) — Mapa Arquitetural

Este documento mapeia as decisões de arquitetura em baixo nível (Cython/C) do módulo `sklearn/tree` em [`references/sklearn_tree/`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree) para consulta rápida durante o desenvolvimento da nossa Hoeffding Tree em C11.

---

## 1. Índice Rápido de Busca (Onde Encontrar Cada Conceito)

| Conceito / Decisão Arquitetural | Arquivo e Linhas Exatas | Como Adaptamos no C11 ($\le 1\text{ MB}$) |
| :--- | :--- | :--- |
| Enum de tipos de split (`SPLIT_LEAF`, `SPLIT_NUMERIC`, etc.) | [`_utils.pxd#L21-L26`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_utils.pxd#L21-L26) | `ht_split_kind_t` (`uint8_t`) dentro de `ht_node_t` |
| Definição da struct `Node` (topologia plana sem ponteiros) | [`_utils.pxd#L28-L48`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_utils.pxd#L28-L48) | `ht_node_t` usando `uint16_t` (`2 B`) em vez de `intp_t` (`8 B`) |
| Layout binário `NODE_DTYPE` e alinhamento de bytes | [`_tree.pyx#L70-L81`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pyx#L70-L81) | Ordenação de campos + `_Static_assert(sizeof(ht_node_t) <= 32)` |
| Separação entre topologia (`Node* nodes`) e contagens (`float64_t* value`) | [`_tree.pxd#L41-L49`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pxd#L41-L49), [`_tree.pyx#L821-L855`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pyx#L821-L855) | Separação entre `node_pool` (`ht_node_t` leve) e `stats_pool` (`ht_leaf_stats_t` pesado, só para folhas ativas) |
| Inserção de nó no array plano (`Tree._add_node`) | [`_tree.pyx#L957-L1015`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pyx#L957-L1015) | Alocação $O(1)$ via *free-list* no `node_pool` estático |
| Predicado inline de roteamento (`goes_left` + `NaN`) | [`_utils.pxd#L63-L85`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_utils.pxd#L63-L85) | Função `static inline bool ht_goes_left(...)` com fallback para filho mais populoso em caso de `NaN` |
| Roteamento iterativo raiz $\to$ folha sem recursão (`_apply_dense`) | [`_tree.pyx#L1032-L1077`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pyx#L1032-L1077) | Loop `while (!ht_node_is_leaf(node))` usando índices `uint16_t` |
| Serialização binária direta (`__getstate__` / `__setstate__`) | [`_tree.pyx#L871-L910`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pyx#L871-L910) | Dump binário direto dos pools (`ht_save` / `ht_load`) sem necessidade de reconstruir ponteiros |
| Interface modular `Criterion` e buffers `sum_total`/`sum_left`/`sum_right` | [`_criterion.pxd#L34-L94`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_criterion.pxd#L34-L94), [`_criterion.pyx#L253-L487`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_criterion.pyx#L253-L487) | `ht_criterion_vtable_t` operando diretamente sobre `(pre_dist, lhs_dist, rhs_dist)` |
| Fórmula otimizada de Entropia (base 2) | [`_criterion.pyx#L532-L603`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_criterion.pyx#L532-L603), [`_utils.pxd#L138-L139`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_utils.pxd#L138-L139) | `ht_info_gain.c` usando `HT_LOG2` com proteção `count_k > 0.0` |
| Fórmula otimizada de Gini (uma única divisão ao final) | [`_criterion.pyx#L605-L688`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_criterion.pyx#L605-L688) | `ht_gini.c` acumulando `sq_count += c * c` e dividindo uma única vez por `total * total` |
| Otimização `proxy_impurity_improvement` vs `impurity_improvement` | [`_criterion.pyx#L147-L200`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_criterion.pyx#L147-L200) | Evitar normalizações desnecessárias no loop interno dos candidatos a split |
| Struct `SplitRecord` alocada na stack para retorno de candidato | [`_splitter.pxd#L14-L44`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_splitter.pxd#L14-L44), [`_splitter.pyx#L319`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_splitter.pyx#L319) | `ht_split_suggestion_t` passada por ponteiro na stack (zero heap) |
| Cálculo seguro do ponto médio sem overflow (`a/2.0 + b/2.0`) | [`_partitioner.pxd#L18-L51`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_partitioner.pxd#L18-L51) | Evitar `(min + max)` direto quando próximo ao limite de `float32` |
| Por que o CART (`DensePartitioner`) gasta $O(N)$ de RAM | [`_partitioner.pxd#L102-L169`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_partitioner.pxd#L102-L169), [`_partitioner.pyx#L78-L170`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_partitioner.pyx#L78-L170) | Substituído pelos observadores incrementais $O(1)$ (Welford Gaussiano e tabela nominal) nas folhas ativas |

---

## 2. Detalhamento das Principais Decisões Arquiteturais

### 2.1 Representação da Árvore em Array Plano (`_utils.pxd` e `_tree.pxd`)
No Scikit-Learn, um nó não aponta para seus filhos via ponteiros de memória (`Node*`), mas sim via índices inteiros (`intp_t left_child`, `intp_t right_child`, onde `_TREE_LEAF = -1` em [`_tree.pyx#L52-L55`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pyx#L52-L55)):
- **Por que o Scikit-Learn separa `Node* nodes` de `float64_t* value` ([`_tree.pxd#L41-L48`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pxd#L41-L48))?**
  1. `sizeof(Node)` precisa ser constante em tempo de compilação para indexação rápida `nodes[i]`.
  2. Durante a inferência (`_apply_dense` em [`_tree.pyx#L1032-L1077`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pyx#L1032-L1077)), o loop de roteamento da raiz até a folha percorre **apenas** o array `nodes` (quente na cache L1), e só acessa `value[leaf_id * value_stride]` uma única vez quando chega na folha!
- **Nossa evolução para Sistemas Embarcados:**
  - Em vez de `realloc` dinâmico (`_resize_c` em [`_tree.pyx#L929-L955`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pyx#L929-L955)), usamos um `node_pool` estático indexado por `uint16_t` (`HT_INVALID_IDX = 0xFFFF`).
  - Separamos radicalmente o estado pesado de aprendizado (`ht_leaf_stats_t`, que só existe enquanto uma folha está ativa) da estrutura permanente do nó (`ht_node_t`).

### 2.2 Roteamento Iterativo e Valores Ausentes (`goes_left` e `_apply_dense`)
Em [`_utils.pxd#L63-L85`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_utils.pxd#L63-L85), a função `cdef inline bint goes_left(...)` centraliza toda a decisão de ir para a esquerda ou direita:
- Primeiro testa `if inlinable_isnan(value): return missing_go_to_left`.
- Depois avalia `SPLIT_NUMERIC` (`value <= threshold`) ou `SPLIT_CATEGORICAL_BITSET`.
- Em [`_tree.pyx#L1059-L1074`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_tree.pyx#L1059-L1074), `_apply_dense` usa `while node.left_child != _TREE_LEAF:` sem nenhuma chamada recursiva, garantindo uso de stack $O(1)$.

### 2.3 Otimizações Numéricas em `Entropy` e `Gini` (`_criterion.pyx`)
- **Gini ([`_criterion.pyx#L622-L645`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_criterion.pyx#L622-L645)):**
  Em vez de dividir $p_c = N_c / N$ dentro do loop de classes ($C$ divisões de ponto flutuante), o Scikit-Learn acumula a soma dos quadrados brutos `sq_count += count_k * count_k` e faz **uma única divisão** ao final:
  $$G = 1.0 - \frac{\sum_{c} N_c^2}{N^2}$$
- **Retorno de Candidatos na Stack (`SplitRecord` em [`_splitter.pxd#L14-L44`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/sklearn_tree/_splitter.pxd#L14-L44)):**
  O avaliador de split recebe um ponteiro `SplitRecord* split` alocado na stack da função chamadora (`_tree.pyx#L205`), preenchendo o melhor candidato sem alocar structs temporárias no heap (diferente do River em Python que instancia objetos `BranchFactory` a cada avaliação).
