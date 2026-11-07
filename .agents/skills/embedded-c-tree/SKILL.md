---
name: embedded-c-tree
description: >-
  Use this skill whenever designing, writing, modifying, or debugging C11 code
  in include/, src/, tests/, or the Makefile for the embedded Hoeffding Tree.
---

# Embedded C11 Decision Tree Skill (`embedded-c-tree`)

Esta skill define o protocolo de engenharia para desenvolver e validar os módulos em C11 (`include/`, `src/`, `tests/`) garantindo o limite rígido de **$\le 1\text{ MB}$ de RAM** e compatibilidade com microcontroladores.

## Checklist de Arquitetura e Memória

1. **Orçamento de Memória e Layout de Structs:**
   - Ordene os campos das structs do maior alinhamento para o menor (`ht_float_t` / `uint32_t` -> `uint16_t` -> `uint8_t`) para evitar *padding* desperdiçado.
   - Inclua `_Static_assert` nos headers/testes para monitorar o tamanho de `ht_node_t` e `ht_leaf_stats_t`.
   - Use índices `uint16_t` (`HT_INVALID_IDX = 0xFFFF`) para `left_child`, `right_child` e `stats_idx`. Nunca armazene ponteiros de 64-bit dentro de `ht_node_t`.

2. **Slab / Pool Allocator (`ht_memory`):**
   - Dois pools independentes com *free-list* intrusiva em $O(1)$:
     - `node_pool`: aloca `ht_node_t` para a raiz e novos filhos no momento do split.
     - `stats_pool`: aloca `ht_leaf_stats_t` apenas para **folhas ativas**.
   - **Reciclagem Obrigatória:**
     - Quando uma folha faz split e vira nó interno: liberar seu `stats_idx` de volta para o `stats_pool` **antes** ou durante a alocação dos dois filhos, definindo `node->stats_idx = HT_INVALID_IDX`.
     - Quando o `stats_pool` ficar cheio: acionar a rotina VFDT de desativação das folhas menos promissoras ($p_l \cdot e_l = \text{total\_weight} - \max_c(\text{class\_counts}[c])$) para recuperar blocos `ht_leaf_stats_t`.

3. **Estabilidade Numérica (`ht_float_t`):**
   - Sempre use macros/funções abstratas compatíveis com `float` e `double` (ex: `HT_LOG2(x)`, `HT_SQRT(x)`, `HT_EXP(x)`, `HT_ERF(x)`, `HT_ISNAN(x)` mapeando para `log2f`/`log2`, `sqrtf`/`sqrt`, etc. em `ht_config.h`).
   - **Algoritmo de Welford (Gaussiana):**
     - Proteja o cálculo da variância ($s^2 = M_2 / (n - 1)$) contra $n < 2$ ou $s^2 \le 10^{-9}$.
   - **Naive Bayes / NBA nas Folhas:**
     - Calcule as probabilidades em espaço logarítmico (`log_pdf`) com *log-sum-exp* trick para evitar *underflow* em `float` 32-bit quando houver muitos atributos.

4. **Validação Obrigatória antes do Commit:**
   Execute sempre os dois modos de compilação e a suíte de testes:
   ```bash
   make clean && make test
   make clean && make test CFLAGS_EXTRA="-DHT_USE_DOUBLE"
   ```
   - Nenhum warning é permitido (`-Wall -Wextra -Werror -pedantic`).
   - Faça commits atômicos por módulo (`feat(memory): ...`, `feat(observer): ...`, `test(observer): ...`).
