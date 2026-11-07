---
name: embedded-c-tree
description: >-
  Use this skill whenever designing, writing, modifying, or debugging C11 code
  in include/, src/, tests/, or the Makefile for the embedded Hoeffding Tree.
---

# Embedded C11 Decision Tree Skill (`embedded-c-tree`)

Esta skill define o protocolo de engenharia para desenvolver e validar os módulos em C11 (`include/`, `src/`, `tests/`) garantindo o limite rígido de **$\le 1\text{ MB}$ de RAM**, compatibilidade com microcontroladores e cobertura progressiva de testes desde o nível de função individual até o fluxo total ponta a ponta.

> **Consulta Rápida às Referências (River & Scikit-Learn):**
> Sempre que precisar conferir uma fórmula, regra de desempate ou layout de struct original durante a implementação em C, consulte diretamente os mapas indexados em:
> - [`.agents/skills/study-reference/references/river_architecture_map.md`](../study-reference/references/river_architecture_map.md)
> - [`.agents/skills/study-reference/references/sklearn_architecture_map.md`](../study-reference/references/sklearn_architecture_map.md)

---

## 1. Checklist de Arquitetura e Memória

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

---

## 2. Protocolo de Testes Progressivos (Da Função ao Fluxo Total)

**Regra Inviolável ("Implementou $\rightarrow$ Testou em Todas as Camadas"):**
Nenhuma parte ou módulo em `src/` é considerado concluído sem que seus testes correspondentes sejam gerados e executados imediatamente após a implementação. **Nunca avance para a implementação do próximo módulo antes de validar a parte atual nos 3 níveis abaixo.**

### Arquitetura Modular da Suíte em `tests/`

A pasta `tests/` deve ser organizada de forma modular, com um cabeçalho de asserções e arquivos dedicados por camada orquestrados por `test_main.c`:

- `tests/test_harness.h`: Macros leves de asserção sem dependências externas (`HT_ASSERT(cond)`, `HT_ASSERT_EQ(a, b)`, `HT_ASSERT_FLOAT_EQ(a, b)` com tolerância adaptativa `1e-5f` para `float` e `1e-12` para `double`) e contadores de testes executados/aprovados.
- `tests/test_memory.c`: Testes unitários, de stress e reciclagem do `ht_memory`.
- `tests/test_observer.c`: Testes das funções de atualização e busca de split em `ht_observer` (Gaussiano de Welford e Categórico).
- `tests/test_criterion.c`: Testes de Gini, Info Gain, Hoeffding Bound e integração com observadores.
- `tests/test_tree.c`: Testes de roteamento, folhas (MC, NB, NBA), ciclo `learn_one`, splits, desativação de folhas e serialização.
- `tests/test_drift.c`: Testes de ADWIN com pool de buckets fixos e subárvores alternativas da HAT.
- `tests/test_e2e_flow.c`: Testes do **fluxo total acumulado** (pipeline ponta a ponta integrando todos os módulos implementados até o momento).
- `tests/test_main.c`: Ponto de entrada único que executa sequencialmente todas as suítes de nível 1, 2 e 3 disponíveis e retorna código `0` apenas se 100% passarem.

### Os 3 Níveis Obrigatórios de Teste a Cada Entrega

Sempre que implementar uma nova parte, escreva testes cobrindo obrigatoriamente:

1. **Nível 1 — Testes de Função / Unidade Isolada (Micro):**
   - Testa cada função pública e helper matemático crítico recém-implementado de forma isolada.
   - Compara as saídas com valores analíticos conhecidos (*ground truth* calculado à mão).
   - Exercita obrigatoriamente **casos de borda (*edge cases*)**:
     - Valores ausentes (`NaN` via `HT_ISNAN`, categorias negativas/desconhecidas).
     - Amostras insuficientes ou degeneradas ($n \in \{0, 1, 2\}$ no Welford, atributos constantes com variância $s^2 = 0$, distribuições de classe 100% puras ou uniformes).
     - Limites físicos e índices inválidos (`HT_INVALID_IDX = 0xFFFF`, tentativa de alocar em pool cheio, liberação repetida ou em ordem reversa).

2. **Nível 2 — Testes de Integração entre Subsistemas (Meso):**
   - Conecta a parte recém-criada aos módulos das camadas anteriores já implementados:
     - **Após `ht_memory`:** Ciclos intensivos de `alloc` $\rightarrow$ `free` $\rightarrow$ `alloc` provando reaproveitamento $O(1)$ da *free-list*, ausência de fragmentação e respeito estrito ao orçamento $\le 1\text{ MB}$.
     - **Após `ht_observer`:** Alimentação de observadores Gaussianos e Categóricos no interior de um bloco `ht_leaf_stats_t` alocado via `ht_memory`, validando estimativas de PDF/CDF e geração de candidatos de corte.
     - **Após `ht_criterion`:** Integração `ht_memory` + `ht_observer` + `ht_criterion` populando múltiplos atributos com distribuições sintéticas separáveis e verificando se a busca seleciona o atributo/threshold ótimo e avalia corretamente o *Hoeffding Bound* ($\Delta G > \epsilon$ ou $\epsilon < \tau$).
     - **Após `ht_tree` / `ht_drift`:** Integração entre roteamento iterativo, predição nas folhas (MC/NB/NBA), execução de split com devolução imediata do `ht_leaf_stats_t` ao `stats_pool` e rotina VFDT de desativação de folhas menos promissoras ($p_l \cdot e_l$).

3. **Nível 3 — Teste de Fluxo Total / Ponta a Ponta Acumulado (Macro / E2E):**
   - Ao final de cada entrega, atualize e execute `tests/test_e2e_flow.c` para exercitar o **fluxo completo do sistema disponível até aquele momento**:
     - Enquanto os módulos de base estão sendo criados, o fluxo E2E simula o pipeline completo de dados desde a entrada `ht_value_t[]` até a decisão de split e reciclagem de memória.
     - Assim que `ht_tree` (e posteriormente `ht_drift`) estiverem implementados, o teste de fluxo total deve rodar um mini-stream sintético determinístico ponta a ponta validando:
       1. Inicialização 100% estática (`ht_init`) sem chamadas ao heap.
       2. Loop *prequential* completo (`ht_predict_proba` / `ht_predict` $\rightarrow$ `ht_learn_one`) comprovando ganho real de acurácia após os splits.
       3. Roteamento e aprendizado robustos na presença de `NaN` e categorias ausentes (fallback para o filho mais populoso).
       4. Pressão deliberada de memória (enchendo o `stats_pool`) para comprovar a desativação/reativação de folhas em pleno voo sem perda de estabilidade.
       5. Ciclo completo de persistência binária (`ht_save` $\rightarrow$ `ht_load`) verificando que a árvore restaurada produz exatamente as mesmas probabilidades e predições que a árvore original.

---

## 3. Validação Obrigatória e Commits Atômicos

Após implementar qualquer parte e seus testes de Nível 1, 2 e 3, execute sempre os dois modos de compilação:

```bash
make clean && make test
make clean && make test CFLAGS_EXTRA="-DHT_USE_DOUBLE"
```

- Nenhum warning é permitido (`-Wall -Wextra -Werror -pedantic`).
- Todos os testes (função, integração e fluxo total) devem passar tanto em `float` (32-bit) quanto em `double` (64-bit).
- Faça commits atômicos mantendo a árvore sempre verde (`feat(<escopo>): ...` e `test(<escopo>): ...`).
