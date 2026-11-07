---
name: river-parity-benchmark
description: >-
  Use this skill when creating or running benchmarks, prequential evaluations,
  synthetic data stream generators, or parity comparisons between the C11
  Hoeffding Tree implementation and Python's River library in benchmarks/.
---

# River Parity & Memory Benchmark Skill (`river-parity-benchmark`)

Esta skill guia a validação experimental da nossa implementação em C11 comparando-a diretamente com `river.tree.HoeffdingTreeClassifier` (e `HoeffdingAdaptiveTreeClassifier` na Fase 3).

## Protocolo de Comparação Justa (Apples-to-Apples)

1. **Alinhamento de Hiperparâmetros entre C e River:**
   Certifique-se de configurar ambos os modelos exatamente com os mesmos parâmetros:
   - `grace_period = 200`
   - `delta = 1e-7`
   - `tau = 0.05`
   - `split_criterion = 'info_gain'` (e testar também `'gini'`)
   - `leaf_prediction = 'nba'` (testar também `'mc'` e `'nb'`)
   - `nb_threshold = 0`
   - `binary_split = True` (nossa árvore em C faz splits estritamente binários)
   - `min_branch_fraction = 0.01`
   - `max_share_to_split = 0.99`
   - `splitter = GaussianSplitter(n_splits=10)`

2. **Avaliação Prequential (Test-Then-Train):**
   Para cada amostra $(x_t, y_t)$ do stream ($t = 1, \dots, N$):
   1. Executar predição $\hat{y}_t = \text{predict}(x_t)$ e atualizar a acurácia acumulada / janela deslizante.
   2. Executar aprendizado incremental $\text{learn\_one}(x_t, y_t)$.
   3. Registrar periodicamente (ex: a cada 1.000 amostras):
      - Acurácia prequential (%)
      - Total de nós da árvore (`n_nodes`), folhas ativas (`n_active_leaves`) e folhas inativas (`n_inactive_leaves`)
      - Bytes de RAM consumidos no `node_pool` + `stats_pool` (verificando que permanece $\le 1\text{ MB}$)
      - Tempo médio por amostra ($\mu\text{s}$/sample)

3. **Verificação de Paridade Numérica (`float` vs `double`):**
   - Rode o executável C compilado com `-DHT_USE_DOUBLE` para verificar se o número de splits e a acurácia convergem com o River (que usa `float64` do Python).
   - Em seguida, rode o executável C no modo padrão `float` (32-bit) para medir o ganho de memória (50% menos RAM nos observadores) e confirmar que a acurácia prequential se mantém equivalente ($\Delta < 0.5\%$).

4. **Benchmarks de Orçamento e Escalonamento de RAM (`bench-memory`):**
   - Ao avaliar diferentes limites de nós ou folhas (ex: 256, 128, 64, 32, 16, 8, 4), gere binários dedicados no `code/Makefile` passando `-DHT_MAX_ACTIVE_LEAVES=<L> -DHT_MAX_NODES=<N>`.
   - **Jamais modifique `code/include/` ou `code/src/`** para criar limitadores de software em runtime; deixe que o tamanho físico do pool compilado governe o limite e meça o `sizeof(ht_tree_t)` real de cada binário.

5. **Commit Atômico:**
   - Faça commits dedicados para scripts e relatórios de benchmark usando o prefixo `bench: ...`.
