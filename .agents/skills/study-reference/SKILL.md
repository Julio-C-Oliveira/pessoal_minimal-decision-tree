---
name: study-reference
description: >-
  Use this skill when studying research papers in references/papers/, analyzing
  or looking up architectural decisions in references/river_tree/ or
  references/sklearn_tree/, or writing technical study notes in docs/.
---

# Study & Reference Mapping Skill (`study-reference`)

Esta skill guia o estudo teórico dos artigos científicos e disponibiliza **Mapas Arquiteturais Indexados (Lookup Maps)** das implementações do **River** e do **Scikit-Learn**.

## 1. Índices de Busca Rápida (Progressive Disclosure Nível 2)

Antes de fazer buscas manuais ou ler arquivos inteiros em `references/river_tree/` ou `references/sklearn_tree/`, **consulte primeiro os mapas arquiteturais já indexados com arquivos e linhas exatas**:

- **[Mapa Arquitetural do River (`river_architecture_map.md`)](./references/river_architecture_map.md)**
  - Contém a localização exata (`#L<início>-L<fim>`), fórmulas e casos de borda de:
    - `HoeffdingTree` / `HoeffdingTreeClassifier` (`learn_one`, `_attempt_to_split`, `_hoeffding_bound`, `_enforce_size_limit`, `calculate_promise`)
    - Nós (`DTBranch`, `NumericBinaryBranch`, `NominalBinaryBranch`, `LeafMajorityClass`, `LeafNaiveBayes`, `LeafNaiveBayesAdaptive`)
    - Observadores (`GaussianSplitter` com 10 cortes e CDF, `NominalSplitterClassif` e comparação com EBST/QO/Histogram)
    - Critérios (`GiniSplitCriterion`, `InfoGainSplitCriterion`, `min_branch_fraction`)
    - Concept Drift (`HoeffdingAdaptiveTreeClassifier`, `AdaBranchClassifier`, `AdaLeafClassifier`, `ADWIN`)
- **[Mapa Arquitetural do Scikit-Learn (`sklearn_architecture_map.md`)](./references/sklearn_architecture_map.md)**
  - Contém a localização exata (`#L<início>-L<fim>`) e decisões em Cython/C de:
    - Array plano `Node* nodes` sem ponteiros e separação do buffer `float64_t* value` (`_tree.pxd`, `_utils.pxd`)
    - Roteamento iterativo sem recursão (`_apply_dense`) e predicado inline `goes_left` (incluindo `NaN`)
    - Serialização binária (`__getstate__` / `__setstate__`)
    - Arquitetura modular de `Criterion` (`Gini` com uma única divisão, `Entropy` base 2, `proxy_impurity_improvement`)
    - Struct `SplitRecord` alocada na stack e comparação entre o `DensePartitioner` batch ($O(N)$ RAM) e nossos observadores online ($O(1)$ RAM).

---

## 2. Procedimento para Escrever Documentação em `docs/`

1. **Consultar os Mapas Indexados Acima:**
   - Leia [river_architecture_map.md](./references/river_architecture_map.md) e/ou [sklearn_architecture_map.md](./references/sklearn_architecture_map.md) para localizar imediatamente as linhas relevantes.
   - Abra apenas as faixas de linhas específicas nos arquivos de `references/` caso precise inspecionar o código literal, ou delegue ao subagente `research` se precisar ler os PDFs em `references/papers/`.

2. **Formato Obrigatório dos Documentos em `docs/`:**
   Todo documento gerado em `docs/` (`01_hoeffding_tree_vfdt.md`, `02_attribute_observers.md`, `03_concept_drift_hat.md`, `04_river_mapping.md`, `05_sklearn_architecture.md`) deve conter:
   - **Fundamentação Matemática:** Equações em LaTeX com definição clara de cada variável.
   - **Casos de Borda Numéricos:** Variância $= 0$, `min == max`, amostra com `NaN`, folha com 1 única classe observada, `min_branch_fraction` ou soma de pesos zero.
   - **Mapeamento para Nossa Arquitetura C11 ($\le 1\text{ MB}$):**
     - Quais structs C (`ht_node_t`, `ht_leaf_stats_t`, `ht_gaussian_obs_t`) armazenam esses dados.
     - Custo exato em bytes (`sizeof`) usando `float` (32-bit) vs `double` (64-bit).

3. **Commit Atômico:**
   - Ao concluir um documento em `docs/`, faça um commit dedicado com prefixo `docs: ...`.
