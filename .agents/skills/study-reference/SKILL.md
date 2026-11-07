---
name: study-reference
description: >-
  Use this skill when studying research papers in references/papers/, analyzing
  the reference implementations in references/river_tree/ or references/sklearn_tree/,
  or writing technical architecture and theoretical study notes in docs/.
---

# Study & Reference Mapping Skill (`study-reference`)

Esta skill guia o estudo teórico dos artigos científicos e o mapeamento das implementações de referência (`river` e `scikit-learn`) para a documentação técnica do projeto em `docs/`.

## Quando Usar
- Ao preencher ou atualizar qualquer documento em `docs/` (`01_hoeffding_tree_vfdt.md`, `02_attribute_observers.md`, `03_concept_drift_hat.md`, `04_river_mapping.md`, `05_sklearn_architecture.md`).
- Ao tirar dúvidas sobre detalhes matemáticos ou casos de borda do algoritmo VFDT, Gaussian Splitter, Naive Bayes Adaptive ou ADWIN.

## Procedimento Passo a Passo

1. **Delegar Leitura Extensa ao Subagente `research` (Se Necessário):**
   - Se a tarefa exigir cruzar mais de 2 arquivos grandes em `references/river_tree/` ou `references/sklearn_tree/`, ou ler páginas de PDFs em `references/papers/`, invoque o subagente `research` pedindo um resumo exato das fórmulas, hiperparâmetros e estruturas de dados.

2. **Cruzar as 3 Fontes Locais:**
   - **Teoria (`references/papers/`):** Identifique a definição matemática original (ex: limite de Hoeffding $\epsilon$, condição de *tie-breaking* $\tau$, desativação de folhas $p_l \cdot e_l$).
   - **Algoritmo Online (`references/river_tree/`):** Verifique como o River resolve casos práticos de ponto flutuante e streaming:
     - `hoeffding_tree_classifier.py` e `hoeffding_tree.py`: fluxo de `learn_one`, `grace_period`, `max_share_to_split`, `min_branch_fraction` e gerenciamento de memória.
     - `splitter/gaussian_splitter.py` e `splitter/nominal_splitter_classif.py`: cálculo dos pontos de corte candidatos (`n_splits=10`), uso de `min_per_class`/`max_per_class` e estimativa das distribuições pós-split (`lhs_dist`, `rhs_dist`) via CDF Gaussiana.
     - `split_criterion/gini_split_criterion.py` e `info_gain_split_criterion.py`: cálculo do mérito, range $R$ (`range_of_merit`) e casos em que o split é rejeitado.
     - `nodes/htc_nodes.py`: como `LeafNaiveBayesAdaptive` rastreia o erro prequential de Majority Class vs Naive Bayes.
   - **Estrutura em Baixo Nível (`references/sklearn_tree/`):** Verifique em `_tree.pxd`, `_utils.pxd`, `_criterion.pxd` e `_splitter.pxd` como organizar arrays planos de nós sem ponteiros e separar `Node` (topologia) de `value` (contadores de classe).

3. **Formato Obrigatório dos Documentos em `docs/`:**
   Todo documento gerado em `docs/` deve conter:
   - **Fundamentação Matemática:** Equações em LaTeX com definição clara de cada variável.
   - **Casos de Borda Numéricos:** O que acontece quando variância $= 0$, `min == max`, amostra com `NaN`, folha com 1 única classe observada, ou soma de pesos zero.
   - **Mapeamento para Nossa Arquitetura C11 ($\le 1\text{ MB}$):**
     - Quais structs C (`ht_node_t`, `ht_leaf_stats_t`, `ht_gaussian_obs_t`) armazenam esses dados.
     - Custo exato em bytes (`sizeof`) usando `float` (32-bit) vs `double` (64-bit).

4. **Commit Atômico:**
   - Ao concluir um documento em `docs/`, faça um commit dedicado com prefixo `docs: ...`.
