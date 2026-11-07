# River Hoeffding Tree (`references/river_tree/`) — Mapa Arquitetural

Este documento mapeia todas as decisões algorítmicas, fórmulas matemáticas, hiperparâmetros e casos de borda da implementação do River em [`references/river_tree/`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree) para consulta rápida durante o desenvolvimento em C11.

---

## 1. Índice Rápido de Busca (Onde Encontrar Cada Conceito)

| Conceito / Decisão Algorítmica | Arquivo e Linhas Exatas | Detalhe Crítico para Paridade em C11 |
| :--- | :--- | :--- |
| Classe base `HoeffdingTree` e limite de Hoeffding ($\epsilon$) | [`hoeffding_tree.py#L24-L112`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree.py#L24-L112) | $\epsilon = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$ (`_hoeffding_bound`, `#L84-L112`) |
| Gerenciamento de memória VFDT (`_enforce_size_limit`) | [`hoeffding_tree.py#L240-L331`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree.py#L240-L331) | Ordena folhas por `calculate_promise()` e desativa as menos promissoras |
| Fluxo principal de treino (`HoeffdingTreeClassifier.learn_one`) | [`hoeffding_tree_classifier.py#L474-L562`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree_classifier.py#L474-L562) | Testa `grace_period` via `weight_seen - last_split_attempt_at >= grace_period` |
| Fluxo de tentativa de split (`_attempt_to_split`) | [`hoeffding_tree_classifier.py#L401-L472`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree_classifier.py#L401-L472) | $\Delta G > \epsilon$ ou $\epsilon < \tau$; inicializa filhos com `children_stats` pós-split |
| Fluxo de inferência (`predict_proba_one`) | [`hoeffding_tree_classifier.py#L563-L580`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree_classifier.py#L563-L580) | `traverse(x, until_leaf=True)` usa `most_common_path()` se faltar atributo |
| Roteamento binário numérico e categórico | [`nodes/branch.py#L47-L120`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/branch.py#L47-L120) | Numérico: `x[f] <= threshold` vai para `0` (left); Nominal: `x[f] == val` vai para `0` |
| Folha base `HTLeaf` (ativação/desativação de observadores) | [`nodes/leaf.py#L11-L191`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/leaf.py#L11-L191) | `deactivate()` libera `self.splitters = None` mantendo `self.stats` |
| `LeafMajorityClass`: pureza, `max_share_to_split` e `calculate_promise` | [`nodes/htc_nodes.py#L11-L105`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py#L11-L105) | `promise = total_seen - max(stats.values())` (`#L63-L78`) |
| `LeafNaiveBayes` e `LeafNaiveBayesAdaptive` (NBA) | [`nodes/htc_nodes.py#L107-L227`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py#L107-L227) | NBA testa MC e NB **antes** de atualizar stats (`#L167-L193`); `disable_attribute` é no-op! |
| Cálculo de Naive Bayes com *Log-Sum-Exp* (`do_naive_bayes_prediction`) | [`utils.py#L39-L103`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/utils.py#L39-L103) | Soma `log(P(c))` + `cond_log_proba(x_i, c)` e normaliza via `max_ll + log(sum(exp(...)))` |
| Observador Numérico `GaussianSplitter` (Welford + 10 cortes + CDF) | [`splitter/gaussian_splitter.py#L12-L103`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/gaussian_splitter.py#L12-L103) | `bin_size = (max - min) / (n_splits + 1)`; usa `min_per_class`/`max_per_class` + CDF |
| Observador Categórico `NominalSplitterClassif` | [`splitter/nominal_splitter_classif.py#L9-L110`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/nominal_splitter_classif.py#L9-L110) | Avalia cada categoria `att_val` em split binário *one-vs-rest* (`#L68-L79`, `#L97-L110`) |
| Critério `GiniSplitCriterion` | [`split_criterion/gini_split_criterion.py#L8-L61`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/gini_split_criterion.py#L8-L61) | `merit = 1.0 - post_gini`; `range_of_merit = 1.0` |
| Critério `InfoGainSplitCriterion` | [`split_criterion/info_gain_split_criterion.py#L8-L80`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/info_gain_split_criterion.py#L8-L80) | `merit = H(pre) - H(post)`; `range_of_merit = log2(max(len(pre_dist), 2))` |
| Filtro `min_branch_fraction` (`num_subsets_greater_than_frac`) | [`split_criterion/info_gain_split_criterion.py#L67-L79`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/info_gain_split_criterion.py#L67-L79) | Exige pelo menos 2 ramos com `W_i / W > min_branch_fraction` (senão retorna `-inf`) |
| Concept Drift: `HoeffdingAdaptiveTreeClassifier` (HAT) | [`hoeffding_adaptive_tree_classifier.py#L22-L287`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_adaptive_tree_classifier.py#L22-L287) | Treino recursivo nos nós internos + votação ponderada por $1/e^2$ |
| Nós Adaptativos `AdaLeafClassifier` e `AdaBranchClassifier` | [`nodes/hatc_nodes.py#L22-L318`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/hatc_nodes.py#L22-L318) | Monitora erro com `ADWIN`, cria `_alternate_tree` e troca quando `bound < old_err - alt_err` |
| Detector de Drift `ADWIN` | [`drift/adwin.py#L7-L138`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/drift/adwin.py#L7-L138) | Histograma exponencial (`max_buckets=5`, `clock=32`, `delta=0.002`) |

---

## 2. Detalhamento das Decisões Críticas e Casos de Borda

### 2.1 Ciclo de Split e Pre-Pruning (`hoeffding_tree_classifier.py`)
1. **Quando tentar o split ([`#L525-L536`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree_classifier.py#L525-L536)):**
   - Somente quando a folha está ativa (`is_active()`) e `weight_seen - last_split_attempt_at >= grace_period` (padrão `200`).
2. **Filtros antes de avaliar observadores:**
   - **Pureza ([`nodes/htc_nodes.py#L80-L94`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py#L80-L94)):** Se menos de 2 classes têm peso $> 0$, não tenta split.
   - **`max_share_to_split` ([`nodes/htc_nodes.py#L54-L61`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py#L54-L61)):** Se $\max_c(w_c) / W > \text{max\_share\_to\_split}$ (padrão `0.99`), aborta a avaliação dos splitters.
3. **Comparação dos 2 melhores candidatos ([`#L416-L432`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree_classifier.py#L416-L432)):**
   - Se `merit_preprune=True`, inclui uma sugestão nula (`feature = None`, `merit = -inf` em `BranchFactory`, [`nodes/leaf.py#L127-L130`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/leaf.py#L127-L130)).
   - Ordena os candidatos por `merit` crescente e compara `best` (`[-1]`) vs `second_best` (`[-2]`).
   - Faz split se `best.merit - second_best.merit > hoeffding_bound` **ou** `hoeffding_bound < tau` (padrão `0.05`).
   - Se o vencedor for o split nulo (`best.feature is None`), a folha é **desativada** ([`#L446-L450`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree_classifier.py#L446-L450)).
4. **Inicialização das folhas filhas ([`#L455-L458`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/hoeffding_tree_classifier.py#L455-L458)):**
   - As duas folhas filhas recém-criadas **não começam zeradas**: elas recebem `children_stats` (`lhs_dist` e `rhs_dist` estimados pelo splitter) como sua distribuição inicial de classes `stats`!

### 2.2 `GaussianSplitter` (`splitter/gaussian_splitter.py`)
- **Sugestão de Pontos de Corte ([`#L74-L90`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/gaussian_splitter.py#L74-L90)):**
  - Encontra `min_value = min(_min_per_class.values())` e `max_value = max(_max_per_class.values())`.
  - Divide o intervalo em `n_splits + 1` partes iguais (`bin_size = (max_value - min_value) / (n_splits + 1.0)`).
  - Gera até `n_splits` (padrão `10`) pontos: `split_value = min_value + bin_size * (i + 1)` para $i \in \{0, \dots, 9\}$, exigindo `min_value < split_value < max_value`.
- **Divisão dos Pesos por Classe via CDF ([`#L92-L103`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/gaussian_splitter.py#L92-L103)):**
  Para cada classe $k$:
  - Se `split_value < _min_per_class[k]`: todo o peso $N_k$ vai para a direita (`rhs_dist[k] = N_k`).
  - Se `split_value >= _max_per_class[k]`: todo o peso $N_k$ vai para a esquerda (`lhs_dist[k] = N_k`).
  - Caso contrário: usa a CDF da normal $\mathcal{N}(\mu_k, \sigma_k^2)$:
    $$\text{lhs\_dist}[k] = \text{CDF}(\text{split\_value}) \cdot N_k, \quad \text{rhs\_dist}[k] = N_k - \text{lhs\_dist}[k]$$

### 2.3 Naive Bayes Adaptive (`nodes/htc_nodes.py` e `utils.py`)
- Em [`LeafNaiveBayesAdaptive.learn_one`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py#L167-L193), sempre que uma nova amostra $(x, y, w)$ chega numa folha ativa, **antes** de atualizar os contadores:
  1. Faz a predição Majority Class (`mc_pred`). Se acertar `y` (ou se `stats` estiver vazio), soma `_mc_correct_weight += w`.
  2. Faz a predição Naive Bayes (`nb_pred`). Se acertar `y`, soma `_nb_correct_weight += w`.
  3. Só depois atualiza `stats` e `splitters`.
- Na hora de prever (`prediction`, [`#L195-L214`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py#L195-L214)): usa Naive Bayes se `is_active() and _nb_correct_weight >= _mc_correct_weight`, caso contrário usa Majority Class.
- **Pegadinha importante ([`#L132-L143`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py#L132-L143), [`#L215-L226`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py#L215-L226)):** Em folhas `NB` e `NBA`, `disable_attribute` é um **no-op** (nunca remove atributos ruins), pois o Naive Bayes precisa de todos os atributos para estimar a probabilidade conjunta.
