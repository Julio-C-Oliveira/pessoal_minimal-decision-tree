# 02 — Observadores de Atributos, Critérios de Impureza e Folhas NBA (Bifet, 2009)

Este documento detalha o tratamento incremental de atributos numéricos e categóricos em fluxos contínuos, os critérios de impureza (*Gini* e *Information Gain*) e as estratégias de predição nas folhas (*Majority Class*, *Naive Bayes* e *Naive Bayes Adaptive*), fundamentados em Bifet (2009) e na implementação de referência do River ([`references/river_tree/`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree)).

---

## 1. Fundamentação Matemática

### 1.1 Observadores de Atributos Numéricos: Comparação e Escolha Arquitetural

Em atributos contínuos ($x_i \in \mathbb{R}$), não é possível manter contadores exatos para cada valor distinto sem que o consumo de memória cresça linearmente com o número de amostras $O(n)$. Na literatura (Bifet, 2009) e em [`references/river_tree/splitter/`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter), destacam-se três abordagens:

| Observador Numérico | Complexidade de RAM por (Atributo, Folha) | Suporte a PDF para Naive Bayes | Viabilidade Embarcada ($\le 1\text{ MB}$) |
| :--- | :--- | :---: | :--- |
| **EBST** (`ebst_splitter.py` / `exhaustive_splitter.py`) | $O(n)$ (árvore binária com cada valor único visto) | Não ($P(x_i \mid c) = 0$) | **Inviável** (estoura a RAM rapidamente em streams contínuos) |
| **Quantis / Histogramas** (`qo_splitter.py` / `histogram_splitter.py`) | $O(B \cdot C)$ (ex: $B = 256$ bins ou hash dinâmico) | Aproximado por bin | **Pesado** ($256 \times 4 \times 8\text{ B} \approx 8\text{ KB}$ por atributo!) |
| **Aproximação Gaussiana** (`gaussian_splitter.py`) | **$O(C)$ estrito** (5 escalares por classe via Welford) | **Sim** (PDF Normal analítica) | **Ideal** ($20\text{ B}$ por classe em `float` 32-bit) |

Portanto, nossa implementação em C11 adota a **Aproximação Gaussiana Incremental via Algoritmo de Welford** ([`gaussian_splitter.py#L12-L103`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/gaussian_splitter.py#L12-L103)), que modela a distribuição condicional de cada atributo numérico $i$ dada a classe $c$ como uma normal $\mathcal{N}(\mu_{i,c}, \sigma_{i,c}^2)$ mantendo também os limites observados $[\min_{i,c}, \max_{i,c}]$.

---

### 1.2 Algoritmo Incremental de Welford e Avaliação de Splits via CDF

Para cada par (atributo numérico $i$, classe $c$) em uma folha ativa, armazenamos 5 escalares: peso acumulado $n_c$, média $\mu_c$, soma dos quadrados das diferenças centradas $M_{2,c}$, valor mínimo $\min_c$ e valor máximo $\max_c$.

#### 1. Atualização Incremental ($O(1)$ por amostra):
Ao observar um valor válido $x \in \mathbb{R}$ da classe $c$ com peso $w > 0$:
- Se $n_c = 0$ (primeira observação da classe $c$ neste atributo):
  $$n_c \leftarrow w, \quad \mu_c \leftarrow x, \quad M_{2,c} \leftarrow 0, \quad \min_c \leftarrow x, \quad \max_c \leftarrow x$$
- Se $n_c > 0$:
  $$\begin{aligned}
  n_c &\leftarrow n_c + w \\
  \delta_1 &= x - \mu_c \\
  \mu_c &\leftarrow \mu_c + \frac{w}{n_c} \delta_1 \\
  \delta_2 &= x - \mu_c \\
  M_{2,c} &\leftarrow M_{2,c} + w \cdot \delta_1 \cdot \delta_2 \\
  \min_c &\leftarrow \min(\min_c, x), \qquad \max_c \leftarrow \max(\max_c, x)
  \end{aligned}$$
- Variância amostral (para $n_c > 1$):
  $$s_c^2 = \frac{M_{2,c}}{n_c - 1}, \qquad \sigma_c = \sqrt{s_c^2}$$

#### 2. Geração dos $k$ Pontos de Corte Candidatos (`_split_point_suggestions`):
Dados os limites por classe na folha, determinamos o intervalo global observado no atributo ([`gaussian_splitter.py#L74-L90`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/gaussian_splitter.py#L74-L90)):

$$x_{\min} = \min_{c: n_c > 0} (\min_c), \qquad x_{\max} = \max_{c: n_c > 0} (\max_c), \qquad \Delta_{\text{bin}} = \frac{x_{\max} - x_{\min}}{k + 1}$$

onde $k = \text{n\_splits}$ (padrão $k = 10$ no River). Isso gera até $k$ candidatos igualmente espaçados:

$$t_j = x_{\min} + j \cdot \Delta_{\text{bin}}, \quad \text{para } j \in \{1, 2, \dots, k\} \text{ tais que } x_{\min} < t_j < x_{\max}$$

#### 3. Estimativa das Distribuições Pós-Split ($D_L, D_R$) via CDF Normal:
Para cada threshold candidato $t_j$ e cada classe observada $c$ (com peso total $n_c$ no atributo), estimamos os pesos que iriam para o filho esquerdo ($x_i \le t_j$) e para o filho direito ($x_i > t_j$) ([`gaussian_splitter.py#L92-L103`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/gaussian_splitter.py#L92-L103)):

$$\begin{aligned}
W_{L,c}(t_j) &= \begin{cases}
0, & \text{se } t_j < \min_c \\
n_c, & \text{se } t_j \ge \max_c \\
\Phi\left(\frac{t_j - \mu_c}{\sigma_c}\right) \cdot n_c, & \text{se } \min_c \le t_j < \max_c
\end{cases} \\[6pt]
W_{R,c}(t_j) &= n_c - W_{L,c}(t_j)
\end{aligned}$$

onde $\Phi(z)$ é a Função de Distribuição Acumulada (CDF) da normal padrão, calculada em C11 via `erff` / `erf` de `<math.h>`:

$$\Phi(z) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)\right]$$

---

### 1.3 Observador de Atributos Categóricos Nominais (*One-vs-Rest*)

Para manter a topologia da árvore **estritamente binária** (todos os nós com apenas `left_child` e `right_child`), atributos categóricos nominais com valores discretos $v \in \{0, 1, \dots, V-1\}$ ($V \le \text{HT\_MAX\_CATEGORIES}$) são avaliados via splits binários do tipo **igualdade vs resto** (*one-vs-rest*, [`nominal_splitter_classif.py#L68-L79, #L97-L110`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/nominal_splitter_classif.py#L68-L79)):

- O observador mantém uma matriz de frequências $N_{v, c}$ (peso acumulado da categoria $v$ na classe $c$) e o total por classe no atributo $N_{\bullet, c} = \sum_{u=0}^{V-1} N_{u, c}$.
- Para cada categoria observada $v$, avalia-se a partição binária:
  - **Filho Esquerdo ($x_i == v$):** $W_{L,c}(v) = N_{v, c}$
  - **Filho Direito ($x_i \neq v$):** $W_{R,c}(v) = N_{\bullet, c} - N_{v, c}$

---

### 1.4 Critérios de Impureza: Gini e Information Gain

Seja $D_{\text{pre}} = \{w_0, \dots, w_{C-1}\}$ a distribuição de classes pré-split (com soma $W = \sum_c w_c$) e sejam $D_L = \{W_{L,0}, \dots, W_{L,C-1}\}$ e $D_R = \{W_{R,0}, \dots, W_{R,C-1}\}$ as distribuições pós-split (com somas $W_L = \sum_c W_{L,c}$, $W_R = \sum_c W_{R,c}$ e $W_{\text{post}} = W_L + W_R$).

#### 1. Filtro de Fração Mínima de Ramo (`min_branch_fraction`):
Antes de computar qualquer impureza, verifica-se se ambos os ramos filhos possuem pelo menos uma fração mínima `min_branch_fraction` (padrão `0.01`, [`info_gain_split_criterion.py#L67-L79`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/info_gain_split_criterion.py#L67-L79)) do peso total $W_{\text{post}}$:

$$\text{Se } \frac{W_L}{W_{\text{post}}} \le \text{min\_branch\_fraction} \quad \lor \quad \frac{W_R}{W_{\text{post}}} \le \text{min\_branch\_fraction} \implies \text{Merit} = -\infty$$

#### 2. Critério de Gini (`GiniSplitCriterion`, [`gini_split_criterion.py#L15-L58`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/gini_split_criterion.py#L15-L58)):
A impureza de Gini de um ramo $D_b$ ($b \in \{L, R\}$) é:

$$\operatorname{Gini}(D_b) = 1 - \sum_{c=0}^{C-1} \left(\frac{W_{b,c}}{W_b}\right)^2$$

No River, o mérito de Gini é definido como $1.0$ menos a impureza pós-split ponderada (de modo que maior mérito = menor impureza pós-split, com amplitude $R = 1.0$):

$$\operatorname{Merit}_{\text{Gini}}(D_L, D_R) = 1.0 - \left(\frac{W_L}{W_{\text{post}}} \operatorname{Gini}(D_L) + \frac{W_R}{W_{\text{post}}} \operatorname{Gini}(D_R)\right), \qquad R_{\text{Gini}} = 1.0$$

#### 3. Critério de Ganho de Informação (`InfoGainSplitCriterion`, [`info_gain_split_criterion.py#L15-L65`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/info_gain_split_criterion.py#L15-L65)):
A entropia de Shannon na base 2 de uma distribuição $D$ com peso total $W$ é calculada usando a identidade algébrica do River ([`#L46-L53`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/split_criterion/info_gain_split_criterion.py#L46-L53)), que realiza **apenas uma divisão ao final**:

$$H(D) = -\sum_{c: w_c > 0} \frac{w_c}{W} \log_2\left(\frac{w_c}{W}\right) = \frac{W \log_2 W - \sum_{c: w_c > 0} w_c \log_2 w_c}{W}$$

O mérito de Ganho de Informação e sua amplitude $R_{\text{IG}}$ são:

$$\operatorname{Merit}_{\text{IG}}(D_{\text{pre}}, D_L, D_R) = H(D_{\text{pre}}) - \left(\frac{W_L}{W_{\text{post}}} H(D_L) + \frac{W_R}{W_{\text{post}}} H(D_R)\right), \qquad R_{\text{IG}} = \log_2\big(\max(C_{\text{obs}}, 2)\big)$$

onde $C_{\text{obs}} = |\{c : w_c > 0\}|$ é o número de classes com peso positivo na folha antes do split.

---

### 1.5 Estratégias de Predição nas Folhas: MC, NB e NBA

As folhas ativas já armazenam $w_c$ e os observadores $P(x_i \mid y = c)$ para avaliar candidatos de split. Bifet (2009) e Holmes et al. (2005) aproveitam essas estatísticas para oferecer três estratégias de predição com custo de RAM quase zero ([`nodes/htc_nodes.py`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py) e [`utils.py#L39-L103`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/utils.py#L39-L103)):

#### 1. Majority Class (`HT_LEAF_MC`):
Retorna a distribuição empírica de classes da folha:

$$\hat{P}_{\text{MC}}(y = c \mid \mathbf{x}) = \frac{w_c}{W}, \qquad \hat{y}_{\text{MC}} = \arg\max_{c} w_c$$

#### 2. Naive Bayes (`HT_LEAF_NB`) com *Log-Sum-Exp Trick*:
Se a folha está ativa e $W \ge \text{nb\_threshold}$ (padrão `0`), calcula a log-verossimilhança conjunta $\ell_c(\mathbf{x})$ em base natural ($\ln$) para cada classe observada $c$ ($w_c > 0$):

$$\ell_c(\mathbf{x}) = \ln\left(\frac{w_c}{W}\right) + \sum_{i=1}^{d} \ln P(x_i \mid y = c)$$

onde a densidade condicional de cada atributo válido $x_i$ é:
- **Atributo Numérico (Gaussiana):**
  $$\ln P(x_i \mid y = c) = -\frac{1}{2}\ln(2\pi s_{i,c}^2) - \frac{(x_i - \mu_{i,c})^2}{2 s_{i,c}^2}$$
- **Atributo Categórico ($x_i = v$):**
  $$P(x_i = v \mid y = c) = \frac{N_{i, v, c}}{w_c}$$

Para converter $\ell_c(\mathbf{x})$ em probabilidades normalizadas sem sofrer *underflow* ou *overflow* em ponto flutuante de 32 bits, aplica-se o **Log-Sum-Exp (LSE) trick** ([`utils.py#L96-L102`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/utils.py#L96-L102)):

$$\ell_{\max} = \max_{c: w_c > 0} \ell_c(\mathbf{x}), \qquad \operatorname{LSE} = \ell_{\max} + \ln \left(\sum_{c: w_c > 0} \exp\big(\ell_c(\mathbf{x}) - \ell_{\max}\big)\right)$$

$$\hat{P}_{\text{NB}}(y = c \mid \mathbf{x}) = \exp\big(\ell_c(\mathbf{x}) - \operatorname{LSE}\big)$$

#### 3. Naive Bayes Adaptive (`HT_LEAF_NBA`, Bifet 2009):
O Naive Bayes converge mais rápido que Majority Class quando a folha tem poucas amostras, mas pode saturar com erro maior em folhas com muitos exemplos se a hipótese de independência condicional for violada. O **Naive Bayes Adaptive (NBA)** monitora *online* o desempenho *prequential* de ambos na própria folha ([`htc_nodes.py#L146-L226`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py#L146-L226)):
- Cada folha ativa mantém dois contadores de acertos: `mc_correct_weight` e `nb_correct_weight` (inicializados em `0.0`).
- **Ordem Estrita no `learn_one(x, y, w)` ([`#L167-L193`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/nodes/htc_nodes.py#L167-L193)):** Quando uma amostra chega a uma folha ativa, **ANTES** de atualizar `class_counts` e os observadores:
  1. Avalia $\hat{y}_{\text{MC}} = \arg\max_c \hat{P}_{\text{MC}}(c \mid \mathbf{x})$. Se a folha ainda estava vazia ou se $\hat{y}_{\text{MC}} == y$, incrementa `mc_correct_weight += w`.
  2. Avalia $\hat{y}_{\text{NB}} = \arg\max_c \hat{P}_{\text{NB}}(c \mid \mathbf{x})$. Se $\hat{y}_{\text{NB}} == y$, incrementa `nb_correct_weight += w`.
  3. Só então atualiza `class_counts[y] += w` e os observadores de atributos.
- **Regra de Predição (`#L210-L213`):**
  $$\hat{P}_{\text{NBA}}(y \mid \mathbf{x}) = \begin{cases}
  \hat{P}_{\text{NB}}(y \mid \mathbf{x}), & \text{se folha ativa } \land \text{ nb\_correct\_weight} \ge \text{mc\_correct\_weight} \\
  \hat{P}_{\text{MC}}(y \mid \mathbf{x}), & \text{caso contrário}
  \end{cases}$$

---

## 2. Casos de Borda Numéricos

1. **Variância Zero ou Amostra Única em Welford ($n_c < 2$ ou $s_c^2 \le 0$):**
   - *No cálculo de split (CDF):* Observe que em [`gaussian_splitter.py#L94-L101`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/gaussian_splitter.py#L94-L101), os testes `if split_value < min_c` e `elif split_value >= max_c` vêm **antes** da chamada à CDF. Se todas as amostras da classe $c$ tiveram exatamente o mesmo valor ($x_1 = x_2 = \dots \implies \min_c == \max_c$ e $s_c^2 = 0$), todo `split_value` satisfaz obrigatoriamente $t_j < \min_c$ ou $t_j \ge \max_c$, nunca entrando no ramo `else` da CDF! Ainda assim, caso $s_c^2 \le 10^{-12}$ por arredondamento de ponto flutuante com $\min_c < \max_c$, se $t_j < \mu_c$ retorna $\Phi = 0.0$, se $t_j > \mu_c$ retorna $\Phi = 1.0$, e se $t_j == \mu_c$ retorna $\Phi = 0.5$.
   - *No cálculo do Naive Bayes (`log_pdf`):* Se $n_c < 2$ ou $s_c^2 \le 0$, no River `Gaussian.pdf(x)` retorna `0.0` $\implies \ln(0) = -\infty$, e `do_naive_bayes_prediction` ([`utils.py#L89-L90`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/utils.py#L89-L90)) ignora termos onde `log_p <= -INFINITY` (`if log_p > -math.inf: log_likelihood[label] += log_p`). Nossa função C11 seguirá exatamente essa guarda.
2. **`min_value == max_value` Global em um Atributo Numérico:**
   - Se todas as classes observadas na folha viram apenas um único valor constante para o atributo $i$ ($x_{\min} == x_{\max}$), então $\Delta_{\text{bin}} = 0$ e $t_j = x_{\min}$. O filtro estrito `x_min < t_j < x_max` ([`gaussian_splitter.py#L88`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/splitter/gaussian_splitter.py#L88)) descarta todos os candidatos, retornando `merit = -INFINITY`.
3. **Amostra com `NaN` ou Categoria Inválida ($< 0$ ou $\ge \text{HT\_MAX\_CATEGORIES}$):**
   - *No `learn_one` da folha:* O contador da classe `node->class_counts[y]` e os observadores dos demais atributos válidos são atualizados normalmente; apenas o observador do atributo $i$ ausente é ignorado (`if (HT_ISNAN(x[i].num)) continue;`).
   - *No Naive Bayes (`do_naive_bayes_prediction`):* Atributos ausentes na amostra de consulta simplesmente contribuem com $0.0$ para a soma de log-verossimilhanças $\ell_c(\mathbf{x})$.
4. **Probabilidade Condicional Zero em Atributo Categórico no Naive Bayes:**
   - Se uma classe $c$ já foi vista na folha ($w_c > 0$), mas a categoria $v$ nunca ocorreu na classe $c$ ($N_{i, v, c} = 0$), `cond_proba` retorna `0.0`. No River ([`utils.py#L91-L93`](file:///home/julio/Documentos/pessoal_minimal-decision-tree/references/river_tree/utils.py#L91-L93)), `math.log(tmp) if tmp > 0 else 0.0` soma `0.0` ao invés de $-\infty$, evitando zerar toda a probabilidade da classe por causa de uma única categoria inédita.
5. **Underflow/Overflow no *Log-Sum-Exp* em `float` 32-bit:**
   - Ao subtrair $\ell_{\max}$ antes de chamar `HT_EXP(ell_c - ell_max)`, o maior argumento de `expf` é exatamente `0.0` ($\exp(0) = 1.0$), garantindo que a soma dentro do logaritmo pertença sempre ao intervalo $[1.0, C]$ — eliminando 100% de risco de *overflow* ou *underflow* do denominador mesmo com dezenas de atributos.

---

## 3. Mapeamento para Nossa Arquitetura C11 ($\le 1\text{ MB}$)

### 3.1 Estruturas C11 dos Observadores e Estado da Folha (`ht_leaf_stats_t`)

Não usamos *vtables* ou ponteiros por atributo; cada atributo em uma folha ativa é representado por uma `union` compacta selecionada pelo schema estático `tree->feature_types[i]`:

```c
/* Observador Gaussiano (Welford) para 1 par (atributo, classe) */
typedef struct {
    ht_float_t count;      /* Peso acumulado n_c */
    ht_float_t mean;       /* Média incremental \mu_c */
    ht_float_t m2;         /* Soma de quadrados centrados M_{2,c} */
    ht_float_t min_val;    /* Mínimo observado na classe c */
    ht_float_t max_val;    /* Máximo observado na classe c */
} ht_gaussian_obs_t;

/* Observador Categórico Nominal para 1 atributo (todas as classes) */
typedef struct {
    ht_float_t counts[HT_MAX_CLASSES][HT_MAX_CATEGORIES];
} ht_categorical_obs_t;

/* Union sem vtable por atributo */
typedef union {
    ht_gaussian_obs_t    num[HT_MAX_CLASSES];
    ht_categorical_obs_t cat;
} ht_feature_obs_t;

/* Estado pesado de aprendizado alocado apenas para folhas ATIVAS no stats_pool */
typedef struct {
    ht_float_t       last_split_attempt_at;
    ht_float_t       mc_correct_weight;
    ht_float_t       nb_correct_weight;
    ht_feature_obs_t observers[HT_MAX_FEATURES];
} ht_leaf_stats_t;
```

### 3.2 Custo Exato em Bytes (`sizeof`) — `float` (32-bit) vs `double` (64-bit)

Considerando a configuração padrão com $F = \text{HT\_MAX\_FEATURES} = 16$, $C = \text{HT\_MAX\_CLASSES} = 4$ e $\text{HT\_MAX\_CATEGORIES} = 5$ (de modo que o observador categórico tem $4 \times 5 = 20$ floats, empatando exatamente em tamanho com os $4 \times 5 = 20$ floats do observador Gaussiano!):

| Estrutura C11 | Cálculo de Campos ($F=16, C=4$) | `float` (32-bit) | `double` (64-bit, `-DHT_USE_DOUBLE`) |
| :--- | :--- | :---: | :---: |
| `ht_gaussian_obs_t` (1 classe) | $5 \times \text{sizeof(ht\_float\_t)}$ | **$20\text{ B}$** | **$40\text{ B}$** |
| `ht_feature_obs_t` (1 atributo, 4 classes) | $4 \times \text{sizeof(ht\_gaussian\_obs\_t)}$ | **$80\text{ B}$** | **$160\text{ B}$** |
| Metadados de `ht_leaf_stats_t` | $3 \times \text{sizeof(ht\_float\_t)}$ (`last_split`, `mc`, `nb`) | **$12\text{ B}$** | **$24\text{ B}$** |
| **`sizeof(ht_leaf_stats_t)` (1 folha ativa)** | $3\text{ floats} + 16 \times 80\text{ B}$ | **$1.292\text{ B}$ ($\approx 1.26\text{ KB}$)** | **$2.584\text{ B}$ ($\approx 2.52\text{ KB}$)** |

#### Orçamento do `stats_pool` (256 Folhas Ativas Simultâneas):
- Em **`float` (32-bit):** $256 \times 1.292\text{ B} = 330.752\text{ B} \approx \mathbf{323.0\text{ KB}}$.
- Em **`double` (64-bit):** $256 \times 2.584\text{ B} = 661.504\text{ B} \approx \mathbf{646.0\text{ KB}}$.
- Somado ao `node_pool` de 4.096 nós ($112\text{ KB}$ em `float` / $192\text{ KB}$ em `double`), a árvore inteira ocupa **$\approx 435\text{ KB}$ em `float`** e **$\approx 838\text{ KB}$ em `double`**, permanecendo com folga abaixo do teto estrito de **$1\text{ MB}$ ($1.048.576\text{ B}$)**.
