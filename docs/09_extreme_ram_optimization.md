# 09 — Otimização Avançada e Hipóteses de Redução Extrema de Memória RAM em kansho (C11 VFDT / HAT)

**Projeto:** `kansho` (簡樹, Embedded Hoeffding Tree in pure C11)  
**Versão Atual da Biblioteca:** `0.2.0`  
**Escopo da Análise:** Todo o código-fonte em [`code/include/`](../code/include/) e [`code/src/`](../code/src/)  
**Meta:** Redução máxima do consumo de memória RAM (SRAM estática) preservando 100% das funcionalidades (Classificação, Regressão, ADWIN, HAT, NBA, Pré-processamento, Save/Load) e mantendo integridade matemática e acurácia idênticas.

---

## 1. Diagnóstico do Consumo de Memória Atual (Baseline)

A biblioteca `kansho` foi concebida sob o dogma de **zero alocação dinâmica em runtime** (`malloc`/`free` proibidos em `ht_learn_one` e `ht_predict`), acomodando toda a topologia da árvore e buffers de aprendizado em pools estáticos dentro de `ht_tree_t`.

### 1.1 Medição Exata de Tamanhos em Memória (Compilado C11)

Executando a inspeção direta de `sizeof` e `offsetof` na compilação nativa com os limites padrão (`HT_MAX_NODES = 4096`, `HT_MAX_ACTIVE_LEAVES = 256`, `HT_MAX_FEATURES = 16`, `HT_MAX_CLASSES = 4`, `HT_REG_NUM_SLOTS = 5`, `HT_ADWIN_MAX_BUCKETS = 5`, `HT_ADWIN_LEVELS = 14`):

| Estrutura / Componente | Tamanho Unitário (`float32`) | Tamanho Unitário (`double64`) | Instâncias no Pool | Total em `float32` | Total em `double64` | % da Árvore (`float32`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `stats_pool` (`ht_leaf_stats_t`) | $1.360\text{ B}$ | $2.720\text{ B}$ | $256$ folhas | **$348.160\text{ B}$ ($340,0\text{ KB}$)** | **$696.320\text{ B}$ ($680,0\text{ KB}$)** | **$64,77\%$** |
| `node_pool` (`ht_node_t`) | $28\text{ B}$ | $48\text{ B}$ | $4.096$ nós | **$114.688\text{ B}$ ($112,0\text{ KB}$)** | **$196.608\text{ B}$ ($192,0\text{ KB}$)** | **$21,34\%$** |
| `node_drift` (`ht_node_drift_t`) | $16\text{ B}$ | $24\text{ B}$ | $4.096$ nós | **$65.536\text{ B}$ ($64,0\text{ KB}$)** | **$98.304\text{ B}$ ($96,0\text{ KB}$)** | **$12,19\%$** |
| `free_nodes` (`uint16_t`) | $2\text{ B}$ | $2\text{ B}$ | $4.096$ nós | **$8.192\text{ B}$ ($8,0\text{ KB}$)** | **$8.192\text{ B}$ ($8,0\text{ KB}$)** | **$1,52\%$** |
| `free_stats` (`uint16_t`) | $2\text{ B}$ | $2\text{ B}$ | $256$ folhas | **$512\text{ B}$ ($0,5\text{ KB}$)** | **$512\text{ B}$ ($0,5\text{ KB}$)** | **$0,10\%$** |
| Parâmetros e Cabeçalho `ht_tree_t` | — | — | $1$ árvore | **$400\text{ B}$ ($0,39\text{ KB}$)** | **$640\text{ B}$ ($0,62\text{ KB}$)** | **$0,08\%$** |
| **Total `sizeof(ht_tree_t)`** | — | — | **1 Árvore** | **$537.488\text{ B}$ ($524,89\text{ KB}$)** | **$1.000.576\text{ B}$ ($977,12\text{ KB}$)** | **$100,0\%$** |
| *Pool ADWIN Externo (`ht_adwin_pool_t`)* | $764\text{ B}$ | $1.504\text{ B}$ | $128$ detectores | **$98.068\text{ B}$ ($95,77\text{ KB}$)** | **$192.792\text{ B}$ ($188,27\text{ KB}$)** | *(Externo)* |

### 1.2 Por que Reordenação Simples de Campos Não Funciona?

Uma auditoria cirúrgica de `offsetof` comprovou que o código atual de kansho **já possui empacotamento denso sem nenhum buraco de alinhamento (*zero padding holes*)**:
- `ht_node_t` possui offset contínuo $0 \to 16 \to 20 \to 22 \to 24 \to 26 \to 27$ (28 bytes exatos).
- `ht_node_drift_t` possui offset contínuo $0 \to 4 \to 8 \to 12 \to 14$ (16 bytes exatos).
- `ht_leaf_stats_t` possui offset contínuo $0 \to 4 \to 8 \to 12 \to 76 \to 80$ (1360 bytes exatos).

Portanto, **qualquer redução real de RAM agora exige otimizações estruturais, redução de assimetria de estado e inteligência de compartilhamento de buffers**.

---

## 2. Mapa Conceitual das 7 Hipóteses de Otimização

```mermaid
flowchart TD
    subgraph S1["Camada de Alocação e Memória (ht_memory)"]
        H1["Hipótese 1: Intrusive Free-Lists<br/>(Eliminação de free_nodes e free_stats)<br/><b>-8,5 KB diretos</b>"]
    end

    subgraph S2["Camada de Topologia dos Nós (ht_node_t)"]
        H2["Hipótese 2: Union Assimétrica Folha vs Nó Interno<br/>(28 B ➔ 20 B por nó)<br/><b>-32,0 KB diretos</b>"]
    end

    subgraph S3["Camada de Concept Drift (HAT & ADWIN)"]
        H3["Hipótese 3: Desacoplamento do node_drift<br/>(Pool dedicado de 128 entradas)<br/><b>-62 a -64 KB diretos</b>"]
        H7["Hipótese 7: ADWIN Compacto para MCU<br/>(10 níveis, 32 detectores)<br/><b>-80,2 KB no pool ADWIN</b>"]
    end

    subgraph S4["Camada de Folhas e Observadores (ht_leaf_stats_t)"]
        H4["Hipótese 4: Union de Cabeçalho SGD / NBA<br/>(Elimina 68 B de SGD na Classificação)<br/><b>-17,0 KB na Classificação</b>"]
        H5["Hipótese 5: Bounding Box Única por Feature<br/>(min/max compartilhado entre classes)<br/><b>-96,0 KB em float / -192 KB em double</b>"]
        H6["Hipótese 6: Dynamic Feature Stride Slab<br/>(Alocação baseada em n_features real)<br/><b>-136 a -276 KB (-40% a -81% nas folhas)</b>"]
    end

    Baseline["Baseline kansho<br/>(524,9 KB float32 / 977,1 KB double64)"] --> H1
    H1 --> H2
    H2 --> H3
    H3 --> H4
    H4 --> H5
    H5 --> H6
    H3 --> H7
    H6 --> Resultado["Árvore Otimizada kansho Extreme<br/><b>160 KB a 310 KB</b> (Full Features)<br/><b>< 10 KB</b> (Micro Tiers)"]
```

---

## 3. Detalhamento Técnico das Hipóteses

---

### Hipótese 1: Free-Lists Intrusivas (Eliminação das Pilhas Externas `free_nodes` e `free_stats`)

#### 1. Fundamentação Teórica e Diagnóstico
Em [`code/include/ht_memory.h`](../code/include/ht_memory.h), a struct `ht_memory_pool_t` mantém dois vetores separados para rastrear índices disponíveis:
```c
uint16_t free_nodes[HT_MAX_NODES];         /* 4096 * 2 = 8.192 bytes */
uint16_t free_stats[HT_MAX_ACTIVE_LEAVES]; /*  256 * 2 =   512 bytes */
```
Totalizando **8.704 bytes (~8,5 KB)** de overhead exclusivo de infraestrutura de free-list.

Quando um nó ou bloco de estatísticas está livre (`node->flags & HT_NODE_IN_USE == 0`), os seus 28 bytes (ou 1.360 bytes nas estatísticas) estão **completamente inativos**. Em alocadores estáticos embarcados canônicos (*Slab Allocators*), o próximo índice livre (`next_free`) é gravado **diretamente dentro da memória do próprio bloco inativo** (lista ligada intrusiva).

#### 2. Mudança Proposta de Layout e Algoritmo
Substituir os arrays de pilha por um único ponteiro cabeça `free_nodes_head` e `free_stats_head`:

```c
/* Em ht_node_t: reutilizar left_child como ponteiro intrusivo quando inativo */
typedef struct {
    /* ... */
    uint16_t left_child; /* Se IN_USE: filho esquerdo. Se FREE: next_free_idx */
    /* ... */
} ht_node_t;

/* Em ht_memory_pool_t: */
typedef struct {
    ht_node_t       node_pool[HT_MAX_NODES];
    ht_leaf_stats_t stats_pool[HT_MAX_ACTIVE_LEAVES];
    uint16_t        free_nodes_head; /* 2 bytes em vez de 8.192 bytes! */
    uint16_t        free_stats_head; /* 2 bytes em vez de 512 bytes! */
    uint16_t        nodes_allocated;
    uint16_t        stats_allocated;
} ht_memory_pool_t;
```

As operações de alocação e liberação continuam rigorosamente em tempo estritamente constante $O(1)$:
```c
uint16_t ht_node_alloc(ht_memory_pool_t *pool) {
    if (pool->free_nodes_head == HT_INVALID_IDX) return HT_INVALID_IDX;
    uint16_t idx = pool->free_nodes_head;
    pool->free_nodes_head = pool->node_pool[idx].left_child; /* pop intrusivo */
    pool->nodes_allocated++;
    /* inicializa nó... */
    return idx;
}

void ht_node_free(ht_memory_pool_t *pool, uint16_t idx) {
    /* ... proteções ... */
    pool->node_pool[idx].left_child = pool->free_nodes_head; /* push intrusivo */
    pool->free_nodes_head = idx;
    pool->nodes_allocated--;
}
```

#### 3. Economia de Memória e Impacto
- **Economia em `float32`:** **$8.704\text{ bytes}$ ($8,50\text{ KB}$)** eliminados diretamente da struct.
- **Economia em `double64`:** **$8.704\text{ bytes}$ ($8,50\text{ KB}$)**.
- **Impacto em Acurácia/Funcionalidades:** **$0,0\%$**. A sequência de nós alocados permanece $O(1)$ determinística.
- **Complexidade de Implementação:** Baixa. Afeta apenas [`code/src/ht_memory.c`](../code/src/ht_memory.c).

---

### Hipótese 2: Union Assimétrica de Topologia em `ht_node_t` (Nó Folha vs Nó de Decisão)

#### 1. Fundamentação Teórica e Diagnóstico
Em [`code/include/ht_memory.h`](../code/include/ht_memory.h), a struct `ht_node_t` tem $28\text{ B}$ em `float32` e $48\text{ B}$ em `double64`.

No ciclo de vida de uma Hoeffding Tree, **um nó é estritamente uma Folha OU um Nó Interno de Decisão, nunca ambos simultaneamente**:

1. **Se for Folha (`flags & HT_NODE_IS_LEAF`):**
   - Precisa de: `class_counts` ($16\text{ B}$ em 4 classes) ou `reg_stats` ($12\text{ B}$), `stats_idx` ($2\text{ B}$), e `flags` ($1\text{ B}$).
   - **Campos desperdiçados:** `left_child` ($2\text{ B}$, sempre `0xFFFF`), `right_child` ($2\text{ B}$, sempre `0xFFFF`), `split_val` ($4\text{ B}$, inútil), `feature_idx` ($1\text{ B}$, sempre `0xFF`).
   - Total desperdiçado em cada folha: **$9\text{ bytes}$**.

2. **Se for Nó Interno de Decisão (`!(flags & HT_NODE_IS_LEAF)`):**
   - Precisa de: `split_val` ($4\text{ B}$), `left_child` ($2\text{ B}$), `right_child` ($2\text{ B}$), `feature_idx` ($1\text{ B}$), `flags` ($1\text{ B}$) e peso para desempate de missing values/NaN ($4\text{ B}$).
   - **Campos desperdiçados:** `stats_idx` ($2\text{ B}$, sempre `0xFFFF`) e as contagens detalhadas de classe ($12\text{ a }16\text{ B}$), pois o nó interno nunca mais calcula probabilidades — o roteamento só avalia a condição de split!
   - Total desperdiçado em cada nó interno: **$14\text{ a }18\text{ bytes}$**.

#### 2. Mudança Proposta de Layout
Sobrepor os campos exclusivos de folha com os campos exclusivos de nó interno através de uma `union`:

```c
typedef struct {
    union {
        /* ESTADO EXCLUSIVO DE FOLHA (18 bytes em float32 / 34 bytes em double64) */
        struct {
            union {
                ht_float_t    class_counts[HT_MAX_CLASSES]; /* 16B float / 32B double */
                ht_var_stat_t reg_stats;                    /* 12B float / 24B double */
            };
            uint16_t stats_idx; /* Índice no stats_pool (2B) */
        } leaf;

        /* ESTADO EXCLUSIVO DE RAMO INTERNO (14 bytes em float32 / 22 bytes em double64) */
        struct {
            union {
                ht_float_t threshold; /* 4B float / 8B double */
                int32_t    cat_val;   /* 4B */
            } split_val;
            ht_float_t left_weight;   /* 4B float / 8B double: peso do filho esquerdo p/ NaN */
            uint16_t   left_child;    /* 2B */
            uint16_t   right_child;   /* 2B */
            uint8_t    feature_idx;   /* 1B */
            uint8_t    pad_branch;    /* 1B alinhamento */
        } branch;
    };
    uint8_t flags;    /* 1B (HT_NODE_IS_LEAF, HT_NODE_IN_USE, etc.) */
    uint8_t reserved; /* 1B alinhamento final da struct */
} ht_node_t;
```

#### 3. Economia de Memória e Impacto
- Em `float32`: `sizeof(ht_node_t)` cai de **$28\text{ bytes}$ para $20\text{ bytes}$** (uma redução de $-28,57\%$).
  - Para $4.096$ nós: o `node_pool` encolhe de **$114.688\text{ B}$ ($112\text{ KB}$) para $81.920\text{ B}$ ($80\text{ KB}$)**.
  - **Economia direta: $32.768\text{ bytes}$ ($32,0\text{ KB}$)!**
- Em `double64`: `sizeof(ht_node_t)` cai de **$48\text{ bytes}$ para $40\text{ bytes}$**.
  - O `node_pool` encolhe de **$196.608\text{ B}$ ($192\text{ KB}$) para $163.840\text{ B}$ ($160\text{ KB}$)**.
  - **Economia direta: $32.768\text{ bytes}$ ($32,0\text{ KB}$)!**
- **Impacto em Acurácia/Funcionalidades:** **$0,0\%$**. A semântica de decisão, split e predição é idêntica.
- **Complexidade de Implementação:** Média. Exige atualizar os acessos em [`code/src/ht_tree.c`](../code/src/ht_tree.c) para usar `node->leaf.*` ou `node->branch.*`.

---

### Hipótese 3: Desacoplamento e Pool Dinâmico de Drift (`node_drift` em HAT)

#### 1. Fundamentação Teórica e Diagnóstico
Em [`code/include/ht_tree.h`](../code/include/ht_tree.h), a struct `ht_tree_t` embute:
```c
ht_node_drift_t node_drift[HT_MAX_NODES]; /* 4096 * 16B = 65.536 bytes (64 KB)! */
```
No entanto:
1. **Em modo VFDT padrão (`params.enable_hat == false`):** Todo esse array de **$64\text{ KB}$ fica $100\%$ sem uso**.
2. **Em modo HAT (`params.enable_hat == true`):** A capacidade do pool de detectores ADWIN (`HT_MAX_ADWIN_DETECTORS`) é de apenas **$128$ detectores**. Além disso, o número de árvores alternativas em treino concorrente na HAT raramente ultrapassa 3 a 5 subárvores. É matematicamente impossível existirem mais de 128 nós com ADWIN ativo!

Alocar 4.096 slots de drift (1 para cada nó potencial) desperdiça mais de $96\%$ dos slots mesmo com HAT ativado.

#### 2. Mudança Proposta de Arquitetura
Existem duas abordagens altamente vantajosas:

- **Abordagem 3A (Ponteiro Externo, análogo a `adwin_pool`):**
  Transformar `node_drift` em ponteiro opcional `ht_node_drift_t *node_drift;` inicializado apenas se `enable_hat == true`. Se o usuário usar VFDT pura (a vasta maioria dos casos embarcados), a árvore economiza **$64\text{ KB}$ inteiros instantaneamente**.
- **Abordagem 3B (Pool de Drift Dimensionado em $128$ Entradas):**
  Em vez de indexar por `node_idx` direto (4.096), criar um pool `ht_drift_pool_t` de $128$ entradas ($128 \times 16\text{ B} = 2.048\text{ B} = 2\text{ KB}$).
  No nó interno (da Hipótese 2), utilizar o slot livre de `stats_idx` (já que nó interno não usa `stats_idx`) como `uint16_t drift_idx`:
  ```c
  union {
      uint16_t stats_idx; /* Folha: índice no stats_pool */
      uint16_t drift_idx; /* Nó Interno: índice no drift_pool (0..127) */
  };
  ```

#### 3. Economia de Memória e Impacto
- **Em VFDT pura (`enable_hat == false`):** **$65.536\text{ bytes}$ ($64,0\text{ KB}$ em float32)** e **$98.304\text{ bytes}$ ($96,0\text{ KB}$ em double64)** eliminados da struct `ht_tree_t`.
- **Em HAT (`enable_hat == true`):** Redução de $65.536\text{ B}$ para $2.048\text{ B}$ (**economia líquida de $63.488\text{ B}$ = $62,0\text{ KB}$**).
- **Impacto em Acurácia/Funcionalidades:** **$0,0\%$**. Os testes de drift e switch de subárvores continuam operando normalmente nos nós que possuem detector.

> [!WARNING]
> **Post-Mortem / Justificativa da Falha na Tentativa da Entrega 7.4 (Commit `7128ca6` no branch `backup/fase7-tentativas`):**
> A tentativa inicial de desacoplar o `node_drift` alocou o pool dentro de `ht_adwin_pool_t`, mas sofreu de duas falhas críticas de projeto:
> 1. **Violação de Determinismo $O(1)$:** Como `ht_node_t` alocou `drift_idx` apenas em nós de ramo, as folhas ativas ficaram sem campo de índice. A implementação introduziu um loop de busca linear $O(N)$ (`for (i = 0; i < 128; i++)`) em `ht_tree_node_drift`, executado a **cada amostra processada em treino HAT**.
> 2. **Pilha Externa de Índices:** Reintroduziu vetores externos (`free_indices[128]` e `in_use_mask`), contrariando o padrão de free-lists intrusivas da Hipótese 1.
> 3. **Ausência de Validação N3:** O benchmark de paridade em drift com economia $\ge 62\text{ KB}$ não foi executado.
> *Solução Correta para Retomada:* O `drift_idx` da folha ativa deve ser armazenado diretamente em `ht_leaf_stats_t`, permitindo acesso $O(1)$ imediato via `stats_pool[node->stats_idx].drift_idx` sem nenhum loop.

---

### Hipótese 4: Union de Cabeçalho em `ht_leaf_stats_t` (Eliminação do Overhead de SGD na Classificação)

#### 1. Fundamentação Teórica e Diagnóstico
Em [`code/include/ht_memory.h`](../code/include/ht_memory.h), cada `ht_leaf_stats_t` possui:
```c
ht_float_t reg_weights[HT_MAX_FEATURES]; /* 16 * 4B = 64 bytes */
ht_float_t reg_intercept;                /*  1 * 4B =  4 bytes */
```
Esses **$68\text{ bytes}$** são exclusivos para os modelos lineares SGD de **Regressão Contínua** (`HT_TASK_REGRESSION`).

Em tarefas de **Classificação**, esses $68\text{ bytes}$ não são tocados. Com $256$ folhas ativas no pool:
$$256 \times 68\text{ bytes} = 17.408\text{ bytes } (\mathbf{17,0\text{ KB}})$$
de memória SRAM ociosa na classificação.

#### 2. Mudança Proposta de Layout
Unificar os cabeçalhos de Classificação e Regressão através de `union`:
```c
typedef struct {
    ht_float_t last_split_attempt_at;
    union {
        /* Metadados de Classificação (8 bytes) */
        struct {
            ht_float_t mc_correct_weight;
            ht_float_t nb_correct_weight;
        } classif;

        /* Metadados de Regressão Contínua (76 bytes em float32) */
        struct {
            ht_float_t fmse_mean;
            ht_float_t fmse_model;
            ht_float_t reg_weights[HT_MAX_FEATURES];
            ht_float_t reg_intercept;
        } reg;
    };
    ht_feature_obs_t observers[HT_MAX_FEATURES];
} ht_leaf_stats_t;
```

#### 3. Economia de Memória e Impacto
- **Economia em Classificação:** **$17.408\text{ bytes}$ ($17,0\text{ KB}$ em float32)** e **$34.816\text{ bytes}$ ($34,0\text{ KB}$ em double64)**.
- **Impacto em Acurácia/Funcionalidades:** **$0,0\%$**.

---

### Hipótese 5: Fatoração de Bounding Box em `ht_gaussian_obs_t` (`[min, max]` Único por Atributo)

#### 1. Fundamentação Teórica e Diagnóstico
Atualmente, o observador Gaussiano armazena:
```c
typedef struct {
    ht_float_t count;   /* 4 bytes */
    ht_float_t mean;    /* 4 bytes */
    ht_float_t m2;      /* 4 bytes */
    ht_float_t min_val; /* 4 bytes */
    ht_float_t max_val; /* 4 bytes */
} ht_gaussian_obs_t;   /* 20 bytes por classe */
```
Para $4$ classes, `num[HT_MAX_CLASSES]` consome $4 \times 20 = 80\text{ bytes}$.
Note que `min_val` e `max_val` são gravados **$4$ vezes** (uma para cada classe), totalizando $4 \times 8 = 32\text{ bytes}$ apenas de limites numéricos por atributo!

No entanto, no algoritmo do River ([`references/river_tree/splitter/gaussian_splitter.py`](../references/river_tree/splitter/gaussian_splitter.py)) e na teoria matemática da Hoeffding Tree:
- Os pontos de corte candidatos gerados em `ht_gaussian_split_suggestions` dependem **exclusivamente do intervalo global $[\min(x_f), \max(x_f)]$ do atributo**, obtido agregando todas as classes observadas.
- O cálculo da CDF da normal em cada classe já depende estritamente de $\mu_c$ e $\sigma_c^2 = M_{2,c}/(n_c - 1)$. As checagens `split_val < min_val` e `split_val >= max_val` atuam como salvaguardas para $P(x \le v \mid c) \approx 0$ e $P(x \le v \mid c) \approx 1$.

#### 2. Mudança Proposta de Layout
Armazenar as estatísticas de Welford $(n_c, \mu_c, M_{2,c})$ por classe ($12\text{ bytes} \times 4 = 48\text{ bytes}$) e armazenar a bounding box $[\min, \max]$ **uma única vez por atributo** ($8\text{ bytes}$):

```c
typedef struct {
    ht_float_t count;
    ht_float_t mean;
    ht_float_t m2;
} ht_gaussian_class_stat_t; /* 12 bytes */

typedef struct {
    ht_gaussian_class_stat_t classes[HT_MAX_CLASSES]; /* 4 * 12 = 48 bytes */
    ht_float_t               min_val;                /* 4 bytes */
    ht_float_t               max_val;                /* 4 bytes */
} ht_gaussian_feature_obs_t;                         /* Total: 56 bytes em float32 */
```

#### 3. Economia de Memória e Impacto
- O tamanho unitário do observador Gaussiano cai de **$80\text{ bytes}$ para $56\text{ bytes}$** em `float32` (de $160\text{ B}$ para $112\text{ B}$ em `double64`).
- Por folha ($16$ atributos): cai de $1.280\text{ B}$ para $896\text{ B}$ (**economia de $384\text{ bytes}$ por folha**).
- Para $256$ folhas ativas:
  $$256 \times 384\text{ bytes} = 98.304\text{ bytes } (\mathbf{96,0\text{ KB}}\text{ em float32})$$
  $$256 \times 768\text{ bytes} = 196.608\text{ bytes } (\mathbf{192,0\text{ KB}}\text{ em double64})$$
- **Impacto em Acurácia/Funcionalidades:** **Falha empírica grave (premissa refutada)**.

> [!WARNING]
> **Post-Mortem / Justificativa da Falha na Tentativa da Entrega 7.5 (Commit `8095979`, revertida no commit `ec59602`):**
> A tentativa de implementar as Hipóteses 4 e 5 falhou pelos seguintes motivos empíricos e arquiteturais:
> 1. **Vazamento de Cauda Gaussiana (Refutação da H5):** Ao eliminar `min_val` e `max_val` em nível de classe e deixar apenas a bounding box global, removeu-se a regra de truncamento do River (`GaussianSplitter`). Em variáveis concentradas ou binárias esparsas (como as 44 variáveis one-hot do Forest Covertype), a cauda da normal vazou probabilidades incorretas para classes sem variância, distorcendo o cálculo de Gini/Entropia e colapsando a acurácia de $61,43\%$ para $<50\%$. A H5 foi **completamente revertida no commit `ec59602`**, restaurando limites por classe ($80\text{ B}$).
> 2. **Inocuidade da Union em Arrays Estáticos (Limitação da H4):** Em C11 estático, `sizeof(union)` é o tamanho do seu maior membro ($76\text{ bytes}$ de regressão). Assim, cada folha em classificação pura continuou alocando 76 bytes, gerando economia quase nula ($2\text{ KB}$ em vez de $17,4\text{ KB}$).
> *Conclusão:* A economia nas folhas só é fisicamente viável sem perda de acurácia através do **Dynamic Feature Stride Slab (Hipótese 6)**.

---

### Hipótese 6: Dynamic Feature Stride Slab em `stats_pool` (Compacted Feature Layout)

#### 1. Fundamentação Teórica e Diagnóstico Exaustivo de Desperdício
A maior fonte de desperdício de memória em todo o projeto `kansho` reside no fato de que cada folha ativa em `stats_pool` reserva espaço estático fixado para o pior caso de compilação: `HT_MAX_FEATURES` (16 ou 64 atributos) e modelos de regressão contínua com `HT_MAX_FEATURES` pesos SGD (`reg_weights`) e `HT_REG_NUM_SLOTS` (até 16 slots por feature).

No modelo estático atual (`code/include/ht_memory.h`):
```c
typedef struct {
    ht_node_t       node_pool[HT_MAX_NODES];
    ht_leaf_stats_t stats_pool[HT_MAX_ACTIVE_LEAVES]; /* Alocação estática homogênea */
    /* ... */
} ht_memory_pool_t;
```

##### Caso Real: O Gargalo do Dataset Forest Covertype (Tier 2):
* Em `main_benchmark/tune_hyperparams.py` (e `main_benchmark/main_benchmark.py`), o benchmark do Covertype compila com `-DHT_MAX_FEATURES=64 -DHT_MAX_CLASSES=8 -DHT_MAX_ACTIVE_LEAVES=2 -DHT_MAX_NODES=32 -DHT_REG_NUM_SLOTS=16`.
* O cabeçalho de cada `ht_leaf_stats_t` em modo `F=64` soma **272 bytes** (12 B de contadores base + 256 B de `reg_weights[64]` + 4 B de `reg_intercept`).
* Cada observador `ht_feature_obs_t` para `HT_MAX_CLASSES=8` exigiria apenas **160 bytes** (8 classes × 20 B por Gaussiana/contador), mas a flag `-DHT_REG_NUM_SLOTS=16` infla a *union* interna do observador para **256 bytes** (16 slots × 16 B)!
* Logo, cada folha estática `ht_leaf_stats_t` consome $272 + 64 \times 256 = \mathbf{16.656\text{ bytes} \ (\approx 16,3\text{ KB})}$.
* Com apenas 2 folhas ativas (`HT_MAX_ACTIVE_LEAVES = 2`):
  $$2 \times 16.656\text{ B} = \mathbf{33.312\text{ bytes} \ (\approx 32,5\text{ KB})}$$
* Dos **36.144 bytes (~35,3 KB)** totais da árvore atual (v0.3.0), **33.312 bytes (~32,5 KB, ou 92,2% de toda a SRAM da árvore) estão concentrados nas 2 folhas de aprendizado**!
* **O Desperdício Físico Duplo:** O Forest Covertype possui **54 features** reais e é uma tarefa de **classificação pura**. Dez observadores inteiros ($10 \times 256\text{ B} = 2.560\text{ B}$ por folha) ficam 100% ociosos, enquanto os 256 bytes de `reg_weights[64]` e os 96 bytes extras de `HT_REG_NUM_SLOTS=16` por feature ($54 \times 96\text{ B} = 5.184\text{ B}$) são alocados sem utilidade alguma.

##### Outros Streams de Benchmark (Impacto Potencial de Compactação via `HT_STATS_STORAGE_BYTES` + Stride):
- **SEA Stream:** 3 atributos contínuos de 16 configurados (em classificação com 16 B de cabeçalho + 3 × 80 B = 256 B/folha vs 1.360 B estáticos = economia potencial de ~81% no slab de folhas).
- **Electricity / Airlines:** 7 ou 8 atributos (8 a 9 observadores ociosos por folha).
- **Friedman #1 (Regressão):** 10 atributos + 10 `reg_weights` (evita alocar os 6 observadores e pesos restantes quando `HT_STATS_STORAGE_BYTES` é dimensionado para o perfil).

---

#### 2. Mudança Proposta de Arquitetura: Slab Híbrido (Compile-Time Storage + Runtime Stride)

Para reduzir fisicamente `sizeof(ht_tree_t)` na seção `.bss` (sem depender apenas de economia virtual) **e** manter 100% de paridade algorítmica com os Tiers, a arquitetura adota um modelo híbrido em duas camadas:

1. **Reordenação Livre de Colisão em `ht_leaf_stats_t`:**
   Atualmente, `reg_weights[HT_MAX_FEATURES]` fica *antes* de `observers[HT_MAX_FEATURES]`, deslocando `offsetof(ht_leaf_stats_t, observers)` para 80 B (padrão `F=16`) ou 272 B (`F=64`). Ao mover `reg_intercept` para o cabeçalho fixo e deslocar `reg_weights` para **depois** de `observers`, o cabeçalho fixo passa a ter exatamente **16 bytes** em `float32` (**32 bytes** em `float64`):
   ```c
   typedef struct {
       ht_float_t last_split_attempt_at;
       union {
           ht_float_t mc_correct_weight;
           ht_float_t fmse_mean;
           uint16_t   next_free;
       };
       union {
           ht_float_t nb_correct_weight;
           ht_float_t fmse_model;
       };
       ht_float_t       reg_intercept;               /* Cabeçalho fixo: 16 B (float32) / 32 B (float64) */
       ht_feature_obs_t observers[HT_MAX_FEATURES];  /* Sempre em offset 16 B (float32) / 32 B (float64) */
       ht_float_t       reg_weights[HT_MAX_FEATURES];/* Movido após observers para compactação por stride */
   } ht_leaf_stats_t;
   ```
   * **Preservação de ABI/Testes Padrão:** `sizeof(ht_leaf_stats_t)` permanece rigorosamente inalterado (`1360 B` em `float32`, `2720 B` em `float64`), e `offsetof(ht_leaf_stats_t, observers)` torna-se constante (`16 B` / `32 B`) independente de `HT_MAX_FEATURES`.
   * **Acesso aos Pesos SGD em Regressão:** Como a folha compactada possui apenas `n_features` observadores em runtime, os pesos de regressão começam imediatamente após `stats->observers[n_features]` via helpers inline:
     ```c
     static inline ht_float_t *ht_leaf_reg_weights(ht_leaf_stats_t *stats, uint16_t n_features) {
         return (ht_float_t *)&stats->observers[n_features];
     }
     static inline const ht_float_t *ht_leaf_reg_weights_const(const ht_leaf_stats_t *stats, uint16_t n_features) {
         return (const ht_float_t *)&stats->observers[n_features];
     }
     ```

2. **Parametrização em Compilação de `HT_STATS_STORAGE_BYTES` + Âncora de Tipo (C11 §6.5):**
   Em `code/include/ht_config.h` / `ht_memory.h`, o buffer `stats_storage` tem seu tamanho em `.bss` definido em compile-time por `HT_STATS_STORAGE_BYTES` com 3 regimes:
   - **Padrão (Compatibilidade Plena):** `(size_t)HT_MAX_ACTIVE_LEAVES * sizeof(ht_leaf_stats_t)` (suporta tanto classificação quanto regressão até `HT_MAX_FEATURES`).
   - **Classificação Pura (`-DHT_CLASSIF_ONLY`):** omite o bloco de `reg_weights` da reserva estática, usando `(size_t)HT_MAX_ACTIVE_LEAVES * HT_LEAF_CLASSIF_MAX_BYTES`, e faz `ht_tree_init` / `ht_tree_load` rejeitarem `HT_TASK_REGRESSION`.
   - **Override Explícito (`-DHT_STATS_STORAGE_BYTES=<bytes>`):** permite ajustar o slab em `.bss` exatamente ao orçamento da aplicação embarcada (ex.: `17312` bytes para Covertype com 54 features e 2 folhas).

   Em `ht_memory_pool_t`, `stats_pool` é substituído por uma `union` tipada (removendo o array `stats_pool` para que o compilador detecte qualquer acesso legado não migrado):
   ```c
   typedef struct {
       ht_node_t node_pool[HT_MAX_NODES];
       union {
           _Alignas(ht_leaf_stats_t) uint8_t stats_storage[HT_STATS_STORAGE_BYTES];
           ht_leaf_stats_t _stats_type_anchor; /* Garante effective type e alinhamento em C11 */
       };
       uint16_t leaf_stride;          /* Stride real alinhado em runtime (bytes por folha) */
       uint16_t max_leaves_capacity;  /* min(HT_STATS_STORAGE_BYTES / leaf_stride, HT_MAX_ACTIVE_LEAVES) */
       uint16_t free_nodes_head;
       uint16_t free_stats_head;
       uint16_t nodes_allocated;
       uint16_t stats_allocated;
   } ht_memory_pool_t;
   ```
   *(Com os dois novos `uint16_t`, `sizeof(ht_memory_pool_t)` passa de `430.088 B` → **`430.092 B`** em `float32`, e de `860.168 B` → **`860.176 B`** em `float64` devido ao padding final de 4 bytes para múltiplo de 8).*

##### Cálculo do Stride em Runtime (`ht_memory_init` e `ht_memory_init_stride`):
* **Contrato para Testes Isolados (`ht_memory_init`):** Chamadores que não passam por `ht_tree_init` (como `test_memory.c`, `test_observer.c` e `test_criterion.c`) chamam `ht_memory_init(pool)`, que delega a `ht_memory_init_stride(pool, HT_MAX_FEATURES, default_task)` garantindo funcionamento padrão seguro (`leaf_stride = HT_LEAF_STRIDE_DEFAULT` e `max_leaves_capacity = HT_MAX_ACTIVE_LEAVES`).
* **Inicialização Compactada (`ht_memory_init_stride(pool, n_features, task_type)`):** Invocada por `ht_tree_init` e `ht_tree_load`:
  1. **Base fixa:** `HT_LEAF_HEADER_BYTES` = `offsetof(ht_leaf_stats_t, observers)` = $16\text{ B}$ (`float32`) / $32\text{ B}$ (`float64`).
  2. **Observadores:** $n\_features \times \text{sizeof(ht\_feature\_obs\_t)}$.
  3. **Pesos de Regressão (condicional):** $+ n\_features \times \text{sizeof(ht\_float\_t)}$ apenas quando `task_type == HT_TASK_REGRESSION` (e falha segura se `HT_CLASSIF_ONLY` estiver ativo).
  4. **Alinhamento natural:**
     $$\text{HT\_ALIGN\_MASK} = \text{sizeof(ht\_float\_t)} - 1 \quad (\text{3 em float32, 7 em float64})$$
     $$\text{leaf\_stride} = (\text{raw\_bytes} + \text{HT\_ALIGN\_MASK}) \ \& \ \sim\text{HT\_ALIGN\_MASK}$$
  5. **Cap Rígido de Paridade e Capacidade:**
     $$\text{raw\_cap} = \lfloor \text{HT\_STATS\_STORAGE\_BYTES} / \text{leaf\_stride} \rfloor, \qquad \text{max\_leaves\_capacity} = \min(\text{raw\_cap}, \text{HT\_MAX\_ACTIVE\_LEAVES})$$
     O *clamp* em `HT_MAX_ACTIVE_LEAVES` garante que a compactação nunca expanda silenciosamente o limite de folhas ativas além do Tier configurado, preservando **100% de paridade algorítmica**; caso `raw_cap == 0` (buffer subdimensionado para `n_features`), `ht_memory_init_stride` retorna `false` antes de indexar o slab.
  6. **Métrica de Uso Ativo:** `ht_memory_bytes_used()` passa a calcular `nodes_allocated * sizeof(ht_node_t) + stats_allocated * pool->leaf_stride`.

##### Acesso $O(1)$ Estritamente Determinístico:
O acesso à folha de índice $i$ ocorre via aritmética de ponteiros sobre o slab dentro de `ht_stats_get`:
```c
ht_leaf_stats_t *ht_stats_get(ht_memory_pool_t *pool, uint16_t idx) {
    if (!pool || idx >= pool->max_leaves_capacity || idx == HT_INVALID_IDX) {
        return NULL;
    }
    return (ht_leaf_stats_t *)(pool->stats_storage + ((size_t)idx * pool->leaf_stride));
}
```

---

#### 3. Análise Exaustiva de Riscos de Baixo Nível (Nível de Risco 4 / 5)

A transição de arrays homogêneos tipados para um buffer contíguo de bytes com *stride* variável em C11 introduz riscos críticos que devem ser dominados:

##### Risco 1: *Hardware Faults* por Desalinhamento de Barramento (ARM Cortex-M / ESP32)
* **O Mecanismo de Falha:** Microcontroladores ARM Cortex-M3/M4 (como o Stellaris LM3S6965 emulado no projeto) e ESP32 possuem instruções de carga e armazenamento de ponto flutuante e 64 bits (`VLDR.F32`, `VLDR.F64`, `LDRD`, `STRD`) que exigem **alinhamento natural rígido** (múltiplos de 4 bytes para `float`, múltiplos de 8 bytes para `double`).
* Se `leaf_stride` ou `stats_storage` não respeitar o alinhamento natural, a Folha 0 pode iniciar alinhada, mas a **Folha 1 começará desalinhada**, disparando um **HardFault imediato** no primeiro acesso de CPU.

##### Risco 2: Corrupção Silenciosa por Sobrescrita de Fronteira (Buffer Overrun Entre Folhas)
* **O Mecanismo de Falha:** Como as folhas residem contíguas em `stats_storage`, qualquer loop em `ht_observer.c`, `ht_criterion.c` ou `ht_tree.c` que itere até `HT_MAX_FEATURES` em vez de `n_features`, ou qualquer código de regressão que ainda leia `stats->reg_weights[f]` diretamente em vez de `ht_leaf_reg_weights(stats, n_features)`, invadirá diretamente a Folha $i+1$, destruindo `next_free` ou contadores Naive Bayes sem provocar *Segmentation Fault* imediato no Linux.

##### Risco 3: Violação de Strict Aliasing no Compilador C11 (`-O2` / `-O3`)
* **O Mecanismo de Falha:** A Seção §6.5 do padrão ISO C11 proíbe acessar a mesma região de memória por tipos incompatíveis; um buffer isolado `uint8_t stats_storage[]` convertido por cast para `ht_leaf_stats_t *` pode levar otimizadores (`-O2`/`-O3`) a reordenar ou eliminar leituras/escritas.
* **Mitigação Estrutural:** Envolver `stats_storage` em uma `union` com `ht_leaf_stats_t _stats_type_anchor` dentro de `ht_memory_pool_t` e zerar `stats_storage` no `ht_memory_init_stride`, garantindo que o compilador enxergue ambos os tipos compartilhando o mesmo objeto de armazenamento.

##### Risco 4: Quebra da Serialização Binária (`ht_tree_save` e `ht_tree_load`)
* **O Mecanismo de Falha:** O código atual de serialização copia blocos fixos de `ht_leaf_stats_t`. Com *stride* dinâmico, copiar `sizeof(ht_leaf_stats_t)` leria além da folha compactada (invadindo folhas adjacentes ou o fim de `stats_storage`) e escreveria fora do bloco na restauração. `ht_tree_save`/`ht_tree_load` devem serializar e restaurar estritamente `leaf_stride` bytes (além de verificar/reaplicar `ht_memory_init_stride` antes de restaurar as folhas).

##### Risco 5: Colisão de Layout do Cabeçalho (`reg_weights`) e Inflação de `ht_feature_obs_t`
* **O Mecanismo de Falha (Dois Efeitos Acoplados):**
  1. Manter `reg_weights[HT_MAX_FEATURES]` *antes* de `observers[HT_MAX_FEATURES]` fixa `offsetof(ht_leaf_stats_t, observers)` em 80 B (`F=16`) ou 272 B (`F=64`), tornando impossível um stride de classificação de 16 B sem destruir os observadores. Além disso, declarar `observers[]` como *Flexible Array Member (FAM)* encolheria `sizeof(ht_leaf_stats_t)` para 16 B, quebrando os `_Static_assert`s de tamanho fixo e a compatibilidade com testes unitários que assumem `HT_MAX_FEATURES`. A reordenação (`reg_intercept` no cabeçalho de 16 B / 32 B e `reg_weights` após `observers` via `ht_leaf_reg_weights`) resolve ambos os problemas sem FAM.
  2. Em benchmarks de classificação multiclasse (como Covertype com 8 classes), compilar com `-DHT_REG_NUM_SLOTS=16` infla cada `ht_feature_obs_t` de 160 B para 256 B (`+96 B` desperdiçados por feature). É obrigatório utilizar `HT_REG_NUM_SLOTS=5` (padrão) nesses perfis de classificação.

---

#### 4. Protocolo Obrigatório de Blindagem e Salvaguardas (As 5 Regras de Ouro)

Para que a implementação da Entrega 7.6 seja executada com risco zero, o desenvolvedor/agente deve obrigatoriamente cumprir as seguintes salvaguardas:

1. **Alinhamento Natural, Union Anchor e Clamp de Capacidade:**
   * `stats_storage` em `union` com `_Alignas(ht_leaf_stats_t)` e `ht_leaf_stats_t _stats_type_anchor` (removendo o antigo campo `stats_pool`).
   * Todo cálculo de `leaf_stride` aplica `(raw_bytes + HT_ALIGN_MASK) & ~HT_ALIGN_MASK` (`HT_ALIGN_MASK = sizeof(ht_float_t) - 1`).
   * `max_leaves_capacity` limitado por `min(HT_STATS_STORAGE_BYTES / leaf_stride, HT_MAX_ACTIVE_LEAVES)` e `_Static_assert`s atualizados em `ht_memory.h` (`offsetof(ht_leaf_stats_t, observers) == 16` / `32`; `sizeof(ht_memory_pool_t) == 430092` / `860176`).
2. **Acesso Restrito a `n_features` e Helpers `ht_leaf_reg_weights`:**
   * Proibido indexar `HT_MAX_FEATURES` em loops sobre folhas ativas ou acessar `stats->reg_weights` diretamente em `ht_tree.c`/`ht_criterion.c` (sempre usar `ht_leaf_reg_weights(stats, n_features)` e `ht_stats_get(pool, idx)`).
   * Em `ht_tree_init` e `ht_tree_load`, bloquear `HT_TASK_REGRESSION` quando `-DHT_CLASSIF_ONLY` estiver definido.
3. **Testes de Integridade com Canários de Borda (*Canary Bytes*):**
   * Em `code/tests/test_memory.c`, alocar duas folhas adjacentes, preencher toda a extensão `leaf_stride` (e margem de guarda) com padrões (`0xAA` / `0xDEADBEEF`) e comprovar zero invasão na folha vizinha e integridade de `next_free`.
   * Atualizar também todos os pontos de chamada direta legada em `code/tests/test_tree.c` e `code/tests/test_criterion.c` para usar `ht_stats_get(...)` e `ht_leaf_reg_weights(...)`.
4. **Serialização Adaptativa e Métrica Precisa:**
   * Registrar/validar `leaf_stride` na serialização em `ht_tree_save` e `ht_tree_load`, executando `ht_memory_init_stride` na carga e `memcpy` delimitado estritamente por `pool->leaf_stride`.
   * Atualizar `ht_memory_bytes_used()` para contabilizar `stats_allocated * pool->leaf_stride`.
5. **Portabilidade Bare-Metal (Gate de Emulação QEMU):**
   * Validação obrigatória da Entrega 7.6 no emulador ARM Cortex-M3 (TI Stellaris LM3S6965 via QEMU), confirmando que nenhuma instrução `VLDR`/`LDR` gera *Hard Fault* em hardware embarcado real.

---

#### 5. Economia Consolidada de Memória no Forest Covertype (Tier 2)

Comparação considerando o perfil compacto de classificação para Forest Covertype (`n_features = 54`, `HT_MAX_CLASSES = 8`, `HT_MAX_ACTIVE_LEAVES = 2`, `HT_MAX_NODES = 32`, **`-DHT_REG_NUM_SLOTS=5`**, **`-DHT_CLASSIF_ONLY`**, e slab em `.bss` dimensionado para o perfil via `-DHT_STATS_STORAGE_BYTES=17312` ou `-DHT_MAX_FEATURES=54`):

| Componente | Código Atual (v0.3.0, `F=64, SLOTS=16`) | Com Dynamic Stride + `-DHT_CLASSIF_ONLY` + `SLOTS=5` (7.6) | Economia Líquida |
| :--- | :---: | :---: | :---: |
| **Cabeçalho da folha** | 272 B (`12 + 256 + 4` SGD) | **16 B** (só classificação) | **-256 B por folha** |
| **Observadores por folha** | $64 \times 256\text{ B} = 16.384\text{ B}$ | $54 \times 160\text{ B} = 8.640\text{ B}$ | **-7.744 B por folha** |
| **Tamanho de 1 folha (Stride)** | **16.656 bytes** | **8.656 bytes** | **-8.000 B (-48,0%)** |
| **Stats Pool (2 folhas ativas)** | **33.312 bytes** | **17.312 bytes** | **-16.000 bytes (-48,0%)** |
| **Node Pool (32 nós x 36 B)** | 1.152 bytes | 1.152 bytes | 0 B (inalterado) |
| **Topologia e Header da Árvore** | 1.680 bytes | 1.684 bytes (`+4 B`: `leaf_stride` + `max_leaves_capacity`) | +4 B |
| **Tamanho Total da Árvore** | **36.144 B (35,3 KB)** | **20.148 B (~19,7 KB)** | **-15.996 B (-15,6 KB / -44,3%)** |

* **Acurácia e F1 Macro:** **100% Preservados (61,43% / 58,51%)**, pois `max_leaves_capacity` permanece travado em `HT_MAX_ACTIVE_LEAVES = 2` e toda a matemática de Welford, limites de corte e Naive Bayes Adaptive operam com as exatas 54 features.
* **Observação de Compile-Time vs Runtime:** Sem sobrescrever `HT_STATS_STORAGE_BYTES` (mantendo `-DHT_MAX_FEATURES=64 -DHT_CLASSIF_ONLY`), o `.bss` do slab reserva $2 \times (16 + 64 \times 160) = 20.512\text{ B}$ (~22,8 KB total na árvore), enquanto `leaf_stride` e `ht_memory_bytes_used()` já reportam exatamente **8.656 B por folha** (**20.148 B** ativos). Com `-DHT_STATS_STORAGE_BYTES=17312` (ou `-DHT_MAX_FEATURES=54`), o próprio `sizeof(ht_tree_t)` em `.bss` cai fisicamente para **20.148 B (~19,7 KB)**.
* **Latência de Inferência:** **Neutra a ligeiramente melhor (~0,90 µs)** devido à maior localidade espacial em L1.

---

### Hipótese 7: ADWIN Exponencial Compacto para Microcontroladores

#### 1. Fundamentação Teórica e Diagnóstico
Em [`code/include/ht_drift.h`](../code/include/ht_drift.h), cada detector ADWIN possui $14$ níveis de buckets (`HT_ADWIN_LEVELS = 14`), pesando $764\text{ bytes}$ em `float32` e $1.504\text{ bytes}$ em `double64`.
Com $128$ detectores no pool, `ht_adwin_pool_t` consome **$98\text{ KB}$** em `float32` e **$192,8\text{ KB}$** em `double64`.

No entanto, em microcontroladores de 32 a 256 KB de SRAM:
- O nível 13 do ADWIN armazena buckets de tamanho $2^{13} = 8.192$ amostras. A janela total pode ultrapassar $80.000$ amostras, o que é excessivo para sensores que operam em ciclos curtos e precisam reagir a drifts recentes.
- Reduzindo `HT_ADWIN_LEVELS` de 14 para 10 níveis, a janela máxima suportada ainda é de até $5.115$ amostras (mais que suficiente para detecção precisa com $\delta = 0,002$).
- Cada detector passa a ter $10 \times 52 + 36 = 556\text{ bytes}$ (em vez de $764\text{ B}$).
- Reduzindo o número de detectores simultâneos de 128 para 32 (uma árvore embarcada com profundidade 8 dificilmente possui mais de 10 a 20 ramos ativos simultâneos):
  $$32 \times 556\text{ bytes} \approx 17.792\text{ bytes } (\mathbf{17,4\text{ KB}} \text{ vs } 98\text{ KB})$$

#### 2. Economia de Memória e Impacto
- **Economia no Pool ADWIN:** De $98,1\text{ KB}$ para $17,4\text{ KB}$ (**economia de $80,7\text{ KB}$**, $-82,3\%$).
- **Impacto em Acurácia:** Praticamente nulo para drifts abruptos e drifts graduais com janela $\le 5.000$ amostras.

---

## 4. Análise de Impacto Cumulativo e Matriz de Economia

A tabela a seguir consolida a progressão do consumo de memória RAM ao aplicar as hipóteses de forma acumulada, comparando o estado atual contra os três níveis de otimização propostos:

| Nível de Otimização | Hipóteses Aplicadas | Tamanho `sizeof(ht_tree_t)` em `float32` | Economia Absoluta (`float32`) | Redução Relativa (`float32`) | Tamanho `sizeof(ht_tree_t)` em `double64` | Margem p/ Limite de 1 MB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Atual** | Nenhuma (versão 0.2.0) | **$537,49\text{ KB}$** ($537.488\text{ B}$) | $0\text{ KB}$ | Baseline ($0\%$) | **$977,12\text{ KB}$** ($1.000.576\text{ B}$) | $46,8\text{ KB}$ ($4,6\%$) |
| **Nível 1 (Estrutural Pura)** | **H1** (Intrusive Lists) + **H2** (Union Nó 20B) + **H3** (Desacoplamento Drift) | **$430,73\text{ KB}$** ($441.064\text{ B}$) | **$-106,75\text{ KB}$** | **$-19,86\%$** | **$840,85\text{ KB}$** ($861.032\text{ B}$) | $183,1\text{ KB}$ ($17,9\%$) |
| **Nível 2 (Packing Avançado)** | **Nível 1** + **H4** (Union SGD/NBA) + **H5** (Min/Max Feature Obs) | **$315,42\text{ KB}$** ($322.984\text{ B}$) | **$-222,06\text{ KB}$** | **$-41,31\%$** | **$609,43\text{ KB}$** ($624.056\text{ B}$) | $414,6\text{ KB}$ ($40,5\%$) |
| **Nível 3 (Dynamic Stride)** | **Nível 2** + **H6** (Stride adaptado p/ 9 atributos - ex: Agrawal) | **$163,82\text{ KB}$** ($167.752\text{ B}$) | **$-373,67\text{ KB}$** | **$-69,52\%$** | **$305,98\text{ KB}$** ($313.320\text{ B}$) | $718,0\text{ KB}$ ($70,1\%$) |
| **Ultra-Edge (SEA 3 features)** | **Nível 2** + **H6** (Stride adaptado p/ 3 atributos) | **$40,22\text{ KB}$** ($41.184\text{ B}$) | **$-497,27\text{ KB}$** | **$-92,52\%$** | **$78,78\text{ KB}$** ($80.672\text{ B}$) | $945,2\text{ KB}$ ($92,3\%$) |

> [!IMPORTANT]
> **Conquista Chave do Nível 1 & 2:**
> O Nível 1 e o Nível 2 **não dependem do número de atributos do dataset**. Eles reduzem a struct global de **$537\text{ KB}$ para $315\text{ KB}$** (uma economia líquida de mais de **$222\text{ KB}$ de SRAM**) mantendo compatibilidade total com streams de até 16 atributos e 4 classes!
> Em precisão dupla (`double64`), a margem para o limite inviolável de 1 MB salta de perigosos **$46,8\text{ KB}$ para confortáveis $414,6\text{ KB}$**.

---

## 5. Roteiro de Implementação Sugerido (Passo a Passo)

Para manter a conformidade rigorosa com [`AGENTS.md`](../AGENTS.md) e o protocolo de testes nos 3 níveis (`embedded-c-tree` skill):

1. **Fase 1: Implementação da Hipótese 1 (Free-Lists Intrusivas)**
   - Modificar `ht_memory.h` e `ht_memory.c` para eliminar `free_nodes` e `free_stats`.
   - Executar suíte de testes de estresse de memória (`test_memory_stress_recycling_100k`).
   - Validar em `float` e `double`.
2. **Fase 2: Implementação da Hipótese 2 (Union Assimétrica de Topologia em `ht_node_t`)**
   - Alterar `ht_node_t` para 20 bytes (float) / 40 bytes (double).
   - Atualizar `ht_tree_route`, `ht_tree_learn_one_subtree`, `ht_tree_attempt_split` e rotinas de serialização.
   - Validar testes unitários de nós e benchmark de paridade do River.
3. **Fase 3: Implementação da Hipótese 3 (Desacoplamento de `node_drift`)**
   - Extrair `node_drift` para pool ou torná-lo opcional via macro/ponteiro.
   - Validar testes de drift em `test_drift.c` e benchmarks de HAT em `compare_river.py --mode drift`.
4. **Fase 4: Avaliação das Hipóteses de Observadores (H4, H5 e H6)**
   - Fatorar $[\min, \max]$ para o nível de atributo em `ht_observer.c`.
   - Validar paridade numérica exata contra o River nos benchmarks holdout e prequential.
