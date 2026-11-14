# feat(memory): Otimização Extrema de Memória SRAM — Redução de 20% a 70% de RAM em C11 sem Perda Funcional

## 1. Contexto e Motivação

O projeto **kansho** (簡樹) foi projetado para indução incremental de árvores de decisão em tempo real (*VFDT / HAT*) em sistemas embarcados e microcontroladores com orçamento estrito de $\le 1\text{ MB}$ de RAM e **zero alocação dinâmica em runtime** (`malloc`/`free` proibidos em `ht_learn_one` e `ht_predict`).

Atualmente, na versão `0.2.0`, a árvore padrão com capacidade para 4.096 nós e 256 folhas ativas consome **524,89 KB em `float32`** e **977,12 KB em `double64`**. Embora o modo `float32` opere confortavelmente abaixo de 1 MB, o modo `double64` está no limite físico (apenas 46,8 KB de margem), e microcontroladores de menor porte (como STM32F4 com 128–256 KB, nRF52840, ou RP2040) exigem reduções ainda mais agressivas de SRAM estática.

Esta Issue consolida uma auditoria estrutural aprofundada do código-fonte em `code/` e formaliza **7 hipóteses concretas de engenharia** que permitem reduzir a SRAM da árvore de **525 KB para ~315 KB (redução universal de 40%)** e até **40 KB (redução de 92% para fluxos com poucas features)**, mantendo:
1. **Zero perda de funcionalidades:** Classificação, Regressão contínua, NBA, ADWIN, HAT, pré-processamento e serialização binária mantidos intactos.
2. **Zero perda de acurácia:** Equivalência matemática e determinismo bit-a-bit preservados.
3. **Zero violações de arquitetura C11:** Padrão `-std=c11`, zero recursão profunda e complexidade $O(1)$ estrita.

---

## 2. Diagnóstico da Estrutura Atual (Baseline)

A medição de `sizeof` e `offsetof` na compilação C11 revelou que o código atual **não possui desperdício por desalinhamento (*zero padding holes*)**:
- `ht_node_t`: offset contínuo $0 \to 16 \to 20 \to 22 \to 24 \to 26 \to 27$ (28 bytes exatos em float32).
- `ht_node_drift_t`: offset contínuo $0 \to 4 \to 8 \to 12 \to 14$ (16 bytes exatos em float32).
- `ht_leaf_stats_t`: offset contínuo $0 \to 4 \to 8 \to 12 \to 76 \to 80$ (1.360 bytes exatos em float32).

### Tabela de Consumo Atual (`sizeof(ht_tree_t) = 537.488 B` / 524,89 KB):
| Componente | Tipo C11 | Tamanho Unitário (`float32`) | Quantidade no Pool | Total em `float32` | Total em `double64` | % da Árvore |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `stats_pool` | `ht_leaf_stats_t` | 1.360 B | 256 | **348.160 B (340,0 KB)** | 696.320 B (680,0 KB) | 64,77% |
| `node_pool` | `ht_node_t` | 28 B | 4.096 | **114.688 B (112,0 KB)** | 196.608 B (192,0 KB) | 21,34% |
| `node_drift` | `ht_node_drift_t` | 16 B | 4.096 | **65.536 B (64,0 KB)** | 98.304 B (96,0 KB) | 12,19% |
| `free_nodes` | `uint16_t[]` | 2 B | 4.096 | **8.192 B (8,0 KB)** | 8.192 B (8,0 KB) | 1,52% |
| `free_stats` | `uint16_t[]` | 2 B | 256 | **512 B (0,5 KB)** | 512 B (0,5 KB) | 0,10% |
| Cabeçalho e Params | Scalars / Ptrs | — | 1 | **400 B (0,39 KB)** | 640 B (0,62 KB) | 0,08% |
| **Total `ht_tree_t`** | — | — | — | **537.488 B (524,89 KB)** | **1.000.576 B (977,12 KB)** | **100,0%** |
| *Pool ADWIN (Externo)* | `ht_adwin_pool_t` | 764 B | 128 | **98.068 B (95,77 KB)** | 192.792 B (188,27 KB) | *(Externo)* |

Portanto, qualquer ganho adicional de memória exige **eliminação de assimetrias de estado, unificação de campos mutuamente exclusivos e slabs com stride adaptativo**.

---

## 3. As 7 Hipóteses de Engenharia para Redução de RAM

```mermaid
flowchart TD
    subgraph H1_H3["Nível 1: Estrutural e Gerenciamento de Memória"]
        H1["<b>H1: Free-Lists Intrusivas</b><br/>Reutiliza bytes inativos de nós e folhas livres.<br/><b>Economia: -8,5 KB</b>"]
        H2["<b>H2: Union Assimétrica em ht_node_t</b><br/>Sobrepõe campos exclusivos de folha e ramo.<br/>28 B ➔ 20 B (-28,6% no pool).<br/><b>Economia: -32,0 KB</b>"]
        H3["<b>H3: Desacoplamento de node_drift</b><br/>Extrai array fixo de 4096 para pool dedicado de 128.<br/><b>Economia: -62,0 a -64,0 KB</b>"]
    end

    subgraph H4_H6["Nível 2 e 3: Folhas e Observadores de Atributos"]
        H4["<b>H4: Union de Cabeçalho SGD / NBA</b><br/>Elimina 68 B de pesos SGD em folhas de classificação.<br/><b>Economia: -17,0 KB</b>"]
        H5["<b>H5: Bounding Box Única por Atributo</b><br/>Fatora [min, max] para fora do array de classes.<br/>80 B ➔ 56 B por observador.<br/><b>Economia: -96,0 KB em float / -192 KB em double</b>"]
        H6["<b>H6: Dynamic Feature Stride Slab</b><br/>Aloca apenas n_features reais no stats_pool.<br/><b>Economia: -136 KB a -276 KB (-40% a -81%)</b>"]
        H7["<b>H7: ADWIN Compacto para MCUs</b><br/>10 níveis e 32 detectores.<br/><b>Economia: -80,2 KB no pool ADWIN</b>"]
    end

    Baseline["Baseline (525 KB)"] --> H1_H3
    H1_H3 --> H4_H6
    H4_H6 --> Result["kansho Extreme<br/><b>315 KB</b> (Full) / <b>< 45 KB</b> (Micro Tiers)"]
```

---

### H1: Free-Lists Intrusivas (Eliminação das Pilhas Externas)
- **Problema:** `free_nodes[4096]` e `free_stats[256]` ocupam 8,7 KB como arrays independentes dentro de `ht_memory_pool_t`.
- **Solução:** Como nós livres (`!(flags & HT_NODE_IN_USE)`) não contêm dados válidos, os primeiros 2 bytes de cada nó livre são reaproveitados para armazenar o índice do próximo nó livre (`next_free`), formando uma pilha ligada intrusiva.
- **Complexidade:** $O(1)$ na alocação e na devolução.
- **Economia:** **8.704 bytes (8,50 KB)** em `float32` e `double64`.

---

### H2: Union Assimétrica de Topologia em `ht_node_t` (28 B $\to$ 20 B)
- **Problema:** Um nó é estritamente uma folha de predição OU um ramo de decisão. Folhas nunca utilizam `left_child`, `right_child`, `split_val`, `feature_idx` (9 bytes ociosos). Ramos nunca utilizam `stats_idx` nem as contagens detalhadas de classe (14 bytes ociosos).
- **Solução:** Estruturar `ht_node_t` com uma `union` anônima discriminada por `flags & HT_NODE_IS_LEAF`:
  ```c
  typedef struct {
      union {
          /* FOLHA (18 bytes em float32) */
          struct {
              union {
                  ht_float_t    class_counts[HT_MAX_CLASSES];
                  ht_var_stat_t reg_stats;
              };
              uint16_t stats_idx;
          } leaf;

          /* RAMO INTERNO (14 bytes em float32) */
          struct {
              union {
                  ht_float_t threshold;
                  int32_t    cat_val;
              } split_val;
              ht_float_t left_weight; /* Peso esquerdo para roteamento de NaN */
              uint16_t   left_child;
              uint16_t   right_child;
              uint8_t    feature_idx;
              uint8_t    pad;
          } branch;
      };
      uint8_t flags;
      uint8_t reserved;
  } ht_node_t;
  ```
- **Economia:** `sizeof(ht_node_t)` reduz de **28 B para 20 B** em float32 (e de **48 B para 40 B** em double64).
  - Economia de 8 bytes $\times$ 4.096 nós = **32.768 bytes (32,0 KB)**.

---

### H3: Desacoplamento e Pool Dedicado de Drift (`node_drift`)
- **Problema:** O array `node_drift[4096]` consome **65,5 KB** incondicionalmente, mesmo quando a árvore é inicializada com VFDT padrão (`enable_hat == false`). Quando HAT está ativo, o pool de detectores ADWIN (`HT_MAX_ADWIN_DETECTORS`) possui apenas 128 instâncias, tornando matematicamente impossível existirem mais de 128 nós com drift ativo.
- **Solução:** 
  1. Em modo VFDT: `node_drift` torna-se opcional (ponteiro `NULL`), economizando **64 KB instantaneamente**.
  2. Em modo HAT: Criar um pool dedicado `ht_drift_pool_t` de 128 posições ($128 \times 16\text{ B} = 2.048\text{ B} = 2\text{ KB}$). O nó interno reaproveita o slot `stats_idx` como `uint16_t drift_idx`.
- **Economia:** **65.536 bytes (64,0 KB)** em VFDT pura; **63.488 bytes (62,0 KB)** em HAT.

---

### H4: Union de Cabeçalho em `ht_leaf_stats_t` (Overhead SGD de Regressão)
- **Problema:** Cada folha ativa reserva 68 bytes para pesos lineares SGD (`reg_weights[16]` e `reg_intercept`), utilizados exclusivamente em regressão contínua. Em classificação, esses 68 bytes permanecem ociosos em todas as 256 folhas ativas ($256 \times 68 = 17,4\text{ KB}$).
- **Solução:** Unificar o cabeçalho de classificação (`mc_correct_weight`, `nb_correct_weight`) e o de regressão (`fmse_mean`, `fmse_model`, `reg_weights`, `reg_intercept`) via `union`.
- **Economia:** **17.408 bytes (17,0 KB)** em tarefas de classificação.

---

### H5: Bounding Box Única por Atributo em `ht_gaussian_obs_t`
- **Problema:** O observador Gaussiano armazena `min_val` e `max_val` dentro da struct de cada classe ($4 \times 8 = 32\text{ bytes}$ por atributo). No entanto, o cálculo de pontos de corte em `ht_gaussian_split_suggestions` depende apenas dos limites globais $[\min(x_f), \max(x_f)]$ agregados de todas as classes.
- **Solução:** Manter as estatísticas de Welford $(n_c, \mu_c, M_{2,c})$ por classe ($12\text{ B} \times 4 = 48\text{ B}$) e armazenar $\min$ e $\max$ uma única vez no nível do atributo ($8\text{ B}$).
- **Economia:** O observador cai de 80 B para **56 B** em float32 (redução de 30%).
  - Em 16 atributos e 256 folhas: economia direta de **98.304 bytes (96,0 KB)** em float32 e **192.068 bytes (192,0 KB)** em double64!

---

### H6: Dynamic Feature Stride Slab em `stats_pool`
- **Problema:** Datasets práticos raramente utilizam 16 atributos simultâneos (ex: SEA usa 3, LED usa 7, Agrawal usa 9, Hyperplane usa 10). Reservar estaticamente 16 observadores em cada folha desperdiça até 80% do `stats_pool`.
- **Solução:** Manter um buffer estático contíguo (`stats_storage`) e calcular o stride da folha em runtime na função `ht_tree_init`:
  $$\text{stride} = \text{sizeof(header)} + n\_features \times \text{sizeof(ht\_feature\_obs\_t)}$$
  O acesso `ht_stats_get(pool, idx)` calcula o deslocamento em $O(1)$: `pool->stats_storage + idx * pool->leaf_stride`.
- **Economia:**
  - **SEA (3 features):** stride cai de 1.360 B para 256 B $\to$ **economia de 274,6 KB (-80,8%)**.
  - **Agrawal (9 features):** stride de 736 B $\to$ **economia de 151,6 KB (-44,6%)**.
  - **Hyperplane (10 features):** stride de 816 B $\to$ **economia de 131,1 KB (-38,6%)**.

---

### H7: ADWIN Compacto para Microcontroladores
- **Problema:** 14 níveis de buckets armazenam janelas de até 80.000 amostras, o que consome 98 KB no pool de 128 detectores.
- **Solução:** Em perfis de microcontroladores com SRAM reduzida, limitar a 10 níveis (janela máxima de ~5.100 amostras) e 32 detectores simultâneos.
- **Economia:** O pool ADWIN cai de 98,1 KB para **17,4 KB (economia de 80,7 KB)**.

---

## 4. Matriz de Impacto Cumulativo

| Nível | Modificações Acumuladas | `sizeof(ht_tree_t)` (`float32`) | Economia Absoluta | Redução % | `sizeof(ht_tree_t)` (`double64`) | Margem p/ 1 MB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Atual** | Nenhuma (v0.2.0) | **524,89 KB** | 0 KB | 0% | **977,12 KB** | 46,8 KB (4,6%) |
| **Nível 1 (Estrutural Pura)** | H1 (Free-Lists) + H2 (Nó 20B) + H3 (Drift Decouple) | **430,73 KB** | **-106,75 KB** | **-19,86%** | **840,85 KB** | 183,1 KB (17,9%) |
| **Nível 2 (Packing Avançado)** | Nível 1 + H4 (SGD Union) + H5 (Min/Max Feature) | **315,42 KB** | **-222,06 KB** | **-41,31%** | **609,43 KB** | 414,6 KB (40,5%) |
| **Nível 3 (Dynamic Stride - 9 feats)** | Nível 2 + H6 (Agrawal 9 features) | **163,82 KB** | **-373,67 KB** | **-69,52%** | **305,98 KB** | 718,0 KB (70,1%) |
| **Ultra-Edge (SEA - 3 feats)** | Nível 2 + H6 (SEA 3 features) | **40,22 KB** | **-497,27 KB** | **-92,52%** | **78,78 KB** | 945,2 KB (92,3%) |

---

## 5. Roteiro de Implementação Sugerido

- [ ] **Etapa 1:** Implementar **H1** (`free_nodes_head` e `free_stats_head` intrusivos em `ht_memory.c`).
- [ ] **Etapa 2:** Implementar **H2** (`union` assimétrica folha vs ramo em `ht_node_t` e ajustes de roteamento).
- [ ] **Etapa 3:** Implementar **H3** (desacoplamento de `node_drift` para pool dedicado de 128 entradas).
- [ ] **Etapa 4:** Validar bateria de testes unitários nos 3 níveis (`make test` em float e double).
- [ ] **Etapa 5:** Implementar **H4** e **H5** (otimização de cabeçalho de folha e observadores Gaussianos).
- [ ] **Etapa 6:** Validar paridade e benchmarks prequential contra o River (`compare_river.py`).
- [ ] **Etapa 7:** Implementar **H6** (Dynamic Stride Slab) para tiers compactos.
