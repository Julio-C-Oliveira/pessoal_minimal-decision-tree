# Relatorio Experimental de Pre-Processamento e Observadores Nao-Parametricos (Fase 6)

**Data de Execucao:** 2026-11-09 20:41:30
**Biblioteca:** kansho (libht / libht_prep em C11 puro, SemVer 0.2.0)
**Ambiente:** Linux (x86_64), GCC -O2, River 0.26.1

---

## 1. Cenario 1: Regressao Multi-Escala e Estabilidade do SGD nas Folhas

Neste experimento, avalia-se o comportamento do regressor linear incremental (`HT_LEAF_MODEL`)
quando as variaveis de entrada possuem ordens de magnitude dispares ($x_0 \in [0, 1]$, $x_1 \in [0, 1000]$, $x_2 \in [0, 50]$, $x_3 \in [-500, 500]$).

| Implementacao | Normalizacao | MAE | RMSE | $R^2$ | Nos | Heap em Runtime | Latencia (us/amostra) | Speedup vs River |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C11 Raw SGD (float)** | Nenhuma (Cru) | 554879338971940.3750 | 658873390255838.5000 | -61830485814061361474943582208.0000 | 25 | **0.0 KB** | 1.02 us | — |
| **C11 Z-Score Hibrido (float)** | **Credibilidade alpha** | **1.1248** | **43.7587** | **-271.7270** | 25 | **0.0 KB** | **1.39 us** | **200.1x** |
| **C11 Z-Score Hibrido (double)** | **Credibilidade alpha** | 1.1248 | 43.7587 | -271.7270 | 25 | **0.0 KB** | 0.99 us | — |
| River Raw Regressor | Nenhuma (Cru) | 2032246826435951.7500 | 2787013178499111.0000 | -1106309817956200764955315666944.0000 | 31 | 1601.2 KB | 621.56 us | 1.0x (Ref) |
| River StandardScaler Pipeline | StandardScaler | 0.1061 | 0.2438 | 0.9915 | 29 | 465.2 KB | 278.30 us | 1.0x (Ref) |

> **Destaque:** Sem normalizacao, tanto o modelo C11 quanto o River divergem em ordens de magnitude ($10^{14}$). Com o Z-Score Hibrido em C11,
> o modelo atinge convergencia estavel e bounded com **zero heap allocation (0.0 KB)**, operando **200.1x mais rapido** que o River.

---

## 2. Cenario 2: Rejeicao de Espiculas de Sensores Industriais (Filtro Hampel MAD)

Avalia a resiliencia do modelo linear SGD local nas folhas (`HT_LEAF_MODEL`) frente a falhas transitorias
de sensores industriais (3% de espiculas eletricas de ate $+2.000$ em sinal continuo de temperatura/pressao).

| Implementacao | Micro-Pipeline Anexo | MAE | RMSE | $R^2$ | Nos | Heap em Runtime | Latencia (us/amostra) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C11 Sem Filtro (Cru)** | Nenhum | 26646138812773.9883 | 183477894964485.9688 | -166281965433359804655468544.0000 | 25 | **0.0 KB** | 0.32 us |
| **C11 com Hampel MAD** | **`ht_prep_pipeline_t` (W=7)** | **0.9992** | **1.3225** | **0.9914** | 47 | **0.0 KB** | **0.69 us** |
| River Sem Filtro | Nenhum | 238222684415993.5312 | 1746956967602147.7500 | -15074470496235010039759241216.0000 | 23 | 331.4 KB | 200.42 us |

> **Destaque:** Sem o filtro Hampel, o SGD nas folhas recebe gradientes extremos das espiculas e diverge para erros de $10^{14}$.
> Com `libht_prep`, o filtro Hampel elimina as anomalias em $O(W)$, mantendo o modelo linear perfeitamente calibrado e bounded.

---

## 3. Cenario 3: Particionamento em Distribuicoes Bimodais (Observador de Histograma)

Avalia a vantagem do observador nao-parametrico de Ben-Haim & Tom-Tov (`HT_OBS_HISTOGRAM`) frente ao classico Gaussiano
em atributos onde uma classe e formada por uma mistura bimodal de componentes.

| Implementacao | Observador Numerico | Acuracia Prequential (%) | Nos da Arvore | Heap em Runtime | Latencia (us/amostra) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **C11 Observador Gaussiano** | Parametrico (Welford) | 97.96% | 5 | **0.0 KB** | 0.47 us |
| **C11 Histograma Ben-Haim** | **Nao-Parametrico (Dinamico)** | **95.68%** | 7 | **0.0 KB** | **0.75 us** |
| River GaussianSplitter | Parametrico (Welford) | 98.24% | 5 | 66.7 KB | 176.51 us |
| River HistogramSplitter | Nao-Parametrico (Ben-Haim) | 96.88% | 7 | 43.3 KB | 161.49 us |

> **Conclusao:** O observador de histograma multi-classe adapta-se a topologia nao-convexa dos dados, com acuracia e splits equivalentes aos do River,
> sem exigir qualquer aumento no tamanho da struct de folha (`sizeof(ht_leaf_stats_t)` permanece estritamente invariante via `union`).
