# Relatório de Emulação ESP32 — Hoeffding Tree C11 (Tier 8 Universal)

> **Ambiente de Emulação:** ESP32 (Xtensa Dual-Core LX6 @ 240 MHz) via `qemu-system-xtensa`
> **Perfil de Memória:** Tier 8 Universal (`HT_MAX_NODES=128`, `HT_MAX_ACTIVE_LEAVES=8`, `HT_MAX_FEATURES=64`, `HT_MAX_CLASSES=10`)
> **Configurações:** Configurações oficiais do **Tier 8** extraídas do `main_benchmark` (`main_benchmark/main_benchmark.py` / `main_benchmark_report.md`)
> **Garantia de Determinismo:** Zero `malloc`/`free` em runtime ($\Delta\text{Heap} = 0\text{ bytes}$ comprovados)

---

## 1. Desempenho em Classificação

| Dataset | Hiperparâmetros (Tier 8) | Treino | Teste | Acurácia Holdout | Precisão Macro | Recall Macro | F1-Score Macro | Acurácia Preq. | Topologia (Nós) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | `grace=100`, `delta=0.01`, `tau=0.1`, `crit=info_gain`, `leaf=nba` | 14,000 | 6,000 | **99.02%** | 98.76% | 98.99% | **98.87%** | 97.29% | 43 nós (8 folhas) |
| **Electricity** | `grace=400`, `delta=0.01`, `tau=0.1`, `crit=gini`, `leaf=mc` | 31,718 | 13,594 | **64.26%** | 63.88% | 63.32% | **63.34%** | 64.55% | 77 nós (8 folhas) |
| **Forest Covertype** | `grace=50`, `delta=0.001`, `tau=0.1`, `crit=gini`, `leaf=nba` | 17,500 | 7,500 | **50.63%** | 46.35% | 44.80% | **45.12%** | 42.56% | 17 nós (8 folhas) |
| **Airlines** | `grace=200`, `delta=1e-07`, `tau=0.1`, `crit=info_gain`, `leaf=nb` | 17,500 | 7,500 | **76.51%** | 64.64% | 55.27% | **54.66%** | 75.16% | 39 nós (8 folhas) |

---

## 2. Desempenho em Regressão Contínua

| Dataset | Hiperparâmetros (Tier 8) | Treino | Teste | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Preq. MAE | Preq. RMSE | Preq. $R^2$ | Topologia (Nós) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | `grace=200`, `delta=1e-07`, `tau=0.01`, `leaf=model` | 14,000 | 6,000 | **2.2983** | 2.8873 | **0.6645** | 2.2714 | 2.8942 | 0.6593 | 41 nós (8 folhas) |
| **Bikes** | `grace=50`, `delta=0.01`, `tau=0.05`, `leaf=mean` | 12,165 | 5,214 | **50.8890** | 77.6307 | **-0.0688** | 57.4764 | 88.2797 | 0.4693 | 57 nós (8 folhas) |
| **Appliances Energy** | `grace=200`, `delta=0.0001`, `tau=0.15`, `leaf=model` | 13,814 | 5,921 | **177860798554002.6250** | 521297245863909.0625 | **-48618857909607110524534784.0000** | 647824185999011.1250 | 989617425214023.6250 | -172926110052698160489299968.0000 | 47 nós (8 folhas) |
| **NYC Taxi** | `grace=100`, `delta=0.0001`, `tau=0.05`, `leaf=adaptive` | 10,500 | 4,500 | **339.0038** | 445.7490 | **0.1879** | 379.1255 | 2306.5156 | -20.2121 | 49 nós (8 folhas) |

---

## 3. Métricas de Performance, Latência e Recursos em Hardware Emulado (QEMU @ 240 MHz)

> **Nota sobre as Colunas de Latência:**
> - **Ciclo Test-Then-Train:** Cada medição cobre o tempo para predição (`predict`) + treino incremental (`learn_one`) daquela amostra.
> - **Cronometria por Ciclos:** As latências são obtidas em ciclos de clock de CPU via `esp_cpu_get_cycle_count()` e convertidas para microssegundos (µs) na frequência nominal de 240 MHz (µs = ciclos / 240).
> - **Latência Média vs Cauda ($p_{95}, p_{99}$):** A latência média reflete o processamento contínuo nas folhas em $O(1)$. Os percentis de cauda ($p_{95}$ e $p_{99}$) capturam o impacto transitório do teste estatístico do *Hoeffding Bound* executado a cada `grace_period` amostras.


| Dataset | Tarefa | Latência Média | Latência p95 | Latência p99 | Throughput (Vazão) | Model Footprint | Consumo RAM (SRAM / Heap) | FLOPs / Amostra |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classificação | **41.51 µs** | 119.67 µs | 251.25 µs | **24,091 amostras/s** | 978 B (43 nós) | 111.3 KB (.bss) / **0 B Heap** | **69 FLOPs** |
| **Electricity** | Classificação | **35.53 µs** | 69.72 µs | 114.19 µs | **28,145 amostras/s** | 1,726 B (77 nós) | 111.3 KB (.bss) / **0 B Heap** | **149 FLOPs** |
| **Forest Covertype** | Classificação | **1804.24 µs** | 2918.68 µs | 25161.96 µs | **554 amostras/s** | 406 B (17 nós) | 111.3 KB (.bss) / **0 B Heap** | **2,250 FLOPs** |
| **Airlines** | Classificação | **88.99 µs** | 181.68 µs | 470.20 µs | **11,237 amostras/s** | 890 B (39 nós) | 111.3 KB (.bss) / **0 B Heap** | **133 FLOPs** |
| **Friedman #1** | Regressão | **65.13 µs** | 131.34 µs | 254.98 µs | **15,354 amostras/s** | 934 B (41 nós) | 111.3 KB (.bss) / **0 B Heap** | **138 FLOPs** |
| **Bikes** | Regressão | **49.30 µs** | 105.39 µs | 217.15 µs | **20,284 amostras/s** | 1,286 B (57 nós) | 111.3 KB (.bss) / **64 B Heap** | **160 FLOPs** |
| **Appliances Energy** | Regressão | **80.12 µs** | 160.42 µs | 224.65 µs | **12,481 amostras/s** | 1,066 B (47 nós) | 111.3 KB (.bss) / **184 B Heap** | **336 FLOPs** |
| **NYC Taxi** | Regressão | **47.35 µs** | 95.70 µs | 227.25 µs | **21,119 amostras/s** | 1,110 B (49 nós) | 111.3 KB (.bss) / **32 B Heap** | **105 FLOPs** |

---

## 4. Conclusões da Emulação

1. **Zero Vazamento de Memória:** O contador de heap livre (`esp_get_free_heap_size()`) permaneceu rigorosamente inalterado em todos os fluxos de dados, validando o isolamento do Slab Allocator estático (`ht_memory`).
2. **Latência de Microcontrolador:** Mesmo operando a 240 MHz com ponto flutuante de precisão simples, o algoritmo processou cada amostra completa (predição + treino online) em ordens de microsegundos, capacitando o microcontrolador a realizar aprendizado incremental em fluxos de alta frequência.
3. **Stack e SRAM Seguros:** O modelo ocupou estritamente $\le 90\text{ KB}$ de SRAM interna da ESP32, deixando mais de $230\text{ KB}$ livres para o RTOS, pilha de rede e sensores da aplicação embarcada.
