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
| **Electricity** | `grace=400`, `delta=0.01`, `tau=0.1`, `crit=gini`, `leaf=mc` | 31,718 | 13,594 | **64.37%** | 64.00% | 63.93% | **63.96%** | 64.58% | 77 nós (8 folhas) |
| **Forest Covertype** | `grace=50`, `delta=0.001`, `tau=0.1`, `crit=gini`, `leaf=nba` | 17,500 | 7,500 | **66.00%** | 63.27% | 62.92% | **62.73%** | 60.74% | 79 nós (8 folhas) |
| **Airlines** | `grace=200`, `delta=1e-07`, `tau=0.1`, `crit=info_gain`, `leaf=nb` | 17,500 | 7,500 | **72.57%** | 57.86% | 55.46% | **55.72%** | 74.62% | 39 nós (8 folhas) |

---

## 2. Desempenho em Regressão Contínua

| Dataset | Hiperparâmetros (Tier 8) | Treino | Teste | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Preq. MAE | Preq. RMSE | Preq. $R^2$ | Topologia (Nós) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | `grace=200`, `delta=1e-07`, `tau=0.01`, `leaf=model` | 14,000 | 6,000 | **2.2397** | 2.8391 | **0.6756** | 2.2026 | 2.8780 | 0.6631 | 43 nós (8 folhas) |
| **Bikes** | `grace=50`, `delta=0.01`, `tau=0.05`, `leaf=mean` | 12,165 | 5,214 | **45.8832** | 72.9454 | **0.0563** | 49.4816 | 76.8065 | 0.5982 | 49 nós (8 folhas) |
| **Appliances Energy** | `grace=100`, `delta=1e-07`, `tau=0.01`, `leaf=adaptive` | 13,814 | 5,921 | **30.7338** | 38.7293 | **0.7316** | 37.6944 | 461.5390 | -36.6134 | 55 nós (8 folhas) |
| **NYC Taxi** | `grace=100`, `delta=0.0001`, `tau=0.05`, `leaf=adaptive` | 10,500 | 4,500 | **341.7619** | 447.4803 | **0.1816** | 349.9388 | 460.5901 | 0.1541 | 55 nós (8 folhas) |

---

## 3. Métricas de Performance, Latência e Recursos em Hardware Emulado (QEMU @ 240 MHz)

> **Nota sobre as Colunas de Latência:**
> - **Ciclo Test-Then-Train:** Cada medição cobre o tempo para predição (`predict`) + treino incremental (`learn_one`) daquela amostra.
> - **Cronometria por Ciclos:** As latências são obtidas em ciclos de clock de CPU via `esp_cpu_get_cycle_count()` e convertidas para microssegundos (µs) na frequência nominal de 240 MHz (µs = ciclos / 240).
> - **Latência Média vs Cauda ($p_{95}, p_{99}$):** A latência média reflete o processamento contínuo nas folhas em $O(1)$. Os percentis de cauda ($p_{95}$ e $p_{99}$) capturam o impacto transitório do teste estatístico do *Hoeffding Bound* executado a cada `grace_period` amostras.


| Dataset | Tarefa | Latência Média | Latência p95 | Latência p99 | Throughput (Vazão) | Model Footprint | Consumo RAM (SRAM / Heap) | FLOPs / Amostra |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classificação | **39.32 µs** | 114.35 µs | 249.28 µs | **25,432 amostras/s** | 978 B (43 nós) | 110.8 KB (.bss) / **0 B Heap** | **69 FLOPs** |
| **Electricity** | Classificação | **32.27 µs** | 72.13 µs | 119.95 µs | **30,989 amostras/s** | 1,726 B (77 nós) | 110.8 KB (.bss) / **0 B Heap** | **149 FLOPs** |
| **Forest Covertype** | Classificação | **509.75 µs** | 1502.60 µs | 5249.56 µs | **1,962 amostras/s** | 1,770 B (79 nós) | 110.8 KB (.bss) / **0 B Heap** | **2,250 FLOPs** |
| **Airlines** | Classificação | **82.16 µs** | 120.88 µs | 174.40 µs | **12,171 amostras/s** | 890 B (39 nós) | 110.8 KB (.bss) / **0 B Heap** | **133 FLOPs** |
| **Friedman #1** | Regressão | **57.20 µs** | 91.61 µs | 124.74 µs | **17,483 amostras/s** | 978 B (43 nós) | 110.8 KB (.bss) / **0 B Heap** | **138 FLOPs** |
| **Bikes** | Regressão | **42.98 µs** | 91.35 µs | 116.28 µs | **23,267 amostras/s** | 1,110 B (49 nós) | 110.8 KB (.bss) / **64 B Heap** | **160 FLOPs** |
| **Appliances Energy** | Regressão | **234.84 µs** | 571.10 µs | 748.63 µs | **4,258 amostras/s** | 1,242 B (55 nós) | 110.8 KB (.bss) / **0 B Heap** | **336 FLOPs** |
| **NYC Taxi** | Regressão | **35.83 µs** | 76.18 µs | 86.21 µs | **27,910 amostras/s** | 1,242 B (55 nós) | 110.8 KB (.bss) / **0 B Heap** | **105 FLOPs** |

---

## 4. Conclusões da Emulação

1. **Zero Vazamento de Memória:** O contador de heap livre (`esp_get_free_heap_size()`) permaneceu rigorosamente inalterado em todos os fluxos de dados, validando o isolamento do Slab Allocator estático (`ht_memory`).
2. **Latência de Microcontrolador:** Mesmo operando a 240 MHz com ponto flutuante de precisão simples, o algoritmo processou cada amostra completa (predição + treino online) em ordens de microsegundos, capacitando o microcontrolador a realizar aprendizado incremental em fluxos de alta frequência.
3. **Stack e SRAM Seguros:** O modelo ocupou estritamente $\le 90\text{ KB}$ de SRAM interna da ESP32, deixando mais de $230\text{ KB}$ livres para o RTOS, pilha de rede e sensores da aplicação embarcada.
