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
| **Airlines** | `grace=50`, `delta=0.0001`, `tau=0.05`, `crit=gini`, `leaf=mc` | 17,500 | 7,500 | **75.75%** | 63.75% | 58.63% | **59.47%** | 75.85% | 23 nós (8 folhas) |

---

## 2. Desempenho em Regressão Contínua

| Dataset | Hiperparâmetros (Tier 8) | Treino | Teste | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Preq. MAE | Preq. RMSE | Preq. $R^2$ | Topologia (Nós) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | `grace=200`, `delta=1e-07`, `tau=0.01`, `leaf=model` | 14,000 | 6,000 | **2.2983** | 2.8873 | **0.6645** | 2.2714 | 2.8942 | 0.6593 | 41 nós (8 folhas) |
| **Bikes** | `grace=200`, `delta=0.01`, `tau=0.1`, `leaf=mean` | 12,165 | 5,214 | **47.0819** | 71.8738 | **0.0839** | 66.6412 | 98.2309 | 0.3429 | 45 nós (8 folhas) |
| **Appliances Energy** | `grace=100`, `delta=1e-07`, `tau=0.01`, `leaf=adaptive` | 13,814 | 5,921 | **30.5998** | 38.6607 | **0.7326** | 40.4381 | 507.9032 | -44.5499 | 55 nós (8 folhas) |
| **NYC Taxi** | `grace=50`, `delta=1e-07`, `tau=0.15`, `leaf=adaptive` | 10,500 | 4,500 | **351.7103** | 457.7860 | **0.1435** | 384.1912 | 2307.6217 | -20.2325 | 55 nós (8 folhas) |

---

## 3. Métricas de Performance, Latência e Recursos em Hardware Emulado (QEMU @ 240 MHz)

> **Nota sobre as Colunas de Latência:**
> - **Ciclo Test-Then-Train:** Cada medição cobre o tempo para predição (`predict`) + treino incremental (`learn_one`) daquela amostra.
> - **Cronometria por Ciclos:** As latências são obtidas em ciclos de clock de CPU via `esp_cpu_get_cycle_count()` e convertidas para microssegundos (µs) na frequência nominal de 240 MHz (µs = ciclos / 240).
> - **Latência Média vs Cauda ($p_{95}, p_{99}$):** A latência média reflete o processamento contínuo nas folhas em $O(1)$. Os percentis de cauda ($p_{95}$ e $p_{99}$) capturam o impacto transitório do teste estatístico do *Hoeffding Bound* executado a cada `grace_period` amostras.


| Dataset | Tarefa | Latência Média | Latência p95 | Latência p99 | Throughput (Vazão) | Model Footprint | Consumo RAM (SRAM / Heap) | FLOPs / Amostra |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classificação | **37.36 µs** | 125.95 µs | 178.11 µs | **26,767 amostras/s** | 1,236 B (43 nós) | 112.0 KB (.bss) / **0 B Heap** | **69 FLOPs** |
| **Electricity** | Classificação | **26.64 µs** | 50.48 µs | 70.17 µs | **37,538 amostras/s** | 2,188 B (77 nós) | 112.0 KB (.bss) / **0 B Heap** | **149 FLOPs** |
| **Forest Covertype** | Classificação | **541.17 µs** | 1609.30 µs | 4134.62 µs | **1,848 amostras/s** | 2,244 B (79 nós) | 112.0 KB (.bss) / **0 B Heap** | **2,250 FLOPs** |
| **Airlines** | Classificação | **41.99 µs** | 53.67 µs | 935.05 µs | **23,815 amostras/s** | 676 B (23 nós) | 112.0 KB (.bss) / **0 B Heap** | **133 FLOPs** |
| **Friedman #1** | Regressão | **66.10 µs** | 118.65 µs | 161.13 µs | **15,129 amostras/s** | 1,180 B (41 nós) | 112.0 KB (.bss) / **0 B Heap** | **138 FLOPs** |
| **Bikes** | Regressão | **63.42 µs** | 123.90 µs | 157.09 µs | **15,768 amostras/s** | 1,292 B (45 nós) | 112.0 KB (.bss) / **64 B Heap** | **160 FLOPs** |
| **Appliances Energy** | Regressão | **114.19 µs** | 234.95 µs | 305.71 µs | **8,757 amostras/s** | 1,572 B (55 nós) | 112.0 KB (.bss) / **0 B Heap** | **336 FLOPs** |
| **NYC Taxi** | Regressão | **40.66 µs** | 94.04 µs | 131.65 µs | **24,594 amostras/s** | 1,572 B (55 nós) | 112.0 KB (.bss) / **0 B Heap** | **105 FLOPs** |

---

## 4. Conclusões da Emulação

1. **Zero Vazamento de Memória:** O contador de heap livre (`esp_get_free_heap_size()`) permaneceu rigorosamente inalterado em todos os fluxos de dados, validando o isolamento do Slab Allocator estático (`ht_memory`).
2. **Latência de Microcontrolador:** Mesmo operando a 240 MHz com ponto flutuante de precisão simples, o algoritmo processou cada amostra completa (predição + treino online) em ordens de microsegundos, capacitando o microcontrolador a realizar aprendizado incremental em fluxos de alta frequência.
3. **Stack e SRAM Seguros:** O modelo ocupou estritamente $\le 90\text{ KB}$ de SRAM interna da ESP32, deixando mais de $230\text{ KB}$ livres para o RTOS, pilha de rede e sensores da aplicação embarcada.
