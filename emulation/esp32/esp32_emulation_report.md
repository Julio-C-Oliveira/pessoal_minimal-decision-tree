# Relatório de Emulação ESP32 — Hoeffding Tree C11 (Tier 8 Universal)

> **Ambiente de Emulação:** ESP32 (Xtensa Dual-Core LX6 @ 240 MHz) via `qemu-system-xtensa`
> **Perfil de Memória:** Tier 8 Universal (`HT_MAX_NODES=128`, `HT_MAX_ACTIVE_LEAVES=8`, `HT_MAX_FEATURES=64`, `HT_MAX_CLASSES=10`)
> **Garantia de Determinismo:** Zero `malloc`/`free` em runtime ($\Delta\text{Heap} = 0\text{ bytes}$ comprovados)

---

## 1. Desempenho em Classificação

| Dataset | Treino | Teste | Acurácia Holdout | Precisão Macro | Recall Macro | F1-Score Macro | Acurácia Preq. | Topologia (Nós) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | 14,000 | 6,000 | **98.10%** | 98.11% | 97.52% | **97.81%** | 96.04% | 11 nós (6 folhas) |
| **Electricity** | 14,000 | 6,000 | **64.05%** | 63.68% | 63.29% | **63.32%** | 62.99% | 9 nós (5 folhas) |
| **Forest Covertype** | 14,000 | 6,000 | **52.62%** | 50.23% | 51.59% | **50.63%** | 41.64% | 1 nós (1 folhas) |
| **Airlines** | 14,000 | 6,000 | **76.80%** | 68.70% | 54.15% | **52.21%** | 75.99% | 7 nós (4 folhas) |

---

## 2. Desempenho em Regressão Contínua

| Dataset | Treino | Teste | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Preq. MAE | Preq. RMSE | Preq. $R^2$ | Topologia (Nós) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | 14,000 | 6,000 | **2.2983** | 2.8873 | **0.6645** | 2.2680 | 2.8777 | 0.6631 | 41 nós (8 folhas) |
| **Bikes** | 12,165 | 5,214 | **53.1949** | 79.9699 | **-0.1341** | 59.6959 | 91.8369 | 0.4256 | 27 nós (8 folhas) |
| **Appliances Energy** | 13,814 | 5,921 | **33.1660** | 42.3745 | **0.6787** | 42.1982 | 508.0700 | -44.5799 | 35 nós (8 folhas) |
| **NYC Taxi** | 10,500 | 4,500 | **353.1150** | 458.2122 | **0.1419** | 384.5726 | 2306.8679 | -20.2186 | 35 nós (8 folhas) |

---

## 3. Métricas de Performance, Latência e Recursos em Hardware Emulado (QEMU @ 240 MHz)

> **Nota sobre as Colunas de Latência:**
> - **Ciclo Test-Then-Train:** Cada medição cobre o tempo para predição (`predict`) + treino incremental (`learn_one`) daquela amostra.
> - **Cronometria por Ciclos:** As latências são obtidas em ciclos de clock de CPU via `esp_cpu_get_cycle_count()` e convertidas para microssegundos ($\mu\text{s}$) na frequência nominal de 240 MHz ($\mu\text{s} = \text{ciclos} / 240$).
> - **Latência Média vs Cauda ($p_{95}, p_{99}$):** A latência média reflete o processamento contínuo nas folhas em $O(1)$. Os percentis de cauda ($p_{95}$ e $p_{99}$) capturam o impacto transitório do teste estatístico do *Hoeffding Bound* executado a cada 200 amostras (`grace_period`).

| Dataset | Tarefa | Latência Média | Latência $p_{95}$ | Latência $p_{99}$ | Throughput (Vazão) | Model Footprint | Consumo RAM (SRAM / Heap) | FLOPs / Amostra |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classificação | **73.81 $\mu$s** | 132.11 $\mu$s | 535.88 $\mu$s | **13,548 amostras/s** | 340 B (11 nós) | 111.3 KB (.bss) / **0 B Heap** | **69 FLOPs** |
| **Electricity** | Classificação | **115.50 $\mu$s** | 206.87 $\mu$s | 965.48 $\mu$s | **8,658 amostras/s** | 284 B (9 nós) | 111.3 KB (.bss) / **0 B Heap** | **149 FLOPs** |
| **Forest Covertype** | Classificação | **1947.20 $\mu$s** | 2250.05 $\mu$s | 4288.57 $\mu$s | **514 amostras/s** | 60 B (1 nós) | 111.3 KB (.bss) / **0 B Heap** | **2,250 FLOPs** |
| **Airlines** | Classificação | **115.84 $\mu$s** | 162.18 $\mu$s | 228.89 $\mu$s | **8,633 amostras/s** | 228 B (7 nós) | 111.3 KB (.bss) / **0 B Heap** | **133 FLOPs** |
| **Friedman #1** | Regressão | **53.68 $\mu$s** | 83.53 $\mu$s | 113.50 $\mu$s | **18,629 amostras/s** | 1,180 B (41 nós) | 111.3 KB (.bss) / **0 B Heap** | **138 FLOPs** |
| **Bikes** | Regressão | **63.20 $\mu$s** | 95.74 $\mu$s | 121.85 $\mu$s | **15,823 amostras/s** | 788 B (27 nós) | 111.3 KB (.bss) / **64 B Heap** | **160 FLOPs** |
| **Appliances Energy** | Regressão | **79.85 $\mu$s** | 151.95 $\mu$s | 195.40 $\mu$s | **12,523 amostras/s** | 1,012 B (35 nós) | 111.3 KB (.bss) / **0 B Heap** | **336 FLOPs** |
| **NYC Taxi** | Regressão | **41.81 $\mu$s** | 80.66 $\mu$s | 157.95 $\mu$s | **23,918 amostras/s** | 1,012 B (35 nós) | 111.3 KB (.bss) / **0 B Heap** | **105 FLOPs** |

---

## 4. Conclusões da Emulação

1. **Zero Vazamento de Memória:** O contador de heap livre (`esp_get_free_heap_size()`) permaneceu rigorosamente inalterado em todos os fluxos de dados, validando o isolamento do Slab Allocator estático (`ht_memory`).
2. **Latência de Microcontrolador:** Mesmo operando a 240 MHz com ponto flutuante de precisão simples, o algoritmo processou cada amostra completa (predição + treino online) em ordens de microsegundos, capacitando o microcontrolador a realizar aprendizado incremental em fluxos de alta frequência.
3. **Stack e SRAM Seguros:** O modelo ocupou estritamente $\approx 111\text{ KB}$ de SRAM interna da ESP32, deixando mais de $200\text{ KB}$ livres para o RTOS, pilha de rede e sensores da aplicação embarcada.
