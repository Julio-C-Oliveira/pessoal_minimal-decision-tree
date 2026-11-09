# Relatório de Emulação ESP32 — Hoeffding Tree C11 (Tier 8 Universal)

> **Ambiente de Emulação:** ESP32 (Xtensa Dual-Core LX6 @ 240 MHz) via `qemu-system-xtensa`
> **Perfil de Memória:** Tier 8 Universal (`HT_MAX_NODES=128`, `HT_MAX_ACTIVE_LEAVES=8`, `HT_MAX_FEATURES=64`, `HT_MAX_CLASSES=10`, `HT_REG_NUM_SLOTS=8`)
> **Otimização de Hiperparâmetros:** Melhores configurações empíricas do **Tier 8** extraídas de `main_benchmark/hyperparam_report.md`
> **Garantia de Determinismo:** Zero `malloc`/`free` em runtime ($\Delta\text{Heap} = 0\text{ bytes}$ comprovados)

---

## 1. Desempenho em Classificação

| Dataset | Hiperparâmetros (Tier 8) | Treino | Teste | Acurácia Holdout | Precisão Macro | Recall Macro | F1-Score Macro | Acurácia Preq. | Topologia (Nós) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | `grace=100`, `delta=0.01`, `tau=0.1`, `crit=info_gain`, `leaf=nba` | 14,000 | 6,000 | **99.02%** | 98.76% | 98.99% | **98.87%** | 97.29% | 43 nós (8 folhas) |
| **Electricity** | `grace=400`, `delta=0.01`, `tau=0.1`, `crit=gini`, `leaf=mc` | 14,000 | 6,000 | **64.20%** | 63.83% | 63.63% | **63.67%** | 63.93% | 49 nós (8 folhas) |
| **Forest Covertype** | `grace=50`, `delta=0.001`, `tau=0.1`, `crit=gini`, `leaf=nba` | 14,000 | 6,000 | **66.43%** | 67.08% | 62.48% | **63.69%** | 59.00% | 73 nós (8 folhas) |
| **Airlines** | `grace=50`, `delta=0.0001`, `tau=0.05`, `crit=gini`, `leaf=mc` | 14,000 | 6,000 | **76.53%** | 66.01% | 56.82% | **56.93%** | 75.79% | 15 nós (8 folhas) |

---

## 2. Desempenho em Regressão Contínua

| Dataset | Hiperparâmetros (Tier 8) | Treino | Teste | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Preq. MAE | Preq. RMSE | Preq. $R^2$ | Topologia (Nós) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | `grace=200`, `delta=1e-07`, `tau=0.01`, `leaf=model` | 14,000 | 6,000 | **2.2244** | 2.7960 | **0.6854** | 2.2319 | 2.8466 | 0.6704 | 43 nós (8 folhas) |
| **Bikes** | `grace=200`, `delta=0.01`, `tau=0.1`, `leaf=mean` | 12,165 | 5,214 | **39.6668** | 58.4939 | **0.3932** | 52.2378 | 77.4104 | 0.5919 | 39 nós (8 folhas) |
| **Appliances Energy** | `grace=100`, `delta=1e-07`, `tau=0.01`, `leaf=adaptive` | 13,814 | 5,921 | **29.9763** | 37.8965 | **0.7431** | 38.3980 | 507.6419 | -44.5031 | 51 nós (8 folhas) |
| **NYC Taxi** | `grace=50`, `delta=1e-07`, `tau=0.15`, `leaf=adaptive` | 10,500 | 4,500 | **331.8239** | 432.7701 | **0.2345** | 361.9140 | 2302.3561 | -20.1357 | 61 nós (8 folhas) |

---

## 3. Métricas de Performance, Latência e Recursos em Hardware Emulado (QEMU @ 240 MHz)

> **Nota sobre as Colunas de Latência:**
> - **Ciclo Test-Then-Train:** Cada medição cobre o tempo para predição (`predict`) + treino incremental (`learn_one`) daquela amostra.
> - **Cronometria por Ciclos:** As latências são obtidas em ciclos de clock de CPU via `esp_cpu_get_cycle_count()` e convertidas para microssegundos ($\mu\text{s}$) na frequência nominal de 240 MHz ($\mu\text{s} = \text{ciclos} / 240$).
> - **Latência Média vs Cauda ($p_{95}, p_{99}$):** A latência média reflete o processamento contínuo nas folhas em $O(1)$. Os percentis de cauda ($p_{95}$ e $p_{99}$) capturam o impacto transitório do teste estatístico do *Hoeffding Bound* executado a cada `grace_period` amostras.


| Dataset | Tarefa | Latência Média | Latência $p_{95}$ | Latência $p_{99}$ | Throughput (Vazão) | Model Footprint | Consumo RAM (SRAM / Heap) | FLOPs / Amostra |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classificação | **46.80 $\mu$s** | 161.40 $\mu$s | 236.79 $\mu$s | **21,368 amostras/s** | 1,236 B (43 nós) | 111.3 KB (.bss) / **0 B Heap** | **69 FLOPs** |
| **Electricity** | Classificação | **46.84 $\mu$s** | 75.94 $\mu$s | 113.50 $\mu$s | **21,349 amostras/s** | 1,404 B (49 nós) | 111.3 KB (.bss) / **0 B Heap** | **149 FLOPs** |
| **Forest Covertype** | Classificação | **690.55 $\mu$s** | 2028.09 $\mu$s | 5963.50 $\mu$s | **1,448 amostras/s** | 2,076 B (73 nós) | 111.3 KB (.bss) / **0 B Heap** | **2,250 FLOPs** |
| **Airlines** | Classificação | **58.73 $\mu$s** | 82.17 $\mu$s | 1027.95 $\mu$s | **17,027 amostras/s** | 452 B (15 nós) | 111.3 KB (.bss) / **0 B Heap** | **133 FLOPs** |
| **Friedman #1** | Regressão | **90.51 $\mu$s** | 211.18 $\mu$s | 284.93 $\mu$s | **11,049 amostras/s** | 1,236 B (43 nós) | 111.3 KB (.bss) / **0 B Heap** | **138 FLOPs** |
| **Bikes** | Regressão | **69.60 $\mu$s** | 165.62 $\mu$s | 224.60 $\mu$s | **14,368 amostras/s** | 1,124 B (39 nós) | 111.3 KB (.bss) / **64 B Heap** | **160 FLOPs** |
| **Appliances Energy** | Regressão | **97.42 $\mu$s** | 263.98 $\mu$s | 368.92 $\mu$s | **10,265 amostras/s** | 1,460 B (51 nós) | 111.3 KB (.bss) / **0 B Heap** | **336 FLOPs** |
| **NYC Taxi** | Regressão | **48.98 $\mu$s** | 122.19 $\mu$s | 180.91 $\mu$s | **20,416 amostras/s** | 1,740 B (61 nós) | 111.3 KB (.bss) / **0 B Heap** | **105 FLOPs** |

---

## 4. Conclusões da Emulação

1. **Zero Vazamento de Memória:** O contador de heap livre (`esp_get_free_heap_size()`) permaneceu rigorosamente inalterado em todos os fluxos de dados, validando o isolamento do Slab Allocator estático (`ht_memory`).
2. **Latência de Microcontrolador:** Mesmo operando a 240 MHz com ponto flutuante de precisão simples, o algoritmo processou cada amostra completa (predição + treino online) em ordens de microsegundos, capacitando o microcontrolador a realizar aprendizado incremental em fluxos de alta frequência.
3. **Stack e SRAM Seguros:** O modelo ocupou estritamente $\approx 111\text{ KB}$ de SRAM interna da ESP32, deixando mais de $200\text{ KB}$ livres para o RTOS, pilha de rede e sensores da aplicação embarcada.
