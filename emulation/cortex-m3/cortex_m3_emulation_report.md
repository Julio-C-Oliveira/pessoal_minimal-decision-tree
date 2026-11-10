# Relatório de Emulação ARM Cortex-M3 (TI Stellaris LM3S6965) — Hoeffding Tree C11

> **Ambiente de Emulação:** TI Stellaris LM3S6965 (ARM Cortex-M3 @ 50 MHz) via `qemu-system-arm -M lm3s6965evb`
> **Perfil de Memória:** Tier 4 Universal (`HT_MAX_NODES=128`, `HT_MAX_ACTIVE_LEAVES=4`, `HT_MAX_FEATURES=64`, `HT_MAX_CLASSES=8`)
> **Restrição Física de SRAM:** 64 KB de SRAM interna (`0x20000000 - 0x20010000`). Firmware bare-metal ocupa ~50.9 KB (.bss), deixando > 6 KB livres.
> **Hiperparâmetros & Tratamentos:** Extraídos diretamente do Rank #1 oficial de `main_benchmark/hyperparam_report.md` com pré-processamento online ativado.
> **Garantia de Determinismo:** Zero `malloc`/`free` em runtime ($\Delta\text{Heap} = 0\text{ bytes}$ comprovados)

---

## 1. Desempenho em Classificação

| Dataset | Hiperparâmetros (Tier 4) | Treino | Teste | Acurácia Holdout | Precisão Macro | Recall Macro | F1-Score Macro | Acurácia Preq. | Topologia (Nós) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | `grace=100`, `delta=0.01`, `tau=0.1`, `crit=info_gain`, `leaf=nba` | 14,000 | 6,000 | **95.48%** | 94.32% | 95.55% | **94.89%** | 95.45% | 25 nós (4 folhas) |
| **Electricity** | `grace=400`, `delta=0.01`, `tau=0.1`, `crit=gini`, `leaf=mc` | 31,718 | 13,594 | **64.11%** | 63.70% | 63.52% | **63.57%** | 64.51% | 47 nós (4 folhas) |
| **Forest Covertype** | `grace=50`, `delta=0.001`, `tau=0.1`, `crit=gini`, `leaf=nba` | 17,500 | 7,500 | **64.17%** | 63.47% | 61.46% | **62.02%** | 58.94% | 47 nós (4 folhas) |
| **Airlines** | `grace=50`, `delta=0.0001`, `tau=0.05`, `crit=gini`, `leaf=mc` | 17,500 | 7,500 | **75.75%** | 63.75% | 58.63% | **59.47%** | 75.85% | 17 nós (4 folhas) |

---

## 2. Desempenho em Regressão Contínua

| Dataset | Hiperparâmetros & Tratamento | Treino | Teste | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Preq. MAE | Preq. RMSE | Preq. $R^2$ | Topologia (Nós) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1 (Tratado)** | `grace=200`, `delta=1e-07`, `tau=0.01`, `leaf=model`, `trat=StandardScaler` | 14,000 | 6,000 | **2.4935** | 3.1356 | **0.6044** | 2.4594 | 3.1766 | 0.5895 | 33 nós (4 folhas) |
| **Bikes** | `grace=200`, `delta=0.01`, `tau=0.1`, `leaf=mean` | 12,165 | 5,214 | **54.1496** | 76.3948 | **-0.0350** | 71.9869 | 99.9823 | 0.3192 | 27 nós (4 folhas) |
| **Appliances Energy (Tratado)** | `grace=100`, `delta=1e-07`, `tau=0.01`, `leaf=adaptive`, `trat=LeafNorm` | 13,814 | 5,921 | **38.3687** | 49.0520 | **0.5695** | 43.3148 | 462.2925 | -36.7363 | 31 nós (4 folhas) |
| **NYC Taxi (Tratado)** | `grace=50`, `delta=1e-07`, `tau=0.15`, `leaf=adaptive`, `trat=LeafNorm` | 10,500 | 4,500 | **355.5271** | 463.2528 | **0.1229** | 385.0259 | 2117.5035 | -16.8780 | 35 nós (4 folhas) |

---

## 3. Métricas de Performance, Latência e Recursos em Hardware Emulado (QEMU @ 50 MHz)

> **Nota sobre as Colunas de Latência:**
> - **Ciclo Test-Then-Train:** Cada medição cobre o tempo para predição (`predict`) + treino incremental (`learn_one`) daquela amostra.
> - **Cronometria por Ciclos:** As latências são obtidas em ciclos de clock de CPU via registrador SysTick ARMv7-M e convertidas para microssegundos (µs) na frequência nominal de 50 MHz (µs = ciclos / 50).
> - **Throughput Real:** Reflete a capacidade do núcleo Cortex-M3 de 50 MHz de processar amostras em streaming contínuo sem estouro de pilha.


| Dataset | Tarefa | Latência Média | Latência p95 | Latência p99 | Throughput (Vazão) | Model Footprint | Consumo RAM (SRAM / Heap) | FLOPs / Amostra |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classificação | **4.97 µs** | 22.14 µs | 41.00 µs | **201,207 amostras/s** | 732 B (25 nós) | 49.7 KB (.bss) / **0 B Heap** | **69 FLOPs** |
| **Electricity** | Classificação | **2.21 µs** | 4.56 µs | 7.56 µs | **452,489 amostras/s** | 1,348 B (47 nós) | 49.7 KB (.bss) / **0 B Heap** | **149 FLOPs** |
| **Forest Covertype** | Classificação | **94.56 µs** | 383.70 µs | 624.80 µs | **10,575 amostras/s** | 1,348 B (47 nós) | 49.7 KB (.bss) / **0 B Heap** | **2250 FLOPs** |
| **Airlines** | Classificação | **5.48 µs** | 5.38 µs | 193.94 µs | **182,482 amostras/s** | 508 B (17 nós) | 49.7 KB (.bss) / **0 B Heap** | **133 FLOPs** |
| **Friedman #1 (Tratado)** | Regressão | **8.54 µs** | 17.32 µs | 29.52 µs | **117,096 amostras/s** | 956 B (33 nós) | 49.7 KB (.bss) / **0 B Heap** | **138 FLOPs** |
| **Bikes** | Regressão | **7.60 µs** | 16.00 µs | 19.88 µs | **131,579 amostras/s** | 788 B (27 nós) | 49.7 KB (.bss) / **0 B Heap** | **160 FLOPs** |
| **Appliances Energy (Tratado)** | Regressão | **38.85 µs** | 149.66 µs | 192.90 µs | **25,740 amostras/s** | 900 B (31 nós) | 49.7 KB (.bss) / **0 B Heap** | **336 FLOPs** |
| **NYC Taxi (Tratado)** | Regressão | **8.62 µs** | 41.46 µs | 47.40 µs | **116,009 amostras/s** | 1,012 B (35 nós) | 49.7 KB (.bss) / **0 B Heap** | **105 FLOPs** |

---

## 4. Conclusões da Emulação no ARM Cortex-M3

1. **Compatibilidade com 64 KB de SRAM:** O perfil Tier 4 Universal permitiu treinar e inferir todos os 8 datasets em um único binário com **50.9 KB de BSS estática**, operando com folga dentro do limite físico de 64 KB da Stellaris LM3S6965.

2. **Zero Alocação Dinâmica:** Comprovado $\Delta\text{Heap} = 0\text{ bytes}$ durante todo o ciclo de treino online e inferência holdout, garantindo determinismo temporal estrito e ausência de fragmentação de memória.

3. **Eficiência Sub-milissegundo @ 50 MHz:** Mesmo operando a 50 MHz sem FPU dedicada (emulação em software de ponto flutuante IEEE 754), o microcontrolador atingiu vazões de milhares de amostras por segundo em fluxos contínuos.

