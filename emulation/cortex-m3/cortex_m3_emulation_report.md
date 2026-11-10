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
| **SEA** | `grace=400`, `delta=0.0001`, `tau=0.15`, `crit=info_gain`, `leaf=nba` | 14,000 | 6,000 | **97.30%** | 97.72% | 96.09% | **96.85%** | 96.50% | 19 nós (4 folhas) |
| **Electricity** | `grace=100`, `delta=1e-07`, `tau=0.05`, `crit=info_gain`, `leaf=nba` | 31,718 | 13,594 | **64.34%** | 63.94% | 63.54% | **63.58%** | 64.28% | 19 nós (4 folhas) |
| **Forest Covertype** | `grace=50`, `delta=0.001`, `tau=0.1`, `crit=gini`, `leaf=nba` | 17,500 | 7,500 | **64.17%** | 63.47% | 61.46% | **62.02%** | 58.94% | 47 nós (4 folhas) |
| **Airlines** | `grace=50`, `delta=0.001`, `tau=0.1`, `crit=gini`, `leaf=nba` | 17,500 | 7,500 | **77.19%** | 67.14% | 56.63% | **56.64%** | 76.14% | 33 nós (4 folhas) |

---

## 2. Desempenho em Regressão Contínua

| Dataset | Hiperparâmetros & Tratamento | Treino | Teste | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Preq. MAE | Preq. RMSE | Preq. $R^2$ | Topologia (Nós) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | `grace=200`, `delta=1e-07`, `tau=0.01`, `leaf=model`, `trat=StandardScaler` | 14,000 | 6,000 | **2.4935** | 3.1356 | **0.6044** | 2.4594 | 3.1766 | 0.5895 | 33 nós (4 folhas) |
| **Bikes** | `grace=200`, `delta=1e-07`, `tau=0.15`, `leaf=mean`, `trat=StandardScaler` | 12,165 | 5,214 | **52.4736** | 73.5308 | **0.0411** | 67.1717 | 94.1011 | 0.3969 | 21 nós (4 folhas) |
| **Appliances Energy** | `grace=100`, `delta=1e-07`, `tau=0.01`, `leaf=adaptive`, `trat=LeafNorm` | 13,814 | 5,921 | **38.3687** | 49.0520 | **0.5695** | 43.3148 | 462.2925 | -36.7363 | 31 nós (4 folhas) |
| **NYC Taxi** | `grace=50`, `delta=1e-07`, `tau=0.1`, `leaf=adaptive`, `trat=LeafNorm` | 10,500 | 4,500 | **355.5271** | 463.2528 | **0.1229** | 385.0259 | 2117.5035 | -16.8780 | 35 nós (4 folhas) |

---

## 3. Métricas de Performance, Latência e Recursos em Hardware Emulado (QEMU @ 50 MHz)

> **Nota sobre as Colunas de Latência:**
> - **Ciclo Test-Then-Train:** Cada medição cobre o tempo para predição (`predict`) + treino incremental (`learn_one`) daquela amostra.
> - **Cronometria por Ciclos:** As latências são obtidas em ciclos de clock de CPU via registrador SysTick ARMv7-M e convertidas para microssegundos (µs) na frequência nominal de 50 MHz (µs = ciclos / 50).
> - **Throughput Real:** Reflete a capacidade do núcleo Cortex-M3 de 50 MHz de processar amostras em streaming contínuo sem estouro de pilha.


| Dataset | Tarefa | Latência Média | Latência p95 | Latência p99 | Throughput (Vazão) | Model Footprint | Consumo RAM (SRAM / Heap) | FLOPs / Amostra |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classificação | **6.88 µs** | 20.88 µs | 27.34 µs | **145,349 amostras/s** | 564 B (19 nós) | 49.7 KB (.bss) / **0 B Heap** | **69 FLOPs** |
| **Electricity** | Classificação | **23.91 µs** | 44.24 µs | 86.12 µs | **41,824 amostras/s** | 564 B (19 nós) | 49.7 KB (.bss) / **0 B Heap** | **149 FLOPs** |
| **Forest Covertype** | Classificação | **99.34 µs** | 396.48 µs | 709.32 µs | **10,066 amostras/s** | 1,348 B (47 nós) | 49.7 KB (.bss) / **0 B Heap** | **2250 FLOPs** |
| **Airlines** | Classificação | **8.20 µs** | 31.64 µs | 45.68 µs | **121,951 amostras/s** | 956 B (33 nós) | 49.7 KB (.bss) / **0 B Heap** | **133 FLOPs** |
| **Friedman #1** | Regressão | **9.31 µs** | 19.64 µs | 33.48 µs | **107,411 amostras/s** | 956 B (33 nós) | 49.7 KB (.bss) / **0 B Heap** | **138 FLOPs** |
| **Bikes** | Regressão | **8.05 µs** | 17.12 µs | 30.28 µs | **124,224 amostras/s** | 620 B (21 nós) | 49.7 KB (.bss) / **0 B Heap** | **160 FLOPs** |
| **Appliances Energy** | Regressão | **41.57 µs** | 153.82 µs | 255.16 µs | **24,056 amostras/s** | 900 B (31 nós) | 49.7 KB (.bss) / **0 B Heap** | **336 FLOPs** |
| **NYC Taxi** | Regressão | **9.20 µs** | 42.10 µs | 50.86 µs | **108,696 amostras/s** | 1,012 B (35 nós) | 49.7 KB (.bss) / **0 B Heap** | **105 FLOPs** |

---

## 4. Conclusões da Emulação no ARM Cortex-M3

1. **Compatibilidade com 64 KB de SRAM:** O perfil Tier 4 Universal permitiu treinar e inferir todos os 8 datasets em um único binário com **50.9 KB de BSS estática**, operando com folga dentro do limite físico de 64 KB da Stellaris LM3S6965.

2. **Zero Alocação Dinâmica:** Comprovado $\Delta\text{Heap} = 0\text{ bytes}$ durante todo o ciclo de treino online e inferência holdout, garantindo determinismo temporal estrito e ausência de fragmentação de memória.

3. **Eficiência Sub-milissegundo @ 50 MHz:** Mesmo operando a 50 MHz sem FPU dedicada (emulação em software de ponto flutuante IEEE 754), o microcontrolador atingiu vazões de milhares de amostras por segundo em fluxos contínuos.

