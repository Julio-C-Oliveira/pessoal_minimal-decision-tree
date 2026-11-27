# Relatório de Emulação ARM Cortex-M3 (TI Stellaris LM3S6965) — Hoeffding Tree C11

> **Ambiente de Emulação:** TI Stellaris LM3S6965 (ARM Cortex-M3 @ 50 MHz) via `qemu-system-arm -M lm3s6965evb`
> **Aritmética / Precisão:** Ponto Fixo Q16.16 Zero-Libm (`-DHT_USE_FIXED_POINT`, sem `-lm` e sem `_printf_float`)
> **Perfil de Memória:** Tier 4 Universal (`HT_MAX_NODES=128`, `HT_MAX_ACTIVE_LEAVES=4`, `HT_MAX_FEATURES=64`, `HT_MAX_CLASSES=8`)
> **Restrição Física de SRAM:** 64 KB de SRAM interna (`0x20000000 - 0x20010000`). Firmware bare-metal ocupa ~50.9 KB (.bss), deixando > 6 KB livres.
> **Hiperparâmetros & Tratamentos:** Extraídos diretamente do Rank #1 oficial de `main_benchmark/hyperparam_report.md` com pré-processamento online ativado.
> **Garantia de Determinismo:** Zero `malloc`/`free` em runtime ($\Delta\text{Heap} = 0\text{ bytes}$ comprovados)

---

## 1. Desempenho em Classificação

| Dataset | Hiperparâmetros (Tier 4) | Treino | Teste | Acurácia Holdout | Precisão Macro | Recall Macro | F1-Score Macro | Acurácia Preq. | Topologia (Nós) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | `grace=400`, `delta=0.0001`, `tau=0.15`, `crit=info_gain`, `leaf=nba` | 14,000 | 6,000 | **96.82%** | 96.79% | 95.87% | **96.31%** | 96.56% | 21 nós (4 folhas) |
| **Electricity** | `grace=100`, `delta=1e-07`, `tau=0.05`, `crit=info_gain`, `leaf=nba` | 31,718 | 13,594 | **64.26%** | 63.85% | 63.50% | **63.54%** | 64.20% | 17 nós (4 folhas) |
| **Forest Covertype** | `grace=50`, `delta=0.001`, `tau=0.1`, `crit=gini`, `leaf=nba` | 17,500 | 7,500 | **64.44%** | 65.47% | 61.04% | **62.08%** | 59.24% | 49 nós (4 folhas) |
| **Airlines** | `grace=50`, `delta=0.001`, `tau=0.1`, `crit=gini`, `leaf=nba` | 17,500 | 7,500 | **77.19%** | 67.14% | 56.63% | **56.64%** | 76.16% | 33 nós (4 folhas) |

---

## 2. Desempenho em Regressão Contínua

| Dataset | Hiperparâmetros & Tratamento | Treino | Teste | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Preq. MAE | Preq. RMSE | Preq. $R^2$ | Topologia (Nós) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | `grace=200`, `delta=1e-07`, `tau=0.01`, `leaf=model`, `trat=StandardScaler` | 14,000 | 6,000 | **2.3755** | 2.9858 | **0.6413** | 2.3876 | 3.0655 | 0.6177 | 27 nós (4 folhas) |
| **Bikes** | `grace=200`, `delta=1e-07`, `tau=0.15`, `leaf=mean`, `trat=StandardScaler` | 12,165 | 5,214 | **58.4292** | 82.9373 | **-0.2199** | 62.0922 | 90.3977 | 0.4435 | 21 nós (4 folhas) |
| **Appliances Energy** | `grace=100`, `delta=1e-07`, `tau=0.01`, `leaf=adaptive`, `trat=LeafNorm` | 13,814 | 5,921 | **37.4663** | 47.6447 | **0.5939** | 42.9370 | 282.5730 | -13.0990 | 31 nós (4 folhas) |
| **NYC Taxi** | `grace=50`, `delta=1e-07`, `tau=0.1`, `leaf=adaptive`, `trat=LeafNorm` | 10,500 | 4,500 | **354.1386** | 463.3722 | **0.1225** | 368.3316 | 572.4643 | -0.3067 | 29 nós (4 folhas) |

---

## 3. Métricas de Performance, Latência e Recursos em Hardware Emulado (QEMU @ 50 MHz)

> **Nota sobre as Colunas de Latência:**
> - **Ciclo Test-Then-Train:** Cada medição cobre o tempo para predição (`predict`) + treino incremental (`learn_one`) daquela amostra.
> - **Cronometria por Ciclos:** As latências são obtidas em ciclos de clock de CPU via registrador SysTick ARMv7-M e convertidas para microssegundos (µs) na frequência nominal de 50 MHz (µs = ciclos / 50).
> - **Throughput Real:** Reflete a capacidade do núcleo Cortex-M3 de 50 MHz de processar amostras em streaming contínuo sem estouro de pilha.


| Dataset | Tarefa | Latência Média | Latência p95 | Latência p99 | Throughput (Vazão) | Model Footprint | Consumo RAM (SRAM / Heap) | FLOPs / Amostra |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classificação | **8.53 µs** | 24.16 µs | 43.40 µs | **117,233 amostras/s** | 494 B (21 nós) | 48.5 KB (.bss) / **0 B Heap** | **69 FLOPs** |
| **Electricity** | Classificação | **26.14 µs** | 52.38 µs | 151.04 µs | **38,256 amostras/s** | 406 B (17 nós) | 48.5 KB (.bss) / **0 B Heap** | **149 FLOPs** |
| **Forest Covertype** | Classificação | **95.84 µs** | 430.64 µs | 775.60 µs | **10,434 amostras/s** | 1,110 B (49 nós) | 48.5 KB (.bss) / **0 B Heap** | **2250 FLOPs** |
| **Airlines** | Classificação | **11.51 µs** | 37.20 µs | 89.36 µs | **86,881 amostras/s** | 758 B (33 nós) | 48.5 KB (.bss) / **0 B Heap** | **133 FLOPs** |
| **Friedman #1** | Regressão | **9.17 µs** | 21.90 µs | 47.72 µs | **109,051 amostras/s** | 626 B (27 nós) | 48.5 KB (.bss) / **0 B Heap** | **138 FLOPs** |
| **Bikes** | Regressão | **8.38 µs** | 19.64 µs | 39.26 µs | **119,332 amostras/s** | 494 B (21 nós) | 48.5 KB (.bss) / **0 B Heap** | **160 FLOPs** |
| **Appliances Energy** | Regressão | **24.85 µs** | 96.08 µs | 151.38 µs | **40,241 amostras/s** | 714 B (31 nós) | 48.5 KB (.bss) / **0 B Heap** | **336 FLOPs** |
| **NYC Taxi** | Regressão | **7.48 µs** | 27.12 µs | 52.52 µs | **133,690 amostras/s** | 670 B (29 nós) | 48.5 KB (.bss) / **0 B Heap** | **105 FLOPs** |

---

## 4. Comparativo Direto A/B no ARM Cortex-M3 (`float32 soft-fp` vs `Q16.16 Zero-Libm`)

### 4.1 Footprint Binário ELF (`arm-none-eabi-size -Os` + Auditoria `arm-none-eabi-nm`)

| Seção / Auditoria | `float32 soft-fp` (`-lm` + `_printf_float`) | `Q16.16 Zero-Libm` (`-DHT_USE_FIXED_POINT`) | Variação |
| :--- | :---: | :---: | :---: |
| **Flash (`.text`)** | `66,220 B` (`64.7 KB`) | **`49,264 B` (`48.1 KB`)** | **`-16,956 B` (`-25.6%`)** |
| **`.data` (SRAM inicializada)** | `468 B` | **`104 B`** | **`-364 B` (`-77.8%`)** |
| **`.bss` (SRAM estática)** | `57,000 B` | **`57,000 B`** | `0 B` (idêntico) |
| **Total (`dec`)** | `123,688 B` | **`106,368 B`** | **`-17,320 B` (`-14.0%`)** |
| **Símbolos `soft-fp` / `libm` / `_printf_float`** | Presentes | **0 (PASS via `arm-none-eabi-nm`)** | **100% eliminados** |

### 4.2 Latência SysTick @ 50 MHz, Speedup e Qualidade Preditiva (8 Datasets)

| Dataset | Tarefa | `float32` Média | `Q16.16` Média | **Speedup** | `float32` p99 | `Q16.16` p99 | `float32` Holdout | `Q16.16` Holdout |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Cls | `20.08 µs` | **`8.53 µs`** | **`2.35×`** | `85.62 µs` | **`43.40 µs`** | `97.30%` | `96.82%` |
| **Electricity** | Cls | `65.57 µs` | **`26.14 µs`** | **`2.51×`** | `374.12 µs` | **`151.04 µs`** | `64.34%` | **`64.26%`** |
| **Forest Covertype** | Cls | `285.41 µs` | **`95.84 µs`** | **`2.98×`** | `1901.28 µs` | **`775.60 µs`** | `64.17%` | **`64.44%`** |
| **Airlines** | Cls | `21.46 µs` | **`11.51 µs`** | **`1.86×`** | `114.76 µs` | **`89.36 µs`** | `77.19%` | **`77.19%`** |
| **Friedman #1** | Reg | `22.68 µs` | **`9.17 µs`** | **`2.47×`** | `73.42 µs` | **`47.72 µs`** | `MAE 2.4935` (`R² 0.6044`) | **`MAE 2.3755` (`R² 0.6413`)** |
| **Bikes** | Reg | `21.15 µs` | **`8.38 µs`** | **`2.52×`** | `65.62 µs` | **`39.26 µs`** | `MAE 52.47` (`Preq 67.17`) | `MAE 58.43` (**`Preq 62.09`**) |
| **Appliances Energy** | Reg | `111.22 µs` | **`24.85 µs`** | **`4.48×`** | `632.64 µs` | **`151.38 µs`** | `MAE 38.37` (`R² 0.5695`) | **`MAE 37.47` (`R² 0.5939`)** |
| **NYC Taxi** | Reg | `27.83 µs` | **`7.48 µs`** | **`3.72×`** | `202.48 µs` | **`52.52 µs`** | `MAE 355.53` (`R² 0.1229`) | **`MAE 354.14` (`R² 0.1225`)** |

---

## 5. Conclusões da Emulação no ARM Cortex-M3

1. **Compatibilidade com 64 KB de SRAM:** O perfil Tier 4 Universal permitiu treinar e inferir todos os 8 datasets em um único binário com **57.0 KB de BSS estática total** (`48.5 KB` da `ht_tree_t`), operando com folga dentro do limite físico de 64 KB da Stellaris LM3S6965.

2. **Zero Alocação Dinâmica:** Comprovado $\Delta\text{Heap} = 0\text{ bytes}$ durante todo o ciclo de treino online e inferência holdout, garantindo determinismo temporal estrito e ausência de fragmentação de memória.

3. **Eficiência Zero-Libm em Ponto Fixo Q16.16 @ 50 MHz:** Operando com aritmética 100% inteira (zero símbolos `__aeabi_f*`/`__aeabi_d*`, zero `_printf_float`, zero `-lm`, `-25.6%` de Flash), o Cortex-M3 atingiu vazões de **10.434 a 133.690 amostras/s** (`1.86×` a `4.48×` mais rápido que `float32 soft-fp`, média `2.86×`).

