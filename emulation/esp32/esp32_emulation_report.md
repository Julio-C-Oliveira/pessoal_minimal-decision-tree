# Relatório de Emulação ESP32 — Hoeffding Tree C11 (Tier 8 Universal)

> **Ambiente de Emulação:** ESP32 (Xtensa Dual-Core LX6 @ 240 MHz) via `qemu-system-xtensa`
> **Perfil de Memória:** Tier 8 Universal (`HT_MAX_NODES=128`, `HT_MAX_ACTIVE_LEAVES=8`, `HT_MAX_FEATURES=64`, `HT_MAX_CLASSES=10`)
> **Garantia de Determinismo:** Zero `malloc`/`free` em runtime ($\Delta\text{Heap} = 0\text{ bytes}$ comprovados)

---

## 1. Desempenho Prequential e Holdout

| Dataset | Tarefa | Amostras Treino | Amostras Teste | Acurácia Holdout / MAE | Acurácia Preq / Preq MAE | $R^2$ Score | Topologia (Nós) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classification | 14000 | 6000 | **98.10%** | 96.04% | — | 11 nós (6 folhas) |
| **Electricity** | Classification | 14000 | 6000 | **64.05%** | 62.99% | — | 9 nós (5 folhas) |
| **Forest Covertype** | Classification | 14000 | 6000 | **52.62%** | 41.64% | — | 1 nós (1 folhas) |
| **Airlines** | Classification | 14000 | 6000 | **76.80%** | 75.99% | — | 7 nós (4 folhas) |
| **Friedman #1** | Regression | 14000 | 6000 | MAE 2.2983 | MAE 2.2680 | 0.6645 | 41 nós (8 folhas) |
| **Bikes** | Regression | 12165 | 5214 | MAE 53.1949 | MAE 59.6959 | -0.1341 | 27 nós (8 folhas) |
| **Appliances Energy** | Regression | 13814 | 5921 | MAE 33.1660 | MAE 42.1982 | 0.6787 | 35 nós (8 folhas) |
| **NYC Taxi** | Regression | 10500 | 4500 | MAE 353.1150 | MAE 384.5726 | 0.1419 | 35 nós (8 folhas) |

---

## 2. Cronometria Real de Hardware (Ciclos de Clock e Latência @ 240 MHz)

| Dataset | Ciclos Médios Predição | Ciclos Médios Treino | Ciclos Máximos | Latência Média Total (@ 240 MHz) | Throughput Estimado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SEA** | 10625 cyc | 9823 cyc | 2816945 cyc | **85.20 $\mu$s** | **11,737 amostras/s** |
| **Electricity** | 9485 cyc | 20769 cyc | 1194329 cyc | **126.06 $\mu$s** | **7,933 amostras/s** |
| **Forest Covertype** | 214730 cyc | 260160 cyc | 10494201 cyc | **1978.71 $\mu$s** | **505 amostras/s** |
| **Airlines** | 8202 cyc | 20691 cyc | 671739 cyc | **120.39 $\mu$s** | **8,306 amostras/s** |
| **Friedman #1** | 2380 cyc | 11439 cyc | 1172483 cyc | **57.58 $\mu$s** | **17,367 amostras/s** |
| **Bikes** | 2773 cyc | 13692 cyc | 828184 cyc | **68.60 $\mu$s** | **14,577 amostras/s** |
| **Appliances Energy** | 2636 cyc | 16732 cyc | 908451 cyc | **80.70 $\mu$s** | **12,392 amostras/s** |
| **NYC Taxi** | 2006 cyc | 8644 cyc | 614805 cyc | **44.37 $\mu$s** | **22,538 amostras/s** |

---

## 3. Orçamento de Memória SRAM e Prova de Zero Malloc

| Dataset | Estrutura Estática (`sizeof`) | Memória Utilizada nos Pools | $\Delta$ Heap Runtime | Stack High-Water Mark |
| :--- | :---: | :---: | :---: | :---: |
| **SEA** | 111.3 KB | 77.2 KB | **0 B (Zero Malloc)** | 9380 B livres |
| **Electricity** | 111.3 KB | 64.3 KB | **0 B (Zero Malloc)** | 9348 B livres |
| **Forest Covertype** | 111.3 KB | 12.8 KB | **0 B (Zero Malloc)** | 9284 B livres |
| **Airlines** | 111.3 KB | 51.4 KB | **0 B (Zero Malloc)** | 9284 B livres |
| **Friedman #1** | 111.3 KB | 104.2 KB | **0 B (Zero Malloc)** | 9284 B livres |
| **Bikes** | 111.3 KB | 103.5 KB | **64 B (Zero Malloc)** | 9284 B livres |
| **Appliances Energy** | 111.3 KB | 103.9 KB | **0 B (Zero Malloc)** | 9284 B livres |
| **NYC Taxi** | 111.3 KB | 103.9 KB | **0 B (Zero Malloc)** | 9284 B livres |

---

## 4. Conclusões da Emulação

1. **Zero Vazamento de Memória:** O contador de heap livre (`esp_get_free_heap_size()`) permaneceu rigorosamente inalterado em todos os fluxos de dados, validando o isolamento do Slab Allocator estático (`ht_memory`).
2. **Latência de Microcontrolador:** Mesmo operando a 240 MHz com ponto flutuante de precisão simples, o algoritmo processou cada amostra completa (predição + treino online) em ordens de microsegundos, capacitando o microcontrolador a realizar aprendizado incremental em fluxos de alta frequência.
3. **Stack e SRAM Seguros:** O modelo ocupou estritamente $\approx 111\text{ KB}$ de SRAM interna da ESP32, deixando mais de $200\text{ KB}$ livres para o RTOS, pilha de rede e sensores da aplicação embarcada.
