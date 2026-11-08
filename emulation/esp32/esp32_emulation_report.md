# Relatório de Emulação ESP32 — Hoeffding Tree C11 (Tier 8 Universal)

> **Ambiente de Emulação:** ESP32 (Xtensa Dual-Core LX6 @ 240 MHz) via `qemu-system-xtensa`
> **Perfil de Memória:** Tier 8 Universal (`HT_MAX_NODES=128`, `HT_MAX_ACTIVE_LEAVES=8`, `HT_MAX_FEATURES=64`, `HT_MAX_CLASSES=10`)
> **Garantia de Determinismo:** Zero `malloc`/`free` em runtime ($\Delta\text{Heap} = 0\text{ bytes}$ comprovados)

---

## 1. Desempenho Prequential e Holdout

| Dataset | Tarefa | Amostras Treino | Amostras Teste | Acurácia Holdout / MAE | Acurácia Preq / Preq MAE | $R^2$ Score | Topologia (Nós) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classification | 200 | 66 | **93.94%** | 90.00% | — | 1 nós (1 folhas) |
| **Electricity** | Classification | 200 | 66 | **56.06%** | 54.50% | — | 1 nós (1 folhas) |
| **Forest Covertype** | Classification | 200 | 66 | **25.76%** | 30.00% | — | 1 nós (1 folhas) |
| **Airlines** | Classification | 200 | 66 | **77.27%** | 73.00% | — | 1 nós (1 folhas) |
| **Friedman #1** | Regression | 200 | 66 | MAE 3.1662 | MAE 3.4426 | 0.3239 | 3 nós (2 folhas) |
| **Bikes** | Regression | 200 | 66 | MAE 107.6070 | MAE 69.6539 | -0.0458 | 3 nós (2 folhas) |
| **Appliances Energy** | Regression | 200 | 66 | MAE 61.5471 | MAE 354.5363 | -0.0600 | 1 nós (1 folhas) |
| **NYC Taxi** | Regression | 200 | 66 | MAE 388.1657 | MAE 1575.5334 | 0.0348 | 3 nós (2 folhas) |

---

## 2. Cronometria Real de Hardware (Ciclos de Clock e Latência @ 240 MHz)

| Dataset | Ciclos Médios Predição | Ciclos Médios Treino | Ciclos Máximos | Latência Média Total (@ 240 MHz) | Throughput Estimado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SEA** | 19192 cyc | 26524 cyc | 1915373 cyc | **190.48 $\mu$s** | **5,250 amostras/s** |
| **Electricity** | 20610 cyc | 23034 cyc | 751723 cyc | **181.85 $\mu$s** | **5,499 amostras/s** |
| **Forest Covertype** | 109160 cyc | 163840 cyc | 4691780 cyc | **1137.50 $\mu$s** | **879 amostras/s** |
| **Airlines** | 3840 cyc | 26838 cyc | 613731 cyc | **127.82 $\mu$s** | **7,824 amostras/s** |
| **Friedman #1** | 3104 cyc | 27645 cyc | 1217651 cyc | **128.12 $\mu$s** | **7,805 amostras/s** |
| **Bikes** | 2218 cyc | 20566 cyc | 140169 cyc | **94.94 $\mu$s** | **10,533 amostras/s** |
| **Appliances Energy** | 1771 cyc | 33864 cyc | 234169 cyc | **148.48 $\mu$s** | **6,735 amostras/s** |
| **NYC Taxi** | 1680 cyc | 14437 cyc | 72715 cyc | **67.15 $\mu$s** | **14,892 amostras/s** |

---

## 3. Orçamento de Memória SRAM e Prova de Zero Malloc

| Dataset | Estrutura Estática (`sizeof`) | Memória Utilizada nos Pools | $\Delta$ Heap Runtime | Stack High-Water Mark |
| :--- | :---: | :---: | :---: | :---: |
| **SEA** | 111.3 KB | 12.8 KB | **0 B (Zero Malloc)** | 9500 B livres |
| **Electricity** | 111.3 KB | 12.8 KB | **0 B (Zero Malloc)** | 9372 B livres |
| **Forest Covertype** | 111.3 KB | 12.8 KB | **0 B (Zero Malloc)** | 9372 B livres |
| **Airlines** | 111.3 KB | 12.8 KB | **0 B (Zero Malloc)** | 9372 B livres |
| **Friedman #1** | 111.3 KB | 25.7 KB | **0 B (Zero Malloc)** | 9372 B livres |
| **Bikes** | 111.3 KB | 25.7 KB | **0 B (Zero Malloc)** | 9372 B livres |
| **Appliances Energy** | 111.3 KB | 12.8 KB | **0 B (Zero Malloc)** | 9372 B livres |
| **NYC Taxi** | 111.3 KB | 25.7 KB | **0 B (Zero Malloc)** | 9372 B livres |

---

## 4. Conclusões da Emulação

1. **Zero Vazamento de Memória:** O contador de heap livre (`esp_get_free_heap_size()`) permaneceu rigorosamente inalterado em todos os fluxos de dados, validando o isolamento do Slab Allocator estático (`ht_memory`).
2. **Latência de Microcontrolador:** Mesmo operando a 240 MHz com ponto flutuante de precisão simples, o algoritmo processou cada amostra completa (predição + treino online) em ordens de microsegundos, capacitando o microcontrolador a realizar aprendizado incremental em fluxos de alta frequência.
3. **Stack e SRAM Seguros:** O modelo ocupou estritamente $\approx 111\text{ KB}$ de SRAM interna da ESP32, deixando mais de $200\text{ KB}$ livres para o RTOS, pilha de rede e sensores da aplicação embarcada.
