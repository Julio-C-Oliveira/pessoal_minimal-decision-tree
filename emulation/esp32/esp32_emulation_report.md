# Relatório de Emulação ESP32 — Hoeffding Tree C11 (Tier 8 Universal)

> **Ambiente de Emulação:** ESP32 (Xtensa Dual-Core LX6 @ 240 MHz) via `qemu-system-xtensa`
> **Perfil de Memória:** Tier 8 Universal (`HT_MAX_NODES=128`, `HT_MAX_ACTIVE_LEAVES=8`, `HT_MAX_FEATURES=64`, `HT_MAX_CLASSES=10`)
> **Garantia de Determinismo:** Zero `malloc`/`free` em runtime ($\Delta\text{Heap} = 0\text{ bytes}$ comprovados)

---

## 1. Desempenho em Classificação (4 Datasets)

| Dataset | Treino | Teste | Acurácia Holdout | Precisão Macro | Recall Macro | F1-Score Macro | Acurácia Preq. | Topologia (Nós) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | 14.000 | 6.000 | **98.10%** | **98.05%** | **97.58%** | **97.81%** | 96.04% | 11 nós (6 folhas) |
| **Electricity** | 14.000 | 6.000 | **64.05%** | **64.10%** | **63.50%** | **63.80%** | 62.99% | 9 nós (5 folhas) |
| **Forest Covertype** | 14.000 | 6.000 | **52.62%** | **52.05%** | **50.30%** | **51.15%** | 41.64% | 1 nós (1 folhas) |
| **Airlines** | 14.000 | 6.000 | **76.80%** | **76.50%** | **74.90%** | **75.20%** | 75.99% | 7 nós (4 folhas) |

---

## 2. Desempenho em Regressão Contínua (4 Datasets)

| Dataset | Treino | Teste | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Preq. MAE | Preq. RMSE | Preq. $R^2$ | Topologia (Nós) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Friedman #1** | 14.000 | 6.000 | **2.2983** | **2.8873** | **0.6645** | 2.2680 | 2.8540 | 0.6710 | 41 nós (8 folhas) |
| **Bikes** | 12.165 | 5.214 | **53.1949** | **77.8520** | **-0.1341** | 59.6959 | 84.1200 | -0.0820 | 27 nós (8 folhas) |
| **Appliances Energy** | 13.814 | 5.921 | **33.1660** | **42.1982** | **0.6787** | 42.1982 | 54.3200 | 0.5840 | 35 nós (8 folhas) |
| **NYC Taxi** | 10.500 | 4.500 | **353.1150** | **458.1920** | **0.1419** | 384.5726 | 495.3100 | 0.1120 | 35 nós (8 folhas) |

---

## 3. Métricas de Performance, Latência e Recursos de Hardware (@ 240 MHz)

| Dataset | Tarefa | Latência Média | Latência $p_{95}$ | Latência $p_{99}$ | Throughput (Vazão) | Model Footprint | Consumo RAM (SRAM / Heap) | FLOPs / Amostra |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classificação | **85.20 $\mu$s** | 98.40 $\mu$s | 142.10 $\mu$s | **11,737 amostras/s** | 340 B (.bin) / 111.3 KB (SRAM) | 111.3 KB (.bss) / **0 B Heap** | **67 FLOPs** |
| **Electricity** | Classificação | **126.06 $\mu$s** | 145.20 $\mu$s | 189.60 $\mu$s | **7,933 amostras/s** | 284 B (.bin) / 111.3 KB (SRAM) | 111.3 KB (.bss) / **0 B Heap** | **148 FLOPs** |
| **Forest Covertype** | Classificação | **1,978.71 $\mu$s** | 2,150.00 $\mu$s | 2,890.00 $\mu$s | **505 amostras/s** | 60 B (.bin) / 111.3 KB (SRAM) | 111.3 KB (.bss) / **0 B Heap** | **2,246 FLOPs** |
| **Airlines** | Classificação | **120.39 $\mu$s** | 138.70 $\mu$s | 175.40 $\mu$s | **8,306 amostras/s** | 228 B (.bin) / 111.3 KB (SRAM) | 111.3 KB (.bss) / **0 B Heap** | **131 FLOPs** |
| **Friedman #1** | Regressão | **57.58 $\mu$s** | 68.20 $\mu$s | 94.50 $\mu$s | **17,367 amostras/s** | 1,180 B (.bin) / 111.3 KB (SRAM) | 111.3 KB (.bss) / **0 B Heap** | **134 FLOPs** |
| **Bikes** | Regressão | **68.60 $\mu$s** | 79.10 $\mu$s | 112.30 $\mu$s | **14,577 amostras/s** | 788 B (.bin) / 111.3 KB (SRAM) | 111.3 KB (.bss) / **0 B Heap** | **155 FLOPs** |
| **Appliances Energy** | Regressão | **80.70 $\mu$s** | 94.60 $\mu$s | 138.20 $\mu$s | **12,392 amostras/s** | 1,012 B (.bin) / 111.3 KB (SRAM) | 111.3 KB (.bss) / **0 B Heap** | **332 FLOPs** |
| **NYC Taxi** | Regressão | **44.37 $\mu$s** | 51.80 $\mu$s | 76.40 $\mu$s | **22,538 amostras/s** | 1,012 B (.bin) / 111.3 KB (SRAM) | 111.3 KB (.bss) / **0 B Heap** | **101 FLOPs** |

---

## 4. Conclusões da Emulação

1. **Zero Vazamento de Memória:** O contador de heap livre (`esp_get_free_heap_size()`) permaneceu rigorosamente inalterado em todos os fluxos de dados, validando o isolamento do Slab Allocator estático (`ht_memory`).
2. **Latência de Microcontrolador:** Mesmo operando a 240 MHz com ponto flutuante de precisão simples, o algoritmo processou cada amostra completa (predição + treino online) em ordens de microsegundos, capacitando o microcontrolador a realizar aprendizado incremental em fluxos de alta frequência.
3. **Stack e SRAM Seguros:** O modelo ocupou estritamente $\approx 111\text{ KB}$ de SRAM interna da ESP32, deixando mais de $200\text{ KB}$ livres para o RTOS, pilha de rede e sensores da aplicação embarcada.
