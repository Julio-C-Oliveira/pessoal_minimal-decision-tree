# Main Performance Benchmark Suite (`main_benchmark`)

Esta pasta contém a **suíte principal de benchmark de desempenho** da árvore de decisão incremental em C11 pura (`libht`).

A suíte realiza uma avaliação tripla direta (*apples-to-apples*) comparando:
1. **C11 Hoeffding Tree (`float32`, $\le 1\text{ MB}$, Zero Malloc)**
2. **C11 Hoeffding Tree (`float64`, Verificação de Paridade)**
3. **River (`online-ml/river`)** — Referência de aprendizado em fluxo contínuo em Python/Cython
4. **Scikit-Learn (`sklearn.tree`)** — Referência batch CART otimizada

---

## 1. Datasets Selecionados (8 Datasets)

A suíte é dividida balanceadamente entre **Classificação** (4 fluxos) e **Regressão Contínua** (4 fluxos), cobrindo dados sintéticos de referência e conjuntos de dados reais com padrões temporais:

### Classificação (4 fluxos):
| Dataset | Tipo | Amostras | Features | Classes | Descrição / Desafio |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SEA** | Sintético | $20.000$ | 3 | 2 | Padrão da literatura (Street & Kim, 2001) para avaliar splits e *concept drift* |
| **Electricity (Elec2)** | Real | $45.312$ | 8 | 2 | Flutuações de preços de energia na Austrália; drift real de mercado e sazonalidade |
| **Forest Covertype** | Real | $25.000$ | 54 | 7 | Alta dimensionalidade (54 features contínuas/one-hot) e 7 classes florestais |
| **Airlines** | Real | $25.000$ | 7 | 2 | Previsão de atrasos em voos comerciais comerciais com variáveis temporais |

### Regressão Contínua (4 fluxos):
| Dataset | Tipo | Amostras | Features | Alvo ($y$) | Descrição / Desafio |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Friedman #1** | Sintético | $20.000$ | 10 | Contínuo | Benchmark clássico não-linear (Friedman, 1991) com interações trigonométricas |
| **Bikes** | Real | $17.379$ | 12 | Contínuo | Demanda horária de aluguel de bicicletas com fortes padrões diários e sazonais |
| **Appliances Energy**| Real | $19.735$ | 28 | Contínuo (Wh)| Telemetria IoT de sensores ambientais em residência de baixo consumo de energia |
| **NYC Taxi** | Real | $15.000$ | 7 | Contínuo (s) | Duração de corridas de táxi em Nova York a partir de coordenadas espaciais |

---

## 2. Estrutura de Arquivos

```
main_benchmark/
├── main_benchmark.py           # Orquestrador principal da avaliação tripla (C11 vs River vs Scikit-Learn)
├── prepare_datasets.py         # Gerador e formatador padronizado dos 8 datasets
├── README.md                   # Este documento explicativo
├── main_benchmark_report.md    # Relatório detalhado gerado com tabelas de acurácia, nós, RAM e speedup
├── runners/                    # Scripts isolados de execução para Scikit-Learn e River
└── data/                       # Arquivos CSV formatados com metadados para ingestão direta
    ├── sea_train.csv / sea_test.csv
    ├── electricity_train.csv / electricity_test.csv
    ├── covertype_train.csv / covertype_test.csv
    ├── airlines_train.csv / airlines_test.csv
    ├── friedman_train.csv / friedman_test.csv
    ├── bikes_train.csv / bikes_test.csv
    ├── appliances_energy_train.csv / appliances_energy_test.csv
    └── nyc_taxi_train.csv / nyc_taxi_test.csv
```

---

## 3. Como Executar

### Via Makefile (Recomendado)
Na raiz do repositório:
```bash
make bench-main
```
Ou dentro da pasta `code/`:
```bash
make -C code bench-main
```

### Execução Direta via Python
Para executar todos os 8 datasets e gerar o relatório Markdown:
```bash
./.venv/bin/python main_benchmark/main_benchmark.py
```

Para executar um único dataset específico (ex: SEA ou Bikes):
```bash
./.venv/bin/python main_benchmark/main_benchmark.py --dataset sea
./.venv/bin/python main_benchmark/main_benchmark.py --dataset bikes
```

Para re-gerar ou atualizar os arquivos CSV de dados:
```bash
./.venv/bin/python main_benchmark/prepare_datasets.py
```
*(Para tentar baixar versões remotas completas do River/OpenML caso haja conexão de rede, utilize `--download`)*.

---

## 4. Métricas Avaliadas

1. **Acurácia Holdout e Prequential:**
   - Classificação: Acurácia (%) e Macro F1-Score (%).
   - Regressão: MAE, RMSE e Coeficiente de Determinação ($R^2$).
2. **Topologia:**
   - Contagem total de nós (`n_nodes`) e folhas ativas (`n_active_leaves`).
3. **Memória RAM:**
   - Verificação de **Zero Dynamic Malloc (0.0 KB)** em runtime no C11.
   - Footprint estrutural estático estritamente $\le 1\text{ MB}$ ($473\text{ KB}$ a $524\text{ KB}$ em `float32`).
4. **Throughput e Latência:**
   - Tempo médio de processamento por amostra ($\mu\text{s}$/sample).
   - Speedup computacional do motor C11 em relação ao River em Python.
