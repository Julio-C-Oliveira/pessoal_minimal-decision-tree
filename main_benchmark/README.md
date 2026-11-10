# Main Performance Benchmark Suite (`main_benchmark`)

Esta pasta contém a **suíte principal de benchmark de desempenho** do **kansho** (`libht` em C11 puro).

A suíte realiza uma avaliação tripla direta (*apples-to-apples*) comparando:
1. **C11 Hoeffding Tree (`float32`, $\le 1\text{ MB}$, Zero Malloc)**
2. **C11 Hoeffding Tree (`float64`, Verificação de Paridade)**
3. **River (`online-ml/river`)** — Referência de aprendizado em fluxo contínuo em Python/Cython
4. **Scikit-Learn (`sklearn.tree`)** — Referência batch CART otimizada

---

## 1. Datasets Selecionados (8 Datasets)

A suíte é dividida balanceadamente entre **Classificação** (4 fluxos) e **Regressão Contínua** (4 fluxos), cobrindo dados sintéticos de referência e conjuntos de dados reais com padrões temporais:

### Classificação (4 fluxos):
| Dataset | Tipo | Amostras Totais | Treino (70%) | Teste (30%) | Features ($D$) | Natureza das Features | Classes do Target ($y$) | Rótulos das Classes / Significado | Desafio do Fluxo |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- | :--- |
| **SEA** | Sintético | $20.000$ | $14.000$ | $6.000$ | 3 | 3 Contínuas ($f_0, f_1, f_2 \in [0, 10]$) | **2** | $\{0, 1\}$ (fronteira linear $f_0 + f_1 > \theta$) | *Abrupt concept drift* a cada 5.000 amostras ($\theta \in \{8, 9, 7, 9.5\}$); $f_2$ é ruído irrelevante |
| **Electricity (Elec2)** | Real | $45.312$ | $31.718$ | $13.594$ | 8 | 8 Contínuas (preços e demandas elétricas em NSW/VIC) | **2** | $\{0: \text{DOWN}, 1: \text{UP}\}$ (movimento relativo do preço) | *Concept drift* real de mercado, alta volatilidade de preços e sazonalidade horária/diária |
| **Forest Covertype** | Real | $25.000$ | $17.500$ | $7.500$ | 54 | 10 Contínuas (topografia) + 4 Binárias (áreas silvestres) + 40 Binárias (tipos de solo) | **7** | $\{1 \dots 7\}$: (1) Spruce-Fir, (2) Lodgepole Pine, (3) Ponderosa Pine, (4) Cottonwood/Willow, (5) Aspen, (6) Douglas-fir, (7) Krummholz | Alta dimensionalidade ($D=54$), espaço esparso (one-hot), desbalanceamento de classes florestais |
| **Airlines** | Real | $25.000$ | $17.500$ | $7.500$ | 7 | 7 Numéricas/Nominais (companhia, aeroporto origem/destino, dia da semana, hora, duração) | **2** | $\{0: \text{No Delay}, 1: \text{Delayed}\}$ (atraso no voo) | Padrões temporais mistos, desbalanceamento de atrasos sazonais e dependência de rotas |

### Regressão Contínua (4 fluxos):
| Dataset | Tipo | Amostras Totais | Treino (70%) | Teste (30%) | Features ($D$) | Natureza das Features | Classes do Target ($y$) | Alvo / Grandeza Física ($y$) | Desafio do Fluxo |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- | :--- |
| **Friedman #1** | Sintético | $20.000$ | $14.000$ | $6.000$ | 10 | 10 Contínuas uniformes ($f_0 \dots f_9 \in [0, 1]$) | **Contínuo** ($y \in \mathbb{R}$) | Sinal contínuo sintético: $10\sin(\pi x_0 x_1) + 20(x_2 - 0.5)^2 + 10x_3 + 5x_4 + \mathcal{N}(0, 1)$ | Não-linearidades trigonométricas e quadráticas com interação entre variáveis ($x_0, x_1$) e atributos puramente ruído ($x_5 \dots x_9$) |
| **Bikes (Bike Sharing)** | Real | $17.379$ | $12.165$ | $5.214$ | 12 | 12 Contínuas e temporais (clima, temperatura normalizada, sensação térmica, umidade, vento, calendário) | **Contínuo** ($y \in [1, 977]$) | Contagem horária de aluguel de bicicletas públicas | Fortes ciclos diários e sazonais, dependência climática não-linear e picos de demanda |
| **Appliances Energy** | Real | $19.735$ | $13.814$ | $5.921$ | 28 | 28 Contínuas (sensores ambientais IoT: temperaturas $T_1 \dots T_9$, umidades $RH_1 \dots RH_9$, pressão, vento, luz) | **Contínuo** ($y \in [10, 1080]\text{ Wh}$) | Consumo energético total dos eletrodomésticos da residência em Watt-hora (Wh) | Sinal ruidoso de sensores IoT com multicolinearidade severa entre cômodos e fortes variações circadianas |
| **NYC Taxi Trip Duration** | Real | $15.000$ | $10.500$ | $4.500$ | 7 | 7 Contínuas e discretas (coordenadas espaciais pickup/dropoff lon/lat, passageiros, hora, dia da semana) | **Contínuo** ($y \in [1, 86400]\text{ s}$) | Duração total da corrida em segundos ($s$) | Distribuição com cauda longa (*heavy tail*), assimetria espacial e congestionamentos urbanos dependentes do tempo |

---

## 2. Estrutura de Arquivos

```
main_benchmark/
├── main_benchmark.py           # Orquestrador principal da avaliação tripla (C11 vs River vs Scikit-Learn)
├── tune_hyperparams.py         # Orquestrador de busca e sensibilidade de hiperparâmetros (C11 vs River vs Sklearn)
├── prepare_datasets.py         # Gerador e formatador padronizado dos 8 datasets
├── README.md                   # Este documento explicativo
├── main_benchmark_report.md    # Relatório detalhado gerado com tabelas de acurácia, nós, RAM e speedup
├── hyperparam_report.md        # Relatório de sensibilidade e curvas de Pareto de hiperparâmetros
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

### Busca e Sensibilidade de Hiperparâmetros (`tune_hyperparams.py`)
Via Makefile (execução padrão rápida nos datasets de referência):
```bash
make bench-tune
```
Personalizando datasets, número de iterações ou competidores:
```bash
make bench-tune TUNE_ARGS="--dataset sea,friedman --engine all --n-iter 20"
```
Ou execução direta via Python:
```bash
# Busca aleatória com 30 iterações por motor em todos os datasets:
./.venv/bin/python main_benchmark/tune_hyperparams.py --dataset all --n-iter 30

# Grid search exaustivo em um dataset específico:
./.venv/bin/python main_benchmark/tune_hyperparams.py --dataset sea --grid

# Avaliação focada em um motor específico:
./.venv/bin/python main_benchmark/tune_hyperparams.py --dataset friedman --engine c11 --n-iter 15
```

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
