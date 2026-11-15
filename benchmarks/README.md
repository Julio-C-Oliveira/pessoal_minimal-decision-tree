# Suíte Especializada de Benchmarks e Paridade Algorítmica (`benchmarks/`)

Esta pasta reúne os scripts, executores nativos e relatórios da **suíte especializada de validação e paridade algorítmica** do **kansho** (`libht` em C11 puro).

O objetivo principal desta suíte é comprovar a **paridade estatística e matemática** da nossa árvore de decisão contra a biblioteca de referência em streaming **[`online-ml/river`](https://github.com/online-ml/river)** e contra o baseline clássico em lote **[`scikit-learn`](https://scikit-learn.org)** (CART), além de avaliar o comportamento do modelo sob **extrema restrição de memória**, **concept drift** e **pipelines de pré-processamento**.

---

## 1. Mapeamento de Tarefas, Scripts e Relatórios

Cada experimento possui um script executor em Python que orquestra a geração de dados, compila os binários em C11 em precisão simples (`float32`) e dupla (`float64`), executa os modelos do River e Scikit-Learn e gera automaticamente o relatório analítico em formato Markdown:

| Tarefa / Experimento | Relatório Markdown Gerado | Script Executor | Dados Utilizados | Como Reproduzir (`make`) |
| :--- | :--- | :--- | :--- | :--- |
| **Classificação Estacionária** | [`benchmark_report.md`](benchmark_report.md) | `compare_river.py --mode static` | SEA, Agrawal, Hyperplane, LED | `make bench` |
| **Concept Drift (`HAT` / `ADWIN2`)** | [`drift_benchmark_report.md`](drift_benchmark_report.md) | `compare_river.py --mode drift` | SEA c/ drift abrupto, Agrawal drift | `make bench-drift` |
| **Regressão Contínua (`VFDT-R` / `HAT-R`)** | [`regression_benchmark_report.md`](regression_benchmark_report.md) | `compare_river.py --mode regression` | Friedman #1, Planes2D | `make bench-regression` |
| **Escalonamento em Extrema Baixa RAM** | [`memory_benchmark_report.md`](memory_benchmark_report.md) | `compare_river.py --mode memory` | SEA, Agrawal (Tier 256, Tier 8, Tier 4) | `make bench-memory` |
| **Micro-Pipelines de Pré-Processamento** | [`prep_benchmark_report.md`](prep_benchmark_report.md) | `compare_prep.py` | Fluxos ruidosos com Hampel MAD, Ben-Haim & Z-Score | `make bench-prep` |
| **Visão Computacional / Limites Teóricos** | [`triple_benchmark_report.md`](triple_benchmark_report.md) | `compare_triple.py` | MNIST ($4 \times 4$ pooling, 16 features: 10C, 4C, Binário) | `make bench-triple` |

---

## 2. Componentes e Estrutura de Arquivos

```text
benchmarks/
├── README.md                     # Este documento explicativo
├── bench_c_runner.c              # Executor C11 nativo de alta precisão (clock_gettime)
├── compare_river.py              # Orquestrador triplo para classificação, drift, regressão e memória
├── compare_prep.py               # Orquestrador para validação dos micro-pipelines de pré-processamento
├── compare_triple.py             # Orquestrador da avaliação no MNIST (10 classes, 4 classes e binário)
├── generate_streams.py           # Gerador determinístico de fluxos sintéticos em CSV
├── generate_mnist.py             # Pré-processador e gerador dos datasets MNIST com pooling TinyML
├── data/                         # Diretório onde os arquivos CSV de fluxo são gravados
├── benchmark_report.md           # Relatório: Paridade em Classificação Estacionária
├── drift_benchmark_report.md     # Relatório: Adaptação a Concept Drift com ADWIN2
├── regression_benchmark_report.md# Relatório: Regressão Contínua (VFDT-R e HAT-R)
├── memory_benchmark_report.md    # Relatório: Escalonamento de RAM (Tier 256 a Tier 4)
├── prep_benchmark_report.md      # Relatório: Micro-Pipelines de Pré-Processamento (libht_prep)
└── triple_benchmark_report.md    # Relatório: Limites teóricos em Visão Computacional (MNIST)
```

### Detalhes dos Componentes:

1. **`bench_c_runner.c`:**
   - Binário C11 stand-alone compilado com flags `-O3` e `-D` específicas para cada experimento.
   - Mede o tempo de processamento por amostra utilizando relógios monotônicos de alta resolução (`clock_gettime(CLOCK_MONOTONIC)`), isolando a execução C11 de qualquer sobrecarga de inicialização ou Garbage Collection do interpretador Python.
   - Fornece métricas de RAM física do modelo via dimensionamento estático do alocador e rastreia o número exato de nós, folhas ativas e profundidade da árvore.

2. **`compare_river.py`:**
   - Script em Python que executa o `online-ml/river` (`HoeffdingTreeClassifier`, `HoeffdingAdaptiveTreeClassifier`, `HoeffdingTreeRegressor`, `HoeffdingAdaptiveTreeRegressor`) e o `scikit-learn` (`DecisionTreeClassifier`, `DecisionTreeRegressor` em modo CART batch) nas mesmas partições determinísticas de dados consumidas pelo executável C11.
   - Suporta múltiplos modos via flag `--mode` (`static`, `drift`, `regression`, `memory` ou `all`).

3. **`compare_prep.py`:**
   - Avalia a biblioteca auxiliar `libht_prep` (módulos de normalização Z-score/Min-Max, filtro mediano de Hampel com janela deslizante MAD $3\sigma$ para rejeição de outliers, quantizador Ben-Haim & Tom-Tov e codificador Space-Saving), comparando acurácia e custo computacional contra equivalentes do River.

4. **`compare_triple.py`:**
   - Executa a avaliação no MNIST com compressão espacial de pixels para mapear os limites do paradigma VFDT diante de dados não-tabulares e documentar analiticamente a quebra das hipóteses de independência do Naive Bayes e as exigências amostrais do *Hoeffding Bound* em espaços multiclasse.

---

## 3. Como Executar

### Via Makefile (Recomendado)
A partir da raiz do repositório:
```bash
# Classificação estacionária (paridade contra River e Sklearn)
make bench

# Adaptação a Concept Drift (HAT com ADWIN2)
make bench-drift

# Regressão contínua (VFDT-R e HAT-R com SGD linear)
make bench-regression

# Escalonamento de memória sob restrição extrema (Tier 256, Tier 8 e Tier 4)
make bench-memory

# Micro-pipelines de pré-processamento (libht_prep)
make bench-prep

# Avaliação no MNIST (10 classes, 4 classes e binário)
make bench-triple
```

### Execução Direta via Python
Utilizando o ambiente virtual configurado no repositório:
```bash
# Executa apenas o modo estático de classificação:
./.venv/bin/python benchmarks/compare_river.py --mode static

# Executa o benchmark de drift:
./.venv/bin/python benchmarks/compare_river.py --mode drift

# Executa o benchmark de regressão:
./.venv/bin/python benchmarks/compare_river.py --mode regression

# Executa o benchmark de escalonamento de memória:
./.venv/bin/python benchmarks/compare_river.py --mode memory

# Executa o benchmark de pré-processamento:
./.venv/bin/python benchmarks/compare_prep.py

# Executa a avaliação tripla no MNIST:
./.venv/bin/python benchmarks/compare_triple.py
```

---

## 4. Relação com Outras Suítes de Benchmark do Repositório

- **[`main_benchmark/`](../main_benchmark/README.md):** Suíte principal de desempenho executada sobre 8 datasets completos (4 de classificação e 4 de regressão), com profiling aprofundado de **Peak RSS real do processo no SO**, comparação direta batch vs streaming e busca de hiperparâmetros (`make bench-main`).
- **[`emulation/esp32/`](../emulation/esp32/esp32_emulation_report.md):** Suíte de emulação Hardware-in-the-Loop em QEMU Xtensa @ 240 MHz com firmware ESP-IDF FreeRTOS, injeção de 187k amostras e cronometria em ciclos de clock de registrador (`make bench-esp32`).
