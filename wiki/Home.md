# Wiki — kansho (Embedded Hoeffding Tree in C11)

Bem-vindo à Wiki oficial do **kansho** (簡樹, *árvore simples*), uma implementação ultraleve de **Hoeffding Trees (`VFDT`, `HAT`, `VFDT-R` / `FIMT-DD` e `HAT-R`)** e detectores de mudança de conceito (**`ADWIN2`**) em **C11 puro**, projetada para aprendizado incremental (*online/streaming*) e inferência em tempo real em microcontroladores e sistemas embarcados com forte restrição de memória (**$\le 1\text{ MB}$ de RAM**, escalável para **$\approx 11\text{ KB}$**, com **zero `malloc`/`free`**).

---

## Índice de Páginas da Wiki

### 1. Guias Práticos e Referência de Uso
- **[Exemplos e Receitas da API](Exemplos-e-Receitas-API.md)**
  Exemplos completos comentados em C11 para Classificação (`VFDT`), Adaptação a *Concept Drift* (`HAT` + `ADWIN2`), Regressão Contínua (`VFDT-R` / `HAT-R`), Atributos Mistos (Numéricos + Categóricos) com valores ausentes (`NaN`) e Serialização Binária para Flash/EEPROM (`ht_tree_save` / `ht_tree_load`).
- **[Referência de Parâmetros e Macros](Referencia-de-Parametros-e-Macros.md)**
  Tabelas detalhadas de todos os hiperparâmetros em runtime (`ht_config_params_t` e `ht_adwin_config_t`), macros de compilação (`ht_config.h`), fórmulas de dimensionamento em bytes e perfis de compilação prontos por categoria de microcontrolador.
- **[Orçamento de Memória e Cross-Compilation](Orcamento-de-Memoria-e-Cross-Compilation.md)**
  Estimativas de consumo em Flash/ROM (`.text` / `.rodata`), tamanho do modelo serializado (`.bin`) por quantidade de nós, consumo de RAM volátil e flags recomendadas de compilação para Linux Embarcado e ARM Cortex-M / ESP32 (`arm-none-eabi-gcc`).

### 2. Internals, Arquitetura e Engenharia
- **[Arquitetura e Fluxogramas](Arquitetura-e-Fluxogramas.md)**
  Por que Hoeffding Tree no lugar do CART, decisões de design em baixo nível, alocador determinístico *Slab/Pool* livre de ponteiros (`uint16_t`) e fluxogramas Mermaid completos (roteamento iterativo, ciclo *Test-Then-Train*, avaliação do *Hoeffding Bound*, reciclagem de memória e substituição de subárvores no `HAT`/`HAT-R`).
- **[Origem: River, Scikit-Learn e Originalidade](Origem-River-Scikit-Learn-e-Originalidade.md)**
  Genealogia técnica e matriz comparativa tripla: o que foi extraído da teoria de streaming do River (`online-ml/river`), o que veio do design de dados e otimização numérica do Scikit-Learn (`sklearn.tree`), e as inovações originais da biblioteca (zero `malloc`, separação topologia/estado, reciclagem imediata e serialização 16-bit).
- **[Versionamento, Releases e Roadmap](Versionamento-Releases-e-Roadmap.md)**
  Arquitetura de versionamento SemVer 2.0.0 em C11 (`VERSION`, `HT_VERSION_*`, `ht_version()`), checklist para publicação de tags e releases no GitHub, e lista de trabalhos futuros e extensões do projeto.

---

## Documentação Teórica e Relatórios no Repositório

Além das páginas desta Wiki, o repositório principal mantém notas matemáticas aprofundadas e relatórios de validação experimental:

### Notas de Estudo e Mapeamento Teórico (`docs/`)
- [`docs/01_hoeffding_tree_vfdt.md`](../docs/01_hoeffding_tree_vfdt.md) — Fundamentos matemáticos do VFDT, *Hoeffding Bound*, *Gini*, *Information Gain* e *Promise Deactivation*.
- [`docs/02_attribute_observers.md`](../docs/02_attribute_observers.md) — Observadores incrementais: Algoritmo de Welford Gaussiano, CDF Normal e tabelas nominais.
- [`docs/03_concept_drift_hat.md`](../docs/03_concept_drift_hat.md) — *Hoeffding Adaptive Tree* (`HAT`), histograma exponencial `ADWIN2` e testes estatísticos de substituição de subárvores.
- [`docs/04_river_mapping.md`](../docs/04_river_mapping.md) — Mapeamento de paridade algorítmica com o `online-ml/river`.
- [`docs/05_sklearn_architecture.md`](../docs/05_sklearn_architecture.md) — Mapeamento de estruturas contíguas sem ponteiros inspirado em `sklearn/tree`.
- [`docs/06_implementation_plan.md`](../docs/06_implementation_plan.md) — Plano de implementação completo concluído (Fases 0 a 4).
- [`docs/07_regression_trees_fimtdd.md`](../docs/07_regression_trees_fimtdd.md) — Árvores de regressão incremental (`FIMT-DD` / `VFDT-R` e `HAT-R`), variância paralela de Chan e folhas lineares com SGD.
- [`docs/08_data_preprocessing_and_nonparametric.md`](../docs/08_data_preprocessing_and_nonparametric.md) — Micro-pipelines embarcados zero-malloc (`libht_prep`), Hampel MAD, Ben-Haim & Tom-Tov e normalização SGD híbrida.

### Relatórios de Benchmark e Paridade (`benchmarks/`)
- [`benchmarks/benchmark_report.md`](../benchmarks/benchmark_report.md) — Paridade de Classificação Estacionária (`VFDT` C11 vs River vs Scikit-Learn).
- [`benchmarks/triple_benchmark_report.md`](../benchmarks/triple_benchmark_report.md) — Benchmark Triplo no dataset MNIST com *profiling* de RAM e Heap.
- [`benchmarks/drift_benchmark_report.md`](../benchmarks/drift_benchmark_report.md) — Adaptação a *Concept Drift* (`HAT` C11 vs River `HAT`).
- [`benchmarks/regression_benchmark_report.md`](../benchmarks/regression_benchmark_report.md) — Regressão Contínua e Drift (`VFDT-R` / `HAT-R` vs River `HTR` / `HATR` vs Scikit-Learn).
- [`benchmarks/prep_benchmark_report.md`](../benchmarks/prep_benchmark_report.md) — Micro-pipelines de pré-processamento (`libht_prep`), normalização Z-score SGD e observador de histograma vs River.

---

## Como Sincronizar esta Pasta com a Aba Wiki do GitHub

O GitHub armazena a aba **Wiki** em um repositório Git independente (`pessoal_minimal-decision-tree.wiki.git`). Sempre que atualizar os arquivos em `wiki/`, você pode publicá-los na aba Wiki do GitHub com:

```bash
# 1. Clone o repositório da Wiki (apenas na primeira vez, após criar a 1ª página na interface web do GitHub)
git clone https://github.com/Julio-C-Oliveira/pessoal_minimal-decision-tree.wiki.git /tmp/mdt-wiki

# 2. Copie os arquivos markdown de wiki/ e envie para o GitHub
cp wiki/*.md /tmp/mdt-wiki/
cd /tmp/mdt-wiki
git add *.md
git commit -m "docs(wiki): sincroniza paginas da Wiki"
git push origin master
```
