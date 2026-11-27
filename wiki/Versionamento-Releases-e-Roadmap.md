# Versionamento, Releases e Roadmap

Esta página descreve a arquitetura de versionamento do projeto em C11 (`SemVer 2.0.0`), como inspecionar a versão da biblioteca em tempo de compilação e execução, o processo de publicação de releases no GitHub e o *roadmap* de possíveis extensões futuras.

---

## 1. Como Funciona o Versionamento em C vs Python

No ecossistema Python moderno, o versionamento centralizado do pacote é definido em um único arquivo de configuração: o [`pyproject.toml`](https://peps.python.org/pep-0621/) (seção `[project] version = "0.1.0"`), sendo comum expor a variável `__version__` no pacote. Ferramentas como `pip`, `build`, `flit` ou `poetry` leem esse arquivo para empacotar e publicar no PyPI.

Em **C11 e bibliotecas de baixo nível / embarcadas** (padrão adotado por SQLite, zlib, libcurl, cJSON, Git e Linux Kernel), não existe um gerenciador de dependências universal mandatório. O versionamento é estruturado em **três camadas complementares**:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Metadata File (Raiz)        │ VERSION ("0.6.0")                      │
│ (Equivalente ao pyproject.toml) │ Lido pelo Makefile, scripts e CI/CD   │
├─────────────────────────────────────────────────────────────────────────┤
│ 2. Single Source of Truth em C │ HT_VERSION_MAJOR / MINOR / PATCH       │
│ (code/include/core/ht_config.h)│ HT_VERSION_STRING ("0.6.0")            │
│                                │ HT_VERSION_CHECK(maj, min, patch)      │
│                                │ const char *ht_version(void);          │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. Git Tags e GitHub Releases  │ v0.6.0 (Tags anotadas no Git)          │
│                                │ GitHub Releases / Tarballs             │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Arquivo de Metadados ([`VERSION`](../VERSION)):** Localizado na raiz do repositório, armazena a versão canônica (ex: `0.6.0`). É a referência única para automações, empacotadores e regras de build.
2. **Macros e Símbolos em C ([`ht_config.h`](../code/include/core/ht_config.h) e [`ht_tree.h`](../code/include/core/ht_tree.h)):** Fornece constantes numéricas para o pré-processador e uma função estática de runtime (`ht_version()`).
3. **Tags do Git (`vX.Y.Z`):** Conectam o histórico de commits do repositório com as versões públicas disponibilizadas no GitHub.

---

## 2. Como Consultar a Versão no Código

A biblioteca permite verificar a versão tanto durante a compilação do firmware quanto em tempo de execução:

### Em Tempo de Compilação (Zero custo de CPU/RAM)
```c
#include "ht_tree.h"

#if !HT_VERSION_CHECK(0, 1, 0)
#error "Este firmware requer kansho >= 0.1.0!"
#endif
```

### Em Tempo de Execução
```c
#include <stdio.h>
#include "ht_tree.h"

void print_system_info(void) {
    /* Retorna ponteiro para string literal em Flash (zero malloc, O(1)) */
    printf("kansho versao: %s\n", ht_version());
}
```

### Via Linha de Comando / Build
```bash
make version
# Saida: kansho (libht) version 0.6.0
```

---

## 3. Política de Versionamento Semântico (SemVer 2.0.0)

O projeto adota o padrão [Semantic Versioning 2.0.0](https://semver.org/lang/pt-BR/): `MAJOR.MINOR.PATCH`:

- **PATCH (`0.1.X`):** Correções de bugs, pequenas melhorias numéricas ou refatorações internas sem alteração na API pública nem no formato binário de serialização (`ht_tree_save` / `ht_tree_load`).
- **MINOR (`0.X.0`):** Adição de novas funcionalidades retrocompatíveis (ex: novo critério de split, novos observadores, novos modos de poda ou novos targets de hardware).
- **MAJOR (`X.0.0`):** Alterações incompatíveis na API pública (`ht_tree_*`), quebra no formato binário de serialização ou mudança na arquitetura de memória dos pools.

---

## 4. Criando e Publicando Tags no GitHub

Para lançar uma nova versão oficial no GitHub:

1. Atualize a versão no arquivo [`VERSION`](../VERSION) e nas macros `HT_VERSION_*` em [`code/include/core/ht_config.h`](../code/include/core/ht_config.h).
2. Execute a suíte de testes garantindo 100% de sucesso nos 3 perfis numéricos (`float32`, `double64` e `Q16.16` sem `-lm`):
   ```bash
   make clean && make test && make clean && make test CFLAGS_EXTRA="-DHT_USE_DOUBLE" && make clean && make test-fixed
   ```
3. Realize o commit das alterações de versão e crie uma **tag anotada**:
   ```bash
   git tag -a v0.6.0 -m "Release v0.6.0: Q16.16 Zero-Libm Fixed-Point Arithmetic (-DHT_USE_FIXED_POINT)"
   ```
4. Envie o commit e a tag para o repositório remoto no GitHub:
   ```bash
   git push origin main
   git push origin v0.6.0
   ```
5. No GitHub, acesse a aba **Releases** $\rightarrow$ **Draft a new release** e selecione a tag `v0.6.0` recém-enviada para gerar as notas de lançamento e os pacotes `.tar.gz` e `.zip` correspondentes.

---

## 4.1 Histórico de Releases (Changelog)

### v0.6.0 (2026-11-27) — Aritmética de Ponto Fixo `Q16.16` Zero-Libm (`-DHT_USE_FIXED_POINT`) (Fase 8)
- **Motor Matemático `Q16.16` Zero-Libm ([`ht_math_fixed`](../code/include/core/ht_math_fixed.h), Entrega 8.2):**
  - Modo de compilação **`-DHT_USE_FIXED_POINT`** (mutuamente exclusivo com `-DHT_USE_DOUBLE`) que substitui `ht_float_t` por inteiro com sinal `int32_t` em formato `Q16.16` ($16$ bits inteiros + $16$ bits fracionários, faixa $[-32768{,}0,\; +32767{,}99998]$, passo $1/65536 \approx 1{,}53\times 10^{-5}$), com **0 bytes de alteração em `sizeof(ht_tree_t)`** (`496.024 B`, idêntico ao `float32`) e sentinela determinística `HT_NAN = INT32_MIN`.
  - Operações saturantes em aritmética inteira 64-bit (`ht_mul`, `ht_div`, `ht_add`, `ht_sub`), raiz quadrada inteira exata (`ht_sqrt` por bit-shift binário), logaritmo natural (`ht_log` via `CLZ` + polinômio minimax), exponencial (`ht_exp` via redução em $\ln 2$), função erro (`ht_erf` / `ht_normal_cdf` via Abramowitz-Stegun 7.1.26) e distribuição de Poisson (`ht_poisson_knuth` via PRNG LCG 32-bit determinístico) — **sem incluir `<math.h>` e sem linkar `-lm`**.
- **Estabilidade Numérica em Fluxos Contínuos (`ht_observer`, `ht_criterion`, `ht_prep`, Entrega 8.3):**
  - **Block-Rescaling** em `ht_gaussian_obs_t` e `ht_var_stat_t` (divide `count` e `m2` por $2$ quando excedem `16384` / faixa segura, mantendo $\sigma^2 = M_2 / (n - 1)$ invariante em fluxos infinitos).
  - **Variância em `Q26.6`** (`ht_var_value_q8`, escala `2^6`) e combinação exata em 64-bit (`ht_var64_t`) no cálculo de Redução de Variância (*Variance Reduction*), suportando regressão contínua de alta amplitude (ex: *NYC Taxi*, *Appliances Energy*) sem saturação de $M_2$ nem estagnação do `grace_period`.
  - Proteção contra *underflow* sub-ULP de `m2` em `ht_gaussian_stddev` (fallback de dispersão uniforme `(max - min) / sqrt(12)` quando `m2 == 0` e `max_val > min_val`).
- **Suporte Completo aos 12 Módulos e Compatibilidade Binária (`core/`, `ensemble/`, `federated/`, Entrega 8.4):**
  - Campo `delta_neg_log10` em `ht_config_params_t` + helper `ht_params_set_delta_pow10(&params, 7)` e `ht_hoeffding_bound_ln` para preservar $\ln(1/\delta)$ com alta precisão mesmo para $\delta = 10^{-7}$ (abaixo do LSB do `Q16.16`).
  - Flags binárias de representação (`HT_SERIAL_FLAG_FIXED`, `HT_COMPACT_FLAG_FIXED`, `HT_FED_FLAG_FIXED`) impedindo carregamento acidental de snapshots `float32`/`double64` em firmware `Q16.16` (e vice-versa).
  - Proteção contra saturação de `faded_mse` em `ht_voting.c` e acumulação de peso imune a Block-Rescaling no `Fed-VFDT STATS-MERGE` (`ht_fed_vfdt.c`).
  - Novos alvos no `Makefile`: **`make lib-fixed`** e **`make test-fixed`** com auditoria automática `nm -u build/libht.a` garantindo **0 símbolos `libm` / `soft-fp`**.
- **Validação Experimental Completa (Entrega 8.5):**
  - **Emulação Hardware-in-the-Loop ARM Cortex-M3 @ 50 MHz sem FPU (`emulation/cortex-m3/cortex_m3_fixed_point_report.md`):**
    - Redução de Flash `.text` de **`66.220 B` para `49.264 B` (`-25,6%`)** e `.data` de `468 B` para `104 B`, mantendo `.bss = 57.000 B` (`0 B` heap).
    - Auditoria `arm-none-eabi-nm` no `.elf` **APROVADA**: **0 símbolos** `__aeabi_f*`, `__aeabi_d*`, `_printf_float` ou `libm`.
    - Speedup real via **SysTick** de **`1,86×` a `4,48×`** (média **`2,86×`**, até `133.690 amostras/s`), com paridade preditiva em todos os cenários (`Electricity` `64,26%` vs `64,34%` em `float32`, `-0,08 pp`, `2,51×` mais rápido).
  - **Paridade Host (8 Datasets × 2 Tiers) e Federada (8 Datasets × Non-IID/IID × 7 Estratégias):**
    - [`main_benchmark/fixed_point_benchmark_report.md`](../main_benchmark/fixed_point_benchmark_report.md): paridade preservada nos 8 datasets em High Tier e Tier 4 (incluindo *NYC Taxi* com `79` nós vs `75` em `float32`).
    - [`main_benchmark/federated_fixed_point_report.md`](../main_benchmark/federated_fixed_point_report.md): 112/112 combinações válidas (`SEA Non-IID STATS-MERGE` `96,78%` idêntico; `Covertype Non-IID STATS-MERGE` `65,53%` vs `65,52%`; `Appliances IID Fed-Forest TOP_M` `MAE 28,73 / R² 0,7662`).
  - **Zero Regressão em `float32` e `double64`:** Validação direta `main` vs `v0.6.0` nos 8 datasets × 2 tiers com **16/16 (`100%`)** de paridade exata em `double64` e `float32`, além de 100% de execução limpa no `tune_hyperparams --replay-report` (`180/180`), `main_benchmark` (`8/8`) e `federated_benchmark` (`112/112`).
- **Suíte de Testes:** **143/143 testes aprovados em `Q16.16`** (`231.542` asserções, zero `-lm`), **138/138 em `float32`** e **138/138 em `double64`** (`231.718` asserções) com zero warnings sob `-std=c11 -Wall -Wextra -Werror -pedantic`.

### v0.5.0 (2026-11-26) — Aprendizado Federado Embarcado (`Fed-Forest` & `Fed-VFDT`) e Arquitetura Modular (Fase 11)
- **Reestruturação Modular de `code/` (Entrega 11.2):**
  - Organização simétrica de `code/include/`, `code/src/` e `code/tests/` em `core/`, `ensemble/` e `federated/`, preservando 100% de retrocompatibilidade de API e headers via `code/Makefile` e firmwares de emulação (`emulation/esp32/`, `emulation/cortex-m3/`).
- **Extensões Cirúrgicas no Núcleo `core/` (Entrega 11.3):**
  - Fusão paralela exata de observadores em [`ht_observer`](../code/include/core/ht_observer.h) (`ht_gaussian_merge` via fórmula de Chan, `ht_categorical_merge`, `ht_hist_merge`, `ht_reg_numeric_merge`, `ht_reg_categorical_merge`).
  - Primitivas de coordenação em [`ht_tree`](../code/include/core/ht_tree.h): flag `HT_TREE_FLAG_DEFER_SPLITS` com **0 bytes extras** em `ht_tree_t`, `ht_tree_evaluate_leaf_split`, `ht_tree_apply_split` e `ht_tree_reset_subtree_to_leaf`.
- **Fundação Compartilhada `ensemble/` (Entrega 11.4):**
  - [`ht_compact_tree`](../code/include/ensemble/ht_compact_tree.h): compactação iterativa BFS de `ht_tree_t` em árvore densa somente-inferência ($\approx 1{,}34\text{ KB}$ para 64 nós) com serialização binária portátil.
  - [`ht_voting`](../code/include/ensemble/ht_voting.h): motor de votação uniforme e ponderado pelo inverso do erro prequential ($1/(\hat{e}_m^2 + \varepsilon)$), compartilhado entre Aprendizado Federado (Fase 11) e Micro-Ensembles (Fase 9).
- **Protocolo Binário `HFED` & Modo 1 `Fed-Forest` (Entrega 11.5):**
  - [`ht_fed_proto`](../code/include/federated/ht_fed_proto.h): framing binário agnóstico de transporte (cabeçalho fixo de 32 B, versionamento e CRC32 IEEE 802.3 via tabela nibble de 64 B).
  - [`ht_fed_forest`](../code/include/federated/ht_fed_forest.h): cliente embarcado com 1 árvore treinável + $M$ árvores remotas compactas e servidor C11 estático com políticas `ALL`, `TOP_M` e `ROUND_ROBIN` (+ `exclude_self`), suportando Classificação e Regressão.
- **Modo 2 `Fed-VFDT` Global & Auditoria ICTAI 2025 (Entregas 11.6 e 11.8):**
  - [`ht_fed_vfdt`](../code/include/federated/ht_fed_vfdt.h): suporta `HT_FED_VFDT_COORDINATED` — alinhado estritamente ao paper *Fed-VFDT* (Silva, Vinagre & Gama, IEEE ICTAI 2025, DOI `10.1109/ICTAI66417.2025.00094`: `QUORUM` por atributo, `MAJORITY` por pluralidade, `BEST-MERIT` `max(G_k)`, `WEIGHTED_MERIT`, `HT_FED_THRESH_MAX_MERIT` e filtro `require_locally_ready`) — e `HT_FED_VFDT_STATS_MERGE` (fusão exata *stateless* de observadores + FedAvg ponderado para SGD, sem contagem dupla).
  - Recuperação de *stragglers* via `topology_version`, log circular `split_log[64]`, fallback automático por `FULL_SNAPSHOT` e reset coordenado por *concept drift* (`HT_FED_CMD_RESET_SUBTREE`).
- **Validação Experimental Completa (Entregas 11.7, 11.9, 11.10 e 11.11):**
  - **Benchmark Real em 8 Datasets (`make bench-main-federated`):** $K=10$ clientes, $R=20$ rodadas, **Non-IID (`Dirichlet α=0.3`)** e **IID** no perfil MCU Tier 8 (pico de SRAM estática por cliente de **77,01 KB** no *Forest Covertype* 54 feat / 7 cls — abaixo de 100 KB). `Fed-VFDT (STATS-MERGE)` saltou de `30,91%` para `65,52%` no *Covertype* Non-IID (+34,61 pp vs local) e `Fed-Forest` reduziu o MAE em `-54,9%` no *Appliances Energy* e `-32,2%` no *Bikes* ([`main_benchmark/federated_benchmark_report.md`](../main_benchmark/federated_benchmark_report.md)).
  - **Transporte Real via Sockets TCP POSIX (`make bench-fed-net` — Gate A):** 5/5 cenários multi-processo aprovados em `0,33 s` com fragmentação de stream (`11–23 B/write`), rejeição de bit-flip via CRC32, Non-IID extremo (100% puro), recuperação real de *straggler* offline (`HT_ERR_STALE_VERSION` → catch-up `TopoV=3/3`) e regressão contínua.
  - **Compilação Cruzada 32-Bit & Emulação QEMU (Gate B):** Zero warnings sob `-std=c11 -Wall -Wextra -Werror -pedantic -Os` em **ARM Cortex-M3** (`arm-none-eabi-gcc 14.2.1` + QEMU `lm3s6965evb`: `11,08 KB` Flash p/ `ensemble/`+`federated/`, `39,66 KB` total) e **ESP32 Xtensa LX6** (ESP-IDF `v5.1.2` + QEMU `esp32`: `11,83 KB` Flash p/ `ensemble/`+`federated/`, `40,61 KB` total), com `data=0` e `bss=0`.
- **Suíte de Testes:** **134/134 testes unitários e de integração aprovados** (`231.385` asserções em `float32` e `double64`) com zero warnings e zero `malloc` em runtime.

### v0.4.1 (2026-11-14) — Otimização de Footprint com Perfil ADWIN Compacto e Omissão de Feat Stats (Entrega 7.7)
- **Perfil de Compilação ADWIN Compacto (`ht_drift`):**
  - Parametrização algébrica das asserções estáticas para os níveis de histograma exponencial do ADWIN via macro `HT_ADWIN_LEVELS` (padrão `14`, configurável para `10` em perfis compactos de microcontroladores como Cortex-M3 / ESP32).
  - Redução física de `sizeof(ht_adwin_t)` de $764\text{ B}$ para **$556\text{ B}$** em `float32` (e de $1.504\text{ B}$ para **$1.088\text{ B}$** em `double64`), reduzindo o `ht_adwin_pool_t` no perfil MCU (`10` níveis, `32` detectores) de **$98.068\text{ B}$ ($95,77\text{ KB}$) para $17.864\text{ B}$ ($17,45\text{ KB}$)** — economia de **$-78,32\text{ KB}$ ($-81,8\%$)**.
  - Adição de teste unitário de saturação de nível superior (`test_adwin_highest_level_saturation_and_drop`), validando a reciclagem de buckets no horizonte temporal máximo sem *underflow* em `total_variance`.
- **Omissão Condicional de `feat_stats` em Classificação (`ht_tree`) e Dimensionamento `.bss`:**
  - Proteção de `feat_stats[HT_MAX_FEATURES]` de regressão sob `#ifndef HT_CLASSIF_ONLY`, eliminando o vetor nos caminhos puros de classificação em `ht_tree.c` e `ht_tree.h`.
  - Parametrização exata de `-DHT_STATS_STORAGE_BYTES` por `n_features` real nos benchmarks (`main_benchmark/tune_hyperparams.py`, `main_benchmark/main_benchmark.py`) e atualização dos perfis compactos nos firmwares QEMU (`emulation/esp32/CMakeLists.txt` e `emulation/cortex-m3/Makefile`).
- **Validação Experimental no Replay Benchmark (8 Datasets):**
  - Forest Covertype (Tier 2, 54 features, 7 classes): redução de RAM de **$22,8\text{ KB}$ para $18,9\text{ KB}$ ($-17,1\%$)**, com acurácia ($61,43\%$) e F1 Macro ($58,51\%$) bit-a-bit idênticos.
  - Reduções consistentes nos benchmarks de streaming e prequential nos 8 datasets avaliados.
- **Suíte de Testes:** 119 testes unitários e de integração 100% aprovados ($227.173$ asserções em `float32` e $227.509$ asserções em `double64`) com zero warnings.

### v0.4.0 (2026-11-14) — Otimização Extrema de Memória SRAM com Dynamic Feature Stride Slab (Entrega 7.6)
- **Dynamic Feature Stride Slab Allocator (`ht_memory`) (Entrega 7.6):**
  - Substituição do vetor estático homogêneo por um slab contíguo em `.bss` (`stats_storage[HT_STATS_STORAGE_BYTES]` em `union` com âncora `ht_leaf_stats_t` via `_Alignas`) inicializado por `ht_memory_init_stride(pool, n_features, task_type)` com alinhamento natural `sizeof(ht_float_t)` e capacidade limitada a `HT_MAX_ACTIVE_LEAVES`.
  - Reordenação de `ht_leaf_stats_t` deslocando `reg_weights` para após `observers` (acesso via `ht_leaf_reg_weights`) e mantendo `offsetof(ht_leaf_stats_t, observers)` fixo em $16\text{ B}$ (`float32`) / $32\text{ B}$ (`float64`).
  - Especialização condicional via macro `-DHT_CLASSIF_ONLY`: omite `reg_weights` na reserva do slab e bloqueia `HT_TASK_REGRESSION`, reduzindo a pegada de SRAM estática em **$-35,4\%$ a $-54\%$** e a pegada de RAM ativa em até **$-49,3\%$** em tarefas de classificação.
  - Teste de integridade com canários de borda (`0xAA` / `0x55`) entre folhas adjacentes em `test_dynamic_stride_canary_boundary_protection` (`test_memory.c`), garantindo zero *buffer overrun*.
- **Validação Experimental no Replay Benchmark (8 Datasets):**
  - Forest Covertype (Tier 2, 54 features, 7 classes): redução de SRAM estática de **35,3 KB para 22,8 KB (-35,4%)** e RAM ativa para **17,9 KB (-49,3%)**, mantendo 100% de paridade preditiva bit-a-bit (Acurácia: 61,43%, F1 Macro: 58,51%).
  - SEA (Tier 1): redução de SRAM estática de **5,0 KB para 2,3 KB (-54,0%)** (Acurácia: 94,87%).
- **Suíte de Testes:** 118 testes unitários e de integração 100% aprovados com zero warnings em `float32` e `double64`.

### v0.3.2 (2026-11-11) — Ciclo Experimental Revertido (Restauração de Bounding Box Gaussiana; ver Post-Mortem da Entrega 7.5)
- **Restauração de Bounding Box por Classe em Observador Gaussiano (`ht_gaussian_obs_t`):**
  - Correção na estimativa de divisão binária numérica restaurando `min_val` e `max_val` em nível de classe (`ht_gaussian_obs_t`), em conformidade com o algoritmo de referência do River (`river.tree.splitter.GaussianSplitter`).
  - Eliminação do vazamento probabilístico de cauda da CDF gaussiana em atributos concentrados (como variáveis binárias/one-hot no Forest Covertype), restaurando a acurácia para **58,68%** (Tier 1) e **61,43%** (Tier 2).
  - *Nota de Histórico:* As tentativas experimentais das Entregas 7.4 e 7.5 (`v0.3.1` e `v0.3.2`) foram formalmente revertidas ao baseline estável da `v0.3.0` (commit `bad03f9`), com post-mortem documentado em [`docs/09_extreme_ram_optimization.md`](../docs/09_extreme_ram_optimization.md), servindo de base para o modelo híbrido validado na `v0.4.0`.
- **Suíte de Testes:** 118 testes unitários e de integração 100% aprovados ($226.977$ asserções) com zero warnings em `float32` e `double64`.

### v0.3.1 (2026-11-11) — Ciclo Experimental Revertido (Tentativas das Entregas 7.4 e 7.5)
- **Experimentos Avaliados e Posteriormente Revertidos (`bad03f9`):**
  - Tentativa de desacoplamento de `node_drift` em pool dedicado (`ht_adwin_pool_t`) (Entrega 7.4) e união de cabeçalho + fatoração de bounding box `[min, max]` por atributo em `ht_gaussian_obs_t` (Entrega 7.5).
  - Identificadas duas limitações empíricas e arquiteturais documentadas no post-mortem: busca linear $O(N)$ na indexação de drift das folhas e vazamento de cauda na CDF gaussiana em variáveis binárias esparsas (refutação empírica da H5).
- **Suíte de Testes:** 118 testes unitários e de integração aprovados ($230.133$ asserções) em `float32` e `double64`.

### v0.3.0 (2026-11-11) — Otimização Extrema de Memória SRAM (Free-lists Intrusivas e Topologia Assimétrica)
- **Free-lists Intrusivas em `ht_memory_pool_t` (Hipótese 1 / Entrega 7.2):**
  - Eliminação completa dos vetores externos de pilha `free_nodes[HT_MAX_NODES]` e `free_stats[HT_MAX_ACTIVE_LEAVES]`, gravando os índices intrusivos `next_free` diretamente dentro das estruturas inativas (`node_pool[idx].next_free` e `stats_pool[idx].next_free`).
  - Economia imediata de **$8.704\text{ bytes}$ ($8,50\text{ KB}$)** de SRAM estática em `float32` e `double64`, preservando complexidade estritamente constante $O(1)$ em alocação e liberação.
- **Topologia Assimétrica de Nós em `ht_node_t` (Hipótese 2 / Entrega 7.3):**
  - Redesenho de `ht_node_t` como união mutuamente exclusiva entre estado de folha (`class_counts` / `reg_stats`, `stats_idx`) e estado de nó interno de decisão (`split_val`, `left_child`, `right_child`, `feature_idx`).
  - Redução física de `sizeof(ht_node_t)` de $28\text{ B}$ para **$20\text{ B}$** por nó em `float32` (configuração padrão de 4 classes) e de $48\text{ B}$ para **$40\text{ B}$** em `double64`, gerando economia direta de **$32.768\text{ bytes}$ ($32,0\text{ KB}$)** no `node_pool` ($4.096$ nós).
  - Introdução da macro de alinhamento dinâmico `HT_NODE_FLAGS_OFFSET`, garantindo posicionamento seguro de `flags` no boundary de multi-classes sem sobreposição de memória.
- **Main Benchmark Atualizado com 8 Datasets:**
  - Resultados consolidados com hiperparâmetros ótimos comprovando paridade preditiva e superioridade contra `River HTC/HTR` com speedups de **$53\times$ a $1.479\times$** por amostra.
  - Consumo intra-treino rigorosamente contido em $\le 1\text{ MB}$ de RAM mesmo em datasets de alta dimensionalidade como Forest Covertype ($703,8\text{ KB}$ vs $57,8\text{ MB}$ do River).
- **Suíte de Testes:** 115 testes unitários e de integração aprovados ($226.035$ asserções) com 0 warnings em `float32` e `double64`.

### v0.2.0 (2026-11-10) — Micro-Pipelines de Pré-Processamento (`libht_prep`), Histogramas Dinâmicos e Normalização SGD
- **Biblioteca de Pré-Processamento Embarcada (`libht_prep`):**
  - `ht_scaler_t`: Normalização online Z-score (Welford incremental) e Min-Max em $O(1)$ de RAM e tempo.
  - `ht_filter_hampel_t`: Filtro mediano deslizante com Desvio Absoluto da Mediana (MAD, $3\sigma$) com buffer circular estático para rejeição determinística de espículas (*outliers*) de sensores industriais.
  - `ht_encoder_freq_t`: Codificador de frequência incremental para variáveis categóricas de alta cardinalidade baseado no algoritmo Space-Saving de Metwally.
  - `ht_quantizer_t`: Quantizador/discretizador online baseado no histograma dinâmico de Ben-Haim & Tom-Tov com estimativa contínua de CDF.
  - `ht_prep_pipeline_t`: Pipeline unificado integrável diretamente à árvore (`ht_tree_set_prep_pipeline`), aplicando transformações de forma transparente em treino e inferência.
- **Normalização SGD Híbrida nas Folhas de Regressão (Item 8):**
  - Implementação de padronização online nos regressores lineares locais com fator de credibilidade $\alpha = \frac{N_{\text{slots}}}{N_{\text{slots}} + 10.0}$, eliminando explosões de gradiente em fluxos multiescala ($MAE \sim 10^{14} \rightarrow 1.12$, 200x mais rápido que River StandardScaler) com zero heap em runtime.
- **Observador Numérico de Histograma Dinâmico (Item 16):**
  - Adição de `HT_OBS_HISTOGRAM` (`ht_hist_obs_t`) para dados multimodais e não-Gaussianos, cabendo perfeitamente dentro da união `ht_feature_obs_t` sem alterar o tamanho de `ht_leaf_stats_t`.
- **Suíte de Testes:** 113 testes unitários e E2E 100% aprovados em precisão simples (`float`) e dupla (`double`).

### v0.1.0 (2026-11-08) — Release Inicial (VFDT, HAT, VFDT-R, HAT-R, ADWIN2)
- Implementação inicial completa das 4 variantes de árvores de decisão em C11 puro dentro do teto estrito de $1\text{ MB}$ de RAM com alocadores *Slab/Pool* estáticos livres de ponteiros (`uint16_t`).
- Detector de drift `ADWIN2` com histograma exponencial fixo e substituição de subárvores.
- Paridade experimental com `online-ml/river` e benchmark de 8 datasets.

---

## 5. Possíveis Trabalhos Futuros e Extensões (Roadmap)

Embora todas as fases entregues até a `v0.6.0` (**Fases 0 a 8 e Fase 11 — Aprendizado Federado**, incluindo Classificação, Adaptação a *Concept Drift* com `ADWIN2`/`HAT`, Regressão Contínua `FIMT-DD`/`HAT-R`, `libht_prep`, otimização extrema de SRAM com *slab* por feature, **aritmética de ponto fixo `Q16.16` Zero-Libm (`-DHT_USE_FIXED_POINT`)** para MCUs sem FPU e Aprendizado Federado `Fed-Forest` & `Fed-VFDT`) estejam 100% concluídas com zero heap dinâmico em runtime e perfis enxutos (`4,7 KB` a `18,9 KB` no núcleo), as análises teóricas e os resultados experimentais apontam prioridades estratégicas claras para as próximas entregas:

### 5.0 Prioridades Estratégicas: Consolidando a Liderança no Nicho Embarcado

> [!TIP]
> **Foco de Evolução (`v0.7.0+`):** Competir ainda mais próximo aos modelos *batch* em fluxos tabulares complexos preservando o teto estrito de SRAM (`< 100 KB`), após a entrega completa da cobertura de MCUs sem FPU (`Q16.16` na `v0.6.0`).

1. **Aritmética de Ponto Fixo (`Q16.16`, `-DHT_USE_FIXED_POINT`) para MCUs sem FPU — Concluído (`v0.6.0`):**
   - Implementação completa em todos os 12 módulos (`core/`, `ensemble/`, `federated/`) sem `<math.h>` e sem `-lm`, validada em ARM Cortex-M3 @ 50 MHz (`-25,6%` de Flash `.text`, speedup de `1,86×–4,48×` via SysTick, `0` símbolos `soft-fp`/`libm`) e nos 8 datasets (local e federado).
2. **Micro-Ensembles com Observadores Compartilhados e Subespaços (`< 100 KB` de SRAM) — Próxima Prioridade (`v0.7.0+`):**
   - Em vez de uma *Adaptive Random Forest* tradicional (que multiplicaria por $M$ o custo dos observadores de Welford/Histograma), implementar um **comitê de 3 a 5 árvores rasas (Tier 2 ou Tier 4)** que **compartilham estatísticas na raiz** ou particionam **subespaços aleatórios de features** sobre o `leaf_stride` dinâmico (`v0.4.0`), elevando drasticamente o Macro F1 no *Forest Covertype* e o $R^2$ nas regressões abaixo de **`100 KB`** (detalhes no [Item #18](#56-estabilidade-numérica-ecossistema-e-ensembles)).

---

### 5.1 Plataformas Embarcadas, Hardware Real e Concorrência Multi-Core

1. **Deploy e Cross-Compilation em Hardware Real (Bare-Metal / RTOS):**
   - Criação de scripts e targets no Makefile para compilação cruzada com toolchains reais (`arm-none-eabi-gcc` para ARM Cortex-M4F / Cortex-M7, `riscv64-unknown-elf-gcc` para RISC-V e ESP-IDF para ESP32).
   - Validação direta em bancada com microcontroladores operando sob FreeRTOS ou Zephyr OS, medindo ciclos exatos de clock (`DWT->CYCCNT`) e consumo de corrente/energia por amostra.

2. **Quantização e Aritmética de Ponto Fixo (`-DHT_USE_FIXED_POINT` — `Q16.16` Zero-Libm) — Concluído (`v0.6.0`):**
   - Suporte nativo completo via flag `-DHT_USE_FIXED_POINT` (`int32_t` `Q16.16`, zero `<math.h>`, zero `-lm`, zero símbolos `soft-fp`) validado em ARM Cortex-M3 sem FPU (`-25,6%` Flash, speedup médio `2,86×` no SysTick) com Block-Rescaling, variância `Q26.6` + combinação 64-bit, `delta_neg_log10` e flags de segurança binária `HT_*_FLAG_FIXED`. *(Extensão futura opcional: perfil ultracompacto `int16` para MCUs de 8/16 bits).*

3. **Validação Prática em ESP32 com Máquina de Estados Completa:**
   - Teste da biblioteca embarcada em uma placa **ESP32** com máquina de estados operacional completa para aplicações de borda:
     - **Modo de Captura de Dados:** Aquisição e pré-processamento de leituras de sensores em tempo real.
     - **Modo de Treinamento:** Aprendizado incremental *on-device* (`ht_tree_learn_one` / `ht_tree_learn_one_reg`) e persistência periódica em memória não-volátil Flash/NVS (`ht_tree_save`).
     - **Modo de Inferência:** Predição em tempo real de ultrabaixa latência (`ht_tree_predict` / `ht_tree_predict_reg`) com restauração a frio do modelo salvo (`ht_tree_load`).

4. **Desacoplamento Assíncrono Multi-Core Lock-Free (SPSC Ring Buffer):**
   - Em microcontroladores dual-core (ESP32 Xtensa/RISC-V, Raspberry Pi RP2040 dual Cortex-M0+, STM32H7 dual Cortex-M7/M4), desacoplar a inferência e o treino:
     - **Core 0 (Prioridade de Tempo Real):** Atende interrupções de sensores a taxas elevadas ($> 10\text{ kHz}$) e executa `ht_tree_predict` com latência estrita sub-microssegundo ($< 1\,\mu\text{s}$), depositando as tuplas $(\mathbf{x}, y)$ em uma fila circular livre de bloqueio (*Single-Producer Single-Consumer Lock-Free Ring Buffer*) alocada em SRAM compartilhada.
     - **Core 1 (Background Worker):** Consome a fila, executa `ht_tree_learn_one` e processa os cálculos estatísticos pesados do Limite de Hoeffding e `ADWIN2` sem bloquear a malha de controle ou a amostragem de dados.

### 5.2 Eficiência Estatística e Limites Adaptativos (Lições do Covertype)

5. **Limites Estatísticos com Variância Empírica (Empirical Bernstein Bound):**
   - No dataset *Covertype* (54 atributos, 7 classes), observou-se que o Limite de Hoeffding clássico $\epsilon_H = \sqrt{\frac{R^2 \ln(1/\delta)}{2n}}$ é excessivamente conservador quando a faixa teórica de impureza $R = \log_2(C)$ é grande, retardando a abertura de splits.
   - Como alternativa, planeja-se integrar o *Empirical Bernstein Bound*:
     $$\epsilon_B = \sqrt{\frac{2 V \ln(3/\delta)}{n}} + \frac{3 R \ln(3/\delta)}{n}$$
     onde $V$ é a variância empírica amostrada dos ganhos de impureza entre os atributos competidores. Quando a dispersão real dos ganhos é reduzida ($V \ll R^2$), $\epsilon_B$ converge dezenas de vezes mais rápido que $\epsilon_H$, destravando divisões em fluxos com muitas classes com uma fração das amostras.

6. **Mecanismo Adaptativo de Desempate ($\tau$-Decay Progressivo):**
   - Quando múltiplos atributos apresentam ganhos praticamente idênticos ($\Delta \bar{G} \approx 0$), o split convencional depende exclusivamente do limiar $\epsilon < \tau$.
   - Para evitar a estagnação prolongada na raiz ou em nós rasos, propõe-se um mecanismo de decaimento adaptativo do parâmetro de empate com base no número de amostras acumuladas $n$ na folha:
     $$\tau_n = \max\left(\tau_{\min}, \; \tau_0 \cdot \left(1 + \frac{\gamma \ln(n)}{\sqrt{n}}\right)\right)$$
     assegurando que nós saturados com alta evidência estatística realizem o split mesmo em cenários de alta simetria.

7. **Seleção e Sub-amostragem Online de Atributos (Random Subspaces):**
   - Em problemas com dezenas ou centenas de atributos, o custo de atualizar os observadores cresce linearmente com $D \times C$.
   - A introdução de sub-amostragem aleatória de atributos por amostra ou por ciclo de avaliação de split (avaliando apenas $k = \lceil \sqrt{D} \rceil$ atributos sorteados) acelera o processamento em até $5\times$ a $10\times$ e quebra a correlação em cenários com grande redundância de sensores.

### 5.3 Robustez e Pré-Condicionamento em Regressão Contínua (Lições do Bikes)

8. **Normalização Online Integrada nas Folhas (Streaming StandardScaler Zero-Malloc) — Concluído (v0.2.0):**
   - No benchmark do dataset *Bikes*, observou-se que atributos com ordens de grandeza muito distintas (ex: temperatura em $[0, 1]$, hora em $[0, 23]$ e umidade em $[0, 100]$) causaram instabilidade e oscilações no gradiente do regressor linear SGD das folhas.
   - **Solução Implementada (v0.2.0):** Padronizador híbrido com fator de credibilidade $\alpha = \frac{N_{\text{slots}}}{N_{\text{slots}} + 10.0}$ combinando os slots de variância com as estatísticas globais da folha via Welford:
     $$z_{j} = \frac{x_{j} - \bar{x}_{j}}{\sqrt{\sigma_{j}^2 + \varepsilon_s}}$$
     onde a média $\bar{x}_j$ e a variância $\sigma_j^2$ são atualizadas incrementalmente com $O(1)$ de RAM e tempo, fornecendo ao regressor SGD um espaço de gradiente esférico e bem condicionado com **0.0 KB de heap em runtime**. Ativado via flag `leaf_sgd_normalize` em `ht_config_params_t`.

9. **Otimizadores com Taxa Adaptativa e Regularização $\ell_2$ (Ridge Online):**
   - Substituição do SGD com taxa de aprendizado estática por otimizadores que incorporam decaimento temporal ou sensibilidade individual por atributo:
     $$\mathbf{w}_{t+1} = \mathbf{w}_t \cdot (1 - \eta_t \lambda_{\text{reg}}) - \eta_t (\hat{y}_t - y_t) \mathbf{z}_t, \quad \eta_t = \frac{\eta_0}{1 + \alpha \sqrt{n}}$$
     onde $\lambda_{\text{reg}}$ atua como regularização Ridge ($\ell_2$), impedindo que folhas recém-divididas com poucas observações extrapolem coeficientes para magnitudes instáveis.

10. **Gating Suave e Limites Físicos de Suporte:**
    - Em tarefas de regressão onde o alvo é intrinsecamente não-negativo (ex: contagem de passageiros no *NYC Taxi* ou demanda de aluguel de bicicletas no *Bikes*), um preditor puramente linear pode produzir valores negativos espúrios em caudas de distribuição.
    - Planeja-se adicionar uma camada de suporte com *soft-clipping* no intervalo observado $[y_{\min}, y_{\max}]$ e transição suave (*blending*) entre a média da folha $\bar{y}$ e o modelo linear $\hat{y}_{\text{model}}$ ponderada pelo inverso da variância residual local.

### 5.4 Divisões Oblíquas e Descorrelação de Atributos (Lições do MNIST)

11. **Árvores de Decisão Oblíquas em Streaming (Streaming Oblique Trees):**
    - Conforme demonstrado na avaliação do MNIST, as árvores com divisões estritamente ortogonais ($x_f \le \theta$) sofrem com o "efeito escada" ao particionar fronteiras contínuas diagonais, exigindo centenas de nós para aproximar uma única reta de separação.
    - A implementação de nós de decisão oblíquos particiona o espaço amostral via combinações lineares:
      $$\sum_{j=1}^{D} w_j x_j + b \le 0$$
      onde os coeficientes $\mathbf{w}$ e o viés $b$ são ajustados online na folha antes do split por um discriminante linear incremental (ex: Perceptron com margem ou SVM linear SGD).

12. **Observadores com Covariância Local em Blocos 2D:**
    - O classificador *Naive Bayes Adaptive* (NBA) nas folhas assume a hipótese de independência condicional entre os atributos: $P(\mathbf{x} \mid C) = \prod P(x_i \mid C)$. Essa premissa é fortemente violada em sinais físicos acoplados (como sensores triaxiais de vibração ou pixels vizinhos).
    - Propõe-se um observador numérico de covariância em blocos compactos de tamanho $2 \times 2$ ou $3 \times 3$, capturando a correlação entre canais ortogonais sem incorrer na sobrecarga cúbica de uma matriz de covariância densa completa.

### 5.5 Gestão Dinâmica de Memória e Poda Online (Lições dos Tiers 4 e 8)

13. **Alocação de Folhas Guiada por Erro Residual (Error-Driven Leaf Budgeting):**
    - Em microcontroladores com orçamento estrito de memória (como o Tier 8 e o Tier 4, limitados a 8 e 4 folhas ativas), as folhas excedentes são desativadas e deixam de acumular estatísticas de split.
    - Como melhoria, propõe-se um escalonador de observadores baseado no erro residual: quando o pool de `ht_leaf_stats_t` se esgota, os observadores são transferidos dinamicamente para as folhas que concentram a maior variância ou maior erro quadrático médio no fluxo recente, concentrando o refinamento onde a árvore mais erra.

14. **Fusão de Folhas (*Leaf Merging*) e Poda Estatística no VFDT Estático:**
    - Diferente da variante adaptativa HAT (que substitui subárvores inteiras ao detectar *drift* via ADWIN), o VFDT estático não possui mecanismo de poda retrospectiva.
    - Planeja-se introduzir uma rotina de fusão de folhas irmãs (*leaf merging*) que avalia se os modelos preditivos dos dois nós filhos convergiram para distribuições idênticas via teste de hipótese bilateral, colapsando-os de volta em um único nó folha e devolvendo os nós excedentes ao pool estático `free_nodes`.

### 5.6 Estabilidade Numérica, Ecosistema e Ensembles

15. **Observadores com Decaimento Exponencial (Fading Memory Welford):**
    - Em fluxos contínuos de centenas de milhões de amostras em precisão simples (`float32`), somatórios acumulados podem sofrer cancelamento catastrófico quando a variância é diminuta em relação à magnitude da média ($\sigma^2 \ll \mu^2$).
    - Para mitigar esse risco e habilitar o rastreamento suave de mudanças lentas (*gradual drift*), planeja-se introduzir observadores com memória atenuada por esquecimento exponencial:
      $$w_t = \lambda w_{t-1} + 1, \quad \mu_t = \mu_{t-1} + \frac{x_t - \mu_{t-1}}{w_t}, \quad M_{2,t} = \lambda M_{2,t-1} + (x_t - \mu_{t-1})(x_t - \mu_t)$$
      com fator de esquecimento $\lambda \in [0{,}999, 0{,}99999]$, garantindo que acumuladores numéricos operem assintoticamente dentro de faixas seguras de representação IEEE-754.

16. **Discretizadores Não-Paramétricos Estáticos (Histograma Ben-Haim & Tom-Tov) — Concluído (v0.2.0):**
    - Para variáveis com distribuições altamente multimodais ou caudas pesadas que violam a suposição de Gaussianidade unimodal de Welford, implementou-se o observador não-paramétrico de Ben-Haim & Tom-Tov (`HT_OBS_HISTOGRAM`, struct `ht_hist_obs_t`).
    - Mantém até 4 bins dinâmicos por padrão (`HT_HIST_MAX_BINS = 4`, configurável via macro) que se fundem deterministicamente quando a capacidade é excedida, calculando probabilidades condicionais via interpolação trapezoidal e gerando candidatos a split no ponto médio dos centróides adjacentes.
    - Ocupa exatamente os mesmos $80\text{ B}$ (em `float32`) da Gaussiana via `union ht_feature_obs_t`, preservando estritamente a invariância de `sizeof(ht_leaf_stats_t)`. Adicionalmente, disponibilizou-se `ht_quantizer_t` em `libht_prep` para discretização/quantização em streaming.

17. **Bindings em Python (CFFI / ctypes / Cython):**
    - Disponibilização de um wrapper Python de alto desempenho expondo a API clássica do Scikit-Learn e River (`fit_one`, `predict_one`, `predict_proba_one`), permitindo que cientistas de dados usem o núcleo ultrarrápido em C11 diretamente em pipelines de experimentação ($200\times$ a $1.000\times$ mais rápido que implementações em Python puro).

18. **Micro-Ensembles com Observadores Compartilhados e Ensembles Online Embarcados (`< 100 KB` SRAM):**
    - Em vez de uma *Adaptive Random Forest* tradicional (que multiplicaria o consumo de RAM dos observadores de Welford/Histograma pelo número $M$ de árvores), implementar um **comitê de 3 a 5 árvores rasas (Tier 2 ou Tier 4)** que **compartilham a mesma estrutura de estatísticas na raiz** ou operam sobre **subespaços aleatórios de features** dimensionados via *slab* (`leaf_stride` da `v0.4.0`), elevando drasticamente o Macro F1 no *Forest Covertype* e o $R^2$ nas regressões mantendo o conjunto estritamente abaixo de **`100 KB` de SRAM**.
    - Como extensão para perfis de maior capacidade (até $1\text{ MB}$), suporte complementar a *Online Bagging* de Oza & Russell (Poisson) ou *Leveraging Bagging* distribuindo o pool estático de nós e detectores entre $M$ árvores leves.

19. **Micro-Pipelines Embarcados de Pré-Processamento Zero-Malloc (`libht_prep`) — Concluído (v0.2.0):**
    - Criação de uma biblioteca companheira modular em C11 estático (`code/include/core/ht_prep.h` e `code/src/core/ht_prep.c`) com **zero alocação dinâmica (`malloc`/`free`)** em runtime:
      - `ht_scaler_t`: Normalização online Z-score ($z = \frac{x - \mu}{\sigma}$) e Min-Max ($[0, 1]$) em $O(1)$ de RAM e tempo via algoritmo de Welford com proteção contra variância nula.
      - `ht_filter_hampel_t`: Filtro mediano deslizante com Desvio Absoluto da Mediana (MAD, $3\sigma$) com buffer circular estático para eliminação de espículas (*outliers*) e ruído elétrico de sensores industriais em tempo real.
      - `ht_encoder_freq_t`: Codificador de frequência incremental para variáveis categóricas de alta cardinalidade via algoritmo determinístico Space-Saving de Metwally.
      - `ht_quantizer_t`: Discretizador/quantizador contínuo online baseado em histogramas dinâmicos de Ben-Haim & Tom-Tov com estimativa trapezoidal de CDF e $k$ bins uniformes.
      - `ht_prep_pipeline_t`: Estrutura de micro-pipeline anexo encadeável à árvore via `ht_tree_set_prep_pipeline`, aplicando transformações antes do treino (`ht_tree_learn_one`) e da inferência (`ht_tree_predict`) sem alocar memória na heap.
