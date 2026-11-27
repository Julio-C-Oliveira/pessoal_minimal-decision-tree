# Aprendizado Federado Embarcado (`Fed-Forest` e `Fed-VFDT`)

Introduzido na versão **`v0.5.0`** (Fase 11), o módulo federado do **kansho** permite que dispositivos embarcados (*edge / IoT*) aprendam colaborativamente sem transmitir dados brutos na rede e mantendo **zero alocação dinâmica (`malloc`/`free`) em runtime**. Todo o estado do cliente e do servidor reside em structs estáticas compactas e troca quadros binários autocontidos através do protocolo **`HFED`**.

> [!NOTE]
> **Base Científica e Créditos:** O **Modo 2A (`Fed-VFDT` Coordenado)** implementa e adapta para microcontroladores C11 o algoritmo **Fed-VFDT** proposto por **Paula Raissa Silva, João Vinagre e João Gama** (*IEEE ICTAI 2025*, DOI [`10.1109/ICTAI66417.2025.00094`](https://doi.org/10.1109/ICTAI66417.2025.00094), código de referência [`paularaissa/project-vfdt`](https://github.com/paularaissa/project-vfdt)), incorporando também o conceito de regressão coordenada do **FHT-R** (*Daniela Menezes, Douglas Vidal, André Riker e Glauco Gonçalves, SBrT 2026*). O **Modo 1 (`Fed-Forest`)**, o **Modo 2B (`Stats-Merge`)**, o protocolo binário **`HFED` + CRC32**, o versionamento com `split_log` / `FULL_SNAPSHOT` e o reset coordenado por *concept drift* são extensões originais do **kansho**. Veja a delimitação detalhada na [Seção 6](#6-créditos-originalidade-e-referências-bibliográficas) e em **[Origem, Linhagem e Originalidade](Origem-River-Scikit-Learn-e-Originalidade.md)**.

---

## 1. Os Dois Paradigmas Federados

A arquitetura oferece dois modos complementares configuráveis via headers públicos em [`code/include/federated/`](../code/include/federated):

| Paradigma | Mecanismo Central | Uplink do Cliente | Downlink do Servidor | Banda Média / Rodada | Ideal Para |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Modo 1 — `Fed-Forest`** *(Autoral)*<br/>([`ht_fed_forest.h`](../code/include/federated/ht_fed_forest.h)) | Cada cliente treina 1 árvore local completa (`ht_tree_t`) e compartilha sua topologia compactada via BFS (`ht_compact_tree_t`, ~1,34 KB em 64 nós). | Árvore compacta somente-inferência + erro prequential. | Comitê de até $M$ árvores pares selecionadas via `ALL`, `TOP_M` ou `ROUND_ROBIN`. | ~738 B – 3,9 KB | Regressão contínua Non-IID, clientes assíncronos e votação adaptativa ($M \le 8$). |
| **Modo 2A — `Fed-VFDT` Coordenado** *(Baseado em Silva, Vinagre & Gama, ICTAI 2025)*<br/>([`ht_fed_vfdt.h`](../code/include/federated/ht_fed_vfdt.h)) | Árvore global única sincronizada por versão (`HT_TREE_FLAG_DEFER_SPLITS`). Clientes avaliam o *Hoeffding Bound* localmente e votam por consenso (`QUORUM`, `MAJORITY`, `MERIT` ou `WEIGHTED_MERIT`). | Propostas compactas de split (`ht_fed_split_proposal_t`) + alerta opcional de drift. | Comandos incrementais de split/reset (`split_log`) ou `FULL_SNAPSHOT`. | **114 B – 564 B** | Links ultrarestritos (LoRaWAN, BLE, NB-IoT, UART) com paridade conceitual ao *Fed-VFDT* (`paularaissa/project-vfdt`). |
| **Modo 2B — `Fed-VFDT Stats-Merge`** *(Autoral)*<br/>([`ht_fed_vfdt.h`](../code/include/federated/ht_fed_vfdt.h)) | Fusão paralela exata sem perda das estatísticas suficientes nas folhas (Chan/Welford, matrizes categóricas, histogramas Ben-Haim, slots de regressão + FedAvg de pesos SGD) sem contagem dupla. | Pacote binário com estatísticas das folhas ativas. | Árvore global evolui com o *Hoeffding Bound* global e transmite deltas via `split_log`. | ~2,0 KB – 53 KB | Cenários **Non-IID extremos** (clientes vendo apenas 1–2 classes ou faixas estreitas de $y$). |

---

## 2. Protocolo Binário `HFED` e Recuperação de *Stragglers*

O transporte é desacoplado de qualquer stack de rede específica (TCP/UDP, UART, BLE, ESP-NOW, LoRa) em [`ht_fed_proto.h`](../code/include/federated/ht_fed_proto.h). A biblioteca produz e consome exclusivamente buffers de bytes em memória fornecidos pelo integrador.

- **Cabeçalho Fixo (`ht_fed_header_t`, exatamente `32 bytes`):**
  - `magic`: `0x44454648U` (`"HFED"` em little-endian) e `proto_version = 1`.
  - `msg_type` (`ht_fed_msg_type_t`):
    1. `HT_FED_MSG_FOREST_CLIENT_TREE` (uplink `Fed-Forest`)
    2. `HT_FED_MSG_FOREST_SERVER_BUNDLE` (downlink `Fed-Forest`)
    3. `HT_FED_MSG_VFDT_SPLIT_PROPOSAL` (uplink `Fed-VFDT` Coordenado)
    4. `HT_FED_MSG_VFDT_LEAF_STATS` (uplink `Fed-VFDT` `Stats-Merge`)
    5. `HT_FED_MSG_VFDT_SPLIT_BROADCAST` (downlink incremental do `split_log`)
    6. `HT_FED_MSG_VFDT_FULL_SNAPSHOT` (downlink completo para ressincronização)
  - `flags` & `float_size`: valida paridade de precisão (`HT_FED_FLAG_DOUBLE`) e tarefa (`HT_FED_FLAG_REG`).
  - `client_id`, `round_id`, `topology_version`: identificam o cliente e impedem aplicação de atualizações fora de sincronia estrutural.
  - `crc32`: soma de verificação IEEE 802.3 calculada via tabela nibble compacta de **64 bytes** (`ht_fed_crc32`), rejeitando pacotes corrompidos com `HT_ERR_CORRUPT_IMAGE`.
- **Recuperação Automática de *Stragglers* (`split_log` + `FULL_SNAPSHOT`):**
  - Se um cliente desconectado enviar um uplink com `topology_version != server->topology_version`, o servidor rejeita a agregação com `HT_ERR_STALE_VERSION`.
  - Ao chamar `ht_fed_vfdt_server_export_update(&server, cid, client_topo_ver, ...)`, o servidor envia apenas os comandos perdidos no buffer circular `split_log[HT_FED_SPLIT_LOG_CAP]` (padrão `64` comandos) caso o atraso caiba no log, ou gera automaticamente um **`HT_FED_MSG_VFDT_FULL_SNAPSHOT`** completo quando o cliente ficou mais de `HT_FED_SPLIT_LOG_CAP` versões para trás.
- **Reset Coordenado por Concept Drift:**
  - Clientes com anomalias locais podem sinalizar `ht_fed_vfdt_client_flag_drift(&client, node_idx)`. Quando o servidor atinge `min_drift_quorum` alertas sobre a mesma subárvore, emite um comando `HT_FED_CMD_RESET_SUBTREE` no `split_log` para reiniciar deterministicamente o ramo em toda a frota.

---

## 3. Exemplos Práticos em C11

### 3.1 Exemplo — Modo 1: `Fed-Forest` (Cliente + Servidor + Pacotes `HFED`)

```c
#include <stddef.h>
#include <stdint.h>
#include "ht_config.h"
#include "ht_tree.h"
#include "ht_fed_forest.h"

static ht_fed_forest_client_t cli;
static ht_fed_forest_server_t srv;
static uint8_t wire_buf[16384];

void run_fed_forest_round(const ht_value_t x[HT_MAX_FEATURES], uint8_t y) {
    ht_config_params_t tp;
    ht_default_params(&tp, /*n_features=*/8, /*n_classes=*/2);

    /* 1. Inicializa cliente #1 e servidor (apenas no setup inicial) */
    ht_fed_forest_client_cfg_t ccfg;
    ht_fed_forest_default_client_config(&ccfg, /*client_id=*/1, &tp);
    ccfg.voting = HT_VOTE_WEIGHTED;
    ht_fed_forest_client_init(&cli, &ccfg);

    ht_fed_forest_server_cfg_t scfg;
    ht_fed_forest_default_server_config(&scfg, HT_TASK_CLASSIFICATION);
    scfg.agg_policy = HT_FED_FOREST_AGG_TOP_M;
    scfg.max_forest_size = 5; /* M <= HT_FED_FOREST_MAX_TREES */
    ht_fed_forest_server_init(&srv, &scfg);

    /* 2. Test-Then-Train local ponderado (local + M arvores pares) */
    uint8_t y_hat = ht_fed_forest_client_predict(&cli, x);
    (void)y_hat;
    ht_fed_forest_client_learn_one(&cli, x, y, 1.0f);

    /* 3. Uplink: exporta arvore BFS compacta em frame HFED com CRC32 */
    size_t tx_bytes = 0;
    if (ht_fed_forest_client_export_update(&cli, wire_buf, sizeof(wire_buf), &tx_bytes) == HT_OK) {
        ht_fed_forest_server_ingest(&srv, wire_buf, tx_bytes);
    }

    /* 4. Downlink: servidor seleciona ate M arvores pares (excluindo o proprio cliente) */
    ht_fed_forest_server_advance_round(&srv);
    size_t rx_bytes = 0;
    if (ht_fed_forest_server_build_forest(&srv, /*target_client_id=*/1,
                                          /*exclude_self=*/true,
                                          wire_buf, sizeof(wire_buf), &rx_bytes) == HT_OK) {
        ht_fed_forest_client_import_forest(&cli, wire_buf, rx_bytes);
    }
}
```

### 3.2 Exemplo — Modo 2: `Fed-VFDT` (`COORDINATED` ou `STATS_MERGE`)

```c
#include <stddef.h>
#include <stdint.h>
#include "ht_config.h"
#include "ht_tree.h"
#include "ht_fed_vfdt.h"

static ht_fed_vfdt_client_t vf_cli;
static ht_fed_vfdt_server_t vf_srv;
static uint8_t fed_wire[65536];

void run_fed_vfdt_round(const ht_value_t x[HT_MAX_FEATURES], uint8_t y) {
    ht_config_params_t tp;
    ht_default_params(&tp, /*n_features=*/8, /*n_classes=*/2);

    /* Escolha o modo: HT_FED_VFDT_COORDINATED (propostas) ou HT_FED_VFDT_STATS_MERGE */
    ht_fed_vfdt_mode_t mode = HT_FED_VFDT_STATS_MERGE;

    ht_fed_vfdt_client_cfg_t ccfg;
    ht_fed_vfdt_default_client_config(&ccfg, /*client_id=*/1, mode, &tp);
    ht_fed_vfdt_client_init(&vf_cli, &ccfg);

    ht_fed_vfdt_server_cfg_t scfg;
    ht_fed_vfdt_default_server_config(&scfg, mode, &tp);
    scfg.consensus = HT_FED_CONSENSUS_QUORUM;
    scfg.threshold_agg = HT_FED_THRESH_MAX_MERIT;
    scfg.min_quorum = 2;
    ht_fed_vfdt_server_init(&vf_srv, &scfg);

    /* 1. Ciclo prequential local (splits autonomos suspensos por HT_TREE_FLAG_DEFER_SPLITS) */
    uint8_t pred = ht_fed_vfdt_client_predict(&vf_cli, x);
    (void)pred;
    ht_fed_vfdt_client_learn_one(&vf_cli, x, y, 1.0f);

    /* 2. Rodada no servidor */
    ht_fed_vfdt_server_begin_round(&vf_srv);

    size_t up_len = 0;
    ht_fed_vfdt_client_export_update(&vf_cli, fed_wire, sizeof(fed_wire), &up_len);

    ht_status_t st = ht_fed_vfdt_server_ingest(&vf_srv, fed_wire, up_len);
    if (st == HT_ERR_STALE_VERSION) {
        /* Cliente straggler desatualizado: exporte delta/snapshot e tente na proxima rodada */
    }

    uint16_t cmds_applied = 0;
    ht_fed_vfdt_server_end_round(&vf_srv, &cmds_applied);

    /* 3. Downlink personalizado a topology_version atual do cliente */
    size_t down_len = 0;
    if (ht_fed_vfdt_server_export_update(&vf_srv, vf_cli.client_id, vf_cli.topology_version,
                                         fed_wire, sizeof(fed_wire), &down_len) == HT_OK) {
        ht_fed_vfdt_client_apply_update(&vf_cli, fed_wire, down_len);
    }
}
```

> [!TIP]
> Para compilar projetos próprios com os módulos federados, adicione os includes e fontes modulares:
> ```bash
> gcc -std=c11 -Os -Icode/include -Icode/include/core -Icode/include/ensemble -Icode/include/federated \
>     code/src/core/*.c code/src/ensemble/*.c code/src/federated/*.c meu_app.c -lm -o app_fed
> ```
> Ou simplesmente linke `code/build/libht.a` gerada via `make lib`.

---

## 4. Macros de Dimensionamento Estático

Além das macros centrais em [`code/include/core/ht_config.h`](../code/include/core/ht_config.h) (documentadas em [Referência de Parâmetros e Macros](Referencia-de-Parametros-e-Macros.md)), a camada modular `ensemble/` + `federated/` expõe:

| Macro | Padrão | Onde é Definida | O que Controla |
| :--- | :---: | :--- | :--- |
| **`HT_COMPACT_MAX_NODES`** | `128` | `ht_config.h` | Capacidade máxima de nós por árvore compactada (`ht_compact_tree_t`, $\approx 2{,}59\text{ KB}$ em `128` nós ou $\approx 1{,}34\text{ KB}$ em `64` nós). |
| **`HT_FED_FOREST_MAX_TREES`** | `8` | `ht_config.h` | Máximo $M$ de árvores pares armazenadas simultaneamente em cada cliente `Fed-Forest`. |
| **`HT_FED_SERVER_MAX_TREES`** | `16` | `ht_config.h` | Capacidade máxima de registros de clientes mantida no servidor `Fed-Forest`. |
| **`HT_FED_MAX_CLIENTS`** | `16` | `ht_config.h` | Capacidade de votos e metadados por cliente nos buffers estáticos de rodada e drift do servidor. |
| **`HT_FED_MAX_PROPOSALS`** | `16` | `ht_config.h` | Limite de propostas por cliente em cada rodada do modo `HT_FED_VFDT_COORDINATED`. |
| **`HT_FED_SPLIT_LOG_CAP`** | `64` | `ht_config.h` | Tamanho do log circular de comandos de topologia para *catch-up* incremental de *stragglers*. |
| **`HT_FED_SERVER_MAX_PROPOSALS`** | `64` | `ht_fed_vfdt.h` | Limite total de propostas acumuladas por rodada no servidor `Fed-VFDT` Coordenado. |

---

## 5. Resultados Experimentais e Validação de Transporte/Hardware

### 5.1 Benchmark Federado nos 8 Datasets Reais (`make bench-main-federated`)

Avaliação completa com **$K = 10$ clientes** e **$R = 20$ rodadas** sob perfil MCU **Tier 8** (`HT_MAX_ACTIVE_LEAVES=8`, `HT_MAX_NODES=128`, `HT_COMPACT_MAX_NODES=64`, `HT_FED_FOREST_MAX_TREES=5`, `float32`, zero `malloc`), particionamento **Non-IID `Dirichlet(alpha=0.3)`** e avaliação no conjunto de teste global (relatório completo em [`main_benchmark/federated_benchmark_report.md`](../main_benchmark/federated_benchmark_report.md)):

| Dataset | Tarefa | Local-Only ($K$ isolados) | Melhor `Fed-Forest` | Melhor `Fed-VFDT` (Coord.) | `Fed-VFDT Stats-Merge` | Central Oracle (T8) | RAM Cli Venc. (Mod + Com) | RAM Srv Venc. (Mod + Com) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SEA** | Classif. (Acc) | 76,41% | 95,06% (`TOP_M`) | 93,68% (`MAJ`/`MERIT`) | **96,78%** | 99,02% | 6,7 + 2,2 = 8,8 KB | 6,7 + 12,0 = 18,6 KB |
| **Electricity** | Classif. (Acc) | 50,65% | 54,78% (`TOP_M`) | 57,61% (`QUORUM`) | **60,37%** | 64,37% | 9,8 + 5,3 = 15,1 KB | 9,8 + 15,1 = 24,9 KB |
| **Forest Covertype** *(54 feat, 7 cls)* | Classif. (Acc) | 30,91% | 25,83% (`RR`) | 35,00% (`MERIT`) | **65,52%** | 67,28% | **65,5 + 59,5 = 125,0 KB** | **65,5 + 75,6 = 141,2 KB** |
| **Airlines** | Classif. (Acc) | 49,60% | **76,39%** (`TOP_M`) | 56,78% (`MAJ`/`MERIT`) | 61,43% | 76,09% | 15,8 + 0,9 = 16,7 KB | 20,8 + 1,1 = 21,9 KB |
| **Friedman #1** | Regr. (MAE) | 2,8288 | 2,8813 (`TOP_M`) | **2,4963** (`QUORUM`) | 2,7318 | 2,1537 | 25,5 + 0,2 = 25,7 KB | 25,2 + 10,0 = 35,3 KB |
| **Bikes** | Regr. (MAE) | 61,6123 | **41,7974** (`RR`) | 73,6779 (`MAJ`) | 46,6150 | 40,4648 | 60,2 + 3,1 = 63,3 KB | 21,2 + 3,3 = 24,5 KB |
| **Appliances Energy** *(28 feat)* | Regr. (MAE) | 78,5876 | **35,4504** (`TOP_M`) | 59,1440 (`MAJ`) | 50,5366 | 29,7630 | 41,1 + 4,7 = 45,8 KB | 22,4 + 4,9 = 27,3 KB |
| **NYC Taxi** | Regr. (MAE) | 397,0585 | **351,5022** (`RR`) | 381,5276 (`QUORUM`) | 372,1993 | 339,2442 | 18,8 + 3,8 = 22,6 KB | 20,8 + 4,1 = 24,9 KB |

- **Orçamento de RAM Decomposto (Modelo vs. Comunicação):** Isolando o **modelo estático**, o pico no cliente foi de **76,95 KB** (`Fed-Forest`, $M=5$) e **65,55 KB** (`Fed-VFDT`). Incluindo também o **buffer half-duplex de comunicação (`wire_buf`) + coordenação**, **`Fed-VFDT Coordenado`** (**66,25 KB** cliente / **82,38 KB** servidor) e **`Fed-Forest`** (**81,00 KB** cliente / **40,76 KB** servidor para 16 árvores) permanecem abaixo de **100 KB**; já **`Fed-VFDT Stats-Merge`** exige buffer TX de até **59,48 KB** no Covertype (**125,04 KB** cliente / **141,16 KB** servidor, stack servidor **7,61 KB**) e **48,68 KB** no Bikes (**102,30 KB** cliente).
- **`Stats-Merge` para Non-IID severo multiclasse:** Recupera +34,61 pp sobre `Local-Only` no Covertype (65,52% vs 30,91%; Macro F1 de 63,04% vs 17,52%).
- **`Fed-Forest` na regressão:** Vence em 3 dos 4 datasets de regressão contínua reduzindo o MAE em até 54,9% (*Appliances Energy*), com servidor compacto (**22,05–40,76 KB** totais).
- **`Fed-VFDT Coordenado` em links lentos:** Transmite apenas **114 a 564 bytes/cliente/rodada** com buffer TX/RX **< 1 KB** observado nos benchmarks de 8 datasets.

### 5.2 Transporte TCP POSIX em Processos Separados (`make bench-fed-net`)

Validado em [`benchmarks/fed_net/`](../benchmarks/fed_net/run_fed_net_tests.py) com **1 processo servidor C11** e **$K = 4$ processos clientes OS independentes** sobre sockets TCP loopback:
- **5/5 cenários aprovados em `0,33 s`** (zero warnings com `-std=c11 -O2 -Wall -Wextra -Werror -pedantic`):
  1. `Fed-Forest` (4 classes Non-IID) com fragmentação TCP de `17 B/write` e injeção de bit-flip (`CRC32` rejeitado com `HT_ERR_CORRUPT_IMAGE` + retransmissão): **98,50% Acc** (`16,8 KB`).
  2. `Fed-VFDT Coordenado` com fragmentação de `13 B/write` e rejeição `CRC32`: **94,25% Acc** (`Splits=3`, `TopoV=3`, `5,4 KB`).
  3. `Fed-VFDT Stats-Merge` com clientes **100% puros (1 classe por processo)** e fragmentação de `19 B/write`: **97,50% Acc** (`Splits=5`, `TopoV=5`, `36,1 KB`).
  4. *Straggler* real desconectado (`client 4` offline nas rodadas 2..6, reconectando na rodada 7) sob fragmentação de `11 B/write`: **96,75% Acc** (`TopoV=3/3`, `Catchups=7`, `28,4 KB`).
  5. `Fed-VFDT Stats-Merge` em regressão contínua Non-IID com fragmentação de `23 B/write`: **RMSE 3,493** (`Splits=21`, `TopoV=21`, `59,0 KB`).

### 5.3 Tamanho de Flash em 32 Bits (`-Os`)

Compilação cruzada validada na entrega `11.11` com zero warnings em **`.data = 0 B`** e **`.bss = 0 B`** internos à biblioteca (`docs/13_federated_learning.md`, Seção 9):

| Alvo Embarcado / Toolchain | `ensemble/` + `federated/` (`.text`) | 11 Módulos Completos (`.text`) |
| :--- | :---: | :---: |
| **ARM Cortex-M3** (`arm-none-eabi-gcc 14.2.1`, Thumb-2) | **11.342 B (11,08 KB)** | **40.614 B (39,66 KB)** |
| **ESP32 Xtensa LX6** (`xtensa-esp32-elf-gcc 12.2.0`) | **12.119 B (11,83 KB)** | **41.582 B (40,61 KB)** |

---

## 6. Créditos, Originalidade e Referências Bibliográficas

### 6.1 Trabalhos Científicos e Código de Referência
1. **Fed-VFDT (Base Principal — Modo 2A Coordenado em Classificação):**
   - **Autores:** Paula Raissa Silva, João Vinagre e João Gama (INESC TEC / Universidade do Porto)
   - **Título:** *Fed-VFDT: Federated Very Fast Decision Trees with Coordinated Splitting Over Data Streams*
   - **Publicação:** *2025 IEEE 37th International Conference on Tools with Artificial Intelligence (ICTAI)*, pp. 653–660
   - **DOI:** [`10.1109/ICTAI66417.2025.00094`](https://doi.org/10.1109/ICTAI66417.2025.00094)
   - **Repositório Oficial:** [`paularaissa/project-vfdt`](https://github.com/paularaissa/project-vfdt) (`client_app/fht.py`, `client_app/client_app.py`, `server_app/server_app.py`, `server_app/my_server.py`)
2. **FHT-R (Referência Complementar — Extensão Coordenada para Regressão):**
   - **Autores:** Daniela Menezes, Douglas Vidal, André Riker e Glauco Gonçalves
   - **Título:** *FHT-R: Federated Online Regression with Coordinated Hoeffding Trees for Air Quality Prediction*
   - **Publicação:** *XLIV Simpósio Brasileiro de Telecomunicações e Processamento de Sinais (SBrT 2026)*

### 6.2 Delimitação de Autoria (O que Veio do `Fed-VFDT` vs. Extensões do `kansho`)
- **Baseado em Silva, Vinagre & Gama (`paularaissa/project-vfdt`) e `FHT-R`:**
  - Coordenação de splits sem envio de dados brutos (splits locais diferidos + propostas avaliadas pelo *Hoeffding Bound*), estratégias de consenso no servidor (`QUORUM` por atributo com `sum(merit)`, `MAJORITY` por pluralidade e `BEST-MERIT` `max(G_k)`), seleção de threshold `HT_FED_THRESH_MAX_MERIT`, filtro `require_locally_ready = true`, broadcast de threshold global único com injeção de `children_stats`, escala `client_grace = G / K` e aplicação do paradigma à regressão online (*FHT-R*).
- **Original / Reprojetado no `kansho` para Microcontroladores C11:**
  - Implementação 100% estática zero-`malloc` em C11 puro (sem Flower, Python, JSON ou Pickle), operação não-bloqueante (sem `_frozen` nem `_instance_buffer`), endereçamento `uint16_t` com `topology_version` + log circular `split_log[64]` + fallback automático `FULL_SNAPSHOT`, **Modo 1 (`Fed-Forest`)** com compactação BFS e votação ponderada, **Modo 2B (`STATS_MERGE`)** com fusão paralela exata sem contagem dupla + FedAvg SGD, protocolo binário **`HFED` (`32 B`) + CRC32** e **reset coordenado por concept drift** (`HT_FED_CMD_RESET_SUBTREE`).

Para a matriz comparativa linha a linha entre `paularaissa/project-vfdt` e o `kansho`, consulte **[Origem, Linhagem e Originalidade](Origem-River-Scikit-Learn-e-Originalidade.md#4-o-que-foi-extraído-do-fed-vfdt-paularaissaproject-vfdt-e-do-fht-r)** e [`docs/13_federated_learning.md`](../docs/13_federated_learning.md).

### 6.3 Citações em BibTeX

```bibtex
@inproceedings{silva2025fedvfdt,
  author    = {Silva, Paula Raissa and Vinagre, Jo{\~a}o and Gama, Jo{\~a}o},
  title     = {Fed-VFDT: Federated Very Fast Decision Trees with Coordinated Splitting Over Data Streams},
  booktitle = {2025 IEEE 37th International Conference on Tools with Artificial Intelligence (ICTAI)},
  pages     = {653--660},
  year      = {2025},
  doi       = {10.1109/ICTAI66417.2025.00094}
}

@inproceedings{menezes2026fhtr,
  author    = {Menezes, Daniela and Vidal, Douglas and Riker, Andr{\'e} and Gon{\c{c}}alves, Glauco},
  title     = {FHT-R: Federated Online Regression with Coordinated Hoeffding Trees for Air Quality Prediction},
  booktitle = {XLIV Simp{\'o}sio Brasileiro de Telecomunica{\c{c}}{\~o}es e Processamento de Sinais (SBrT 2026)},
  year      = {2026}
}
```
