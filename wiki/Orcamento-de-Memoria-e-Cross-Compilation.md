# Orçamento de Memória e Cross-Compilation (Flash / RAM)

Para sistemas embarcados e dispositivos fracos (microcontroladores ARM Cortex-M, ESP32, RISC-V, ou SoCs Linux compactos como MIPS OpenWrt), a alocação de memória se divide em duas grandezas fundamentais:
1. **Armazenamento Permanente / Código Compilado (Flash / ROM / Disco):** Onde fica gravado o binário executável (`.text` + `.rodata`) e os arquivos de modelo serializados transferidos do PC.
2. **Memória de Trabalho Volátil (RAM / `.bss`):** Onde reside o estado operacional da árvore (`ht_tree_t`) durante o treino ou inferência.

---

## 1. Peso do Código Compilado no Dispositivo (Flash / ROM / Binário)

Ao compilar a biblioteca no PC para gerar os arquivos binários gravados no dispositivo, o consumo em armazenamento depende dos módulos linkados (`core/`, `ensemble/`, `federated/`) e das flags de compilação:

| Modo de Uso / Funcionalidades Linkadas | Seção `.text` (Código C11 em `-Os`) | Seção `.rodata` / `.data` / `.bss` | Medição Real 32-bit / Status |
| :--- | :---: | :---: | :--- |
| **Núcleo `core/` Completo em `-Os` (`6` módulos)**<br/>*(VFDT, HAT, VFDT-R, HAT-R, ADWIN2, `libht_prep`, serialização)* | **$\approx 28{,}6\text{ KB}$** *(Cortex-M3)*<br/>**$\approx 28{,}8\text{ KB}$** *(ESP32)* | `.rodata < 0,7 KB`<br/>`.data = 0 B`, `.bss = 0 B` | Medido em `arm-none-eabi-gcc 14.2.1` e `xtensa-esp32-elf-gcc 12.2.0` (`-Os`). |
| **Camada Federada + Ensembles (`5` módulos em `-Os`)**<br/>*(`ht_compact_tree`, `ht_voting`, `ht_fed_proto`, `ht_fed_forest`, `ht_fed_vfdt`)* | **`11,08 KB` (`11.342 B`)** *(Cortex-M3)*<br/>**`11,83 KB` (`12.119 B`)** *(ESP32)* | `.data = 0 B`<br/>`.bss = 0 B` | Medido por módulo em `emulation/cortex-m3/` e `emulation/esp32/`. |
| **Biblioteca Completa (`11` módulos em `-Os`)** | **`39,66 KB` (`40.614 B`)** *(Cortex-M3)*<br/>**`40,61 KB` (`41.582 B`)** *(ESP32)* | `.data = 0 B`<br/>`.bss = 0 B` | Todos os 11 módulos C11 compilados com `-std=c11 -Wall -Wextra -Werror -pedantic -Os`. |
| **Somente Inferência em MCU (`--gc-sections`)**<br/>*(Sem treino incremental, regressão ou drift)* | **$\approx 6\text{ a } 8\text{ KB}$** *(estimativa `--gc-sections`)* | $< 0,7\text{ KB}$ | Descarta observadores e critérios com `-ffunction-sections -fdata-sections`. |

> [!NOTE]
> - **Zero overhead estático interno na biblioteca:** A `libht` possui `0 B` em `.data` e `0 B` em `.bss` internos. Todo o footprint de execução reside exclusivamente nas instâncias alocadas pelo usuário (`ht_tree_t`, `ht_fed_forest_client_t`, `ht_fed_vfdt_client_t`).
> - **Microcontroladores de 32 bits (ARM Thumb-2 / Xtensa LX6):** Na prática em hardware emulado, o firmware bare-metal completo do **ARM Cortex-M3** (`lm3s6965evb`, Tier 4 sem FPU) ocupa **66.220 B de Flash** e **57.468 B de RAM ($\le 64\text{ KB}$)** em `float32` (`soft-fp`), caindo para **`49.264 B` de Flash (`-25,6%`)** e **`57.104 B` de RAM** com **`-DHT_USE_FIXED_POINT`** (`Q16.16` zero-`libm`, `v0.6.0`, speedup SysTick de `1,86×–4,48×` e `0` símbolos `soft-fp`/`libm`); no **ESP32**, **186.263 B de Flash** e **148.904 B de DRAM**.

---

## 2. Peso do Arquivo do Modelo Serializado (Transferência PC $\rightarrow$ Dispositivo)

Caso o fluxo de trabalho seja treinar ou pré-treinar o modelo no PC e transferir apenas o arquivo do modelo para o dispositivo rodar inferência em campo (`ht_tree_save` / `ht_tree_load`):

- **Estrutura Binária:**
  - **Cabeçalho Fixo (`ht_serial_header_t`):** Fixos **$172\text{ bytes}$** em `float32` (**$224\text{ bytes}$** em `float64`; contém número mágico `"VFDT"`, versão, parâmetros de schema e contadores).
  - **Custo por Nó Serializado no `.bin` (`uint16_t idx` + `ht_node_t`):** Apenas **$22\text{ bytes}$ por nó** em `float32` ($2\text{ B}$ do índice + $20\text{ B}$ do nó estrutural; ou **$42\text{ bytes}$** em `float64`).
  - **Modo Somente-Inferência (`include_training_stats = false`):** Descarta os observadores pesados de atributos (`ht_leaf_stats_t`). O arquivo binário contém estritamente o cabeçalho e os nós estruturais em uso!

### Tabela de Tamanho do Arquivo `.bin` do Modelo para Inferência

$$\text{Tamanho do Arquivo (Bytes)} = 172\text{ B (Header)} + N \times 22\text{ B (onde } N \text{ é o número de nós da árvore)}$$

| Quantidade de Nós na Árvore ($N$) | Tipo de Complexidade do Modelo | Peso do Arquivo Transferido (`.bin`, `float32`) |
| :---: | :--- | :---: |
| **$1\text{ nó}$** | Raiz inicial (sem splits) | **$194\text{ bytes}$** ($< 0,2\text{ KB}$) |
| **$10\text{ nós}$** | Modelo compacto (poucas regras) | **$392\text{ bytes}$** ($< 0,4\text{ KB}$) |
| **$25\text{ nós}$** | Modelo pequeno | **$722\text{ bytes}$** ($< 0,8\text{ KB}$) |
| **$50\text{ nós}$** | Modelo intermediário típico | **$1.272\text{ bytes}$** ($\approx 1,2\text{ KB}$) |
| **$100\text{ nós}$** | Modelo detalhado de alta acurácia | **$2.372\text{ bytes}$** ($\approx 2,3\text{ KB}$) |
| **$250\text{ nós}$** | Modelo complexo | **$5.672\text{ bytes}$** ($\approx 5,5\text{ KB}$) |
| **$500\text{ nós}$** | Árvore de grande porte | **$11.172\text{ bytes}$** ($\approx 10,9\text{ KB}$) |

> [!TIP]
> Mesmo uma árvore profunda com **$100\text{ nós}$** pesa **menos de 2,4 KB de arquivo**, podendo ser enviada facilmente por conexões ultralentas (BLE, LoRaWAN, Zigbee, UART serial de 9600 bps) ou gravada em memórias EEPROM/Flash minúsculas. Além disso, para ensembles e `Fed-Forest`, a compactação densa via BFS (`ht_compact_tree_t`) consome apenas **`1.376 B` ($\approx 1{,}34\text{ KB}$)** em RAM para `HT_COMPACT_MAX_NODES=64` (ou **`2.656 B`** para `128`).
>
> Caso deseje transferir o modelo completo com suporte a **continuar o aprendizado em campo** (`include_training_stats = true`), somam-se $2\text{ B} + 1.360\text{ B} = 1.362\text{ bytes}$ por folha ativa salva (na configuração padrão $F=16, C=4$).

---

## 3. Consumo de Memória Volátil de Execução (RAM)

Com os limites padrão ($F = 16$ atributos, $C = 4$ classes, `4096` nós e `256` folhas ativas):
- **Observador por Atributo (`ht_feature_obs_t`):** Exatamente **$80\text{ B}$** em `float32` ($160\text{ B}$ em `float64`) tanto para Classificação quanto para Regressão.
- **Estado de Aprendizado por Folha Ativa (`ht_leaf_stats_t`):**
  $$16 \times 80\text{ B} + \text{pesos SGD} + \text{metadados NBA/fMSE} = \mathbf{1.360\text{ B por folha ativa}} \text{ (em float32)}$$
  *(Com o Dynamic Feature Stride Slab `ht_memory_init_stride` introduzido na `v0.4.0`, o espaço efetivo por folha reduz proporcionalmente ao `n_features` real do problema.)*
- **Nó Estrutural (`ht_node_t` — Internos e Folhas Inativas):**
  $$\text{union } (16\text{ B}: \text{class\_counts}[4] \text{ ou } \text{split/índices em folha}) + \text{índices/flags compactos } (4\text{ B}) = \mathbf{20\text{ B por nó}} \text{ (em float32)}$$
  *(A free-list intrusiva reutiliza bytes internos dos nós e folhas livres — **0 bytes extras** para pilhas `free_nodes`/`free_stats`.)*
- **Consumo Total de `sizeof(ht_tree_t)`:**
  - **Em `float` (32-bit):** **`496.024 B` ($\approx \mathbf{484,4\text{ KB}}$)** — menos da metade do orçamento de 1 MB, suportando $256$ folhas ativas e $4.096$ nós.
  - **Em `double` (64-bit):** **`959.112 B` ($\approx \mathbf{936,6\text{ KB}}$)** — validado em tempo de compilação via `_Static_assert(sizeof(ht_tree_t) <= 1024 * 1024)`.

---

## 4. Flags Recomendadas de Compilação no PC para Dispositivos Fracos

Para gerar binários ultracompactos no PC antes de enviar para o dispositivo, inclua os headers modulares (`-Icode/include -Icode/include/core`) e utilize o conjunto de flags abaixo (adicione `-Icode/include/ensemble -Icode/include/federated code/src/ensemble/*.c code/src/federated/*.c` caso utilize Aprendizado Federado ou árvores compactas):

### A) Dispositivos com Linux Embarcado (Raspberry Pi Zero, Orange Pi, OpenWrt, etc.)
```bash
# 1. Compilação do núcleo com otimização de tamanho (-Os) e descarte de código morto
gcc -std=c11 -Os -ffunction-sections -fdata-sections \
    -Icode/include -Icode/include/core code/src/core/*.c main.c -lm \
    -Wl,--gc-sections -s -o app_inferencia

# 2. Remoção garantida de qualquer tabela de símbolos de debug remanescente:
strip --strip-all app_inferencia
```
*O executável stripped final pesará apenas algumas dezenas de KB, pronto para ser enviado via SCP/SSH.*

### B) Microcontroladores Bare-Metal / RTOS (ARM Cortex-M, STM32, ESP32)
```bash
# 1. Compilação cruzada do núcleo com FPU em hardware (exemplo ARM Cortex-M4F):
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 \
    -std=c11 -Os -ffunction-sections -fdata-sections \
    -Icode/include -Icode/include/core code/src/core/*.c main.c -lm \
    -Wl,--gc-sections -o firmware.elf

# 2. Compilação cruzada ZERO-LIBM em ponto fixo Q16.16 para MCUs SEM FPU (Cortex-M0/M0+/M3, sem -lm!):
arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -mfloat-abi=soft \
    -DHT_USE_FIXED_POINT -std=c11 -Os -ffunction-sections -fdata-sections \
    -Icode/include -Icode/include/core code/src/core/*.c main.c \
    -Wl,--gc-sections -o firmware_q16.elf

# 3. Extração da imagem binária limpa (.bin) para gravação na Flash:
arm-none-eabi-objcopy -O binary firmware_q16.elf firmware_q16.bin
```
*O firmware final do núcleo C11 na Flash do microcontrolador ocupará **$\approx 28{,}6\text{ KB}$ completo** (ou **$< 10\text{ KB}$** somente inferência com `--gc-sections`), e o modo `-DHT_USE_FIXED_POINT` elimina 100% da biblioteca `libm` e da emulação `soft-fp` (`-25,6%` no binário completo Cortex-M3).*

### C) Redimensionando a RAM para Microcontroladores com Pouca Memória (< 128 KB de RAM)
Se o microcontrolador possuir apenas 32 KB, 64 KB ou 128 KB de memória RAM (como STM32F103, ESP8266 ou RP2040), não é necessário alocar os $484,4\text{ KB}$ da configuração padrão. Basta reduzir as macros de capacidade no momento da compilação:

```bash
# Configuração ultracompacta (~30,8 KB de SRAM no núcleo):
gcc -std=c11 -Os -DHT_MAX_NODES=256 -DHT_MAX_ACTIVE_LEAVES=32 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2 \
    -Icode/include -Icode/include/core code/src/core/*.c main.c -lm -Wl,--gc-sections -s -o app_compacto
```
