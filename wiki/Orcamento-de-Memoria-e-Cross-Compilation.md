# Orçamento de Memória e Cross-Compilation (Flash / RAM)

Para sistemas embarcados e dispositivos fracos (microcontroladores ARM Cortex-M, ESP32, RISC-V, ou SoCs Linux compactos como MIPS OpenWrt), a alocação de memória se divide em duas grandezas fundamentais:
1. **Armazenamento Permanente / Código Compilado (Flash / ROM / Disco):** Onde fica gravado o binário executável (`.text` + `.rodata`) e os arquivos de modelo serializados transferidos do PC.
2. **Memória de Trabalho Volátil (RAM / `.bss`):** Onde reside o estado operacional da árvore (`ht_tree_t`) durante o treino ou inferência.

---

## 1. Peso do Código Compilado no Dispositivo (Flash / ROM / Binário)

Ao compilar a biblioteca no PC para gerar os arquivos binários que serão gravados no dispositivo, o tamanho real em armazenamento depende das funcionalidades utilizadas e das flags do compilador:

| Modo de Uso / Funcionalidades Linkadas | Seção `.text` (Código C11 em `-Os`) | Seção `.rodata` / `.data` | Consumo em Flash (Cortex-M / RISC-V) | Executável ELF Linux *Stripped* |
| :--- | :---: | :---: | :---: | :---: |
| **Apenas Inferência** (`ht_tree_predict` / `ht_tree_predict_proba` / `ht_tree_predict_reg`) com `--gc-sections` | **$\approx 8,2\text{ KB}$** | $< 0,7\text{ KB}$ | **$\approx 6\text{ a } 8\text{ KB}$** | $\approx 19\text{ KB}$ |
| **Treino Incremental Padrão (VFDT) + Inferência** (sem detecção de drift ou regressão contínua) | **$\approx 19,4\text{ KB}$** | $< 0,7\text{ KB}$ | **$\approx 16\text{ a } 20\text{ KB}$** | $\approx 27\text{ KB}$ |
| **Biblioteca Completa em `-Os`** (Classificação + Regressão VFDT-R + Drift HAT/ADWIN2 + Serializador) | **$\approx 28,5\text{ KB}$** | $< 0,7\text{ KB}$ | **$\approx 22\text{ a } 28\text{ KB}$** | $\approx 39\text{ KB}$ |
| **Biblioteca Completa em `-O2`** (otimizada para velocidade e vetorização de loops) | **$\approx 42,5\text{ KB}$** | $< 0,8\text{ KB}$ | $\approx 35\text{ a } 42\text{ KB}$ | $\approx 44\text{ KB}$ |

> [!NOTE]
> - **Zero overhead de BSS na biblioteca:** A `libht` não possui buffers globais ou variáveis estáticas internas ocultas. Todo o footprint de execução reside exclusivamente na instância `ht_tree_t` alocada pelo usuário.
> - **Microcontroladores de 32 bits (ARM Thumb-2 / RV32IMC):** Em processadores como STM32, nRF52, ESP32 e RP2040/RP2350, o conjunto de instruções comprimidas de 16 bits atinge densidade de código ainda maior, ficando tipicamente em **$\le 8\text{ KB}$ de Flash** para inferência e **$\le 25\text{ KB}$ de Flash** para a biblioteca inteira com treino.

---

## 2. Peso do Arquivo do Modelo Serializado (Transferência PC $\rightarrow$ Dispositivo)

Caso o fluxo de trabalho seja treinar ou pré-treinar o modelo no PC e transferir apenas o arquivo do modelo para o dispositivo rodar inferência em campo (`ht_tree_save` / `ht_tree_load`):

- **Estrutura Binária:**
  - **Cabeçalho Fixo (`ht_serial_header_t`):** Fixos **$170\text{ bytes}$** (contém número mágico `"VFDT"`, versão, metadados de configuração e contadores).
  - **Custo por Nó da Árvore (`ht_node_t`):** Apenas **$28\text{ bytes}$ por nó** em `float32` ($48\text{ bytes}$ em `float64`).
  - **Modo Somente-Inferência (`include_training_stats = false`):** Descarta os observadores pesados de atributos (`ht_leaf_stats_t`). O arquivo binário contém estritamente o cabeçalho e os nós estruturais em uso!

### Tabela de Tamanho do Arquivo `.bin` do Modelo para Inferência

$$\text{Tamanho do Arquivo (Bytes)} = 170\text{ B (Header)} + N \times 28\text{ B (onde } N \text{ é o número de nós da árvore)}$$

| Quantidade de Nós na Árvore ($N$) | Tipo de Complexidade do Modelo | Peso do Arquivo Transferido (`.bin`) |
| :---: | :--- | :---: |
| **$1\text{ nó}$** | Raiz inicial (sem splits) | **$198\text{ bytes}$** ($< 0,2\text{ KB}$) |
| **$10\text{ nós}$** | Modelo compacto (poucas regras) | **$450\text{ bytes}$** ($< 0,5\text{ KB}$) |
| **$25\text{ nós}$** | Modelo pequeno | **$870\text{ bytes}$** ($< 0,9\text{ KB}$) |
| **$50\text{ nós}$** | Modelo intermediário típico | **$1.570\text{ bytes}$** ($\approx 1,5\text{ KB}$) |
| **$100\text{ nós}$** | Modelo detalhado de alta acurácia | **$2.970\text{ bytes}$** ($\approx 2,9\text{ KB}$) |
| **$250\text{ nós}$** | Modelo complexo | **$7.170\text{ bytes}$** ($\approx 7,0\text{ KB}$) |
| **$500\text{ nós}$** | Árvore de grande porte | **$14.170\text{ bytes}$** ($\approx 13,8\text{ KB}$) |

> [!TIP]
> Mesmo uma árvore bem profunda com **$100\text{ nós}$** pesa **menos de 3 KB de arquivo**, podendo ser enviada facilmente por conexões ultralentas (BLE, LoRaWAN, Zigbee, UART serial de 9600 bps) ou gravada em memórias EEPROM/Flash minúsculas sem nenhum desperdício de armazenamento.
>
> Caso deseje transferir o modelo com suporte a **continuar o aprendizado em campo** (`include_training_stats = true`), somam-se $1.360\text{ bytes}$ por folha ativa salva (para a configuração padrão $F=16, C=4$).

---

## 3. Consumo de Memória Volátil de Execução (RAM)

Com os limites padrão ($F = 16$ atributos, $C = 4$ classes, `4096` nós e `256` folhas ativas):
- **Observador por Atributo (`ht_feature_obs_t`):** Exatamente **$80\text{ B}$** em `float32` ($160\text{ B}$ em `float64`) tanto para Classificação quanto para Regressão.
- **Estado de Aprendizado por Folha Ativa (`ht_leaf_stats_t`):**
  $$16 \times 80\text{ B} + \text{pesos SGD} + \text{metadados NBA/fMSE} = \mathbf{1.360\text{ B por folha ativa}} \text{ (em float32)}$$
- **Nó Estrutural (`ht_node_t` — Internos e Folhas Inativas):**
  $$\text{union alvo } (16\text{ B}) + \text{split } (4\text{ B}) + \text{índices } (3 \times 2\text{ B}) + \text{flags } (2\text{ B}) = \mathbf{28\text{ B por nó}} \text{ (em float32)}$$
- **Consumo Total de `sizeof(ht_tree_t)`:**
  - **Em `float` (32-bit):** $\approx \mathbf{537\text{ KB}}$ (metade do orçamento de 1 MB, suportando $256$ folhas ativas e $4.096$ nós).
  - **Em `double` (64-bit):** $\approx \mathbf{1.000\text{ KB}}$ (validado em tempo de compilação via `_Static_assert(sizeof(ht_tree_t) <= 1024 * 1024)`).

---

## 4. Flags Recomendadas de Compilação no PC para Dispositivos Fracos

Para gerar binários ultracompactos no PC antes de enviar para o dispositivo, utilize o conjunto de flags abaixo:

### A) Dispositivos com Linux Embarcado (Raspberry Pi Zero, Orange Pi, OpenWrt, etc.)
```bash
# 1. Compilação com otimização de tamanho (-Os) e seções separadas para descarte de código morto
gcc -std=c11 -Os -ffunction-sections -fdata-sections \
    -Icode/include code/src/*.c main.c -lm \
    -Wl,--gc-sections -s -o app_inferencia

# 2. Remoção garantida de qualquer tabela de símbolos de debug remanescente:
strip --strip-all app_inferencia
```
*O executável stripped final pesará entre **19 KB e 39 KB**, pronto para ser enviado via SCP/SSH ou pen drive.*

### B) Microcontroladores Bare-Metal / RTOS (ARM Cortex-M, STM32, ESP32)
```bash
# 1. Compilação cruzada (exemplo ARM Cortex-M4F):
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 \
    -std=c11 -Os -ffunction-sections -fdata-sections \
    -Icode/include code/src/*.c main.c -lm \
    -Wl,--gc-sections -o firmware.elf

# 2. Extração da imagem binária limpa (.bin) para gravação na Flash:
arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
```
*O firmware final na Flash do microcontrolador ocupará apenas **~8 KB a 28 KB**.*

### C) Redimensionando a RAM para Microcontroladores com Pouca Memória (< 128 KB de RAM)
Se o microcontrolador possuir apenas 32 KB, 64 KB ou 128 KB de memória RAM (como STM32F103, ESP8266 ou RP2040), não é necessário alocar os $537\text{ KB}$ da configuração padrão. Basta reduzir as macros de capacidade no momento da compilação:

```bash
# Configuração ultracompacta: consome menos de 35 KB de RAM no total!
gcc -std=c11 -Os -DHT_MAX_NODES=256 -DHT_MAX_ACTIVE_LEAVES=32 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2 \
    -Icode/include code/src/*.c main.c -lm -Wl,--gc-sections -s -o app_compacto
```
