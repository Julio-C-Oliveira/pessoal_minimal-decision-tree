# Referência Completa de Parâmetros e Macros (`ht_config.h` / `ht_tree.h`)

Esta página documenta todos os hiperparâmetros configuráveis em tempo de execução (`ht_config_params_t` e `ht_adwin_config_t`), bem como todas as macros de compilação (`ht_config.h`) que controlam o dimensionamento físico da árvore na memória RAM.

---

## 1. Parâmetros de Configuração em Runtime (`ht_config_params_t`)

Todos os campos abaixo pertencem à struct [`ht_config_params_t`](../code/include/ht_tree.h) e são inicializados automaticamente pelas funções `ht_default_params()`, `ht_default_hat_params()`, `ht_default_reg_params()` ou `ht_default_hat_reg_params()`:

| Parâmetro (`ht_config_params_t`) | Tipo | Padrão (VFDT / HAT) | Padrão (VFDT-R / HAT-R) | Descrição e Efeito no Modelo |
| :--- | :--- | :---: | :---: | :--- |
| `task_type` | `ht_task_type_t` | `HT_TASK_CLASSIFICATION` | `HT_TASK_REGRESSION` | Define se a árvore opera em modo de Classificação Multiclasse ou Regressão Contínua. |
| `n_features` | `uint8_t` | *(Definido pelo usuário)* | *(Definido pelo usuário)* | Número de atributos de entrada no fluxo ($1 \dots \text{HT\_MAX\_FEATURES}$). |
| `n_classes` | `uint8_t` | *(Definido pelo usuário)* | `1` | Número de classes alvo na classificação ($2 \dots \text{HT\_MAX\_CLASSES}$). Ignorado na regressão. |
| `feature_types[i]` | `ht_feature_type_t` | `HT_FEAT_NUMERIC` | `HT_FEAT_NUMERIC` | Tipo de cada coluna $i$: `HT_FEAT_NUMERIC` (contínuo) ou `HT_FEAT_CATEGORICAL` (inteiro $0 \dots \text{HT\_MAX\_CATEGORIES}-1$). |
| `criterion` | `ht_criterion_type_t` | `HT_CRIT_INFO_GAIN` | `HT_CRIT_VARIANCE_REDUCTION` | Critério de impureza: `HT_CRIT_INFO_GAIN` (Entropia base 2), `HT_CRIT_GINI` (Impureza de Gini) ou `HT_CRIT_VARIANCE_REDUCTION` (Redução de Variância com `ddof=1`). |
| `leaf_prediction` | `ht_leaf_pred_t` | `HT_LEAF_NBA` | `HT_LEAF_ADAPTIVE` | **Classificação:** `HT_LEAF_MC` (Classe Majoritária), `HT_LEAF_NB` (Naive Bayes), `HT_LEAF_NBA` (Adaptativo entre MC e NB).<br/>**Regressão:** `HT_LEAF_MEAN` (Média), `HT_LEAF_MODEL` (Linear SGD), `HT_LEAF_ADAPTIVE` (Alterna entre Média e SGD via fMSE). |
| `grace_period` | `uint16_t` | `200` | `200` | Peso acumulado de amostras ($n_{\min}$) que uma folha ativa aguarda entre avaliações consecutivas de candidatos de split. Valores menores aceleram o crescimento da árvore ao custo de mais CPU. |
| `max_depth` | `uint16_t` | `64` | `64` | Profundidade máxima permitida para a árvore (raiz = `0`). Ao atingir `max_depth`, a folha é automaticamente desativada. |
| `delta` ($\delta$) | `ht_float_t` | `1e-7` | `1e-7` | Nível de significância do *Hoeffding Bound* ($1 - \delta$ é a confiança estatística do split). Deve estar em $(0, 1)$. Valores menores exigem mais evidência antes de dividir. |
| `tau` ($\tau$) | `ht_float_t` | `0.05` | `0.05` | Limiar de desempate (*tie-breaking*). Se dois atributos têm mérito muito parecido e o limite de Hoeffding cai abaixo de $\tau$ ($\epsilon < \tau$), o split é forçado no melhor candidato atual. |
| `min_branch_fraction` | `ht_float_t` | `0.01` (`1%`) | `0.01` (`1%`) | Fração mínima do peso total da folha que cada ramo filho (esquerdo e direito) deve receber para que um candidato de split seja considerado válido ($0 < f \le 0.5$). |
| `max_share_to_split` | `ht_float_t` | `0.99` (`99%`) | `0.99` | *(Classificação)* Se a classe majoritária na folha já representa mais que essa fração do peso total, a folha é considerada suficientemente pura e não tenta fazer split. |
| `nb_threshold` | `ht_float_t` | `0.0` | `0.0` | *(Classificação c/ `HT_LEAF_NB`)* Peso mínimo acumulado na folha antes de habilitar o uso do Naive Bayes (evita instabilidade com pouquíssimas amostras). |
| `merit_preprune` | `bool` | `true` | `true` | Quando `true`, inclui o candidato de "não dividir" (*null split*) na competição do Hoeffding Bound, evitando splits com ganho irrelevante. |
| `enable_hat` | `bool` | `false` (VFDT) / `true` (HAT) | `false` (VFDT-R) / `true` (HAT-R) | Ativa o monitoramento de *Concept Drift* via detectores ADWIN2 e cultivo de subárvores alternativas em cada nó interno. |
| `bootstrap_sampling` | `bool` | `false` (VFDT) / `true` (HAT) | `false` (VFDT-R) / `true` (HAT-R) | Quando `enable_hat = true`, multiplica o peso de cada amostra recebida por $k \sim \text{Poisson}(1)$, aumentando a diversidade estatística durante transições de drift. |
| `drift_window_threshold` | `uint16_t` | `300` | `300` | *(HAT / HAT-R)* Número mínimo de amostras observadas tanto no ramo principal quanto na subárvore alternativa antes de avaliar a substituição (`SWITCH`) ou poda (`PRUNE`). |
| `switch_significance` ($\alpha$) | `ht_float_t` | `0.05` | `0.05` | *(HAT / HAT-R)* Nível de significância estatística para substituir um ramo principal por sua subárvore alternativa (ou podá-la). |
| `adwin_delta` | `ht_float_t` | `0.002` | `0.002` | *(HAT / HAT-R)* Parâmetro de confiança $\delta_{\text{ADWIN}}$ dos detectores de drift alocados para os nós da árvore. |
| `learning_rate` ($\eta_0$) | `ht_float_t` | `0.01` | `0.01` | *(Regressão `MODEL` / `ADAPTIVE`)* Taxa de aprendizado base do modelo linear treinado via *Stochastic Gradient Descent* (SGD) nas folhas. |
| `l2_regularization` ($\lambda$) | `ht_float_t` | `1e-4` | `1e-4` | *(Regressão `MODEL` / `ADAPTIVE`)* Penalidade de regularização L2 e fator de decaimento da taxa de aprendizado ($\eta_t = \eta_0 / (1 + \lambda n)$). |
| `model_selector_decay` ($\alpha$) | `ht_float_t` | `0.95` | `0.95` | *(Regressão `ADAPTIVE`)* Fator de esquecimento exponencial do erro quadrático médio ponderado (*Faded MSE*: $\text{fmse}_t = \alpha \cdot \text{fmse}_{t-1} + e_t^2$). |
| `min_samples_split` | `ht_float_t` | `5.0` | `5.0` | *(Regressão)* Peso mínimo de amostras exigido em cada ramo filho (esquerdo e direito) para validar um candidato de split de regressão. |

---

## 2. Parâmetros do Detector ADWIN (`ht_adwin_config_t`)

O detector [`ht_adwin_t`](../code/include/ht_drift.h) também pode ser usado de forma independente (*standalone*) para monitorar qualquer fluxo de dados numéricos ou taxas de erro:

| Parâmetro (`ht_adwin_config_t`) | Tipo | Valor Padrão | Significado |
| :--- | :--- | :---: | :--- |
| `delta` | `ht_float_t` | `0.002` | Nível de significância para o teste de corte de janela $\epsilon_{\text{cut}}$. Valores maiores tornam o detector mais sensível a pequenas mudanças. |
| `clock` | `uint16_t` | `32` | Intervalo de passos entre verificações de corte da janela (reduz custo computacional amortizado para $O(1)$). |
| `min_window_length` | `uint16_t` | `5` | Tamanho mínimo de cada subjanela ($W_0$ e $W_1$) exigido ao testar uma hipótese de bipartição. |
| `grace_period` | `uint16_t` | `10` | Número mínimo de elementos totais na janela antes de habilitar os testes de detecção de drift. |
| `max_buckets` ($M$) | `uint8_t` | `5` | Número máximo de buckets de capacidade $2^k$ por nível antes de comprimir os 2 mais antigos em um bucket de capacidade $2^{k+1}$ no nível seguinte. |

---

## 3. Macros de Compilação para Dimensionamento de RAM (`ht_config.h`)

Como a biblioteca **não utiliza alocação dinâmica (`malloc`/`free`)**, todo o espaço de memória da árvore (`ht_tree_t`) é definido estaticamente em tempo de compilação através das macros em [`code/include/ht_config.h`](../code/include/ht_config.h).

Você pode sobrescrever qualquer uma dessas macros diretamente no comando do compilador usando a flag `-D<MACRO>=<VALOR>` (ou via `make CFLAGS_EXTRA="-D..."`) sem precisar editar os arquivos `.h` do código-fonte.

### 3.1 Tabela Completa de Macros e Seus Efeitos

| Macro de Compilação | Valor Padrão | O que Dimensiona na Arquitetura | Efeito Prático ao Reduzir ou Aumentar |
| :--- | :---: | :--- | :--- |
| **`HT_USE_DOUBLE`** | *(Indefinido)* | Tipo base `ht_float_t`: quando omitido usa `float` ($4\text{ B}$); quando definido (`-DHT_USE_DOUBLE`) usa `double` ($8\text{ B}$). | **Omitido (`float`):** corta o uso de RAM pela metade e usa FPU nativa de 32 bits em MCUs.<br/>**Definido (`double`):** dobra o consumo de RAM, usado para validação de paridade numérica exata no PC contra o Python River. |
| **`HT_MAX_FEATURES`** | `16` | Quantidade máxima de colunas/atributos por amostra (`observers[HT_MAX_FEATURES]` e `reg_weights[HT_MAX_FEATURES]` em cada folha ativa). | **Reduzir:** economiza **$84\text{ B}$ por atributo removido em cada folha ativa** ($80\text{ B}$ do observador + $4\text{ B}$ do peso SGD). Se o seu sensor tem apenas 4 variáveis, defina `-DHT_MAX_FEATURES=4` para economizar ~75% de RAM nas folhas!<br/>**Limite:** `params.n_features` nunca pode ultrapassar `HT_MAX_FEATURES`. |
| **`HT_MAX_CLASSES`** | `4` | Quantidade máxima de classes na classificação (`class_counts[HT_MAX_CLASSES]` em todos os nós e número de Gaussianas de Welford por atributo numérico). | **Reduzir (ex: `=2` para classificação binária):** reduz o tamanho de cada nó estrutural de $28\text{ B}$ para $24\text{ B}$ e reduz o observador Gaussiano de $80\text{ B}$ para $40\text{ B}$.<br/>**Aumentar (ex: `=10` para MNIST):** permite problemas com mais classes, aumentando proporcionalmente o tamanho do observador ($20\text{ B} \times C$) e do nó ($4\text{ B} \times C$). |
| **`HT_MAX_CATEGORIES`** | `5` | Cardinalidade máxima de atributos categóricos discretos (valores inteiros aceitos de $0$ até $\text{HT\_MAX\_CATEGORIES}-1$). | **Reduzir:** diminui a matriz de frequências `counts[HT_MAX_CLASSES][HT_MAX_CATEGORIES]` e `cat_stats[HT_MAX_CATEGORIES]`.<br/>**Aumentar:** permite variáveis nominais com mais categorias distintas. Valores recebidos $\ge \text{HT\_MAX\_CATEGORIES}$ ou $< 0$ são tratados como ausentes (*missing*). |
| **`HT_MAX_NODES`** | `4096` | Capacidade total do pool estrutural `node_pool[HT_MAX_NODES]` (soma de nós internos de decisão + folhas ativas + folhas inativas). | **Efeito no Modelo:** define o tamanho máximo que a árvore pode atingir. Cada nó consome apenas **$24\text{ a }28\text{ B}$** (mais $16\text{ B}$ de metadados de drift em `node_drift`).<br/>**Se encher:** a árvore para de realizar novos splits, mas **continua aprendendo estatísticas nas folhas atuais e fazendo predições normalmente** sem travar. |
| **`HT_MAX_ACTIVE_LEAVES`** | `256` | Capacidade do `stats_pool[HT_MAX_ACTIVE_LEAVES]` (máximo de folhas pesadas `ht_leaf_stats_t` aprendendo estatísticas de split simultaneamente). | **É o principal controle de RAM!** Cada folha ativa custa **$1.360\text{ B}$** na configuração padrão.<br/>**Se encher:** aciona automaticamente a **Desativação por Promessa (*VFDT Memory Management*)** — a folha menos promissora é convertida em folha inativa (devolvendo seus $1.360\text{ B}$ ao pool, mas continuando apta a predizer) para dar lugar a novas folhas mais promissoras.<br/>**Para Somente-Inferência:** se o dispositivo apenas carrega um modelo pronto (`ht_tree_load`) e nunca chama `learn_one`, você pode compilar com **`-DHT_MAX_ACTIVE_LEAVES=1`**, reduzindo a RAM das folhas em **99,6%**! |
| **`HT_GAUSSIAN_SPLITS`** | `10` | Número de pontos de corte candidatos testados uniformemente entre `[min_val, max_val]` em cada atributo numérico de classificação. | **Não altera a RAM estática** (usado apenas no loop de avaliação de split).<br/>**Reduzir (ex: `=5`):** acelera o cálculo de tentativa de split na CPU em 2x com perda mínima de precisão no ponto de corte.<br/>**Aumentar (ex: `=20`):** testa mais limiares por atributo ao custo de mais ciclos de CPU a cada `grace_period`. |
| **`HT_REG_NUM_SLOTS`** | `5` | Quantidade de centróides/slots ordenados `(valor, count, mean, m2)` mantidos por atributo numérico na regressão (`ht_reg_numeric_obs_t`). | Cada slot ocupa $16\text{ B}$ em `float32` ($5 \times 16\text{ B} = 80\text{ B}$, empatando exatamente com o observador de classificação na `union`).<br/>**Aumentar (ex: `=10`):** aumenta a resolução de quantização de limiares na regressão contínua, aumentando o tamanho de `ht_feature_obs_t`. |
| **`HT_MAX_ADWIN_DETECTORS`** | `128` | Capacidade de detectores de mudança de conceito no `ht_adwin_pool_t` (usado apenas quando `enable_hat = true`). | Cada detector ADWIN ocupa **$764\text{ B}$** em `float32`. Se você **não usa HAT/Drift** (`enable_hat = false`), nem precisa instanciar `ht_adwin_pool_t` no seu código (`tree.adwin_pool = NULL`), economizando 100% dessa memória! Se usar HAT em MCU pequeno, reduza para `16` ou `32`. |
| **`HT_ADWIN_MAX_BUCKETS`** | `5` | Máximo de buckets ($M$) de capacidade $2^k$ por nível antes de comprimir os 2 mais antigos no nível $k+1$. | Controla o compromisso entre resolução temporal e memória dentro do detector ADWIN ($M=5$ é o padrão clássico de Bifet & Gavaldà). Reduzir para `3` ou `4` economiza RAM dentro de cada detector ADWIN. |
| **`HT_ADWIN_LEVELS`** | `14` | Quantidade de níveis logarítmicos do histograma exponencial de cada detector ADWIN. | Com `14` níveis e $M=5$, cada detector cobre janelas históricas de até $5 \times (2^{14}-1) = 81.915$ amostras sem perda. Reduzir para `10` ($5.115$ amostras) economiza $192\text{ B}$ por detector ADWIN. |

---

### 3.2 Como as Macros Determinam o Tamanho em Bytes (Fórmulas de Alocação)

Em precisão simples (`float` de $4\text{ bytes}$), o compilador dimensiona as estruturas automaticamente segundo as relações:

1. **Tamanho do Observador de 1 Atributo (`sizeof(ht_feature_obs_t)`):**
   $$\text{Obs} = \max\Big(20 \times C,\; 4 \times C \times K,\; 16 \times S_{\text{reg}},\; 12 \times K\Big)\text{ bytes}$$
   *(Onde $C = \text{HT\_MAX\_CLASSES}$, $K = \text{HT\_MAX\_CATEGORIES}$, $S_{\text{reg}} = \text{HT\_REG\_NUM\_SLOTS}$. No padrão: $20 \times 4 = \mathbf{80\text{ B}}$).*

2. **Tamanho de 1 Folha Ativa (`sizeof(ht_leaf_stats_t)`):**
   $$\text{Folha Ativa} = 16 + (4 \times F) + (F \times \text{Obs})\text{ bytes}$$
   *(Onde $F = \text{HT\_MAX\_FEATURES}$. No padrão com $F=16$: $16 + 64 + 1280 = \mathbf{1.360\text{ B}}$).*

3. **Tamanho de 1 Nó Estrutural (`sizeof(ht_node_t)`):**
   $$\text{Nó} = \max(4 \times C,\; 12) + 12\text{ bytes}$$
   *(No padrão com $C=4$: $16 + 12 = \mathbf{28\text{ B}}$. Para $C=2$: $12 + 12 = \mathbf{24\text{ B}}$).*

4. **RAM Total da Árvore (`sizeof(ht_tree_t)`):**
   $$\text{RAM Árvore} \approx \text{HT\_MAX\_NODES} \times (\text{Nó} + 18\text{ B}) + \text{HT\_MAX\_ACTIVE\_LEAVES} \times (\text{Folha Ativa} + 2\text{ B})$$

---

### 3.3 Perfis Prontos de Compilação por Categoria de Dispositivo (`gcc -D...`)

Abaixo estão combinações testadas de macros para diferentes orçamentos de hardware:

| Perfil de Hardware Alvo | Flags de Macros (`CFLAGS`) | Tamanho do Nó (`ht_node_t`) | Tamanho da Folha Ativa (`ht_leaf_stats_t`) | Consumo Total de RAM (`sizeof(ht_tree_t)`) |
| :--- | :--- | :---: | :---: | :---: |
| **1. Somente Inferência em Campo**<br/>*(Modelo treinado no PC e carregado via `ht_tree_load`)* | `-DHT_MAX_NODES=256 -DHT_MAX_ACTIVE_LEAVES=1 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2` | $24\text{ B}$ | $688\text{ B}$ | **$\approx 11,3\text{ KB}$** |
| **2. MCU Ultra-Fraco (32 a 64 KB RAM)**<br/>*(ex: STM32F103, Cortex-M0+, Arduino Zero)* | `-DHT_MAX_NODES=128 -DHT_MAX_ACTIVE_LEAVES=16 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2 -DHT_MAX_CATEGORIES=4 -DHT_REG_NUM_SLOTS=4` | $24\text{ B}$ | $560\text{ B}$ | **$\approx 14,2\text{ KB}$** |
| **3. MCU Compacto (64 a 128 KB RAM)**<br/>*(ex: STM32F401, nRF52832, ESP8266)* | `-DHT_MAX_NODES=256 -DHT_MAX_ACTIVE_LEAVES=32 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2` | $24\text{ B}$ | $688\text{ B}$ | **$\approx 32,2\text{ KB}$** |
| **4. MCU Intermediário (256 a 512 KB RAM)**<br/>*(ex: ESP32, RP2040, RP2350, STM32F4/H7)* | `-DHT_MAX_NODES=1024 -DHT_MAX_ACTIVE_LEAVES=64 -DHT_MAX_FEATURES=12 -DHT_MAX_CLASSES=4` | $28\text{ B}$ | $1.024\text{ B}$ | **$\approx 110,3\text{ KB}$** |
| **5. Configuração Padrão ($\le 1\text{ MB}$ RAM)**<br/>*(Sem flags extras — padrão do repositório)* | *(Padrão: `NODES=4096`, `LEAVES=256`, `FEATURES=16`, `CLASSES=4`)* | $28\text{ B}$ | $1.360\text{ B}$ | **$\approx 524,7\text{ KB}$** |
| **6. Multiclasse Pesado (10 Classes)**<br/>*(ex: Reconhecimento de dígitos MNIST)* | `-DHT_MAX_NODES=2048 -DHT_MAX_ACTIVE_LEAVES=128 -DHT_MAX_FEATURES=16 -DHT_MAX_CLASSES=10` | $52\text{ B}$ | $3.280\text{ B}$ | **$\approx 550,4\text{ KB}$** |

**Exemplo de compilação usando o Makefile do projeto com um perfil customizado:**
```bash
make clean && make lib CFLAGS_EXTRA="-Os -DHT_MAX_NODES=256 -DHT_MAX_ACTIVE_LEAVES=32 -DHT_MAX_FEATURES=8 -DHT_MAX_CLASSES=2"
```
