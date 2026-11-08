# Versionamento, Releases e Roadmap

Esta página descreve a arquitetura de versionamento do projeto em C11 (`SemVer 2.0.0`), como inspecionar a versão da biblioteca em tempo de compilação e execução, o processo de publicação de releases no GitHub e o *roadmap* de possíveis extensões futuras.

---

## 1. Como Funciona o Versionamento em C vs Python

No ecossistema Python moderno, o versionamento centralizado do pacote é definido em um único arquivo de configuração: o [`pyproject.toml`](https://peps.python.org/pep-0621/) (seção `[project] version = "0.1.0"`), sendo comum expor a variável `__version__` no pacote. Ferramentas como `pip`, `build`, `flit` ou `poetry` leem esse arquivo para empacotar e publicar no PyPI.

Em **C11 e bibliotecas de baixo nível / embarcadas** (padrão adotado por SQLite, zlib, libcurl, cJSON, Git e Linux Kernel), não existe um gerenciador de dependências universal mandatório. O versionamento é estruturado em **três camadas complementares**:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Metadata File (Raiz)        │ VERSION ("0.1.0")                      │
│ (Equivalente ao pyproject.toml) │ Lido pelo Makefile, scripts e CI/CD   │
├─────────────────────────────────────────────────────────────────────────┤
│ 2. Single Source of Truth em C │ HT_VERSION_MAJOR / MINOR / PATCH       │
│ (code/include/ht_config.h)     │ HT_VERSION_STRING ("0.1.0")            │
│                                │ HT_VERSION_CHECK(maj, min, patch)      │
│                                │ const char *ht_version(void);          │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. Git Tags e GitHub Releases  │ v0.1.0 (Tags anotadas no Git)          │
│                                │ GitHub Releases / Tarballs             │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Arquivo de Metadados ([`VERSION`](../VERSION)):** Localizado na raiz do repositório, armazena a versão canônica (ex: `0.1.0`). É a referência única para automações, empacotadores e regras de build.
2. **Macros e Símbolos em C ([`ht_config.h`](../code/include/ht_config.h) e [`ht_tree.h`](../code/include/ht_tree.h)):** Fornece constantes numéricas para o pré-processador e uma função estática de runtime (`ht_version()`).
3. **Tags do Git (`vX.Y.Z`):** Conectam o histórico de commits do repositório com as versões públicas disponibilizadas no GitHub.

---

## 2. Como Consultar a Versão no Código

A biblioteca permite verificar a versão tanto durante a compilação do firmware quanto em tempo de execução:

### Em Tempo de Compilação (Zero custo de CPU/RAM)
```c
#include "ht_tree.h"

#if !HT_VERSION_CHECK(0, 1, 0)
#error "Este firmware requer Minimal Decision Tree >= 0.1.0!"
#endif
```

### Em Tempo de Execução
```c
#include <stdio.h>
#include "ht_tree.h"

void print_system_info(void) {
    /* Retorna ponteiro para string literal em Flash (zero malloc, O(1)) */
    printf("Minimal Decision Tree versao: %s\n", ht_version());
}
```

### Via Linha de Comando / Build
```bash
make version
# Saida: Minimal Decision Tree (libht) version 0.1.0
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

1. Atualize a versão no arquivo [`VERSION`](../VERSION) e nas macros `HT_VERSION_*` em [`code/include/ht_config.h`](../code/include/ht_config.h).
2. Execute a suíte de testes garantindo 100% de sucesso em precisão simples (`float`) e dupla (`double`):
   ```bash
   make clean && make test && make clean && make test CFLAGS_EXTRA="-DHT_USE_DOUBLE"
   ```
3. Realize o commit das alterações de versão e crie uma **tag anotada**:
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0: Embedded Hoeffding Tree in C11 (VFDT, HAT, VFDT-R, HAT-R, ADWIN2)"
   ```
4. Envie o commit e a tag para o repositório remoto no GitHub:
   ```bash
   git push origin main
   git push origin v0.1.0
   ```
5. No GitHub, acesse a aba **Releases** $\rightarrow$ **Draft a new release** e selecione a tag `v0.1.0` recém-enviada para gerar as notas de lançamento e os pacotes `.tar.gz` e `.zip` correspondentes.

---

## 5. Possíveis Trabalhos Futuros e Extensões (Roadmap)

Embora todas as fases planejadas (Fases 0 a 4, incluindo Classificação, Adaptação a *Concept Drift* com `ADWIN2`/`HAT` e Regressão Contínua `FIMT-DD`/`HAT-R`) estejam 100% concluídas com zero heap dinâmico em runtime e conformidade estrita com o teto de 1 MB de RAM, as análises teóricas e as constatações experimentais do **Main Benchmark com 8 Datasets** apontam direções promissoras de evolução estruturadas em 6 eixos temáticos:

### 5.1 Plataformas Embarcadas, Hardware Real e Concorrência Multi-Core

1. **Deploy e Cross-Compilation em Hardware Real (Bare-Metal / RTOS):**
   - Criação de scripts e targets no Makefile para compilação cruzada com toolchains reais (`arm-none-eabi-gcc` para ARM Cortex-M4F / Cortex-M7, `riscv64-unknown-elf-gcc` para RISC-V e ESP-IDF para ESP32).
   - Validação direta em bancada com microcontroladores operando sob FreeRTOS ou Zephyr OS, medindo ciclos exatos de clock (`DWT->CYCCNT`) e consumo de corrente/energia por amostra.

2. **Quantização e Aritmética de Ponto Fixo (Fixed-Point / INT8 / Q16.16):**
   - Suporte a microcontroladores de baixíssimo custo desprovidos de FPU de ponto flutuante em hardware (ex: ARM Cortex-M0/M0+ ou microcontroladores de 8/16 bits).
   - Substituição de `ht_float_t` por inteiros escalados em ponto fixo Q16.16 (16 bits de parte inteira e 16 bits de fração) com tabelas de consulta (*LUT*) ou algoritmo CORDIC para aproximações de $\ln(x)$, $\exp(x)$ e $\sqrt{x}$.

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

8. **Normalização Online Integrada nas Folhas (Streaming StandardScaler Zero-Malloc):**
   - No benchmark do dataset *Bikes*, observou-se que atributos com ordens de grandeza muito distintas (ex: temperatura em $[0, 1]$, hora em $[0, 23]$ e umidade em $[0, 100]$) causaram instabilidade e oscilações no gradiente do regressor linear SGD das folhas.
   - O projeto prevê um padronizador online leve baseado no algoritmo de Welford integrado nas folhas:
     $$z_{j} = \frac{x_{j} - \bar{x}_{j}}{\sqrt{\sigma_{j}^2 + \varepsilon_s}}$$
     onde a média $\bar{x}_j$ e a variância $\sigma_j^2$ são atualizadas incrementalmente com $O(1)$ de RAM e tempo, fornecendo ao regressor SGD um espaço de gradiente esférico e bem condicionado.

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

16. **Discretizadores Não-Paramétricos Estáticos (T-Digest / Histograma Ben-Haim & Tom-Tov):**
    - Para variáveis com distribuições altamente multimodais ou caudas pesadas que violam a suposição de Gaussianidade unimodal de Welford, propõe-se observadores de histograma dinâmico em tamanho estático pré-dimensionado (ex: 32 a 64 centróides em buffer contíguo), permitindo estimar quantis exatos para splits com baixo footprint de memória.

17. **Bindings em Python (CFFI / ctypes / Cython):**
    - Disponibilização de um wrapper Python de alto desempenho expondo a API clássica do Scikit-Learn e River (`fit_one`, `predict_one`, `predict_proba_one`), permitindo que cientistas de dados usem o núcleo ultrarrápido em C11 diretamente em pipelines de experimentação ($200\times$ a $1.000\times$ mais rápido que implementações em Python puro).

18. **Ensembles Online Embarcados (Streaming Random Forests / Online Bagging):**
    - Implementação de algoritmos de ensemble em streaming (*Online Bagging* de Oza & Russell com distribuição de Poisson, ou *Leveraging Bagging*), dividindo o pool estático de nós e detectores entre $M$ árvores leves (ex: 5 a 10 árvores de 32 nós cada) operando colaborativamente dentro do orçamento máximo de $1\text{ MB}$ de RAM.

19. **Micro-Pipelines Embarcados de Pré-Processamento Zero-Malloc (`libht_prep`):**
    - Criação de uma biblioteca companheira em C11 estático contendo componentes modulares de pré-processamento online determinístico:
      - `ht_scaler_t`: Normalização online Z-score e Min-Max.
      - `ht_filter_hampel_t`: Filtro mediano de Hampel em janelas estáticas para eliminação de espículas (*outliers*) em tempo real de sensores industriais.
      - `ht_encoder_freq_t`: Codificador de frequência incremental para variáveis categóricas de alta cardinalidade.
