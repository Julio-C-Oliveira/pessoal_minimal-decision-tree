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

Embora todas as fases planejadas (Fases 0 a 4, incluindo Classificação, Adaptação a Concept Drift com ADWIN2/HAT e Regressão Contínua FIMT-DD/HAT-R) estejam 100% concluídas com zero heap dinâmico em runtime e conformidade estrita com o teto de 1 MB de RAM, o projeto pode ser expandido nas seguintes frentes:

1. **Deploy e Cross-Compilation em Hardware Real (Bare-Metal / RTOS):**
   - Targets de compilação cruzada para arquiteturas embarcadas reais (`arm-none-eabi-gcc` para ARM Cortex-M4F / Cortex-M7, RISC-V e ESP-IDF para ESP32).
   - Validação direta em microcontroladores rodando FreeRTOS ou Zephyr OS, medindo ciclos de clock e consumo de energia por amostra.

2. **Quantização e Aritmética de Ponto Fixo (Fixed-Point / INT8 / Q16.16):**
   - Suporte a microcontroladores de baixíssimo custo desprovidos de FPU de ponto flutuante em hardware (ex: ARM Cortex-M0/M0+).
   - Substituição configurável de `ht_float_t` por inteiros escalados com aproximação de ponto fixo (CORDIC / look-up tables) para logaritmos e funções exponenciais.

3. **Bindings em Python (CFFI / ctypes / Cython):**
   - Pacote Python com interface compatível com Scikit-Learn e River (`learn_one`, `predict_one`, `predict_proba_one`), permitindo usar a biblioteca C11 diretamente em pipelines de Machine Learning em streaming com velocidade nativa ($110\times$ a $400\times$ mais rápido que Python puro).

4. **Ensembles Online Embarcados (Streaming Random Forests / Online Bagging):**
   - Implementação de algoritmos de ensemble em streaming (*Online Bagging*, *Leveraging Bagging* ou *Streaming Random Patches*).
   - Particionamento do pool estático de nós e detectores ADWIN entre múltiplas árvores leves ($M$ árvores) dentro do mesmo buffer estático de $\le 1\text{ MB}$.

5. **Validação Prática em ESP32 com Máquina de Estados Completa:**
   - Testar a biblioteca embarcada diretamente em uma placa **ESP32** estruturada com estados de operação prontos:
     - **Modo de Captura de Dados:** Aquisição e pré-processamento de leituras de sensores em tempo real.
     - **Modo de Treinamento:** Aprendizado incremental *on-device* (`ht_tree_learn_one` / `ht_tree_learn_one_reg`) e persistência do modelo em Flash/NVS (`ht_tree_save`).
     - **Modo de Inferência:** Predição em tempo real de ultrabaixa latência (`ht_tree_predict` / `ht_tree_predict_reg`) com possibilidade de carregar o modelo salvo (`ht_tree_load`).
