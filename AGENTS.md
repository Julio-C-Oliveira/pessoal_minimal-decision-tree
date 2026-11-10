# AGENTS.md — kansho (Embedded Hoeffding Tree in C11)

Este repositório desenvolve **kansho** (簡樹, "árvore simples"): uma **Hoeffding Tree (VFDT / HAT)** ultraleve em **C11 puro**, projetada para treino incremental (*online*) e inferência em tempo real em sistemas embarcados com **$\le 1\text{ MB}$ de RAM**.

## 1. Regras Invioláveis de Arquitetura e Código C11

- **Padrão e Dependências:** C11 estrito (`-std=c11 -Wall -Wextra -Werror -pedantic`), utilizando apenas headers da biblioteca padrão (`<stdint.h>`, `<stddef.h>`, `<stdbool.h>`, `<math.h>`, `<string.h>`).
- **Zero Alocação Dinâmica em Runtime:** É **proibido** usar `malloc`, `calloc`, `realloc` ou `free` nos caminhos de treino (`ht_learn_one`) e inferência (`ht_predict`). Toda a memória deve vir dos *Slab/Pool Allocators* estáticos gerenciados por `ht_memory`.
- **Indexação Livre de Ponteiros (`uint16_t`):**
  - Referências entre nós (`left_child`, `right_child`) e para estados de aprendizado (`stats_idx`) devem usar índices `uint16_t` (`HT_INVALID_IDX = 0xFFFF`), garantindo menor consumo de RAM e relocabilidade binária direta (serialização para Flash/disco).
- **Separação entre Topologia e Estado de Aprendizado:**
  - `ht_node_t` (**Leve**): Estrutura permanente do nó (filhos, condição de split binário, `class_counts` e `stats_idx`).
  - `ht_leaf_stats_t` (**Pesado**): Observadores de atributos (Gaussianas de Welford e tabelas categóricas) + contadores do *Naive Bayes Adaptive*. Deve ser **imediatamente devolvido ao pool** quando a folha sofrer split ou for desativada pelo gerenciamento de memória do VFDT.
- **Precisão Configurável (`ht_float_t`):**
  - Todo cálculo de ponto flutuante deve usar `ht_float_t` (`float` 32-bit por padrão; `double` 64-bit quando compilado com `-DHT_USE_DOUBLE` para testes de paridade com o River).
- **Sem Recursão Profunda:** O roteamento da raiz até a folha deve usar loops iterativos (`while`), nunca recursão na pilha de chamadas (stack limitada em microcontroladores).
- **Dimensionamento Estrito via Macros de Compilação (Zero Redundância em Runtime):**
  - Limites de capacidade (`HT_MAX_NODES`, `HT_MAX_ACTIVE_LEAVES`, `HT_MAX_FEATURES`, `HT_MAX_CLASSES`, `HT_MAX_CATEGORIES`) são definidos exclusivamente em tempo de compilação (`code/include/ht_config.h` ou flags `-D` no `Makefile`) para reduzir fisicamente `sizeof(ht_tree_t)` na SRAM.
  - É **proibido** duplicar esses limites como parâmetros dinâmicos em `ht_config_params_t` ou adicionar checagens extras em `code/src/` para restringir em runtime o que o alocador estático (`ht_memory`) já impõe nativamente por exaustão do pool (`free_nodes_top`, `free_stats_top`).
- **Trava de Minimalismo antes de tocar em `code/`:**
  - Tarefas de benchmark (`benchmarks/`) não devem alterar `code/include/` ou `code/src/`. Antes de propor qualquer novo campo em structs ou lógica extra no núcleo C11, verifique se o mecanismo nativo de macros + pools estáticos já resolve o problema; em caso de dúvida, pergunte ao usuário antes de modificar `code/`.

## 2. Regra de Commits Atômicos (Git)

- **Um Conceito Lógico por Commit:** Nunca misture mudanças de naturezas distintas no mesmo commit (ex: não misture notas teóricas em `docs/`, adição de referências em `references/`, configuração de agentes em `.agents/` e implementação em `src/`).
- **Estado Sempre Verde e Gate de Testes (Condicional a Código C11):**
  - **Obrigatoriedade de Testes:** O gate de compilação sem warnings (`-Wall -Wextra -Werror -pedantic`) e execução de 100% dos testes unitários e de integração (em `float` e `double`) é **obrigatório exclusivamente quando houver modificações em código C ou arquivos de build/teste** (`code/include/`, `code/src/`, `code/tests/`, `code/Makefile` ou `Makefile`).
  - **Modificações Exclusivas de Documentação/Metadados:** Quando o commit contiver apenas alterações documentais ou conceituais (ex: `README.md`, `docs/`, `wiki/`, `.agents/`, `VERSION` ou `references/`), **não** há necessidade de rodar a suíte de testes antes de commitar.
  - Caso ocorram alterações em código e algum teste falhe ou ocorra erro/warning de compilação, é **expressamente proibido commitar**. O agente deve obrigatoriamente diagnosticar o problema, consertar os erros e validar novamente antes de commitar.
- **Mensagens Padronizadas (Conventional Commits):**
  - `docs:` criação/atualização de documentação em `docs/` ou `README.md`
  - `feat(<escopo>):` nova funcionalidade em C (`memory`, `observer`, `criterion`, `tree`, `drift` em `code/`)
  - `fix(<escopo>):` correção de bug ou precisão numérica
  - `test(<escopo>):` adição ou ajuste de testes unitários em `code/tests/`
  - `bench:` scripts e executáveis de validação/benchmark em `benchmarks/`
  - `chore:` build (`Makefile`, `code/Makefile`), materiais de referência (`references/`) ou configuração de agentes (`AGENTS.md`, `.agents/`)
- **Versionamento Semântico e Releases (SemVer 2.0.0):**
  - O projeto adota estritamente Semantic Versioning (`MAJOR.MINOR.PATCH`).
  - A fonte única de verdade é mantida sincronizada em duas pontas: o arquivo `VERSION` (na raiz do projeto, para Make/scripts) e as macros `HT_VERSION_MAJOR`, `HT_VERSION_MINOR`, `HT_VERSION_PATCH`, `HT_VERSION_STRING` em `code/include/ht_config.h` (para o código C11), além das asserções de consistência em `code/tests/test_main.c`.
  - **Avaliação Obrigatória de Upgrade de Versão:** Após **cada commit realizado** (ou entrega de bloco lógico/etapa), o agente deve avaliar obrigatoriamente se as alterações acumuladas justificam ou demandam um upgrade de versão SemVer:
    - **`PATCH` (0.x.+1):** correções de bugs (`fix`), correções de precisão numérica, refatorações internas ou otimizações de performance sem alteração da API pública nem quebra de compatibilidade.
    - **`MINOR` (0.+1.0):** nova funcionalidade retrocompatível (`feat`), novos módulos/algoritmos (ex: novos observadores, novos tipos de split, novas estratégias de drift), expansão de structs ou utilitários da biblioteca.
    - **`MAJOR` (+1.0.0):** alterações incompatíveis na API pública (`BREAKING CHANGE`), mudanças drásticas de layout de memória/estruturas ou marcos de estabilização do projeto (ex: release `1.0.0`).
  - Quando um upgrade de versão for indicado, ele deve ser executado em um **commit atômico dedicado** (`chore(version): bump version to X.Y.Z`), sincronizando `VERSION`, `code/include/ht_config.h` e `code/tests/test_main.c`, com validação prévia de 100% dos testes.
  - Nunca crie tags Git sem antes validar 100% dos testes unitários e de integração em precisão simples (`float`) e dupla (`double`).
  - Releases no GitHub devem obrigatoriamente usar tags anotadas no padrão `vX.Y.Z` (ex: `git tag -a v0.1.0 -m "Release v0.1.0: ..."`).

## 3. Uso de Contexto, Referências e Subagentes

- **Materiais de Referência Locais (`references/`):**
  - `references/papers/`: Artigos originais (Domingos & Hulten 2000/2001).
  - `references/river_tree/`: Código de referência do `online-ml/river` (`river/tree` e `river/drift`).
  - `references/sklearn_tree/`: Código de referência do `scikit-learn` (`sklearn/tree` em Cython/C).
  - **Não modifique** os arquivos dentro de `references/` — eles são apenas fonte de consulta.
- **Delegação para Subagentes (`research`):**
  - Sempre que precisar explorar múltiplos arquivos de referência em `references/river_tree/`, `references/sklearn_tree/` ou ler PDFs em `references/papers/`, delegue a pesquisa ao subagente `research` para manter a janela de contexto principal limpa e focada na implementação.

## 4. Skills do Projeto (`.agents/skills/`) e Plano Persistente (`docs/06_implementation_plan.md`)

- **Plano de Implementação Persistente Multi-Chat:** O arquivo [`docs/06_implementation_plan.md`](docs/06_implementation_plan.md) é a fonte única de verdade para o plano de implementação entre diferentes chats e agentes. Consulte-o antes de iniciar qualquer implementação e mantenha seu checklist atualizado a cada entrega concluída.

Ative a skill correspondente antes de iniciar cada tipo de tarefa:
- **`study-reference`**: Para estudar os papers e mapear algoritmos do River / estruturas do Scikit-Learn em `docs/`.
- **`implementation-plan`**: Para criar, consultar ou atualizar o plano de implementação persistente em [`docs/06_implementation_plan.md`](docs/06_implementation_plan.md) entre diferentes chats e agentes.
- **`embedded-c-tree`**: Para implementar, revisar ou otimizar módulos em `code/include/`, `code/src/` e `code/tests/` respeitando o orçamento de 1 MB e o protocolo de testes em 3 níveis.
- **`river-parity-benchmark`**: Para validar acurácia *prequential*, número de splits e consumo de RAM em `benchmarks/` contra o River.
- **`sync-wiki`**: Para sincronizar e publicar automaticamente as alterações de `wiki/` no repositório oficial da GitHub Wiki (`pessoal_minimal-decision-tree.wiki.git`).
