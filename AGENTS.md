# AGENTS.md — Minimal Decision Tree (Embedded Hoeffding Tree in C11)

Este repositório desenvolve uma **Hoeffding Tree (VFDT / HAT)** ultraleve em **C11 puro**, projetada para treino incremental (*online*) e inferência em tempo real em sistemas embarcados com **$\le 1\text{ MB}$ de RAM**.

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

## 2. Regra de Commits Atômicos (Git)

- **Um Conceito Lógico por Commit:** Nunca misture mudanças de naturezas distintas no mesmo commit (ex: não misture notas teóricas em `docs/`, adição de referências em `references/`, configuração de agentes em `.agents/` e implementação em `src/`).
- **Estado Sempre Verde:** Todo commit que toque em `include/`, `src/` ou `tests/` deve compilar sem warnings e passar em todos os testes unitários (tanto em `float` quanto em `double`).
- **Mensagens Padronizadas (Conventional Commits):**
  - `docs:` criação/atualização de documentação em `docs/` ou `README.md`
  - `feat(<escopo>):` nova funcionalidade em C (`memory`, `observer`, `criterion`, `tree`, `drift`)
  - `fix(<escopo>):` correção de bug ou precisão numérica
  - `test(<escopo>):` adição ou ajuste de testes unitários em `tests/`
  - `bench:` scripts e executáveis de validação/benchmark em `benchmarks/`
  - `chore:` build (`Makefile`), materiais de referência (`references/`) ou configuração de agentes (`AGENTS.md`, `.agents/`)

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
- **`embedded-c-tree`**: Para implementar, revisar ou otimizar módulos em `include/`, `src/` e `tests/` respeitando o orçamento de 1 MB e o protocolo de testes em 3 níveis.
- **`river-parity-benchmark`**: Para validar acurácia *prequential*, número de splits e consumo de RAM em `benchmarks/` contra o River.
