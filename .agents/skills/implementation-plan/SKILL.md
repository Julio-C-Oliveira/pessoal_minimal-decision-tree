---
name: implementation-plan
description: >-
  Use this skill whenever creating, reviewing, consulting, or updating the
  persistent cross-chat implementation plan in docs/06_implementation_plan.md,
  or before starting/completing any implementation milestone across phases.
---

# Persistent Implementation Plan Skill (`implementation-plan`)

Como o desenvolvimento deste projeto pode ocorrer ao longo de **múltiplos chats independentes e diferentes agentes/subagentes**, o plano técnico de implementação **não** deve ficar restrito a artefatos temporários de uma única conversa.

A fonte única de verdade (*Single Source of Truth*) para o plano de implementação, contratos de API em C11, layouts de memória e status de progresso entre sessões é o arquivo versionado no repositório:

- **[`docs/06_implementation_plan.md`](../../../docs/06_implementation_plan.md)**

---

## 1. Protocolo para Qualquer Chat ou Agente Ao Iniciar uma Tarefa

1. **Leitura Obrigatória de Contexto:**
   - Leia [`docs/06_implementation_plan.md`](../../../docs/06_implementation_plan.md) para identificar exatamente:
     - Qual etapa/módulo está `[x]` (concluído), `[-]` (em andamento) ou `[ ]` (pendente).
     - Os contratos exatos de structs, assinaturas de funções públicas, constantes e limites de memória ($\le 1\text{ MB}$) acordados para cada header em `include/`.
     - A matriz de testes de **Nível 1 (Função/Unidade)**, **Nível 2 (Integração)** e **Nível 3 (Fluxo Total / E2E)** exigida para a etapa atual.
2. **Consulta Rápida à Fundamentação (`docs/01` a `docs/05`):**
   - Caso precise revisar uma fórmula matemática ou detalhe de paridade durante a execução do plano, consulte os estudos já consolidados em:
     - [`docs/01_hoeffding_tree_vfdt.md`](../../../docs/01_hoeffding_tree_vfdt.md)
     - [`docs/02_attribute_observers.md`](../../../docs/02_attribute_observers.md)
     - [`docs/03_concept_drift_hat.md`](../../../docs/03_concept_drift_hat.md)
     - [`docs/04_river_mapping.md`](../../../docs/04_river_mapping.md)
     - [`docs/05_sklearn_architecture.md`](../../../docs/05_sklearn_architecture.md)

---

## 2. Estrutura Obrigatória do `docs/06_implementation_plan.md`

Sempre que criar ou atualizar [`docs/06_implementation_plan.md`](../../../docs/06_implementation_plan.md), mantenha as seguintes seções sincronizadas com o código real:

1. **Quadro de Status das Entregas (Checklist Executável Multi-Chat):**
   - Dividido em entregas atômicas (**Entrega 1.1 a 1.5** na Fase 1, **Entrega 2.1** na Fase 2, **Entrega 3.1 a 3.3** na Fase 3), indicando os arquivos exatos (`include/`, `src/`, `tests/`), o status atual (`[ ]`, `[-]`, `[x]`) e o hash do commit quando concluído.
2. **Orçamento Consolidado de Memória ($\le 1\text{ MB}$) e `_Static_assert`:**
   - Tamanhos exatos em bytes (`sizeof`) em modo `float` (32-bit) e `double` (64-bit) para `ht_node_t`, `ht_leaf_stats_t`, `ht_split_candidate_t`, `ht_adwin_state_t` e `ht_tree_t`.
3. **Especificação Técnica por Módulo (`include/` e `src/`):**
   - Definições completas das structs e assinaturas C11 de:
     - `ht_config.h` e `Makefile`
     - `ht_memory.h` / `ht_memory.c`
     - `ht_observer.h` / `ht_observer.c`
     - `ht_criterion.h` / `ht_criterion.c`
     - `ht_tree.h` / `ht_tree.c`
     - `ht_drift.h` / `ht_drift.c`
4. **Plano de Verificação em 3 Níveis (`tests/` e `benchmarks/`):**
   - Casos de teste obrigatórios de Nível 1 (Micro), Nível 2 (Meso) e Nível 3 (Macro E2E em `tests/test_e2e_flow.c`), além dos comandos de validação dupla (`float` e `double`).

---

## 3. Regra de Sincronização Pós-Entrega

Ao concluir e testar qualquer entrega prevista no plano:
1. Faça os commits atômicos de código e testes (`feat(<escopo>): ...`, `test(<escopo>): ...`).
2. Atualize o status da entrega (`[x]`) em [`docs/06_implementation_plan.md`](../../../docs/06_implementation_plan.md) e [`README.md`](../../../README.md) em um commit `docs:` dedicado para que o próximo chat ou agente saiba exatamente de onde continuar.
