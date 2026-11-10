---
name: version-upgrade
description: >-
  Use this skill whenever evaluating version upgrade eligibility after commits,
  presenting justifications for user approval, and atomically updating VERSION,
  code/include/ht_config.h, code/tests/test_main.c, and the Wiki release notes.
---

# Semantic Versioning Upgrade Skill (`version-upgrade`)

No projeto **kansho**, o versionamento segue estritamente o padrão [Semantic Versioning 2.0.0](https://semver.org/lang/pt-BR/) (`MAJOR.MINOR.PATCH`). Como uma biblioteca em C11 para sistemas embarcados sem dependência de gerenciadores como o `pip` ou `cargo`, a versão canônica reside em **três pontos de contato obrigatórios** mantidos em perfeita sincronia:

1. **[`VERSION`](../../../VERSION):** Arquivo de texto simples na raiz do repositório, lido por regras de build (`Makefile`, `code/Makefile`), scripts de empacotamento e suítes de benchmark em Python.
2. **[`code/include/ht_config.h`](../../../code/include/ht_config.h):** Símbolos do pré-processador (`HT_VERSION_MAJOR`, `HT_VERSION_MINOR`, `HT_VERSION_PATCH`, `HT_VERSION_STRING`, `HT_VERSION_CHECK`), garantindo checagens de compatibilidade em tempo de compilação com zero overhead de RAM/Flash.
3. **[`code/tests/test_main.c`](../../../code/tests/test_main.c):** Asserções unitárias que validam as macros, a função de runtime `ht_version()` e a macro `HT_VERSION_CHECK`.

---

## 1. Regra de Ouro: Proibição de Upgrade Autônomo

> [!IMPORTANT]
> **O agente nunca deve realizar upgrades de versão de forma autônoma sem confirmação prévia do usuário.**
> Toda alteração de versão exige obrigatoriamente:
> 1. Execução do modo de diagnóstico (`bump_version.py analyze`).
> 2. Apresentação formal ao usuário da versão atual, versão proposta e justificativas técnicas detalhadas baseadas no SemVer.
> 3. Confirmação explícita do usuário autorizando o bump (e indicando se deseja ou não a criação de tag Git anotada).

---

## 2. Critérios de Avaliação SemVer 2.0.0 (Escopo Estrito em `code/`)

> [!IMPORTANT]
> **A versão SemVer da biblioteca reflete única e exclusivamente a solução em `code/` (`code/include/`, `code/src/`, `code/tests/`, `code/Makefile`).**
> Commits que alteram apenas `.agents/` (skills, regras), `benchmarks/`, `docs/` ou `wiki/` são classificados como ferramentas ou metadados de suporte e **nunca desencadeiam upgrade de versão**.

Ao analisar os commits acumulados na linha principal desde o último bump de `VERSION`:

| Tipo | Critério (Somente em `code/`) | Exemplos no Kansho |
| :--- | :--- | :--- |
| **`PATCH`** (`0.x.+1`) | Correções de bugs retrocompatíveis (`fix:`), correções de precisão numérica ou otimizações internas no código em `code/` sem alteração da API pública. | `fix(memory): restringir assert`, correções de precisão em Welford/ADWIN2. |
| **`MINOR`** (`0.+1.0`) | Novas funcionalidades públicas retrocompatíveis (`feat:`) em `code/`, novos observadores, novos splits ou expansão de structs da biblioteca. | Adição de `HT_OBS_HISTOGRAM`, pipeline `libht_prep`, novos kernels de regressão. |
| **`MAJOR`** (`+1.0.0`) | Alterações incompatíveis na API pública (`BREAKING CHANGE`) em `code/include/`, mudanças drásticas de layout de memória/estruturas ou marcos (`1.0.0`). | Alteração na assinatura de `ht_learn_one` / `ht_predict`, alteração de formato de serialização binária (`HT_SERIAL_VERSION`). |
| **`NONE`** (Sem bump) | Commits sem alterações em `code/` (ex: `docs:`, `wiki/`, `bench:`, `.agents/`, `chore:`) ou apenas testes/build sem impacto na biblioteca. | Nova skill em `.agents/skills/`, documentação em `docs/` ou `wiki/`, scripts de tuning. |

---

## 3. Fluxo Operacional da Skill

### Fase 1: Diagnóstico Pós-Commit
Após realizar qualquer commit (ou entrega de bloco lógico), o agente executa a análise:

```bash
python3 .agents/skills/version-upgrade/scripts/bump_version.py analyze
```

O comando analisa o histórico Git entre o último commit que tocou em `VERSION` e `HEAD`, imprimindo um parecer com tabela de commits classificados e a recomendação SemVer.

### Fase 2: Apresentação e Consulta ao Usuário
Se a recomendação for diferente de `NONE`, o agente apresenta o parecer ao usuário e pergunta:
> *"Você aprova o upgrade de versão de `X.Y.Z` para `A.B.C` (TIPO)? Deseja também gerar a tag Git anotada `vA.B.C`?"*

### Fase 3: Aplicação Atômica
Após a aprovação do usuário, o agente executa o script no modo `apply`:

```bash
# Apenas bump com commit atômico e validação de testes
python3 .agents/skills/version-upgrade/scripts/bump_version.py apply <patch|minor|major|X.Y.Z> --commit

# Bump com commit atômico e criação de tag Git anotada
python3 .agents/skills/version-upgrade/scripts/bump_version.py apply <patch|minor|major|X.Y.Z> --commit --tag -m "Release vX.Y.Z: ..."

# Simulação (dry-run) para inspecionar os diffs sem alterar arquivos
python3 .agents/skills/version-upgrade/scripts/bump_version.py apply <patch|minor|major|X.Y.Z> --dry-run
```

O modo `apply` realiza as seguintes etapas de forma determinística:
1. Valida que a árvore de trabalho está limpa (`git status --porcelain`).
2. Atualiza sincronizadamente `VERSION`, `code/include/ht_config.h` e `code/tests/test_main.c`.
3. Executa o **Gate de Testes Obrigatório**:
   - `make clean && make test` (precisão simples `float`)
   - `make clean && make test CFLAGS_EXTRA="-DHT_USE_DOUBLE"` (precisão dupla `double`)
   - *Se algum teste falhar, os arquivos são revertidos automaticamente e o processo é abortado.*
4. Se a flag `--commit` for informada, gera o commit atômico padronizado: `chore(version): bump version to X.Y.Z`.
5. Se a flag `--tag` for informada, cria a tag anotada correspondente (`vX.Y.Z`).

---

## 4. Protocolo Pós-Upgrade: Atualização do Changelog na Wiki

Após o commit atômico de upgrade de versão:
1. Atualize a seção `## 4.1 Histórico de Releases (Changelog)` em [`wiki/Versionamento-Releases-e-Roadmap.md`](../../../wiki/Versionamento-Releases-e-Roadmap.md), documentando a nova versão lançada com seus respectivos módulos e correções.
2. Realize o commit isolado de documentação:
   ```bash
   git add wiki/Versionamento-Releases-e-Roadmap.md
   git commit -m "docs(wiki): documentar release vX.Y.Z no changelog"
   ```
3. Publique as alterações na Wiki oficial usando a skill [`sync-wiki`](../sync-wiki/SKILL.md):
   ```bash
   python3 .agents/skills/sync-wiki/scripts/sync_wiki.py
   ```
