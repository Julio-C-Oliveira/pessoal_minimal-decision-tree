---
name: sync-wiki
description: >-
  Use this skill whenever modifying, reviewing, or publishing changes to markdown
  files in wiki/ to ensure the official GitHub Wiki repository
  (pessoal_minimal-decision-tree.wiki.git) is kept in sync with the main repository.
---

# GitHub Wiki Synchronization Skill (`sync-wiki`)

No ecossistema do GitHub, a aba **Wiki** de um repositório é gerenciada como um **repositório Git independente** (`https://github.com/Julio-C-Oliveira/pessoal_minimal-decision-tree.wiki.git`) desacoplado da árvore de commits da branch `main`.

Para manter a consistência e integridade da documentação:
- A pasta versionada **[`wiki/`](../../../wiki/Home.md)** na raiz do projeto é a **fonte canônica de desenvolvimento e versionamento local**.
- Esta skill garante que, sempre que qualquer arquivo em `wiki/` for modificado, os arquivos sejam replicados, traduzidos e enviados para a GitHub Wiki oficial.

---

## 1. Regra de Ouro da Sincronização

> [!IMPORTANT]
> **Sempre que modificar qualquer arquivo em `wiki/`:**
> 1. Realize o commit e push das alterações no repositório principal (`origin/main`).
> 2. Execute a sincronização da GitHub Wiki oficial utilizando o script automatizado da skill para garantir paridade imediata.

---

## 2. Execução Automatizada (`sync_wiki.py`)

A skill disponibiliza um script em Python que automatiza todo o ciclo de clone, cópia, tradução de links e push:

```bash
# Executa a sincronização com a mensagem do último commit
python3 .agents/skills/sync-wiki/scripts/sync_wiki.py

# Ou especificando uma mensagem de commit customizada
python3 .agents/skills/sync-wiki/scripts/sync_wiki.py -m "docs(wiki): atualiza roadmap com novos itens"

# Para verificar as alterações antes de enviar
python3 .agents/skills/sync-wiki/scripts/sync_wiki.py --dry-run
```

### O que o Script Realiza Automaticamente:
1. **Clonagem Dinâmica:** Clona ou atualiza o repositório `pessoal_minimal-decision-tree.wiki.git` em diretório temporário (`/tmp/mdt-wiki`).
2. **Tradução Inteligente de Links:**
   - **Links para Arquivos Externos (`../`):** Transforma referências relativas do repositório local (ex: `../VERSION`, `../code/include/ht_tree.h`, `../docs/01_...`) em URLs canônicas completas (`https://github.com/Julio-C-Oliveira/pessoal_minimal-decision-tree/blob/main/...`), garantindo que funcionem na interface web da Wiki.
   - **Links Internos da Wiki:** Converte referências entre páginas `[Texto](Pagina.md)` para a sintaxe limpa nativa do GitHub Wiki `[Texto](Pagina)`.
3. **Verificação de Idempotência:** Inspeciona `git status --porcelain`. Se nenhuma alteração for detectada, não cria commits vazios.
4. **Commit & Push Atômico:** Adiciona as mudanças, comita com a mensagem especificada e envia via `git push origin master`.
5. **Limpeza:** Remove os arquivos temporários de forma segura após o push.

---

## 3. Protocolo de Fallback Manual (Se o Script não Estiver Disponível)

Caso precise sincronizar manualmente sem o script:

```bash
# 1. Clone o repositório da wiki em um diretório temporário
git clone https://github.com/Julio-C-Oliveira/pessoal_minimal-decision-tree.wiki.git /tmp/mdt-wiki

# 2. Copie os arquivos markdown de wiki/
cp wiki/*.md /tmp/mdt-wiki/

# 3. Acesse o clone temporário
cd /tmp/mdt-wiki

# 4. Ajuste links relativos externos para links absolutos do GitHub, se houver
# 5. Verifique status e envie
git status
git add *.md
git commit -m "docs(wiki): sincroniza páginas da wiki com o repositório principal"
git push origin master

# 6. Remova o diretório temporário
rm -rf /tmp/mdt-wiki
```

---

## 4. Convenção de Commits

Sempre utilize a convenção do repositório:
- `docs(wiki): ...` (ex: `docs(wiki): sincroniza roadmap de trabalhos futuros com benchmarks`).
