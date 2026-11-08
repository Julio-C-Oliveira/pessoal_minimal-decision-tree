#!/usr/bin/env python3
"""
Script de sincronização automática entre a pasta wiki/ do repositório principal
e o repositório oficial da GitHub Wiki (pessoal_minimal-decision-tree.wiki.git).

Funcionalidades:
1. Clona ou atualiza o repositório da GitHub Wiki em um diretório temporário.
2. Copia os arquivos markdown de wiki/ para a raiz do repositório da Wiki.
3. Converte links relativos para fora da Wiki (ex: ../VERSION, ../code/, ../docs/)
   em links canônicos do GitHub para garantir navegação funcional na interface web.
4. Converte referências entre páginas da wiki para a sintaxe nativa do GitHub Wiki
   (ex: [Nome](Pagina.md) -> [Nome](Pagina)).
5. Identifica se houve alterações reais (git status/diff).
6. Executa commit atômico e push para a branch master do GitHub Wiki.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/Julio-C-Oliveira/pessoal_minimal-decision-tree"
WIKI_REPO_URL = f"{REPO_URL}.wiki.git"
DEFAULT_TMP_DIR = "/tmp/mdt-wiki"

def run_cmd(cmd, cwd=None, check=True):
    res = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=True)
    if check and res.returncode != 0:
        print(f"Erro ao executar: {cmd}", file=sys.stderr)
        print(f"STDOUT:\n{res.stdout}", file=sys.stderr)
        print(f"STDERR:\n{res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)
    return res

def transform_markdown_links(content: str) -> str:
    """
    Ajusta links markdown para o ecossistema da GitHub Wiki.
    1. Transforma links locais entre páginas da wiki [Texto](Pagina.md) -> [Texto](Pagina)
    2. Transforma links para arquivos externos da raiz do repositório (../caminho)
       em links absolutos no GitHub (https://github.com/.../blob/main/caminho ou tree/main/caminho)
    """
    # 1. Ajuste de links para arquivos externos do repositório
    def replace_external_link(match):
        text = match.group(1)
        rel_path = match.group(2)
        # Normaliza caminho removendo prefixo ../
        clean_path = rel_path.lstrip("./")
        # Se for diretório ou arquivos específicos de include/src, usa tree/main
        if clean_path.startswith("code/include") and not clean_path.endswith((".h", ".c")):
            url = f"{REPO_URL}/tree/main/{clean_path}"
        elif clean_path.startswith("code/src") and not clean_path.endswith((".h", ".c")):
            url = f"{REPO_URL}/tree/main/{clean_path}"
        elif clean_path.endswith((".h", ".c", ".md", ".py", ".sh", ".txt", ".csv")) or clean_path == "VERSION":
            url = f"{REPO_URL}/blob/main/{clean_path}"
        else:
            url = f"{REPO_URL}/tree/main/{clean_path}"
        return f"[{text}]({url})"

    # Regex para links relativos saindo da pasta wiki (ex: [Texto](../VERSION) ou [Texto](../code/...))
    content = re.sub(r'\[([^\]]+)\]\(\.\./([^\)]+)\)', replace_external_link, content)

    # 2. Ajuste de links entre páginas internas da própria Wiki (ex: [Texto](Pagina.md) -> [Texto](Pagina))
    # Exceção: não alterar URLs completas com http:// ou https://
    def replace_internal_wiki_link(match):
        text = match.group(1)
        target = match.group(2)
        if target.startswith("http://") or target.startswith("https://") or target.startswith("#"):
            return match.group(0)
        # Se apontar para um arquivo .md interno
        if target.endswith(".md"):
            clean_target = target[:-3]
            return f"[{text}]({clean_target})"
        return match.group(0)

    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_internal_wiki_link, content)

    return content

def main():
    parser = argparse.ArgumentParser(description="Sincronizador automático da GitHub Wiki.")
    parser.add_argument("--message", "-m", help="Mensagem de commit para a wiki.")
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR, help="Diretório temporário para clonagem.")
    parser.add_argument("--dry-run", action="store_true", help="Apenas mostra alterações sem commitar nem fazer push.")
    parser.add_argument("--wiki-src", default="wiki", help="Diretório fonte das páginas da wiki no repositório.")
    args = parser.parse_args()

    wiki_src = os.path.abspath(args.wiki_src)
    if not os.path.isdir(wiki_src):
        print(f"Diretório fonte '{wiki_src}' não encontrado!", file=sys.stderr)
        sys.exit(1)

    # Determina mensagem de commit se não especificada
    commit_msg = args.message
    if not commit_msg:
        res = run_cmd("git log -1 --pretty=%B", check=False)
        if res.returncode == 0 and res.stdout.strip():
            last_commit_line = res.stdout.strip().split("\n")[0]
            if not last_commit_line.startswith("docs(wiki):"):
                commit_msg = f"docs(wiki): {last_commit_line}"
            else:
                commit_msg = last_commit_line
        else:
            commit_msg = "docs(wiki): sincroniza páginas da Wiki com o repositório principal"

    print(f"[*] Repositório Wiki: {WIKI_REPO_URL}")
    print(f"[*] Diretório de sincronização: {args.tmp_dir}")
    print(f"[*] Mensagem de commit: {commit_msg}")

    # Clona ou atualiza o repositório da Wiki
    if os.path.exists(args.tmp_dir):
        print(f"[*] Atualizando clone existente em {args.tmp_dir}...")
        run_cmd("git fetch origin && git reset --hard origin/master", cwd=args.tmp_dir)
    else:
        print(f"[*] Clonando GitHub Wiki em {args.tmp_dir}...")
        run_cmd(f"git clone {WIKI_REPO_URL} {args.tmp_dir}")

    # Copia e processa todos os arquivos .md de wiki_src
    files_processed = 0
    for filename in os.listdir(wiki_src):
        if filename.endswith(".md"):
            src_file = os.path.join(wiki_src, filename)
            dst_file = os.path.join(args.tmp_dir, filename)

            with open(src_file, "r", encoding="utf-8") as f:
                content = f.read()

            transformed_content = transform_markdown_links(content)

            with open(dst_file, "w", encoding="utf-8") as f:
                f.write(transformed_content)

            files_processed += 1

    print(f"[*] {files_processed} arquivos markdown processados e copiados.")

    # Verifica status do git no repositório da wiki
    res = run_cmd("git status --porcelain", cwd=args.tmp_dir)
    status_output = res.stdout.strip()

    if not status_output:
        print("[+] A GitHub Wiki já está perfeitamente sincronizada. Nenhuma alteração detectada.")
        if not args.dry_run:
            shutil.rmtree(args.tmp_dir, ignore_errors=True)
        return

    print(f"[*] Alterações detectadas na GitHub Wiki:\n{status_output}")

    if args.dry_run:
        print("[*] Modo --dry-run: diff das alterações:")
        diff_res = run_cmd("git diff", cwd=args.tmp_dir)
        print(diff_res.stdout)
        return

    # Executa commit e push
    print("[*] Adicionando arquivos e realizando commit...")
    run_cmd("git add -A", cwd=args.tmp_dir)
    run_cmd(f'git commit -m "{commit_msg}"', cwd=args.tmp_dir)

    print("[*] Enviando alterações para a GitHub Wiki (git push origin master)...")
    run_cmd("git push origin master", cwd=args.tmp_dir)

    print("[+] GitHub Wiki sincronizada e publicada com sucesso!")

    # Limpeza
    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    print("[*] Diretório temporário limpo.")

if __name__ == "__main__":
    main()
