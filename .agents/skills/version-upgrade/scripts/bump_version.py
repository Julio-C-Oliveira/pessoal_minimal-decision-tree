#!/usr/bin/env python3
"""
Script de automação para avaliação e upgrade atômico de versão SemVer 2.0.0 (MAJOR.MINOR.PATCH)
do projeto kansho (libht).

Funcionalidades:
1. --analyze: Avalia os commits acumulados desde o último bump de VERSION e emite parecer
   técnico (NONE, PATCH, MINOR, MAJOR) formatado para apresentação e aprovação do usuário.
2. --apply <patch|minor|major|X.Y.Z>: Atualiza atomicamente a tríade de arquivos:
   - VERSION
   - code/include/ht_config.h
   - code/tests/test_main.c
3. Gate de testes obrigatório (make test em float e double com CFLAGS_EXTRA="-DHT_USE_DOUBLE").
4. Commit atômico padronizado (chore(version): bump version to X.Y.Z).
5. Suporte opcional a tag Git anotada (vX.Y.Z).
"""

import argparse
import difflib
import os
import re
import subprocess
import sys

def run_cmd(cmd, cwd=None, check=True):
    res = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=True)
    if check and res.returncode != 0:
        print(f"Erro ao executar comando: {cmd}", file=sys.stderr)
        if res.stdout:
            print(f"STDOUT:\n{res.stdout}", file=sys.stderr)
        if res.stderr:
            print(f"STDERR:\n{res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)
    return res

def get_repo_root():
    res = run_cmd("git rev-parse --show-toplevel", check=False)
    if res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip()
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

def parse_semver(version_str):
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str.strip())
    if not m:
        raise ValueError(f"Versão inválida '{version_str}'. Deve seguir o formato MAJOR.MINOR.PATCH (ex: 0.2.1)")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def format_semver(maj, min_, patch):
    return f"{maj}.{min_}.{patch}"

def read_version_files(repo_root):
    version_file = os.path.join(repo_root, "VERSION")
    config_file = os.path.join(repo_root, "code/include/ht_config.h")
    test_file = os.path.join(repo_root, "code/tests/test_main.c")

    if not os.path.isfile(version_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {version_file}")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {config_file}")
    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {test_file}")

    with open(version_file, "r", encoding="utf-8") as f:
        version_content = f.read()
    with open(config_file, "r", encoding="utf-8") as f:
        config_content = f.read()
    with open(test_file, "r", encoding="utf-8") as f:
        test_content = f.read()

    # Parse VERSION
    v_raw = version_content.strip()
    v_maj, v_min, v_patch = parse_semver(v_raw)

    # Parse ht_config.h
    m_maj = re.search(r"#define\s+HT_VERSION_MAJOR\s+(\d+)U", config_content)
    m_min = re.search(r"#define\s+HT_VERSION_MINOR\s+(\d+)U", config_content)
    m_patch = re.search(r"#define\s+HT_VERSION_PATCH\s+(\d+)U", config_content)
    m_str = re.search(r'#define\s+HT_VERSION_STRING\s+"([^"]+)"', config_content)

    if not (m_maj and m_min and m_patch and m_str):
        raise ValueError(f"Não foi possível extrair macros HT_VERSION_* de {config_file}")

    c_maj, c_min, c_patch = int(m_maj.group(1)), int(m_min.group(1)), int(m_patch.group(1))
    c_str = m_str.group(1)

    # Parse test_main.c
    t_maj = re.search(r"HT_ASSERT_EQ\(HT_VERSION_MAJOR,\s*(\d+)U\);", test_content)
    t_min = re.search(r"HT_ASSERT_EQ\(HT_VERSION_MINOR,\s*(\d+)U\);", test_content)
    t_patch = re.search(r"HT_ASSERT_EQ\(HT_VERSION_PATCH,\s*(\d+)U\);", test_content)

    t_maj_val = int(t_maj.group(1)) if t_maj else None
    t_min_val = int(t_min.group(1)) if t_min else None
    t_patch_val = int(t_patch.group(1)) if t_patch else None

    # Consistency check
    is_consistent = (
        (v_maj == c_maj == t_maj_val) and
        (v_min == c_min == t_min_val) and
        (v_patch == c_patch == t_patch_val) and
        (v_raw == c_str)
    )

    return {
        "version_file": version_file,
        "config_file": config_file,
        "test_file": test_file,
        "version_content": version_content,
        "config_content": config_content,
        "test_content": test_content,
        "current_tuple": (v_maj, v_min, v_patch),
        "current_str": v_raw,
        "is_consistent": is_consistent,
        "details": {
            "VERSION": v_raw,
            "ht_config.h": f"{c_maj}.{c_min}.{c_patch} (str={c_str})",
            "test_main.c": f"{t_maj_val}.{t_min_val}.{t_patch_val}" if t_maj_val is not None else "N/A"
        }
    }

def analyze_commits(repo_root):
    # 1. Encontra o commit base onde VERSION foi alterado pela última vez
    res_base = run_cmd("git log -n 1 --format=%H -- VERSION", cwd=repo_root, check=False)
    base_commit = res_base.stdout.strip()

    if not base_commit:
        # Fallback para a última tag ou commit inicial
        res_tag = run_cmd("git describe --tags --abbrev=0", cwd=repo_root, check=False)
        if res_tag.returncode == 0 and res_tag.stdout.strip():
            base_commit = res_tag.stdout.strip()
        else:
            res_root = run_cmd("git rev-list --max-parents=0 HEAD", cwd=repo_root, check=False)
            base_commit = res_root.stdout.strip()

    # Informações do commit base
    res_base_info = run_cmd(f"git log -1 --format='%h %s (%cd)' --date=short {base_commit}", cwd=repo_root, check=False)
    base_info = res_base_info.stdout.strip()

    # 2. Lista commits de base_commit até HEAD
    res_log = run_cmd(f"git log {base_commit}..HEAD --oneline", cwd=repo_root, check=False)
    log_lines = [l.strip() for l in res_log.stdout.strip().split("\n") if l.strip()]

    # 3. Classifica os commits
    classified_commits = []
    has_breaking = False
    has_feat = False
    has_fix = False

    for line in log_lines:
        parts = line.split(" ", 1)
        commit_hash = parts[0]
        subject = parts[1] if len(parts) > 1 else ""

        # Checa commit completo por BREAKING CHANGE
        res_body = run_cmd(f"git log -1 --format=%B {commit_hash}", cwd=repo_root, check=False)
        body = res_body.stdout

        category = "other"
        if "BREAKING CHANGE" in body or re.search(r"^[a-z]+(\([^\)]+\))?!:", subject):
            category = "breaking"
            has_breaking = True
        elif subject.startswith("feat") or re.search(r"^feat(\([^\)]+\))?:", subject):
            category = "feat"
            has_feat = True
        elif subject.startswith("fix") or re.search(r"^fix(\([^\)]+\))?:", subject):
            category = "fix"
            has_fix = True
        elif subject.startswith("docs"):
            category = "docs"
        elif subject.startswith("bench"):
            category = "bench"
        elif subject.startswith("test"):
            category = "test"
        elif subject.startswith("chore"):
            category = "chore"

        classified_commits.append({
            "hash": commit_hash,
            "subject": subject,
            "category": category
        })

    # Determina sugestão SemVer
    if has_breaking:
        recommendation = "MAJOR"
        reason = "Detectadas alterações com quebra de compatibilidade (BREAKING CHANGE / API breaking)."
    elif has_feat:
        recommendation = "MINOR"
        reason = "Detectadas novas funcionalidades retrocompatíveis ('feat:')."
    elif has_fix:
        recommendation = "PATCH"
        reason = "Detectadas correções de bugs ou ajustes numéricos ('fix:') sem novas funcionalidades."
    elif len(classified_commits) == 0:
        recommendation = "NONE"
        reason = "Nenhum commit realizado desde o último bump de versão."
    else:
        recommendation = "NONE"
        reason = "Todos os commits acumulados são exclusivos de documentação, benchmarks, testes ou build/configuração ('docs:', 'bench:', 'test:', 'chore:')."

    return {
        "base_commit": base_commit,
        "base_info": base_info,
        "commits": classified_commits,
        "recommendation": recommendation,
        "reason": reason
    }

def generate_updated_contents(data, target_version):
    new_maj, new_min, new_patch = target_version
    new_str = format_semver(new_maj, new_min, new_patch)

    # 1. VERSION
    new_version_content = f"{new_str}\n"

    # 2. code/include/ht_config.h
    cfg = data["config_content"]
    cfg = re.sub(r"(#define\s+HT_VERSION_MAJOR\s+)\d+U", rf"\g<1>{new_maj}U", cfg)
    cfg = re.sub(r"(#define\s+HT_VERSION_MINOR\s+)\d+U", rf"\g<1>{new_min}U", cfg)
    cfg = re.sub(r"(#define\s+HT_VERSION_PATCH\s+)\d+U", rf"\g<1>{new_patch}U", cfg)
    cfg = re.sub(r'(#define\s+HT_VERSION_STRING\s+)"[^"]+"', rf'\g<1>"{new_str}"', cfg)
    new_config_content = cfg

    # 3. code/tests/test_main.c
    test = data["test_content"]
    test = re.sub(r"(HT_ASSERT_EQ\(HT_VERSION_MAJOR,\s*)\d+U(\);)", rf"\g<1>{new_maj}U\g<2>", test)
    test = re.sub(r"(HT_ASSERT_EQ\(HT_VERSION_MINOR,\s*)\d+U(\);)", rf"\g<1>{new_min}U\g<2>", test)
    test = re.sub(r"(HT_ASSERT_EQ\(HT_VERSION_PATCH,\s*)\d+U(\);)", rf"\g<1>{new_patch}U\g<2>", test)
    test = re.sub(
        r"HT_ASSERT\(HT_VERSION_CHECK\(\d+,\s*\d+,\s*\d+\)\);",
        f"HT_ASSERT(HT_VERSION_CHECK({new_maj}, {new_min}, {new_patch}));",
        test
    )
    test = re.sub(
        r"HT_ASSERT\(!HT_VERSION_CHECK\(\d+,\s*\d+,\s*\d+\)\);",
        f"HT_ASSERT(!HT_VERSION_CHECK({new_maj}, {new_min}, {new_patch + 1}));",
        test,
        count=1
    )
    test = re.sub(
        r"HT_ASSERT\(!HT_VERSION_CHECK\(1,\s*0,\s*0\)\);",
        f"HT_ASSERT(!HT_VERSION_CHECK({new_maj + 1}, 0, 0));",
        test
    )
    test = re.sub(
        r'HT_ASSERT_EQ\(strcmp\(HT_VERSION_STRING,\s*"[^"]+"\),\s*0\);',
        f'HT_ASSERT_EQ(strcmp(HT_VERSION_STRING, "{new_str}"), 0);',
        test
    )
    test = re.sub(
        r'HT_ASSERT_EQ\(strcmp\(ht_version\(\),\s*"[^"]+"\),\s*0\);',
        f'HT_ASSERT_EQ(strcmp(ht_version(), "{new_str}"), 0);',
        test
    )
    new_test_content = test

    return {
        "new_version_content": new_version_content,
        "new_config_content": new_config_content,
        "new_test_content": new_test_content,
        "new_version_str": new_str
    }

def show_diff(old_content, new_content, filename):
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}"
    )
    return "".join(diff)

def cmd_analyze(repo_root):
    data = read_version_files(repo_root)
    analysis = analyze_commits(repo_root)

    curr_maj, curr_min, curr_patch = data["current_tuple"]
    curr_str = data["current_str"]

    if analysis["recommendation"] == "MAJOR":
        prop_str = format_semver(curr_maj + 1, 0, 0)
    elif analysis["recommendation"] == "MINOR":
        prop_str = format_semver(curr_maj, curr_min + 1, 0)
    elif analysis["recommendation"] == "PATCH":
        prop_str = format_semver(curr_maj, curr_min, curr_patch + 1)
    else:
        prop_str = curr_str

    print("# Parecer Técnico de Upgrade de Versão (SemVer 2.0.0)\n")
    print(f"- **Versão Atual:** `{curr_str}`")
    print(f"- **Versão Proposta:** `{prop_str}`")
    print(f"- **Recomendação SemVer:** `{analysis['recommendation']}`")
    print(f"- **Diagnóstico:** {analysis['reason']}")
    print(f"- **Consistência da Tríade de Arquivos:** {'Consistente [OK]' if data['is_consistent'] else 'Divergente [ALERTA]'}")

    if not data["is_consistent"]:
        print("\n> [!WARNING]")
        print("> Detalhes da divergência encontrada antes do bump:")
        for k, v in data["details"].items():
            print(f"> - `{k}`: {v}")

    print(f"\n### Commits Analisados desde o último bump (`{analysis['base_info']}`):")
    if not analysis["commits"]:
        print("*Nenhum commit encontrado.*")
    else:
        print("| Hash | Categoria | Mensagem |")
        print("| :--- | :--- | :--- |")
        for c in analysis["commits"]:
            cat_badge = f"`{c['category']}`"
            print(f"| `{c['hash']}` | {cat_badge} | {c['subject']} |")

    print("\n---")
    if analysis["recommendation"] != "NONE":
        print(f"**Ação Recomendada:** Apresente este parecer ao usuário e pergunte:")
        print(f"> *\"Você aprova o upgrade de versão de `{curr_str}` para `{prop_str}` ({analysis['recommendation']})? Deseja também gerar a tag Git anotada `v{prop_str}`?\"*")
    else:
        print("**Ação Recomendada:** Nenhuma alteração de versão necessária no momento.")

def cmd_apply(repo_root, target, dry_run=False, no_test=False, commit=False, tag=False, tag_msg=None):
    data = read_version_files(repo_root)
    curr_maj, curr_min, curr_patch = data["current_tuple"]
    curr_str = data["current_str"]

    if target.lower() == "patch":
        target_tuple = (curr_maj, curr_min, curr_patch + 1)
    elif target.lower() == "minor":
        target_tuple = (curr_maj, curr_min + 1, 0)
    elif target.lower() == "major":
        target_tuple = (curr_maj + 1, 0, 0)
    else:
        target_tuple = parse_semver(target)

    target_str = format_semver(*target_tuple)
    print(f"[*] Preparando upgrade de versão: {curr_str} -> {target_str}")

    # Verifica status da árvore antes de iniciar (exceto em dry-run)
    if not dry_run:
        res_st = run_cmd("git status --porcelain", cwd=repo_root, check=False)
        dirty_lines = [l for l in res_st.stdout.strip().split("\n") if l.strip() and not l.endswith("bump_version.py")]
        if dirty_lines:
            print("Erro: A árvore de trabalho possui modificações não commitadas. Limpe ou commite antes de prosseguir:", file=sys.stderr)
            for l in dirty_lines:
                print(f"  {l}", file=sys.stderr)
            sys.exit(1)

    updated = generate_updated_contents(data, target_tuple)

    # Exibe diffs
    diff_v = show_diff(data["version_content"], updated["new_version_content"], "VERSION")
    diff_c = show_diff(data["config_content"], updated["new_config_content"], "code/include/ht_config.h")
    diff_t = show_diff(data["test_content"], updated["new_test_content"], "code/tests/test_main.c")

    if dry_run:
        print("[*] Modo --dry-run: Nenhuma alteração gravada em disco. Diff planejado:\n")
        print(diff_v or "Sem alterações em VERSION")
        print(diff_c or "Sem alterações em code/include/ht_config.h")
        print(diff_t or "Sem alterações em code/tests/test_main.c")
        return

    # Escreve os arquivos em disco
    print("[*] Atualizando VERSION...")
    with open(data["version_file"], "w", encoding="utf-8") as f:
        f.write(updated["new_version_content"])

    print("[*] Atualizando code/include/ht_config.h...")
    with open(data["config_file"], "w", encoding="utf-8") as f:
        f.write(updated["new_config_content"])

    print("[*] Atualizando code/tests/test_main.c...")
    with open(data["test_file"], "w", encoding="utf-8") as f:
        f.write(updated["new_test_content"])

    # Executa o Gate de Testes
    if not no_test:
        print("[*] Executando Gate de Testes (1/2: precisão simples float)...")
        res1 = run_cmd("make clean && make test", cwd=repo_root, check=False)
        if res1.returncode != 0:
            print("FALHA no teste em float! Revertendo arquivos...", file=sys.stderr)
            run_cmd("git checkout -- VERSION code/include/ht_config.h code/tests/test_main.c", cwd=repo_root)
            sys.exit(res1.returncode)

        print("[*] Executando Gate de Testes (2/2: precisão dupla double)...")
        res2 = run_cmd("make clean && make test CFLAGS_EXTRA=\"-DHT_USE_DOUBLE\"", cwd=repo_root, check=False)
        if res2.returncode != 0:
            print("FALHA no teste em double! Revertendo arquivos...", file=sys.stderr)
            run_cmd("git checkout -- VERSION code/include/ht_config.h code/tests/test_main.c", cwd=repo_root)
            sys.exit(res2.returncode)

        print("[+] 100% dos testes unitários e de integração aprovados em float e double!")

    # Commit atômico
    if commit:
        commit_msg = f"chore(version): bump version to {target_str}"
        print(f"[*] Criando commit atômico: '{commit_msg}'...")
        run_cmd(f"git add VERSION code/include/ht_config.h code/tests/test_main.c", cwd=repo_root)
        run_cmd(f'git commit -m "{commit_msg}"', cwd=repo_root)
        print("[+] Commit realizado com sucesso!")

    # Tag Git anotada
    if tag:
        tag_name = f"v{target_str}"
        default_tag_msg = f"Release {tag_name}: kansho embedded Hoeffding Tree in C11"
        final_tag_msg = tag_msg if tag_msg else default_tag_msg
        print(f"[*] Criando tag anotada '{tag_name}'...")
        run_cmd(f'git tag -a "{tag_name}" -m "{final_tag_msg}"', cwd=repo_root)
        print(f"[+] Tag {tag_name} criada com sucesso!")

    print(f"\n[+] Upgrade para v{target_str} concluído com sucesso!")
    print(f"[*] Lembrete: Se aplicável, atualize a seção 4.1 de 'wiki/Versionamento-Releases-e-Roadmap.md' e sincronize via sync-wiki.")

def main():
    parser = argparse.ArgumentParser(description="Automação de avaliação e bump de versão SemVer 2.0.0 (kansho)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando analyze
    subparsers.add_parser("analyze", help="Avalia commits recentes e emite parecer SemVer para aprovação do usuário")

    # Subcomando apply
    parser_apply = subparsers.add_parser("apply", help="Aplica o upgrade de versão na tríade de arquivos")
    parser_apply.add_argument("target", help="Tipo de incremento ('patch', 'minor', 'major') ou versão explícita ('0.2.1')")
    parser_apply.add_argument("--dry-run", action="store_true", help="Simula as alterações sem gravar nem commitar")
    parser_apply.add_argument("--no-test", action="store_true", help="Pula a execução da suíte de testes")
    parser_apply.add_argument("--commit", action="store_true", help="Gera o commit atômico padronizado")
    parser_apply.add_argument("--tag", action="store_true", help="Cria tag Git anotada correspondente")
    parser_apply.add_argument("--tag-message", "-m", help="Mensagem da tag anotada")

    args = parser.parse_args()
    repo_root = get_repo_root()

    if args.command == "analyze":
        cmd_analyze(repo_root)
    elif args.command == "apply":
        cmd_apply(
            repo_root,
            args.target,
            dry_run=args.dry_run,
            no_test=args.no_test,
            commit=args.commit,
            tag=args.tag,
            tag_msg=args.tag_message
        )

if __name__ == "__main__":
    main()
