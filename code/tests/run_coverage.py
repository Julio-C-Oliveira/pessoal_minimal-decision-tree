#!/usr/bin/env python3
"""
run_coverage.py — Automated gcov line & branch coverage gate for kansho (libht).

Builds and runs code/tests/ with GCC --coverage (-O0) in both float32 and Q16.16
modes, prints a per-module summary table, and enforces coverage floors:
  - Total lines executed >= 92.0%
  - Total branches taken >= 80.0%
  - Every C module in code/src/ >= 85.0% lines executed

Uses only Python stdlib + gcc + gcov (zero external dependencies).
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile


def collect_sources(code_dir):
    src_files = []
    for sub in ("core", "ensemble", "federated"):
        p = os.path.join(code_dir, "src", sub)
        for f in sorted(os.listdir(p)):
            if f.endswith(".c"):
                src_files.append(os.path.abspath(os.path.join(p, f)))

    test_files = [os.path.abspath(os.path.join(code_dir, "tests", "test_main.c"))]
    for sub in ("core", "ensemble", "federated"):
        p = os.path.join(code_dir, "tests", sub)
        for f in sorted(os.listdir(p)):
            if f.endswith(".c"):
                test_files.append(os.path.abspath(os.path.join(p, f)))

    return src_files, test_files


def run_mode(code_dir, mode_name, extra_cflags, ldflags, cc):
    src_files, test_files = collect_sources(code_dir)
    inc_flags = [
        f"-I{os.path.abspath(os.path.join(code_dir, 'include'))}",
        f"-I{os.path.abspath(os.path.join(code_dir, 'include', 'core'))}",
        f"-I{os.path.abspath(os.path.join(code_dir, 'include', 'ensemble'))}",
        f"-I{os.path.abspath(os.path.join(code_dir, 'include', 'federated'))}",
    ]
    test_inc = [f"-I{os.path.abspath(os.path.join(code_dir, 'tests'))}"]

    with tempfile.TemporaryDirectory(prefix=f"kansho_cov_{mode_name}_") as workdir:
        objs = []
        for sf in src_files:
            oname = os.path.join(workdir, os.path.basename(sf).replace(".c", ".o"))
            cmd = [
                cc, "-std=c11", "-O0", "--coverage",
                "-Wall", "-Wextra", "-Werror", "-pedantic",
            ] + inc_flags + extra_cflags + ["-c", sf, "-o", oname]
            subprocess.check_call(cmd)
            objs.append(oname)

        for tf in test_files:
            oname = os.path.join(workdir, "t_" + os.path.basename(tf).replace(".c", ".o"))
            cmd = [
                cc, "-std=c11", "-O0",
                "-Wall", "-Wextra", "-Werror", "-pedantic",
            ] + inc_flags + test_inc + extra_cflags + ["-c", tf, "-o", oname]
            subprocess.check_call(cmd)
            objs.append(oname)

        exe = os.path.join(workdir, "test_runner_cov")
        subprocess.check_call([cc, "--coverage"] + objs + ldflags + ["-o", exe])
        res = subprocess.run([exe], capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-2000:])
            print(res.stderr[-2000:], file=sys.stderr)
            raise RuntimeError(f"Test runner failed under coverage mode '{mode_name}' (rc={res.returncode})")

        rows = []
        total_lines = 0
        covered_lines = 0
        total_branches = 0
        taken_branches = 0

        for sf in src_files:
            base = os.path.basename(sf)
            oname = os.path.join(workdir, base.replace(".c", ".o"))
            out = subprocess.check_output(
                ["gcov", "-b", "-o", workdir, oname],
                cwd=workdir,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for block in out.split("File ")[1:]:
                header = block.splitlines()[0]
                if base in header:
                    lm = re.search(r"Lines executed:([0-9.]+)% of (\d+)", block)
                    tm = re.search(r"Taken at least once:([0-9.]+)% of (\d+)", block)
                    if lm:
                        lp, ln = float(lm.group(1)), int(lm.group(2))
                        tp, tn = (float(tm.group(1)), int(tm.group(2))) if tm else (100.0, 0)
                        lc = int(round(lp * ln / 100.0))
                        tc = int(round(tp * tn / 100.0))
                        total_lines += ln
                        covered_lines += lc
                        total_branches += tn
                        taken_branches += tc
                        rows.append((base, lp, ln, tp, tn))
                    break

        tot_lp = (100.0 * covered_lines / total_lines) if total_lines else 0.0
        tot_bp = (100.0 * taken_branches / total_branches) if total_branches else 0.0
        return rows, tot_lp, total_lines, tot_bp, total_branches


def main():
    parser = argparse.ArgumentParser(description="kansho gcov coverage gate")
    parser.add_argument("--code-dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--min-total-lines", type=float, default=92.0)
    parser.add_argument("--min-total-branches", type=float, default=80.0)
    parser.add_argument("--min-module-lines", type=float, default=85.0)
    parser.add_argument("--no-enforce", action="store_true")
    args = parser.parse_args()

    modes = [
        ("float32", [], ["-lm"]),
        ("Q16.16", ["-DHT_USE_FIXED_POINT"], []),
    ]

    failures = []
    for mode_name, extra, ld in modes:
        rows, tot_lp, tot_ln, tot_bp, tot_bn = run_mode(args.code_dir, mode_name, extra, ld, args.cc)
        print(f"\n=== kansho gcov Coverage Report ({mode_name}) ===")
        print(f"{'Module':22s} | {'Lines %':>8s} ({'Total':>5s}) | {'Branches Taken %':>16s} ({'Total':>5s})")
        print("-" * 68)
        for base, lp, ln, tp, tn in rows:
            flag = "  <-- BELOW MIN" if lp < args.min_module_lines else ""
            print(f"{base:22s} | {lp:7.2f}% ({ln:5d}) | {tp:15.2f}% ({tn:5d}){flag}")
            if lp < args.min_module_lines:
                failures.append(f"[{mode_name}] {base} lines {lp:.2f}% < {args.min_module_lines:.1f}%")
        print("-" * 68)
        print(f"{'TOTAL':22s} | {tot_lp:7.2f}% ({tot_ln:5d}) | {tot_bp:15.2f}% ({tot_bn:5d})")

        if tot_lp < args.min_total_lines:
            failures.append(f"[{mode_name}] TOTAL lines {tot_lp:.2f}% < {args.min_total_lines:.1f}%")
        if tot_bp < args.min_total_branches:
            failures.append(f"[{mode_name}] TOTAL branches {tot_bp:.2f}% < {args.min_total_branches:.1f}%")

    if failures and not args.no_enforce:
        print("\nCOVERAGE GATE FAILED:", file=sys.stderr)
        for f in failures:
            print("  -", f, file=sys.stderr)
        return 1

    print("\nCOVERAGE GATE PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
