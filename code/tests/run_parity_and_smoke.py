#!/usr/bin/env python3
"""
run_parity_and_smoke.py — Hybrid cross-precision parity & runner smoke gate.

Validates float32 vs double64 vs Q16.16 end-to-end across:
  Block A (always runs, self-contained):
    - Sub-ULP narrow-feature classification (Electricity Cortex-M3 regime)
    - High-amplitude regression + Block-Rescaling (NYC Taxi regime)
    - Abrupt concept drift recovery (HAT + ADWIN2)
    - Federated Non-IID classification (Fed-VFDT STATS-MERGE & Fed-Forest TOP_M)
    - Federated high-amplitude regression (Fed-Forest TOP_M & Fed-VFDT STATS-MERGE)
  Block B (runs automatically when main_benchmark/data/ is present locally):
    - Deterministic slices (3000 train / 1000 test) of Electricity, Bikes,
      NYC Taxi, and Covertype across float32, double64, and Q16.16
  Block C:
    - Real POSIX TCP multi-process federated suite (make bench-fed-net)

Uses only Python stdlib + gcc + make (zero external dependencies).
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile


def collect_core_srcs(code_dir):
    p = os.path.join(code_dir, "src", "core")
    return [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".c")]


def collect_all_srcs(code_dir):
    srcs = []
    for sub in ("core", "ensemble", "federated"):
        p = os.path.join(code_dir, "src", sub)
        srcs.extend(os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".c"))
    return srcs


def compile_runner(cc, code_dir, srcs, driver_src, out_bin, extra_defs):
    inc = [
        f"-I{os.path.join(code_dir, 'include')}",
        f"-I{os.path.join(code_dir, 'include', 'core')}",
        f"-I{os.path.join(code_dir, 'include', 'ensemble')}",
        f"-I{os.path.join(code_dir, 'include', 'federated')}",
    ]
    cmd = (
        [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic"]
        + inc
        + extra_defs
        + srcs
        + [driver_src, "-lm", "-o", out_bin]
    )
    subprocess.check_call(cmd)


def run_bench_json(bin_path, csv_path, extra_args, workdir, tag):
    jpath = os.path.join(workdir, f"{tag}.json")
    cmd = [bin_path, csv_path, "--json", jpath] + extra_args
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"bench runner failed ({tag}): rc={res.returncode}\n{res.stderr}\n{res.stdout}")
    with open(jpath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_fed_json(bin_path, args):
    res = subprocess.run([bin_path] + args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"fed runner failed: rc={res.returncode}\n{res.stderr}\n{res.stdout}")
    for line in reversed(res.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"No JSON line in fed runner output:\n{res.stdout}")


def write_csv(path, stream, task, n_features, n_classes, rows):
    ftypes = ",".join(["0"] * n_features)
    fnames = ",".join(f"f{i}" for i in range(n_features))
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# stream={stream} task={task} n_samples={len(rows)} "
            f"n_features={n_features} n_classes={n_classes} "
            f"feature_types={ftypes} feature_names={fnames}\n"
        )
        for r in rows:
            f.write(",".join(r) + "\n")


def slice_real_csv(src_path, dst_path, max_rows):
    with open(src_path, "r", encoding="utf-8") as fin:
        header = fin.readline()
        data = []
        for line in fin:
            if not line.strip():
                continue
            data.append(line)
            if len(data) >= max_rows:
                break
    # Update n_samples= in header metadata
    parts = header.strip().split()
    new_parts = [f"n_samples={len(data)}" if p.startswith("n_samples=") else p for p in parts]
    with open(dst_path, "w", encoding="utf-8") as fout:
        fout.write(" ".join(new_parts) + "\n")
        fout.writelines(data)


def check_finite(val, label):
    if val is None or not math.isfinite(float(val)):
        raise AssertionError(f"Non-finite metric for {label}: {val}")


def run_block_a(workdir, bench_bins, fed_bins):
    print("\n=== Block A: Deterministic Extreme Synthetic Parity (float32 vs double64 vs Q16.16) ===")
    rng = random.Random(20261127)

    # 1. Sub-ULP narrow-feature classification (Electricity Cortex-M3 regime)
    sub_train, sub_test = [], []
    for idx in range(3600):
        narrow = 0.0001 + (idx * 0.000005)
        x1 = rng.uniform(0.0, 10.0)
        x2 = rng.uniform(-1.0, 1.0)
        y = 1 if (x1 + 0.5 * x2 > 5.0) else 0
        row = [f"{narrow:.6f}", f"{x1:.6f}", f"{x2:.6f}", str(y)]
        (sub_train if idx < 3000 else sub_test).append(row)
    tr_p = os.path.join(workdir, "sub_tr.csv")
    te_p = os.path.join(workdir, "sub_te.csv")
    write_csv(tr_p, "synth_sub_ulp", "classification", 3, 2, sub_train)
    write_csv(te_p, "synth_sub_ulp", "classification", 3, 2, sub_test)

    res_sub = {}
    for mode, bpath in bench_bins.items():
        res_sub[mode] = run_bench_json(
            bpath, tr_p,
            ["--task", "classification", "--obs", "gaussian", "--leaf", "nba",
             "--grace-period", "50", "--delta", "1e-4", "--test-csv", te_p],
            workdir, f"sub_{mode}"
        )
    a32 = res_sub["float32"]["holdout_accuracy"]
    a64 = res_sub["double64"]["holdout_accuracy"]
    aq16 = res_sub["Q16.16"]["holdout_accuracy"]
    for m, r in res_sub.items():
        check_finite(r["holdout_accuracy"], f"sub_ulp/{m}")
        assert r["final_nodes"] >= 3, f"sub_ulp/{m} failed to split (nodes={r['final_nodes']})"
    assert abs(a32 - a64) <= 0.01, f"sub_ulp f32 vs f64 diverged: {a32:.4f} vs {a64:.4f}"
    assert abs(aq16 - a32) <= 0.03, f"sub_ulp Q16 vs f32 diverged: {aq16:.4f} vs {a32:.4f}"
    print(f"  [PASS] Sub-ULP Classif   | f64={a64*100:.2f}% f32={a32*100:.2f}% Q16={aq16*100:.2f}% (nodes={res_sub['Q16.16']['final_nodes']})")

    # 2. High-amplitude long-stream regression (NYC Taxi regime)
    reg_train, reg_test = [], []
    for idx in range(5800):
        bucket = rng.randint(0, 3)
        x0 = bucket * 10.0 + rng.uniform(0.0, 2.0)
        x1 = rng.uniform(0.0, 5.0)
        x2 = rng.uniform(0.0, 10.0)
        base = [250.0, 1100.0, 2300.0, 3500.0][bucket]
        y = base + 12.0 * x1 + rng.uniform(-20.0, 20.0)
        row = [f"{x0:.6f}", f"{x1:.6f}", f"{x2:.6f}", f"{y:.6f}"]
        (reg_train if idx < 5000 else reg_test).append(row)
    tr_r = os.path.join(workdir, "reg_tr.csv")
    te_r = os.path.join(workdir, "reg_te.csv")
    write_csv(tr_r, "synth_high_reg", "regression", 3, 0, reg_train)
    write_csv(te_r, "synth_high_reg", "regression", 3, 0, reg_test)

    res_reg = {}
    for mode, bpath in bench_bins.items():
        res_reg[mode] = run_bench_json(
            bpath, tr_r,
            ["--task", "regression", "--leaf", "adaptive", "--leaf-normalize",
             "--grace-period", "100", "--delta", "1e-4", "--test-csv", te_r],
            workdir, f"reg_{mode}"
        )
    m32 = res_reg["float32"]["holdout_mae"]
    m64 = res_reg["double64"]["holdout_mae"]
    mq16 = res_reg["Q16.16"]["holdout_mae"]
    for m, r in res_reg.items():
        check_finite(r["holdout_mae"], f"high_reg/{m}")
        check_finite(r["holdout_r2"], f"high_reg_r2/{m}")
        assert r["final_nodes"] >= 3, f"high_reg/{m} failed to split (nodes={r['final_nodes']})"
        assert r["holdout_r2"] > 0.85, f"high_reg/{m} poor R2={r['holdout_r2']:.4f}"
    assert abs(m32 - m64) / max(m64, 1.0) <= 0.03, f"high_reg f32 vs f64 MAE diverged: {m32:.2f} vs {m64:.2f}"
    assert mq16 <= m32 * 1.15 + 2.0, f"high_reg Q16 degraded vs f32 MAE: {mq16:.2f} vs {m32:.2f}"
    print(f"  [PASS] High-Amp Reg      | MAE f64={m64:.2f} f32={m32:.2f} Q16={mq16:.2f} (Q16 R2={res_reg['Q16.16']['holdout_r2']:.4f}, nodes={res_reg['Q16.16']['final_nodes']})")

    # 3. Abrupt concept drift (HAT + ADWIN2)
    drift_rows = []
    for idx in range(3000):
        x0 = rng.uniform(0.0, 10.0)
        x1 = rng.uniform(0.0, 10.0)
        y = (1 if x0 > 5.0 else 0) if idx < 1500 else (0 if x0 > 5.0 else 1)
        drift_rows.append([f"{x0:.6f}", f"{x1:.6f}", str(y)])
    dr_p = os.path.join(workdir, "drift.csv")
    write_csv(dr_p, "synth_drift", "classification", 2, 2, drift_rows)

    res_dr = {}
    for mode, bpath in bench_bins.items():
        res_dr[mode] = run_bench_json(
            bpath, dr_p,
            ["--task", "classification", "--hat", "--bootstrap-sampling", "0",
             "--grace-period", "40", "--delta", "1e-3", "--drift-window-threshold", "50"],
            workdir, f"drift_{mode}"
        )
    for m, r in res_dr.items():
        check_finite(r["final_accuracy"], f"drift/{m}")
        assert r["final_accuracy"] >= 0.78, f"drift/{m} failed to recover (acc={r['final_accuracy']:.4f})"
    assert abs(res_dr["Q16.16"]["final_accuracy"] - res_dr["float32"]["final_accuracy"]) <= 0.05
    print(
        f"  [PASS] HAT Abrupt Drift  | f64={res_dr['double64']['final_accuracy']*100:.2f}% "
        f"f32={res_dr['float32']['final_accuracy']*100:.2f}% Q16={res_dr['Q16.16']['final_accuracy']*100:.2f}%"
    )

    # 4. Federated Non-IID classification (4 clients, 6 rounds)
    fed_cls_dir = os.path.join(workdir, "fed_cls")
    os.makedirs(fed_cls_dir)
    for k in range(4):
        crows = []
        # Non-IID skew: clients 0,1 mostly class 0; clients 2,3 mostly class 1
        for _ in range(400):
            c = (0 if rng.random() < 0.85 else 1) if k < 2 else (1 if rng.random() < 0.85 else 0)
            x0 = rng.uniform(0.0, 4.0) if c == 0 else rng.uniform(6.0, 10.0)
            x1 = rng.uniform(0.0, 10.0)
            crows.append([f"{x0:.6f}", f"{x1:.6f}", str(c)])
        write_csv(os.path.join(fed_cls_dir, f"client_{k}.csv"), "fed_cls", "classification", 2, 2, crows)
    fed_cls_te = os.path.join(workdir, "fed_cls_te.csv")
    write_csv(
        fed_cls_te, "fed_cls", "classification", 2, 2,
        [[f"{(rng.uniform(0.0, 4.0) if (i % 2 == 0) else rng.uniform(6.0, 10.0)):.6f}",
          f"{rng.uniform(0.0, 10.0):.6f}", str(i % 2)] for i in range(400)]
    )

    for strat in ("fed_vfdt_merge", "fed_forest_topm"):
        r_f = {}
        for mode, fbin in fed_bins.items():
            r_f[mode] = run_fed_json(
                fbin, ["--clients-dir", fed_cls_dir, "--test-csv", fed_cls_te,
                       "--strategy", strat, "--n-clients", "4", "--rounds", "6",
                       "--grace", "60", "--delta", "1e-3"]
            )
            check_finite(r_f[mode]["holdout_acc"], f"{strat}/{mode}")
            assert r_f[mode]["holdout_acc"] >= 88.0, f"{strat}/{mode} low acc={r_f[mode]['holdout_acc']}"
        assert abs(r_f["float32"]["holdout_acc"] - r_f["double64"]["holdout_acc"]) <= 1.0
        assert abs(r_f["Q16.16"]["holdout_acc"] - r_f["float32"]["holdout_acc"]) <= 3.0
        print(
            f"  [PASS] Fed {strat:15s} | f64={r_f['double64']['holdout_acc']:.2f}% "
            f"f32={r_f['float32']['holdout_acc']:.2f}% Q16={r_f['Q16.16']['holdout_acc']:.2f}%"
        )

    # 5. Federated high-amplitude regression (4 clients, 6 rounds, y in [250, 3500])
    fed_reg_dir = os.path.join(workdir, "fed_reg")
    os.makedirs(fed_reg_dir)
    for k in range(4):
        crows = []
        for _ in range(400):
            b = rng.randint(0, 3)
            x0 = b * 10.0 + rng.uniform(0.0, 2.0)
            x1 = rng.uniform(0.0, 5.0)
            y = [250.0, 1100.0, 2300.0, 3500.0][b] + rng.uniform(-25.0, 25.0)
            crows.append([f"{x0:.6f}", f"{x1:.6f}", f"{y:.6f}"])
        write_csv(os.path.join(fed_reg_dir, f"client_{k}.csv"), "fed_reg", "regression", 2, 0, crows)
    fed_reg_te = os.path.join(workdir, "fed_reg_te.csv")
    write_csv(
        fed_reg_te, "fed_reg", "regression", 2, 0,
        [[f"{((i % 4) * 10.0 + rng.uniform(0.0, 2.0)):.6f}",
          f"{rng.uniform(0.0, 5.0):.6f}",
          f"{([250.0, 1100.0, 2300.0, 3500.0][i % 4]):.6f}"] for i in range(400)]
    )

    for strat in ("fed_forest_topm", "fed_vfdt_merge"):
        r_f = {}
        for mode, fbin in fed_bins.items():
            r_f[mode] = run_fed_json(
                fbin, ["--clients-dir", fed_reg_dir, "--test-csv", fed_reg_te,
                       "--strategy", strat, "--n-clients", "4", "--rounds", "6",
                       "--grace", "80", "--delta", "1e-3", "--leaf", "adaptive", "--leaf-normalize"]
            )
            check_finite(r_f[mode]["holdout_mae"], f"fed_reg_{strat}/{mode}")
            check_finite(r_f[mode]["holdout_r2"], f"fed_reg_r2_{strat}/{mode}")
            assert r_f[mode]["holdout_r2"] > 0.70, f"fed_reg {strat}/{mode} poor R2={r_f[mode]['holdout_r2']:.4f}"
            if strat == "fed_vfdt_merge":
                assert r_f[mode]["avg_nodes"] >= 3.0, f"fed_reg merge/{mode} failed to split"
        print(
            f"  [PASS] FedReg {strat:12s} | MAE f64={r_f['double64']['holdout_mae']:.1f} "
            f"f32={r_f['float32']['holdout_mae']:.1f} Q16={r_f['Q16.16']['holdout_mae']:.1f} "
            f"(Q16 R2={r_f['Q16.16']['holdout_r2']:.4f})"
        )


def run_block_b(workdir, repo_root, bench_bins, wide_bins):
    data_dir = os.path.join(repo_root, "main_benchmark", "data")
    if not os.path.isdir(data_dir):
        print("\n=== Block B: Real-Dataset Smoke Slices [SKIPPED — main_benchmark/data/ not present] ===")
        return

    print("\n=== Block B: Conditional Real-Dataset Smoke Slices (3000 train / 1000 test) ===")
    specs = [
        ("electricity", "classification", bench_bins,
         ["--task", "classification", "--obs", "gaussian", "--leaf", "nba", "--grace-period", "100", "--delta", "1e-5"],
         0.01, 0.04),
        ("bikes", "regression", bench_bins,
         ["--task", "regression", "--leaf", "mean", "--grace-period", "100", "--delta", "1e-4"],
         0.25, 0.20),
        ("nyc_taxi", "regression", bench_bins,
         ["--task", "regression", "--leaf", "adaptive", "--leaf-normalize", "--grace-period", "100", "--delta", "1e-4"],
         0.03, 0.25),
        ("covertype", "classification", wide_bins,
         ["--task", "classification", "--obs", "gaussian", "--leaf", "nba", "--grace-period", "50", "--delta", "1e-2"],
         0.02, 0.06),
    ]

    for ds, task, bins, extra, tol_f32_f64, tol_q16_f32 in specs:
        tr_src = os.path.join(data_dir, f"{ds}_train.csv")
        te_src = os.path.join(data_dir, f"{ds}_test.csv")
        if not (os.path.exists(tr_src) and os.path.exists(te_src)):
            print(f"  [SKIP] {ds} (files missing)")
            continue
        tr_sl = os.path.join(workdir, f"{ds}_tr.csv")
        te_sl = os.path.join(workdir, f"{ds}_te.csv")
        slice_real_csv(tr_src, tr_sl, 3000)
        slice_real_csv(te_src, te_sl, 1000)

        res = {}
        for mode, bpath in bins.items():
            res[mode] = run_bench_json(bpath, tr_sl, extra + ["--test-csv", te_sl], workdir, f"real_{ds}_{mode}")

        if task == "classification":
            v64 = res["double64"]["holdout_accuracy"]
            v32 = res["float32"]["holdout_accuracy"]
            vq16 = res["Q16.16"]["holdout_accuracy"]
            for m, v in (("double64", v64), ("float32", v32), ("Q16.16", vq16)):
                check_finite(v, f"{ds}/{m}")
            assert abs(v32 - v64) <= tol_f32_f64, f"{ds}: f32={v32:.4f} vs f64={v64:.4f}"
            assert abs(vq16 - v32) <= tol_q16_f32, f"{ds}: Q16={vq16:.4f} vs f32={v32:.4f}"
            print(
                f"  [PASS] {ds:12s} (cls) | f64={v64*100:.2f}% f32={v32*100:.2f}% Q16={vq16*100:.2f}% "
                f"(nodes: f32={res['float32']['final_nodes']} Q16={res['Q16.16']['final_nodes']})"
            )
        else:
            v64 = res["double64"]["holdout_mae"]
            v32 = res["float32"]["holdout_mae"]
            vq16 = res["Q16.16"]["holdout_mae"]
            for m, v in (("double64", v64), ("float32", v32), ("Q16.16", vq16)):
                check_finite(v, f"{ds}/{m}")
                assert res[m]["final_nodes"] >= 3, f"{ds}/{m} failed to split (nodes={res[m]['final_nodes']})"
            assert abs(v32 - v64) / max(v64, 1.0) <= tol_f32_f64, f"{ds}: MAE f32={v32:.2f} vs f64={v64:.2f}"
            assert vq16 <= v32 * (1.0 + tol_q16_f32) + 2.0, f"{ds}: MAE Q16={vq16:.2f} degraded vs f32={v32:.2f}"
            print(
                f"  [PASS] {ds:12s} (reg) | MAE f64={v64:.2f} f32={v32:.2f} Q16={vq16:.2f} "
                f"(nodes: f32={res['float32']['final_nodes']} Q16={res['Q16.16']['final_nodes']})"
            )


def run_block_c(code_dir):
    print("\n=== Block C: Real POSIX TCP Federated Network Suite (bench-fed-net) ===")
    res = subprocess.run(["make", "-C", code_dir, "bench-fed-net"], capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-1500:])
        print(res.stderr[-1500:], file=sys.stderr)
        raise RuntimeError(f"bench-fed-net failed (rc={res.returncode})")
    for line in res.stdout.splitlines():
        if "PASS" in line or "SCENARIO" in line or "SUMMARY" in line or "5/5" in line:
            print("  " + line.strip())
    print("  [PASS] Real POSIX TCP federated suite green")


def main():
    parser = argparse.ArgumentParser(description="kansho cross-precision parity & runner smoke gate")
    parser.add_argument("--code-dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--skip-tcp", action="store_true")
    args = parser.parse_args()

    code_dir = os.path.abspath(args.code_dir)
    repo_root = os.path.abspath(os.path.join(code_dir, ".."))
    core_srcs = collect_core_srcs(code_dir)
    all_srcs = collect_all_srcs(code_dir)
    bench_driver = os.path.join(repo_root, "benchmarks", "bench_c_runner.c")
    fed_driver = os.path.join(repo_root, "main_benchmark", "fed_c_runner.c")

    with tempfile.TemporaryDirectory(prefix="kansho_parity_") as workdir:
        print("[*] Compiling 3-precision bench_c_runner & fed_c_runner binaries...")
        prec_defs = {
            "float32": [],
            "double64": ["-DHT_USE_DOUBLE"],
            "Q16.16": ["-DHT_USE_FIXED_POINT"],
        }
        bench_bins = {}
        wide_bins = {}
        fed_bins = {}
        for mode, pdefs in prec_defs.items():
            b = os.path.join(workdir, f"bench_{mode}")
            compile_runner(args.cc, code_dir, core_srcs, bench_driver, b, pdefs)
            bench_bins[mode] = b

            w = os.path.join(workdir, f"bench_wide_{mode}")
            compile_runner(
                args.cc, code_dir, core_srcs, bench_driver, w,
                pdefs + ["-DHT_MAX_FEATURES=64", "-DHT_MAX_CLASSES=8",
                         "-DHT_MAX_ACTIVE_LEAVES=16", "-DHT_MAX_NODES=256"],
            )
            wide_bins[mode] = w

            fb = os.path.join(workdir, f"fed_{mode}")
            compile_runner(
                args.cc, code_dir, all_srcs, fed_driver, fb,
                pdefs + ["-DHT_MAX_NODES=128", "-DHT_MAX_ACTIVE_LEAVES=8",
                         "-DHT_COMPACT_MAX_NODES=64", "-DHT_FED_FOREST_MAX_TREES=8",
                         "-DHT_FED_SERVER_MAX_TREES=16", "-DHT_FED_SERVER_MAX_PROPOSALS=128"],
            )
            fed_bins[mode] = fb

        run_block_a(workdir, bench_bins, fed_bins)
        run_block_b(workdir, repo_root, bench_bins, wide_bins)

    if not args.skip_tcp:
        run_block_c(code_dir)

    print("\nCROSS-PRECISION PARITY & RUNNER SMOKE GATE PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
