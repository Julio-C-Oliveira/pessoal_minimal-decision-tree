#!/usr/bin/env python3
"""
Fixed-Point Q16.16 vs Float32 Benchmark Suite (Phase 8 — Delivery 8.5).

Evaluates Kansho across all 8 flagship datasets from main_benchmark/data
comparing IEEE 754 float32 vs Zero-Libm Q16.16 Fixed-Point (-DHT_USE_FIXED_POINT)
on both High Tier (Tier 256 / Tier 40) and MCU Tier 4 (64 KB SRAM profile).

Produces:
- main_benchmark/fixed_point_benchmark_report.md
- main_benchmark/fixed_point_benchmark_results.json
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from main_benchmark import (  # noqa: E402
    DATASET_OPTIMAL_CONFIGS,
    build_c11_cli_args,
    run_isolated_c_tier,
)
from tune_hyperparams import calculate_safe_capacity  # noqa: E402

DATASETS = [
    ("sea", "SEA", "classification", False),
    ("electricity", "Electricity", "classification", False),
    ("covertype", "Forest Covertype", "classification", True),
    ("airlines", "Airlines", "classification", False),
    ("friedman", "Friedman #1", "regression", False),
    ("bikes", "Bikes", "regression", False),
    ("appliances_energy", "Appliances Energy", "regression", True),
    ("nyc_taxi", "NYC Taxi", "regression", False),
]


def build_runner_binary(
    tier: int,
    reg_slots: int,
    is_high_dim: bool,
    task: str,
    fixed_point: bool,
) -> str:
    code_dir = os.path.join(PROJECT_ROOT, "code")
    build_dir = os.path.join(code_dir, "build", "fixed_bench")
    os.makedirs(build_dir, exist_ok=True)

    mode_tag = "q16" if fixed_point else "f32"
    dim_tag = "hd" if is_high_dim else "std"
    task_tag = "cls" if task == "classification" else f"reg_s{reg_slots}"
    bin_path = os.path.join(build_dir, f"bench_{mode_tag}_t{tier}_{dim_tag}_{task_tag}")

    if os.path.exists(bin_path):
        return bin_path

    leaves, max_nodes = calculate_safe_capacity(tier, reg_slots, is_high_dim, task)
    defs = [
        f"-DHT_MAX_ACTIVE_LEAVES={leaves}",
        f"-DHT_MAX_NODES={max_nodes}",
        "-DHT_ADWIN_LEVELS=10",
        "-DHT_MAX_ADWIN_DETECTORS=32",
    ]
    if is_high_dim:
        defs.extend(["-DHT_MAX_FEATURES=64", "-DHT_MAX_CLASSES=8"])
    else:
        defs.extend(["-DHT_MAX_FEATURES=16", "-DHT_MAX_CLASSES=4"])

    if task == "classification":
        defs.append("-DHT_CLASSIF_ONLY")
    else:
        defs.append(f"-DHT_REG_NUM_SLOTS={reg_slots}")

    if fixed_point:
        defs.append("-DHT_USE_FIXED_POINT")

    inc_flags = [
        f"-I{os.path.join(code_dir, 'include')}",
        f"-I{os.path.join(code_dir, 'include', 'core')}",
        f"-I{os.path.join(code_dir, 'include', 'ensemble')}",
        f"-I{os.path.join(code_dir, 'include', 'federated')}",
    ]

    core_dir = os.path.join(code_dir, "src", "core")
    core_srcs = sorted(
        os.path.join(core_dir, f) for f in os.listdir(core_dir) if f.endswith(".c")
    )
    bench_src = os.path.join(PROJECT_ROOT, "benchmarks", "bench_c_runner.c")

    cmd = [
        "gcc",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        *inc_flags,
        *defs,
        *core_srcs,
        bench_src,
        "-lm",
        "-o",
        bin_path,
    ]
    subprocess.run(cmd, check=True)
    return bin_path


def generate_markdown_report(results: List[Dict[str, Any]], report_path: str) -> None:
    lines: List[str] = []
    lines.append("# Fixed-Point Q16.16 vs Float32 Benchmark Report (Phase 8 — Delivery 8.5)")
    lines.append("")
    lines.append(
        "Direct A/B parity evaluation across all 8 flagship datasets of `main_benchmark/data/` "
        "comparing **IEEE 754 `float32`** vs **Zero-Libm `Q16.16` Fixed-Point (`-DHT_USE_FIXED_POINT`)** "
        "using identical tuned hyperparameters on both **High Tier (`Tier 256` / `Tier 40`)** and **MCU `Tier 4`**."
    )
    lines.append("")

    for tier_key, tier_title in [("high", "High Tier (Tier 256 / Tier 40)"), ("tier4", "MCU Tier 4 (4 Active Leaves / 64 Nodes)")]:
        lines.append(f"## {tier_title}")
        lines.append("")
        lines.append("### Classification Datasets (4)")
        lines.append("")
        lines.append("| Dataset | Float32 Holdout Acc | Q16.16 Holdout Acc | Delta Acc (pp) | Float32 Preq Acc | Q16.16 Preq Acc | Float32 Nodes | Q16.16 Nodes | Float32 RAM (KB) | Q16.16 RAM (KB) | Float32 us/s | Q16.16 us/s |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for row in results:
            if row["task"] != "classification":
                continue
            f32 = row[tier_key]["float32"]
            q16 = row[tier_key]["q16"]
            delta_acc = q16["holdout_acc"] - f32["holdout_acc"]
            lines.append(
                f"| **{row['name']}** | {f32['holdout_acc']:.2f}% | {q16['holdout_acc']:.2f}% | "
                f"{delta_acc:+.2f} pp | {f32['prequential_acc']:.2f}% | {q16['prequential_acc']:.2f}% | "
                f"{f32['n_nodes']} | {q16['n_nodes']} | {f32['intra_train_ram_kb']:.1f} | "
                f"{q16['intra_train_ram_kb']:.1f} | {f32['avg_us_per_sample']:.2f} | {q16['avg_us_per_sample']:.2f} |"
            )

        lines.append("")
        lines.append("### Regression Datasets (4)")
        lines.append("")
        lines.append("| Dataset | Float32 MAE | Q16.16 MAE | Delta MAE | Float32 RMSE | Q16.16 RMSE | Float32 R² | Q16.16 R² | Float32 Nodes | Q16.16 Nodes | Float32 us/s | Q16.16 us/s |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for row in results:
            if row["task"] != "regression":
                continue
            f32 = row[tier_key]["float32"]
            q16 = row[tier_key]["q16"]
            delta_mae = q16["holdout_mae"] - f32["holdout_mae"]
            lines.append(
                f"| **{row['name']}** | {f32['holdout_mae']:.4f} | {q16['holdout_mae']:.4f} | "
                f"{delta_mae:+.4f} | {f32['holdout_rmse']:.4f} | {q16['holdout_rmse']:.4f} | "
                f"{f32['holdout_r2']:.4f} | {q16['holdout_r2']:.4f} | "
                f"{f32['n_nodes']} | {q16['n_nodes']} | {f32['avg_us_per_sample']:.2f} | {q16['avg_us_per_sample']:.2f} |"
            )

        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q16.16 vs float32 benchmark across 8 main datasets.")
    parser.add_argument("--data-dir", default=os.path.join(SCRIPT_DIR, "data"))
    parser.add_argument("--report", default=os.path.join(SCRIPT_DIR, "fixed_point_benchmark_report.md"))
    parser.add_argument("--json", default=os.path.join(SCRIPT_DIR, "fixed_point_benchmark_results.json"))
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []

    print("================================================================================")
    print(" PHASE 8 BENCHMARK: IEEE 754 float32 vs ZERO-LIBM Q16.16 (-DHT_USE_FIXED_POINT)")
    print("================================================================================")

    for d_id, name, task, is_high_dim in DATASETS:
        train_csv = os.path.join(args.data_dir, f"{d_id}_train.csv")
        test_csv = os.path.join(args.data_dir, f"{d_id}_test.csv")
        opt_cfg = DATASET_OPTIMAL_CONFIGS[d_id]
        high_tier = 40 if is_high_dim else 256

        print(f"\n>>> Dataset: {name} [{task.upper()}]")
        entry: Dict[str, Any] = {"id": d_id, "name": name, "task": task, "high_tier": high_tier}

        for tier_key, tier_num, cfg_key in [
            ("high", high_tier, f"kansho_t{high_tier}"),
            ("tier4", 4, "kansho_t4"),
        ]:
            params = opt_cfg[cfg_key]
            slots = params.get("reg_num_slots", 16) if task == "regression" else 16
            cli_args = build_c11_cli_args(params, task)

            bin_f32 = build_runner_binary(tier_num, slots, is_high_dim, task, fixed_point=False)
            bin_q16 = build_runner_binary(tier_num, slots, is_high_dim, task, fixed_point=True)

            res_f32 = run_isolated_c_tier(bin_f32, train_csv, test_csv, task, f"f32_t{tier_num}", extra_args=cli_args)
            res_q16 = run_isolated_c_tier(bin_q16, train_csv, test_csv, task, f"q16_t{tier_num}", extra_args=cli_args)

            entry[tier_key] = {"float32": res_f32, "q16": res_q16}

            if task == "classification":
                delta = res_q16["holdout_acc"] - res_f32["holdout_acc"]
                print(
                    f"  [Tier {tier_num:>3}] float32 Acc={res_f32['holdout_acc']:6.2f}% ({res_f32['avg_us_per_sample']:5.2f} us/s) | "
                    f"Q16.16 Acc={res_q16['holdout_acc']:6.2f}% ({res_q16['avg_us_per_sample']:5.2f} us/s) | Delta={delta:+.2f} pp"
                )
            else:
                delta = res_q16["holdout_mae"] - res_f32["holdout_mae"]
                print(
                    f"  [Tier {tier_num:>3}] float32 MAE={res_f32['holdout_mae']:8.4f} R2={res_f32['holdout_r2']:7.4f} | "
                    f"Q16.16 MAE={res_q16['holdout_mae']:8.4f} R2={res_q16['holdout_r2']:7.4f} | DeltaMAE={delta:+.4f}"
                )

        results.append(entry)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    generate_markdown_report(results, args.report)
    print(f"\nSaved report to {args.report} and JSON to {args.json}")


if __name__ == "__main__":
    main()
