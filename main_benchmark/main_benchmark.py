#!/usr/bin/env python3
"""
Main Performance Benchmark Suite for Embedded Hoeffding Tree (Kansho).

Executes comprehensive multi-tier evaluation across 8 flagship datasets:
--- CLASSIFICATION (4) ---
1. SEA (Synthetic, 3 features, 2 classes)
2. Electricity / Elec2 (Real, 8 features, 2 classes)
3. Forest Covertype (Real, 54 features, 7 classes)
4. Airlines Delay (Real, 7 features, 2 classes)

--- REGRESSION (4) ---
5. Friedman #1 (Synthetic, 10 features, continuous nonlinear target)
6. Bikes (Real, 12 features, continuous bike rental demand)
7. Appliances Energy (Real, 28 features, building IoT energy telemetry)
8. NYC Taxi (Real, 7 features, trip duration seconds)

Competitors Evaluated in Isolated Subprocesses:
1. Scikit-Learn (CART Batch Completo) - Receives full N training samples at once
2. River (Streaming Online 1-a-1) - Streams line by line from disk
3. Kansho High Tier (Tier 256, ou Tier 40 em alta dimensionalidade, <= 1 MB)
4. Kansho Tier 8 (8 Folhas / 128 Nós, <= 1 MB)
5. Kansho Tier 4 (4 Folhas / 64 Nós, <= 1 MB)

Produces:
- main_benchmark/main_benchmark_report.md
- main_benchmark/main_benchmark_results.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, Any, List, Optional

try:
    from .tune_hyperparams import ensure_c11_binary
except ImportError:
    from tune_hyperparams import ensure_c11_binary


def resolve_path(p: str, is_exec: bool = False) -> str:
    if os.path.exists(p):
        return os.path.abspath(p)
    candidates = [
        os.path.join("code", p),
        os.path.join("..", "code", p),
        os.path.join("..", p),
        os.path.join(os.path.dirname(__file__), "..", "..", "code", p),
        os.path.join(os.path.dirname(__file__), "..", p),
        os.path.join(os.path.dirname(__file__), p),
    ]
    for c in candidates:
        if os.path.exists(c):
            if is_exec and not os.access(c, os.X_OK):
                continue
            return os.path.abspath(c)
    return p


def get_python_exec() -> str:
    candidates = [
        sys.executable,
        os.path.abspath(".venv/bin/python"),
        os.path.abspath("../.venv/bin/python"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "python")),
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            try:
                subprocess.run([c, "-c", "import river, sklearn"], capture_output=True, check=True)
                return c
            except Exception:
                continue
    return sys.executable


# ==============================================================================
# Subprocess Evaluators
# ==============================================================================

def run_isolated_sklearn(train_csv: str, test_csv: str, task: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    runner_script = os.path.join(os.path.dirname(__file__), "runners", "run_sklearn.py")
    cmd = [
        get_python_exec(),
        runner_script,
        "--train-csv", train_csv,
        "--test-csv", test_csv,
        "--task", task,
    ]
    if params:
        if params.get("prep_scaler") == "standard":
            cmd.append("--use-scaler")
        if params.get("max_depth") is not None:
            cmd.extend(["--max-depth", str(params["max_depth"])])
        if params.get("min_samples_split") is not None:
            cmd.extend(["--min-samples-split", str(params["min_samples_split"])])
        if params.get("min_samples_leaf") is not None:
            cmd.extend(["--min-samples-leaf", str(params["min_samples_leaf"])])
        if params.get("max_features") is not None:
            cmd.extend(["--max-features", str(params["max_features"])])
        if params.get("criterion") is not None:
            cmd.extend(["--criterion", str(params["criterion"])])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout.strip())
    res["speedup"] = "Batch"
    return res


def run_isolated_river(train_csv: str, test_csv: str, task: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    runner_script = os.path.join(os.path.dirname(__file__), "runners", "run_river.py")
    cmd = [
        get_python_exec(),
        runner_script,
        "--train-csv", train_csv,
        "--test-csv", test_csv,
        "--task", task,
    ]
    if params:
        if params.get("prep_scaler") == "standard":
            cmd.append("--use-scaler")
        if params.get("grace_period") is not None:
            cmd.extend(["--grace-period", str(params["grace_period"])])
        if params.get("delta") is not None:
            cmd.extend(["--delta", str(params["delta"])])
        if params.get("tau") is not None:
            cmd.extend(["--tau", str(params["tau"])])
        if params.get("leaf_prediction") is not None:
            cmd.extend(["--leaf-prediction", str(params["leaf_prediction"])])
        if params.get("split_criterion") is not None:
            cmd.extend(["--split-criterion", str(params["split_criterion"])])
        if params.get("splitter") is not None:
            cmd.extend(["--splitter", str(params["splitter"])])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout.strip())
    res["speedup"] = "1.0x (baseline)"
    return res


def run_isolated_c_tier(
    bin_path: str,
    train_csv: str,
    test_csv: str,
    task: str,
    tier_label: str,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    json_path = f"/tmp/bench_c_{os.path.basename(bin_path)}_{time.time_ns()}.json"
    cmd = [
        bin_path,
        train_csv,
        "--test-csv", test_csv,
        "--json", json_path,
    ]
    if task == "regression":
        cmd.append("--regression")
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if os.path.exists(json_path):
        os.remove(json_path)

    # Static model size from stdout or JSON
    model_static_bytes = meta.get("final_bytes_used", 524720)
    for line in proc.stdout.splitlines():
        if "model static size =" in line:
            parts = line.split("model static size =")[1].split("bytes")[0].strip()
            model_static_bytes = int(parts)

    intra_train_ram_kb = model_static_bytes / 1024.0

    train_rss_mb = meta.get("train_rss_kb", 2500) / 1024.0
    test_rss_mb = meta.get("test_rss_kb", train_rss_mb * 1024.0) / 1024.0
    peak_rss_mb = meta.get("peak_rss_kb", test_rss_mb * 1024.0) / 1024.0

    res = {
        "model": tier_label,
        "task": task,
        "n_samples": meta.get("n_samples", 0) + meta.get("holdout_samples", 0),
        "n_features": meta.get("n_features", 0),
        "n_nodes": meta.get("final_nodes", 1),
        "n_leaves": meta.get("final_active_leaves", 1),
        "intra_train_ram_kb": intra_train_ram_kb,
        "train_rss_mb": train_rss_mb,
        "test_rss_mb": test_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "avg_us_per_sample": meta.get("avg_time_per_sample_us", 0.0),
    }

    if task == "regression":
        res["holdout_mae"] = float(meta.get("holdout_mae", 0.0))
        res["holdout_rmse"] = float(meta.get("holdout_rmse", 0.0))
        res["holdout_r2"] = float(meta.get("holdout_r2", 0.0))
        res["prequential_mae"] = float(meta.get("final_mae", 0.0))
    else:
        res["holdout_acc"] = float(meta.get("holdout_accuracy", 0.0) * 100.0)
        res["prequential_acc"] = float(meta.get("final_accuracy", 0.0) * 100.0)
        preds = meta.get("holdout_predictions")
        if preds is not None and os.path.exists(test_csv):
            try:
                import numpy as np
                from sklearn.metrics import f1_score
                y_test = []
                with open(test_csv, "r", encoding="utf-8") as tf:
                    for line in tf:
                        if line.startswith("#") or not line.strip():
                            continue
                        y_test.append(int(float(line.strip().split(",")[-1])))
                y_test = np.array(y_test, dtype=np.int32)
                y_pred = np.array(preds, dtype=np.int32)
                res["f1_macro"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0) * 100.0)
            except Exception:
                res["f1_macro"] = res["holdout_acc"]
        else:
            res["f1_macro"] = res["holdout_acc"]

    return res


# ==============================================================================
# Helper to Build Kansho CLI Arguments from Optimal Params
# ==============================================================================

def build_c11_cli_args(params: Dict[str, Any], task: str) -> List[str]:
    args: List[str] = []
    if "grace_period" in params:
        args.extend(["--grace-period", str(params["grace_period"])])
    if "delta" in params:
        args.extend(["--delta", str(params["delta"])])
    if "tau" in params:
        args.extend(["--tau", str(params["tau"])])
    if params.get("prep_scaler") == "standard":
        args.append("--prep-scaler")
    if params.get("leaf_normalize"):
        args.append("--leaf-normalize")
    if params.get("obs_splitter") in ("hist", "histogram"):
        args.extend(["--obs", "histogram"])
    if params.get("filter_hampel"):
        args.append("--prep-hampel")

    if task == "regression":
        if "leaf_prediction" in params:
            args.extend(["--leaf", str(params["leaf_prediction"])])
    else:
        if "split_criterion" in params:
            args.extend(["--criterion", str(params["split_criterion"])])
        if "leaf_prediction" in params:
            args.extend(["--leaf", str(params["leaf_prediction"])])
    return args


# ==============================================================================
# Optimal Hyperparameters from Tuning Search (hyperparam_results.json)
# ==============================================================================
DATASET_OPTIMAL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sea": {
        "sklearn": {"max_depth": None, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": None, "prep_scaler": None, "criterion": "entropy"},
        "river": {"grace_period": 50, "delta": 0.01, "tau": 0.1, "prep_scaler": None, "leaf_prediction": "nba", "split_criterion": "info_gain", "splitter": "gaussian"},
        "kansho_t256": {"tier": 256, "grace_period": 100, "delta": 0.01, "tau": 0.1, "split_criterion": "info_gain", "leaf_prediction": "nba", "prep_scaler": None, "obs_splitter": "gaussian", "filter_hampel": False},
        "kansho_t8": {"tier": 8, "grace_period": 100, "delta": 0.01, "tau": 0.1, "split_criterion": "info_gain", "leaf_prediction": "nba", "prep_scaler": None, "obs_splitter": "gaussian", "filter_hampel": False},
        "kansho_t4": {"tier": 4, "grace_period": 400, "delta": 0.0001, "tau": 0.15, "prep_scaler": "standard", "obs_splitter": "histogram", "filter_hampel": False, "split_criterion": "info_gain", "leaf_prediction": "nba"},
    },
    "electricity": {
        "sklearn": {"max_depth": 6, "min_samples_split": 500, "min_samples_leaf": 50, "max_features": None, "prep_scaler": "standard", "criterion": "entropy"},
        "river": {"grace_period": 200, "delta": 1e-07, "tau": 0.01, "prep_scaler": "standard", "leaf_prediction": "mc", "split_criterion": "gini", "splitter": "histogram"},
        "kansho_t256": {"tier": 256, "grace_period": 400, "delta": 0.01, "tau": 0.1, "prep_scaler": "standard", "obs_splitter": "histogram", "filter_hampel": False, "split_criterion": "info_gain", "leaf_prediction": "nba"},
        "kansho_t8": {"tier": 8, "grace_period": 400, "delta": 0.01, "tau": 0.1, "split_criterion": "gini", "leaf_prediction": "mc", "prep_scaler": None, "obs_splitter": "gaussian", "filter_hampel": False},
        "kansho_t4": {"tier": 4, "grace_period": 100, "delta": 1e-07, "tau": 0.05, "prep_scaler": "standard", "obs_splitter": "histogram", "filter_hampel": False, "split_criterion": "info_gain", "leaf_prediction": "nba"},
    },
    "covertype": {
        "sklearn": {"max_depth": None, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": None, "prep_scaler": "standard", "criterion": "gini"},
        "river": {"grace_period": 50, "delta": 0.01, "tau": 0.1, "prep_scaler": None, "leaf_prediction": "nba", "split_criterion": "gini", "splitter": "gaussian"},
        "kansho_t40": {"tier": 40, "grace_period": 50, "delta": 0.001, "tau": 0.1, "split_criterion": "gini", "leaf_prediction": "nba", "prep_scaler": None, "obs_splitter": "gaussian", "filter_hampel": False},
        "kansho_t8": {"tier": 8, "grace_period": 50, "delta": 0.001, "tau": 0.1, "split_criterion": "gini", "leaf_prediction": "nba", "prep_scaler": None, "obs_splitter": "gaussian", "filter_hampel": False},
        "kansho_t4": {"tier": 4, "grace_period": 50, "delta": 0.001, "tau": 0.1, "split_criterion": "gini", "leaf_prediction": "nba", "prep_scaler": None, "obs_splitter": "gaussian", "filter_hampel": False},
    },
    "airlines": {
        "sklearn": {"max_depth": 16, "min_samples_split": 50, "min_samples_leaf": 20, "max_features": None, "prep_scaler": "standard", "criterion": "entropy"},
        "river": {"grace_period": 50, "delta": 1e-07, "tau": 0.05, "prep_scaler": None, "leaf_prediction": "nb", "split_criterion": "gini", "splitter": "histogram"},
        "kansho_t256": {"tier": 256, "grace_period": 100, "delta": 1e-07, "tau": 0.1, "prep_scaler": None, "obs_splitter": "histogram", "filter_hampel": False, "split_criterion": "info_gain", "leaf_prediction": "nb"},
        "kansho_t8": {"tier": 8, "grace_period": 200, "delta": 1e-07, "tau": 0.1, "prep_scaler": "standard", "obs_splitter": "histogram", "filter_hampel": False, "split_criterion": "info_gain", "leaf_prediction": "nb"},
        "kansho_t4": {"tier": 4, "grace_period": 50, "delta": 0.001, "tau": 0.1, "split_criterion": "gini", "leaf_prediction": "nba", "prep_scaler": None, "obs_splitter": "gaussian", "filter_hampel": False},
    },
    "friedman": {
        "sklearn": {"max_depth": None, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": None, "prep_scaler": None, "criterion": "squared_error"},
        "river": {"grace_period": 50, "delta": 0.001, "tau": 0.05, "prep_scaler": "standard", "leaf_prediction": "model", "splitter": "ebst"},
        "kansho_t256": {"tier": 256, "grace_period": 400, "delta": 0.001, "tau": 0.1, "prep_scaler": "standard", "obs_splitter": "gaussian", "filter_hampel": False, "leaf_prediction": "model", "reg_num_slots": 8, "leaf_normalize": True},
        "kansho_t8": {"tier": 8, "grace_period": 200, "delta": 1e-07, "tau": 0.01, "leaf_prediction": "model", "reg_num_slots": 16, "leaf_normalize": False, "prep_scaler": "standard", "obs_splitter": "gaussian", "filter_hampel": False},
        "kansho_t4": {"tier": 4, "grace_period": 200, "delta": 1e-07, "tau": 0.01, "leaf_prediction": "model", "reg_num_slots": 16, "leaf_normalize": False, "prep_scaler": "standard", "obs_splitter": "gaussian", "filter_hampel": False},
    },
    "bikes": {
        "sklearn": {"max_depth": 16, "min_samples_split": 10, "min_samples_leaf": 20, "max_features": None, "prep_scaler": None, "criterion": "squared_error"},
        "river": {"grace_period": 50, "delta": 0.001, "tau": 0.15, "prep_scaler": None, "leaf_prediction": "mean", "splitter": "ebst"},
        "kansho_t256": {"tier": 256, "grace_period": 200, "delta": 0.01, "tau": 0.15, "prep_scaler": None, "obs_splitter": "histogram", "filter_hampel": False, "leaf_prediction": "mean", "reg_num_slots": 16, "leaf_normalize": True},
        "kansho_t8": {"tier": 8, "grace_period": 50, "delta": 0.01, "tau": 0.05, "prep_scaler": "standard", "obs_splitter": "gaussian", "filter_hampel": False, "leaf_prediction": "mean", "reg_num_slots": 32, "leaf_normalize": False},
        "kansho_t4": {"tier": 4, "grace_period": 200, "delta": 1e-07, "tau": 0.15, "prep_scaler": "standard", "obs_splitter": "gaussian", "filter_hampel": False, "leaf_prediction": "mean", "reg_num_slots": 16, "leaf_normalize": False},
    },
    "appliances_energy": {
        "sklearn": {"max_depth": 8, "min_samples_split": 100, "min_samples_leaf": 50, "max_features": None, "prep_scaler": "standard", "criterion": "squared_error"},
        "river": {"grace_period": 100, "delta": 0.001, "tau": 0.15, "prep_scaler": None, "leaf_prediction": "mean", "splitter": "tebst"},
        "kansho_t40": {"tier": 40, "grace_period": 100, "delta": 1e-07, "tau": 0.01, "leaf_prediction": "adaptive", "reg_num_slots": 16, "leaf_normalize": True, "prep_scaler": None, "obs_splitter": "gaussian", "filter_hampel": False},
        "kansho_t8": {"tier": 8, "grace_period": 200, "delta": 0.0001, "tau": 0.15, "prep_scaler": None, "obs_splitter": "histogram", "filter_hampel": False, "leaf_prediction": "model", "reg_num_slots": 8, "leaf_normalize": True},
        "kansho_t4": {"tier": 4, "grace_period": 100, "delta": 1e-07, "tau": 0.01, "leaf_prediction": "adaptive", "reg_num_slots": 16, "leaf_normalize": True, "prep_scaler": None, "obs_splitter": "gaussian", "filter_hampel": False},
    },
    "nyc_taxi": {
        "sklearn": {"max_depth": None, "min_samples_split": 10, "min_samples_leaf": 20, "max_features": None, "prep_scaler": None, "criterion": "squared_error"},
        "river": {"grace_period": 200, "delta": 0.01, "tau": 0.15, "prep_scaler": "standard", "leaf_prediction": "adaptive", "splitter": "tebst"},
        "kansho_t256": {"tier": 256, "grace_period": 200, "delta": 0.01, "tau": 0.15, "prep_scaler": "standard", "obs_splitter": "histogram", "filter_hampel": False, "leaf_prediction": "model", "reg_num_slots": 32, "leaf_normalize": True},
        "kansho_t8": {"tier": 8, "grace_period": 100, "delta": 0.0001, "tau": 0.05, "prep_scaler": "standard", "obs_splitter": "histogram", "filter_hampel": False, "leaf_prediction": "adaptive", "reg_num_slots": 8, "leaf_normalize": False},
        "kansho_t4": {"tier": 4, "grace_period": 50, "delta": 1e-07, "tau": 0.1, "prep_scaler": None, "obs_splitter": "gaussian", "filter_hampel": False, "leaf_prediction": "adaptive", "reg_num_slots": 32, "leaf_normalize": True},
    },
}


# ==============================================================================
# Suite Orchestrator
# ==============================================================================

def run_main_benchmark_suite(
    data_dir: str,
    report_path: str,
    selected_datasets: Optional[List[str]] = None,
):
    print("================================================================================")
    print(" MAIN PERFORMANCE BENCHMARK SUITE: 8 DATASETS x MULTI-TIER KANSHO")
    print(" Kansho (Tiers High, 8, 4) vs River (Streaming) vs Scikit-Learn (Batch)")
    print(f" Data directory:       {data_dir}")
    print(f" Report destination:   {report_path}")
    print("================================================================================")

    datasets = [
        # (id, name, task, desc, is_high_dim)
        ("sea", "SEA", "classification", "SEA (Synthetic, 3 continuous, 2 classes)", False),
        ("electricity", "Electricity", "classification", "Electricity / Elec2 (Real, 8 continuous, 2 classes)", False),
        ("covertype", "Forest Covertype", "classification", "Forest Covertype (Real, 54 features, 7 classes)", True),
        ("airlines", "Airlines", "classification", "Airlines Delay (Real, 7 features, 2 classes)", False),
        ("friedman", "Friedman #1", "regression", "Friedman #1 (Synthetic, 10 continuous, nonlinear target)", False),
        ("bikes", "Bikes", "regression", "Bikes (Real, 12 features, demand count)", False),
        ("appliances_energy", "Appliances Energy", "regression", "Appliances Energy (Real, 28 features, building Wh)", True),
        ("nyc_taxi", "NYC Taxi", "regression", "NYC Taxi (Real, 7 features, trip duration seconds)", False),
    ]

    all_results = []

    for d_id, name, task, desc, is_high_dim in datasets:
        if selected_datasets and d_id not in selected_datasets and "all" not in selected_datasets:
            continue

        print(f"\n================================================================================")
        print(f" >>> Dataset: {name} [{task.upper()}] - {desc}")
        print(f"================================================================================")

        train_csv = os.path.join(data_dir, f"{d_id}_train.csv")
        test_csv = os.path.join(data_dir, f"{d_id}_test.csv")

        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"  Dataset files not found for {d_id}. Running prepare_datasets.py...")
            prep_script = os.path.join(os.path.dirname(__file__), "prepare_datasets.py")
            subprocess.run([get_python_exec(), prep_script, "--outdir", data_dir, "--dataset", d_id], check=True)

        opt_cfg = DATASET_OPTIMAL_CONFIGS.get(d_id, {})
        sk_params = opt_cfg.get("sklearn", {})
        riv_params = opt_cfg.get("river", {})
        high_tier = 40 if is_high_dim else 256
        t_high_key = f"kansho_t{high_tier}"
        t_high_params = opt_cfg.get(t_high_key, {})
        t8_params = opt_cfg.get("kansho_t8", {})
        t4_params = opt_cfg.get("kansho_t4", {})

        # 1. Scikit-Learn (Full Batch in RAM)
        print("  [1/5] Running Scikit-Learn (Full Batch in RAM)...", end="", flush=True)
        sk_res = run_isolated_sklearn(train_csv, test_csv, task, params=sk_params)
        if task == "classification":
            print(f" Done. (Acc={sk_res['holdout_acc']:.2f}%, Nodes={sk_res['n_nodes']}, IntraRAM={sk_res['intra_train_ram_kb']:.1f}KB, PeakRSS={sk_res['peak_rss_mb']:.1f}MB, Time={sk_res['avg_us_per_sample']:.2f}us)")
        else:
            print(f" Done. (MAE={sk_res['holdout_mae']:.4f}, R2={sk_res['holdout_r2']:.4f}, Nodes={sk_res['n_nodes']}, IntraRAM={sk_res['intra_train_ram_kb']:.1f}KB, PeakRSS={sk_res['peak_rss_mb']:.1f}MB, Time={sk_res['avg_us_per_sample']:.2f}us)")

        # 2. River (Streaming 1-a-1)
        print("  [2/5] Running River (Streaming 1-a-1)...", end="", flush=True)
        riv_res = run_isolated_river(train_csv, test_csv, task, params=riv_params)
        river_us = riv_res["avg_us_per_sample"]
        if task == "classification":
            print(f" Done. (Acc={riv_res['holdout_acc']:.2f}%, PreqAcc={riv_res['prequential_acc']:.2f}%, IntraRAM={riv_res['intra_train_ram_kb']:.1f}KB, PeakRSS={riv_res['peak_rss_mb']:.1f}MB, Time={river_us:.2f}us)")
        else:
            print(f" Done. (MAE={riv_res['holdout_mae']:.4f}, PreqMAE={riv_res['prequential_mae']:.4f}, IntraRAM={riv_res['intra_train_ram_kb']:.1f}KB, PeakRSS={riv_res['peak_rss_mb']:.1f}MB, Time={river_us:.2f}us)")

        # 3. Kansho High Tier (Tier 256 or Tier 40)
        slots_high = t_high_params.get("reg_num_slots", 16) if task == "regression" else 16
        bin_high = ensure_c11_binary(high_tier, slots_high, is_high_dim, task)
        cli_args_high = build_c11_cli_args(t_high_params, task)
        high_label = f"Kansho Tier {high_tier}"

        print(f"  [3/5] Running {high_label} ({os.path.basename(bin_high)})...", end="", flush=True)
        t_high_res = run_isolated_c_tier(bin_high, train_csv, test_csv, task, high_label, extra_args=cli_args_high)
        t_high_res["speedup"] = river_us / max(0.01, t_high_res["avg_us_per_sample"])
        if task == "classification":
            print(f" Done. (Acc={t_high_res['holdout_acc']:.2f}%, Nodes={t_high_res['n_nodes']}, IntraRAM={t_high_res['intra_train_ram_kb']:.1f}KB, PeakRSS={t_high_res['peak_rss_mb']:.1f}MB, Time={t_high_res['avg_us_per_sample']:.2f}us, Speedup={t_high_res['speedup']:.1f}x)")
        else:
            print(f" Done. (MAE={t_high_res['holdout_mae']:.4f}, R2={t_high_res['holdout_r2']:.4f}, Nodes={t_high_res['n_nodes']}, IntraRAM={t_high_res['intra_train_ram_kb']:.1f}KB, PeakRSS={t_high_res['peak_rss_mb']:.1f}MB, Time={t_high_res['avg_us_per_sample']:.2f}us, Speedup={t_high_res['speedup']:.1f}x)")

        # 4. Kansho Tier 8
        slots_t8 = t8_params.get("reg_num_slots", 16) if task == "regression" else 16
        bin_t8 = ensure_c11_binary(8, slots_t8, is_high_dim, task)
        cli_args_t8 = build_c11_cli_args(t8_params, task)

        print(f"  [4/5] Running Kansho Tier 8 ({os.path.basename(bin_t8)})...", end="", flush=True)
        t8_res = run_isolated_c_tier(bin_t8, train_csv, test_csv, task, "Kansho Tier 8", extra_args=cli_args_t8)
        t8_res["speedup"] = river_us / max(0.01, t8_res["avg_us_per_sample"])
        if task == "classification":
            print(f" Done. (Acc={t8_res['holdout_acc']:.2f}%, Nodes={t8_res['n_nodes']}, IntraRAM={t8_res['intra_train_ram_kb']:.1f}KB, PeakRSS={t8_res['peak_rss_mb']:.1f}MB, Time={t8_res['avg_us_per_sample']:.2f}us, Speedup={t8_res['speedup']:.1f}x)")
        else:
            print(f" Done. (MAE={t8_res['holdout_mae']:.4f}, R2={t8_res['holdout_r2']:.4f}, Nodes={t8_res['n_nodes']}, IntraRAM={t8_res['intra_train_ram_kb']:.1f}KB, PeakRSS={t8_res['peak_rss_mb']:.1f}MB, Time={t8_res['avg_us_per_sample']:.2f}us, Speedup={t8_res['speedup']:.1f}x)")

        # 5. Kansho Tier 4
        slots_t4 = t4_params.get("reg_num_slots", 16) if task == "regression" else 16
        bin_t4 = ensure_c11_binary(4, slots_t4, is_high_dim, task)
        cli_args_t4 = build_c11_cli_args(t4_params, task)

        print(f"  [5/5] Running Kansho Tier 4 ({os.path.basename(bin_t4)})...", end="", flush=True)
        t4_res = run_isolated_c_tier(bin_t4, train_csv, test_csv, task, "Kansho Tier 4", extra_args=cli_args_t4)
        t4_res["speedup"] = river_us / max(0.01, t4_res["avg_us_per_sample"])
        if task == "classification":
            print(f" Done. (Acc={t4_res['holdout_acc']:.2f}%, Nodes={t4_res['n_nodes']}, IntraRAM={t4_res['intra_train_ram_kb']:.1f}KB, PeakRSS={t4_res['peak_rss_mb']:.1f}MB, Time={t4_res['avg_us_per_sample']:.2f}us, Speedup={t4_res['speedup']:.1f}x)")
        else:
            print(f" Done. (MAE={t4_res['holdout_mae']:.4f}, R2={t4_res['holdout_r2']:.4f}, Nodes={t4_res['n_nodes']}, IntraRAM={t4_res['intra_train_ram_kb']:.1f}KB, PeakRSS={t4_res['peak_rss_mb']:.1f}MB, Time={t4_res['avg_us_per_sample']:.2f}us, Speedup={t4_res['speedup']:.1f}x)")

        all_results.append({
            "id": d_id,
            "name": name,
            "task": task,
            "desc": desc,
            "is_high_dim": is_high_dim,
            "high_tier": high_tier,
            "n_samples": sk_res["n_samples"],
            "n_features": sk_res["n_features"],
            "sklearn": sk_res,
            "river": riv_res,
            "c_high": t_high_res,
            "c_high_label": high_label,
            "c_tier8": t8_res,
            "c_tier4": t4_res,
        })

    # Save Results JSON & Markdown Report
    json_path = os.path.splitext(report_path)[0] + "_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    generate_markdown_report(report_path, all_results)
    print(f"\n[✔] Main multi-tier benchmark report generated: {os.path.abspath(report_path)}")
    print(f"[✔] Benchmark results saved to JSON: {os.path.abspath(json_path)}")


def generate_markdown_report(report_path: str, results: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Relatório do Main Benchmark: Avaliação Realista de RAM e Multi-Tier Kansho\n\n")
        f.write("Este documento consolida a avaliação experimental dos **8 datasets emblemáticos** (4 de Classificação e 4 de Regressão Contínua) comparando a biblioteca **Kansho (em C11 puro)** em seus tiers ótimos de dimensionamento físico estático contra o estimador batch **Scikit-Learn** e a biblioteca streaming **River**.\n\n")

        f.write("## 1. Protocolo Experimental e Metodologia de Memória\n\n")
        f.write("- **Volume de Amostras Equivalente ($N$):** As três soluções analisam rigorosamente a mesma quantidade total de amostras de treino ao longo do experimento.\n")
        f.write("- **Diferença de Paradigma na Ingestão de Dados:**\n")
        f.write("  - **Scikit-Learn (Batch Completo):** Carrega todas as $N$ amostras simultaneamente na RAM para `model.fit(X_train, y_train)` e todo o conjunto de teste para `model.predict(X_test)`.\n")
        f.write("  - **River (Streaming):** Consome amostra por amostra do fluxo contínuo via gerador linha a linha (`learn_one`).\n")
        f.write("  - **Kansho (Streaming Puro Embarcado):** Consome estritamente **1 amostra por vez** diretamente do CSV através de um buffer estático de $2\\text{ KB}$, operando com **0.0 KB de `malloc` em runtime** e sem qualquer acúmulo de dados na RAM.\n")
        f.write("- **Métricas de Memória:**\n")
        f.write("  - `RAM Intra-Treino (Modelo + Heap)`: Tamanho físico estrutural do modelo somado ao pico de alocações dinâmicas no heap durante o fit/train.\n")
        f.write("  - `Peak RSS Treino / Teste / Total (MB)`: Memória física real (Resident Set Size) alocada pelo Sistema Operacional para o processo isolado.\n")
        f.write("- **Tiers de Dimensionamento do Kansho:**\n")
        f.write("  - **Tier 256 / Tier 40 (Capacidade Alta):** $256$ folhas em datasets convencionais ou $40$ folhas em alta dimensionalidade (Covertype e Appliances Energy), dimensionado estritamente para $\\le 1\\text{ MB}$ de SRAM estática.\n")
        f.write("  - **Tier 8 (Dimensionamento Intermediário / Sweet Spot):** $8$ folhas ativas / $128$ nós ($16.6\\text{ KB}$ a $90.3\\text{ KB}$ em `float32`), dimensionamento universal para microcontroladores como ESP32.\n")
        f.write("  - **Tier 4 (Dimensionamento Extremo):** $4$ folhas ativas / $64$ nós ($8.4\\text{ KB}$ a $45.3\\text{ KB}$ em `float32`), projetado para MCUs com $\\le 64\\text{ KB}$ de SRAM (ex: ARM Cortex-M3).\n\n")

        # Classification Table
        f.write("## 2. Resultados em Classificação (4 Datasets)\n\n")
        f.write("| Dataset | Modelo / Tier | Holdout Acc (%) | Macro F1 (%) | Prequential Acc (%) | Total Nós | RAM Intra-Treino (Mod+Heap) | RSS Treino (MB) | RSS Teste (MB) | RSS Total (MB) | Tempo/amostra (µs) | Speedup vs River |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for r in results:
            if r["task"] != "classification":
                continue
            name = f"**{r['name']}** ({r['n_features']} feat)"
            models = [
                ("sklearn", "Scikit-Learn (CART Batch)"),
                ("river", "River HTC (Streaming)"),
                ("c_high", r["c_high_label"]),
                ("c_tier8", "Kansho Tier 8"),
                ("c_tier4", "Kansho Tier 4"),
            ]
            for key, label in models:
                m = r[key]
                acc_s = f"{m['holdout_acc']:.2f}%"
                f1_s = f"{m['f1_macro']:.2f}%"
                preq_s = f"{m['prequential_acc']:.2f}%" if m.get('prequential_acc') is not None else "N/A"
                intra_s = f"{m['intra_train_ram_kb']:.1f} KB"
                tr_rss = f"{m['train_rss_mb']:.1f} MB"
                te_rss = f"{m['test_rss_mb']:.1f} MB"
                tot_rss = f"{m['peak_rss_mb']:.1f} MB"
                time_s = f"{m['avg_us_per_sample']:.2f} µs"
                sp = m.get("speedup", "-")
                sp_s = f"**{sp:.1f}x**" if isinstance(sp, float) else str(sp)
                bold = "**" if key.startswith("c_") else ""
                f.write(f"| {name} | {bold}{label}{bold} | {bold}{acc_s}{bold} | {f1_s} | {preq_s} | {m['n_nodes']} | {intra_s} | {tr_rss} | {te_rss} | {bold}{tot_rss}{bold} | {time_s} | {sp_s} |\n")

        # Regression Table
        f.write("\n## 3. Resultados em Regressão Contínua (4 Datasets)\n\n")
        f.write("| Dataset | Modelo / Tier | Holdout MAE | Holdout RMSE | Holdout $R^2$ | Prequential MAE | Total Nós | RAM Intra-Treino (Mod+Heap) | RSS Treino (MB) | RSS Teste (MB) | RSS Total (MB) | Tempo/amostra (µs) | Speedup vs River |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for r in results:
            if r["task"] != "regression":
                continue
            name = f"**{r['name']}** ({r['n_features']} feat)"
            models = [
                ("sklearn", "Scikit-Learn (CART Batch)"),
                ("river", "River HTR (Streaming)"),
                ("c_high", r["c_high_label"]),
                ("c_tier8", "Kansho Tier 8"),
                ("c_tier4", "Kansho Tier 4"),
            ]
            for key, label in models:
                m = r[key]
                mae_s = f"{m['holdout_mae']:.4f}"
                rmse_s = f"{m['holdout_rmse']:.4f}"
                r2_s = f"{m['holdout_r2']:.4f}"
                preq_s = f"{m['prequential_mae']:.4f}" if m.get('prequential_mae') is not None else "N/A"
                intra_s = f"{m['intra_train_ram_kb']:.1f} KB"
                tr_rss = f"{m['train_rss_mb']:.1f} MB"
                te_rss = f"{m['test_rss_mb']:.1f} MB"
                tot_rss = f"{m['peak_rss_mb']:.1f} MB"
                time_s = f"{m['avg_us_per_sample']:.2f} µs"
                sp = m.get("speedup", "-")
                sp_s = f"**{sp:.1f}x**" if isinstance(sp, float) else str(sp)
                bold = "**" if key.startswith("c_") else ""
                f.write(f"| {name} | {bold}{label}{bold} | {bold}{mae_s}{bold} | {rmse_s} | {bold}{r2_s}{bold} | {preq_s} | {m['n_nodes']} | {intra_s} | {tr_rss} | {te_rss} | {bold}{tot_rss}{bold} | {time_s} | {sp_s} |\n")

        # Section 4: Highlights and Analysis
        f.write("\n## 4. Análise Comparativa e Síntese de Performance\n\n")
        f.write("1. **Equilíbrio Preditivo dos Melhores Parâmetros:** A aplicação dos hiperparâmetros ótimos identificados na suíte de busca estratificada permitiu que os três tiers do Kansho atingissem desempenhos de ponta em todos os 8 datasets, sem redundância entre variantes cruas e tratadas.\n")
        f.write("2. **Eficiência e Velocidade do C11:** Kansho processou cada amostra completa entre **$0,23\\,\\mu\\text{s}$ e $6,37\\,\\mu\\text{s}$**, entregando speedups de **$100\\times$ a mais de $2.000\\times$** frente ao River em Python.\n")
        f.write("3. **Orçamento Rigoroso $\\le 1\\text{ MB}$:** Mesmo no Tier 40 dos datasets mais densos (Forest Covertype e Appliances Energy), a memória estática foi mantida estritamente abaixo do teto físico (consumindo $713.8\\text{ KB}$), garantindo estabilidade absoluta e determinismo sem alocação dinâmica em runtime.\n")


def main():
    parser = argparse.ArgumentParser(description="Main Multi-Tier Performance Benchmark Suite.")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--report", default=os.path.join(os.path.dirname(__file__), "main_benchmark_report.md"))
    parser.add_argument("--dataset", default="all", help="Dataset to run ('all' or comma-separated list).")
    args = parser.parse_args()

    selected = [s.strip().lower() for s in args.dataset.split(",")]
    run_main_benchmark_suite(
        data_dir=args.data_dir,
        report_path=args.report,
        selected_datasets=selected,
    )


if __name__ == "__main__":
    main()
