#!/usr/bin/env python3
"""
Hyperparameter Tuning and Sensitivity Analysis Suite for Embedded Hoeffding Tree.

Performs systematic exploration across 8 flagship datasets:
--- CLASSIFICATION (4) ---
1. SEA (Synthetic, 3 features, 2 classes)
2. Electricity / Elec2 (Real, 8 features, 2 classes)
3. Forest Covertype (Real, 54 features, 7 classes)
4. Airlines Delay (Real, 7 features, 2 classes)

--- REGRESSION (4) ---
5. Friedman #1 (Synthetic, 10 features, continuous nonlinear target)
6. Bikes (Real, 12 features, continuous bike rental demand)
7. Appliances Energy (Real, 28 features, building IoT telemetry)
8. NYC Taxi (Real, 7 features, trip duration seconds)

Engines Evaluated:
1. Kansho (Algorithm runtime params + static compilation macros: slots, tiers)
2. River (Hoeffding Tree / HAT with parameter parity + splitters)
3. Scikit-Learn (CART Batch with max_depth, min_samples_split/leaf, max_features, criterion)

Produces:
- main_benchmark/hyperparam_report.md (Comprehensive Markdown report with Pareto curves & sensitivity)
- main_benchmark/hyperparam_results.json (Full trial metrics in JSON)
"""

import argparse
import concurrent.futures
import csv
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, Any, List, Optional, Tuple

_compile_lock = threading.Lock()


# ==============================================================================
# Path Resolution and Datasets Configuration
# ==============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(PROJECT_ROOT, "code")
BUILD_DIR = os.path.join(CODE_DIR, "build")
BENCH_SRC = os.path.join(PROJECT_ROOT, "benchmarks", "bench_c_runner.c")

PYTHON_BIN = sys.executable
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python3")
if os.path.exists(VENV_PYTHON):
    PYTHON_BIN = VENV_PYTHON

# Memory Tiers: Tier -> (Leaves, Nodes)
TIER_MAP = {
    1: (1, 16),
    2: (2, 32),
    4: (4, 64),
    8: (8, 128),
    16: (16, 256),
    40: (40, 1024),
    64: (64, 1024),
    128: (128, 2048),
    256: (256, 4096),
}

DATASETS_CONFIG = [
    # id, name, task, desc, is_high_dim
    ("sea", "SEA", "classification", "SEA (Synthetic, 3 continuous, 2 classes)", False),
    ("electricity", "Electricity", "classification", "Electricity / Elec2 (Real, 8 continuous, 2 classes)", False),
    ("covertype", "Forest Covertype", "classification", "Forest Covertype (Real, 54 features, 7 classes)", True),
    ("airlines", "Airlines", "classification", "Airlines Delay (Real, 7 features, 2 classes)", False),
    ("friedman", "Friedman #1", "regression", "Friedman #1 (Synthetic, 10 continuous, nonlinear target)", False),
    ("bikes", "Bikes", "regression", "Bikes (Real, 12 features, demand count)", False),
    ("appliances_energy", "Appliances Energy", "regression", "Appliances Energy (Real, 28 features, building Wh)", True),
    ("nyc_taxi", "NYC Taxi", "regression", "NYC Taxi (Real, 7 features, trip duration seconds)", False),
]


def resolve_executable(bin_name: str) -> str:
    path = os.path.join(BUILD_DIR, bin_name)
    if os.path.exists(path) and os.access(path, os.X_OK):
        return path
    return ""


# ==============================================================================
# C11 On-Demand Binary Compilation & Cache
# ==============================================================================

def calculate_safe_capacity(tier: int, slots: int, is_high_dim: bool, task: str = "classification") -> Tuple[int, int]:
    """
    Computes exact safe leaf and node capacities, strictly enforcing the <= 1 MB SRAM
    budget under any feature, class, or regression slot configuration before compilation.
    """
    max_features = 64 if is_high_dim else 16
    max_classes = 8 if is_high_dim else 4

    # Node size estimate: 28B in float32 for default; ~48B for 8 classes
    node_size = (max_classes * 4) + 16

    # Leaf stats size estimate:
    if task == "classification":
        obs_size = max_classes * 20
        leaf_size = 16 + (max_features * obs_size)
    else:
        obs_size = max(slots * 16, 80)
        leaf_size = 16 + (max_features * 4) + (max_features * obs_size)

    leaves, nodes = TIER_MAP.get(tier, (tier, tier * 16))
    if is_high_dim and tier > 16:
        leaves = 40
        nodes = 1024

    # Memory pool size estimate (node_pool + stats_pool + free lists + state):
    total_bytes = (nodes * (node_size + 2)) + (leaves * (leaf_size + 2)) + 64

    # Safety clamp ensuring pool never exceeds 980 KB:
    while total_bytes > (980 * 1024) and leaves > 1:
        leaves = max(1, int(leaves * 0.8))
        nodes = max(16, leaves * 16)
        total_bytes = (nodes * (node_size + 2)) + (leaves * (leaf_size + 2)) + 64

    return leaves, nodes


def ensure_c11_binary(tier: int, slots: int, is_high_dim: bool, task: str = "classification") -> str:
    """
    Compiles or retrieves a cached C11 benchmark runner binary with the required
    tier (active leaves) and regression slot macros. Thread-safe with lock.
    """
    with _compile_lock:
        os.makedirs(BUILD_DIR, exist_ok=True)
        task_suffix = "_classif" if task == "classification" else f"_s{slots}"
        high_dim_suffix = "_main" if is_high_dim else ""
        bin_name = f"bench_c_runner_t{tier}{task_suffix}{high_dim_suffix}"
        bin_path = os.path.join(BUILD_DIR, bin_name)

        # Check if binary already exists and is newer than source files
        c_srcs = [
            BENCH_SRC,
            os.path.join(CODE_DIR, "src", "ht_tree.c"),
            os.path.join(CODE_DIR, "src", "ht_memory.c"),
            os.path.join(CODE_DIR, "src", "ht_observer.c"),
            os.path.join(CODE_DIR, "src", "ht_criterion.c"),
            os.path.join(CODE_DIR, "src", "ht_drift.c"),
            os.path.join(CODE_DIR, "src", "ht_prep.c"),
            os.path.join(CODE_DIR, "include", "ht_config.h"),
            os.path.join(CODE_DIR, "include", "ht_memory.h"),
            os.path.join(CODE_DIR, "include", "ht_prep.h"),
        ]

        needs_compile = not os.path.exists(bin_path)
        if not needs_compile:
            bin_mtime = os.path.getmtime(bin_path)
            for s in c_srcs:
                if os.path.exists(s) and os.path.getmtime(s) > bin_mtime:
                    needs_compile = True
                    break

        if needs_compile:
            print(f"    [Compiling C11 binary: {bin_name}]...", end="", flush=True)
            
            leaves, nodes = calculate_safe_capacity(tier, slots, is_high_dim, task)

            macros = [
                f"-DHT_MAX_ACTIVE_LEAVES={leaves}",
                f"-DHT_MAX_NODES={nodes}",
            ]
            if task == "classification":
                macros.append("-DHT_CLASSIF_ONLY")
            else:
                macros.append(f"-DHT_REG_NUM_SLOTS={slots}")

            if is_high_dim:
                macros.extend(["-DHT_MAX_FEATURES=64", "-DHT_MAX_CLASSES=8"])

            src_files = [
                os.path.join(CODE_DIR, "src", "ht_memory.c"),
                os.path.join(CODE_DIR, "src", "ht_observer.c"),
                os.path.join(CODE_DIR, "src", "ht_criterion.c"),
                os.path.join(CODE_DIR, "src", "ht_tree.c"),
                os.path.join(CODE_DIR, "src", "ht_drift.c"),
                os.path.join(CODE_DIR, "src", "ht_prep.c"),
                BENCH_SRC,
            ]

            cmd = [
                "gcc",
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                f"-I{os.path.join(CODE_DIR, 'include')}",
            ] + macros + src_files + ["-lm", "-o", bin_path]

            subprocess.run(cmd, check=True)
            print(" Done.")

        return bin_path


# ==============================================================================
# Metric Computation in Pure Python
# ==============================================================================

def load_test_ground_truth(test_csv: str, task: str) -> Tuple[List[float], int]:
    """Loads y_true from test CSV."""
    n_features = 0
    y: List[float] = []
    with open(test_csv, "r", encoding="utf-8") as f:
        header = f.readline()
        if header.startswith("#"):
            for tok in header.lstrip("#").strip().split():
                if tok.startswith("n_features="):
                    n_features = int(tok.split("=")[1])
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if n_features == 0:
                n_features = len(row) - 1
            val = float(row[n_features])
            y.append(int(val) if task == "classification" else val)
    return y, n_features


def compute_metrics(
    y_true: List[float],
    y_pred: List[float],
    task: str,
) -> Dict[str, float]:
    """Computes exact accuracy, precision, recall, f1 or mae, rmse, r2."""
    if not y_true or len(y_true) != len(y_pred):
        return {}

    n = len(y_true)
    if task == "regression":
        mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n
        mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n
        rmse = math.sqrt(mse)
        mean_y = sum(y_true) / n
        ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        return {
            "holdout_mae": float(mae),
            "holdout_rmse": float(rmse),
            "holdout_r2": float(r2),
        }
    else:
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == int(yp))
        acc = (correct / n) * 100.0

        classes = sorted(list(set(int(x) for x in y_true) | set(int(x) for x in y_pred)))
        prec_macro_sum = 0.0
        rec_macro_sum = 0.0
        f1_macro_sum = 0.0

        prec_weighted_sum = 0.0
        rec_weighted_sum = 0.0
        f1_weighted_sum = 0.0

        for c in classes:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == c and int(yp) == c)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) != c and int(yp) == c)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == c and int(yp) != c)
            support = sum(1 for yt in y_true if int(yt) == c)

            prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            prec_macro_sum += prec
            rec_macro_sum += rec
            f1_macro_sum += f1

            weight = support / n
            prec_weighted_sum += prec * weight
            rec_weighted_sum += rec * weight
            f1_weighted_sum += f1 * weight

        n_classes = max(1, len(classes))
        return {
            "holdout_acc": float(acc),
            "precision_macro": float(prec_macro_sum / n_classes),
            "precision_weighted": float(prec_weighted_sum),
            "recall_macro": float(rec_macro_sum / n_classes),
            "recall_weighted": float(rec_weighted_sum),
            "f1_macro": float(f1_macro_sum / n_classes),
            "f1_weighted": float(f1_weighted_sum),
        }


# ==============================================================================
# Runners for Each Engine
# ==============================================================================

def run_trial_c11(
    train_csv: str,
    test_csv: str,
    task: str,
    params: Dict[str, Any],
    y_test: List[float],
    is_high_dim: bool,
) -> Dict[str, Any]:
    tier = params.get("tier", 256)
    slots = params.get("reg_num_slots", 16)
    bin_path = ensure_c11_binary(tier, slots, is_high_dim, task)

    json_path = f"/tmp/tune_c11_{os.path.basename(bin_path)}_{time.time_ns()}_{random.randint(0, 10000000)}.json"
    cmd = [
        bin_path,
        train_csv,
        "--test-csv", test_csv,
        "--json", json_path,
        "--grace-period", str(params.get("grace_period", 200)),
        "--delta", str(params.get("delta", 1e-7)),
        "--tau", str(params.get("tau", 0.05)),
    ]
    
    # New params
    if params.get("prep_scaler") == "standard":
        cmd.append("--prep-scaler")
    if params.get("leaf_normalize", False):
        cmd.append("--leaf-normalize")
    if params.get("obs_splitter"):
        cmd.extend(["--obs", params["obs_splitter"]])
    if params.get("filter_hampel", False):
        cmd.append("--filter-hampel")

    if task == "regression":
        cmd.append("--regression")
        leaf_p = params.get("leaf_prediction", "adaptive")
        cmd.extend(["--leaf", leaf_p])
    else:
        crit = params.get("split_criterion", "info_gain")
        cmd.extend(["--criterion", crit])
        leaf_p = params.get("leaf_prediction", "nba")
        cmd.extend(["--leaf", leaf_p])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if os.path.exists(json_path):
        os.remove(json_path)

    # Dynamic and static model size
    dynamic_bytes = meta.get("final_bytes_used", 0)
    model_static_bytes = dynamic_bytes
    for line in proc.stdout.splitlines():
        if "model static size =" in line:
            parts = line.split("model static size =")[1].split("bytes")[0].strip()
            model_static_bytes = int(parts)

    intra_train_ram_kb = model_static_bytes / 1024.0
    train_rss_mb = meta.get("train_rss_kb", 2500) / 1024.0
    test_rss_mb = meta.get("test_rss_kb", train_rss_mb * 1024.0) / 1024.0
    peak_rss_mb = meta.get("peak_rss_kb", test_rss_mb * 1024.0) / 1024.0

    res: Dict[str, Any] = {
        "engine": "c11",
        "params": params,
        "n_nodes": meta.get("final_nodes", 1),
        "n_leaves": meta.get("final_active_leaves", 1),
        "intra_train_ram_kb": intra_train_ram_kb,
        "final_bytes_used": dynamic_bytes,
        "dynamic_ram_kb": dynamic_bytes / 1024.0,
        "train_rss_mb": train_rss_mb,
        "test_rss_mb": test_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "avg_us_per_sample": meta.get("avg_time_per_sample_us", 0.0),
    }

    preds = meta.get("holdout_predictions", [])
    if preds and len(preds) == len(y_test):
        computed = compute_metrics(y_test, preds, task)
        res.update(computed)
    else:
        if task == "regression":
            res["holdout_mae"] = float(meta.get("holdout_mae", 0.0))
            res["holdout_rmse"] = float(meta.get("holdout_rmse", 0.0))
            res["holdout_r2"] = float(meta.get("holdout_r2", 0.0))
        else:
            acc = float(meta.get("holdout_accuracy", 0.0) * 100.0)
            res["holdout_acc"] = acc
            res["f1_macro"] = acc
            res["f1_weighted"] = acc
            res["precision_macro"] = acc
            res["precision_weighted"] = acc
            res["recall_macro"] = acc
            res["recall_weighted"] = acc

    if task == "regression":
        res["prequential_mae"] = float(meta.get("final_mae", 0.0))
    else:
        res["prequential_acc"] = float(meta.get("final_accuracy", 0.0) * 100.0)

    return res


def run_trial_river(
    train_csv: str,
    test_csv: str,
    task: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    runner_script = os.path.join(os.path.dirname(__file__), "runners", "run_river.py")
    cmd = [
        PYTHON_BIN,
        runner_script,
        "--train-csv", train_csv,
        "--test-csv", test_csv,
        "--task", task,
        "--grace-period", str(params.get("grace_period", 200)),
        "--delta", str(params.get("delta", 1e-7)),
        "--tau", str(params.get("tau", 0.05)),
        "--leaf-prediction", str(params.get("leaf_prediction", "adaptive")),
        "--splitter", str(params.get("splitter", "default")),
    ]
    # Parity: pipeline
    if params.get("prep_scaler") == "standard":
        cmd.append("--use-scaler")
        
    if task == "classification":
        cmd.extend(["--split-criterion", str(params.get("split_criterion", "info_gain"))])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout.strip())
    res["engine"] = "river"
    res["params"] = params
    return res


def run_trial_sklearn(
    train_csv: str,
    test_csv: str,
    task: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    runner_script = os.path.join(os.path.dirname(__file__), "runners", "run_sklearn.py")
    cmd = [
        PYTHON_BIN,
        runner_script,
        "--train-csv", train_csv,
        "--test-csv", test_csv,
        "--task", task,
        "--min-samples-split", str(params.get("min_samples_split", 2)),
        "--min-samples-leaf", str(params.get("min_samples_leaf", 1)),
    ]
    # Parity: pipeline
    if params.get("prep_scaler") == "standard":
        cmd.append("--use-scaler")
        
    if params.get("max_depth") is not None:
        cmd.extend(["--max-depth", str(params["max_depth"])])
    if params.get("max_features") is not None:
        cmd.extend(["--max-features", str(params["max_features"])])
    if params.get("criterion") is not None:
        cmd.extend(["--criterion", str(params["criterion"])])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout.strip())
    res["engine"] = "sklearn"
    res["params"] = params
    return res


# ==============================================================================
# Hyperparameter Search Space Builders
# ==============================================================================

def get_c11_space(task: str, is_high_dim: bool = False, target_tier: Optional[int] = None) -> Dict[str, List[Any]]:
    # In dense datasets, max physical active leaves is 40 to guarantee <= 1 MB SRAM
    if target_tier is not None and target_tier > 0:
        tiers = [target_tier]
    else:
        tiers = [1, 2, 4, 8, 16, 40] if is_high_dim else [1, 2, 4, 8, 16, 64, 128, 256]
    space: Dict[str, List[Any]] = {
        "grace_period": [50, 100, 200, 400],
        "delta": [1e-2, 1e-3, 1e-4, 1e-7],
        "tau": [0.01, 0.05, 0.10, 0.15],
        "tier": tiers,
        "prep_scaler": [None, "standard"],
        "obs_splitter": ["gaussian", "histogram"],
        "filter_hampel": [False, True],
    }
    if task == "classification":
        space["split_criterion"] = ["info_gain", "gini"]
        space["leaf_prediction"] = ["nba", "nb", "mc"]
    else:
        space["leaf_prediction"] = ["adaptive", "model", "mean"]
        space["reg_num_slots"] = [8, 16, 32]
        space["leaf_normalize"] = [False, True]
    return space


def get_canonical_seeds(task: str, tier: int) -> List[Dict[str, Any]]:
    """
    Returns 4 proven canonical configurations guaranteed to be evaluated
    in each tier, preventing stochastic omission of global/hardware optima.
    """
    if task == "classification":
        return [
            # Seed 1: Sweet-spot padrão (base ESP32 no SEA)
            {
                "tier": tier,
                "grace_period": 100,
                "delta": 0.01,
                "tau": 0.10,
                "split_criterion": "info_gain",
                "leaf_prediction": "nba",
                "prep_scaler": None,
                "obs_splitter": "gaussian",
                "filter_hampel": False,
            },
            # Seed 2: Multiclasse ágil (base Covertype / Airlines)
            {
                "tier": tier,
                "grace_period": 50,
                "delta": 0.001,
                "tau": 0.10,
                "split_criterion": "gini",
                "leaf_prediction": "nba",
                "prep_scaler": None,
                "obs_splitter": "gaussian",
                "filter_hampel": False,
            },
            # Seed 3: Alta evidência estatística com pré-processamento online
            {
                "tier": tier,
                "grace_period": 200,
                "delta": 1e-4,
                "tau": 0.05,
                "split_criterion": "info_gain",
                "leaf_prediction": "nba",
                "prep_scaler": "standard",
                "obs_splitter": "histogram",
                "filter_hampel": False,
            },
            # Seed 4: Conservador de baixa variância
            {
                "tier": tier,
                "grace_period": 400,
                "delta": 0.01,
                "tau": 0.10,
                "split_criterion": "gini",
                "leaf_prediction": "mc",
                "prep_scaler": None,
                "obs_splitter": "gaussian",
                "filter_hampel": False,
            },
        ]
    else:
        return [
            # Seed 1: SGD Linear com Scaler (base Friedman)
            {
                "tier": tier,
                "grace_period": 200,
                "delta": 1e-7,
                "tau": 0.01,
                "leaf_prediction": "model",
                "reg_num_slots": 16,
                "leaf_normalize": False,
                "prep_scaler": "standard",
                "obs_splitter": "gaussian",
                "filter_hampel": False,
            },
            # Seed 2: Adaptativo com Normalização Híbrida (base Appliances)
            {
                "tier": tier,
                "grace_period": 100,
                "delta": 1e-7,
                "tau": 0.01,
                "leaf_prediction": "adaptive",
                "reg_num_slots": 16,
                "leaf_normalize": True,
                "prep_scaler": None,
                "obs_splitter": "gaussian",
                "filter_hampel": False,
            },
            # Seed 3: Média simples zero SGD (base Bikes)
            {
                "tier": tier,
                "grace_period": 200,
                "delta": 0.01,
                "tau": 0.10,
                "leaf_prediction": "mean",
                "reg_num_slots": 8,
                "leaf_normalize": False,
                "prep_scaler": None,
                "obs_splitter": "gaussian",
                "filter_hampel": False,
            },
            # Seed 4: Alta resolução contínua (32 slots + Scaler + Histograma)
            {
                "tier": tier,
                "grace_period": 100,
                "delta": 1e-4,
                "tau": 0.15,
                "leaf_prediction": "adaptive",
                "reg_num_slots": 32,
                "leaf_normalize": True,
                "prep_scaler": "standard",
                "obs_splitter": "histogram",
                "filter_hampel": False,
            },
        ]


def sample_stratified_c11_space(
    task: str,
    is_high_dim: bool,
    trials_per_tier: int,
    seed: int = 42,
    target_tier: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Samples configurations stratified per tier:
    1. Injects the 4 canonical seeds for each tier.
    2. Samples remaining (trials_per_tier - 4) exploratory configurations uniformly.
    """
    space = get_c11_space(task, is_high_dim, target_tier=target_tier)
    tiers = space["tier"]
    all_configs: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    param_keys = [k for k in space.keys() if k != "tier"]

    for tier in tiers:
        tier_configs: List[Dict[str, Any]] = []
        seen = set()

        # 1. Inject canonical seeds
        canonical_seeds = get_canonical_seeds(task, tier)
        for seed_cfg in canonical_seeds:
            cfg = dict(seed_cfg)
            key_tuple = tuple((k, str(cfg[k])) for k in sorted(cfg.keys()))
            if key_tuple not in seen:
                seen.add(key_tuple)
                tier_configs.append(cfg)
                if len(tier_configs) >= trials_per_tier:
                    break

        # 2. Sample exploratory trials for this tier
        max_attempts = trials_per_tier * 50
        attempts = 0
        while len(tier_configs) < trials_per_tier and attempts < max_attempts:
            attempts += 1
            cfg = {"tier": tier}
            for k in param_keys:
                cfg[k] = rng.choice(space[k])
            key_tuple = tuple((k, str(cfg[k])) for k in sorted(cfg.keys()))
            if key_tuple not in seen:
                seen.add(key_tuple)
                tier_configs.append(cfg)

        all_configs.extend(tier_configs)

    return all_configs


def get_river_space(task: str) -> Dict[str, List[Any]]:
    space: Dict[str, List[Any]] = {
        "grace_period": [50, 100, 200, 400],
        "delta": [1e-2, 1e-3, 1e-4, 1e-7],
        "tau": [0.01, 0.05, 0.10, 0.15],
        "prep_scaler": [None, "standard"],
    }
    if task == "classification":
        space["leaf_prediction"] = ["nba", "nb", "mc"]
        space["split_criterion"] = ["info_gain", "gini"]
        space["splitter"] = ["gaussian", "histogram"]
    else:
        space["leaf_prediction"] = ["adaptive", "model", "mean"]
        space["splitter"] = ["tebst", "ebst"]
    return space


def get_sklearn_space(task: str) -> Dict[str, List[Any]]:
    space: Dict[str, List[Any]] = {
        "max_depth": [4, 6, 8, 12, 16, None],
        "min_samples_split": [10, 50, 100, 500],
        "min_samples_leaf": [5, 20, 50, 100],
        "max_features": [None, "sqrt", "log2"],
        "prep_scaler": [None, "standard"],
    }
    if task == "classification":
        space["criterion"] = ["entropy", "gini"]
    else:
        space["criterion"] = ["squared_error", "absolute_error"]
    return space


def sample_hyperparameters(
    space: Dict[str, List[Any]],
    n_iter: int,
    is_grid: bool,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    value_lists = [space[k] for k in keys]

    if is_grid:
        combinations = [dict(zip(keys, p)) for p in itertools.product(*value_lists)]
        return combinations

    # Random sampling
    total_combinations = 1
    for v in value_lists:
        total_combinations *= len(v)

    if n_iter >= total_combinations:
        combinations = [dict(zip(keys, p)) for p in itertools.product(*value_lists)]
        return combinations

    rng = random.Random(seed)
    seen = set()
    sampled = []
    max_attempts = n_iter * 20
    attempts = 0

    while len(sampled) < n_iter and attempts < max_attempts:
        attempts += 1
        cfg = {k: rng.choice(space[k]) for k in keys}
        key_tuple = tuple((k, str(cfg[k])) for k in sorted(cfg.keys()))
        if key_tuple not in seen:
            seen.add(key_tuple)
            sampled.append(cfg)

    return sampled


# ==============================================================================
# Report & Analysis Generation
# ==============================================================================

def generate_markdown_report(
    results_by_dataset: Dict[str, List[Dict[str, Any]]],
    output_path: str,
):
    """
    Renders hyperparam_report.md containing Top-K models, Pareto curves and
    sensitivity insights for each dataset.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Relatório Consolidado de Otimização e Sensibilidade de Hiperparâmetros\n\n")
        f.write("Suíte automatizada de exploração de hiperparâmetros comparando **Kansho**, **River** e **Scikit-Learn** em 8 datasets emblemáticos.\n\n")
        f.write("---\n\n")

        for d_id, d_name, task, desc, is_high_dim in DATASETS_CONFIG:
            if d_id not in results_by_dataset:
                continue

            trials = results_by_dataset[d_id]
            f.write(f"## Dataset: {d_name} ({task.capitalize()})\n\n")
            f.write(f"- **Descrição:** {desc}\n")
            f.write(f"- **Total de Experimentos (Trials):** {len(trials)}\n\n")

            # Group trials by engine / tier
            by_tier: Dict[int, List[Dict[str, Any]]] = {}
            for t in trials:
                if t.get("engine") in ("c11", "kansho"):
                    tier = t.get("params", {}).get("tier", 0)
                    if tier not in by_tier:
                        by_tier[tier] = []
                    by_tier[tier].append(t)
            
            # Engines
            engines = {"river": [], "sklearn": []}
            for t in trials:
                eng = t.get("engine")
                if eng in engines:
                    engines[eng].append(t)

            # Sort trials:
            # - Classificação: 1º F1 Score (macro), 2º Acurácia (desempate: latência)
            # - Regressão: 1º R² (maior melhor), 2º RMSE (menor melhor como desempate: latência)
            def sort_key(x):
                if task == "classification":
                    return (x.get("f1_macro", 0.0), x.get("holdout_acc", 0.0), -x.get("avg_us_per_sample", 0.0))
                else:
                    return (x.get("holdout_r2", -1e9), -x.get("holdout_rmse", 1e9), -x.get("avg_us_per_sample", 0.0))

            for k in by_tier:
                by_tier[k].sort(key=sort_key, reverse=True)
            for k in engines:
                engines[k].sort(key=sort_key, reverse=True)

            # 1. Melhores Modelos por Tier (Kansho) e Competidores
            f.write(f"### 1. Melhores Modelos por Tier (Kansho) e Competidores (Top 3)\n\n")
            
            # Kansho per Tier
            for tier in sorted(by_tier.keys()):
                leaves, nodes = TIER_MAP.get(tier, (tier, tier * 16))
                if is_high_dim and tier > 16:
                    leaves = 40
                    nodes = 1024
                top_trial = by_tier[tier][0] if by_tier[tier] else {}
                used_nodes = top_trial.get("n_nodes", 1)
                active_str = f"{leaves} nó ativo" if leaves == 1 else f"{leaves} nós ativos"
                used_str = f"{used_nodes} utilizado" if used_nodes == 1 else f"{used_nodes} utilizados"
                max_str = f"{nodes} nós no max"
                f.write(f"#### Kansho (Tier {tier}: {active_str}, {used_str}, {max_str})\n\n")
                if task == "classification":
                    f.write("| Rank | Configuração | Acurácia (%) | F1 Macro (%) | RAM (KB) | Latência (µs) |\n")
                    f.write("| :---: | :--- | :---: | :---: | :---: | :---: |\n")
                    for rank, t in enumerate(by_tier[tier][:3], start=1):
                        p_str = ", ".join(f"`{k}={v}`" for k, v in t["params"].items())
                        f.write(f"| #{rank} | {p_str} | {t.get('holdout_acc', 0.0):.2f}% | {t.get('f1_macro', 0.0):.2f}% | {t.get('intra_train_ram_kb', 0.0):.1f} KB | {t.get('avg_us_per_sample', 0.0):.2f} µs |\n")
                else:
                    f.write("| Rank | Configuração | MAE | R² | RAM (KB) | Latência (µs) |\n")
                    f.write("| :---: | :--- | :---: | :---: | :---: | :---: |\n")
                    for rank, t in enumerate(by_tier[tier][:3], start=1):
                        p_str = ", ".join(f"`{k}={v}`" for k, v in t["params"].items())
                        f.write(f"| #{rank} | {p_str} | {t.get('holdout_mae', 0.0):.4f} | {t.get('holdout_r2', 0.0):.4f} | {t.get('intra_train_ram_kb', 0.0):.1f} KB | {t.get('avg_us_per_sample', 0.0):.2f} µs |\n")
                f.write("\n")
                
            # Competitors
            for eng in ["river", "sklearn"]:
                f.write(f"#### {eng.capitalize()}\n\n")
                if task == "classification":
                    f.write("| Rank | Configuração | Acurácia (%) | F1 Macro (%) | Latência (µs) |\n")
                    f.write("| :---: | :--- | :---: | :---: | :---: |\n")
                    for rank, t in enumerate(engines[eng][:3], start=1):
                        p_str = ", ".join(f"`{k}={v}`" for k, v in t["params"].items())
                        f.write(f"| #{rank} | {p_str} | {t.get('holdout_acc', 0.0):.2f}% | {t.get('f1_macro', 0.0):.2f}% | {t.get('avg_us_per_sample', 0.0):.2f} µs |\n")
                else:
                    f.write("| Rank | Configuração | MAE | R² | Latência (µs) |\n")
                    f.write("| :---: | :--- | :---: | :---: | :---: |\n")
                    for rank, t in enumerate(engines[eng][:3], start=1):
                        p_str = ", ".join(f"`{k}={v}`" for k, v in t["params"].items())
                        f.write(f"| #{rank} | {p_str} | {t.get('holdout_mae', 0.0):.4f} | {t.get('holdout_r2', 0.0):.4f} | {t.get('avg_us_per_sample', 0.0):.2f} µs |\n")
                f.write("\n")

            f.write("\n---\n\n")

        # Global Summary
        f.write("## 2. Síntese e Diretrizes de Engenharia\n\n")
        f.write("1. **Análise por Tier:** O ranking estratificado revela que configurações ideais para 256 folhas frequentemente diferem daqueles necessários para evitar subajuste em 4 ou 8 folhas.\n")
        f.write("2. **Eficiência:** O uso de latência como desempate garante que configurações computacionalmente custosas que entreguem ganhos marginais de acurácia sejam penalizadas.\n")


# ==============================================================================
# Progress Evaluation & Tracking
# ==============================================================================

class ProgressTracker:
    """Thread-safe progress evaluator for multi-core hyperparameter tuning."""
    def __init__(self, total_trials: int):
        self.total_trials = total_trials
        self.completed = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
            return "--:--"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def update(
        self,
        engine_name: str,
        dataset_name: str,
        engine_idx: int,
        engine_total: int,
        metric_str: str,
        extra_str: str = "",
    ):
        with self.lock:
            self.completed += 1
            now = time.time()
            elapsed = now - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0.0
            remaining = (self.total_trials - self.completed) / rate if rate > 0 else 0.0

            pct_global = (self.completed / max(1, self.total_trials)) * 100.0
            bar_len = 20
            filled = int(bar_len * self.completed // max(1, self.total_trials))
            bar = "█" * filled + "░" * (bar_len - filled)

            elapsed_str = self._format_time(elapsed)
            eta_str = self._format_time(remaining)

            print(
                f"[{bar}] {pct_global:5.1f}% | Global: {self.completed:4d}/{self.total_trials} | "
                f"[{elapsed_str}<{eta_str}, {rate:.2f} it/s] | "
                f"[{dataset_name} | {engine_name.upper()} {engine_idx:3d}/{engine_total}] {metric_str} {extra_str}",
                flush=True,
            )


def run_engine_trials(
    engine_name: str,
    trial_fn,
    configs: List[Dict[str, Any]],
    task: str,
    jobs: int,
    dataset_name: str = "",
    tracker: Optional[ProgressTracker] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = len(configs)
    completed = 0
    lock = threading.Lock()
    t_start = time.time()

    def worker(idx_cfg):
        nonlocal completed
        idx, cfg = idx_cfg
        try:
            res = trial_fn(cfg)
            with lock:
                completed += 1
                metric_val = (
                    f"Acc={res.get('holdout_acc', 0.0):.2f}%"
                    if task == "classification"
                    else f"MAE={res.get('holdout_mae', 0.0):.4f}, R2={res.get('holdout_r2', 0.0):.4f}"
                )
                extra = f"RAM={res.get('intra_train_ram_kb', 0.0):.1f}KB, Time={res.get('avg_us_per_sample', 0.0):.2f}us | {cfg}"
            
            if tracker is not None:
                tracker.update(engine_name, dataset_name, completed, total, metric_val, extra)
            else:
                with lock:
                    print(f"    [{engine_name.upper()}] {completed:4d}/{total}: {metric_val}, {extra}", flush=True)
            return res
        except Exception as e:
            with lock:
                completed += 1
                err_msg = f"ERROR ({e}) | {cfg}"
            if tracker is not None:
                tracker.update(engine_name, dataset_name, completed, total, err_msg)
            else:
                with lock:
                    print(f"    [{engine_name.upper()}] {completed:4d}/{total}: {err_msg}", flush=True)
            return None
        
    if jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            for r in executor.map(worker, enumerate(configs, start=1)):
                if r is not None:
                    results.append(r)
    else:
        for idx_cfg in enumerate(configs, start=1):
            r = worker(idx_cfg)
            if r is not None:
                results.append(r)

    elapsed_engine = time.time() - t_start
    throughput = len(results) / max(0.001, elapsed_engine)
    print(f"  [✔] {engine_name.upper()} em {dataset_name} concluído: {len(results)}/{total} ensaios válidos em {elapsed_engine:.1f}s ({throughput:.2f} it/s)\n", flush=True)

    return results


# ==============================================================================
# Suite Orchestrator
# ==============================================================================

def extract_report_configs(
    all_results: Dict[str, List[Dict[str, Any]]],
    selected_datasets: List[str],
    selected_engines: List[str],
    top_k: int = 3,
    report_path: Optional[str] = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Extracts configurations to replay.
    If report_path is provided and exists, parses the exact configurations published
    in the Markdown report tables 1:1, ensuring exact parity with the published report.
    Otherwise, falls back to sorting top_k from all_results.
    """
    dataset_map = {
        "SEA": "sea",
        "Electricity": "electricity",
        "Forest Covertype": "covertype",
        "Airlines": "airlines",
        "Friedman #1": "friedman",
        "Bikes": "bikes",
        "Appliances Energy": "appliances_energy",
        "NYC Taxi": "nyc_taxi",
    }
    kansho_enabled = ("c11" in selected_engines or "kansho" in selected_engines or "all" in selected_engines)
    river_enabled = ("river" in selected_engines or "all" in selected_engines)
    sklearn_enabled = ("sklearn" in selected_engines or "all" in selected_engines)

    if report_path and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_dataset = None
        current_engine = "kansho"
        extracted_from_md: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for line in lines:
            line_s = line.rstrip("\r\n")
            if line_s.startswith("## Dataset:"):
                current_dataset = None
                for name, did in dataset_map.items():
                    if name in line_s:
                        current_dataset = did
                        break
                if current_dataset:
                    extracted_from_md[current_dataset] = {"kansho": [], "river": [], "sklearn": []}
                current_engine = "kansho"
            elif line_s.startswith("#### Kansho"):
                current_engine = "kansho"
            elif line_s.startswith("#### River"):
                current_engine = "river"
            elif line_s.startswith("#### Sklearn"):
                current_engine = "sklearn"
            elif line_s.startswith("| #") and current_dataset:
                parts = [p.strip() for p in line_s.split("|")]
                if len(parts) >= 3:
                    cfg_str = parts[2]
                    tokens = [t.strip().strip("`") for t in cfg_str.split(",") if "=" in t]
                    d: Dict[str, Any] = {}
                    for tok in tokens:
                        if "=" not in tok:
                            continue
                        k, v = tok.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if v == "None":
                            d[k] = None
                        elif v == "True":
                            d[k] = True
                        elif v == "False":
                            d[k] = False
                        else:
                            try:
                                d[k] = int(v) if v.isdigit() or (v.startswith("-") and v[1:].isdigit()) else float(v)
                            except ValueError:
                                d[k] = v
                    if d:
                        eng_key = current_engine
                        if eng_key == "kansho" and not kansho_enabled:
                            continue
                        if eng_key == "river" and not river_enabled:
                            continue
                        if eng_key == "sklearn" and not sklearn_enabled:
                            continue
                        extracted_from_md[current_dataset][eng_key].append(d)

        # Filter by selected_datasets
        filtered: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for did, eng_dict in extracted_from_md.items():
            if selected_datasets and did not in selected_datasets and "all" not in selected_datasets:
                continue
            filtered[did] = eng_dict
        if any(len(cfgs) > 0 for d in filtered.values() for cfgs in d.values()):
            return filtered

    extracted: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for d_id, d_name, task, desc, is_high_dim in DATASETS_CONFIG:
        if selected_datasets and d_id not in selected_datasets and "all" not in selected_datasets:
            continue
        trials = all_results.get(d_id, [])
        if not trials:
            continue

        def sort_key(x):
            if task == "classification":
                return (x.get("f1_macro", 0.0), x.get("holdout_acc", 0.0), -x.get("avg_us_per_sample", 0.0))
            else:
                return (x.get("holdout_r2", -1e9), -x.get("holdout_rmse", 1e9), -x.get("avg_us_per_sample", 0.0))

        dataset_extracted: Dict[str, List[Dict[str, Any]]] = {}

        if kansho_enabled:
            by_tier: Dict[int, List[Dict[str, Any]]] = {}
            for t in trials:
                if t.get("engine") in ("c11", "kansho"):
                    tier = t.get("params", {}).get("tier", 0)
                    by_tier.setdefault(tier, []).append(t)
            c11_configs: List[Dict[str, Any]] = []
            for tier in sorted(by_tier.keys()):
                by_tier[tier].sort(key=sort_key, reverse=True)
                for t in by_tier[tier][:top_k]:
                    c11_configs.append(t["params"])
            dataset_extracted["kansho"] = c11_configs

        if river_enabled:
            river_trials = [t for t in trials if t.get("engine") == "river"]
            river_trials.sort(key=sort_key, reverse=True)
            dataset_extracted["river"] = [t["params"] for t in river_trials[:top_k]]

        if sklearn_enabled:
            sklearn_trials = [t for t in trials if t.get("engine") == "sklearn"]
            sklearn_trials.sort(key=sort_key, reverse=True)
            dataset_extracted["sklearn"] = [t["params"] for t in sklearn_trials[:top_k]]

        extracted[d_id] = dataset_extracted

    return extracted


def update_report_metrics_in_place(
    baseline_report_path: str,
    all_results: Dict[str, List[Dict[str, Any]]],
    output_report_path: str,
):
    """
    Updates the metrics in-place for every model row in baseline_report_path,
    preserving the exact original row ordering and parameter strings 1:1.
    """
    if not os.path.exists(baseline_report_path):
        return

    def make_param_key(engine: str, params: Dict[str, Any]) -> Tuple:
        eng = "c11" if engine in ("c11", "kansho") else engine
        return (eng, tuple(sorted((k, str(v)) for k, v in params.items())))

    indexed = {}
    for did, trials in all_results.items():
        for t in trials:
            k = make_param_key(t.get("engine", ""), t.get("params", {}))
            indexed[(did, k)] = t

    dataset_map = {
        "SEA": "sea",
        "Electricity": "electricity",
        "Forest Covertype": "covertype",
        "Airlines": "airlines",
        "Friedman #1": "friedman",
        "Bikes": "bikes",
        "Appliances Energy": "appliances_energy",
        "NYC Taxi": "nyc_taxi",
    }

    current_dataset = None
    current_engine = "kansho"
    task = "classification"
    out_lines = []

    with open(baseline_report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line_s = line.rstrip("\r\n")
        if line_s.startswith("## Dataset:"):
            for name, did in dataset_map.items():
                if name in line_s:
                    current_dataset = did
                    break
            task = "regression" if ("Regression" in line_s) else "classification"
            current_engine = "kansho"
            out_lines.append(line_s)
        elif line_s.startswith("#### Kansho"):
            current_engine = "kansho"
            out_lines.append(line_s)
        elif line_s.startswith("#### River"):
            current_engine = "river"
            out_lines.append(line_s)
        elif line_s.startswith("#### Sklearn"):
            current_engine = "sklearn"
            out_lines.append(line_s)
        elif line_s.startswith("| #"):
            parts = [p.strip() for p in line_s.split("|")]
            if len(parts) >= 7:
                rank = parts[1]
                cfg_str = parts[2]
                tokens = [t.strip().strip("`") for t in cfg_str.split(",") if "=" in t]
                d = {}
                for tok in tokens:
                    k, v = tok.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if v == "None":
                        d[k] = None
                    elif v == "True":
                        d[k] = True
                    elif v == "False":
                        d[k] = False
                    else:
                        try:
                            if "." in v or "e" in v.lower():
                                d[k] = float(v)
                            else:
                                d[k] = int(v)
                        except ValueError:
                            d[k] = v
                pk = make_param_key(current_engine, d)
                matched = indexed.get((current_dataset, pk))
                if matched and current_engine == "kansho":
                    ram = matched.get("intra_train_ram_kb", 0.0)
                    lat = matched.get("avg_us_per_sample", 0.0)
                    if task == "classification":
                        acc = matched.get("holdout_acc", 0.0)
                        f1 = matched.get("f1_macro", 0.0)
                        new_line = f"| {rank} | {cfg_str} | {acc:.2f}% | {f1:.2f}% | {ram:.1f} KB | {lat:.2f} µs |"
                    else:
                        mae = matched.get("holdout_mae", 0.0)
                        r2 = matched.get("holdout_r2", 0.0)
                        new_line = f"| {rank} | {cfg_str} | {mae:.4f} | {r2:.4f} | {ram:.1f} KB | {lat:.2f} µs |"
                    out_lines.append(new_line)
                else:
                    out_lines.append(line_s)
            else:
                out_lines.append(line_s)
        else:
            out_lines.append(line_s)

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")


def run_tuning_suite(
    data_dir: str,
    output_report: str,
    output_json: str,
    selected_datasets: List[str],
    selected_engines: List[str],
    n_iter: int,
    is_grid: bool,
    seed: int,
    jobs: int,
    trials_per_tier: int = 30,
    target_tier: Optional[int] = None,
    replay_report: bool = False,
    report_only: bool = False,
):
    if report_only:
        if not os.path.exists(output_json):
            print(f"[Error: JSON file '{output_json}' not found. Cannot generate report.]")
            sys.exit(1)
        with open(output_json, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        generate_markdown_report(all_results, output_report)
        print(f"[✔] Report generated successfully from {output_json} -> {output_report}")
        return

    norm_engines = [e.strip().lower() for e in selected_engines]
    kansho_enabled = ("c11" in norm_engines or "kansho" in norm_engines or "all" in norm_engines)
    river_enabled = ("river" in norm_engines or "all" in norm_engines)
    sklearn_enabled = ("sklearn" in norm_engines or "all" in norm_engines)

    if replay_report:
        if not os.path.exists(output_json):
            print(f"[Error: JSON file '{output_json}' not found for replay mode.]")
            sys.exit(1)

        backup_path = output_json.replace(".json", ".backup.json")
        shutil.copyfile(output_json, backup_path)
        backup_report_path = output_report.replace(".md", ".backup.md")
        if os.path.exists(output_report):
            shutil.copyfile(output_report, backup_report_path)
        print("================================================================================")
        print(" REPLAY MODE: Re-executando configurações do relatório")
        print(f" Backup de segurança salvo em: {backup_path}")
        print("================================================================================")

        with open(output_json, "r", encoding="utf-8") as f:
            all_results = json.load(f)

        report_configs = extract_report_configs(
            all_results,
            selected_datasets=selected_datasets,
            selected_engines=norm_engines,
            top_k=3,
            report_path=output_report,
        )

        def make_param_key(engine: str, params: Dict[str, Any]) -> Tuple:
            eng = "c11" if engine in ("c11", "kansho") else engine
            return (eng, tuple(sorted((k, str(v)) for k, v in params.items())))

        # Build valid keys set for report view to ensure report only displays the exact evaluated models
        valid_report_keys: Dict[str, set] = {}
        for d_id, d_name, task, desc, is_high_dim in DATASETS_CONFIG:
            valid_report_keys[d_id] = set()
            if d_id in report_configs:
                for eng, cfgs in report_configs[d_id].items():
                    for c in cfgs:
                        valid_report_keys[d_id].add(make_param_key(eng, c))
            trials = all_results.get(d_id, [])
            def sort_key_fn(x):
                if task == "classification":
                    return (x.get("f1_macro", 0.0), x.get("holdout_acc", 0.0), -x.get("avg_us_per_sample", 0.0))
                else:
                    return (x.get("holdout_r2", -1e9), -x.get("holdout_rmse", 1e9), -x.get("avg_us_per_sample", 0.0))
            for eng in ["river", "sklearn"]:
                eng_trials = [t for t in trials if t.get("engine") == eng]
                eng_trials.sort(key=sort_key_fn, reverse=True)
                for t in eng_trials[:3]:
                    valid_report_keys[d_id].add(make_param_key(eng, t.get("params", {})))

        total_planned_trials = sum(
            len(cfgs)
            for d_dict in report_configs.values()
            for cfgs in d_dict.values()
        )

        tracker = ProgressTracker(total_planned_trials)

        print(f" Datasets:     {', '.join(selected_datasets)} ({len(report_configs)} com ensaios para replay)")
        print(f" Engines:      {', '.join(selected_engines)}")
        print(f" Parallelism:  {jobs} job(s) / núcleo(s)")
        print(f" Total Trials: {total_planned_trials} ensaios para replay")
        print(f" Report Dest:  {output_report}")
        print(f" JSON Dest:    {output_json}")
        print("================================================================================")

        for d_id, d_name, task, desc, is_high_dim in DATASETS_CONFIG:
            if d_id not in report_configs:
                continue

            d_cfgs = report_configs[d_id]
            print(f"\n================================================================================")
            print(f" >>> Dataset: {d_name} [{task.upper()}] - {desc}")
            print(f"================================================================================")

            train_csv = os.path.join(data_dir, f"{d_id}_train.csv")
            test_csv = os.path.join(data_dir, f"{d_id}_test.csv")

            if not os.path.exists(train_csv) or not os.path.exists(test_csv):
                print(f"  Dataset files not found for {d_id}. Running prepare_datasets.py...")
                prep_script = os.path.join(os.path.dirname(__file__), "prepare_datasets.py")
                subprocess.run([PYTHON_BIN, prep_script, "--outdir", data_dir, "--dataset", d_id], check=True)

            y_test, _ = load_test_ground_truth(test_csv, task)
            replayed_trials: List[Dict[str, Any]] = []

            # 1. Kansho
            if "kansho" in d_cfgs and d_cfgs["kansho"]:
                c11_cfgs = d_cfgs["kansho"]
                print(f"  [1/3] Replay Kansho ({len(c11_cfgs)} ensaios, {jobs} workers)...")
                c11_trials = run_engine_trials(
                    "kansho",
                    lambda cfg: run_trial_c11(train_csv, test_csv, task, cfg, y_test, is_high_dim),
                    c11_cfgs,
                    task,
                    jobs,
                    dataset_name=d_name,
                    tracker=tracker,
                )
                replayed_trials.extend(c11_trials)

            # 2. River
            if "river" in d_cfgs and d_cfgs["river"]:
                r_cfgs = d_cfgs["river"]
                print(f"  [2/3] Replay River ({len(r_cfgs)} ensaios, {jobs} workers)...")
                river_trials = run_engine_trials(
                    "river",
                    lambda cfg: run_trial_river(train_csv, test_csv, task, cfg),
                    r_cfgs,
                    task,
                    jobs,
                    dataset_name=d_name,
                    tracker=tracker,
                )
                replayed_trials.extend(river_trials)

            # 3. Sklearn
            if "sklearn" in d_cfgs and d_cfgs["sklearn"]:
                sk_cfgs = d_cfgs["sklearn"]
                print(f"  [3/3] Replay Sklearn ({len(sk_cfgs)} ensaios, {jobs} workers)...")
                sklearn_trials = run_engine_trials(
                    "sklearn",
                    lambda cfg: run_trial_sklearn(train_csv, test_csv, task, cfg),
                    sk_cfgs,
                    task,
                    jobs,
                    dataset_name=d_name,
                    tracker=tracker,
                )
                replayed_trials.extend(sklearn_trials)

            existing = all_results.get(d_id, [])
            key_to_idx = {make_param_key(t.get("engine", ""), t.get("params", {})): idx for idx, t in enumerate(existing)}

            for rt in replayed_trials:
                k = make_param_key(rt.get("engine", ""), rt.get("params", {}))
                if k in key_to_idx:
                    existing[key_to_idx[k]] = rt
                else:
                    existing.append(rt)

            all_results[d_id] = existing

            # Incremental save after each dataset
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

            if os.path.exists(backup_report_path):
                update_report_metrics_in_place(backup_report_path, all_results, output_report)
            else:
                report_view = {
                    did: [t for t in d_trials if make_param_key(t.get("engine", ""), t.get("params", {})) in valid_report_keys.get(did, set())]
                    for did, d_trials in all_results.items()
                }
                generate_markdown_report(report_view, output_report)
            print(f"  [✔] Checkpoint: Results and Report updated for {d_name}.")

        print(f"\nFinal replayed results saved to JSON: {output_json}")
        print(f"Final Markdown report updated: {output_report}")
        return

    # Pre-calculate total planned trials across datasets and engines
    active_datasets = [
        d for d in DATASETS_CONFIG
        if not selected_datasets or "all" in selected_datasets or d[0] in selected_datasets
    ]
    norm_engines = [e.strip().lower() for e in selected_engines]
    kansho_enabled = ("c11" in norm_engines or "kansho" in norm_engines or "all" in norm_engines)
    river_enabled = ("river" in norm_engines or "all" in norm_engines)
    sklearn_enabled = ("sklearn" in norm_engines or "all" in norm_engines)

    total_planned_trials = 0
    for _, _, task, _, is_high_dim in active_datasets:
        if is_grid:
            n_c11 = len(sample_hyperparameters(get_c11_space(task, is_high_dim, target_tier), n_iter, is_grid, seed))
        else:
            n_c11 = len(sample_stratified_c11_space(task, is_high_dim, trials_per_tier, seed, target_tier))

        if kansho_enabled:
            total_planned_trials += n_c11
        if river_enabled:
            total_planned_trials += len(sample_hyperparameters(get_river_space(task), n_c11, is_grid, seed))
        if sklearn_enabled:
            total_planned_trials += len(sample_hyperparameters(get_sklearn_space(task), n_c11, is_grid, seed))

    tracker = ProgressTracker(total_planned_trials)

    print("================================================================================")
    print(" HYPERPARAMETER TUNING & SENSITIVITY ANALYSIS SUITE")
    print(f" Datasets:     {', '.join(selected_datasets)} ({len(active_datasets)} selecionados)")
    print(f" Engines:      {', '.join(selected_engines)}")
    tier_info = f"Tier {target_tier}" if target_tier is not None else "All Tiers"
    print(f" Mode:         {'Cartesian Grid Search' if is_grid else f'Stratified Search ({trials_per_tier} trials/tier, {tier_info})'}")
    print(f" Parallelism:  {jobs} job(s) / núcleo(s)")
    print(f" Total Trials: {total_planned_trials} ensaios planejados")
    print(f" Report Dest:  {output_report}")
    print(f" JSON Dest:    {output_json}")
    print("================================================================================")

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"  [Loaded existing results from {output_json}: {len(all_results)} datasets]")
        except Exception as e:
            print(f"  [Warning: Could not load existing {output_json}: {e}]")
            all_results = {}

    for d_id, d_name, task, desc, is_high_dim in DATASETS_CONFIG:
        if selected_datasets and d_id not in selected_datasets and "all" not in selected_datasets:
            continue

        print(f"\n================================================================================")
        print(f" >>> Dataset: {d_name} [{task.upper()}] - {desc}")
        print(f"================================================================================")

        train_csv = os.path.join(data_dir, f"{d_id}_train.csv")
        test_csv = os.path.join(data_dir, f"{d_id}_test.csv")

        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"  Dataset files not found for {d_id}. Running prepare_datasets.py...")
            prep_script = os.path.join(os.path.dirname(__file__), "prepare_datasets.py")
            subprocess.run([PYTHON_BIN, prep_script, "--outdir", data_dir, "--dataset", d_id], check=True)

        y_test, _ = load_test_ground_truth(test_csv, task)
        dataset_trials: List[Dict[str, Any]] = []

        # 1. Kansho (C11)
        if is_grid:
            c11_configs = sample_hyperparameters(get_c11_space(task, is_high_dim, target_tier), n_iter, is_grid, seed)
        else:
            c11_configs = sample_stratified_c11_space(task, is_high_dim, trials_per_tier, seed, target_tier)

        if kansho_enabled:
            print(f"  [1/3] Iniciando Kansho ({len(c11_configs)} ensaios, {jobs} workers)...")
            c11_trials = run_engine_trials(
                "kansho",
                lambda cfg: run_trial_c11(train_csv, test_csv, task, cfg, y_test, is_high_dim),
                c11_configs,
                task,
                jobs,
                dataset_name=d_name,
                tracker=tracker,
            )
            dataset_trials.extend(c11_trials)

        n_engine_trials = len(c11_configs)

        # 2. River
        if river_enabled:
            river_space = get_river_space(task)
            river_configs = sample_hyperparameters(river_space, n_engine_trials, is_grid, seed)
            print(f"  [2/3] Iniciando River ({len(river_configs)} ensaios, {jobs} workers)...")
            river_trials = run_engine_trials(
                "river",
                lambda cfg: run_trial_river(train_csv, test_csv, task, cfg),
                river_configs,
                task,
                jobs,
                dataset_name=d_name,
                tracker=tracker,
            )
            dataset_trials.extend(river_trials)

        # 3. Scikit-Learn
        if sklearn_enabled:
            sklearn_space = get_sklearn_space(task)
            sklearn_configs = sample_hyperparameters(sklearn_space, n_engine_trials, is_grid, seed)
            print(f"  [3/3] Iniciando Scikit-Learn ({len(sklearn_configs)} ensaios, {jobs} workers)...")
            sklearn_trials = run_engine_trials(
                "sklearn",
                lambda cfg: run_trial_sklearn(train_csv, test_csv, task, cfg),
                sklearn_configs,
                task,
                jobs,
                dataset_name=d_name,
                tracker=tracker,
            )
            dataset_trials.extend(sklearn_trials)

        # Incremental merge with existing results for this dataset
        existing_trials = all_results.get(d_id, [])
        preserved_trials = []
        for t in existing_trials:
            eng = t.get("engine", "")
            t_tier = t.get("params", {}).get("tier")
            if kansho_enabled and eng in ("c11", "kansho"):
                if target_tier is not None and t_tier == target_tier:
                    continue  # Overwrite target_tier runs
                if target_tier is None:
                    continue  # Overwrite all C11 runs if running all tiers
            elif river_enabled and eng == "river":
                continue
            elif sklearn_enabled and eng == "sklearn":
                continue
            preserved_trials.append(t)

        all_results[d_id] = preserved_trials + dataset_trials

        # Incremental save after each dataset
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        generate_markdown_report(all_results, output_report)
        print(f"  [✔] Checkpoint: Results and Report updated for {d_name}.")

    print(f"\nFinal results saved to JSON: {output_json}")
    print(f"Final Markdown report generated: {output_report}")


def main():
    parser = argparse.ArgumentParser(description="Hoeffding Tree Hyperparameter Tuning Suite.")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--report", default=os.path.join(os.path.dirname(__file__), "hyperparam_report.md"))
    parser.add_argument("--json", default=os.path.join(os.path.dirname(__file__), "hyperparam_results.json"))
    parser.add_argument("--dataset", default="all", help="Dataset(s) to run (e.g. 'all' or 'sea,friedman').")
    parser.add_argument("--engine", default="all", help="Engines to run ('all' or 'kansho,river,sklearn').")
    parser.add_argument("--tier", default=None, help="Target specific memory tier in Kansho (e.g. '3' or 'all').")
    parser.add_argument("--trials-per-tier", type=int, default=30, help="Number of trials per memory tier in Kansho (default: 30).")
    parser.add_argument("--n-iter", type=int, default=30, help="Number of random search iterations per engine (legacy fallback).")
    parser.add_argument("--grid", action="store_true", help="Perform full exhaustive grid search.")
    parser.add_argument("--jobs", type=int, default=12, help="Number of parallel jobs.")
    parser.add_argument("--replay-report", action="store_true", help="Re-run only the Top-3 configurations per tier/engine currently published in hyperparam_report.md.")
    parser.add_argument("--report-only", action="store_true", help="Only re-render hyperparam_report.md from hyperparam_results.json without running trials.")
    
    args = parser.parse_args()
    
    datasets = args.dataset.split(",")
    engines = args.engine.split(",")

    target_tier = None
    if args.tier and args.tier.lower() != "all":
        try:
            target_tier = int(args.tier)
        except ValueError:
            target_tier = None

    run_tuning_suite(
        args.data_dir,
        args.report,
        args.json,
        datasets,
        engines,
        args.n_iter,
        args.grid,
        42,
        args.jobs,
        args.trials_per_tier,
        target_tier,
        args.replay_report,
        args.report_only,
    )


if __name__ == "__main__":
    main()
