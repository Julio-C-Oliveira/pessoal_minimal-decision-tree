#!/usr/bin/env python3
"""
Isolated Scikit-Learn Batch Runner for Main Benchmark.

Runs as an independent subprocess:
- Loads the full training dataset in RAM (representing true batch requirement).
- Performs model.fit() in a single batch.
- Measures train phase Peak RSS via OS (getrusage).
- Loads the full test dataset in RAM and performs model.predict().
- Measures test phase and total Peak RSS via OS.
- Computes intra-train model structural size + fit heap allocation (tracemalloc).
- Outputs standardized JSON to stdout.
"""

import argparse
import json
import resource
import sys
import time
import tracemalloc
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def get_peak_rss_mb() -> float:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_csv(path: str):
    n_features = 0
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if header.startswith("#"):
            tokens = header.lstrip("#").strip().split()
            for tok in tokens:
                if tok.startswith("n_features="):
                    n_features = int(tok.split("=")[1])
    data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float32)
    if n_features == 0:
        n_features = data.shape[1] - 1
    X = data[:, :n_features]
    y = data[:, n_features]
    return X, y, n_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--task", default="classification", choices=["classification", "regression"])
    args = parser.parse_args()

    tracemalloc.start()
    t_train0 = time.perf_counter()

    # Full batch ingestion in RAM
    X_train, y_train, n_features = load_csv(args.train_csv)

    if args.task == "regression":
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
    else:
        y_train = y_train.astype(np.int32)
        model = DecisionTreeClassifier(criterion="entropy", random_state=42)
        model.fit(X_train, y_train)

    train_time_s = time.perf_counter() - t_train0
    _, peak_heap_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    train_rss_mb = get_peak_rss_mb()

    # Test phase
    t_test0 = time.perf_counter()
    X_test, y_test, _ = load_csv(args.test_csv)
    y_pred = model.predict(X_test)
    test_time_s = time.perf_counter() - t_test0

    peak_rss_mb = get_peak_rss_mb()
    test_rss_mb = peak_rss_mb

    tree_ = model.tree_
    model_struct_bytes = (
        tree_.node_count * 96
        + tree_.value.nbytes
        + tree_.threshold.nbytes
        + tree_.feature.nbytes
        + tree_.children_left.nbytes
        + tree_.children_right.nbytes
    )
    intra_train_ram_kb = (model_struct_bytes + peak_heap_bytes) / 1024.0

    total_samples = len(X_train) + len(X_test)
    avg_us_per_sample = ((train_time_s + test_time_s) * 1e6) / max(1, total_samples)

    res = {
        "model": "Scikit-Learn (CART Batch)",
        "task": args.task,
        "n_samples": total_samples,
        "n_features": n_features,
        "n_nodes": int(tree_.node_count),
        "n_leaves": int(tree_.n_leaves),
        "intra_train_ram_kb": intra_train_ram_kb,
        "train_rss_mb": train_rss_mb,
        "test_rss_mb": test_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "avg_us_per_sample": avg_us_per_sample,
    }

    if args.task == "regression":
        res["holdout_mae"] = float(mean_absolute_error(y_test, y_pred))
        res["holdout_rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        res["holdout_r2"] = float(r2_score(y_test, y_pred))
        res["prequential_mae"] = None
    else:
        y_test = y_test.astype(np.int32)
        y_pred = y_pred.astype(np.int32)
        res["holdout_acc"] = float(accuracy_score(y_test, y_pred) * 100.0)
        res["f1_macro"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0) * 100.0)
        res["prequential_acc"] = None

    print(json.dumps(res))


if __name__ == "__main__":
    main()
