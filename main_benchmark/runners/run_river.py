#!/usr/bin/env python3
"""
Isolated River Streaming Runner for Main Benchmark.

Runs as an independent subprocess:
- Consumes dataset line by line (strictly 1 sample at a time via generator).
- Performs online incremental learning (learn_one).
- Measures train phase Peak RSS via OS (getrusage).
- Consumes test dataset line by line (predict_one).
- Measures test phase and total Peak RSS via OS.
- Computes intra-train model structural size + dynamic heap allocation (tracemalloc).
- Outputs standardized JSON to stdout.
"""

import argparse
import csv
import json
import resource
import sys
import time
import tracemalloc
import numpy as np
import river
from river import tree, metrics
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


def stream_csv(path: str):
    """Yields (x_dict, y_val, n_features, feat_types, feat_names) one sample at a time."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        n_features = 0
        feat_types = []
        if header.startswith("#"):
            tokens = header.lstrip("#").strip().split()
            for tok in tokens:
                if tok.startswith("n_features="):
                    n_features = int(tok.split("=")[1])
                elif tok.startswith("feature_types="):
                    feat_types = [int(x) for x in tok.split("=")[1].split(",")]

        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if n_features == 0:
                n_features = len(row) - 1
            if not feat_types:
                feat_types = [0] * n_features

            x_dict = {
                i: int(row[i]) if (i < len(feat_types) and feat_types[i] == 1) else float(row[i])
                for i in range(n_features)
            }
            y_val = float(row[n_features])
            yield x_dict, y_val, n_features, feat_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--task", default="classification", choices=["classification", "regression"])
    args = parser.parse_args()

    tracemalloc.start()
    t_train0 = time.perf_counter()

    first_sample = next(stream_csv(args.train_csv))
    n_features = first_sample[2]
    feat_types = first_sample[3]
    nominal_attrs = [i for i, t in enumerate(feat_types) if t == 1]

    if args.task == "regression":
        model = tree.HoeffdingTreeRegressor(
            grace_period=200,
            delta=1e-7,
            tau=0.05,
            leaf_prediction="adaptive",
            min_samples_split=5,
            binary_split=True,
            nominal_attributes=nominal_attrs if nominal_attrs else None,
        )
        mae_metric = metrics.MAE()
    else:
        model = tree.HoeffdingTreeClassifier(
            grace_period=200,
            delta=1e-7,
            tau=0.05,
            split_criterion="info_gain",
            leaf_prediction="nba",
            nominal_attributes=nominal_attrs if nominal_attrs else None,
            splitter=tree.splitter.GaussianSplitter(n_splits=10),
            binary_split=True,
        )
        acc_metric = metrics.Accuracy()

    # Streaming Training (1 sample per iteration)
    train_samples = 0
    for x_dict, y_val, _, _ in stream_csv(args.train_csv):
        train_samples += 1
        if args.task == "regression":
            pred = model.predict_one(x_dict)
            if pred is not None:
                mae_metric.update(y_val, pred)
            model.learn_one(x_dict, y_val)
        else:
            y_label = int(y_val)
            pred = model.predict_one(x_dict)
            if pred is not None:
                acc_metric.update(y_label, pred)
            model.learn_one(x_dict, y_label)

    train_time_s = time.perf_counter() - t_train0
    _, peak_heap_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    train_rss_mb = get_peak_rss_mb()

    # Streaming Testing (1 sample per iteration)
    t_test0 = time.perf_counter()
    y_trues = []
    y_preds = []
    test_samples = 0

    for x_dict, y_val, _, _ in stream_csv(args.test_csv):
        test_samples += 1
        pred = model.predict_one(x_dict)
        if args.task == "regression":
            y_trues.append(y_val)
            y_preds.append(float(pred) if pred is not None else 0.0)
        else:
            y_trues.append(int(y_val))
            y_preds.append(int(pred) if pred is not None else 0)

    test_time_s = time.perf_counter() - t_test0

    peak_rss_mb = get_peak_rss_mb()
    test_rss_mb = peak_rss_mb

    n_nodes = getattr(model, "n_nodes", len(list(model.iter_nodes())) if hasattr(model, "iter_nodes") else 1)
    n_leaves = getattr(model, "n_active_leaves", 1) + getattr(model, "n_inactive_leaves", 0)

    model_struct_bytes = sys.getsizeof(model) + (n_nodes * 128)
    intra_train_ram_kb = (model_struct_bytes + peak_heap_bytes) / 1024.0

    total_samples = train_samples + test_samples
    avg_us_per_sample = ((train_time_s + test_time_s) * 1e6) / max(1, total_samples)

    res = {
        "model": "River (" + ("HTR" if args.task == "regression" else "HTC") + ")",
        "task": args.task,
        "n_samples": total_samples,
        "n_features": n_features,
        "n_nodes": n_nodes,
        "n_leaves": n_leaves,
        "intra_train_ram_kb": intra_train_ram_kb,
        "train_rss_mb": train_rss_mb,
        "test_rss_mb": test_rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "avg_us_per_sample": avg_us_per_sample,
    }

    if args.task == "regression":
        res["holdout_mae"] = float(mean_absolute_error(y_trues, y_preds))
        res["holdout_rmse"] = float(np.sqrt(mean_squared_error(y_trues, y_preds)))
        res["holdout_r2"] = float(r2_score(y_trues, y_preds))
        res["prequential_mae"] = float(mae_metric.get())
    else:
        res["holdout_acc"] = float(accuracy_score(y_trues, y_preds) * 100.0)
        res["f1_macro"] = float(f1_score(y_trues, y_preds, average="macro", zero_division=0) * 100.0)
        res["prequential_acc"] = float(acc_metric.get() * 100.0)

    print(json.dumps(res))


if __name__ == "__main__":
    main()
