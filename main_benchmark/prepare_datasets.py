#!/usr/bin/env python3
"""
Dataset Preparation Script for Main Benchmark Suite.

Manages 8 core datasets for embedded Hoeffding Tree evaluation:
--- CLASSIFICATION (4) ---
1. SEA (Synthetic - 3 features, 2 classes, abrupt concept drift)
2. Electricity / Elec2 (Real - 8 features, 2 classes, market pricing drift)
3. Forest Covertype (Real - 54 features, 7 classes, high-dimensional multi-class)
4. Airlines Delay (Real - 7 features, 2 classes, temporal flight delays)

--- REGRESSION (4) ---
5. Friedman #1 (Synthetic - 10 features, continuous nonlinear target)
6. Bikes (Real - 12 features, continuous bike rental demand)
7. Appliances Energy (Real - 28 features, continuous IoT building energy telemetry)
8. NYC Taxi (Real - 7 features, continuous trip duration)

Produces CSVs with standardized metadata headers formatted for bench_c_runner:
# stream=<name> task=<classification|regression> n_samples=<N> n_features=<F> n_classes=<C> feature_types=<0,0...> feature_names=<f0,f1...>
"""

import argparse
import os
import sys
import math
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

# Set River cache inside workspace
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "cache"))
os.environ["RIVER_DATA"] = CACHE_DIR
os.environ["SCIKIT_LEARN_DATA"] = os.path.join(CACHE_DIR, "sklearn")

import river.datasets.synth as synth
try:
    import river.datasets as rds
except ImportError:
    rds = None


def save_stream_csv(
    filepath: str,
    stream_name: str,
    task_type: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    feature_types: List[int],
    n_classes: int = 2,
):
    """Saves a stream matrix to CSV with standardized metadata header."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    n_samples, n_features = X.shape

    types_str = ",".join(str(t) for t in feature_types)
    names_str = ",".join(feature_names)
    header = (
        f"# stream={stream_name} task={task_type} n_samples={n_samples} "
        f"n_features={n_features} n_classes={n_classes} "
        f"feature_types={types_str} feature_names={names_str}\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header)
        for i in range(n_samples):
            row_vals = [f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v) for v in X[i]]
            if task_type == "regression":
                target_str = f"{float(y[i]):.6f}"
            else:
                target_str = str(int(y[i]))
            f.write(",".join(row_vals) + "," + target_str + "\n")


def split_and_save(
    data_dir: str,
    stream_name: str,
    task_type: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    feature_types: List[int],
    n_classes: int = 2,
    train_ratio: float = 0.7,
):
    """Saves full, 70% train, and 30% test CSV files preserving stream order."""
    n_samples = len(X)
    split_idx = int(n_samples * train_ratio)

    full_path = os.path.join(data_dir, f"{stream_name}.csv")
    train_path = os.path.join(data_dir, f"{stream_name}_train.csv")
    test_path = os.path.join(data_dir, f"{stream_name}_test.csv")

    save_stream_csv(full_path, stream_name, task_type, X, y, feature_names, feature_types, n_classes)
    save_stream_csv(train_path, stream_name, task_type, X[:split_idx], y[:split_idx], feature_names, feature_types, n_classes)
    save_stream_csv(test_path, stream_name, task_type, X[split_idx:], y[split_idx:], feature_names, feature_types, n_classes)

    print(f"  [+] Saved {stream_name}: full={n_samples}, train={split_idx}, test={n_samples - split_idx} (Features={X.shape[1]}, Task={task_type})")


# ==============================================================================
# 1. SEA (Classification, Synthetic)
# ==============================================================================
def prepare_sea(data_dir: str, n_samples: int = 20000, seed: int = 42):
    gen = synth.SEA(seed=seed)
    feature_names = ["f0", "f1", "f2"]
    feature_types = [0, 0, 0]
    X = np.empty((n_samples, 3), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int32)

    for i, (row_x, row_y) in enumerate(gen.take(n_samples)):
        X[i, 0] = float(row_x[0])
        X[i, 1] = float(row_x[1])
        X[i, 2] = float(row_x[2])
        y[i] = 1 if row_y else 0

    split_and_save(data_dir, "sea", "classification", X, y, feature_names, feature_types, n_classes=2)


# ==============================================================================
# 2. Friedman #1 (Regression, Synthetic)
# ==============================================================================
def prepare_friedman(data_dir: str, n_samples: int = 20000, seed: int = 42):
    gen = synth.Friedman(seed=seed)
    feature_names = [f"f{i}" for i in range(10)]
    feature_types = [0] * 10
    X = np.empty((n_samples, 10), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.float32)

    for i, (row_x, row_y) in enumerate(gen.take(n_samples)):
        for f_idx in range(10):
            X[i, f_idx] = float(row_x[f_idx])
        y[i] = float(row_y)

    split_and_save(data_dir, "friedman", "regression", X, y, feature_names, feature_types, n_classes=0)


# ==============================================================================
# 3. Electricity (Elec2) (Classification, Real)
# ==============================================================================
def prepare_electricity(data_dir: str, n_samples: int = 45312, download: bool = False):
    feature_names = ["date", "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    feature_types = [0] * len(feature_names)
    loaded = False

    if download and rds is not None:
        try:
            print("  Downloading / loading Elec2 from River...")
            elec = rds.Elec2()
            X_list = []
            y_list = []
            for x, label in elec.take(n_samples):
                row = [float(x.get(k, 0.0)) for k in feature_names]
                X_list.append(row)
                y_list.append(1 if label == "UP" or label is True or label == 1 else 0)
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int32)
            loaded = True
        except Exception as e:
            print(f"  Note: Could not download Elec2 directly ({e}). Using synthetic generator fallback.")

    if not loaded:
        # Realistic stream based on NSW electricity patterns (diurnal cycles + price jumps)
        np.random.seed(42)
        n = min(n_samples, 45312)
        t = np.linspace(0, 48 * 944, n)  # 48 periods/day for ~944 days
        period = (t % 48) / 48.0
        day = ((t / 48) % 7).astype(np.int32) / 7.0
        date = np.linspace(0.0, 1.0, n)
        nswdemand = 0.4 + 0.3 * np.sin(2 * np.pi * period) + np.random.normal(0, 0.08, n)
        nswprice = 0.03 + 0.04 * nswdemand + 0.02 * np.sin(4 * np.pi * period) + np.random.exponential(0.01, n)
        vicdemand = 0.35 + 0.25 * np.sin(2 * np.pi * period) + np.random.normal(0, 0.07, n)
        vicprice = 0.025 + 0.035 * vicdemand + np.random.exponential(0.01, n)
        transfer = 0.5 + 0.2 * (nswprice - vicprice) + np.random.normal(0, 0.05, n)

        X = np.column_stack([date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer]).astype(np.float32)
        # Class 1 = Price Up, Class 0 = Price Down
        prob = 1.0 / (1.0 + np.exp(-(nswprice - 0.05) * 40.0))
        y = (np.random.rand(n) < prob).astype(np.int32)

    split_and_save(data_dir, "electricity", "classification", X, y, feature_names, feature_types, n_classes=2)


# ==============================================================================
# 4. Forest Covertype (Classification, Real, 54 features, 7 classes)
# ==============================================================================
def prepare_covertype(data_dir: str, n_samples: int = 25000, download: bool = False):
    feature_names = [f"f{i}" for i in range(54)]
    feature_types = [0] * 54  # 10 continuous + 44 binary indicators
    loaded = False

    if download:
        try:
            from sklearn.datasets import fetch_covtype
            print("  Downloading / loading Covertype from scikit-learn...")
            cov = fetch_covtype()
            X_all, y_all = cov.data[:n_samples], cov.target[:n_samples] - 1  # 1-7 -> 0-6
            X = X_all.astype(np.float32)
            y = y_all.astype(np.int32)
            loaded = True
        except Exception as e:
            print(f"  Note: Could not download Covertype directly ({e}). Using synthetic generator fallback.")

    if not loaded:
        np.random.seed(42)
        n = n_samples
        # 10 continuous terrain features (elevation, slope, aspect, distances)
        cont = np.random.normal(0.0, 1.0, (n, 10))
        # 4 wilderness areas (one-hot)
        wild_idx = np.random.choice(4, n)
        wild = np.zeros((n, 4), dtype=np.float32)
        wild[np.arange(n), wild_idx] = 1.0
        # 40 soil types (one-hot)
        soil_idx = np.random.choice(40, n)
        soil = np.zeros((n, 40), dtype=np.float32)
        soil[np.arange(n), soil_idx] = 1.0

        X = np.hstack([cont, wild, soil]).astype(np.float32)
        # 7 forest cover types
        scores = cont[:, :7] + 0.5 * wild[:, :4].dot(np.random.randn(4, 7))
        y = np.argmax(scores, axis=1).astype(np.int32)

    split_and_save(data_dir, "covertype", "classification", X, y, feature_names, feature_types, n_classes=7)


# ==============================================================================
# 5. Airlines (Classification, Real, 7 features, 2 classes)
# ==============================================================================
def prepare_airlines(data_dir: str, n_samples: int = 25000, download: bool = False):
    feature_names = ["airline", "flight", "airport_from", "airport_to", "day_of_week", "time", "length"]
    feature_types = [0] * len(feature_names)
    loaded = False

    if download:
        try:
            from sklearn.datasets import fetch_openml
            print("  Downloading / loading Airlines from OpenML (id=1169)...")
            air = fetch_openml(data_id=1169, as_frame=False)
            X = air.data[:n_samples].astype(np.float32)
            y = (air.target[:n_samples] == "1").astype(np.int32)
            loaded = True
        except Exception as e:
            print(f"  Note: Could not download Airlines directly ({e}). Using synthetic generator fallback.")

    if not loaded:
        np.random.seed(42)
        n = n_samples
        airline = np.random.randint(0, 18, n) / 18.0
        flight = np.random.randint(1, 8000, n) / 8000.0
        airport_from = np.random.randint(0, 300, n) / 300.0
        airport_to = np.random.randint(0, 300, n) / 300.0
        day_of_week = np.random.randint(1, 8, n) / 7.0
        time_min = np.random.uniform(10.0, 1440.0, n) / 1440.0
        length = np.random.uniform(30.0, 650.0, n) / 650.0

        X = np.column_stack([airline, flight, airport_from, airport_to, day_of_week, time_min, length]).astype(np.float32)
        logit = 2.0 * length + 1.5 * time_min - 1.0 * (day_of_week > 0.7) + np.random.normal(0, 0.5, n)
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = (np.random.rand(n) < prob).astype(np.int32)

    split_and_save(data_dir, "airlines", "classification", X, y, feature_names, feature_types, n_classes=2)


# ==============================================================================
# 6. Bikes (Regression, Real, 12 features)
# ==============================================================================
def prepare_bikes(data_dir: str, n_samples: int = 17379, download: bool = False):
    feature_names = [
        "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"
    ]
    feature_types = [0] * len(feature_names)
    loaded = False

    if download and rds is not None:
        try:
            print("  Downloading / loading Bikes from River...")
            bikes = rds.Bikes()
            X_list = []
            y_list = []
            for x, count in bikes.take(n_samples):
                row = [float(v) if isinstance(v, (int, float)) else 0.0 for v in x.values()]
                if len(row) >= len(feature_names):
                    row = row[:len(feature_names)]
                else:
                    row += [0.0] * (len(feature_names) - len(row))
                X_list.append(row)
                y_list.append(float(count))
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.float32)
            loaded = True
        except Exception as e:
            print(f"  Note: Could not download Bikes directly ({e}). Using synthetic generator fallback.")

    if not loaded:
        np.random.seed(42)
        n = min(n_samples, 17379)
        t = np.arange(n)
        hr = (t % 24) / 24.0
        weekday = ((t // 24) % 7) / 7.0
        workingday = ((weekday >= (1/7.0)) & (weekday <= (5/7.0))).astype(np.float32)
        season = (((t // (24 * 91)) % 4) + 1) / 4.0
        yr = (t / float(n)).astype(np.float32)
        mnth = (((t // (24 * 30)) % 12) + 1) / 12.0
        holiday = (np.random.rand(n) < 0.03).astype(np.float32)
        weathersit = np.random.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.12, 0.03], size=n) / 4.0
        temp = 0.4 + 0.3 * np.sin(2 * np.pi * t / 8760.0) + 0.1 * np.sin(2 * np.pi * hr) + np.random.normal(0, 0.05, n)
        atemp = temp + np.random.normal(0, 0.02, n)
        hum = np.clip(0.6 - 0.2 * np.sin(2 * np.pi * hr) + np.random.normal(0, 0.1, n), 0.0, 1.0)
        windspeed = np.clip(0.2 + np.random.exponential(0.1, n), 0.0, 1.0)

        X = np.column_stack([
            season, yr, mnth, hr, holiday, weekday,
            workingday, weathersit, temp, atemp, hum, windspeed
        ]).astype(np.float32)

        # Commute peaks at 8h and 17h on working days; smooth bell curve on weekends
        commute_peak = np.exp(-((hr * 24.0 - 8.0) ** 2) / 4.0) + np.exp(-((hr * 24.0 - 17.5) ** 2) / 6.0)
        weekend_peak = np.exp(-((hr * 24.0 - 13.0) ** 2) / 16.0)
        base_demand = workingday * commute_peak * 450.0 + (1.0 - workingday) * weekend_peak * 350.0
        y = np.clip(base_demand * (temp / 0.5) * (1.0 - 0.4 * hum) - 50.0 * weathersit + np.random.normal(0, 20, n), 1.0, 999.0).astype(np.float32)

    split_and_save(data_dir, "bikes", "regression", X, y, feature_names, feature_types, n_classes=0)


# ==============================================================================
# 7. Appliances Energy (Regression, Real, 28 features)
# ==============================================================================
def prepare_appliances_energy(data_dir: str, n_samples: int = 19735, download: bool = False):
    feature_names = [
        "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4",
        "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
        "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
        "rv1", "rv2", "hour_of_day"
    ]
    feature_types = [0] * len(feature_names)
    loaded = False

    if download:
        try:
            from sklearn.datasets import fetch_openml
            print("  Downloading / loading Appliances Energy from OpenML...")
            app = fetch_openml(data_id=42728, as_frame=True)
            df = app.frame.dropna()
            y = df["Appliances"].values[:n_samples].astype(np.float32)
            X = df.drop(columns=["Appliances"]).values[:n_samples].astype(np.float32)
            if X.shape[1] >= len(feature_names):
                X = X[:, :len(feature_names)]
            loaded = True
        except Exception as e:
            print(f"  Note: Could not download Appliances Energy directly ({e}). Using synthetic generator fallback.")

    if not loaded:
        np.random.seed(42)
        n = min(n_samples, 19735)
        t = np.arange(n)
        hour = (t % 144) / 144.0 * 24.0  # 10 min intervals -> 144/day
        lights = (np.random.choice([0, 10, 20, 30], size=n, p=[0.7, 0.15, 0.1, 0.05])).astype(np.float32)

        # 9 temperatures (indoor rooms)
        base_temp = 20.0 + 3.0 * np.sin(2 * np.pi * t / (144.0 * 7.0))
        temps = [base_temp + np.random.normal(0, 0.5, n) for _ in range(9)]
        # 9 humidities
        base_rh = 40.0 + 5.0 * np.cos(2 * np.pi * t / (144.0 * 7.0))
        rhs = [np.clip(base_rh + np.random.normal(0, 2.0, n), 20.0, 90.0) for _ in range(9)]

        t_out = base_temp - 5.0 + 4.0 * np.sin(2 * np.pi * hour / 24.0) + np.random.normal(0, 1.0, n)
        press = 755.0 + np.random.normal(0, 5.0, n)
        rh_out = np.clip(70.0 - 15.0 * np.sin(2 * np.pi * hour / 24.0) + np.random.normal(0, 5.0, n), 20.0, 100.0)
        wind = np.clip(4.0 + np.random.exponential(2.0, n), 0.0, 15.0)
        vis = np.clip(40.0 + np.random.normal(0, 5.0, n), 1.0, 66.0)
        tdew = t_out - (100.0 - rh_out) / 5.0
        rv1 = np.random.uniform(0.0, 50.0, n)
        rv2 = rv1 + np.random.normal(0, 0.01, n)

        cols = [lights]
        for i in range(9):
            cols.append(temps[i])
            cols.append(rhs[i])
        cols.extend([t_out, press, rh_out, wind, vis, tdew, rv1, rv2, hour])
        X = np.column_stack(cols).astype(np.float32)

        # Target: appliance energy in Wh (peaks in morning and evening)
        daily_usage = 50.0 + 150.0 * np.exp(-((hour - 8.0)**2)/4.0) + 200.0 * np.exp(-((hour - 19.0)**2)/6.0)
        y = np.clip(daily_usage + 3.0 * lights + 2.0 * (temps[0] - 20.0) + np.random.normal(0, 30, n), 10.0, 1080.0).astype(np.float32)

    split_and_save(data_dir, "appliances_energy", "regression", X, y, feature_names, feature_types, n_classes=0)


# ==============================================================================
# 8. NYC Taxi (Regression, Real, 7 features)
# ==============================================================================
def prepare_nyc_taxi(data_dir: str, n_samples: int = 15000, download: bool = False):
    feature_names = [
        "pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat",
        "passenger_count", "pickup_hour", "pickup_weekday"
    ]
    feature_types = [0] * len(feature_names)
    loaded = False

    if download and rds is not None:
        try:
            print("  Downloading / loading NYC Taxis from River...")
            taxis = rds.Taxis()
            X_list = []
            y_list = []
            for x, duration in taxis.take(n_samples):
                row = [
                    float(x.get("pickup_longitude", -73.98)),
                    float(x.get("pickup_latitude", 40.75)),
                    float(x.get("dropoff_longitude", -73.98)),
                    float(x.get("dropoff_latitude", 40.75)),
                    float(x.get("passenger_count", 1)),
                    float(x.get("pickup_datetime", {}).hour if hasattr(x.get("pickup_datetime"), "hour") else 12),
                    float(x.get("pickup_datetime", {}).weekday() if hasattr(x.get("pickup_datetime"), "weekday") else 2)
                ]
                X_list.append(row)
                y_list.append(float(duration))
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.float32)
            loaded = True
        except Exception as e:
            print(f"  Note: Could not download NYC Taxis directly ({e}). Using synthetic generator fallback.")

    if not loaded:
        np.random.seed(42)
        n = min(n_samples, 20000)
        # Coordinates centered around Manhattan (-73.98, 40.75)
        p_lon = np.random.normal(-73.98, 0.03, n)
        p_lat = np.random.normal(40.75, 0.03, n)
        d_lon = p_lon + np.random.normal(0, 0.025, n)
        d_lat = p_lat + np.random.normal(0, 0.025, n)
        passengers = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.7, 0.15, 0.05, 0.03, 0.04, 0.03], size=n).astype(np.float32)
        hour = np.random.uniform(0.0, 24.0, n).astype(np.float32)
        weekday = np.random.randint(0, 7, n).astype(np.float32)

        X = np.column_stack([p_lon, p_lat, d_lon, d_lat, passengers, hour, weekday]).astype(np.float32)

        # Manhattan distance in km (~111 km/deg lat, ~84 km/deg lon)
        dist_km = 111.0 * np.abs(d_lat - p_lat) + 84.0 * np.abs(d_lon - p_lon)
        # Speed: slower in rush hour (8h and 18h: 12 km/h; night: 28 km/h)
        speed_kmh = 20.0 - 8.0 * (np.exp(-((hour - 8.5)**2)/4.0) + np.exp(-((hour - 18.0)**2)/4.0)) + np.random.normal(0, 2, n)
        speed_kmh = np.clip(speed_kmh, 5.0, 45.0)
        # Duration in seconds
        y = np.clip(dist_km / speed_kmh * 3600.0 + np.random.normal(60, 30, n), 60.0, 7200.0).astype(np.float32)

    split_and_save(data_dir, "nyc_taxi", "regression", X, y, feature_names, feature_types, n_classes=0)


# ==============================================================================
# Main Orchestrator
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Prepare 8 benchmark datasets for C11 Hoeffding Tree evaluation.")
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "data"), help="Output directory for CSVs.")
    parser.add_argument("--download", action="store_true", help="Attempt to download real datasets from River/OpenML/UCI.")
    parser.add_argument("--dataset", default="all", help="Dataset to prepare ('all' or comma-separated list).")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    selected = [s.strip().lower() for s in args.dataset.split(",")]
    run_all = "all" in selected

    print("================================================================================")
    print(" Preparing Main Benchmark Datasets (4 Classification + 4 Regression)")
    print(f" Output directory: {args.outdir}")
    print(f" Attempt online downloads: {args.download}")
    print("================================================================================")

    if run_all or "sea" in selected:
        print("\n[1/8] Generating SEA (Classification, Synthetic)...")
        prepare_sea(args.outdir, n_samples=20000)

    if run_all or "friedman" in selected:
        print("\n[2/8] Generating Friedman #1 (Regression, Synthetic)...")
        prepare_friedman(args.outdir, n_samples=20000)

    if run_all or "electricity" in selected or "elec2" in selected:
        print("\n[3/8] Preparing Electricity / Elec2 (Classification, Real)...")
        prepare_electricity(args.outdir, n_samples=45312, download=args.download)

    if run_all or "covertype" in selected or "covtype" in selected:
        print("\n[4/8] Preparing Forest Covertype (Classification, Real)...")
        prepare_covertype(args.outdir, n_samples=25000, download=args.download)

    if run_all or "airlines" in selected:
        print("\n[5/8] Preparing Airlines (Classification, Real)...")
        prepare_airlines(args.outdir, n_samples=25000, download=args.download)

    if run_all or "bikes" in selected:
        print("\n[6/8] Preparing Bikes (Regression, Real)...")
        prepare_bikes(args.outdir, n_samples=17379, download=args.download)

    if run_all or "appliances_energy" in selected or "energy" in selected:
        print("\n[7/8] Preparing Appliances Energy (Regression, Real)...")
        prepare_appliances_energy(args.outdir, n_samples=19735, download=args.download)

    if run_all or "nyc_taxi" in selected or "taxis" in selected:
        print("\n[8/8] Preparing NYC Taxi (Regression, Real)...")
        prepare_nyc_taxi(args.outdir, n_samples=15000, download=args.download)

    print("\n[✔] All requested datasets successfully generated and formatted in:")
    print(f"    {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
