#!/usr/bin/env python3
"""
Stream generator for C11 Hoeffding Tree vs River Parity Benchmarks.

Generates reproducible synthetic datasets from river.datasets.synth:
- SEA (3 continuous numeric features, 2 classes)
- Agrawal (9 numeric features, 2 classes)
- Hyperplane (10 continuous numeric features, 2 classes)
- LED (7 binary features, 2 classes: even vs odd)
- STAGGER (3 nominal categorical features, 2 classes)

Saves CSV streams with an explicit metadata header line understood by both
bench_c_runner and compare_river.py.
"""

import argparse
import os
import sys
import numpy as np
from river.datasets import synth


def generate_sea(n_samples: int, seed: int):
    gen = synth.SEA(seed=seed)
    feature_names = [f"f{i}" for i in range(3)]
    feature_types = [0, 0, 0]  # numeric
    samples = []
    for x, y in gen.take(n_samples):
        # x is dict with keys 0, 1, 2; y is bool
        row_x = [float(x[i]) for i in range(3)]
        row_y = 1 if y else 0
        samples.append((row_x, row_y))
    return feature_names, feature_types, 2, samples


def generate_agrawal(n_samples: int, seed: int):
    gen = synth.Agrawal(seed=seed)
    keys = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue", "hyears", "loan"]
    feature_types = [0] * len(keys)  # treated as numeric in standard River benchmarks
    samples = []
    for x, y in gen.take(n_samples):
        row_x = [float(x[k]) for k in keys]
        row_y = int(y)
        samples.append((row_x, row_y))
    return keys, feature_types, 2, samples


def generate_hyperplane(n_samples: int, seed: int):
    gen = synth.Hyperplane(seed=seed)
    n_features = 10
    feature_names = [f"f{i}" for i in range(n_features)]
    feature_types = [0] * n_features
    samples = []
    for x, y in gen.take(n_samples):
        row_x = [float(x[i]) for i in range(n_features)]
        row_y = int(y)
        samples.append((row_x, row_y))
    return feature_names, feature_types, 2, samples


def generate_led(n_samples: int, seed: int, noise_percentage: float = 0.10):
    gen = synth.LED(seed=seed, noise_percentage=noise_percentage)
    n_features = 7
    feature_names = [f"seg_{i}" for i in range(n_features)]
    feature_types = [0] * n_features  # 0 or 1 numeric/binary
    samples = []
    for x, y in gen.take(n_samples):
        row_x = [float(x[i]) for i in range(n_features)]
        # Map 10 digits to 2 classes (even=0, odd=1) to fit within HT_MAX_CLASSES=4
        row_y = int(y % 2)
        samples.append((row_x, row_y))
    return feature_names, feature_types, 2, samples


def generate_stagger(n_samples: int, seed: int):
    gen = synth.STAGGER(seed=seed)
    keys = ["size", "color", "shape"]
    feature_types = [1, 1, 1]  # 1 = categorical nominal
    samples = []
    for x, y in gen.take(n_samples):
        row_x = [int(x[k]) for k in keys]
        row_y = int(y)
        samples.append((row_x, row_y))
    return keys, feature_types, 2, samples


def generate_sea_drift_abrupt(n_samples: int, seed: int):
    """SEA concept drift with abrupt transition from variant 0 to variant 3 at midpoint."""
    feature_names = [f"f{i}" for i in range(3)]
    feature_types = [0, 0, 0]
    position = n_samples // 2

    gen1 = synth.SEA(seed=seed, variant=0)
    gen2 = synth.SEA(seed=seed + 1, variant=3)

    samples = []
    for i, (x, y) in enumerate(gen1.take(position)):
        samples.append(([float(x[k]) for k in range(3)], 1 if y else 0))
    for i, (x, y) in enumerate(gen2.take(n_samples - position)):
        samples.append(([float(x[k]) for k in range(3)], 1 if y else 0))
    return feature_names, feature_types, 2, samples


def generate_sea_drift_gradual(n_samples: int, seed: int, width: int = 2000):
    """SEA concept drift with gradual transition from variant 0 to variant 3 around midpoint."""
    feature_names = [f"f{i}" for i in range(3)]
    feature_types = [0, 0, 0]
    position = n_samples // 2
    w = min(width, n_samples // 4)
    start = position - (w // 2)
    end = position + (w // 2)

    gen1 = synth.SEA(seed=seed, variant=0)
    gen2 = synth.SEA(seed=seed + 1, variant=3)
    rng = np.random.RandomState(seed)

    iter1 = iter(gen1)
    iter2 = iter(gen2)
    samples = []

    for i in range(n_samples):
        x1, y1 = next(iter1)
        x2, y2 = next(iter2)
        if i < start:
            x, y = x1, y1
        elif i > end:
            x, y = x2, y2
        else:
            p2 = float(i - start) / float(w)
            x, y = (x2, y2) if rng.rand() < p2 else (x1, y1)
        samples.append(([float(x[k]) for k in range(3)], 1 if y else 0))

    return feature_names, feature_types, 2, samples


def generate_agrawal_drift_abrupt(n_samples: int, seed: int):
    """Agrawal stream with abrupt transition from function 0 to function 2 at midpoint."""
    keys = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue", "hyears", "loan"]
    feature_types = [0] * len(keys)
    position = n_samples // 2

    gen1 = synth.Agrawal(seed=seed, classification_function=0)
    gen2 = synth.Agrawal(seed=seed + 1, classification_function=2)

    samples = []
    for i, (x, y) in enumerate(gen1.take(position)):
        samples.append(([float(x[k]) for k in keys], int(y)))
    for i, (x, y) in enumerate(gen2.take(n_samples - position)):
        samples.append(([float(x[k]) for k in keys], int(y)))
    return keys, feature_types, 2, samples


def generate_agrawal_drift_gradual(n_samples: int, seed: int, width: int = 2000):
    """Agrawal stream with gradual transition from function 0 to function 2 around midpoint."""
    keys = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue", "hyears", "loan"]
    feature_types = [0] * len(keys)
    position = n_samples // 2
    w = min(width, n_samples // 4)
    start = position - (w // 2)
    end = position + (w // 2)

    gen1 = synth.Agrawal(seed=seed, classification_function=0)
    gen2 = synth.Agrawal(seed=seed + 1, classification_function=2)
    rng = np.random.RandomState(seed)

    iter1 = iter(gen1)
    iter2 = iter(gen2)
    samples = []

    for i in range(n_samples):
        x1, y1 = next(iter1)
        x2, y2 = next(iter2)
        if i < start:
            x, y = x1, y1
        elif i > end:
            x, y = x2, y2
        else:
            p2 = float(i - start) / float(w)
            x, y = (x2, y2) if rng.rand() < p2 else (x1, y1)
        samples.append(([float(x[k]) for k in keys], int(y)))

    return keys, feature_types, 2, samples


def generate_hyperplane_drift(n_samples: int, seed: int):
    """Hyperplane stream with continuous rotating hyperplane drift."""
    n_features = 10
    feature_names = [f"f{i}" for i in range(n_features)]
    feature_types = [0] * n_features

    gen = synth.Hyperplane(seed=seed, n_features=n_features, n_drift_features=2, mag_change=0.001)
    samples = []
    for x, y in gen.take(n_samples):
        row_x = [float(x[i]) for i in range(n_features)]
        row_y = int(y)
        samples.append((row_x, row_y))
    return feature_names, feature_types, 2, samples


STREAM_GENERATORS = {
    "sea": generate_sea,
    "agrawal": generate_agrawal,
    "hyperplane": generate_hyperplane,
    "led": generate_led,
    "stagger": generate_stagger,
    "sea_drift_abrupt": generate_sea_drift_abrupt,
    "sea_drift_gradual": generate_sea_drift_gradual,
    "agrawal_drift_abrupt": generate_agrawal_drift_abrupt,
    "agrawal_drift_gradual": generate_agrawal_drift_gradual,
    "hyperplane_drift": generate_hyperplane_drift,
}


def write_csv(filepath: str, stream_name: str, feature_names: list, feature_types: list, n_classes: int, samples: list):
    n_samples = len(samples)
    n_features = len(feature_names)
    types_str = ",".join(str(t) for t in feature_types)
    names_str = ",".join(feature_names)

    with open(filepath, "w", encoding="utf-8") as f:
        # Header line with structured metadata
        f.write(
            f"# stream={stream_name} n_samples={n_samples} n_features={n_features} "
            f"n_classes={n_classes} feature_types={types_str} feature_names={names_str}\n"
        )
        for row_x, row_y in samples:
            line_vals = [str(val) for val in row_x]
            line_vals.append(str(row_y))
            f.write(",".join(line_vals) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic streams for Hoeffding Tree parity benchmarks.")
    parser.add_argument("--outdir", default="benchmarks/data", help="Output directory for CSV stream files.")
    parser.add_argument("--samples", type=int, default=20000, help="Number of samples to generate per stream.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stream generators.")
    parser.add_argument("--streams", nargs="+", choices=list(STREAM_GENERATORS.keys()), default=list(STREAM_GENERATORS.keys()),
                        help="Subset of streams to generate.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("==================================================")
    print(" Hoeffding Tree Synthetic Stream Generator")
    print(f" Output directory: {args.outdir}")
    print(f" Default samples:  {args.samples}")
    print(f" Random seed:      {args.seed}")
    print("==================================================")

    for stream_name in args.streams:
        n_samples = args.samples
        if stream_name == "stagger" and n_samples > 10000:
            n_samples = 10000  # STAGGER is periodic and simple, 10k is standard

        print(f"Generating stream: {stream_name} ({n_samples} samples)...", end="", flush=True)
        gen_fn = STREAM_GENERATORS[stream_name]
        names, types, n_classes, samples = gen_fn(n_samples, args.seed)
        out_path = os.path.join(args.outdir, f"{stream_name}.csv")
        write_csv(out_path, stream_name, names, types, n_classes, samples)
        file_size_kb = os.path.getsize(out_path) / 1024.0
        print(f" Done -> {out_path} ({file_size_kb:.1f} KB)")

    print("All streams generated successfully.")


if __name__ == "__main__":
    main()
