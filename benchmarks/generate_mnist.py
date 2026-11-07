#!/usr/bin/env python3
"""
MNIST Stream Generator with Spatial Average Pooling (TinyML / Edge Vision).

Processes the local MNIST dataset (784 features, 28x28 pixels) into
embedded-friendly streams with 16 continuous spatial features via 4x4 pooling:
- mnist_10class.csv: 10 digit classes (0..9)
- mnist_4class.csv: 4 digit classes (0, 1, 2, 3)
- mnist_binary.csv: 2 classes (even vs odd digits)

Header format is fully compatible with bench_c_runner and compare_triple.py.
"""

import argparse
import os
import sys
import numpy as np
from sklearn.datasets import fetch_openml, load_digits


def load_mnist_raw(cache_dir: str):
    try:
        print(f"Loading MNIST from cache ({cache_dir})...", end="", flush=True)
        mnist = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            parser="liac-arff",
            data_home=cache_dir,
        )
        X = mnist.data.astype(np.float32)
        y = mnist.target.astype(np.int32)
        print(f" Done. Loaded {len(X)} samples (28x28).")
        return X, y, 28
    except Exception as e:
        print(f"\nWarning: Failed to load fetch_openml: {e}")
        print("Falling back to local load_digits (8x8)...", end="", flush=True)
        digits = load_digits()
        X = digits.data.astype(np.float32)
        y = digits.target.astype(np.int32)
        print(f" Done. Loaded {len(X)} samples (8x8).")
        return X, y, 8


def pool_images_to_16_features(X_raw: np.ndarray, img_dim: int) -> np.ndarray:
    """Pools square images to 4x4 blocks (16 features) normalized to [0.0, 1.0]."""
    n_samples = len(X_raw)
    if img_dim == 28:
        # 28x28 -> 4x4 blocks of 7x7 pixels
        X_imgs = X_raw.reshape(n_samples, 28, 28)
        X_pooled = X_imgs.reshape(n_samples, 4, 7, 4, 7).mean(axis=(2, 4))
        X_norm = (X_pooled / 255.0).reshape(n_samples, 16)
    elif img_dim == 8:
        # 8x8 -> 4x4 blocks of 2x2 pixels
        X_imgs = X_raw.reshape(n_samples, 8, 8)
        X_pooled = X_imgs.reshape(n_samples, 4, 2, 4, 2).mean(axis=(2, 4))
        X_norm = (X_pooled / 16.0).reshape(n_samples, 16)
    else:
        raise ValueError(f"Unsupported image dimension: {img_dim}")
    return X_norm


def write_csv_stream(
    filepath: str,
    stream_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
):
    n_samples, n_features = X.shape
    feature_names = [f"p_{r}_{c}" for r in range(4) for c in range(4)]
    feature_types = [0] * n_features  # 0 = numeric continuous
    types_str = ",".join(str(t) for t in feature_types)
    names_str = ",".join(feature_names)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(
            f"# stream={stream_name} n_samples={n_samples} n_features={n_features} "
            f"n_classes={n_classes} feature_types={types_str} feature_names={names_str}\n"
        )
        for i in range(n_samples):
            line_vals = [f"{v:.6f}" for v in X[i]]
            line_vals.append(str(int(y[i])))
            f.write(",".join(line_vals) + "\n")


def generate_all_mnist_streams(out_dir: str, cache_dir: str, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)

    X_raw, y_raw, img_dim = load_mnist_raw(cache_dir)
    X_pooled = pool_images_to_16_features(X_raw, img_dim)

    # 1. mnist_10class (20,000 samples)
    n_10c = min(len(X_pooled), 20000)
    indices_10c = np.random.permutation(len(X_pooled))[:n_10c]
    p_10c = os.path.join(out_dir, "mnist_10class.csv")
    print(f"Writing {p_10c} ({n_10c} samples, 10 classes)...", end="", flush=True)
    write_csv_stream(p_10c, "mnist_10class", X_pooled[indices_10c], y_raw[indices_10c], n_classes=10)
    print(f" Done. ({os.path.getsize(p_10c)/1024:.1f} KB)")

    # 2. mnist_4class (digits 0, 1, 2, 3 - 10,000 samples)
    mask_4c = (y_raw < 4)
    X_4c = X_pooled[mask_4c]
    y_4c = y_raw[mask_4c]
    n_4c = min(len(X_4c), 10000)
    indices_4c = np.random.permutation(len(X_4c))[:n_4c]
    p_4c = os.path.join(out_dir, "mnist_4class.csv")
    print(f"Writing {p_4c} ({n_4c} samples, 4 classes)...", end="", flush=True)
    write_csv_stream(p_4c, "mnist_4class", X_4c[indices_4c], y_4c[indices_4c], n_classes=4)
    print(f" Done. ({os.path.getsize(p_4c)/1024:.1f} KB)")

    # 3. mnist_binary (even=0 vs odd=1 - 20,000 samples)
    y_bin = y_raw % 2
    n_bin = min(len(X_pooled), 20000)
    indices_bin = np.random.permutation(len(X_pooled))[:n_bin]
    p_bin = os.path.join(out_dir, "mnist_binary.csv")
    print(f"Writing {p_bin} ({n_bin} samples, 2 classes)...", end="", flush=True)
    write_csv_stream(p_bin, "mnist_binary", X_pooled[indices_bin], y_bin[indices_bin], n_classes=2)
    print(f" Done. ({os.path.getsize(p_bin)/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Generate pooled MNIST streams for Triple Benchmark.")
    parser.add_argument("--outdir", default="benchmarks/data", help="Output directory.")
    parser.add_argument("--cachedir", default="benchmarks/data/sklearn_cache", help="Cache directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    generate_all_mnist_streams(
        out_dir=os.path.abspath(args.outdir),
        cache_dir=os.path.abspath(args.cachedir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
