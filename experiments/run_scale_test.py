"""
Experiment 16: Scalability to 500+ Channels.

Generates synthetic data with increasing channel counts and measures:
1. Compute time per window (ms)
2. F1 at 50% bandwidth
3. Memory usage

Validates the O(wdk) scaling claim from the paper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time
import tracemalloc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.triage.pipeline import TriagePipeline
from src.baselines import VarianceSampling

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def generate_synthetic(n_channels, n_samples=5000, anomaly_ratio=0.1, seed=42):
    """Generate synthetic multi-channel data with anomalies."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_channels)

    # Add correlation structure (groups of 10)
    n_groups = max(1, n_channels // 10)
    for g in range(n_groups):
        start = g * 10
        end = min(start + 10, n_channels)
        latent = rng.randn(n_samples)
        for ch in range(start, end):
            X[:, ch] = 0.6 * latent + 0.4 * X[:, ch]

    # Add anomalies
    y = np.zeros(n_samples, dtype=int)
    n_anomaly = int(n_samples * anomaly_ratio)
    anom_start = n_samples // 2
    y[anom_start:anom_start + n_anomaly] = 1
    affected = rng.choice(n_channels, size=min(5, n_channels), replace=False)
    for ch in affected:
        X[anom_start:anom_start + n_anomaly, ch] += 3.0

    return X, y


def run_experiment():
    print("=" * 70, flush=True)
    print("EXPERIMENT 16: Scalability to 500+ Channels", flush=True)
    print("=" * 70, flush=True)

    channel_counts = [8, 25, 50, 100, 200, 300, 500, 750, 1000]
    n_samples = 5000
    budget = 0.5
    rows = []

    for d in channel_counts:
        print(f"\n  d={d} channels...", end=" ", flush=True)
        X, y = generate_synthetic(d, n_samples=n_samples)

        split = int(0.7 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        k = min(10, d - 1)

        # PCA-Triage timing
        tracemalloc.start()
        t0 = time.time()
        pipe = TriagePipeline(
            n_components=k, window_size=50, budget=budget,
            forgetting_factor=1.0, min_rate=0.05,
            reconstruction_method='linear', scorer='hybrid', alpha=0.7)
        X_tr = pipe.process_stream(X_train, seed=42)
        X_te = pipe.process_stream(X_test, seed=43)
        pca_time = time.time() - t0
        pca_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # Peak MB
        tracemalloc.stop()

        n_windows = n_samples // 50
        ms_per_window = (pca_time / n_windows) * 1000

        # F1
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_train)
        f1_pca = f1_score(y_test, clf.predict(X_te), average='weighted')

        # Variance baseline timing
        t0 = time.time()
        var = VarianceSampling(budget=budget, window_size=50)
        X_tr_v = var.process_stream(X_train, seed=42)
        X_te_v = var.process_stream(X_test, seed=43)
        var_time = time.time() - t0
        var_ms = (var_time / n_windows) * 1000

        clf2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf2.fit(X_tr_v, y_train)
        f1_var = f1_score(y_test, clf2.predict(X_te_v), average='weighted')

        rows.append({
            'channels': d, 'pca_ms_per_window': ms_per_window,
            'pca_total_s': pca_time, 'pca_peak_mb': pca_mem,
            'pca_f1': f1_pca, 'var_ms_per_window': var_ms,
            'var_f1': f1_var,
            'under_5ms': ms_per_window < 5.0,
        })
        edge = "✓" if ms_per_window < 5.0 else "✗"
        print(f"PCA: {ms_per_window:.2f} ms/win, F1={f1_pca:.3f} | "
              f"Var: {var_ms:.2f} ms/win, F1={f1_var:.3f} | "
              f"Edge target (<5ms): {edge}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'scale_test.csv'), index=False)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SCALABILITY SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    max_edge = df[df['under_5ms']]['channels'].max() if df['under_5ms'].any() else 0
    print(f"  Max channels under 5ms edge target: {max_edge}", flush=True)
    print(f"  At d=500: {df[df['channels']==500]['pca_ms_per_window'].values[0]:.2f} ms/window", flush=True)
    print(f"  At d=1000: {df[df['channels']==1000]['pca_ms_per_window'].values[0]:.2f} ms/window", flush=True)
    print(f"\nSaved to {RESULTS_DIR}/scale_test.csv", flush=True)


if __name__ == "__main__":
    run_experiment()
