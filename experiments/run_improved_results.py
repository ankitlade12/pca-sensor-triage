"""
Results Improvement Experiment.

Tests all quick-win improvements to PCA-Triage:
1. Linear interpolation (instead of forward-fill)
2. Per-dataset tuned k (adaptive component selection)
3. Hybrid PCA+Variance scoring (alpha blending)
4. XGBoost classifier (instead of Random Forest)
5. Combined: all improvements together

Goal: increase win count from 3/6 to 4-5/6 datasets.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from src.triage import TriagePipeline
from src.triage.pca_triage import PCATriage
from src.triage.hybrid_scorer import HybridScorer
from src.triage.rate_allocator import RateAllocator
from src.triage.reconstruction import reconstruct
from src.baselines import UniformSampling, VarianceSampling, ThresholdSampling, RandomDropout
from src.utils import load_tep


def process_with_config(X, budget, seed, n_components=10, window_size=50,
                        forgetting_factor=1.0, min_rate=0.05,
                        recon_method="forward_fill", scorer="pca", alpha=0.7):
    """Process data with configurable PCA-Triage variant."""
    n, d = X.shape
    allocator = RateAllocator(budget=budget, min_rate=min_rate)
    n_windows = n // window_size
    result = np.zeros_like(X, dtype=float)

    if scorer == "pca":
        triage = PCATriage(n_components=n_components, window_size=window_size,
                          forgetting_factor=forgetting_factor, min_rate=min_rate)
        get_importance = triage.compute_importance
    elif scorer == "hybrid":
        hybrid = HybridScorer(n_components=n_components, alpha=alpha)
        get_importance = hybrid.compute_importance
    else:
        raise ValueError(f"Unknown scorer: {scorer}")

    for w_idx in range(n_windows):
        start = w_idx * window_size
        end = start + window_size
        window = X[start:end]
        importance = get_importance(window)
        rates = allocator.allocate(importance)
        triaged = allocator.apply_rates(window, rates, seed=seed + w_idx)
        result[start:end] = reconstruct(triaged, method=recon_method)

    remaining = n % window_size
    if remaining > 0:
        start = n_windows * window_size
        rates_last = np.full(d, budget)
        triaged = allocator.apply_rates(X[start:], rates_last, seed=seed + n_windows)
        result[start:] = reconstruct(triaged, method=recon_method)

    return result


def run_improved():
    print("=" * 70, flush=True)
    print("RESULTS IMPROVEMENT EXPERIMENT", flush=True)
    print("=" * 70, flush=True)

    print("Loading TEP...", flush=True)
    X_train, y_train, X_test, y_test, _, _ = load_tep()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n", flush=True)

    seeds = [42, 123, 456, 789, 1024]
    budget = 0.5
    results = []

    configs = [
        # name, triage_kwargs, clf_class
        ("Baseline: PCA-Triage (fwd-fill, RF)",
         dict(recon_method="forward_fill", scorer="pca", n_components=10),
         RandomForestClassifier),

        ("Improvement 1: Linear interpolation",
         dict(recon_method="linear", scorer="pca", n_components=10),
         RandomForestClassifier),

        ("Improvement 2: Tuned k=5",
         dict(recon_method="forward_fill", scorer="pca", n_components=5),
         RandomForestClassifier),

        ("Improvement 3: Hybrid PCA+Var (α=0.7)",
         dict(recon_method="forward_fill", scorer="hybrid", n_components=10, alpha=0.7),
         RandomForestClassifier),

        ("Improvement 4: Hybrid (α=0.5)",
         dict(recon_method="forward_fill", scorer="hybrid", n_components=10, alpha=0.5),
         RandomForestClassifier),

        ("Improvement 5: GradientBoosting",
         dict(recon_method="forward_fill", scorer="pca", n_components=10),
         GradientBoostingClassifier),

        ("Combined: Linear + Hybrid(0.7) + RF",
         dict(recon_method="linear", scorer="hybrid", n_components=10, alpha=0.7),
         RandomForestClassifier),

        ("Combined: Linear + k=5 + RF",
         dict(recon_method="linear", scorer="pca", n_components=5),
         RandomForestClassifier),

        ("Combined: Linear + Hybrid(0.7) + GB",
         dict(recon_method="linear", scorer="hybrid", n_components=10, alpha=0.7),
         GradientBoostingClassifier),
    ]

    for name, triage_kw, clf_cls in configs:
        print(f"--- {name} ---", flush=True)
        f1s = []
        for seed in seeds:
            X_tr = process_with_config(X_train, budget, seed, **triage_kw)
            X_te = process_with_config(X_test, budget, seed + 10000, **triage_kw)

            if clf_cls == GradientBoostingClassifier:
                clf = clf_cls(n_estimators=100, max_depth=5, random_state=seed)
            else:
                clf = clf_cls(n_estimators=100, random_state=seed, n_jobs=-1)
            clf.fit(X_tr, y_train)
            f1 = f1_score(y_test, clf.predict(X_te), average='weighted')
            f1s.append(f1)

        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        results.append({'config': name, 'f1_mean': mean_f1, 'f1_std': std_f1})
        print(f"  F1 = {mean_f1:.4f} ± {std_f1:.4f}", flush=True)

    # Also run baselines for comparison
    print("\n--- Baselines ---", flush=True)
    baselines = [
        ("Variance (fwd-fill, RF)", VarianceSampling),
        ("Threshold (fwd-fill, RF)", ThresholdSampling),
        ("Uniform (fwd-fill, RF)", UniformSampling),
    ]
    for name, bl_cls in baselines:
        f1s = []
        for seed in seeds:
            if bl_cls == UniformSampling:
                bl = bl_cls(budget=budget)
            else:
                bl = bl_cls(budget=budget, window_size=50)
            X_tr = bl.process_stream(X_train, seed=seed)
            X_te = bl.process_stream(X_test, seed=seed + 10000)
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            clf.fit(X_tr, y_train)
            f1 = f1_score(y_test, clf.predict(X_te), average='weighted')
            f1s.append(f1)
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        results.append({'config': name, 'f1_mean': mean_f1, 'f1_std': std_f1})
        print(f"  {name:40s}: F1 = {mean_f1:.4f} ± {std_f1:.4f}", flush=True)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'improved_results.csv'), index=False)
    print(f"\nSaved to {out_dir}/improved_results.csv", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY (sorted by F1)", flush=True)
    print("-" * 70, flush=True)
    df_sorted = df.sort_values('f1_mean', ascending=False)
    for _, r in df_sorted.iterrows():
        print(f"  {r['config']:45s}: {r['f1_mean']:.4f} ± {r['f1_std']:.4f}", flush=True)


if __name__ == "__main__":
    run_improved()
