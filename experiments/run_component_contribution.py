"""
Experiment 7: Component Contribution Analysis.

Measures each PCA-Triage component's individual contribution by disabling
components one at a time (via parameter extremes) and comparing to the
full method. Inspired by RG-TTA's component analysis approach.

Components tested:
1. Full PCA-Triage (all components enabled)
2. No PCA (Variance baseline — removes correlation exploitation)
3. No smoothing (lambda=0 — purely instantaneous scores)
4. No min-rate floor (min_rate=0 — channels can be fully silenced)
5. No proportional allocation (Threshold baseline — binary include/exclude)
6. No weighting (unweighted loadings V[i,j]^2 instead of sigma_i * V[i,j]^2)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.triage import TriagePipeline
from src.baselines import VarianceSampling, ThresholdSampling, UniformSampling
from src.utils import load_tep


def run_component_contribution():
    print("=" * 60)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 60)

    print("\nLoading TEP dataset...")
    X_train, y_train, X_test, y_test, sensor_cols, _ = load_tep()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    seeds = [42, 123, 456, 789, 1024]
    budget = 0.5
    results = []

    def evaluate(method_name, get_data_fn):
        """Evaluate a configuration across all seeds."""
        f1s = []
        for seed in seeds:
            X_tr, X_te = get_data_fn(seed)
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            clf.fit(X_tr, y_train)
            f1 = f1_score(y_test, clf.predict(X_te), average='weighted')
            f1s.append(f1)
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        results.append({
            'component': method_name,
            'f1_mean': mean_f1,
            'f1_std': std_f1,
            'n_seeds': len(seeds),
        })
        print(f"  {method_name:40s}: F1 = {mean_f1:.4f} +/- {std_f1:.4f}")
        return mean_f1

    # 1. Full PCA-Triage (all components)
    print("\n1. Full PCA-Triage (all components)")
    def full_pca(seed):
        pipe = TriagePipeline(n_components=10, window_size=50, budget=budget,
                              forgetting_factor=1.0, min_rate=0.05)
        X_tr = pipe.process_stream(X_train, seed=seed)
        pipe.reset()
        X_te = pipe.process_stream(X_test, seed=seed + 10000)
        return X_tr, X_te
    full_f1 = evaluate("Full PCA-Triage", full_pca)

    # 2. No PCA → Variance baseline (removes correlation exploitation)
    print("\n2. No PCA (Variance baseline)")
    def no_pca(seed):
        var = VarianceSampling(budget=budget, window_size=50)
        X_tr = var.process_stream(X_train, seed=seed)
        X_te = var.process_stream(X_test, seed=seed + 10000)
        return X_tr, X_te
    evaluate("No PCA (Variance)", no_pca)

    # 3. No smoothing (lambda → 0, instantaneous scores)
    print("\n3. No smoothing (lambda=0.001)")
    def no_smoothing(seed):
        pipe = TriagePipeline(n_components=10, window_size=50, budget=budget,
                              forgetting_factor=0.001, min_rate=0.05)
        X_tr = pipe.process_stream(X_train, seed=seed)
        pipe.reset()
        X_te = pipe.process_stream(X_test, seed=seed + 10000)
        return X_tr, X_te
    evaluate("No smoothing (lambda=0.001)", no_smoothing)

    # 4. No min-rate floor (channels can be fully silenced)
    print("\n4. No min-rate floor (min_rate=0.001)")
    def no_floor(seed):
        pipe = TriagePipeline(n_components=10, window_size=50, budget=budget,
                              forgetting_factor=1.0, min_rate=0.001)
        X_tr = pipe.process_stream(X_train, seed=seed)
        pipe.reset()
        X_te = pipe.process_stream(X_test, seed=seed + 10000)
        return X_tr, X_te
    evaluate("No min-rate floor (r_min=0.001)", no_floor)

    # 5. No proportional allocation → Threshold (binary include/exclude)
    print("\n5. No proportional allocation (Threshold baseline)")
    def no_proportional(seed):
        thr = ThresholdSampling(budget=budget, window_size=50)
        X_tr = thr.process_stream(X_train, seed=seed)
        X_te = thr.process_stream(X_test, seed=seed + 10000)
        return X_tr, X_te
    evaluate("No proportional (Threshold)", no_proportional)

    # 6. No data-driven allocation → Uniform baseline
    print("\n6. No data-driven allocation (Uniform baseline)")
    def no_datadriven(seed):
        uni = UniformSampling(budget=budget)
        X_tr = uni.process_stream(X_train, seed=seed)
        X_te = uni.process_stream(X_test, seed=seed + 10000)
        return X_tr, X_te
    evaluate("No data-driven (Uniform)", no_datadriven)

    # 7. Aggressive smoothing (lambda=0.5 — fast forgetting)
    print("\n7. Aggressive smoothing (lambda=0.5)")
    def fast_forget(seed):
        pipe = TriagePipeline(n_components=10, window_size=50, budget=budget,
                              forgetting_factor=0.5, min_rate=0.05)
        X_tr = pipe.process_stream(X_train, seed=seed)
        pipe.reset()
        X_te = pipe.process_stream(X_test, seed=seed + 10000)
        return X_tr, X_te
    evaluate("Aggressive smoothing (lambda=0.5)", fast_forget)

    # 8. Few components (k=2 — minimal PCA)
    print("\n8. Minimal PCA (k=2)")
    def minimal_pca(seed):
        pipe = TriagePipeline(n_components=2, window_size=50, budget=budget,
                              forgetting_factor=1.0, min_rate=0.05)
        X_tr = pipe.process_stream(X_train, seed=seed)
        pipe.reset()
        X_te = pipe.process_stream(X_test, seed=seed + 10000)
        return X_tr, X_te
    evaluate("Minimal PCA (k=2)", minimal_pca)

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'component_contribution.csv'), index=False)
    print(f"\nSaved to {out_dir}/component_contribution.csv")

    # Print contribution summary
    print("\n" + "=" * 60)
    print("COMPONENT CONTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"{'Component removed':<42s} {'F1':>8s} {'Delta':>8s} {'%':>8s}")
    print("-" * 68)
    for r in results:
        delta = r['f1_mean'] - full_f1
        pct = (delta / full_f1) * 100
        sign = "+" if delta >= 0 else ""
        print(f"{r['component']:<42s} {r['f1_mean']:>8.4f} {sign}{delta:>7.4f} {sign}{pct:>7.2f}%")


if __name__ == "__main__":
    run_component_contribution()
