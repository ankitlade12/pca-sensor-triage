"""
Experiment 11: Reconstruction Method Comparison.

Compares forward-fill, linear interpolation, and zero-fill reconstruction
on TEP at multiple bandwidth levels with PCA-Triage importance scoring.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.triage.pca_triage import PCATriage
from src.triage.rate_allocator import RateAllocator
from src.triage.reconstruction import reconstruct, compute_reconstruction_error
from src.utils import load_tep


def run_reconstruction_comparison():
    print("=" * 70, flush=True)
    print("RECONSTRUCTION METHOD COMPARISON", flush=True)
    print("=" * 70, flush=True)

    print("Loading TEP...", flush=True)
    X_train, y_train, X_test, y_test, sensor_cols, _ = load_tep()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}", flush=True)

    seeds = [42, 123, 456]
    budgets = [0.1, 0.3, 0.5, 0.7]
    recon_methods = ["forward_fill", "linear", "zero"]
    window_size = 50
    results = []

    for budget in budgets:
        print(f"\n--- Budget = {budget:.0%} ---", flush=True)

        for recon_method in recon_methods:
            f1s = []
            recon_errors = []

            for seed in seeds:
                # PCA importance scoring
                triage = PCATriage(n_components=10, window_size=window_size,
                                   forgetting_factor=1.0, min_rate=0.05)
                allocator = RateAllocator(budget=budget, min_rate=0.05)
                n_train = X_train.shape[0]
                n_windows = n_train // window_size
                recon_train = np.zeros_like(X_train, dtype=float)

                for w_idx in range(n_windows):
                    start = w_idx * window_size
                    end = start + window_size
                    window = X_train[start:end]
                    importance = triage.compute_importance(window)
                    rates = allocator.allocate(importance)
                    triaged = allocator.apply_rates(window, rates, seed=seed + w_idx)
                    recon_train[start:end] = reconstruct(triaged, method=recon_method)

                # Handle tail
                remaining = n_train % window_size
                if remaining > 0:
                    start = n_windows * window_size
                    rates_last = np.full(X_train.shape[1], budget)
                    triaged = allocator.apply_rates(X_train[start:], rates_last, seed=seed + n_windows)
                    recon_train[start:] = reconstruct(triaged, method=recon_method)

                # Reconstruction error on train
                err = compute_reconstruction_error(X_train, recon_train)
                recon_errors.append(err['nrmse_mean'])

                # Process test
                triage.reset()
                triage2 = PCATriage(n_components=10, window_size=window_size,
                                    forgetting_factor=1.0, min_rate=0.05)
                n_test = X_test.shape[0]
                n_win_te = n_test // window_size
                recon_test = np.zeros_like(X_test, dtype=float)

                for w_idx in range(n_win_te):
                    start = w_idx * window_size
                    end = start + window_size
                    window = X_test[start:end]
                    importance = triage2.compute_importance(window)
                    rates = allocator.allocate(importance)
                    triaged = allocator.apply_rates(window, rates, seed=seed + 10000 + w_idx)
                    recon_test[start:end] = reconstruct(triaged, method=recon_method)

                remaining = n_test % window_size
                if remaining > 0:
                    start = n_win_te * window_size
                    rates_last = np.full(X_test.shape[1], budget)
                    triaged = allocator.apply_rates(X_test[start:], rates_last, seed=seed + 10000 + n_win_te)
                    recon_test[start:] = reconstruct(triaged, method=recon_method)

                # Classify
                clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                clf.fit(recon_train, y_train)
                f1 = f1_score(y_test, clf.predict(recon_test), average='weighted')
                f1s.append(f1)

            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            mean_nrmse = np.mean(recon_errors)

            results.append({
                'budget': budget,
                'reconstruction': recon_method,
                'f1_mean': mean_f1,
                'f1_std': std_f1,
                'nrmse_mean': mean_nrmse,
            })
            print(f"  {recon_method:15s}: F1 = {mean_f1:.4f} ± {std_f1:.4f}, NRMSE = {mean_nrmse:.4f}", flush=True)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'reconstruction_comparison.csv'), index=False)
    print(f"\nSaved to {out_dir}/reconstruction_comparison.csv", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print(f"{'Budget':>7s}  {'forward_fill':>14s}  {'linear':>14s}  {'zero':>14s}", flush=True)
    print("-" * 55, flush=True)
    for budget in budgets:
        row = f"{budget:>7.0%}"
        for rm in recon_methods:
            r = df[(df['budget'] == budget) & (df['reconstruction'] == rm)]
            row += f"  {r['f1_mean'].values[0]:>14.3f}"
        print(row, flush=True)


if __name__ == "__main__":
    run_reconstruction_comparison()
