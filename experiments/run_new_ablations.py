"""
Run adaptive k ablation, ensemble scorer ablation, and updated statistical tests.

1. Adaptive k: Compare fixed k vs adaptive k (variance threshold) on TEP
2. Ensemble: Compare single k vs multi-k ensemble on TEP
3. Wilcoxon/Friedman: Re-run with 7 datasets
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from src.triage.pipeline import TriagePipeline
from src.baselines import (
    UniformSampling, ThresholdSampling, VarianceSampling,
    RandomDropout, MutualInfoSampling,
)
from src.utils.data_loader import get_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate(X_train, y_tr, X_test, y_te, pipe_kwargs, seeds=[42, 123, 456]):
    f1s = []
    for seed in seeds:
        pipe = TriagePipeline(**pipe_kwargs, budget=0.5)
        X_tr = pipe.process_stream(X_train, seed=seed)
        X_te = pipe.process_stream(X_test, seed=seed + 10000)
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        f1s.append(f1_score(y_te, clf.predict(X_te), average='weighted'))
    return np.mean(f1s), np.std(f1s), f1s


# ============================================================
# 1. ADAPTIVE K ABLATION (TEP)
# ============================================================
def run_adaptive_k_ablation():
    print("=" * 60, flush=True)
    print("ABLATION 1: Adaptive k (TEP @ 50% BW)", flush=True)
    print("=" * 60, flush=True)

    result = get_dataset('tep')
    X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
    le = LabelEncoder()
    y_tr, y_te = le.fit_transform(y_train), le.transform(y_test)

    configs = [
        ("Fixed k=3", dict(n_components=3, window_size=50, forgetting_factor=1.0,
                           min_rate=0.05, scorer='pca', reconstruction_method='linear')),
        ("Fixed k=5", dict(n_components=5, window_size=50, forgetting_factor=1.0,
                           min_rate=0.05, scorer='pca', reconstruction_method='linear')),
        ("Fixed k=10", dict(n_components=10, window_size=50, forgetting_factor=1.0,
                            min_rate=0.05, scorer='pca', reconstruction_method='linear')),
        ("Fixed k=15", dict(n_components=15, window_size=50, forgetting_factor=1.0,
                            min_rate=0.05, scorer='pca', reconstruction_method='linear')),
        ("Fixed k=20", dict(n_components=20, window_size=50, forgetting_factor=1.0,
                            min_rate=0.05, scorer='pca', reconstruction_method='linear')),
        ("Adaptive k (95%)", dict(n_components=20, window_size=50, forgetting_factor=1.0,
                                  min_rate=0.05, scorer='adaptive_k',
                                  variance_threshold=0.95, reconstruction_method='linear')),
        ("Adaptive k (90%)", dict(n_components=20, window_size=50, forgetting_factor=1.0,
                                  min_rate=0.05, scorer='adaptive_k',
                                  variance_threshold=0.90, reconstruction_method='linear')),
        ("Adaptive k (99%)", dict(n_components=20, window_size=50, forgetting_factor=1.0,
                                  min_rate=0.05, scorer='adaptive_k',
                                  variance_threshold=0.99, reconstruction_method='linear')),
    ]

    rows = []
    for name, cfg in configs:
        mean_f1, std_f1, _ = evaluate(X_train, y_tr, X_test, y_te, cfg)
        rows.append({'config': name, 'f1_mean': mean_f1, 'f1_std': std_f1})
        print(f"  {name:25s}: F1={mean_f1:.4f} ± {std_f1:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'adaptive_k_ablation.csv'), index=False)
    print(f"\nSaved to {RESULTS_DIR}/adaptive_k_ablation.csv\n", flush=True)
    return df


# ============================================================
# 2. ENSEMBLE SCORER ABLATION (TEP)
# ============================================================
def run_ensemble_ablation():
    print("=" * 60, flush=True)
    print("ABLATION 2: Ensemble Scorer (TEP @ 50% BW)", flush=True)
    print("=" * 60, flush=True)

    result = get_dataset('tep')
    X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
    le = LabelEncoder()
    y_tr, y_te = le.fit_transform(y_train), le.transform(y_test)

    configs = [
        ("Single k=5", dict(n_components=5, window_size=50, forgetting_factor=1.0,
                            min_rate=0.05, scorer='pca', reconstruction_method='linear')),
        ("Single k=10", dict(n_components=10, window_size=50, forgetting_factor=1.0,
                             min_rate=0.05, scorer='pca', reconstruction_method='linear')),
        ("Ensemble [3,5,10]", dict(n_components=10, window_size=50, forgetting_factor=1.0,
                                   min_rate=0.05, scorer='ensemble',
                                   k_values=[3, 5, 10], reconstruction_method='linear')),
        ("Ensemble [5,10,15]", dict(n_components=15, window_size=50, forgetting_factor=1.0,
                                    min_rate=0.05, scorer='ensemble',
                                    k_values=[5, 10, 15], reconstruction_method='linear')),
        ("Ensemble [3,5,10,15]", dict(n_components=15, window_size=50, forgetting_factor=1.0,
                                      min_rate=0.05, scorer='ensemble',
                                      k_values=[3, 5, 10, 15], reconstruction_method='linear')),
        ("Hybrid α=0.8 (v2 default)", dict(n_components=10, window_size=50, forgetting_factor=1.0,
                                            min_rate=0.05, scorer='hybrid', alpha=0.8,
                                            sharpness=1.5, reconstruction_method='linear')),
    ]

    rows = []
    for name, cfg in configs:
        mean_f1, std_f1, _ = evaluate(X_train, y_tr, X_test, y_te, cfg)
        rows.append({'config': name, 'f1_mean': mean_f1, 'f1_std': std_f1})
        print(f"  {name:30s}: F1={mean_f1:.4f} ± {std_f1:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'ensemble_ablation.csv'), index=False)
    print(f"\nSaved to {RESULTS_DIR}/ensemble_ablation.csv\n", flush=True)
    return df


# ============================================================
# 3. WILCOXON / FRIEDMAN TESTS (7 datasets)
# ============================================================
def run_statistical_tests():
    print("=" * 60, flush=True)
    print("STATISTICAL TESTS: Wilcoxon + Friedman (7 datasets)", flush=True)
    print("=" * 60, flush=True)

    datasets = ['tep', 'smd', 'msl', 'psm', 'hai', 'skab', 'swat']
    methods = ['PCA-Triage', 'Uniform', 'Threshold', 'Variance', 'Random Dropout']

    # Collect F1 at budget=0.5 per dataset per method
    results = {}
    for ds in datasets:
        path = os.path.join(RESULTS_DIR, f'pareto_{ds}.csv')
        if not os.path.exists(path):
            print(f"  SKIP {ds} — no pareto CSV", flush=True)
            continue
        df = pd.read_csv(path)
        df50 = df[df['budget'] == 0.5]
        for method in methods:
            sub = df50[df50['method'] == method]
            if len(sub) > 0:
                key = (ds, method)
                results[key] = sub['f1'].mean()

    # Build matrix: rows=datasets, cols=methods
    matrix = []
    ds_used = []
    for ds in datasets:
        row = []
        skip = False
        for method in methods:
            if (ds, method) in results:
                row.append(results[(ds, method)])
            else:
                skip = True
                break
        if not skip:
            matrix.append(row)
            ds_used.append(ds)

    matrix = np.array(matrix)
    print(f"\n  Datasets used: {ds_used}", flush=True)
    print(f"  Matrix shape: {matrix.shape}", flush=True)

    # Friedman test
    friedman_stat, friedman_p = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
    ranks = np.zeros_like(matrix)
    for i in range(len(ds_used)):
        ranks[i] = len(methods) + 1 - stats.rankdata(matrix[i])  # higher F1 = lower rank
    mean_ranks = ranks.mean(axis=0)

    print(f"\n  Friedman chi2={friedman_stat:.3f}, p={friedman_p:.4f}", flush=True)
    print(f"  Mean ranks (lower=better):", flush=True)
    for method, rank in zip(methods, mean_ranks):
        marker = " *** BEST" if rank == mean_ranks.min() else ""
        print(f"    {method:20s}: {rank:.2f}{marker}", flush=True)

    # Wilcoxon signed-rank tests: PCA-Triage vs each baseline
    pca_col = 0  # PCA-Triage is first
    wilcoxon_rows = []
    print(f"\n  Wilcoxon signed-rank (PCA-Triage vs baseline, one-sided):", flush=True)
    for j, method in enumerate(methods[1:], 1):
        diff = matrix[:, pca_col] - matrix[:, j]
        wins = (diff > 0).sum()
        losses = (diff < 0).sum()
        ties = (diff == 0).sum()
        try:
            stat, p_two = stats.wilcoxon(matrix[:, pca_col], matrix[:, j],
                                          alternative='greater')
            p_val = p_two
        except ValueError:
            p_val = 1.0
            stat = 0

        sig = "*" if p_val < 0.05 else ""
        wilcoxon_rows.append({
            'baseline': method, 'wins': wins, 'losses': losses, 'ties': ties,
            'p_value': p_val, 'significant': p_val < 0.05
        })
        print(f"    vs {method:20s}: W={wins} L={losses} T={ties} p={p_val:.4f} {sig}", flush=True)

    # Save
    friedman_df = pd.DataFrame({
        'method': methods,
        'mean_rank': mean_ranks,
    })
    friedman_df.to_csv(os.path.join(RESULTS_DIR, 'friedman_ranks_v2.csv'), index=False)

    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    wilcoxon_df.to_csv(os.path.join(RESULTS_DIR, 'wilcoxon_tests_v2.csv'), index=False)

    print(f"\n  Saved to {RESULTS_DIR}/friedman_ranks_v2.csv and wilcoxon_tests_v2.csv\n", flush=True)


if __name__ == "__main__":
    run_adaptive_k_ablation()
    run_ensemble_ablation()
    run_statistical_tests()
    print("\nAll done!", flush=True)
