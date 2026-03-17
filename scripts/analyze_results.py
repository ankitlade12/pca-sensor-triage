#!/usr/bin/env python3
"""
Analyze experiment results and generate summary tables.

Usage:
    python scripts/analyze_results.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results')

datasets = ['tep', 'skab', 'smd', 'msl', 'psm', 'hai']
methods = ['PCA-Triage', 'Uniform', 'Threshold', 'Variance', 'Random Dropout', 'Mutual Info']


def load_pareto(ds):
    path = os.path.join(RESULTS_DIR, f'pareto_{ds}.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def summary_at_budget(budget=0.5):
    """Print summary table at given budget."""
    print(f"\n{'='*80}")
    print(f"SUMMARY AT {int(budget*100)}% BANDWIDTH")
    print(f"{'='*80}")
    print(f"{'Method':20s}", end="")
    for ds in datasets:
        print(f" | {ds.upper():>8s}", end="")
    print(f" | {'Wins':>5s}")
    print("-" * 80)

    for method in methods:
        print(f"{method:20s}", end="")
        wins = 0
        for ds in datasets:
            df = load_pareto(ds)
            if df is None:
                print(f" | {'N/A':>8s}", end="")
                continue
            sub = df[(df['method'] == method) & (df['budget'] == budget)]
            if len(sub) == 0:
                print(f" | {'N/A':>8s}", end="")
                continue
            f1 = sub['f1'].mean()
            # Check if winner
            all_at_b = df[df['budget'] == budget].groupby('method')['f1'].mean()
            if method == all_at_b.idxmax():
                wins += 1
                print(f" | {f1:>7.4f}*", end="")
            else:
                print(f" | {f1:>8.4f}", end="")
        print(f" | {wins:>5d}")


def wilcoxon_analysis(budget=0.5):
    """Run Wilcoxon tests: PCA-Triage vs each baseline."""
    print(f"\n{'='*80}")
    print(f"WILCOXON SIGNED-RANK TESTS AT {int(budget*100)}% BANDWIDTH")
    print(f"{'='*80}")

    for ds in datasets:
        df = load_pareto(ds)
        if df is None:
            continue
        at_b = df[df['budget'] == budget]
        pca = at_b[at_b['method'] == 'PCA-Triage']['f1'].values

        print(f"\n{ds.upper()}:")
        for method in methods:
            if method == 'PCA-Triage':
                continue
            other = at_b[at_b['method'] == method]['f1'].values
            n = min(len(pca), len(other))
            if n < 3:
                print(f"  vs {method:20s}: insufficient data (n={n})")
                continue
            try:
                _, p = wilcoxon(pca[:n], other[:n], alternative='greater')
                sig = '*' if p < 0.05 else ''
                print(f"  vs {method:20s}: p={p:.4f} {sig}")
            except Exception as e:
                print(f"  vs {method:20s}: {e}")


if __name__ == '__main__':
    summary_at_budget(0.5)
    summary_at_budget(0.3)
    wilcoxon_analysis(0.5)
