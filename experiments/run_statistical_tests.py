"""
LEGACY statistical analysis script.

This file preserves an older exploratory workflow based on
`experiments/results/pareto_{dataset}.csv` and includes Nemenyi post-hoc tests.
It is NOT the canonical paper statistics path.

For the paper's current results, use:
    experiments/run_statistical_tests_v1.py

Canonical source for the paper:
    paper/tables/table2_results_50pct.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import warnings
warnings.filterwarnings('ignore')


def load_all_50pct():
    """Load all dataset results at 50% bandwidth, averaged per seed."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    datasets = ['tep', 'smd', 'msl', 'psm', 'hai', 'skab']

    all_data = []
    for ds in datasets:
        f = os.path.join(results_dir, f'pareto_{ds}.csv')
        df = pd.read_csv(f)
        b50 = df[df['budget'] == 0.5]
        for _, row in b50.iterrows():
            all_data.append({
                'dataset': ds.upper(),
                'method': row['method'],
                'seed': row['seed'],
                'f1': row['f1'],
            })
    return pd.DataFrame(all_data)


def friedman_test(df):
    """Run Friedman test across all datasets and methods."""
    print("=" * 70)
    print("FRIEDMAN TEST: Overall Ranking Across Datasets")
    print("=" * 70)

    # Average over seeds per dataset
    avg = df.groupby(['dataset', 'method'])['f1'].mean().reset_index()

    # Pivot: rows=datasets, columns=methods
    pivot = avg.pivot(index='dataset', columns='method', values='f1')
    methods = list(pivot.columns)

    print(f"\nDatasets: {list(pivot.index)}")
    print(f"Methods: {methods}")

    # Friedman test
    arrays = [pivot[m].values for m in methods]
    stat, p = stats.friedmanchisquare(*arrays)

    print(f"\nFriedman chi-squared = {stat:.2f}")
    print(f"p-value = {p:.2e}")
    print(f"Significant (p < 0.05): {'YES' if p < 0.05 else 'NO'}")

    # Compute mean ranks
    ranks_df = pivot.rank(axis=1, ascending=False)
    mean_ranks = ranks_df.mean().sort_values()

    print(f"\nMean Ranks (lower = better):")
    print("-" * 40)
    for method, rank in mean_ranks.items():
        print(f"  {method:20s}: {rank:.2f}")

    return pivot, mean_ranks, stat, p


def nemenyi_posthoc(pivot):
    """Run Nemenyi post-hoc test."""
    print("\n" + "=" * 70)
    print("NEMENYI POST-HOC TEST")
    print("=" * 70)

    # scikit-posthocs Nemenyi test
    result = sp.posthoc_nemenyi_friedman(pivot.values)
    result.index = pivot.columns
    result.columns = pivot.columns

    print("\nP-value matrix (significant pairs have p < 0.05):")
    print(result.round(4).to_string())

    # Extract significant pairs
    print("\nSignificant pairs (p < 0.05):")
    methods = list(pivot.columns)
    sig_pairs = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            p = result.iloc[i, j]
            if p < 0.05:
                sig_pairs.append((methods[i], methods[j], p))
                print(f"  {methods[i]} vs {methods[j]}: p = {p:.4f}")

    if not sig_pairs:
        print("  (none)")

    return result


def wilcoxon_pairwise(df):
    """Run Wilcoxon signed-rank tests: PCA-Triage vs each baseline, per dataset."""
    print("\n" + "=" * 70)
    print("WILCOXON SIGNED-RANK TESTS: PCA-Triage vs Each Baseline")
    print("=" * 70)

    datasets = df['dataset'].unique()
    methods = [m for m in df['method'].unique() if m != 'PCA-Triage']

    results = []

    for ds in sorted(datasets):
        print(f"\n--- {ds} ---")
        ds_data = df[df['dataset'] == ds]
        pca_data = ds_data[ds_data['method'] == 'PCA-Triage'].sort_values('seed')['f1'].values

        for method in methods:
            baseline_data = ds_data[ds_data['method'] == method].sort_values('seed')['f1'].values

            if len(pca_data) != len(baseline_data) or len(pca_data) < 3:
                print(f"  vs {method:20s}: SKIPPED (insufficient paired data)")
                continue

            diff = pca_data - baseline_data
            mean_diff = np.mean(diff)

            try:
                stat, p = stats.wilcoxon(pca_data, baseline_data, alternative='two-sided')
                sig = "*" if p < 0.05 else ""
                print(f"  vs {method:20s}: delta={mean_diff:+.4f}  W={stat:.1f}  p={p:.4f} {sig}")
            except Exception as e:
                # If all differences are 0 or too few samples
                p = 1.0
                stat = 0
                sig = ""
                print(f"  vs {method:20s}: delta={mean_diff:+.4f}  (test failed: {e})")

            results.append({
                'dataset': ds,
                'baseline': method,
                'pca_mean': np.mean(pca_data),
                'baseline_mean': np.mean(baseline_data),
                'delta': mean_diff,
                'W_stat': stat,
                'p_value': p,
                'significant': p < 0.05,
            })

    results_df = pd.DataFrame(results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: PCA-Triage Wins (p < 0.05)")
    print("=" * 70)

    sig_wins = results_df[(results_df['significant']) & (results_df['delta'] > 0)]
    sig_losses = results_df[(results_df['significant']) & (results_df['delta'] < 0)]

    print(f"Significant wins:   {len(sig_wins)}/{len(results_df)}")
    print(f"Significant losses: {len(sig_losses)}/{len(results_df)}")

    for _, row in sig_wins.iterrows():
        print(f"  WIN:  {row['dataset']:5s} vs {row['baseline']:20s} (delta={row['delta']:+.4f}, p={row['p_value']:.4f})")
    for _, row in sig_losses.iterrows():
        print(f"  LOSS: {row['dataset']:5s} vs {row['baseline']:20s} (delta={row['delta']:+.4f}, p={row['p_value']:.4f})")

    return results_df


def critical_difference_values(n_datasets, n_methods, alpha=0.05):
    """Compute Nemenyi critical difference."""
    from scipy.stats import studentized_range
    q_alpha = studentized_range.ppf(1 - alpha, n_methods, np.inf)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    return cd


if __name__ == "__main__":
    df = load_all_50pct()
    print(f"Loaded {len(df)} data points across {df['dataset'].nunique()} datasets, "
          f"{df['method'].nunique()} methods, {df['seed'].nunique()} seeds\n")

    # 1. Friedman test
    pivot, mean_ranks, chi2, friedman_p = friedman_test(df)

    # 2. Nemenyi post-hoc
    nemenyi_result = nemenyi_posthoc(pivot)

    # 3. Wilcoxon per dataset
    wilcoxon_df = wilcoxon_pairwise(df)

    # 4. Critical difference
    n_ds = pivot.shape[0]
    n_m = pivot.shape[1]
    try:
        cd = critical_difference_values(n_ds, n_m)
        print(f"\nNemenyi Critical Difference (alpha=0.05): CD = {cd:.2f}")
        print("Methods whose mean rank difference < CD are statistically indistinguishable.")
    except Exception:
        print("\n(Could not compute CD — studentized_range may not be available)")

    # Save all results
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    wilcoxon_df.to_csv(os.path.join(out_dir, 'wilcoxon_tests.csv'), index=False)

    ranks_df = pd.DataFrame({'method': mean_ranks.index, 'mean_rank': mean_ranks.values})
    ranks_df.to_csv(os.path.join(out_dir, 'friedman_ranks.csv'), index=False)

    print(f"\nSaved to {out_dir}/wilcoxon_tests.csv and friedman_ranks.csv")
