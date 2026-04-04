"""
Statistical tests on V1 base results (6 datasets, 3 seeds, no per-dataset tuning).

Computes:
1. Friedman test with Kendall's W effect size
2. Wilcoxon signed-rank (PCA-Triage vs each baseline) with Holm correction
3. Rank-biserial effect sizes for Wilcoxon

Reads from: experiments/results/pareto_{tep,smd,msl,psm,hai,skab}.csv
Writes to: experiments/results/friedman_ranks_v1.csv, wilcoxon_tests_v1.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
PAPER_TABLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'tables')
DATASETS = ['tep', 'smd', 'msl', 'psm', 'hai', 'skab']
UNSUPERVISED = ['PCA-Triage', 'Uniform', 'Threshold', 'Variance', 'Random Dropout']

# Canonical source: paper/tables/table2_results_50pct.csv (V1 base, 5 seeds, RF-100)
# NOT experiments/results/pareto_{ds}.csv (which contains V2 tuned results)
CANONICAL_SOURCE = 'paper/tables/table2_results_50pct.csv'


def load_v1_50pct():
    """Load canonical V1 base results from table2_results_50pct.csv.

    This CSV contains seed-averaged F1 values (not per-seed). For Wilcoxon
    tests we use dataset-level pairing (6 paired observations).
    """
    path = os.path.join(PAPER_TABLES_DIR, 'table2_results_50pct.csv')
    df = pd.read_csv(path)

    col_map = {
        'PCA-Triage': 'PCA_f1_mean',
        'Uniform': 'Uniform_f1',
        'Threshold': 'Threshold_f1',
        'Variance': 'Variance_f1',
        'Random Dropout': 'Random Dropout_f1',
        'Mutual Info': 'Mutual Info_f1',
    }

    rows = []
    for _, row in df.iterrows():
        ds = row['dataset'].upper()
        for method, col in col_map.items():
            rows.append({
                'dataset': ds,
                'method': method,
                'f1': row[col],
            })
    return pd.DataFrame(rows)


def friedman_analysis(df):
    print("=" * 70)
    print(f"FRIEDMAN TEST (5 unsupervised methods x 6 datasets)")
    print(f"Source: {CANONICAL_SOURCE}")
    print("=" * 70)

    unsup = df[df['method'].isin(UNSUPERVISED)]
    pivot = unsup.pivot(index='dataset', columns='method', values='f1')

    methods = list(pivot.columns)
    arrays = [pivot[m].values for m in methods]
    chi2, p = stats.friedmanchisquare(*arrays)

    n = len(pivot)  # datasets
    k = len(methods)  # methods
    kendall_w = chi2 / (n * (k - 1))

    ranks_df = pivot.rank(axis=1, ascending=False)
    mean_ranks = ranks_df.mean().sort_values()

    print(f"\nDatasets: {list(pivot.index)}")
    print(f"Methods: {methods}")
    print(f"Friedman chi^2 = {chi2:.3f}, p = {p:.4f}")
    print(f"Kendall's W = {kendall_w:.3f}")
    print(f"Significant (p < 0.05): {'YES' if p < 0.05 else 'NO'}")
    print(f"\nMean Ranks (lower = better):")
    for method, rank in mean_ranks.items():
        print(f"  {method:20s}: {rank:.2f}")

    # Save
    out = pd.DataFrame({
        'method': mean_ranks.index,
        'mean_rank': mean_ranks.values,
    })
    out.to_csv(os.path.join(RESULTS_DIR, 'friedman_ranks_v1.csv'), index=False)

    return pivot, mean_ranks, chi2, p, kendall_w


def wilcoxon_analysis(df):
    print("\n" + "=" * 70)
    print("WILCOXON SIGNED-RANK (PCA-Triage vs each baseline, 6 datasets)")
    print("With Holm correction for 4 comparisons")
    print("=" * 70)

    baselines = ['Uniform', 'Variance', 'Threshold', 'Random Dropout']
    pivot = df[df['method'].isin(UNSUPERVISED)].pivot(
        index='dataset', columns='method', values='f1'
    )

    results = []
    for bl in baselines:
        pca_vals = pivot['PCA-Triage'].values
        bl_vals = pivot[bl].values
        diff = pca_vals - bl_vals

        wins = np.sum(diff > 0)
        losses = np.sum(diff < 0)
        ties = np.sum(diff == 0)

        try:
            w_stat, p_raw = stats.wilcoxon(
                pca_vals, bl_vals, alternative='greater', zero_method='wilcox'
            )
        except ValueError:
            w_stat, p_raw = 0.0, 1.0

        n_nonzero = np.sum(diff != 0)
        if n_nonzero > 0:
            # rank-biserial r: W is T+ (sum of positive ranks) for alternative='greater'
            r_effect = (2.0 * w_stat) / (n_nonzero * (n_nonzero + 1) / 2) - 1.0
        else:
            r_effect = 0.0

        results.append({
            'baseline': bl,
            'wins': int(wins),
            'losses': int(losses),
            'ties': int(ties),
            'W': w_stat,
            'p_raw': p_raw,
            'effect_r': r_effect,
            'mean_delta': np.mean(diff),
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('p_raw')

    # Holm correction
    m = len(res_df)
    holm_thresholds = [0.05 / (m - i) for i in range(m)]
    res_df['holm_threshold'] = holm_thresholds
    sig_so_far = True
    corrected_sig = []
    for _, row in res_df.iterrows():
        if sig_so_far and row['p_raw'] <= row['holm_threshold']:
            corrected_sig.append(True)
        else:
            sig_so_far = False
            corrected_sig.append(False)
    res_df['sig_holm'] = corrected_sig

    print(f"\n{'Baseline':20s} {'W/L/T':8s} {'W':6s} {'p_raw':8s} {'Holm_thr':8s} {'Sig':5s} {'r':6s} {'delta':8s}")
    print("-" * 80)
    for _, row in res_df.iterrows():
        sig_mark = '*' if row['sig_holm'] else ''
        print(f"{row['baseline']:20s} {row['wins']}/{row['losses']}/{row['ties']}     "
              f"{row['W']:6.1f} {row['p_raw']:8.4f} {row['holm_threshold']:8.4f} "
              f"{sig_mark:5s} {row['effect_r']:6.3f} {row['mean_delta']:+8.4f}")

    res_df.to_csv(os.path.join(RESULTS_DIR, 'wilcoxon_tests_v1.csv'), index=False)

    print(f"\nNote: With n=6 paired observations, the minimum achievable one-sided")
    print(f"p-value is ~0.016. After Holm correction for 4 tests, significance")
    print(f"requires p < 0.0125 for the smallest. Statistical power is limited.")

    return res_df


def main():
    df = load_v1_50pct()
    print(f"Loaded {len(df)} rows: {df['dataset'].nunique()} datasets, "
          f"{df['method'].nunique()} methods")
    print(f"Source: {CANONICAL_SOURCE}\n")

    pivot, ranks, chi2, p_fried, w_fried = friedman_analysis(df)
    wilcox_df = wilcoxon_analysis(df)

    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print(f"\nFriedman: chi^2 = {chi2:.2f}, p = {p_fried:.3f}, Kendall's W = {w_fried:.3f}")
    print(f"PCA-Triage mean rank: {ranks.loc['PCA-Triage']:.2f}")
    print(f"\nWilcoxon (Holm-corrected):")
    for _, row in wilcox_df.iterrows():
        sig = '*' if row['sig_holm'] else 'n.s.'
        print(f"  vs {row['baseline']:20s}: p={row['p_raw']:.3f} ({sig}), r={row['effect_r']:.3f}")


if __name__ == '__main__':
    main()
