#!/usr/bin/env python3
"""
Generate all publication figures from experiment results.

Usage:
    python paper/generate_all_figures.py

Reads from experiments/results/ and writes to paper/figures/.
All figures are 300 DPI, colorblind-friendly palette.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Consistent style
sns.set_style('whitegrid')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

PALETTE = sns.color_palette('colorblind', 6)
COLORS = {
    'PCA-Triage': PALETTE[0], 'Uniform': PALETTE[1], 'Threshold': PALETTE[2],
    'Variance': PALETTE[3], 'Random Dropout': PALETTE[4], 'Mutual Info': PALETTE[5],
}
MARKERS = {
    'PCA-Triage': 'o', 'Uniform': 's', 'Threshold': '^',
    'Variance': 'D', 'Random Dropout': 'v', 'Mutual Info': 'P',
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def fig_pareto_curves():
    """Figure 2: Pareto curves across all datasets."""
    # Include all datasets that have pareto CSVs
    all_datasets = ['tep', 'smd', 'psm', 'msl', 'hai', 'skab', 'swat', 'wadi']
    datasets = [ds for ds in all_datasets
                if os.path.exists(os.path.join(RESULTS_DIR, f'pareto_{ds}.csv'))]
    methods = list(COLORS.keys())

    ncols = 4 if len(datasets) > 6 else 3
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols, idx % ncols] if nrows > 1 else axes[idx % ncols]
        path = os.path.join(RESULTS_DIR, f'pareto_{ds}.csv')
        if not os.path.exists(path):
            ax.set_title(f'{ds.upper()} — NO DATA')
            continue
        df = pd.read_csv(path)
        agg = df.groupby(['method', 'budget']).agg(
            f1_mean=('f1', 'mean'), f1_std=('f1', 'std')
        ).reset_index()
        for method in methods:
            sub = agg[agg['method'] == method].sort_values('budget')
            if len(sub) == 0:
                continue
            ax.errorbar(sub['budget'] * 100, sub['f1_mean'], yerr=sub['f1_std'],
                        label=method, color=COLORS[method], marker=MARKERS[method],
                        markersize=6, linewidth=2, capsize=2)
        ax.set_xlabel('Bandwidth (%)')
        ax.set_ylabel('F1')
        ax.set_title(f'{ds.upper()}', fontsize=13, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')
        ax.set_xlim(5, 95)

    # Hide unused axes
    total_axes = nrows * ncols
    for idx in range(len(datasets), total_axes):
        ax = axes[idx // ncols, idx % ncols] if nrows > 1 else axes[idx % ncols]
        ax.set_visible(False)

    plt.suptitle('Pareto Curves: Accuracy vs Bandwidth', fontsize=15, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'pareto_curves.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


def fig_compute_cost():
    """Figure 4: Compute cost bar chart."""
    path = os.path.join(RESULTS_DIR, 'compute_profile.json')
    if not os.path.exists(path):
        print('  SKIP compute_cost — no data')
        return
    with open(path) as f:
        results = json.load(f)

    methods = list(results.keys())
    times = [results[m]['ms_per_window'] for m in methods]
    colors = [COLORS.get(m, '#999') for m in methods]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(methods, times, color=colors, edgecolor='black', linewidth=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{t:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Time per Window (ms)', fontsize=12)
    ax.set_title('Compute Cost per Triage Decision', fontsize=13, fontweight='bold')
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.4, label='5ms edge target')
    ax.legend(fontsize=10)
    plt.xticks(rotation=15, fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'compute_cost.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


def fig_ablation_grid():
    """Figure 5: Ablation 2x2 grid."""
    path = os.path.join(os.path.dirname(__file__), '..', 'paper', 'tables', 'table3_ablation.csv')
    if not os.path.exists(path):
        print('  SKIP ablation_grid — no data')
        return
    df = pd.read_csv(path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    params = [('k', 'Number of Components (k)', 'steelblue'),
              ('w', 'Window Size (w)', 'coral'),
              ('lambda', 'Forgetting Factor (λ)', 'seagreen')]

    for idx, (param, xlabel, color) in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        sub = df[df['param'] == param].sort_values('value')
        ax.errorbar(sub['value'], sub['f1_mean'], yerr=sub['f1_std'],
                    fmt='o-', color=color, linewidth=2, markersize=8, capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('F1 Score')
        ax.set_title(f'({chr(97 + idx)}) Effect of {param}')

    ax = axes[1, 1]
    sub = df[df['param'] == 'score_formula']
    if len(sub) > 0:
        colors_f = ['steelblue', 'coral', 'seagreen'][:len(sub)]
        bars = ax.bar(range(len(sub)), sub['f1_mean'].values, color=colors_f,
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub['value'].values, fontsize=9)
        for bar, f in zip(bars, sub['f1_mean'].values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f'{f:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('F1 Score')
    ax.set_title('(d) Score Formula Comparison')

    fig.suptitle('Ablation Studies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'ablation_grid.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


def fig_scalability():
    """Figure 6: Scalability plot."""
    path = os.path.join(RESULTS_DIR, 'scalability.json')
    if not os.path.exists(path):
        print('  SKIP scalability — no data')
        return
    with open(path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, fmt, color, label in [
        ('PCA-Triage', 'o-', COLORS['PCA-Triage'], 'PCA-Triage O(wdk)'),
        ('Variance', 's--', COLORS['Variance'], 'Variance O(wd)'),
        ('Attention', '^-.', '#e74c3c', 'Attention O(d²w)'),
    ]:
        if method in data:
            ax.plot(data['dims'], data[method], fmt, color=color,
                    linewidth=2.5, markersize=9, label=label)

    ax.set_xlabel('Number of Channels (d)', fontsize=12)
    ax.set_ylabel('Time per Window (ms)', fontsize=12)
    ax.set_title('Scalability: Compute Time vs Sensor Channels', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(y=5, color='red', linestyle=':', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'scalability.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print('Generating all figures...')
    fig_pareto_curves()
    fig_compute_cost()
    fig_ablation_grid()
    fig_scalability()
    print('Done.')
