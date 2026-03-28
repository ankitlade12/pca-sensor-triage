"""
Plotting utilities for generating publication-quality IEEE figures.

Style: grayscale-friendly, serif fonts matching IEEEtran,
300 DPI, consistent marker/line styles, colorblind-safe palette.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- IEEE-compatible global style ---
plt.rcParams.update({
    'savefig.dpi': 300,
    'figure.dpi': 150,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.framealpha': 0.9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
sns.set_style('whitegrid', {
    'axes.edgecolor': '0.2',
    'grid.color': '0.85',
})

# IEEE column width: 3.5in (single), 7.16in (double)
IEEE_COL_WIDTH = 3.5
IEEE_DBL_WIDTH = 7.16

# Grayscale-friendly palette with distinct markers/linestyles
# Colors chosen to be distinguishable in both color and B&W print
METHOD_COLORS = {
    'PCA-Triage': '#0072B2',    # blue
    'Uniform': '#D55E00',       # vermilion
    'Threshold': '#009E73',     # green
    'Variance': '#CC79A7',      # pink
    'Random Dropout': '#F0E442',# yellow
    'Mutual Info': '#56B4E9',   # light blue
    'Attention': '#E69F00',     # orange
    'Autoencoder': '#999999',   # gray
}
METHOD_MARKERS = {
    'PCA-Triage': 'o',
    'Uniform': 's',
    'Threshold': '^',
    'Variance': 'D',
    'Random Dropout': 'v',
    'Mutual Info': 'P',
    'Attention': 'X',
    'Autoencoder': 'h',
}
METHOD_LINESTYLES = {
    'PCA-Triage': '-',
    'Uniform': '--',
    'Threshold': '-.',
    'Variance': ':',
    'Random Dropout': '--',
    'Mutual Info': '-',
    'Attention': '-.',
    'Autoencoder': ':',
}


def plot_pareto_curves(
    dataframes: Dict[str, pd.DataFrame],
    output_path: str,
    methods: Optional[List[str]] = None,
):
    """Plot multi-panel Pareto curves with shaded ±1σ error bands (IEEE style).

    Parameters
    ----------
    dataframes : dict
        Mapping of dataset_name → DataFrame with columns [method, budget, f1].
    output_path : str
        Path to save figure.
    methods : list, optional
        Methods to include. Defaults to all.
    """
    n_panels = len(dataframes)
    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(IEEE_DBL_WIDTH, 2.4 * nrows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    for idx, (ds_name, df) in enumerate(dataframes.items()):
        ax = axes[idx]
        agg = df.groupby(['method', 'budget']).agg(
            f1_mean=('f1', 'mean'), f1_std=('f1', 'std')
        ).reset_index()

        plot_methods = methods or list(agg['method'].unique())
        for method in plot_methods:
            if method not in METHOD_COLORS:
                continue
            sub = agg[agg['method'] == method].sort_values('budget')
            if len(sub) == 0:
                continue
            x = sub['budget'].values * 100
            y = sub['f1_mean'].values
            yerr = sub['f1_std'].values
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES.get(method, '-')

            ax.fill_between(x, y - yerr, y + yerr, alpha=0.12, color=color)
            ax.plot(x, y, label=method, color=color,
                    marker=METHOD_MARKERS.get(method, 'o'),
                    linestyle=ls)

        ax.set_xlabel('Bandwidth (%)')
        ax.set_ylabel('F1 Score')
        ax.set_title(ds_name, fontweight='bold')
        ax.legend(loc='lower right', frameon=True)
        ax.set_xlim(5, 95)

    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Pareto Curves: Accuracy vs Bandwidth (6 Real-World Datasets)',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_importance_heatmap(
    importance: np.ndarray,
    sensor_cols: List[str],
    window_size: int,
    sampling_interval: int = 3,
    fault_onset_sample: Optional[int] = None,
    title: str = 'Channel Importance Over Time',
    output_path: str = 'importance_heatmap.png',
):
    """Plot channel importance heatmap over time (IEEE style)."""
    window_times = np.arange(importance.shape[0]) * window_size * sampling_interval

    fig, ax = plt.subplots(figsize=(IEEE_DBL_WIDTH, 3.5))
    im = ax.imshow(importance.T, aspect='auto', cmap='YlOrRd',
                   extent=[0, window_times[-1], len(sensor_cols) - 0.5, -0.5])

    if fault_onset_sample is not None:
        fault_time = fault_onset_sample * sampling_interval
        ax.axvline(x=fault_time, color='white', linewidth=2, linestyle='--',
                   label='Fault onset')
        ax.legend(loc='upper right')

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Channel')
    step = max(1, len(sensor_cols) // 10)
    ax.set_yticks(range(0, len(sensor_cols), step))
    ax.set_yticklabels([sensor_cols[i] for i in range(0, len(sensor_cols), step)])
    ax.set_title(title, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Importance Score')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_compute_comparison(
    results: Dict[str, Dict],
    output_path: str,
):
    """Plot compute cost bar chart (IEEE style)."""
    methods = list(results.keys())
    times = [results[m]['ms_per_window'] for m in methods]
    colors = [METHOD_COLORS.get(m, '#999') for m in methods]

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.5))
    bars = ax.bar(methods, times, color=colors, edgecolor='0.3', linewidth=0.5)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f'{t:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Time per Window (ms)')
    ax.set_title('Compute Cost per Triage Decision', fontweight='bold')
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.4, label='5 ms edge target')
    ax.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_ablation_grid(
    ablation_df: pd.DataFrame,
    output_path: str,
):
    """Plot 2x2 ablation grid (IEEE style)."""
    fig, axes = plt.subplots(2, 2, figsize=(IEEE_DBL_WIDTH, 4.5))

    params = [
        ('k', 'Number of Components (k)', '#0072B2'),
        ('w', 'Window Size (w)', '#D55E00'),
        ('lambda', 'Forgetting Factor (λ)', '#009E73'),
    ]

    for idx, (param, xlabel, color) in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        sub = ablation_df[ablation_df['param'] == param].sort_values('value')
        ax.errorbar(sub['value'], sub['f1_mean'], yerr=sub['f1_std'],
                    fmt='o-', color=color, capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('F1 Score')
        ax.set_title(f'({chr(97 + idx)}) Effect of {param}', fontweight='bold')

    ax = axes[1, 1]
    sub = ablation_df[ablation_df['param'] == 'score_formula']
    if len(sub) > 0:
        colors_f = ['#0072B2', '#D55E00', '#009E73'][:len(sub)]
        bars = ax.bar(sub['value'], sub['f1_mean'], color=colors_f,
                      edgecolor='0.3', linewidth=0.5)
        for bar, f in zip(bars, sub['f1_mean']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f'{f:.4f}', ha='center', fontsize=7, fontweight='bold')
        ax.set_ylabel('F1 Score')
        ax.set_title('(d) Score Formula Comparison', fontweight='bold')

    fig.suptitle('Ablation Studies — PCA-Triage on TEP (50% Bandwidth)',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scalability(
    dims: List[int],
    times: Dict[str, List[float]],
    output_path: str,
):
    """Plot scalability: compute time vs number of channels (IEEE style)."""
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, 2.5))

    styles = {
        'PCA-Triage': ('o-', METHOD_COLORS['PCA-Triage'], 'PCA-Triage $O(wdk)$'),
        'Variance': ('s--', METHOD_COLORS['Variance'], 'Variance $O(wd)$'),
        'Attention': ('^-.', METHOD_COLORS['Attention'], 'Attention $O(d^2w)$'),
    }

    for method, (fmt, color, label) in styles.items():
        if method in times:
            ax.plot(dims, times[method], fmt, color=color, label=label)

    ax.set_xlabel('Number of Channels ($d$)')
    ax.set_ylabel('Time per Window (ms)')
    ax.set_title('Scalability: Compute Time vs Channels', fontweight='bold')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(y=5, color='red', linestyle=':', alpha=0.4)
    ax.text(dims[0] * 1.2, 6, '5 ms edge target', fontsize=7, color='red', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
