"""
Plotting utilities for generating publication-quality figures.

All figures use consistent styling: colorblind-friendly palette,
300 DPI, proper font sizes, and consistent marker/line styles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Consistent style
sns.set_style('whitegrid')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

# Method styling
PALETTE = sns.color_palette('colorblind', 7)
METHOD_COLORS = {
    'PCA-Triage': PALETTE[0],
    'Uniform': PALETTE[1],
    'Threshold': PALETTE[2],
    'Variance': PALETTE[3],
    'Random Dropout': PALETTE[4],
    'Mutual Info': PALETTE[5],
    'Attention': '#e74c3c',
}
METHOD_MARKERS = {
    'PCA-Triage': 'o',
    'Uniform': 's',
    'Threshold': '^',
    'Variance': 'D',
    'Random Dropout': 'v',
    'Mutual Info': 'P',
    'Attention': 'X',
}


def plot_pareto_curves(
    dataframes: Dict[str, pd.DataFrame],
    output_path: str,
    methods: Optional[List[str]] = None,
):
    """Plot multi-panel Pareto curves (accuracy vs bandwidth).

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
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    for ax, (ds_name, df) in zip(axes, dataframes.items()):
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
            ax.errorbar(
                sub['budget'] * 100, sub['f1_mean'], yerr=sub['f1_std'],
                label=method, color=METHOD_COLORS[method],
                marker=METHOD_MARKERS.get(method, 'o'),
                markersize=7, linewidth=2, capsize=3,
            )

        ax.set_xlabel('Bandwidth Budget (%)', fontsize=12)
        ax.set_ylabel('Weighted F1 Score', fontsize=12)
        ax.set_title(f'{ds_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim(5, 95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    """Plot channel importance heatmap over time."""
    window_times = np.arange(importance.shape[0]) * window_size * sampling_interval

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(importance.T, aspect='auto', cmap='YlOrRd',
                   extent=[0, window_times[-1], len(sensor_cols) - 0.5, -0.5])

    if fault_onset_sample is not None:
        fault_time = fault_onset_sample * sampling_interval
        ax.axvline(x=fault_time, color='white', linewidth=2.5, linestyle='--',
                   label='Fault onset')
        ax.legend(loc='upper right', fontsize=10)

    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_yticks(range(0, len(sensor_cols), max(1, len(sensor_cols) // 10)))
    ax.set_yticklabels(
        [sensor_cols[i] for i in range(0, len(sensor_cols), max(1, len(sensor_cols) // 10))],
        fontsize=8
    )
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Importance Score')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_compute_comparison(
    results: Dict[str, Dict],
    output_path: str,
):
    """Plot compute cost bar chart."""
    methods = list(results.keys())
    times = [results[m]['ms_per_window'] for m in methods]
    colors = [METHOD_COLORS.get(m, '#999') for m in methods]

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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_grid(
    ablation_df: pd.DataFrame,
    output_path: str,
):
    """Plot 2×2 ablation grid (k, w, lambda, score formula)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    params = [
        ('k', 'Number of Components (k)', 'steelblue'),
        ('w', 'Window Size (w)', 'coral'),
        ('lambda', 'Forgetting Factor (λ)', 'seagreen'),
    ]

    for idx, (param, xlabel, color) in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        sub = ablation_df[ablation_df['param'] == param].sort_values('value')
        ax.errorbar(sub['value'], sub['f1_mean'], yerr=sub['f1_std'],
                    fmt='o-', color=color, linewidth=2, markersize=8, capsize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('F1 Score')
        ax.set_title(f'({chr(97 + idx)}) Effect of {param}')

    # Score formula (bar chart)
    ax = axes[1, 1]
    sub = ablation_df[ablation_df['param'] == 'score_formula']
    if len(sub) > 0:
        colors_f = ['steelblue', 'coral', 'seagreen'][:len(sub)]
        bars = ax.bar(sub['value'], sub['f1_mean'], color=colors_f,
                      edgecolor='black', linewidth=0.5)
        for bar, f in zip(bars, sub['f1_mean']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f'{f:.4f}', ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel('F1 Score')
        ax.set_title('(d) Score Formula Comparison')

    fig.suptitle('Ablation Studies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scalability(
    dims: List[int],
    times: Dict[str, List[float]],
    output_path: str,
):
    """Plot scalability: compute time vs number of channels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {
        'PCA-Triage': ('o-', METHOD_COLORS['PCA-Triage'], 'PCA-Triage O(wdk)'),
        'Variance': ('s--', METHOD_COLORS['Variance'], 'Variance O(wd)'),
        'Attention': ('^-.', '#e74c3c', 'Attention O(d²w)'),
    }

    for method, (fmt, color, label) in styles.items():
        if method in times:
            ax.plot(dims, times[method], fmt, color=color, linewidth=2.5,
                    markersize=9, label=label)

    ax.set_xlabel('Number of Channels (d)', fontsize=12)
    ax.set_ylabel('Time per Window (ms)', fontsize=12)
    ax.set_title('Scalability: Compute Time vs Sensor Channels', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(y=5, color='red', linestyle=':', alpha=0.4)
    ax.text(dims[0] * 1.2, 5.5, '5ms edge target', fontsize=9, color='red', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
