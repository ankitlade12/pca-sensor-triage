"""
Experiment 3: Adaptivity Under Fault Onset.

Analyzes how PCA-Triage reallocates bandwidth when system dynamics change.
Produces adaptivity heatmaps and reaction time analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.triage import TriagePipeline
from src.utils import load_tep_single_fault


def analyze_fault_adaptivity(
    fault_num: int,
    run: int = 1,
    n_components: int = 10,
    window_size: int = 25,
    budget: float = 0.5,
    forgetting_factor: float = 0.85,
):
    """Analyze how importance shifts after fault onset."""
    X, sensor_cols = load_tep_single_fault(fault_num, run=run)

    pipe = TriagePipeline(
        n_components=n_components, window_size=window_size,
        budget=budget, forgetting_factor=forgetting_factor, min_rate=0.05,
    )
    pipe.process_stream(X, seed=42)

    importance = pipe.get_importance_matrix()
    rates = pipe.get_rate_matrix()

    # Pre-fault (window 0) vs post-fault (window 3)
    pre = importance[0]
    post = importance[min(3, len(importance) - 1)]
    delta = post - pre

    top_gainers = np.argsort(delta)[::-1][:5]
    top_losers = np.argsort(delta)[:5]

    return {
        'fault_num': fault_num,
        'importance': importance,
        'rates': rates,
        'sensor_cols': sensor_cols,
        'top_gainers': [(sensor_cols[i], float(delta[i])) for i in top_gainers],
        'top_losers': [(sensor_cols[i], float(delta[i])) for i in top_losers],
    }


def measure_reaction_times(
    fault_nums=[1, 2, 4, 5],
    lambdas=[0.80, 0.85, 0.90, 0.95, 0.99, 1.0],
    window_size=25,
):
    """Measure reaction time (windows until top-5 channels change)."""
    results = []

    for fault_num in fault_nums:
        X, sensor_cols = load_tep_single_fault(fault_num, run=1)

        for lam in lambdas:
            pipe = TriagePipeline(
                n_components=10, window_size=window_size, budget=0.5,
                forgetting_factor=lam, min_rate=0.05,
            )
            pipe.process_stream(X, seed=42)
            importance = pipe.get_importance_matrix()

            pre_top5 = set(np.argsort(importance[0])[::-1][:5])
            reaction = len(importance) - 1
            for wi in range(1, len(importance)):
                cur_top5 = set(np.argsort(importance[wi])[::-1][:5])
                if cur_top5 != pre_top5:
                    reaction = wi - 1
                    break

            results.append({
                'fault': fault_num,
                'lambda': lam,
                'reaction_windows': reaction,
                'reaction_minutes': reaction * window_size * 3,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== Fault Adaptivity Analysis ===\n")

    for fault in [1, 2, 4, 5]:
        result = analyze_fault_adaptivity(fault)
        print(f"Fault {fault}:")
        print(f"  Top gainers: {result['top_gainers'][:3]}")
        print(f"  Top losers:  {result['top_losers'][:3]}")
        print()

    print("=== Reaction Time Analysis ===\n")
    df = measure_reaction_times()

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'reaction_times.csv'), index=False)

    pivot = df.pivot_table(values='reaction_windows', index='fault', columns='lambda')
    print(pivot.to_string())
    print(f"\nSaved to {out_dir}/reaction_times.csv")
