"""
Experiment 5: Scalability Analysis.

Measures compute time vs number of sensor channels for all methods.
Demonstrates O(wdk) scaling for PCA-Triage vs O(d²w) for attention.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import json
import numpy as np

from src.triage import TriagePipeline
from src.baselines import VarianceSampling, AttentionSampling


def run_scalability(dims=None, w=100, n_windows=50):
    if dims is None:
        dims = [10, 25, 50, 100, 200, 500]

    results = {'dims': dims, 'PCA-Triage': [], 'Variance': [], 'Attention': []}

    print(f"SCALABILITY: Time per window vs channels (w={w}, n_windows={n_windows})")
    print(f"{'d':>5s} | {'PCA-Triage':>12s} | {'Variance':>12s} | {'Attention':>12s}")
    print('-' * 55)

    for d in dims:
        data = np.random.randn(n_windows * w, d)

        for name, create_fn in [
            ('PCA-Triage', lambda d=d: TriagePipeline(
                n_components=min(10, d-1), window_size=w, budget=0.5,
                forgetting_factor=1.0, min_rate=0.05)),
            ('Variance', lambda: VarianceSampling(budget=0.5, window_size=w)),
            ('Attention', lambda: AttentionSampling(budget=0.5, window_size=w, hidden_dim=32)),
        ]:
            method = create_fn()
            t0 = time.perf_counter()
            method.process_stream(data, seed=42)
            elapsed = (time.perf_counter() - t0) / n_windows * 1000
            results[name].append(elapsed)

        print(f"{d:5d} | {results['PCA-Triage'][-1]:10.2f}ms | "
              f"{results['Variance'][-1]:10.2f}ms | {results['Attention'][-1]:10.2f}ms")

    return results


if __name__ == "__main__":
    results = run_scalability()

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'scalability.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}/scalability.json")
