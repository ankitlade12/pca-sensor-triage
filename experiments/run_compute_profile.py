"""
Experiment 2: Compute Cost Profiling.

Profiles wall-clock time, peak memory, and parameter count
for all methods under laptop and edge-simulated conditions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import json
import tracemalloc
import numpy as np

from src.triage import TriagePipeline
from src.baselines import (
    UniformSampling, ThresholdSampling, VarianceSampling,
    RandomDropout, AttentionSampling,
)


def profile_method(name, create_fn, data, n_iter):
    """Profile a single method."""
    method = create_fn()
    tracemalloc.start()

    t0 = time.perf_counter()
    _ = method.process_stream(data, seed=42)
    elapsed = time.perf_counter() - t0

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ms_per_window = (elapsed / n_iter) * 1000
    peak_mb = peak / 1024 / 1024

    n_params = 0
    if hasattr(method, 'n_parameters'):
        n_params = method.n_parameters
    elif hasattr(method, 'triage'):
        n_params = method.triage.n_components * data.shape[1]

    return {
        'ms_per_window': ms_per_window,
        'peak_mb': peak_mb,
        'n_params': n_params,
        'total_time': elapsed,
    }


def run_profiling(d=52, w=100, n_iter=200, single_threaded=False):
    if single_threaded:
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

    np.random.seed(42)
    data = np.random.randn(n_iter * w, d)

    methods = {
        'PCA-Triage': lambda: TriagePipeline(
            n_components=10, window_size=w, budget=0.5,
            forgetting_factor=1.0, min_rate=0.05),
        'Uniform': lambda: UniformSampling(budget=0.5),
        'Threshold': lambda: ThresholdSampling(budget=0.5, window_size=w),
        'Variance': lambda: VarianceSampling(budget=0.5, window_size=w),
        'Random Dropout': lambda: RandomDropout(budget=0.5, window_size=w),
        'Attention': lambda: AttentionSampling(budget=0.5, window_size=w, hidden_dim=32),
    }

    mode = "edge-simulated" if single_threaded else "laptop"
    print(f"{'='*60}")
    print(f"COMPUTE PROFILING ({mode}): {n_iter} windows, w={w}, d={d}")
    print(f"{'='*60}")
    print(f"{'Method':20s} | {'ms/window':>12s} | {'Peak MB':>8s} | {'Params':>8s}")
    print('-' * 55)

    results = {}
    for name, create_fn in methods.items():
        res = profile_method(name, create_fn, data, n_iter)
        results[name] = res
        print(f"{name:20s} | {res['ms_per_window']:10.3f}ms | {res['peak_mb']:6.2f}MB | {res['n_params']:>8d}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge', action='store_true', help='Simulate edge constraints')
    parser.add_argument('--d', type=int, default=52, help='Number of channels')
    parser.add_argument('--w', type=int, default=100, help='Window size')
    parser.add_argument('--n-iter', type=int, default=200, help='Number of windows')
    args = parser.parse_args()

    results = run_profiling(d=args.d, w=args.w, n_iter=args.n_iter,
                           single_threaded=args.edge)

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    suffix = '_edge' if args.edge else ''
    with open(os.path.join(out_dir, f'compute_profile{suffix}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/compute_profile{suffix}.json")
