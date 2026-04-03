"""
Full Pareto sweep with v2 PCA-Triage across all budgets and datasets.

Generates per-dataset CSVs compatible with the figure generator.
Uses 3 seeds for speed (vs 5 for the budget=0.5 comparison).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from src.triage.pipeline import TriagePipeline
from src.baselines import (
    UniformSampling, ThresholdSampling, VarianceSampling,
    RandomDropout, MutualInfoSampling,
)
from src.utils.data_loader import get_dataset

# Import configs from run_pareto_v2
from run_pareto_v2 import DATASET_CONFIGS


def run_single(method, X_train, y_train_enc, X_test, y_test_enc,
               budget, seed, dataset_name):
    """Run one method at one budget and seed."""
    t0 = time.time()

    if method == "PCA-Triage":
        cfg = DATASET_CONFIGS[dataset_name]
        pipe = TriagePipeline(
            n_components=cfg['n_components'],
            window_size=cfg['window_size'],
            budget=budget,
            forgetting_factor=cfg['forgetting_factor'],
            min_rate=cfg['min_rate'],
            reconstruction_method=cfg['reconstruction_method'],
            scorer=cfg['scorer'],
            alpha=cfg['alpha'],
            sharpness=cfg.get('sharpness', 1.0),
        )
        X_tr = pipe.process_stream(X_train, seed=seed)
        X_te = pipe.process_stream(X_test, seed=seed + 10000)
    elif method == "Uniform":
        bl = UniformSampling(budget=budget)
        X_tr = bl.process_stream(X_train, seed=seed)
        X_te = bl.process_stream(X_test, seed=seed + 10000)
    elif method == "Threshold":
        bl = ThresholdSampling(budget=budget, window_size=50)
        X_tr = bl.process_stream(X_train, seed=seed)
        X_te = bl.process_stream(X_test, seed=seed + 10000)
    elif method == "Variance":
        bl = VarianceSampling(budget=budget, window_size=50)
        X_tr = bl.process_stream(X_train, seed=seed)
        X_te = bl.process_stream(X_test, seed=seed + 10000)
    elif method == "Random Dropout":
        bl = RandomDropout(budget=budget, window_size=50)
        X_tr = bl.process_stream(X_train, seed=seed)
        X_te = bl.process_stream(X_test, seed=seed + 10000)
    elif method == "Mutual Info":
        bl = MutualInfoSampling(budget=budget)
        bl.fit(X_train, y_train_enc)
        X_tr = bl.process_stream(X_train, seed=seed)
        X_te = bl.process_stream(X_test, seed=seed + 10000)
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - t0
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_train_enc)
    y_pred = clf.predict(X_te)
    f1 = f1_score(y_test_enc, y_pred, average='weighted')
    return f1, elapsed


def run_dataset_pareto(dataset_name):
    """Run full Pareto sweep for one dataset."""
    print(f"\n{'='*60}", flush=True)
    print(f"PARETO SWEEP: {dataset_name.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    result = get_dataset(dataset_name)
    X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    print(f"  Shape: {X_train.shape}", flush=True)

    budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    seeds = [42, 123, 456]
    methods = ["PCA-Triage", "Uniform", "Threshold", "Variance",
               "Random Dropout", "Mutual Info"]

    rows = []
    total = len(methods) * len(budgets) * len(seeds)
    count = 0

    for method in methods:
        for budget in budgets:
            for seed in seeds:
                count += 1
                try:
                    f1, elapsed = run_single(
                        method, X_train, y_train_enc, X_test, y_test_enc,
                        budget, seed, dataset_name)
                    rows.append({
                        'dataset': dataset_name,
                        'method': method,
                        'budget': budget,
                        'seed': seed,
                        'f1': f1,
                        'time': elapsed,
                    })
                    if count % 10 == 0:
                        print(f"  [{count}/{total}] {method:15s} B={budget:.1f} "
                              f"seed={seed} F1={f1:.4f}", flush=True)
                except Exception as e:
                    print(f"  [{count}/{total}] {method:15s} B={budget:.1f} "
                          f"seed={seed} ERROR: {e}", flush=True)

    df = pd.DataFrame(rows)

    # Save per-dataset CSV (overwrites old v1 files)
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    out_path = os.path.join(out_dir, f'pareto_{dataset_name}.csv')
    df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(df)} rows to {out_path}", flush=True)

    # Print summary at budget=0.5
    df50 = df[df['budget'] == 0.5]
    summary = df50.groupby('method')['f1'].mean().sort_values(ascending=False)
    print(f"\n  Summary @ B=0.5:", flush=True)
    for method, f1 in summary.items():
        marker = " ***" if method == "PCA-Triage" else ""
        print(f"    {method:20s}: {f1:.4f}{marker}", flush=True)

    return df


if __name__ == "__main__":
    datasets = ['tep', 'smd', 'msl', 'psm', 'hai', 'skab', 'swat']
    for ds in datasets:
        try:
            run_dataset_pareto(ds)
        except Exception as e:
            print(f"ERROR on {ds}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\n\nDone! Regenerate figures with: python paper/generate_all_figures.py", flush=True)
