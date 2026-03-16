"""
Experiment 1: Pareto Curves — Accuracy vs Bandwidth Trade-off.

For each method × bandwidth level × random seed, runs the full pipeline
and logs downstream task performance. Produces the paper's key figure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.triage import TriagePipeline
from src.baselines import (
    UniformSampling, ThresholdSampling, VarianceSampling,
    RandomDropout, MutualInfoSampling,
)


def run_single(method_name, X_train, y_train, X_test, y_test,
               budget, seed, n_components=10, window_size=50):
    """Run a single method at a given budget and seed."""

    t0 = time.time()

    if method_name == "PCA-Triage":
        pipe = TriagePipeline(
            n_components=n_components, window_size=window_size,
            budget=budget, forgetting_factor=1.0, min_rate=0.05,
        )
        X_tr = pipe.process_stream(X_train, seed=seed)
        X_te = pipe.process_stream(X_test, seed=seed + 10000)

    elif method_name == "Uniform":
        uni = UniformSampling(budget=budget)
        X_tr = uni.process_stream(X_train, seed=seed)
        X_te = uni.process_stream(X_test, seed=seed + 10000)

    elif method_name == "Threshold":
        thr = ThresholdSampling(budget=budget, window_size=window_size)
        X_tr = thr.process_stream(X_train, seed=seed)
        X_te = thr.process_stream(X_test, seed=seed + 10000)

    elif method_name == "Variance":
        var = VarianceSampling(budget=budget, window_size=window_size)
        X_tr = var.process_stream(X_train, seed=seed)
        X_te = var.process_stream(X_test, seed=seed + 10000)

    elif method_name == "Random Dropout":
        rd = RandomDropout(budget=budget, window_size=window_size)
        X_tr = rd.process_stream(X_train, seed=seed)
        X_te = rd.process_stream(X_test, seed=seed + 10000)

    elif method_name == "Mutual Info":
        mi = MutualInfoSampling(budget=budget)
        mi.fit(X_train, y_train)
        X_tr = mi.process_stream(X_train, seed=seed)
        X_te = mi.process_stream(X_test, seed=seed + 10000)

    elif method_name == "Full Data":
        X_tr = X_train.copy()
        X_te = X_test.copy()

    else:
        raise ValueError(f"Unknown method: {method_name}")

    elapsed = time.time() - t0

    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return {"f1": f1, "time": elapsed}


def run_pareto_experiment(
    X_train, y_train, X_test, y_test,
    dataset_name="TEP",
    budgets=None,
    seeds=None,
    methods=None,
):
    """Run full Pareto experiment across methods, budgets, seeds."""

    if budgets is None:
        budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]
    if methods is None:
        methods = ["PCA-Triage", "Uniform", "Threshold", "Variance",
                   "Random Dropout", "Mutual Info"]

    results = []
    total = len(methods) * len(budgets) * len(seeds)
    count = 0

    for method in methods:
        for budget in budgets:
            for seed in seeds:
                count += 1
                if method == "Full Data" and budget < 1.0:
                    continue

                try:
                    res = run_single(method, X_train, y_train, X_test, y_test,
                                     budget=budget, seed=seed)
                    results.append({
                        "dataset": dataset_name,
                        "method": method,
                        "budget": budget,
                        "seed": seed,
                        "f1": res["f1"],
                        "time": res["time"],
                    })
                    print(f"  [{count}/{total}] {method:15s} B={budget:.1f} seed={seed:4d} "
                          f"F1={res['f1']:.4f} ({res['time']:.1f}s)")
                except Exception as e:
                    print(f"  [{count}/{total}] {method:15s} B={budget:.1f} seed={seed:4d} ERROR: {e}")
                    results.append({
                        "dataset": dataset_name,
                        "method": method,
                        "budget": budget,
                        "seed": seed,
                        "f1": np.nan,
                        "time": np.nan,
                    })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import pyreadr

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

    # --- TEP ---
    print("=" * 60)
    print("LOADING TEP DATASET")
    print("=" * 60)

    ff = pyreadr.read_r(os.path.join(DATA_DIR, 'TEP_FaultFree_Training.RData'))
    ff_df = ff[list(ff.keys())[0]]
    ft = pyreadr.read_r(os.path.join(DATA_DIR, 'TEP_Faulty_Training.RData'))
    ft_df = ft[list(ft.keys())[0]]

    sensor_cols = [c for c in ff_df.columns if c.startswith('xmeas_') or c.startswith('xmv_')]
    fault_types = [0, 1, 2, 4, 5]

    X_tr_parts, y_tr_parts, X_te_parts, y_te_parts = [], [], [], []
    for fault in fault_types:
        if fault == 0:
            tr = ff_df[ff_df['simulationRun'] <= 20]
            te = ff_df[(ff_df['simulationRun'] > 21) & (ff_df['simulationRun'] <= 30)]
        else:
            tr = ft_df[(ft_df['faultNumber'] == fault) & (ft_df['simulationRun'] <= 20)]
            te = ft_df[(ft_df['faultNumber'] == fault) & (ft_df['simulationRun'] > 21) & (ft_df['simulationRun'] <= 30)]
        X_tr_parts.append(tr[sensor_cols].values)
        y_tr_parts.append(np.full(len(tr), fault))
        X_te_parts.append(te[sensor_cols].values)
        y_te_parts.append(np.full(len(te), fault))

    X_train = np.vstack(X_tr_parts)
    y_train = np.concatenate(y_tr_parts)
    X_test = np.vstack(X_te_parts)
    y_test = np.concatenate(y_te_parts)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"Train: {X_train_s.shape}, Test: {X_test_s.shape}")
    print()

    # Run with fewer seeds for speed (3 instead of 5)
    df_tep = run_pareto_experiment(
        X_train_s, y_train, X_test_s, y_test,
        dataset_name="TEP",
        seeds=[42, 123, 456],
        budgets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )

    # Save results
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    df_tep.to_csv(os.path.join(os.path.dirname(__file__), 'results', 'pareto_tep.csv'), index=False)
    print(f"\nSaved {len(df_tep)} results to experiments/results/pareto_tep.csv")
