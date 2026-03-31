"""
Experiment 9: Per-Fault-Type Breakdown on TEP.

Evaluates PCA-Triage and baselines on each TEP fault type individually
(binary: normal vs fault_i) at 50% bandwidth.

Optimized: loads TEP data ONCE and caches it.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from src.triage import TriagePipeline
from src.baselines import UniformSampling, VarianceSampling, ThresholdSampling
import pyreadr

FAULT_NAMES = {
    1: "A/C feed ratio step",
    2: "B composition step",
    4: "Reactor CW temp step",
    5: "Condenser CW temp step",
    6: "A feed loss",
    7: "C header pressure loss",
    11: "Reactor CW temp random",
    12: "Condenser CW temp random",
    13: "Reaction kinetics drift",
    14: "Reactor CW valve stick",
}


def run_per_fault():
    print("=" * 70, flush=True)
    print("PER-FAULT-TYPE BREAKDOWN ON TEP (50% Bandwidth)", flush=True)
    print("=" * 70, flush=True)

    # Load data ONCE
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    print("Loading TEP data (one-time)...", flush=True)
    ff = pyreadr.read_r(os.path.join(DATA_DIR, 'TEP_FaultFree_Training.RData'))
    ff_df = ff[list(ff.keys())[0]]
    ft = pyreadr.read_r(os.path.join(DATA_DIR, 'TEP_Faulty_Training.RData'))
    ft_df = ft[list(ft.keys())[0]]
    sensor_cols = [c for c in ff_df.columns
                   if c.startswith('xmeas_') or c.startswith('xmv_')]
    print(f"Loaded. FF: {len(ff_df)}, FT: {len(ft_df)}, {len(sensor_cols)} channels", flush=True)

    # Cache normal data
    tr_norm = ff_df[ff_df['simulationRun'] <= 20][sensor_cols].values
    te_norm = ff_df[(ff_df['simulationRun'] > 21) & (ff_df['simulationRun'] <= 30)][sensor_cols].values

    fault_types = [1, 2, 4, 5, 6, 7, 11, 12, 13, 14]
    seeds = [42, 123, 456]
    budget = 0.5
    results = []

    for fault_num in fault_types:
        print(f"\n--- IDV({fault_num}): {FAULT_NAMES[fault_num]} ---", flush=True)

        # Extract fault data
        tr_fault_df = ft_df[(ft_df['faultNumber'] == fault_num) & (ft_df['simulationRun'] <= 20)]
        te_fault_df = ft_df[(ft_df['faultNumber'] == fault_num) &
                            (ft_df['simulationRun'] > 21) & (ft_df['simulationRun'] <= 30)]

        if len(tr_fault_df) == 0 or len(te_fault_df) == 0:
            print(f"  SKIPPED: no data for fault {fault_num}", flush=True)
            continue

        tr_fault = tr_fault_df[sensor_cols].values
        te_fault = te_fault_df[sensor_cols].values

        X_train_raw = np.vstack([tr_norm, tr_fault])
        y_train_raw = np.concatenate([np.zeros(len(tr_norm)), np.ones(len(tr_fault))])
        X_test_raw = np.vstack([te_norm, te_fault])
        y_test_raw = np.concatenate([np.zeros(len(te_norm)), np.ones(len(te_fault))])

        method_configs = [
            ('PCA-Triage', 'pca'),
            ('Variance', 'var'),
            ('Threshold', 'thr'),
            ('Uniform', 'uni'),
            ('Full Data', 'full'),
        ]

        for method_name, method_key in method_configs:
            f1s = []
            for seed in seeds:
                # Shuffle
                rng = np.random.RandomState(seed)
                tr_idx = rng.permutation(len(y_train_raw))
                te_idx = rng.permutation(len(y_test_raw))
                X_train = X_train_raw[tr_idx]
                y_train = y_train_raw[tr_idx]
                X_test = X_test_raw[te_idx]
                y_test = y_test_raw[te_idx]

                # Scale
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                if method_key == 'pca':
                    pipe = TriagePipeline(n_components=10, window_size=50,
                                         budget=budget, forgetting_factor=1.0, min_rate=0.05)
                    X_tr = pipe.process_stream(X_train_s, seed=seed)
                    pipe.reset()
                    X_te = pipe.process_stream(X_test_s, seed=seed + 10000)
                elif method_key == 'var':
                    pipe = VarianceSampling(budget=budget, window_size=50)
                    X_tr = pipe.process_stream(X_train_s, seed=seed)
                    X_te = pipe.process_stream(X_test_s, seed=seed + 10000)
                elif method_key == 'thr':
                    pipe = ThresholdSampling(budget=budget, window_size=50)
                    X_tr = pipe.process_stream(X_train_s, seed=seed)
                    X_te = pipe.process_stream(X_test_s, seed=seed + 10000)
                elif method_key == 'uni':
                    pipe = UniformSampling(budget=budget)
                    X_tr = pipe.process_stream(X_train_s, seed=seed)
                    X_te = pipe.process_stream(X_test_s, seed=seed + 10000)
                elif method_key == 'full':
                    X_tr = X_train_s
                    X_te = X_test_s

                clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                clf.fit(X_tr, y_train)
                f1 = f1_score(y_test, clf.predict(X_te), average='weighted')
                f1s.append(f1)

            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            results.append({
                'fault': fault_num,
                'fault_name': FAULT_NAMES[fault_num],
                'method': method_name,
                'f1_mean': mean_f1,
                'f1_std': std_f1,
            })
            print(f"  {method_name:15s}: F1 = {mean_f1:.4f} +/- {std_f1:.4f}", flush=True)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'per_fault_breakdown.csv'), index=False)
    print(f"\nSaved to {out_dir}/per_fault_breakdown.csv", flush=True)

    # Summary
    print("\n" + "=" * 90, flush=True)
    print(f"{'Fault':<30s}  {'PCA-T':>7s}  {'Var':>7s}  {'Thr':>7s}  {'Uni':>7s}  {'Full':>7s}  {'Best':>10s}", flush=True)
    print("-" * 90, flush=True)

    pca_wins = 0
    for fault_num in fault_types:
        fdf = df[df['fault'] == fault_num]
        if len(fdf) == 0:
            continue
        vals = {}
        for m in ['PCA-Triage', 'Variance', 'Threshold', 'Uniform', 'Full Data']:
            row = fdf[fdf['method'] == m]
            vals[m] = row['f1_mean'].values[0] if len(row) > 0 else float('nan')

        unsup = {k: v for k, v in vals.items() if k != 'Full Data'}
        best_unsup = max(unsup, key=unsup.get)
        if best_unsup == 'PCA-Triage':
            pca_wins += 1

        name = f"IDV({fault_num}): {FAULT_NAMES[fault_num]}"
        print(f"{name:<30s}  {vals['PCA-Triage']:>7.3f}  {vals['Variance']:>7.3f}  "
              f"{vals['Threshold']:>7.3f}  {vals['Uniform']:>7.3f}  {vals['Full Data']:>7.3f}  "
              f"{best_unsup:>10s}", flush=True)

    print(f"\nPCA-Triage best unsupervised on {pca_wins}/{len(fault_types)} fault types", flush=True)


if __name__ == "__main__":
    run_per_fault()
