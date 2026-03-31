"""
Experiment 10: Autoencoder Baseline Comparison.

Evaluates autoencoder-based channel importance against PCA-Triage
and other baselines on TEP, SMD, and SKAB at 50% bandwidth.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.triage import TriagePipeline
from src.baselines import (
    UniformSampling, VarianceSampling, AutoencoderSampling
)
from src.utils import load_tep


def run_autoencoder_baseline():
    print("=" * 70, flush=True)
    print("AUTOENCODER BASELINE COMPARISON", flush=True)
    print("=" * 70, flush=True)

    seeds = [42, 123, 456]
    budget = 0.5
    results = []

    # --- TEP ---
    print("\nLoading TEP...", flush=True)
    X_train, y_train, X_test, y_test, _, _ = load_tep()
    print(f"TEP: Train {X_train.shape}, Test {X_test.shape}", flush=True)

    methods = {
        'PCA-Triage': lambda: TriagePipeline(
            n_components=10, window_size=50, budget=budget,
            forgetting_factor=1.0, min_rate=0.05),
        'Autoencoder': lambda: AutoencoderSampling(
            budget=budget, window_size=50, bottleneck=10,
            n_epochs=50, lr=0.01, min_rate=0.05),
        'Variance': lambda: VarianceSampling(budget=budget, window_size=50),
        'Uniform': lambda: UniformSampling(budget=budget),
    }

    for ds_name, X_tr_raw, y_tr, X_te_raw, y_te in [
        ('TEP', X_train, y_train, X_test, y_test),
    ]:
        print(f"\n--- {ds_name} ---", flush=True)
        for method_name, method_fn in methods.items():
            f1s = []
            for seed in seeds:
                pipe = method_fn()
                if method_name == 'PCA-Triage':
                    X_tr = pipe.process_stream(X_tr_raw, seed=seed)
                    pipe.reset()
                    X_te = pipe.process_stream(X_te_raw, seed=seed + 10000)
                else:
                    X_tr = pipe.process_stream(X_tr_raw, seed=seed)
                    X_te = pipe.process_stream(X_te_raw, seed=seed + 10000)

                clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                clf.fit(X_tr, y_tr)
                f1 = f1_score(y_te, clf.predict(X_te), average='weighted')
                f1s.append(f1)

            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            results.append({
                'dataset': ds_name,
                'method': method_name,
                'f1_mean': mean_f1,
                'f1_std': std_f1,
            })
            print(f"  {method_name:15s}: F1 = {mean_f1:.4f} +/- {std_f1:.4f}", flush=True)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'autoencoder_baseline.csv'), index=False)
    print(f"\nSaved to {out_dir}/autoencoder_baseline.csv", flush=True)


if __name__ == "__main__":
    run_autoencoder_baseline()
