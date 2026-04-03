"""
Experiment 15: Real-Time Deployment Simulation.

Simulates non-ideal edge deployment conditions:
1. Latency jitter — random delays on window arrival
2. Packet loss — entire windows dropped at random
3. Sensor noise — Gaussian noise added to channels
4. Clock drift — channels arrive at slightly different rates

Tests whether PCA-Triage degrades gracefully under these conditions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from src.triage.pipeline import TriagePipeline
from src.baselines import UniformSampling, VarianceSampling
from src.utils.data_loader import get_dataset

from run_pareto_v2 import DATASET_CONFIGS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def apply_latency_jitter(data, max_delay=5, seed=42):
    """Randomly shift rows by 0-max_delay positions (simulates arrival jitter)."""
    rng = np.random.RandomState(seed)
    result = data.copy()
    n, d = data.shape
    for j in range(d):
        delay = rng.randint(0, max_delay + 1)
        if delay > 0:
            result[delay:, j] = data[:-delay, j]
            result[:delay, j] = data[0, j]  # repeat first value
    return result


def apply_packet_loss(data, loss_rate=0.05, window_size=50, seed=42):
    """Drop entire windows at random (simulates network packet loss)."""
    rng = np.random.RandomState(seed)
    result = data.copy()
    n_windows = len(data) // window_size
    for w in range(n_windows):
        if rng.random() < loss_rate:
            start = w * window_size
            end = start + window_size
            # Replace lost window with previous window (or zeros if first)
            if w > 0:
                prev_start = (w - 1) * window_size
                result[start:end] = result[prev_start:prev_start + window_size]
            else:
                result[start:end] = 0.0
    return result


def apply_sensor_noise(data, noise_std=0.1, seed=42):
    """Add Gaussian noise to simulate sensor imprecision."""
    rng = np.random.RandomState(seed)
    return data + rng.randn(*data.shape) * noise_std


def apply_clock_drift(data, max_drift=3, seed=42):
    """Simulate per-channel clock drift (channels arrive at different rates)."""
    rng = np.random.RandomState(seed)
    result = data.copy()
    n, d = data.shape
    for j in range(d):
        drift = rng.randint(-max_drift, max_drift + 1)
        if drift > 0:
            result[drift:, j] = data[:-drift, j]
            result[:drift, j] = data[0, j]
        elif drift < 0:
            result[:drift, j] = data[-drift:, j]
            result[drift:, j] = data[-1, j]
    return result


def apply_combined(data, seed=42):
    """Apply all perturbations simultaneously (worst case)."""
    data = apply_latency_jitter(data, max_delay=3, seed=seed)
    data = apply_packet_loss(data, loss_rate=0.05, window_size=50, seed=seed + 1)
    data = apply_sensor_noise(data, noise_std=0.05, seed=seed + 2)
    data = apply_clock_drift(data, max_drift=2, seed=seed + 3)
    return data


PERTURBATIONS = {
    'Clean (baseline)': lambda x, s: x,
    'Jitter (±5 samples)': lambda x, s: apply_latency_jitter(x, max_delay=5, seed=s),
    'Packet loss (5%)': lambda x, s: apply_packet_loss(x, loss_rate=0.05, seed=s),
    'Packet loss (10%)': lambda x, s: apply_packet_loss(x, loss_rate=0.10, seed=s),
    'Sensor noise (σ=0.1)': lambda x, s: apply_sensor_noise(x, noise_std=0.1, seed=s),
    'Sensor noise (σ=0.3)': lambda x, s: apply_sensor_noise(x, noise_std=0.3, seed=s),
    'Clock drift (±3)': lambda x, s: apply_clock_drift(x, max_drift=3, seed=s),
    'Combined (all)': lambda x, s: apply_combined(x, seed=s),
}


def run_experiment():
    print("=" * 70, flush=True)
    print("EXPERIMENT 15: Real-Time Deployment Simulation", flush=True)
    print("=" * 70, flush=True)

    # Use TEP as primary dataset
    datasets_to_test = ['tep', 'smd', 'psm']
    seeds = [42, 123, 456]
    all_rows = []

    for ds_name in datasets_to_test:
        print(f"\n--- {ds_name.upper()} ---", flush=True)
        result = get_dataset(ds_name)
        X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
        le = LabelEncoder()
        y_tr, y_te = le.fit_transform(y_train), le.transform(y_test)

        cfg = DATASET_CONFIGS.get(ds_name, DATASET_CONFIGS['tep'])

        for perturb_name, perturb_fn in PERTURBATIONS.items():
            f1s_pca, f1s_var, f1s_uni = [], [], []

            for seed in seeds:
                # Apply perturbation to test data only (train is "clean historical")
                X_test_perturbed = perturb_fn(X_test.copy(), seed)

                # PCA-Triage
                pipe = TriagePipeline(
                    n_components=cfg['n_components'], window_size=cfg['window_size'],
                    budget=0.5, forgetting_factor=cfg['forgetting_factor'],
                    min_rate=cfg['min_rate'], reconstruction_method=cfg['reconstruction_method'],
                    scorer=cfg['scorer'], alpha=cfg['alpha'],
                    sharpness=cfg.get('sharpness', 1.0))
                X_tr = pipe.process_stream(X_train, seed=seed)

                pipe2 = TriagePipeline(
                    n_components=cfg['n_components'], window_size=cfg['window_size'],
                    budget=0.5, forgetting_factor=cfg['forgetting_factor'],
                    min_rate=cfg['min_rate'], reconstruction_method=cfg['reconstruction_method'],
                    scorer=cfg['scorer'], alpha=cfg['alpha'],
                    sharpness=cfg.get('sharpness', 1.0))
                X_te = pipe2.process_stream(X_test_perturbed, seed=seed + 10000)

                clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
                clf.fit(X_tr, y_tr)
                f1s_pca.append(f1_score(y_te, clf.predict(X_te), average='weighted'))

                # Variance baseline
                var = VarianceSampling(budget=0.5, window_size=50)
                X_tr_v = var.process_stream(X_train, seed=seed)
                var2 = VarianceSampling(budget=0.5, window_size=50)
                X_te_v = var2.process_stream(X_test_perturbed, seed=seed + 10000)
                clf2 = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
                clf2.fit(X_tr_v, y_tr)
                f1s_var.append(f1_score(y_te, clf2.predict(X_te_v), average='weighted'))

                # Uniform baseline
                uni = UniformSampling(budget=0.5)
                X_tr_u = uni.process_stream(X_train, seed=seed)
                uni2 = UniformSampling(budget=0.5)
                X_te_u = uni2.process_stream(X_test_perturbed, seed=seed + 10000)
                clf3 = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
                clf3.fit(X_tr_u, y_tr)
                f1s_uni.append(f1_score(y_te, clf3.predict(X_te_u), average='weighted'))

            pca_mean = np.mean(f1s_pca)
            var_mean = np.mean(f1s_var)
            uni_mean = np.mean(f1s_uni)

            all_rows.append({
                'dataset': ds_name, 'perturbation': perturb_name,
                'pca_f1': pca_mean, 'pca_std': np.std(f1s_pca),
                'var_f1': var_mean, 'var_std': np.std(f1s_var),
                'uni_f1': uni_mean, 'uni_std': np.std(f1s_uni),
                'pca_wins': pca_mean > max(var_mean, uni_mean),
            })

            winner = "PCA" if pca_mean > max(var_mean, uni_mean) else ("Var" if var_mean > uni_mean else "Uni")
            print(f"  {perturb_name:25s}: PCA={pca_mean:.4f} Var={var_mean:.4f} Uni={uni_mean:.4f} → {winner}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'realtime_simulation.csv'), index=False)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("ROBUSTNESS SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    pca_win_rate = df['pca_wins'].mean()
    print(f"PCA-Triage wins {df['pca_wins'].sum()}/{len(df)} conditions ({pca_win_rate:.0%})", flush=True)

    # Degradation from clean
    for ds in datasets_to_test:
        sub = df[df['dataset'] == ds]
        clean = sub[sub['perturbation'] == 'Clean (baseline)']['pca_f1'].values[0]
        combined = sub[sub['perturbation'] == 'Combined (all)']['pca_f1'].values[0]
        degradation = (clean - combined) / clean * 100
        print(f"  {ds.upper()}: Clean={clean:.4f} → Combined={combined:.4f} (degradation: {degradation:.1f}%)", flush=True)

    print(f"\nSaved to {RESULTS_DIR}/realtime_simulation.csv", flush=True)


if __name__ == "__main__":
    run_experiment()
