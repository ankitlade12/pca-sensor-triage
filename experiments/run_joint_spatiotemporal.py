"""
Experiment 17: Joint Spatial-Temporal Optimization.

Combines PCA-Triage (spatial: which channels) with Send-on-Delta
(temporal: which samples within each channel). Tests whether the
combination achieves better bandwidth savings than either alone.

Also implements OGD (Online Gradient Descent) regret-optimal baseline
for comparison (Phase 3.2).
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
from src.baselines.send_on_delta import SendOnDelta, JointSpatialTemporal
from src.baselines import UniformSampling, VarianceSampling
from src.triage.rate_allocator import RateAllocator
from src.utils.data_loader import get_dataset

from run_pareto_v2 import DATASET_CONFIGS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


class OGDAllocator:
    """Online Gradient Descent rate allocation (regret-optimal baseline).

    Uses OGD to minimize reconstruction error online, serving as an
    upper bound for what an online algorithm can achieve.

    Parameters
    ----------
    budget : float
        Total bandwidth budget.
    lr : float
        Learning rate for gradient updates.
    min_rate : float
        Minimum per-channel rate.
    """

    def __init__(self, budget=0.5, lr=0.01, min_rate=0.05):
        self.budget = budget
        self.lr = lr
        self.min_rate = min_rate
        self._rates = None

    def process_stream(self, data, seed=42):
        n, d = data.shape
        allocator = RateAllocator(budget=self.budget, min_rate=self.min_rate)
        window_size = 50
        n_windows = n // window_size
        result = np.zeros_like(data, dtype=float)

        # Initialize uniform
        if self._rates is None:
            self._rates = np.full(d, self.budget)

        rng = np.random.RandomState(seed)

        for w_idx in range(n_windows):
            start = w_idx * window_size
            end = start + window_size
            window = data[start:end]

            # Apply current rates
            mask = rng.random((window_size, d)) < self._rates[np.newaxis, :]
            triaged = window.copy().astype(float)
            triaged[~mask] = np.nan
            recon = pd.DataFrame(triaged).interpolate(method='linear', axis=0,
                                                       limit_direction='both').fillna(0.0).values
            result[start:end] = recon

            # Compute per-channel reconstruction error (gradient signal)
            recon_error = np.mean((window - recon) ** 2, axis=0)  # (d,)

            # OGD update: increase rates for high-error channels
            gradient = recon_error / (recon_error.sum() + 1e-10)
            self._rates = self._rates + self.lr * gradient

            # Project onto budget simplex
            self._rates = np.clip(self._rates, self.min_rate, 1.0)
            excess = self._rates.mean() - self.budget
            if excess > 0:
                self._rates = self._rates - excess

            self._rates = np.clip(self._rates, self.min_rate, 1.0)

        # Handle tail
        remaining = n % window_size
        if remaining > 0:
            start = n_windows * window_size
            mask = rng.random((remaining, d)) < self._rates[np.newaxis, :]
            triaged = data[start:].copy().astype(float)
            triaged[~mask] = np.nan
            result[start:] = pd.DataFrame(triaged).interpolate(
                method='linear', axis=0, limit_direction='both').fillna(0.0).values

        return result


def run_experiment():
    print("=" * 70, flush=True)
    print("EXPERIMENT 17: Joint Spatial-Temporal + OGD Baseline", flush=True)
    print("=" * 70, flush=True)

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
        n_ch = X_train.shape[1]

        methods = {
            'PCA-Triage (spatial only)': lambda X, s: TriagePipeline(
                n_components=cfg['n_components'], window_size=cfg['window_size'],
                budget=0.5, forgetting_factor=cfg['forgetting_factor'],
                min_rate=cfg['min_rate'], reconstruction_method='linear',
                scorer=cfg['scorer'], alpha=cfg['alpha'],
                sharpness=cfg.get('sharpness', 1.0)).process_stream(X, seed=s),

            'Send-on-Delta (temporal only, δ=0.1)': lambda X, s: SendOnDelta(
                delta=0.1).process_stream(X, seed=s),

            'Send-on-Delta (temporal only, δ=0.3)': lambda X, s: SendOnDelta(
                delta=0.3).process_stream(X, seed=s),

            'Joint PCA+SoD (δ=0.1)': lambda X, s: JointSpatialTemporal(
                spatial_budget=0.7, delta=0.1,
                n_components=min(cfg['n_components'], n_ch - 1),
                window_size=cfg['window_size']).process_stream(X, seed=s),

            'Joint PCA+SoD (δ=0.3)': lambda X, s: JointSpatialTemporal(
                spatial_budget=0.7, delta=0.3,
                n_components=min(cfg['n_components'], n_ch - 1),
                window_size=cfg['window_size']).process_stream(X, seed=s),

            'OGD (regret-optimal)': lambda X, s: OGDAllocator(
                budget=0.5, lr=0.01).process_stream(X, seed=s),

            'Variance': lambda X, s: VarianceSampling(
                budget=0.5, window_size=50).process_stream(X, seed=s),

            'Uniform': lambda X, s: UniformSampling(
                budget=0.5).process_stream(X, seed=s),
        }

        for method_name, method_fn in methods.items():
            f1s = []
            for seed in seeds:
                try:
                    X_tr = method_fn(X_train, seed)
                    X_te = method_fn(X_test, seed + 10000)
                    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
                    clf.fit(X_tr, y_tr)
                    f1 = f1_score(y_te, clf.predict(X_te), average='weighted')
                    f1s.append(f1)
                except Exception as e:
                    print(f"    {method_name}: seed {seed} ERROR: {e}", flush=True)

            if f1s:
                mean_f1 = np.mean(f1s)
                std_f1 = np.std(f1s)
                all_rows.append({
                    'dataset': ds_name, 'method': method_name,
                    'f1_mean': mean_f1, 'f1_std': std_f1,
                })
                marker = " ***" if 'PCA' in method_name and 'Joint' not in method_name else ""
                print(f"  {method_name:40s}: F1={mean_f1:.4f} ± {std_f1:.4f}{marker}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'joint_spatiotemporal.csv'), index=False)

    print(f"\n{'='*70}", flush=True)
    print("KEY FINDINGS", flush=True)
    print(f"{'='*70}", flush=True)
    for ds in datasets_to_test:
        sub = df[df['dataset'] == ds].sort_values('f1_mean', ascending=False)
        print(f"\n  {ds.upper()}:", flush=True)
        for _, row in sub.iterrows():
            print(f"    {row['method']:40s}: {row['f1_mean']:.4f}", flush=True)

    print(f"\nSaved to {RESULTS_DIR}/joint_spatiotemporal.csv", flush=True)


if __name__ == "__main__":
    run_experiment()
