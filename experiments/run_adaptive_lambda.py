"""
Experiment 12: Adaptive Lambda (Time-Varying Forgetting Factor).

Tests an adaptive forgetting factor that:
- Increases λ during stable periods (favoring accuracy)
- Decreases λ after detected importance shifts (favoring adaptivity)

This aims to close the accuracy-adaptivity gap between λ=1.0 (best F1)
and λ=0.85 (fastest reaction).

Design:
- Detect shift via L2 distance between consecutive importance vectors
- If shift > threshold: decrease λ (fast adaptation)
- If shift < threshold: increase λ (stability)
- λ bounded in [λ_min, λ_max]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.triage.rate_allocator import RateAllocator
from src.triage.reconstruction import reconstruct
from src.utils import load_tep


class AdaptiveLambdaPCATriage:
    """PCA-Triage with adaptive forgetting factor.

    λ adapts based on importance shift magnitude:
    - Large shift → decrease λ (fast adaptation)
    - Small shift → increase λ (stability)
    """

    def __init__(
        self,
        n_components=10,
        window_size=50,
        lambda_min=0.80,
        lambda_max=0.99,
        lambda_init=0.95,
        shift_threshold=0.05,
        lambda_increase=0.02,
        lambda_decrease=0.10,
        min_rate=0.05,
    ):
        self.n_components = n_components
        self.window_size = window_size
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lam = lambda_init
        self.shift_threshold = shift_threshold
        self.lambda_increase = lambda_increase
        self.lambda_decrease = lambda_decrease
        self.min_rate = min_rate

        self.ipca = IncrementalPCA(n_components=n_components)
        self._fitted = False
        self._importance_history = None
        self._prev_raw_scores = None
        self.lambda_log = []

    def compute_importance(self, window):
        w, d = window.shape

        if not self._fitted:
            self.ipca.fit(window)
            self._fitted = True
        else:
            self.ipca.partial_fit(window)

        V = self.ipca.components_
        sigma = self.ipca.singular_values_

        scores = np.zeros(d)
        for i in range(len(sigma)):
            scores += sigma[i] * V[i, :] ** 2
        scores = scores / scores.sum()

        # Detect shift and adapt λ
        if self._prev_raw_scores is not None:
            shift = np.linalg.norm(scores - self._prev_raw_scores)
            if shift > self.shift_threshold:
                self.lam = max(self.lambda_min, self.lam - self.lambda_decrease)
            else:
                self.lam = min(self.lambda_max, self.lam + self.lambda_increase)
        self._prev_raw_scores = scores.copy()
        self.lambda_log.append(self.lam)

        # Smooth
        if self._importance_history is None:
            self._importance_history = scores
        else:
            self._importance_history = self.lam * self._importance_history + (1 - self.lam) * scores

        smoothed = self._importance_history / self._importance_history.sum()
        return smoothed

    def reset(self):
        self.ipca = IncrementalPCA(n_components=self.n_components)
        self._fitted = False
        self._importance_history = None
        self._prev_raw_scores = None
        self.lam = (self.lambda_min + self.lambda_max) / 2
        self.lambda_log = []


def process_with_adaptive(X, budget, seed, **kwargs):
    """Process data with adaptive lambda PCA-Triage."""
    triage = AdaptiveLambdaPCATriage(**kwargs)
    allocator = RateAllocator(budget=budget, min_rate=kwargs.get('min_rate', 0.05))
    n, d = X.shape
    ws = kwargs.get('window_size', 50)
    n_windows = n // ws
    recon = np.zeros_like(X, dtype=float)

    for w_idx in range(n_windows):
        start = w_idx * ws
        end = start + ws
        window = X[start:end]
        importance = triage.compute_importance(window)
        rates = allocator.allocate(importance)
        triaged = allocator.apply_rates(window, rates, seed=seed + w_idx)
        recon[start:end] = reconstruct(triaged, method="forward_fill")

    remaining = n % ws
    if remaining > 0:
        start = n_windows * ws
        rates_last = np.full(d, budget)
        triaged = allocator.apply_rates(X[start:], rates_last, seed=seed + n_windows)
        recon[start:] = reconstruct(triaged, method="forward_fill")

    return recon, triage


def run_adaptive_lambda():
    print("=" * 70, flush=True)
    print("ADAPTIVE LAMBDA EXPERIMENT", flush=True)
    print("=" * 70, flush=True)

    print("Loading TEP...", flush=True)
    X_train, y_train, X_test, y_test, _, _ = load_tep()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}", flush=True)

    seeds = [42, 123, 456, 789, 1024]
    budget = 0.5
    results = []

    configs = [
        ("Fixed λ=1.0 (best accuracy)", dict(n_components=10, window_size=50, min_rate=0.05),
         "fixed_1.0"),
        ("Fixed λ=0.85 (fast reaction)", dict(n_components=10, window_size=50, min_rate=0.05),
         "fixed_0.85"),
        ("Adaptive λ ∈ [0.80, 0.99]", dict(
            n_components=10, window_size=50, min_rate=0.05,
            lambda_min=0.80, lambda_max=0.99, lambda_init=0.95,
            shift_threshold=0.05, lambda_increase=0.02, lambda_decrease=0.10),
         "adaptive"),
        ("Adaptive λ ∈ [0.85, 0.99] (conservative)", dict(
            n_components=10, window_size=50, min_rate=0.05,
            lambda_min=0.85, lambda_max=0.99, lambda_init=0.95,
            shift_threshold=0.03, lambda_increase=0.01, lambda_decrease=0.08),
         "adaptive_conservative"),
    ]

    for name, kwargs, key in configs:
        print(f"\n--- {name} ---", flush=True)
        f1s = []
        for seed in seeds:
            if key == "fixed_1.0":
                from src.triage import TriagePipeline
                pipe = TriagePipeline(n_components=10, window_size=50, budget=budget,
                                     forgetting_factor=1.0, min_rate=0.05)
                X_tr = pipe.process_stream(X_train, seed=seed)
                pipe.reset()
                X_te = pipe.process_stream(X_test, seed=seed + 10000)
            elif key == "fixed_0.85":
                from src.triage import TriagePipeline
                pipe = TriagePipeline(n_components=10, window_size=50, budget=budget,
                                     forgetting_factor=0.85, min_rate=0.05)
                X_tr = pipe.process_stream(X_train, seed=seed)
                pipe.reset()
                X_te = pipe.process_stream(X_test, seed=seed + 10000)
            else:
                X_tr, triage_tr = process_with_adaptive(X_train, budget, seed, **kwargs)
                X_te, triage_te = process_with_adaptive(X_test, budget, seed + 10000, **kwargs)
                if seed == 42:
                    lam_log = triage_tr.lambda_log
                    print(f"    λ range: [{min(lam_log):.3f}, {max(lam_log):.3f}], "
                          f"mean={np.mean(lam_log):.3f}, final={lam_log[-1]:.3f}", flush=True)

            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            clf.fit(X_tr, y_train)
            f1 = f1_score(y_test, clf.predict(X_te), average='weighted')
            f1s.append(f1)

        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        results.append({
            'config': name,
            'key': key,
            'f1_mean': mean_f1,
            'f1_std': std_f1,
        })
        print(f"  F1 = {mean_f1:.4f} ± {std_f1:.4f}", flush=True)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'adaptive_lambda.csv'), index=False)
    print(f"\nSaved to {out_dir}/adaptive_lambda.csv", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        print(f"  {r['config']:45s}: F1 = {r['f1_mean']:.4f} ± {r['f1_std']:.4f}", flush=True)


if __name__ == "__main__":
    run_adaptive_lambda()
