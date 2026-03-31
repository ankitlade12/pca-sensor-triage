"""
Experiment 8: Controlled Correlation Validation.

Validates Theorem 1: PCA-Triage's advantage over variance-based allocation
increases with inter-channel correlation, because PCA detects redundancy.

Design rationale (mirrors real-world TEP scenario):
- d=40 channels total
- Group A (10 channels): informative + highly correlated with each other
  → PCA detects redundancy, down-weights duplicates, frees bandwidth
  → Variance treats each as equally important (wastes bandwidth)
- Group B (10 channels): informative + independent
  → Both methods recognize these; PCA can allocate MORE budget here
  because it freed budget from Group A
- Group C (20 channels): noise (independent, uninformative)
  → Both methods should down-weight these

We vary rho (within Group A correlation) from 0 to 0.95 and measure
PCA-Triage vs Variance. At rho=0, PCA ≈ Variance (no redundancy).
At high rho, PCA should win by concentrating Group A into fewer
effective channels and giving more budget to Group B.
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


def generate_redundancy_data(n_samples, rho, seed=42):
    """Generate data where PCA's redundancy detection matters.

    Group A (ch 0-9):  Informative, within-group correlation = rho
    Group B (ch 10-19): Informative, independent
    Group C (ch 20-39): Pure noise, independent

    Fault signal: Group A and B channels shift mean by +1.5 under fault.
    Group C is always noise.
    """
    rng = np.random.RandomState(seed)
    d = 40
    n_A, n_B, n_C = 10, 10, 20

    # Build covariance: Group A correlated, rest independent
    cov = np.eye(d)
    for i in range(n_A):
        for j in range(n_A):
            if i != j:
                cov[i, j] = rho

    # Class 0: normal operation
    n0 = n_samples // 2
    X0 = rng.multivariate_normal(np.zeros(d), cov, size=n0)

    # Class 1: fault — shift Group A and Group B means
    n1 = n_samples - n0
    mean1 = np.zeros(d)
    mean1[:n_A] = 1.5       # Group A: informative (but correlated)
    mean1[n_A:n_A+n_B] = 1.5  # Group B: informative (independent)
    # Group C: no shift (noise)
    X1 = rng.multivariate_normal(mean1, cov, size=n1)

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    # Shuffle
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    # Split
    split = int(0.7 * len(y))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split])
    X_test = scaler.transform(X[split:])

    return X_train, y[:split], X_test, y[split:]


def run_correlation_validation():
    print("=" * 70)
    print("CONTROLLED CORRELATION VALIDATION (Theorem 1)")
    print("=" * 70)
    print("Design: 40 channels = 10 correlated-informative (A) +")
    print("        10 independent-informative (B) + 20 noise (C)")
    print("Varying: within-Group-A correlation rho")

    rho_values = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    seeds = [42, 123, 456, 789, 1024]
    budget = 0.5
    n_samples = 6000

    results = []

    for rho in rho_values:
        print(f"\n--- rho = {rho:.2f} ---")

        method_configs = [
            ('PCA-Triage', 'pca'),
            ('Variance', 'var'),
            ('Threshold', 'thr'),
            ('Uniform', 'uni'),
        ]

        for method_name, method_key in method_configs:
            f1s = []
            for seed in seeds:
                X_train, y_train, X_test, y_test = generate_redundancy_data(
                    n_samples=n_samples, rho=rho, seed=seed)

                if method_key == 'pca':
                    pipe = TriagePipeline(n_components=10, window_size=50,
                                         budget=budget, forgetting_factor=1.0,
                                         min_rate=0.05)
                    X_tr = pipe.process_stream(X_train, seed=seed)
                    pipe.reset()
                    X_te = pipe.process_stream(X_test, seed=seed + 10000)
                elif method_key == 'var':
                    pipe = VarianceSampling(budget=budget, window_size=50)
                    X_tr = pipe.process_stream(X_train, seed=seed)
                    X_te = pipe.process_stream(X_test, seed=seed + 10000)
                elif method_key == 'thr':
                    pipe = ThresholdSampling(budget=budget, window_size=50)
                    X_tr = pipe.process_stream(X_train, seed=seed)
                    X_te = pipe.process_stream(X_test, seed=seed + 10000)
                elif method_key == 'uni':
                    pipe = UniformSampling(budget=budget)
                    X_tr = pipe.process_stream(X_train, seed=seed)
                    X_te = pipe.process_stream(X_test, seed=seed + 10000)

                clf = RandomForestClassifier(n_estimators=100, random_state=seed,
                                             n_jobs=-1)
                clf.fit(X_tr, y_train)
                f1 = f1_score(y_test, clf.predict(X_te), average='weighted')
                f1s.append(f1)

            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            results.append({
                'rho': rho,
                'method': method_name,
                'f1_mean': mean_f1,
                'f1_std': std_f1,
                'n_seeds': len(seeds),
            })
            print(f"  {method_name:15s}: F1 = {mean_f1:.4f} +/- {std_f1:.4f}")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'correlation_validation.csv'), index=False)
    print(f"\nSaved to {out_dir}/correlation_validation.csv")

    # Summary table
    print("\n" + "=" * 70)
    print("THEOREM 1 VALIDATION: PCA advantage increases with correlation")
    print("=" * 70)
    print(f"{'rho':>5s}  {'PCA-Triage':>12s}  {'Variance':>12s}  {'PCA-Var':>8s}  {'Uniform':>12s}  {'PCA-Uni':>8s}")
    print("-" * 65)

    for rho in rho_values:
        pca = df[(df['rho'] == rho) & (df['method'] == 'PCA-Triage')]['f1_mean'].values[0]
        var = df[(df['rho'] == rho) & (df['method'] == 'Variance')]['f1_mean'].values[0]
        uni = df[(df['rho'] == rho) & (df['method'] == 'Uniform')]['f1_mean'].values[0]
        print(f"{rho:5.2f}  {pca:12.4f}  {var:12.4f}  {pca-var:+8.4f}  {uni:12.4f}  {pca-uni:+8.4f}")


if __name__ == "__main__":
    run_correlation_validation()
