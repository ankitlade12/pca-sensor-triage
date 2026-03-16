"""
Experiment 4: Ablation Studies.

Systematically varies each hyperparameter (k, w, lambda, score formula)
while holding others at defaults. Produces Table 3 and Figure 5.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.triage import TriagePipeline
from src.utils import load_tep


def run_ablation(config_path: str = None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'ablation.yaml')

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ablation = cfg['ablation']
    defaults = ablation['defaults']
    seeds = ablation['seeds']

    print("Loading TEP dataset...")
    X_train, y_train, X_test, y_test, sensor_cols, _ = load_tep()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    def eval_config(k, w, lam):
        f1s = []
        for seed in seeds:
            pipe = TriagePipeline(
                n_components=k, window_size=w, budget=defaults['budget'],
                forgetting_factor=lam, min_rate=defaults['min_rate'],
            )
            X_tr = pipe.process_stream(X_train, seed=seed)
            X_te = pipe.process_stream(X_test, seed=seed + 10000)
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            clf.fit(X_tr, y_train)
            f1s.append(f1_score(y_test, clf.predict(X_te), average='weighted'))
        return np.mean(f1s), np.std(f1s)

    results = []

    # Ablation 1: k
    print("\nAblation 1: Number of components (k)")
    for k in ablation['k_values']:
        mean, std = eval_config(k=k, w=defaults['window_size'], lam=defaults['forgetting_factor'])
        results.append({'param': 'k', 'value': k, 'f1_mean': mean, 'f1_std': std})
        print(f"  k={k:3d}: F1 = {mean:.4f} ± {std:.4f}")

    # Ablation 2: w
    print("\nAblation 2: Window size (w)")
    for w in ablation['w_values']:
        mean, std = eval_config(k=defaults['n_components'], w=w, lam=defaults['forgetting_factor'])
        results.append({'param': 'w', 'value': w, 'f1_mean': mean, 'f1_std': std})
        print(f"  w={w:3d}: F1 = {mean:.4f} ± {std:.4f}")

    # Ablation 3: lambda
    print("\nAblation 3: Forgetting factor (lambda)")
    for lam in ablation['lambda_values']:
        mean, std = eval_config(k=defaults['n_components'], w=defaults['window_size'], lam=lam)
        results.append({'param': 'lambda', 'value': lam, 'f1_mean': mean, 'f1_std': std})
        print(f"  λ={lam:.2f}: F1 = {mean:.4f} ± {std:.4f}")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, 'ablation_results.csv'), index=False)
    print(f"\nSaved to {out_dir}/ablation_results.csv")


if __name__ == "__main__":
    run_ablation()
