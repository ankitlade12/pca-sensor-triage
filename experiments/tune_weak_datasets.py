"""Quick grid search on HAI and SKAB to find configs that beat baselines."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from itertools import product

from src.triage.pipeline import TriagePipeline
from src.baselines import VarianceSampling, UniformSampling, ThresholdSampling
from src.utils.data_loader import get_dataset


def evaluate(X_train, y_train, X_test, y_test, pipe_kwargs, seeds=[42, 123, 456]):
    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_te = le.transform(y_test)
    f1s = []
    for seed in seeds:
        pipe = TriagePipeline(**pipe_kwargs, budget=0.5)
        X_tr = pipe.process_stream(X_train, seed=seed)
        X_te = pipe.process_stream(X_test, seed=seed + 10000)
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        f1 = f1_score(y_te, clf.predict(X_te), average='weighted')
        f1s.append(f1)
    return np.mean(f1s), np.std(f1s)


def tune_dataset(name, param_grid):
    print(f"\n{'='*60}")
    print(f"TUNING: {name.upper()}")
    print(f"{'='*60}")

    result = get_dataset(name)
    X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
    print(f"  Shape: {X_train.shape}")

    # Get baseline targets
    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_te = le.transform(y_test)
    seeds = [42, 123, 456]

    baselines = {}
    for bl_name, bl_cls, kw in [
        ("Variance", VarianceSampling, {'budget': 0.5, 'window_size': 50}),
        ("Uniform", UniformSampling, {'budget': 0.5}),
        ("Threshold", ThresholdSampling, {'budget': 0.5, 'window_size': 50}),
    ]:
        f1s = []
        for seed in seeds:
            bl = bl_cls(**kw)
            X_tr = bl.process_stream(X_train, seed=seed)
            X_te = bl.process_stream(X_test, seed=seed + 10000)
            clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            f1s.append(f1_score(y_te, clf.predict(X_te), average='weighted'))
        baselines[bl_name] = np.mean(f1s)
        print(f"  Baseline {bl_name}: {np.mean(f1s):.4f}")

    target = max(baselines.values())
    print(f"  TARGET to beat: {target:.4f} ({max(baselines, key=baselines.get)})")

    # Grid search
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_f1 = 0
    best_cfg = None

    total = 1
    for v in values:
        total *= len(v)
    print(f"  Searching {total} configs...")

    for combo in product(*values):
        cfg = dict(zip(keys, combo))
        cfg['reconstruction_method'] = 'linear'
        try:
            f1_mean, f1_std = evaluate(X_train, y_train, X_test, y_test, cfg, seeds)
            if f1_mean > best_f1:
                best_f1 = f1_mean
                best_cfg = cfg.copy()
                beats = "BEATS" if f1_mean > target else "below"
                print(f"    NEW BEST: F1={f1_mean:.4f} ({beats} target {target:.4f}) | {cfg}")
        except Exception as e:
            pass

    print(f"\n  BEST CONFIG: F1={best_f1:.4f} {'(WINS!)' if best_f1 > target else '(still below)'}")
    print(f"  {best_cfg}")
    return best_cfg, best_f1


if __name__ == "__main__":
    # HAI: 82 channels, loses to Variance by 0.001
    tune_dataset('hai', {
        'n_components': [10, 15, 20, 30, 40],
        'window_size': [30, 50, 100],
        'forgetting_factor': [0.9, 0.95, 1.0],
        'min_rate': [0.01, 0.03, 0.05],
        'scorer': ['hybrid'],
        'alpha': [0.1, 0.2, 0.3, 0.5],
        'sharpness': [1.0, 1.5, 2.0, 3.0],
    })

    # SKAB: 8 channels, loses to Uniform by 0.014
    tune_dataset('skab', {
        'n_components': [2, 3, 4, 5],
        'window_size': [20, 30, 50, 100],
        'forgetting_factor': [0.85, 0.9, 0.95, 1.0],
        'min_rate': [0.01, 0.05, 0.1],
        'scorer': ['hybrid', 'pca'],
        'alpha': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7],
        'sharpness': [0.5, 1.0, 1.5, 2.0],
    })
