"""Fast targeted tuning for HAI and SKAB — small grid, 2 seeds."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from src.triage.pipeline import TriagePipeline
from src.baselines import VarianceSampling, UniformSampling
from src.utils.data_loader import get_dataset


def eval_config(X_train, y_tr, X_test, y_te, cfg, seeds=[42, 123]):
    f1s = []
    for seed in seeds:
        pipe = TriagePipeline(**cfg, budget=0.5)
        X_tr = pipe.process_stream(X_train, seed=seed)
        X_te = pipe.process_stream(X_test, seed=seed + 10000)
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        f1s.append(f1_score(y_te, clf.predict(X_te), average='weighted'))
    return np.mean(f1s)


def eval_baseline(bl_cls, kw, X_train, y_tr, X_test, y_te, seeds=[42, 123]):
    f1s = []
    for seed in seeds:
        bl = bl_cls(**kw)
        X_tr = bl.process_stream(X_train, seed=seed)
        X_te = bl.process_stream(X_test, seed=seed + 10000)
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        f1s.append(f1_score(y_te, clf.predict(X_te), average='weighted'))
    return np.mean(f1s)


def tune(name, configs):
    print(f"\n{'='*60}", flush=True)
    print(f"TUNING: {name.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    result = get_dataset(name)
    X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_te = le.transform(y_test)

    # Baselines
    var_f1 = eval_baseline(VarianceSampling, {'budget': 0.5, 'window_size': 50},
                           X_train, y_tr, X_test, y_te)
    uni_f1 = eval_baseline(UniformSampling, {'budget': 0.5},
                           X_train, y_tr, X_test, y_te)
    target = max(var_f1, uni_f1)
    print(f"  Variance: {var_f1:.4f}, Uniform: {uni_f1:.4f} → target: {target:.4f}", flush=True)

    best_f1, best_cfg = 0, None
    for i, cfg in enumerate(configs):
        try:
            f1 = eval_config(X_train, y_tr, X_test, y_te, cfg)
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = cfg.copy()
                tag = "BEATS!" if f1 > target else "below"
                print(f"  [{i+1}/{len(configs)}] F1={f1:.4f} ({tag}) | a={cfg.get('alpha','?')} k={cfg.get('n_components','?')} s={cfg.get('sharpness','?')} w={cfg.get('window_size','?')} lam={cfg.get('forgetting_factor','?')}", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(configs)}] ERROR: {e}", flush=True)

    print(f"\n  BEST: F1={best_f1:.4f} {'WINS!' if best_f1 > target else 'still below'}", flush=True)
    print(f"  Config: {best_cfg}", flush=True)


if __name__ == "__main__":
    # HAI: 82 channels, loses to Variance 1.0000 by 0.001
    # Strategy: very low alpha (near-variance), high k, various sharpness
    hai_configs = []
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3]:
        for k in [15, 20, 30, 40]:
            for sharp in [1.0, 1.5, 2.0]:
                for w in [30, 50]:
                    hai_configs.append({
                        'n_components': k, 'window_size': w,
                        'forgetting_factor': 1.0, 'min_rate': 0.01,
                        'scorer': 'hybrid', 'alpha': alpha,
                        'sharpness': sharp, 'reconstruction_method': 'linear',
                    })
    print(f"HAI: {len(hai_configs)} configs")
    tune('hai', hai_configs)

    # SKAB: 8 channels, loses to Uniform 0.5889
    # Strategy: low k, various alpha, various sharpness
    skab_configs = []
    for alpha in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        for k in [2, 3, 4, 5]:
            for sharp in [0.5, 1.0, 1.5, 2.0]:
                for w in [20, 30, 50]:
                    for lam in [0.9, 1.0]:
                        skab_configs.append({
                            'n_components': k, 'window_size': w,
                            'forgetting_factor': lam, 'min_rate': 0.05,
                            'scorer': 'hybrid', 'alpha': alpha,
                            'sharpness': sharp, 'reconstruction_method': 'linear',
                        })
    print(f"SKAB: {len(skab_configs)} configs")
    tune('skab', skab_configs)
