"""
Run adaptive k and ensemble scorer ablations across ALL 7 Pareto-evaluated datasets.
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
from src.utils.data_loader import get_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATASETS = ['tep', 'smd', 'msl', 'psm', 'hai', 'skab', 'swat']


def evaluate(X_train, y_tr, X_test, y_te, pipe_kwargs, seeds=[42, 123, 456]):
    f1s = []
    for seed in seeds:
        pipe = TriagePipeline(**pipe_kwargs, budget=0.5)
        X_tr = pipe.process_stream(X_train, seed=seed)
        X_te = pipe.process_stream(X_test, seed=seed + 10000)
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        f1s.append(f1_score(y_te, clf.predict(X_te), average='weighted'))
    return np.mean(f1s), np.std(f1s)


def run_all():
    all_rows = []

    for ds in DATASETS:
        print(f"\n{'='*60}", flush=True)
        print(f"DATASET: {ds.upper()}", flush=True)
        print(f"{'='*60}", flush=True)

        result = get_dataset(ds)
        X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
        le = LabelEncoder()
        y_tr, y_te = le.fit_transform(y_train), le.transform(y_test)
        n_ch = X_train.shape[1]
        print(f"  Channels: {n_ch}", flush=True)

        # Adaptive k configs
        k_max = min(20, n_ch - 1)
        configs = [
            ("Fixed k=5", dict(n_components=min(5, k_max), window_size=50,
                               forgetting_factor=1.0, min_rate=0.05,
                               scorer='pca', reconstruction_method='linear')),
            ("Fixed k=10", dict(n_components=min(10, k_max), window_size=50,
                                forgetting_factor=1.0, min_rate=0.05,
                                scorer='pca', reconstruction_method='linear')),
            ("Adaptive k (90%)", dict(n_components=k_max, window_size=50,
                                      forgetting_factor=1.0, min_rate=0.05,
                                      scorer='adaptive_k', variance_threshold=0.90,
                                      reconstruction_method='linear')),
            ("Adaptive k (95%)", dict(n_components=k_max, window_size=50,
                                      forgetting_factor=1.0, min_rate=0.05,
                                      scorer='adaptive_k', variance_threshold=0.95,
                                      reconstruction_method='linear')),
            ("Ensemble [3,5,10]", dict(n_components=min(10, k_max), window_size=50,
                                       forgetting_factor=1.0, min_rate=0.05,
                                       scorer='ensemble',
                                       k_values=[min(3, k_max), min(5, k_max), min(10, k_max)],
                                       reconstruction_method='linear')),
            ("Hybrid v2", dict(n_components=min(10, k_max), window_size=50,
                               forgetting_factor=1.0, min_rate=0.05,
                               scorer='hybrid', alpha=0.7, sharpness=1.5,
                               reconstruction_method='linear')),
        ]

        for name, cfg in configs:
            try:
                mean_f1, std_f1 = evaluate(X_train, y_tr, X_test, y_te, cfg)
                all_rows.append({
                    'dataset': ds, 'config': name,
                    'f1_mean': mean_f1, 'f1_std': std_f1,
                })
                print(f"  {name:25s}: F1={mean_f1:.4f} ± {std_f1:.4f}", flush=True)
            except Exception as e:
                print(f"  {name:25s}: ERROR {e}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'multi_dataset_ablation.csv'), index=False)
    print(f"\nSaved to {RESULTS_DIR}/multi_dataset_ablation.csv", flush=True)

    # Summary: which scorer wins per dataset?
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY: Best scorer per dataset", flush=True)
    print(f"{'='*60}", flush=True)
    for ds in DATASETS:
        sub = df[df['dataset'] == ds].sort_values('f1_mean', ascending=False)
        if len(sub) > 0:
            best = sub.iloc[0]
            print(f"  {ds:6s}: {best['config']:25s} F1={best['f1_mean']:.4f}", flush=True)


if __name__ == "__main__":
    run_all()
