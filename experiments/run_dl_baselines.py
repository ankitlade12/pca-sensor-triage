"""
Experiment 14: Deep Learning Baselines (LSTM-Attention + Transformer-Attention).

Compares PCA-Triage against learned channel importance methods.
Key question: does deep learning's expressiveness justify its compute cost?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from src.triage.pipeline import TriagePipeline
from src.baselines.lstm_attention import LSTMAttentionSampling, TransformerAttentionSampling
from src.baselines import UniformSampling, VarianceSampling
from src.utils.data_loader import get_dataset

from run_pareto_v2 import DATASET_CONFIGS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATASETS = ['tep', 'smd', 'msl', 'psm', 'hai', 'skab', 'swat']


def evaluate_method(name, process_fn, X_train, y_tr, X_test, y_te, seeds=[42, 123]):
    """Evaluate a triage method with timing."""
    f1s, times = [], []
    for seed in seeds:
        t0 = time.time()
        X_tr = process_fn(X_train, seed)
        X_te = process_fn(X_test, seed + 10000)
        elapsed = time.time() - t0

        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        f1 = f1_score(y_te, clf.predict(X_te), average='weighted')
        f1s.append(f1)
        times.append(elapsed)
    return np.mean(f1s), np.std(f1s), np.mean(times)


def run_dataset(ds_name):
    print(f"\n{'='*60}", flush=True)
    print(f"DATASET: {ds_name.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    result = get_dataset(ds_name)
    X_train, y_train, X_test, y_test = result[0], result[1], result[2], result[3]
    le = LabelEncoder()
    y_tr, y_te = le.fit_transform(y_train), le.transform(y_test)
    n_ch = X_train.shape[1]
    print(f"  Shape: {X_train.shape}, Channels: {n_ch}", flush=True)

    # Subsample for DL baselines (they're slow on full data)
    max_samples = 20000
    if len(X_train) > max_samples:
        rng = np.random.RandomState(42)
        idx_tr = rng.choice(len(X_train), max_samples, replace=False)
        idx_te = rng.choice(len(X_test), min(max_samples // 2, len(X_test)), replace=False)
        X_train_sub = X_train[idx_tr]
        y_tr_sub = y_tr[idx_tr]
        X_test_sub = X_test[idx_te]
        y_te_sub = y_te[idx_te]
        print(f"  Subsampled to {max_samples} train, {len(idx_te)} test (DL baselines)", flush=True)
    else:
        X_train_sub, y_tr_sub = X_train, y_tr
        X_test_sub, y_te_sub = X_test, y_te

    rows = []

    # PCA-Triage v2 (on subsampled data for fair comparison)
    cfg = DATASET_CONFIGS.get(ds_name, DATASET_CONFIGS['tep'])
    def pca_fn(X, seed):
        pipe = TriagePipeline(
            n_components=cfg['n_components'], window_size=cfg['window_size'],
            budget=0.5, forgetting_factor=cfg['forgetting_factor'],
            min_rate=cfg['min_rate'], reconstruction_method=cfg['reconstruction_method'],
            scorer=cfg['scorer'], alpha=cfg['alpha'],
            sharpness=cfg.get('sharpness', 1.0))
        return pipe.process_stream(X, seed=seed)

    f1, std, t = evaluate_method("PCA-Triage v2", pca_fn,
                                  X_train_sub, y_tr_sub, X_test_sub, y_te_sub)
    rows.append({'dataset': ds_name, 'method': 'PCA-Triage v2', 'f1_mean': f1,
                 'f1_std': std, 'time_s': t, 'params': 0})
    print(f"  PCA-Triage v2          : F1={f1:.4f} ± {std:.4f} ({t:.1f}s, 0 params) ***", flush=True)

    # Variance baseline
    def var_fn(X, seed):
        return VarianceSampling(budget=0.5, window_size=50).process_stream(X, seed=seed)
    f1, std, t = evaluate_method("Variance", var_fn,
                                  X_train_sub, y_tr_sub, X_test_sub, y_te_sub)
    rows.append({'dataset': ds_name, 'method': 'Variance', 'f1_mean': f1,
                 'f1_std': std, 'time_s': t, 'params': 0})
    print(f"  Variance               : F1={f1:.4f} ± {std:.4f} ({t:.1f}s, 0 params)", flush=True)

    # LSTM-Attention
    try:
        def lstm_fn(X, seed):
            lstm = LSTMAttentionSampling(budget=0.5, window_size=50,
                                          hidden_size=32, n_epochs=3, lr=0.001)
            return lstm.process_stream(X, seed=seed)
        f1, std, t = evaluate_method("LSTM-Attention", lstm_fn,
                                      X_train_sub, y_tr_sub, X_test_sub, y_te_sub)
        # Count params
        temp_lstm = LSTMAttentionSampling(budget=0.5, window_size=50, hidden_size=32)
        temp_lstm._init_model(n_ch)
        n_params = temp_lstm.n_params
        rows.append({'dataset': ds_name, 'method': 'LSTM-Attention', 'f1_mean': f1,
                     'f1_std': std, 'time_s': t, 'params': n_params})
        print(f"  LSTM-Attention          : F1={f1:.4f} ± {std:.4f} ({t:.1f}s, {n_params:,} params)", flush=True)
    except Exception as e:
        print(f"  LSTM-Attention          : ERROR {e}", flush=True)

    # Transformer-Attention
    try:
        def tfm_fn(X, seed):
            tfm = TransformerAttentionSampling(budget=0.5, window_size=50,
                                                d_model=32, n_heads=4, n_epochs=3, lr=0.001)
            return tfm.process_stream(X, seed=seed)
        f1, std, t = evaluate_method("Transformer-Attention", tfm_fn,
                                      X_train_sub, y_tr_sub, X_test_sub, y_te_sub)
        temp_tfm = TransformerAttentionSampling(budget=0.5, window_size=50, d_model=32, n_heads=4)
        temp_tfm._init_model(n_ch)
        n_params = temp_tfm.n_params
        rows.append({'dataset': ds_name, 'method': 'Transformer-Attention', 'f1_mean': f1,
                     'f1_std': std, 'time_s': t, 'params': n_params})
        print(f"  Transformer-Attention   : F1={f1:.4f} ± {std:.4f} ({t:.1f}s, {n_params:,} params)", flush=True)
    except Exception as e:
        print(f"  Transformer-Attention   : ERROR {e}", flush=True)

    return rows


if __name__ == "__main__":
    all_rows = []
    for ds in DATASETS:
        try:
            rows = run_dataset(ds)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ERROR on {ds}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'dl_baselines.csv'), index=False)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY: PCA-Triage vs Deep Learning Baselines", flush=True)
    print(f"{'='*60}", flush=True)
    for ds in DATASETS:
        sub = df[df['dataset'] == ds].sort_values('f1_mean', ascending=False)
        if len(sub) > 0:
            best = sub.iloc[0]
            pca = sub[sub['method'] == 'PCA-Triage v2']
            pca_f1 = pca.iloc[0]['f1_mean'] if len(pca) > 0 else 0
            print(f"  {ds:6s}: Winner={best['method']:25s} F1={best['f1_mean']:.4f} "
                  f"(PCA: {pca_f1:.4f})", flush=True)

    print(f"\nSaved to {RESULTS_DIR}/dl_baselines.csv", flush=True)
