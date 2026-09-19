"""Causal sensor-allocation benchmark with mask-derived communication audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from src.baselines import ThresholdSampling, UniformSampling, VarianceSampling
from src.triage import TriagePipeline, compute_reconstruction_error
from src.utils.data_loader import get_dataset
from src.utils.evaluation import detection_metrics, summarize_acquisition

METHODS = ("PCA-Triage", "Variance", "Threshold", "Uniform", "Full Data")
QUANTILES = (0.95, 0.975, 0.99, 0.995, 0.999)


def acquire_stream(method: str, data: np.ndarray, budget: float, seed: int):
    """Start an independent stream with fresh PCA, rate, and reconstruction state."""
    if method == "PCA-Triage":
        transformer = TriagePipeline(
            n_components=min(10, data.shape[1]), window_size=50, budget=budget,
            forgetting_factor=0.95, min_rate=min(0.05, budget / 2),
            reconstruction_method="forward_fill", scorer="pca",
            update_source="reconstructed", hard_budget=True,
        )
    elif method == "Variance":
        transformer = VarianceSampling(budget=budget, window_size=50)
    elif method == "Threshold":
        transformer = ThresholdSampling(budget=budget, window_size=50)
    elif method == "Uniform":
        transformer = UniformSampling(budget=budget, window_size=50)
    elif method == "Full Data":
        records = []
        for start in range(0, len(data), 50):
            window = data[start:start + 50]
            observed = int(np.isfinite(window).sum())
            if observed != window.size:
                raise ValueError("Full-rate input must be finite")
            records.append(dict(window_samples=len(window), channels=data.shape[1],
                                available_values=window.size, transmitted_values=observed,
                                allowed_values=window.size, excess_values=0))
        return data.copy(), records
    else:
        raise ValueError(f"Unknown method: {method}")
    reconstructed = transformer.process_stream(data, seed=seed)
    return reconstructed, transformer.allocator.observation_log


def audit_rows(records, *, dataset, method, budget, seed, partition, stream_id="0"):
    return [dict(dataset=dataset, method=method, requested_budget=budget, seed=seed,
                 partition=partition, stream_id=stream_id, window=i, **row)
            for i, row in enumerate(records)]


def fit_detector(X_train: np.ndarray, seed: int, max_fit_samples: int = 20_000):
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(X_train), min(len(X_train), max_fit_samples), replace=False))
    fit_data = X_train[idx]
    detector = IsolationForest(n_estimators=100, max_samples=min(2048, len(fit_data)),
                               contamination="auto", random_state=seed, n_jobs=-1)
    detector.fit(fit_data)
    scores = -detector.score_samples(fit_data)
    return detector, {q: float(np.quantile(scores, q)) for q in QUANTILES}


def run_dataset(dataset: str, budget: float, seed: int):
    X_train, y_train, X_test, y_test, _, _ = get_dataset(dataset)
    if np.any(np.asarray(y_train) != 0):
        raise ValueError("This protocol requires an entirely normal training stream")
    fixed_detector, fixed_thresholds = fit_detector(X_train, seed)
    rows, audits, sensitivity = [], [], []
    for method in METHODS:
        started = time.perf_counter()
        train_recon, train_records = acquire_stream(method, X_train, budget, seed)
        test_recon, test_records = acquire_stream(method, X_test, budget, seed + 10_000)
        effective_budget = 1.0 if method == "Full Data" else budget
        usage = summarize_acquisition(test_records, effective_budget)
        for partition, records in (("train", train_records), ("test", test_records)):
            audits.extend(audit_rows(records, dataset=dataset, method=method, budget=budget,
                                    seed=seed, partition=partition))
        reconstruction = compute_reconstruction_error(X_test, test_recon)
        for protocol in ("method_retrained", "fixed_full_data"):
            if protocol == "method_retrained" and method != "Full Data":
                detector, thresholds = fit_detector(train_recon, seed)
            else:
                detector, thresholds = fixed_detector, fixed_thresholds
            scores = -detector.score_samples(test_recon)
            common = dict(dataset=dataset, method=method, protocol=protocol,
                          budget=effective_budget, requested_budget=budget, seed=seed)
            for quantile, threshold in thresholds.items():
                metrics = detection_metrics(y_test, scores >= threshold)
                sensitivity.append(dict(**common, quantile=quantile, threshold=threshold, **metrics))
                if quantile == 0.99:
                    rows.append(dict(**common, **usage, rmse=reconstruction["rmse"],
                                     mae=reconstruction["mae"], nrmse_mean=reconstruction["nrmse_mean"],
                                     elapsed_s=time.perf_counter() - started, **metrics))
    return rows, audits, sensitivity


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(args, data_files):
    source_files = sorted(Path("src").rglob("*.py")) + [
        Path("experiments/run_causal_benchmark.py"), Path("experiments/run_causal_tep.py")]
    return dict(
        protocol_version="2026-09-06-mask-audited", created_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv, python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, sklearn=sklearn.__version__,
        datasets=getattr(args, "datasets", ["tep"]), budgets=args.budgets, seeds=args.seeds,
        methods=METHODS, window_size=50, n_components=10, smoothing=0.95, min_rate=0.05,
        score_order="normalize covariance diagonal, then smooth",
        stream_initialization="all allocation and reconstruction state reset per partition/run",
        accounting="counts from acquisition masks; hypothetical float32 payload, 4 bytes/value",
        seed_interpretation="joint acquisition, detector, and fitting-subsample seeds",
        threshold_quantiles=QUANTILES, primary_quantile=0.99,
        source_sha256={str(p): file_hash(p) for p in source_files},
        data_sha256={str(p): file_hash(p) for p in data_files},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["smd", "psm"], default=["smd", "psm"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--output", type=Path, default=Path("experiments/results/causal_benchmark.csv"))
    args = parser.parse_args()
    rows, audits, sensitivity = [], [], []
    for dataset in args.datasets:
        for budget in args.budgets:
            for seed in args.seeds:
                result, ledger, thresholds = run_dataset(dataset, budget, seed)
                rows.extend(result)
                audits.extend(ledger)
                sensitivity.extend(thresholds)
                print(f"completed dataset={dataset} budget={budget:.2f} seed={seed}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    pd.DataFrame(audits).to_csv(args.output.with_suffix(".windows.csv.gz"), index=False)
    pd.DataFrame(sensitivity).to_csv(args.output.with_suffix(".thresholds.csv"), index=False)
    data_files = []
    if "smd" in args.datasets:
        data_files += [Path(f"data/raw/omni_temp/ServerMachineDataset/{part}/machine-1-1.txt")
                       for part in ("train", "test", "test_label")]
    if "psm" in args.datasets:
        data_files += [Path(f"data/raw/psm_temp/data/{part}.csv") for part in ("train", "test", "test_label")]
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest(args, data_files), indent=2) + "\n")
    print(f"saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
