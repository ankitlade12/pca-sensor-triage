"""Causal, run-separated Tennessee Eastman Process evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from run_causal_benchmark import METHODS, acquire_stream, audit_rows, manifest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.triage import compute_reconstruction_error
from src.utils.evaluation import detection_metrics, summarize_acquisition

DATA_DIR = Path("data/raw")


def load_runs(
    fault_types=tuple(range(21)),
    train_run_end=5,
    test_run_start=21,
    test_run_end=25,
    fault_onset_sample=20,
):
    """Return independent scaled runs; no online state crosses a run boundary."""
    fault_free = next(
        iter(pyreadr.read_r(DATA_DIR / "TEP_FaultFree_Training.RData").values())
    )
    faulty = next(iter(pyreadr.read_r(DATA_DIR / "TEP_Faulty_Training.RData").values()))
    sensors = [
        c for c in fault_free.columns if c.startswith("xmeas_") or c.startswith("xmv_")
    ]

    train_records = []
    test_records = []
    for fault in fault_types:
        source = fault_free if fault == 0 else faulty[faulty["faultNumber"] == fault]
        for run_id, run_df in source.groupby("simulationRun", sort=True):
            if run_id <= train_run_end:
                target = train_records
            elif test_run_start <= run_id <= test_run_end:
                target = test_records
            else:
                continue
            X = run_df.sort_values("sample")[sensors].to_numpy(dtype=float)
            samples = run_df.sort_values("sample")["sample"].to_numpy()
            if fault == 0:
                y = np.zeros(len(X), dtype=int)
            else:
                y = np.where(samples >= fault_onset_sample, fault, 0)
            target.append((int(fault), int(run_id), X, y))

    scaler = StandardScaler().fit(np.vstack([record[2] for record in train_records]))
    train_records = [
        (fault, run, scaler.transform(X), y) for fault, run, X, y in train_records
    ]
    test_records = [
        (fault, run, scaler.transform(X), y) for fault, run, X, y in test_records
    ]
    return train_records, test_records


def transform_runs(method, records, budget, seed, partition):
    transformed, audits = [], []
    for index, (fault, run, X, y) in enumerate(records):
        X_recon, ledger = acquire_stream(method, X, budget, seed + index)
        transformed.append((fault, run, X_recon, y))
        audits.extend(audit_rows(ledger, dataset="tep", method=method, budget=budget,
                                seed=seed, partition=partition, stream_id=f"{fault}:{run}"))
    return transformed, audits


def stack(records):
    return np.vstack([r[2] for r in records]), np.concatenate([r[3] for r in records])


def run_once(train_records, test_records, budget, seed):
    X_train_full, y_train = stack(train_records)
    X_test_full, y_test = stack(test_records)
    fixed = RandomForestClassifier(
        n_estimators=50, class_weight="balanced_subsample", random_state=seed, n_jobs=-1
    ).fit(X_train_full, y_train)
    rows = []
    per_fault_rows = []
    audits = []
    run_lengths = [len(r[3]) for r in test_records]

    for method in METHODS:
        started = time.perf_counter()
        train_recon, train_audit = transform_runs(method, train_records, budget, seed, "train")
        test_recon, test_audit = transform_runs(method, test_records, budget, seed + 10_000, "test")
        # Ledger seed identifies the experimental replicate, not its sampling offset.
        for row in test_audit:
            row["seed"] = seed
        audits.extend(train_audit + test_audit)
        X_train, _ = stack(train_recon)
        X_test, _ = stack(test_recon)
        usage = summarize_acquisition(test_audit, 1.0 if method == "Full Data" else budget)
        error = compute_reconstruction_error(X_test_full, X_test)

        for protocol in ("method_retrained", "fixed_full_data"):
            if protocol == "method_retrained":
                classifier = RandomForestClassifier(
                    n_estimators=50,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ).fit(X_train, y_train)
            else:
                classifier = fixed
            predicted = classifier.predict(X_test)
            binary = detection_metrics(y_test != 0, predicted != 0, run_lengths=run_lengths)
            rows.append(
                {
                    "dataset": "tep",
                    "method": method,
                    "protocol": protocol,
                    "budget": 1.0 if method == "Full Data" else budget,
                    "seed": seed,
                    "requested_budget": budget,
                    "weighted_f1": f1_score(y_test, predicted, average="weighted"),
                    "macro_f1": f1_score(y_test, predicted, average="macro"),
                    "macro_precision": precision_score(
                        y_test, predicted, average="macro", zero_division=0
                    ),
                    "macro_recall": recall_score(
                        y_test, predicted, average="macro", zero_division=0
                    ),
                    "rmse": error["rmse"],
                    "mae": error["mae"],
                    "nrmse_mean": error["nrmse_mean"],
                    **usage,
                    "elapsed_s": time.perf_counter() - started,
                    **{f"binary_{key}": value for key, value in binary.items()},
                }
            )

            offset = 0
            grouped_predictions = {}
            for fault, _, _, labels in test_recon:
                run_predicted = predicted[offset : offset + len(labels)]
                offset += len(labels)
                if fault == 0:
                    continue
                grouped_predictions.setdefault(fault, [[], []])
                grouped_predictions[fault][0].append(labels)
                grouped_predictions[fault][1].append(run_predicted)
            for fault, (label_parts, prediction_parts) in grouped_predictions.items():
                fault_labels = np.concatenate(label_parts)
                fault_predictions = np.concatenate(prediction_parts)
                fault_binary = detection_metrics(
                    fault_labels != 0, fault_predictions != 0,
                    run_lengths=[len(part) for part in label_parts],
                )
                active = fault_labels == fault
                per_fault_rows.append(
                    {
                        "dataset": "tep",
                        "method": method,
                        "protocol": protocol,
                        "budget": 1.0 if method == "Full Data" else budget,
                        "seed": seed,
                        "fault_type": fault,
                        "exact_fault_recall": float(
                            np.mean(fault_predictions[active] == fault)
                        ),
                        **{f"binary_{key}": value for key, value in fault_binary.items()},
                    }
                )
    return rows, per_fault_rows, audits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/results/causal_tep.csv")
    )
    args = parser.parse_args()
    train_records, test_records = load_runs()
    rows = []
    per_fault_rows = []
    audits = []
    for budget in args.budgets:
        for seed in args.seeds:
            overall, by_fault, ledger = run_once(train_records, test_records, budget, seed)
            rows.extend(overall)
            per_fault_rows.extend(by_fault)
            audits.extend(ledger)
            print(f"completed dataset=tep budget={budget:.2f} seed={seed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    per_fault_output = args.output.with_name(f"{args.output.stem}_per_fault.csv")
    pd.DataFrame(per_fault_rows).to_csv(per_fault_output, index=False)
    pd.DataFrame(audits).to_csv(args.output.with_suffix(".windows.csv.gz"), index=False)
    provenance = manifest(args, [DATA_DIR / "TEP_FaultFree_Training.RData", DATA_DIR / "TEP_Faulty_Training.RData"])
    provenance.update(train_runs=[1, 2, 3, 4, 5], test_runs=[21, 22, 23, 24, 25],
                      fault_onset_sample=20, label_rule="sample >= 20", event_boundaries="per run",
                      rmse_aggregation="pooled samples and channels")
    args.output.with_suffix(".manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"saved {len(rows)} rows to {args.output}")
    print(f"saved {len(per_fault_rows)} rows to {per_fault_output}")


if __name__ == "__main__":
    main()
