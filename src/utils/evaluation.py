"""Leakage-resistant point, event, and communication metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def contiguous_events(labels: Iterable[int]) -> list[tuple[int, int]]:
    """Return half-open ``(start, end)`` intervals for positive events."""
    y = np.asarray(labels, dtype=int).reshape(-1)
    padded = np.pad(y > 0, (1, 1), constant_values=False)
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def detection_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], run_lengths: Iterable[int] | None = None,
) -> dict[str, float]:
    """Pool point metrics while preserving independent run boundaries for events.

    Delays are in samples and conditional on detection; missed events are counted
    separately. No event or predicted alarm interval can span two runs.
    """
    truth = (np.asarray(y_true).reshape(-1) > 0).astype(int)
    pred = (np.asarray(y_pred).reshape(-1) > 0).astype(int)
    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred must have equal length")

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    lengths = [len(truth)] if run_lengths is None else list(run_lengths)
    if any(n < 0 for n in lengths) or sum(lengths) != len(truth):
        raise ValueError("run_lengths must partition the complete predictions")
    true_events, pred_events = [], []
    offset = 0
    for length in lengths:
        true_events.extend((a + offset, b + offset) for a, b in contiguous_events(truth[offset:offset + length]))
        pred_events.extend((a + offset, b + offset) for a, b in contiguous_events(pred[offset:offset + length]))
        offset += length
    delays = []
    detected = 0
    for start, end in true_events:
        hits = np.flatnonzero(pred[start:end])
        if len(hits):
            detected += 1
            delays.append(float(hits[0]))

    false_events = sum(not np.any(truth[ps:pe]) for ps, pe in pred_events)
    n_true_events = len(true_events)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "specificity": float(specificity),
        "event_recall": detected / n_true_events if n_true_events else 0.0,
        "mean_detection_delay": float(np.mean(delays)) if delays else float("nan"),
        "median_detection_delay": float(np.median(delays)) if delays else float("nan"),
        "false_alarm_events": float(false_events),
        "false_alarms_per_1000": 1000.0 * false_events / len(truth) if len(truth) else 0.0,
        "n_events": float(n_true_events),
        "detected_events": float(detected),
        "missed_events": float(n_true_events - detected),
    }


def summarize_acquisition(records: list[dict], requested_budget: float) -> dict:
    """Aggregate an audit ledger populated from actual sampling masks."""
    total = sum(row["available_values"] for row in records)
    transmitted = sum(row["transmitted_values"] for row in records)
    realized = transmitted / total if total else 0.0
    return {
        "available_values": total,
        "transmitted_values": transmitted,
        "transmitted_payload_bytes": transmitted * 4,
        "payload_bytes_per_value": 4,
        "realized_bandwidth": realized,
        "budget_excess": max(0.0, realized - requested_budget),
        "max_window_budget_excess": max(
            (row["excess_values"] / row["available_values"] for row in records if row["available_values"]),
            default=0.0,
        ),
        "max_window_excess_values": max((row["excess_values"] for row in records), default=0),
        "n_windows": len(records),
    }


def communication_metrics(
    observed_mask: np.ndarray,
    requested_budget: float,
    bytes_per_value: int = 4,
) -> dict[str, float]:
    """Measure realized communication from a boolean observation mask."""
    mask = np.asarray(observed_mask, dtype=bool)
    transmitted = int(mask.sum())
    total = int(mask.size)
    realized = transmitted / total if total else 0.0
    return {
        "transmitted_values": float(transmitted),
        "transmitted_bytes_payload": float(transmitted * bytes_per_value),
        "realized_bandwidth": float(realized),
        "budget_excess": float(max(0.0, realized - requested_budget)),
        "budget_slack": float(max(0.0, requested_budget - realized)),
    }
