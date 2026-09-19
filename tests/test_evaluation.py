import numpy as np

from src.utils.evaluation import communication_metrics, contiguous_events, detection_metrics


def test_contiguous_events():
    assert contiguous_events([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]


def test_detection_metrics_include_delay_and_false_alarms():
    truth = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    pred = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    metrics = detection_metrics(truth, pred)
    assert metrics["event_recall"] == 1.0
    assert metrics["mean_detection_delay"] == 0.5
    assert metrics["false_alarm_events"] == 1.0
    assert metrics["fpr"] > 0


def test_communication_metrics_measure_actual_mask():
    mask = np.array([[True, False], [True, False]])
    metrics = communication_metrics(mask, requested_budget=0.5)
    assert metrics["realized_bandwidth"] == 0.5
    assert metrics["transmitted_bytes_payload"] == 8.0
    assert metrics["budget_excess"] == 0.0


def test_run_boundaries_do_not_merge_false_alarm_with_true_event():
    truth = [0, 0, 1, 1]
    prediction = [0, 1, 1, 1]
    metrics = detection_metrics(truth, prediction, run_lengths=[2, 2])
    assert metrics["false_alarm_events"] == 1
    assert metrics["detected_events"] == 1
    assert metrics["missed_events"] == 0
    assert metrics["mean_detection_delay"] == 0


def test_undetected_event_delay_is_undefined():
    metrics = detection_metrics([0, 1, 1], [0, 0, 0])
    assert metrics["missed_events"] == 1
    assert np.isnan(metrics["mean_detection_delay"])


def test_acquisition_summary_reports_observed_excess():
    from src.utils.evaluation import summarize_acquisition
    metrics = summarize_acquisition([
        dict(available_values=20, transmitted_values=12, excess_values=2),
        dict(available_values=10, transmitted_values=4, excess_values=0),
    ], 0.5)
    assert metrics["transmitted_values"] == 16
    assert metrics["transmitted_payload_bytes"] == 64
    assert metrics["max_window_budget_excess"] == 0.1
