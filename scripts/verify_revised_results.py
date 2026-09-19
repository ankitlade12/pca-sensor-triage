"""Reconcile published results with mask-derived ledgers and source manifests."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify_ledger(path):
    data = pd.read_csv(path)
    ledger = pd.read_csv(path.with_suffix('.windows.csv.gz'))
    keys = ['dataset', 'method', 'requested_budget', 'seed']
    assert not ledger.duplicated(keys + ['partition', 'stream_id', 'window']).any()
    assert set(ledger.partition) == {'train', 'test'}
    assert (ledger.available_values == ledger.window_samples * ledger.channels).all()
    assert (ledger.transmitted_values >= 0).all()
    assert (ledger.transmitted_values <= ledger.available_values).all()
    effective = np.where(ledger.method == 'Full Data', 1.0, ledger.requested_budget)
    allowed = np.floor(effective * ledger.window_samples * ledger.channels).astype(int)
    assert np.array_equal(ledger.allowed_values, allowed)
    excess = np.maximum(0, ledger.transmitted_values - allowed)
    assert np.array_equal(ledger.excess_values, excess)
    assert (ledger.transmitted_values == allowed).all(), 'quota not actually satisfied'
    test = ledger[ledger.partition == 'test'].groupby(keys).agg(
        counted_values=('transmitted_values', 'sum'), counted_available=('available_values', 'sum'),
        counted_windows=('window', 'size'), max_excess=('excess_values', 'max'))
    actual = data.merge(test, on=keys, how='left', validate='many_to_one')
    assert len(actual) == len(data) and actual.counted_values.notna().all()
    assert (actual.transmitted_values == actual.counted_values).all()
    assert (actual.available_values == actual.counted_available).all()
    assert (actual.n_windows == actual.counted_windows).all()
    assert np.allclose(actual.realized_bandwidth, actual.counted_values / actual.counted_available, atol=1e-12)
    assert (actual.transmitted_payload_bytes == 4 * actual.counted_values).all()
    assert (actual.max_window_excess_values == actual.max_excess).all()
    assert (actual.max_window_budget_excess == 0).all()
    assert (actual.budget_excess == 0).all()
    manifest = json.loads(path.with_suffix('.manifest.json').read_text())
    assert manifest['protocol_version'] == '2026-09-06-mask-audited'
    for filename, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, f'Source changed since experiment: {filename}'
    return data


def verify_anomaly_results():
    path = ROOT / 'experiments/results/causal_benchmark.csv'
    data = verify_ledger(path)
    assert len(data) == 2 * 3 * 5 * 5 * 2
    assert set(data.dataset) == {'smd', 'psm'}
    keys = ['dataset', 'method', 'protocol', 'requested_budget', 'seed']
    assert not data.duplicated(keys).any()
    for metric in ['precision', 'recall', 'f1', 'fpr', 'specificity', 'event_recall']:
        assert data[metric].between(0, 1).all()
    assert (data.missed_events + data.detected_events == data.n_events).all()
    sensitivity = pd.read_csv(path.with_suffix('.thresholds.csv'))
    assert set(sensitivity['quantile']) == {0.95, 0.975, 0.99, 0.995, 0.999}
    assert len(sensitivity) == len(data) * 5
    main = sensitivity[sensitivity['quantile'] == 0.99].merge(data, on=keys, suffixes=('_sweep', '_main'), validate='one_to_one')
    assert np.allclose(main.f1_sweep, main.f1_main)


def verify_tep_results():
    path = ROOT / 'experiments/results/causal_tep.csv'
    data = verify_ledger(path)
    assert len(data) == 1 * 5 * 5 * 2
    assert set(data.dataset) == {'tep'}
    assert (data.binary_n_events == 100).all()
    assert (data.binary_missed_events + data.binary_detected_events == 100).all()
    per_fault = pd.read_csv(ROOT / 'experiments/results/causal_tep_per_fault.csv')
    assert set(per_fault.fault_type) == set(range(1, 21))
    assert len(per_fault) == 5 * 5 * 2 * 20
    assert (per_fault.binary_n_events == 5).all()


def verify_manuscript_scope():
    text = (ROOT / 'paper/main_revised.tex').read_text()
    for claim in ['0.961', '0.970', 'best unsupervised', 'scales to 500 channels', 'Corrected experiments are in progress']:
        assert claim not in text, f'legacy claim: {claim}'
    assert 'MSL & 55' not in text
    for name in ['causal_abstract.tex', 'causal_summary.tex', 'causal_metrics.tex', 'causal_thresholds.tex']:
        assert (ROOT / 'paper/tables' / name).is_file()
    assert 'MSL' not in (ROOT / 'paper/tables/causal_summary.tex').read_text()


if __name__ == '__main__':
    verify_anomaly_results()
    verify_tep_results()
    verify_manuscript_scope()
    print('Revised artifacts verified against per-window masks and experiment source hashes')
