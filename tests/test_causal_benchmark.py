import numpy as np
import pytest

from experiments.run_causal_benchmark import METHODS, acquire_stream
from src.utils.evaluation import summarize_acquisition


@pytest.mark.parametrize('method', METHODS)
def test_benchmark_counts_each_window_and_resets_independent_streams(method):
    data = np.random.RandomState(8).normal(size=(107, 4))
    reconstructed, ledger = acquire_stream(method, data, 0.3, 42)
    again, other = acquire_stream(method, data, 0.3, 42)
    assert np.array_equal(reconstructed, again)
    assert ledger == other
    assert [row['window_samples'] for row in ledger] == [50, 50, 7]
    budget = 1.0 if method == 'Full Data' else 0.3
    usage = summarize_acquisition(ledger, budget)
    assert usage['transmitted_values'] == sum(int(np.floor(budget * n * 4)) for n in [50, 50, 7])
    assert usage['max_window_excess_values'] == 0
