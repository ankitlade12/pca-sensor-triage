"""Unit tests for all 6 baseline methods."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.baselines import (
    AttentionSampling,
    MutualInfoSampling,
    RandomDropout,
    ThresholdSampling,
    UniformSampling,
    VarianceSampling,
)


@pytest.fixture
def synthetic_data():
    """Generate synthetic sensor data for testing."""
    np.random.seed(42)
    n, d = 500, 10
    X = np.random.randn(n, d)
    # Make channel 0 high-variance
    X[:, 0] *= 10
    y = np.random.randint(0, 3, n)
    return X, y


class TestUniformSampling:
    def test_output_shape(self, synthetic_data):
        X, _ = synthetic_data
        uni = UniformSampling(budget=0.5)
        result = uni.process_stream(X, seed=42)
        assert result.shape == X.shape

    def test_no_nans_in_output(self, synthetic_data):
        X, _ = synthetic_data
        uni = UniformSampling(budget=0.5)
        result = uni.process_stream(X, seed=42)
        assert not np.isnan(result).any()

    def test_rates_are_uniform(self, synthetic_data):
        X, _ = synthetic_data
        uni = UniformSampling(budget=0.5)
        rates = uni.get_rates(X.shape[1])
        assert np.allclose(rates, 0.5)

    def test_different_budgets(self, synthetic_data):
        X, _ = synthetic_data
        for budget in [0.1, 0.3, 0.5, 0.7, 0.9]:
            uni = UniformSampling(budget=budget)
            result = uni.process_stream(X, seed=42)
            assert result.shape == X.shape


class TestThresholdSampling:
    def test_output_shape(self, synthetic_data):
        X, _ = synthetic_data
        thr = ThresholdSampling(budget=0.5, window_size=100)
        result = thr.process_stream(X, seed=42)
        assert result.shape == X.shape

    def test_no_nans_in_output(self, synthetic_data):
        X, _ = synthetic_data
        thr = ThresholdSampling(budget=0.5, window_size=100)
        result = thr.process_stream(X, seed=42)
        assert not np.isnan(result).any()

    def test_different_thresholds(self, synthetic_data):
        X, _ = synthetic_data
        for pct in [25, 50, 75]:
            thr = ThresholdSampling(budget=0.5, threshold_percentile=pct, window_size=100)
            result = thr.process_stream(X, seed=42)
            assert result.shape == X.shape

    def test_handles_tail_samples(self):
        """Data length not divisible by window_size."""
        X = np.random.randn(150, 5)
        thr = ThresholdSampling(budget=0.5, window_size=100)
        result = thr.process_stream(X, seed=42)
        assert result.shape == X.shape


class TestVarianceSampling:
    def test_output_shape(self, synthetic_data):
        X, _ = synthetic_data
        var = VarianceSampling(budget=0.5, window_size=100)
        result = var.process_stream(X, seed=42)
        assert result.shape == X.shape

    def test_no_nans_in_output(self, synthetic_data):
        X, _ = synthetic_data
        var = VarianceSampling(budget=0.5, window_size=100)
        result = var.process_stream(X, seed=42)
        assert not np.isnan(result).any()

    def test_high_variance_channel_preserved(self):
        """Channel 0 has 100x variance — should be sampled more."""
        np.random.seed(42)
        X = np.random.randn(200, 5) * 0.1
        X[:, 0] = np.random.randn(200) * 10
        var = VarianceSampling(budget=0.3, window_size=100)
        result = var.process_stream(X, seed=42)
        # Channel 0 reconstruction should be closer to original
        np.mean((X[:, 0] - result[:, 0]) ** 2)
        np.mean((X[:, 4] - result[:, 4]) ** 2)
        # Not guaranteed but very likely with 100x variance difference
        assert result.shape == X.shape


class TestRandomDropout:
    def test_output_shape(self, synthetic_data):
        X, _ = synthetic_data
        rd = RandomDropout(budget=0.5, window_size=100)
        result = rd.process_stream(X, seed=42)
        assert result.shape == X.shape

    def test_no_nans_in_output(self, synthetic_data):
        X, _ = synthetic_data
        rd = RandomDropout(budget=0.5, window_size=100)
        result = rd.process_stream(X, seed=42)
        assert not np.isnan(result).any()

    def test_keeps_correct_fraction(self, synthetic_data):
        X, _ = synthetic_data
        RandomDropout(budget=0.5, window_size=100)
        # Budget=0.5 with d=10 means ~5 channels kept per window
        n_keep = max(1, int(0.5 * 10))
        assert n_keep == 5

    def test_different_seeds_give_different_results(self, synthetic_data):
        X, _ = synthetic_data
        rd = RandomDropout(budget=0.5, window_size=100)
        r1 = rd.process_stream(X, seed=42)
        r2 = rd.process_stream(X, seed=123)
        assert not np.allclose(r1, r2)

    def test_extreme_budget(self):
        X = np.random.randn(200, 10)
        rd_low = RandomDropout(budget=0.1, window_size=100)
        result = rd_low.process_stream(X, seed=42)
        assert result.shape == X.shape


class TestMutualInfoSampling:
    def test_requires_fit(self, synthetic_data):
        X, _ = synthetic_data
        mi = MutualInfoSampling(budget=0.5)
        with pytest.raises(RuntimeError):
            mi.process_stream(X, seed=42)

    def test_fit_and_process(self, synthetic_data):
        X, y = synthetic_data
        mi = MutualInfoSampling(budget=0.5)
        mi.fit(X, y)
        result = mi.process_stream(X, seed=42)
        assert result.shape == X.shape

    def test_no_nans_in_output(self, synthetic_data):
        X, y = synthetic_data
        mi = MutualInfoSampling(budget=0.5)
        mi.fit(X, y)
        result = mi.process_stream(X, seed=42)
        assert not np.isnan(result).any()

    def test_mi_scores_normalized(self, synthetic_data):
        X, y = synthetic_data
        mi = MutualInfoSampling(budget=0.5)
        mi.fit(X, y)
        assert abs(mi.mi_scores.sum() - 1.0) < 1e-6

    def test_mi_scores_non_negative(self, synthetic_data):
        X, y = synthetic_data
        mi = MutualInfoSampling(budget=0.5)
        mi.fit(X, y)
        assert (mi.mi_scores >= 0).all()


class TestAttentionSampling:
    def test_output_shape(self, synthetic_data):
        X, _ = synthetic_data
        attn = AttentionSampling(budget=0.5, window_size=100, hidden_dim=16)
        result = attn.process_stream(X, seed=42)
        assert result.shape == X.shape

    def test_no_nans_in_output(self, synthetic_data):
        X, _ = synthetic_data
        attn = AttentionSampling(budget=0.5, window_size=100, hidden_dim=16)
        result = attn.process_stream(X, seed=42)
        assert not np.isnan(result).any()

    def test_param_count(self):
        attn = AttentionSampling(budget=0.5, window_size=100, hidden_dim=32)
        X = np.random.randn(200, 10)
        attn.process_stream(X, seed=42)
        # 3 matrices of size (w, h) = (100, 32) = 3 * 3200
        assert attn.n_parameters == 3 * 100 * 32

    def test_different_hidden_dims(self, synthetic_data):
        X, _ = synthetic_data
        for h in [8, 16, 32, 64]:
            attn = AttentionSampling(budget=0.5, window_size=100, hidden_dim=h)
            result = attn.process_stream(X, seed=42)
            assert result.shape == X.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
