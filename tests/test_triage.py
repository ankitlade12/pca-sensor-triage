"""Unit tests for PCA-Triage core modules."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.triage import PCATriage, HybridScorer, RateAllocator, reconstruct, compute_reconstruction_error, TriagePipeline


class TestPCATriage:
    def test_importance_scores_sum_to_one(self):
        triage = PCATriage(n_components=3, window_size=50)
        window = np.random.randn(50, 10)
        scores = triage.compute_importance(window)
        assert scores.shape == (10,)
        assert abs(scores.sum() - 1.0) < 1e-6

    def test_importance_scores_non_negative(self):
        triage = PCATriage(n_components=3, window_size=50)
        window = np.random.randn(50, 10)
        scores = triage.compute_importance(window)
        assert (scores >= 0).all()

    def test_multiple_windows_update(self):
        triage = PCATriage(n_components=3, window_size=50, forgetting_factor=0.9)
        for _ in range(5):
            window = np.random.randn(50, 10)
            scores = triage.compute_importance(window)
        assert scores.shape == (10,)
        assert abs(scores.sum() - 1.0) < 1e-6

    def test_high_variance_channel_gets_high_importance(self):
        triage = PCATriage(n_components=3, window_size=100)
        window = np.random.randn(100, 10) * 0.1
        window[:, 0] = np.random.randn(100) * 10  # channel 0 has 100x variance
        scores = triage.compute_importance(window)
        assert scores[0] == scores.max()

    def test_reset(self):
        triage = PCATriage(n_components=3, window_size=50)
        window = np.random.randn(50, 10)
        triage.compute_importance(window)
        assert triage._fitted
        triage.reset()
        assert not triage._fitted


class TestRateAllocator:
    def test_rates_respect_budget(self):
        alloc = RateAllocator(budget=0.5, min_rate=0.05)
        scores = np.random.dirichlet(np.ones(10))
        rates = alloc.allocate(scores)
        assert rates.shape == (10,)
        assert rates.mean() <= 0.5 + 1e-6

    def test_rates_above_minimum(self):
        alloc = RateAllocator(budget=0.5, min_rate=0.05)
        scores = np.random.dirichlet(np.ones(10))
        rates = alloc.allocate(scores)
        assert (rates >= 0.05 - 1e-6).all()

    def test_rates_below_one(self):
        alloc = RateAllocator(budget=0.9, min_rate=0.05)
        scores = np.zeros(10)
        scores[0] = 1.0  # all importance to one channel
        rates = alloc.allocate(scores)
        assert (rates <= 1.0 + 1e-6).all()

    def test_uniform_importance_gives_uniform_rates(self):
        alloc = RateAllocator(budget=0.5, min_rate=0.05)
        scores = np.ones(10) / 10
        rates = alloc.allocate(scores)
        assert np.allclose(rates, rates[0], atol=1e-6)

    def test_apply_rates_creates_nans(self):
        alloc = RateAllocator(budget=0.5, min_rate=0.05)
        data = np.ones((100, 5))
        rates = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        triaged = alloc.apply_rates(data, rates, seed=42)
        # Low-rate channels should have more NaNs
        nan_fracs = np.isnan(triaged).mean(axis=0)
        assert nan_fracs[0] > nan_fracs[4]

    def test_effective_bandwidth(self):
        alloc = RateAllocator(budget=0.5, min_rate=0.05)
        rates = np.array([0.3, 0.5, 0.7])
        assert abs(alloc.get_effective_bandwidth(rates) - 0.5) < 1e-6


class TestReconstruction:
    def test_forward_fill(self):
        data = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, np.nan]])
        recon = reconstruct(data, method="forward_fill")
        assert not np.isnan(recon).any()
        assert recon[1, 0] == 1.0  # forward filled

    def test_zero_fill(self):
        data = np.array([[1.0, np.nan], [np.nan, 2.0]])
        recon = reconstruct(data, method="zero")
        assert recon[0, 1] == 0.0
        assert recon[1, 0] == 0.0

    def test_reconstruction_error_metrics(self):
        original = np.array([[1.0, 2.0], [3.0, 4.0]])
        recon = np.array([[1.1, 2.0], [3.0, 3.8]])
        metrics = compute_reconstruction_error(original, recon)
        assert metrics["mse"] > 0
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert len(metrics["nrmse_per_channel"]) == 2


class TestTriagePipeline:
    def test_pipeline_output_shape(self):
        pipe = TriagePipeline(n_components=3, window_size=25, budget=0.5)
        data = np.random.randn(100, 10)
        recon = pipe.process_stream(data, seed=42)
        assert recon.shape == data.shape

    def test_pipeline_logs(self):
        pipe = TriagePipeline(n_components=3, window_size=25, budget=0.5)
        data = np.random.randn(100, 10)
        pipe.process_stream(data, seed=42)
        assert len(pipe.importance_log) == 4  # 100 / 25 = 4 windows
        assert len(pipe.rate_log) == 4
        assert pipe.get_importance_matrix().shape == (4, 10)

    def test_pipeline_reset(self):
        pipe = TriagePipeline(n_components=3, window_size=25, budget=0.5)
        data = np.random.randn(100, 10)
        pipe.process_stream(data, seed=42)
        pipe.reset()
        assert len(pipe.importance_log) == 0


class TestHybridScorer:
    def test_scores_sum_to_one(self):
        scorer = HybridScorer(n_components=3, alpha=0.7)
        window = np.random.randn(50, 10)
        scores = scorer.compute_importance(window)
        assert scores.shape == (10,)
        assert abs(scores.sum() - 1.0) < 1e-6

    def test_pure_variance_mode(self):
        scorer = HybridScorer(n_components=3, alpha=0.0)
        window = np.random.randn(50, 10) * 0.1
        window[:, 0] = np.random.randn(50) * 10
        scores = scorer.compute_importance(window)
        assert scores[0] == scores.max()

    def test_pure_pca_mode(self):
        scorer = HybridScorer(n_components=3, alpha=1.0)
        window = np.random.randn(50, 10)
        scores = scorer.compute_importance(window)
        assert scores.shape == (10,)
        assert abs(scores.sum() - 1.0) < 1e-6

    def test_smoothing_over_windows(self):
        scorer = HybridScorer(n_components=3, alpha=0.5, forgetting_factor=0.9)
        for _ in range(5):
            window = np.random.randn(50, 10)
            scores = scorer.compute_importance(window)
        assert abs(scores.sum() - 1.0) < 1e-6

    def test_reset(self):
        scorer = HybridScorer(n_components=3, alpha=0.7)
        scorer.compute_importance(np.random.randn(50, 10))
        scorer.reset()
        assert not scorer._fitted
        assert scorer._importance_history is None


class TestSharpening:
    def test_sharpness_concentrates_rates(self):
        # Use 20 channels so no single channel hits the 1.0 cap
        scores = np.random.dirichlet(np.ones(20))
        alloc_flat = RateAllocator(budget=0.5, min_rate=0.01, sharpness=1.0)
        alloc_sharp = RateAllocator(budget=0.5, min_rate=0.01, sharpness=3.0)
        rates_flat = alloc_flat.allocate(scores)
        rates_sharp = alloc_sharp.allocate(scores)
        top = scores.argmax()
        bot = scores.argmin()
        # Sharpened should give more to top channel, less to bottom
        assert rates_sharp[top] >= rates_flat[top]
        assert rates_sharp[bot] <= rates_flat[bot]

    def test_sharpness_one_is_identity(self):
        alloc = RateAllocator(budget=0.5, min_rate=0.05, sharpness=1.0)
        scores = np.random.dirichlet(np.ones(10))
        rates = alloc.allocate(scores)
        assert rates.mean() <= 0.5 + 1e-6


class TestPipelineV2:
    def test_hybrid_pipeline(self):
        pipe = TriagePipeline(
            n_components=3, window_size=25, budget=0.5,
            scorer='hybrid', alpha=0.7,
        )
        data = np.random.randn(100, 10)
        recon = pipe.process_stream(data, seed=42)
        assert recon.shape == data.shape
        assert not np.isnan(recon).any()

    def test_linear_interpolation_default(self):
        pipe = TriagePipeline(
            n_components=3, window_size=25, budget=0.5,
        )
        assert pipe.reconstruction_method == 'linear'

    def test_sharpened_pipeline(self):
        pipe = TriagePipeline(
            n_components=3, window_size=25, budget=0.5,
            sharpness=2.0,
        )
        data = np.random.randn(100, 10)
        recon = pipe.process_stream(data, seed=42)
        assert recon.shape == data.shape
        assert not np.isnan(recon).any()

    def test_linear_recon(self):
        data = np.array([[1.0, np.nan, 3.0], [np.nan, 2.0, np.nan]])
        recon = reconstruct(data.T, method="linear")
        assert not np.isnan(recon).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
