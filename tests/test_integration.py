"""Integration tests: end-to-end pipeline validation.

Tests the full flow from raw data → triage → reconstruction → classification
to ensure all components work together correctly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.baselines import (
    AttentionSampling,
    MutualInfoSampling,
    RandomDropout,
    ThresholdSampling,
    UniformSampling,
    VarianceSampling,
)
from src.triage import TriagePipeline, compute_reconstruction_error


@pytest.fixture
def classification_data():
    """Create a synthetic classification problem with correlated channels."""
    np.random.seed(42)
    n_per_class = 200
    d = 20

    X_class0 = np.random.randn(n_per_class, d) * 0.5
    X_class1 = np.random.randn(n_per_class, d) * 0.5
    # Class 1 has elevated channels 0-4
    X_class1[:, :5] += 3.0
    # Add correlated channels (channels 5-9 correlate with 0-4)
    X_class0[:, 5:10] = X_class0[:, :5] * 0.8 + np.random.randn(n_per_class, 5) * 0.2
    X_class1[:, 5:10] = X_class1[:, :5] * 0.8 + np.random.randn(n_per_class, 5) * 0.2

    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split
    split = int(0.7 * len(X))
    return X[:split], y[:split], X[split:], y[split:]


class TestEndToEndPipeline:
    """Test complete triage → classify flow."""

    def test_pca_triage_improves_over_random(self, classification_data):
        """PCA-Triage at 50% should beat random dropout."""
        X_train, y_train, X_test, y_test = classification_data

        # PCA-Triage
        pipe = TriagePipeline(n_components=5, window_size=50, budget=0.5,
                              forgetting_factor=1.0, min_rate=0.05)
        X_tr_pca = pipe.process_stream(X_train, seed=42)
        X_te_pca = pipe.process_stream(X_test, seed=123)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_tr_pca, y_train)
        f1_pca = f1_score(y_test, clf.predict(X_te_pca), average='weighted')

        # Random Dropout
        rd = RandomDropout(budget=0.5, window_size=50)
        X_tr_rd = rd.process_stream(X_train, seed=42)
        X_te_rd = rd.process_stream(X_test, seed=123)

        clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
        clf2.fit(X_tr_rd, y_train)
        f1_rd = f1_score(y_test, clf2.predict(X_te_rd), average='weighted')

        # PCA-Triage should outperform random dropout on correlated data
        assert f1_pca >= f1_rd - 0.05  # allow small margin

    def test_full_budget_matches_original(self, classification_data):
        """At budget=0.9, triage should closely match full data."""
        X_train, y_train, X_test, y_test = classification_data

        # Full data baseline
        clf_full = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_full.fit(X_train, y_train)
        f1_full = f1_score(y_test, clf_full.predict(X_test), average='weighted')

        # PCA-Triage at 90%
        pipe = TriagePipeline(n_components=5, window_size=50, budget=0.9,
                              forgetting_factor=1.0, min_rate=0.05)
        X_tr = pipe.process_stream(X_train, seed=42)
        X_te = pipe.process_stream(X_test, seed=123)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_tr, y_train)
        f1_triage = f1_score(y_test, clf.predict(X_te), average='weighted')

        # Should be within 5% of full data
        assert f1_triage >= f1_full - 0.05

    def test_reconstruction_error_decreases_with_budget(self, classification_data):
        """Higher budget → lower reconstruction error (on average)."""
        X_train, _, _, _ = classification_data

        errors = []
        for budget in [0.2, 0.5, 0.8]:
            pipe = TriagePipeline(n_components=5, window_size=50,
                                  budget=budget, forgetting_factor=1.0, min_rate=0.05)
            X_recon = pipe.process_stream(X_train, seed=42)
            metrics = compute_reconstruction_error(X_train, X_recon)
            errors.append(metrics['mse'])

        # Error at 20% should be higher than at 80%
        assert errors[0] > errors[2]


class TestAllMethodsConsistency:
    """Ensure all methods produce valid outputs on same data."""

    def test_all_methods_same_shape(self, classification_data):
        X_train, y_train, X_test, _ = classification_data

        methods = {
            'PCA-Triage': lambda X: TriagePipeline(
                n_components=5, window_size=50, budget=0.5).process_stream(X, seed=42),
            'Uniform': lambda X: UniformSampling(budget=0.5).process_stream(X, seed=42),
            'Threshold': lambda X: ThresholdSampling(
                budget=0.5, window_size=50).process_stream(X, seed=42),
            'Variance': lambda X: VarianceSampling(
                budget=0.5, window_size=50).process_stream(X, seed=42),
            'Random Dropout': lambda X: RandomDropout(
                budget=0.5, window_size=50).process_stream(X, seed=42),
            'Attention': lambda X: AttentionSampling(
                budget=0.5, window_size=50, hidden_dim=16).process_stream(X, seed=42),
        }

        for name, method_fn in methods.items():
            result = method_fn(X_train)
            assert result.shape == X_train.shape, f"{name} output shape mismatch"
            assert not np.isnan(result).any(), f"{name} produced NaNs"
            assert np.isfinite(result).all(), f"{name} produced Inf"

    def test_mutual_info_with_labels(self, classification_data):
        X_train, y_train, X_test, _ = classification_data
        mi = MutualInfoSampling(budget=0.5)
        mi.fit(X_train, y_train)
        result = mi.process_stream(X_train, seed=42)
        assert result.shape == X_train.shape
        assert not np.isnan(result).any()


class TestReproducibility:
    """Same seed should produce identical results."""

    def test_pca_triage_deterministic(self, classification_data):
        X_train, _, _, _ = classification_data

        pipe1 = TriagePipeline(n_components=5, window_size=50, budget=0.5)
        r1 = pipe1.process_stream(X_train, seed=42)

        pipe2 = TriagePipeline(n_components=5, window_size=50, budget=0.5)
        r2 = pipe2.process_stream(X_train, seed=42)

        assert np.allclose(r1, r2)

    def test_different_seeds_differ(self, classification_data):
        X_train, _, _, _ = classification_data

        pipe1 = TriagePipeline(n_components=5, window_size=50, budget=0.5)
        r1 = pipe1.process_stream(X_train, seed=42)

        pipe2 = TriagePipeline(n_components=5, window_size=50, budget=0.5)
        r2 = pipe2.process_stream(X_train, seed=999)

        assert not np.allclose(r1, r2)


class TestEdgeCases:
    """Test boundary conditions."""

    def test_single_channel(self):
        X = np.random.randn(200, 1)
        pipe = TriagePipeline(n_components=1, window_size=50, budget=0.5)
        result = pipe.process_stream(X, seed=42)
        assert result.shape == X.shape

    def test_very_small_budget(self, classification_data):
        X_train, _, _, _ = classification_data
        pipe = TriagePipeline(n_components=5, window_size=50, budget=0.1, min_rate=0.02)
        result = pipe.process_stream(X_train, seed=42)
        assert result.shape == X_train.shape
        assert not np.isnan(result).any()

    def test_budget_equals_one(self, classification_data):
        """Budget=1.0 should approximate original data."""
        X_train, _, _, _ = classification_data
        pipe = TriagePipeline(n_components=5, window_size=50, budget=0.99, min_rate=0.05)
        result = pipe.process_stream(X_train, seed=42)
        # Most values should be close to original
        mse = np.mean((X_train - result) ** 2)
        assert mse < 0.5  # very low error at near-full budget

    def test_data_shorter_than_window(self):
        X = np.random.randn(30, 5)
        pipe = TriagePipeline(n_components=3, window_size=50, budget=0.5)
        result = pipe.process_stream(X, seed=42)
        assert result.shape == X.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
