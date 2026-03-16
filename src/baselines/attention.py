"""
Baseline 6: Attention-Based Channel Importance.

Single-head self-attention over channels to compute importance weights.
This is the compute-expensive upper-bound comparison target.
"""

import numpy as np
import pandas as pd
from src.triage.rate_allocator import RateAllocator


class AttentionSampling:
    """Self-attention based channel importance scoring.

    Computes single-head self-attention over channels per window
    to derive importance weights for bandwidth allocation.

    Parameters
    ----------
    budget : float
        Total bandwidth budget.
    window_size : int
        Window size for attention computation.
    min_rate : float
        Minimum sampling rate.
    hidden_dim : int
        Dimension for Q, K, V projections.
    """

    def __init__(
        self,
        budget: float = 0.5,
        window_size: int = 100,
        min_rate: float = 0.05,
        hidden_dim: int = 32,
    ):
        self.budget = budget
        self.window_size = window_size
        self.min_rate = min_rate
        self.hidden_dim = hidden_dim
        self.allocator = RateAllocator(budget=budget, min_rate=min_rate)
        self._params_initialized = False
        self.W_q = None
        self.W_k = None
        self.W_v = None

    def _init_params(self, d):
        """Initialize random projection matrices."""
        rng = np.random.RandomState(42)
        h = self.hidden_dim
        self.W_q = rng.randn(d, h) / np.sqrt(h)
        self.W_k = rng.randn(d, h) / np.sqrt(h)
        self.W_v = rng.randn(d, h) / np.sqrt(h)
        self._params_initialized = True

    def _compute_attention_importance(self, window):
        """Compute channel importance via self-attention.

        Treats each channel as a "token" with its time-series as embedding.
        """
        w, d = window.shape

        if not self._params_initialized:
            self._init_params(w)

        # Channel embeddings: each channel's time series is its embedding
        # X shape: (d, w) — d channels, each with w time steps
        X = window.T  # (d, w)

        # Q, K, V projections: (d, w) @ (w, h) = (d, h)
        Q = X @ self.W_q  # (d, h)
        K = X @ self.W_k  # (d, h)
        V = X @ self.W_v  # (d, h)

        # Attention scores: (d, d)
        h = self.hidden_dim
        scores = Q @ K.T / np.sqrt(h)  # (d, d)

        # Softmax
        scores_exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        attention = scores_exp / scores_exp.sum(axis=1, keepdims=True)  # (d, d)

        # Channel importance = sum of attention weights received
        importance = attention.sum(axis=0)  # (d,)
        importance = importance / importance.sum()

        return importance

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        n, d = data.shape
        n_windows = n // self.window_size
        reconstructed = np.zeros_like(data, dtype=float)

        for w_idx in range(n_windows):
            start = w_idx * self.window_size
            end = start + self.window_size
            window = data[start:end]

            importance = self._compute_attention_importance(window)
            rates = self.allocator.allocate(importance)
            triaged = self.allocator.apply_rates(window, rates, seed=seed + w_idx)
            recon = pd.DataFrame(triaged).ffill().bfill().fillna(0.0).values
            reconstructed[start:end] = recon

        remaining_n = n % self.window_size
        if remaining_n > 0:
            start = n_windows * self.window_size
            rates_last = np.full(d, self.budget)
            triaged = self.allocator.apply_rates(data[start:], rates_last, seed=seed + n_windows)
            reconstructed[start:] = pd.DataFrame(triaged).ffill().bfill().fillna(0.0).values

        return reconstructed

    @property
    def n_parameters(self):
        if self._params_initialized:
            return 3 * self.W_q.size
        return 0
