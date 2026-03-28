"""
Baseline 7: Autoencoder-Based Channel Importance.

Trains a simple autoencoder on each window and uses per-channel
reconstruction error as importance: channels that are harder to
reconstruct from the bottleneck representation are more informative.

This is a learned, unsupervised baseline that uses neural networks
instead of PCA for dimensionality reduction.
"""

import numpy as np
import pandas as pd

from src.triage.rate_allocator import RateAllocator


class AutoencoderSampling:
    """Autoencoder-based adaptive sampling.

    Uses a simple 1-hidden-layer autoencoder trained per window.
    Importance = per-channel MSE reconstruction error (higher = more important).

    Parameters
    ----------
    budget : float
        Total bandwidth budget.
    window_size : int
        Window for training autoencoder.
    bottleneck : int
        Hidden layer size (analogous to PCA k).
    n_epochs : int
        Training epochs per window.
    lr : float
        Learning rate.
    min_rate : float
        Minimum sampling rate.
    """

    def __init__(
        self,
        budget: float = 0.5,
        window_size: int = 100,
        bottleneck: int = 10,
        n_epochs: int = 50,
        lr: float = 0.01,
        min_rate: float = 0.05,
    ):
        self.budget = budget
        self.window_size = window_size
        self.bottleneck = bottleneck
        self.n_epochs = n_epochs
        self.lr = lr
        self.min_rate = min_rate
        self.allocator = RateAllocator(budget=budget, min_rate=min_rate)

    def _train_and_score(self, window: np.ndarray) -> np.ndarray:
        """Train autoencoder on window and return per-channel importance."""
        n, d = window.shape

        # Simple autoencoder: d -> bottleneck -> d
        # Using numpy only (no torch dependency) for portability
        # Xavier initialization for stability
        rng = np.random.RandomState(42)
        scale1 = np.sqrt(2.0 / (d + self.bottleneck))
        scale2 = np.sqrt(2.0 / (self.bottleneck + d))
        W1 = rng.randn(d, self.bottleneck) * scale1
        b1 = np.zeros(self.bottleneck)
        W2 = rng.randn(self.bottleneck, d) * scale2
        b2 = np.zeros(d)

        # Normalize window
        mu = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True) + 1e-8
        X = (window - mu) / std

        for _ in range(self.n_epochs):
            # Forward
            pre_act = X @ W1 + b1
            pre_act = np.clip(pre_act, -10, 10)  # prevent overflow
            hidden = np.tanh(pre_act)             # (n, bottleneck)
            recon = hidden @ W2 + b2              # (n, d)

            # Loss gradient (MSE)
            diff = recon - X                      # (n, d)

            # Backward
            dW2 = hidden.T @ diff / n             # (bottleneck, d)
            db2 = diff.mean(axis=0)               # (d,)
            dhidden = diff @ W2.T                  # (n, bottleneck)
            dhidden *= (1 - hidden ** 2)          # tanh derivative
            dW1 = X.T @ dhidden / n               # (d, bottleneck)
            db1 = dhidden.mean(axis=0)            # (bottleneck,)

            # NaN guard + gradient clipping
            grads = [dW1, db1, dW2, db2]
            if any(np.any(np.isnan(g)) for g in grads):
                break  # stop training if NaN detected
            for g in grads:
                np.clip(g, -1.0, 1.0, out=g)

            # Update
            W1 -= self.lr * dW1
            b1 -= self.lr * db1
            W2 -= self.lr * dW2
            b2 -= self.lr * db2

        # Final reconstruction error per channel
        hidden = np.tanh(np.clip(X @ W1 + b1, -10, 10))
        recon = hidden @ W2 + b2
        per_channel_mse = np.mean((recon - X) ** 2, axis=0)  # (d,)

        # Importance = reconstruction error (harder to reconstruct = more unique info)
        importance = per_channel_mse
        imp_sum = importance.sum()
        if imp_sum > 0:
            importance = importance / imp_sum
        else:
            importance = np.ones(d) / d

        return importance

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        n, d = data.shape
        n_windows = n // self.window_size
        reconstructed = np.zeros_like(data, dtype=float)

        for w_idx in range(n_windows):
            start = w_idx * self.window_size
            end = start + self.window_size
            window = data[start:end]

            importance = self._train_and_score(window)
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
