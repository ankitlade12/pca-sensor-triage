"""
Baseline 10: Online Gradient Descent (OGD) Rate Allocation.

A regret-optimal online allocation baseline that updates per-channel
rates via gradient steps on reconstruction error. Serves as an
upper bound for what an online learning algorithm can achieve.
"""

import numpy as np
import pandas as pd


class OGDAllocator:
    """Online Gradient Descent rate allocation.

    Parameters
    ----------
    budget : float
        Total bandwidth budget.
    lr : float
        Learning rate for gradient updates.
    min_rate : float
        Minimum per-channel rate.
    """

    def __init__(self, budget=0.5, lr=0.01, min_rate=0.05):
        self.budget = budget
        self.lr = lr
        self.min_rate = min_rate
        self._rates = None

    def process_stream(self, data, seed=42):
        n, d = data.shape
        window_size = 50
        n_windows = n // window_size
        result = np.zeros_like(data, dtype=float)

        if self._rates is None:
            self._rates = np.full(d, self.budget)

        rng = np.random.RandomState(seed)

        for w_idx in range(n_windows):
            start = w_idx * window_size
            end = start + window_size
            window = data[start:end]

            mask = rng.random((window_size, d)) < self._rates[np.newaxis, :]
            triaged = window.copy().astype(float)
            triaged[~mask] = np.nan
            recon = pd.DataFrame(triaged).interpolate(
                method='linear', axis=0, limit_direction='both'
            ).fillna(0.0).values
            result[start:end] = recon

            recon_error = np.mean((window - recon) ** 2, axis=0)
            gradient = recon_error / (recon_error.sum() + 1e-10)
            self._rates = self._rates + self.lr * gradient

            self._rates = np.clip(self._rates, self.min_rate, 1.0)
            excess = self._rates.mean() - self.budget
            if excess > 0:
                self._rates = self._rates - excess
            self._rates = np.clip(self._rates, self.min_rate, 1.0)

        remaining = n % window_size
        if remaining > 0:
            start = n_windows * window_size
            mask = rng.random((remaining, d)) < self._rates[np.newaxis, :]
            triaged = data[start:].copy().astype(float)
            triaged[~mask] = np.nan
            result[start:] = pd.DataFrame(triaged).interpolate(
                method='linear', axis=0, limit_direction='both'
            ).fillna(0.0).values

        return result
