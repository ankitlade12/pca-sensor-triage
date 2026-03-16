"""
Baseline 3: Variance-Based Sampling.

Allocates sampling rates proportional to rolling variance of each channel.
Channels with higher variance get more bandwidth.
This is the most natural non-PCA data-driven baseline.
"""

import numpy as np
import pandas as pd
from src.triage.rate_allocator import RateAllocator


class VarianceSampling:
    """Variance-based adaptive sampling.

    Parameters
    ----------
    budget : float
        Total bandwidth budget.
    window_size : int
        Window for computing variance.
    min_rate : float
        Minimum sampling rate.
    """

    def __init__(
        self,
        budget: float = 0.5,
        window_size: int = 100,
        min_rate: float = 0.05,
    ):
        self.budget = budget
        self.window_size = window_size
        self.min_rate = min_rate
        self.allocator = RateAllocator(budget=budget, min_rate=min_rate)

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        n, d = data.shape
        n_windows = n // self.window_size
        reconstructed = np.zeros_like(data, dtype=float)

        for w_idx in range(n_windows):
            start = w_idx * self.window_size
            end = start + self.window_size
            window = data[start:end]

            # Per-channel variance as importance score
            var_scores = np.var(window, axis=0)
            var_sum = var_scores.sum()
            if var_sum > 0:
                importance = var_scores / var_sum
            else:
                importance = np.ones(d) / d

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
