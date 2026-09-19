"""
Baseline 3: Variance-Based Sampling.

Allocates sampling rates proportional to rolling variance of each channel.
Channels with higher variance get more bandwidth.
This is the most natural non-PCA data-driven baseline.
"""

import numpy as np

from src.triage.rate_allocator import RateAllocator
from src.triage.reconstruction import reconstruct


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
        self._next_rates = None

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        n, d = data.shape
        n_windows = n // self.window_size
        reconstructed = np.zeros_like(data, dtype=float)
        rates = (
            self._next_rates.copy()
            if self._next_rates is not None
            else np.full(d, self.budget)
        )
        last_values = np.zeros(d, dtype=float)

        for w_idx in range(n_windows):
            start = w_idx * self.window_size
            end = start + self.window_size
            window = data[start:end]

            # Apply rates determined from the preceding window.
            triaged = self.allocator.apply_rates(window, rates, seed=seed + w_idx)
            recon = reconstruct(triaged, method="forward_fill", initial_values=last_values)
            reconstructed[start:end] = recon
            last_values = recon[-1].copy()

            # Update after acquisition for use by the next window.
            var_scores = np.var(recon, axis=0)
            var_sum = var_scores.sum()
            if var_sum > 0:
                importance = var_scores / var_sum
            else:
                importance = np.ones(d) / d

            rates = self.allocator.allocate(importance)
            self._next_rates = rates.copy()

        remaining_n = n % self.window_size
        if remaining_n > 0:
            start = n_windows * self.window_size
            triaged = self.allocator.apply_rates(data[start:], rates, seed=seed + n_windows)
            reconstructed[start:] = reconstruct(
                triaged, method="forward_fill", initial_values=last_values
            )

        return reconstructed
