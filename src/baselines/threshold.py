"""
Baseline 2: Threshold-Based Adaptive Sampling.

Channels whose recent absolute change exceeds a threshold get full rate.
Others get minimum rate. Inspired by Send-on-Delta approaches.
"""

import numpy as np

from src.triage.rate_allocator import RateAllocator
from src.triage.reconstruction import reconstruct


class ThresholdSampling:
    """Threshold-based adaptive sampling.

    Parameters
    ----------
    budget : float
        Total bandwidth budget.
    threshold_percentile : float
        Percentile of absolute changes to use as threshold.
        Channels above this threshold get high rate, others get min_rate.
    window_size : int
        Window for computing changes.
    min_rate : float
        Minimum sampling rate.
    """

    def __init__(
        self,
        budget: float = 0.5,
        threshold_percentile: float = 50.0,
        window_size: int = 100,
        min_rate: float = 0.05,
    ):
        self.budget = budget
        self.threshold_percentile = threshold_percentile
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

            triaged = self.allocator.apply_rates(window, rates, seed=seed + w_idx)
            recon = reconstruct(triaged, method="forward_fill", initial_values=last_values)
            reconstructed[start:end] = recon
            last_values = recon[-1].copy()

            # Compute the next rates from the causally reconstructed window.
            changes = np.mean(np.abs(np.diff(recon, axis=0)), axis=0)

            # Threshold: channels above percentile get high rate
            threshold = np.percentile(changes, self.threshold_percentile)
            active = changes >= threshold

            # Allocate: active channels share budget, others get min_rate
            rates = np.full(d, self.min_rate)
            n_active = active.sum()
            if n_active > 0:
                total_budget = self.budget * d
                remaining = total_budget - self.min_rate * d
                rates[active] += remaining / n_active

            rates = np.clip(rates, self.min_rate, 1.0)
            self._next_rates = rates.copy()

        # Handle tail
        remaining_n = n % self.window_size
        if remaining_n > 0:
            start = n_windows * self.window_size
            triaged = self.allocator.apply_rates(data[start:], rates, seed=seed + n_windows)
            reconstructed[start:] = reconstruct(
                triaged, method="forward_fill", initial_values=last_values
            )

        return reconstructed
