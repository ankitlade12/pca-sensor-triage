"""
Baseline 2: Threshold-Based Adaptive Sampling.

Channels whose recent absolute change exceeds a threshold get full rate.
Others get minimum rate. Inspired by Send-on-Delta approaches.
"""

import numpy as np
import pandas as pd
from src.triage.rate_allocator import RateAllocator


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

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        n, d = data.shape
        n_windows = n // self.window_size
        reconstructed = np.zeros_like(data, dtype=float)

        for w_idx in range(n_windows):
            start = w_idx * self.window_size
            end = start + self.window_size
            window = data[start:end]

            # Compute per-channel mean absolute change
            changes = np.mean(np.abs(np.diff(window, axis=0)), axis=0)

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

            triaged = self.allocator.apply_rates(window, rates, seed=seed + w_idx)
            recon = pd.DataFrame(triaged).ffill().bfill().values
            reconstructed[start:end] = recon

        # Handle tail
        remaining_n = n % self.window_size
        if remaining_n > 0:
            start = n_windows * self.window_size
            rates_last = np.full(d, self.budget)
            triaged = self.allocator.apply_rates(data[start:], rates_last, seed=seed + n_windows)
            reconstructed[start:] = pd.DataFrame(triaged).ffill().bfill().values

        return reconstructed
