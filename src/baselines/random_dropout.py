"""
Baseline 4: Random Channel Dropout.

Randomly drops X% of channels per window (full dropout, not rate reduction).
Remaining channels get full sampling rate.
"""

import numpy as np
import pandas as pd


class RandomDropout:
    """Random channel dropout baseline.

    Parameters
    ----------
    budget : float
        Fraction of channels to keep per window.
    window_size : int
        Window size for dropout decisions.
    """

    def __init__(self, budget: float = 0.5, window_size: int = 100):
        self.budget = budget
        self.window_size = window_size

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        rng = np.random.RandomState(seed)
        n, d = data.shape
        n_keep = max(1, int(self.budget * d))
        n_windows = n // self.window_size
        reconstructed = np.zeros_like(data, dtype=float)

        for w_idx in range(n_windows):
            start = w_idx * self.window_size
            end = start + self.window_size
            window = data[start:end].copy()

            # Randomly select channels to keep
            keep_idx = rng.choice(d, size=n_keep, replace=False)
            mask = np.zeros(d, dtype=bool)
            mask[keep_idx] = True

            # Zero out dropped channels, then fill with column mean
            result = window.copy()
            result[:, ~mask] = np.nan
            result = pd.DataFrame(result).ffill().bfill().values
            # For fully NaN columns, use 0
            result = np.nan_to_num(result, nan=0.0)
            reconstructed[start:end] = result

        remaining_n = n % self.window_size
        if remaining_n > 0:
            start = n_windows * self.window_size
            keep_idx = rng.choice(d, size=n_keep, replace=False)
            mask = np.zeros(d, dtype=bool)
            mask[keep_idx] = True
            result = data[start:].copy().astype(float)
            result[:, ~mask] = np.nan
            result = pd.DataFrame(result).ffill().bfill().values
            result = np.nan_to_num(result, nan=0.0)
            reconstructed[start:] = result

        return reconstructed
