"""
Rate Allocator: Converts importance scores to per-channel sampling rates.

Given importance scores and a bandwidth budget B, allocates per-channel
sampling rates proportional to importance, subject to:
  - sum(rates) <= B * d  (total budget constraint)
  - rates[j] >= min_rate  (minimum rate floor)
  - rates[j] <= 1.0       (maximum is full rate)
"""

import numpy as np


class RateAllocator:
    """Allocate per-channel sampling rates from importance scores.

    Parameters
    ----------
    budget : float
        Bandwidth budget as fraction of uniform full-rate (0, 1].
        E.g., 0.5 means total bandwidth = 50% of sampling all channels at 1.0.
    min_rate : float
        Minimum sampling rate for any channel.
    sharpness : float
        Power-law exponent for importance sharpening (default 1.0 = linear).
        Values > 1 concentrate more bandwidth on top channels.
    """

    def __init__(self, budget: float = 0.5, min_rate: float = 0.05,
                 sharpness: float = 1.0):
        if not 0 < budget <= 1:
            raise ValueError(f"budget must be in (0, 1], got {budget}")
        if not 0 <= min_rate < budget:
            raise ValueError(f"min_rate must be in [0, budget), got {min_rate}")

        self.budget = budget
        self.min_rate = min_rate
        self.sharpness = sharpness

    def allocate(self, importance_scores: np.ndarray) -> np.ndarray:
        """Allocate sampling rates proportional to importance.

        Parameters
        ----------
        importance_scores : np.ndarray, shape (d,)
            Per-channel importance scores (should sum to ~1).

        Returns
        -------
        rates : np.ndarray, shape (d,)
            Per-channel sampling rates in [min_rate, 1.0],
            satisfying sum(rates) <= budget * d.
        """
        d = len(importance_scores)
        total_budget = self.budget * d

        # Start with minimum rate floor for all channels
        rates = np.full(d, self.min_rate)
        remaining_budget = total_budget - self.min_rate * d

        if remaining_budget <= 0:
            # Budget is too small even for minimum rates — distribute equally
            return np.full(d, total_budget / d)

        # Normalize importance scores
        scores = importance_scores.copy()
        scores = np.maximum(scores, 0)  # ensure non-negative
        score_sum = scores.sum()
        if score_sum > 0:
            scores = scores / score_sum
        else:
            scores = np.ones(d) / d

        # Apply sharpening: concentrate budget on top channels
        if self.sharpness != 1.0:
            scores = scores ** self.sharpness
            s_sum = scores.sum()
            if s_sum > 0:
                scores = scores / s_sum

        # Allocate remaining budget proportionally to importance
        additional = scores * remaining_budget
        rates += additional

        # Clip to [min_rate, 1.0] and redistribute excess
        excess = np.maximum(rates - 1.0, 0)
        rates = np.minimum(rates, 1.0)

        if excess.sum() > 0:
            # Redistribute excess to channels below 1.0
            below_max = rates < 1.0
            if below_max.any():
                redistribute = excess.sum()
                sub_scores = scores[below_max]
                sub_scores = sub_scores / sub_scores.sum() if sub_scores.sum() > 0 else np.ones(below_max.sum()) / below_max.sum()
                rates[below_max] += sub_scores * redistribute
                rates = np.minimum(rates, 1.0)

        return rates

    def apply_rates(
        self, data: np.ndarray, rates: np.ndarray, seed: int = 42
    ) -> np.ndarray:
        """Apply sampling rates to data by masking samples.

        For each channel j, independently keep each sample with
        probability rates[j]. Masked samples are set to NaN.

        Parameters
        ----------
        data : np.ndarray, shape (n, d)
            Full-rate sensor data.
        rates : np.ndarray, shape (d,)
            Per-channel sampling rates.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        triaged : np.ndarray, shape (n, d)
            Data with sub-sampled channels (NaN for dropped samples).
        """
        rng = np.random.RandomState(seed)
        n, d = data.shape
        mask = rng.random((n, d)) < rates[np.newaxis, :]
        triaged = data.copy().astype(float)
        triaged[~mask] = np.nan
        return triaged

    def get_effective_bandwidth(self, rates: np.ndarray) -> float:
        """Compute actual bandwidth usage as fraction of full rate."""
        return rates.mean()
