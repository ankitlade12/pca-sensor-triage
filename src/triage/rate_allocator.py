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
        # Counts come from the actual acquisition mask, not the requested rate.
        self.observation_log = []

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

        while excess.sum() > 1e-12:
            # Redistribute excess to channels below 1.0
            below_max = rates < 1.0
            if below_max.any():
                redistribute = excess.sum()
                sub_scores = scores[below_max]
                sub_scores = sub_scores / sub_scores.sum() if sub_scores.sum() > 0 else np.ones(below_max.sum()) / below_max.sum()
                rates[below_max] += sub_scores * redistribute
                excess = np.maximum(rates - 1.0, 0)
                rates = np.minimum(rates, 1.0)
            else:
                break

        return rates

    def apply_rates(
        self,
        data: np.ndarray,
        rates: np.ndarray,
        seed: int = 42,
        hard_budget: bool = True,
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
        hard_budget : bool
            If true, allocate integer channel quotas whose total equals the
            window budget. If false, use independent Bernoulli sampling and
            enforce the budget only in expectation.

        Returns
        -------
        triaged : np.ndarray, shape (n, d)
            Data with sub-sampled channels (NaN for dropped samples).
        """
        rng = np.random.RandomState(seed)
        n, d = data.shape
        if not np.isfinite(data).all():
            raise ValueError("Acquisition input must contain only finite measurements")
        if not np.isfinite(rates).all():
            raise ValueError("Sampling rates must be finite")
        if hard_budget:
            # Convert expected rates into integer per-channel quotas while
            # enforcing the aggregate window budget exactly (up to flooring).
            target_total = min(n * d, int(np.floor(self.budget * n * d)))
            expected = np.clip(rates, 0.0, 1.0) * n
            quotas = np.floor(expected).astype(int)
            quotas = np.minimum(quotas, n)

            remainder = target_total - int(quotas.sum())
            if remainder > 0:
                fractions = expected - quotas
                order = np.argsort(-fractions)
                while remainder > 0:
                    eligible = order[quotas[order] < n]
                    if len(eligible) == 0:
                        break
                    take = eligible[:remainder]
                    quotas[take] += 1
                    remainder -= len(take)
            elif remainder < 0:
                fractions = expected - quotas
                order = np.argsort(fractions)
                while remainder < 0:
                    eligible = order[quotas[order] > 0]
                    if len(eligible) == 0:
                        break
                    take = eligible[: min(-remainder, len(eligible))]
                    quotas[take] -= 1
                    remainder += len(take)

            mask = np.zeros((n, d), dtype=bool)
            for j, quota in enumerate(quotas):
                if quota:
                    keep = rng.choice(n, size=quota, replace=False)
                    mask[keep, j] = True
        else:
            mask = rng.random((n, d)) < rates[np.newaxis, :]
        transmitted = int(mask.sum())
        allowed = int(np.floor(self.budget * n * d))
        self.observation_log.append({
            "window_samples": n,
            "channels": d,
            "available_values": int(mask.size),
            "transmitted_values": transmitted,
            "allowed_values": allowed,
            "excess_values": max(0, transmitted - allowed),
        })
        triaged = data.copy().astype(float)
        triaged[~mask] = np.nan
        return triaged

    def get_effective_bandwidth(self, rates: np.ndarray) -> float:
        """Compute actual bandwidth usage as fraction of full rate."""
        return rates.mean()
