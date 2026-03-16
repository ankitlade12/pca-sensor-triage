"""
Baseline 1: Uniform Sampling.

Every channel is sampled at the same reduced rate = budget.
The simplest possible baseline — no intelligence in allocation.
"""

import numpy as np
from src.triage.rate_allocator import RateAllocator


class UniformSampling:
    """Uniform sampling baseline — same rate for all channels."""

    def __init__(self, budget: float = 0.5, min_rate: float = 0.05):
        self.budget = budget
        self.allocator = RateAllocator(budget=budget, min_rate=min_rate)

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        n, d = data.shape
        rates = np.full(d, self.budget)
        triaged = self.allocator.apply_rates(data, rates, seed=seed)

        # Reconstruct via forward fill
        import pandas as pd
        recon = pd.DataFrame(triaged).ffill().bfill().fillna(0.0).values
        return recon

    def get_rates(self, d: int) -> np.ndarray:
        return np.full(d, self.budget)
