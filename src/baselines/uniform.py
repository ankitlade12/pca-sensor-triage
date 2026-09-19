"""
Baseline 1: Uniform Sampling.

Every channel is sampled at the same reduced rate = budget.
The simplest possible baseline — no intelligence in allocation.
"""

import numpy as np

from src.triage.rate_allocator import RateAllocator
from src.triage.reconstruction import reconstruct


class UniformSampling:
    """Uniform sampling baseline — same rate for all channels."""

    def __init__(
        self, budget: float = 0.5, min_rate: float = 0.05, window_size: int = 50
    ):
        self.budget = budget
        self.window_size = window_size
        self.allocator = RateAllocator(budget=budget, min_rate=min_rate)

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        n, d = data.shape
        rates = np.full(d, self.budget)
        reconstructed = np.zeros_like(data, dtype=float)
        last_values = np.zeros(d, dtype=float)
        for w_idx, start in enumerate(range(0, n, self.window_size)):
            end = min(start + self.window_size, n)
            triaged = self.allocator.apply_rates(
                data[start:end], rates, seed=seed + w_idx, hard_budget=True
            )
            recon = reconstruct(triaged, method="forward_fill", initial_values=last_values)
            reconstructed[start:end] = recon
            if len(recon):
                last_values = recon[-1].copy()
        return reconstructed

    def get_rates(self, d: int) -> np.ndarray:
        return np.full(d, self.budget)
