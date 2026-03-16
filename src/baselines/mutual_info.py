"""
Baseline 5: Mutual Information-Based Sampling.

Uses mutual information between each channel and the fault label
to allocate sampling rates. Channels with higher MI get more bandwidth.

Note: This baseline requires labels, making it supervised —
unlike our unsupervised PCA-Triage method.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from src.triage.rate_allocator import RateAllocator


class MutualInfoSampling:
    """Mutual information-based adaptive sampling.

    Uses MI between each sensor and fault labels to rank channels.
    This is an information-theoretic baseline — supervised, batch.

    Parameters
    ----------
    budget : float
        Total bandwidth budget.
    min_rate : float
        Minimum sampling rate.
    """

    def __init__(self, budget: float = 0.5, min_rate: float = 0.05):
        self.budget = budget
        self.min_rate = min_rate
        self.allocator = RateAllocator(budget=budget, min_rate=min_rate)
        self._mi_scores = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_neighbors: int = 5):
        """Compute MI scores from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Training sensor data.
        y : np.ndarray, shape (n,)
            Fault labels.
        """
        self._mi_scores = mutual_info_classif(
            X, y, n_neighbors=n_neighbors, random_state=42
        )
        # Normalize
        mi_sum = self._mi_scores.sum()
        if mi_sum > 0:
            self._mi_scores = self._mi_scores / mi_sum
        else:
            self._mi_scores = np.ones(X.shape[1]) / X.shape[1]

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        if self._mi_scores is None:
            raise RuntimeError("Must call fit() first with labeled data.")

        rates = self.allocator.allocate(self._mi_scores)
        triaged = self.allocator.apply_rates(data, rates, seed=seed)
        recon = pd.DataFrame(triaged).ffill().bfill().values
        return recon

    @property
    def mi_scores(self):
        return self._mi_scores
