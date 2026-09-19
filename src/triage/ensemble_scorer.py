"""
Ensemble Scorer: Blends importance scores from multiple PCA configurations.

Runs PCA with different k values and averages the resulting importance
scores. This produces more robust channel rankings that are less
sensitive to the choice of k.
"""

from typing import List, Optional

import numpy as np
from sklearn.decomposition import IncrementalPCA


class EnsembleScorer:
    """Ensemble of PCA scorers with different component counts.

    Parameters
    ----------
    k_values : list of int
        Component counts to ensemble (e.g., [3, 5, 10, 15]).
    weights : list of float or None
        Per-k weights for averaging. None = equal weight.
    forgetting_factor : float
        Exponential smoothing for blended importance scores.
    """

    def __init__(
        self,
        k_values: List[int] = None,
        weights: List[float] = None,
        forgetting_factor: float = 1.0,
    ):
        if not 0 <= forgetting_factor <= 1:
            raise ValueError("forgetting_factor must be in [0, 1]")
        if k_values is None:
            k_values = [3, 5, 10]
        self.k_values = k_values
        self.weights = weights or [1.0 / len(k_values)] * len(k_values)
        self.forgetting_factor = forgetting_factor

        self._ipcas = [IncrementalPCA(n_components=k) for k in k_values]
        self._fitted = [False] * len(k_values)
        self._importance_history: Optional[np.ndarray] = None
        self._n_score_updates = 0

    def compute_importance(self, window: np.ndarray) -> np.ndarray:
        """Compute ensemble importance scores."""
        w, d = window.shape
        ensemble_scores = np.zeros(d)

        for idx, (ipca, k, weight) in enumerate(
            zip(self._ipcas, self.k_values, self.weights)
        ):
            # Skip if k > min(w, d)
            if k > min(w, d):
                ensemble_scores += weight * np.ones(d) / d
                continue

            if not self._fitted[idx]:
                ipca.fit(window)
                self._fitted[idx] = True
            else:
                ipca.partial_fit(window)

            V = ipca.components_
            eigenvalues = ipca.explained_variance_

            scores = np.zeros(d)
            for i in range(len(eigenvalues)):
                scores += eigenvalues[i] * V[i, :] ** 2

            s_sum = scores.sum()
            if s_sum > 0:
                scores = scores / s_sum
            else:
                scores = np.ones(d) / d

            ensemble_scores += weight * scores

        # Normalize
        ensemble_scores = ensemble_scores / ensemble_scores.sum()

        # Exponential smoothing
        if self._importance_history is None:
            self._importance_history = ensemble_scores.copy()
            self._n_score_updates = 1
        else:
            lam = self.forgetting_factor
            if lam == 1.0:
                n = self._n_score_updates
                self._importance_history = (
                    n * self._importance_history + ensemble_scores
                ) / (n + 1)
            else:
                self._importance_history = (
                    lam * self._importance_history + (1 - lam) * ensemble_scores
                )
            self._n_score_updates += 1

        smoothed = self._importance_history / self._importance_history.sum()
        return smoothed

    def reset(self):
        self._ipcas = [IncrementalPCA(n_components=k) for k in self.k_values]
        self._fitted = [False] * len(self.k_values)
        self._importance_history = None
        self._n_score_updates = 0
