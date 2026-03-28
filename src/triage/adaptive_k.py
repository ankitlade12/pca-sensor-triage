"""
Adaptive k selection: Automatically choose the number of PCA components
based on cumulative explained variance threshold.

Instead of a fixed k, fits PCA with k_max components and retains only
those needed to explain `variance_threshold` fraction of variance.
This adapts to the dataset's intrinsic dimensionality.
"""

from typing import Optional

import numpy as np
from sklearn.decomposition import IncrementalPCA


class AdaptiveKPCATriage:
    """PCA-Triage with automatic component selection.

    Parameters
    ----------
    k_max : int
        Maximum number of components to consider.
    variance_threshold : float
        Cumulative explained variance ratio to retain (e.g., 0.95 = 95%).
    window_size : int
        Samples per sliding window.
    forgetting_factor : float
        Exponential smoothing for importance scores.
    """

    def __init__(
        self,
        k_max: int = 20,
        variance_threshold: float = 0.95,
        window_size: int = 50,
        forgetting_factor: float = 1.0,
    ):
        self.k_max = k_max
        self.variance_threshold = variance_threshold
        self.window_size = window_size
        self.forgetting_factor = forgetting_factor

        self.ipca = IncrementalPCA(n_components=k_max)
        self._fitted = False
        self._importance_history: Optional[np.ndarray] = None
        self._effective_k: int = k_max

    def compute_importance(self, window: np.ndarray) -> np.ndarray:
        """Compute importance with adaptive k selection."""
        w, d = window.shape

        # Ensure k_max doesn't exceed min(w, d)
        actual_k = min(self.k_max, w, d)
        if actual_k != self.ipca.n_components:
            self.ipca = IncrementalPCA(n_components=actual_k)
            self._fitted = False

        if not self._fitted:
            self.ipca.fit(window)
            self._fitted = True
        else:
            self.ipca.partial_fit(window)

        V = self.ipca.components_  # (k, d)
        sigma = self.ipca.singular_values_  # (k,)

        # Determine effective k from cumulative variance
        evr = self.ipca.explained_variance_ratio_
        cumvar = np.cumsum(evr)
        k_eff = np.searchsorted(cumvar, self.variance_threshold) + 1
        k_eff = max(1, min(k_eff, len(sigma)))
        self._effective_k = k_eff

        # Compute importance using only the first k_eff components
        scores = np.zeros(d)
        for i in range(k_eff):
            scores += sigma[i] * V[i, :] ** 2

        scores_sum = scores.sum()
        if scores_sum > 0:
            scores = scores / scores_sum
        else:
            scores = np.ones(d) / d

        # Exponential smoothing
        if self._importance_history is None:
            self._importance_history = scores
        else:
            lam = self.forgetting_factor
            self._importance_history = lam * self._importance_history + (1 - lam) * scores

        smoothed = self._importance_history / self._importance_history.sum()
        return smoothed

    @property
    def effective_k(self) -> int:
        """Number of components actually used in the last window."""
        return self._effective_k

    def reset(self):
        self.ipca = IncrementalPCA(n_components=self.k_max)
        self._fitted = False
        self._importance_history = None
        self._effective_k = self.k_max
