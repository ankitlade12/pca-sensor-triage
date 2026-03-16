"""
PCA-Triage: Streaming PCA-based channel importance scoring.

Computes per-channel importance scores from incremental PCA loadings
using the weighted loadings formula: score_j = sum_i sigma_i * |V[i,j]|^2

This converts PCA's subspace information into actionable channel-level
importance rankings for bandwidth allocation.
"""

import numpy as np
from sklearn.decomposition import IncrementalPCA
from typing import Optional


class PCATriage:
    """Streaming PCA-based sensor channel importance scorer.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain (k).
    window_size : int
        Number of samples per sliding window (w).
    forgetting_factor : float
        Exponential forgetting factor lambda in (0, 1].
        1.0 = no forgetting (equal weight to all history).
        0.9 = recent windows weighted ~10x more than 10-windows-ago.
    min_rate : float
        Minimum sampling rate floor for any channel.
    """

    def __init__(
        self,
        n_components: int = 10,
        window_size: int = 100,
        forgetting_factor: float = 0.95,
        min_rate: float = 0.05,
    ):
        self.n_components = n_components
        self.window_size = window_size
        self.forgetting_factor = forgetting_factor
        self.min_rate = min_rate

        self.ipca = IncrementalPCA(n_components=n_components)
        self._fitted = False
        self._importance_history: Optional[np.ndarray] = None
        self._n_channels: Optional[int] = None

    def compute_importance(self, window: np.ndarray) -> np.ndarray:
        """Compute per-channel importance scores from a data window.

        Parameters
        ----------
        window : np.ndarray, shape (w, d)
            A window of sensor observations. w = samples, d = channels.

        Returns
        -------
        scores : np.ndarray, shape (d,)
            Per-channel importance scores, normalized to sum to 1.
        """
        w, d = window.shape
        self._n_channels = d

        # Fit or partial_fit the incremental PCA
        if not self._fitted:
            self.ipca.fit(window)
            self._fitted = True
        else:
            self.ipca.partial_fit(window)

        # Extract components (V) and singular values (sigma)
        # ipca.components_ has shape (k, d) — rows are eigenvectors
        # ipca.singular_values_ has shape (k,)
        V = self.ipca.components_  # (k, d)
        sigma = self.ipca.singular_values_  # (k,)

        # Weighted loadings: score_j = sum_i sigma_i * V[i,j]^2
        scores = np.zeros(d)
        for i in range(len(sigma)):
            scores += sigma[i] * V[i, :] ** 2

        # Normalize to sum to 1
        scores = scores / scores.sum()

        # Apply exponential forgetting to smooth importance over time
        if self._importance_history is None:
            self._importance_history = scores
        else:
            lam = self.forgetting_factor
            self._importance_history = lam * self._importance_history + (1 - lam) * scores

        # Re-normalize after blending
        smoothed = self._importance_history / self._importance_history.sum()

        return smoothed

    def get_raw_importance(self, window: np.ndarray) -> np.ndarray:
        """Compute raw (unsmoothed) importance scores for a single window.

        Useful for analysis and visualization without affecting internal state.
        """
        w, d = window.shape

        ipca_temp = IncrementalPCA(n_components=self.n_components)
        ipca_temp.fit(window)

        V = ipca_temp.components_
        sigma = ipca_temp.singular_values_

        scores = np.zeros(d)
        for i in range(len(sigma)):
            scores += sigma[i] * V[i, :] ** 2

        return scores / scores.sum()

    def reset(self):
        """Reset the triage model to initial state."""
        self.ipca = IncrementalPCA(n_components=self.n_components)
        self._fitted = False
        self._importance_history = None
        self._n_channels = None

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Return explained variance ratio from latest PCA fit."""
        if self._fitted:
            return self.ipca.explained_variance_ratio_
        return None

    @property
    def n_channels(self) -> Optional[int]:
        return self._n_channels
