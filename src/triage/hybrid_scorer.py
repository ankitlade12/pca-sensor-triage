"""
Hybrid Importance Scorer: Combines PCA loadings with per-channel variance.

On datasets with limited correlation structure (e.g., SKAB with 8 sensors),
pure PCA may not capture enough structure. Blending PCA importance with
raw variance importance can improve allocation on such datasets.
"""

import numpy as np
from sklearn.decomposition import IncrementalPCA
from typing import Optional


class HybridScorer:
    """Combines PCA weighted loadings with per-channel variance.

    score_j = alpha * pca_score_j + (1 - alpha) * variance_score_j

    Parameters
    ----------
    n_components : int
        Number of PCA components.
    alpha : float
        Blending weight. 1.0 = pure PCA, 0.0 = pure variance.
    """

    def __init__(self, n_components: int = 10, alpha: float = 0.7):
        self.n_components = n_components
        self.alpha = alpha
        self.ipca = IncrementalPCA(n_components=n_components)
        self._fitted = False

    def compute_importance(self, window: np.ndarray) -> np.ndarray:
        w, d = window.shape

        # PCA importance
        if not self._fitted:
            self.ipca.fit(window)
            self._fitted = True
        else:
            self.ipca.partial_fit(window)

        V = self.ipca.components_
        sigma = self.ipca.singular_values_
        pca_scores = np.zeros(d)
        for i in range(len(sigma)):
            pca_scores += sigma[i] * V[i, :] ** 2
        pca_scores = pca_scores / pca_scores.sum() if pca_scores.sum() > 0 else np.ones(d) / d

        # Variance importance
        var_scores = np.var(window, axis=0)
        var_scores = var_scores / var_scores.sum() if var_scores.sum() > 0 else np.ones(d) / d

        # Blend
        blended = self.alpha * pca_scores + (1 - self.alpha) * var_scores
        return blended / blended.sum()

    def reset(self):
        self.ipca = IncrementalPCA(n_components=self.n_components)
        self._fitted = False
