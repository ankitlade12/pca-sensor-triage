"""
Hybrid Importance Scorer: Combines PCA loadings with per-channel variance.

On datasets with limited correlation structure (e.g., SKAB with 8 sensors),
pure PCA may not capture enough structure. Blending PCA importance with
raw variance importance can improve allocation on such datasets.
"""


import numpy as np
from sklearn.decomposition import IncrementalPCA


class HybridScorer:
    """Combines PCA weighted loadings with per-channel variance.

    score_j = alpha * pca_score_j + (1 - alpha) * variance_score_j

    Parameters
    ----------
    n_components : int
        Number of PCA components.
    alpha : float
        Blending weight. 1.0 = pure PCA, 0.0 = pure variance.
    forgetting_factor : float
        Exponential smoothing for importance scores over time.
    """

    def __init__(self, n_components: int = 10, alpha: float = 0.7,
                 forgetting_factor: float = 1.0):
        if not 0 <= forgetting_factor <= 1:
            raise ValueError("forgetting_factor must be in [0, 1]")
        self.n_components = n_components
        self.alpha = alpha
        self.forgetting_factor = forgetting_factor
        self.ipca = IncrementalPCA(n_components=n_components)
        self._fitted = False
        self._importance_history = None
        self._n_score_updates = 0

    def compute_importance(self, window: np.ndarray) -> np.ndarray:
        w, d = window.shape

        # PCA importance
        if not self._fitted:
            self.ipca.fit(window)
            self._fitted = True
        else:
            self.ipca.partial_fit(window)

        V = self.ipca.components_
        eigenvalues = self.ipca.explained_variance_
        pca_scores = np.zeros(d)
        for i in range(len(eigenvalues)):
            pca_scores += eigenvalues[i] * V[i, :] ** 2
        pca_scores = pca_scores / pca_scores.sum() if pca_scores.sum() > 0 else np.ones(d) / d

        # Variance importance
        var_scores = np.var(window, axis=0)
        var_scores = var_scores / var_scores.sum() if var_scores.sum() > 0 else np.ones(d) / d

        # Blend
        blended = self.alpha * pca_scores + (1 - self.alpha) * var_scores
        scores = blended / blended.sum()

        # Exponential smoothing over time
        if self._importance_history is None:
            self._importance_history = scores.copy()
            self._n_score_updates = 1
        else:
            lam = self.forgetting_factor
            if lam == 1.0:
                n = self._n_score_updates
                self._importance_history = (n * self._importance_history + scores) / (n + 1)
            else:
                self._importance_history = lam * self._importance_history + (1 - lam) * scores
            self._n_score_updates += 1

        smoothed = self._importance_history / self._importance_history.sum()
        return smoothed

    def reset(self):
        self.ipca = IncrementalPCA(n_components=self.n_components)
        self._fitted = False
        self._importance_history = None
        self._n_score_updates = 0
