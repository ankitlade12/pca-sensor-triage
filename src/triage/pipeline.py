"""
PCA-Triage Pipeline: End-to-end streaming sensor triage.

Combines PCATriage, RateAllocator, and reconstruction into a single
pipeline that processes sensor data window-by-window.
"""

import numpy as np
from typing import Optional
from .pca_triage import PCATriage
from .hybrid_scorer import HybridScorer
from .rate_allocator import RateAllocator
from .reconstruction import reconstruct


class TriagePipeline:
    """End-to-end PCA-Triage pipeline.

    Parameters
    ----------
    n_components : int
        Number of PCA components (k).
    window_size : int
        Samples per window (w).
    budget : float
        Bandwidth budget as fraction of full rate.
    forgetting_factor : float
        Exponential forgetting for importance smoothing.
    min_rate : float
        Minimum per-channel sampling rate.
    reconstruction_method : str
        How to interpolate dropped samples.
    scorer : str
        Scoring method: "pca" for pure PCA, "hybrid" for PCA+Variance blend.
    alpha : float
        Blending weight for hybrid scorer (1.0 = pure PCA, 0.0 = pure variance).
    """

    def __init__(
        self,
        n_components: int = 10,
        window_size: int = 100,
        budget: float = 0.5,
        forgetting_factor: float = 0.95,
        min_rate: float = 0.05,
        reconstruction_method: str = "linear",
        scorer: str = "pca",
        alpha: float = 0.7,
        sharpness: float = 1.0,
    ):
        self.scorer_type = scorer
        if scorer == "hybrid":
            self.triage = HybridScorer(
                n_components=n_components, alpha=alpha,
                forgetting_factor=forgetting_factor,
            )
        else:
            self.triage = PCATriage(
                n_components=n_components,
                window_size=window_size,
                forgetting_factor=forgetting_factor,
                min_rate=min_rate,
            )
        self.allocator = RateAllocator(
            budget=budget, min_rate=min_rate, sharpness=sharpness,
        )
        self.reconstruction_method = reconstruction_method
        self.window_size = window_size

        # Logging
        self.importance_log = []
        self.rate_log = []
        self.bandwidth_log = []

    def process_stream(
        self,
        data: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """Process a full data stream window-by-window.

        Parameters
        ----------
        data : np.ndarray, shape (n, d)
            Full sensor data stream.
        seed : int
            Random seed for rate application.

        Returns
        -------
        reconstructed : np.ndarray, shape (n, d)
            Reconstructed data after triage.
        """
        n, d = data.shape
        n_windows = n // self.window_size
        reconstructed = np.zeros_like(data, dtype=float)

        for w_idx in range(n_windows):
            start = w_idx * self.window_size
            end = start + self.window_size
            window = data[start:end]

            # Step 1: Compute importance scores
            importance = self.triage.compute_importance(window)
            self.importance_log.append(importance.copy())

            # Step 2: Allocate rates
            rates = self.allocator.allocate(importance)
            self.rate_log.append(rates.copy())
            self.bandwidth_log.append(self.allocator.get_effective_bandwidth(rates))

            # Step 3: Apply rates (sub-sample)
            triaged = self.allocator.apply_rates(
                window, rates, seed=seed + w_idx
            )

            # Step 4: Reconstruct
            recon = reconstruct(triaged, method=self.reconstruction_method)
            reconstructed[start:end] = recon

        # Handle remaining samples (tail < window_size)
        remaining = n % self.window_size
        if remaining > 0:
            start = n_windows * self.window_size
            # Use last known rates if available, else uniform
            if self.rate_log:
                rates = self.rate_log[-1]
            else:
                rates = np.full(d, self.allocator.budget)

            triaged = self.allocator.apply_rates(
                data[start:], rates, seed=seed + n_windows
            )
            recon = reconstruct(triaged, method=self.reconstruction_method)
            reconstructed[start:] = recon

        return reconstructed

    def get_importance_matrix(self) -> np.ndarray:
        """Return importance scores as (n_windows, d) matrix."""
        if self.importance_log:
            return np.array(self.importance_log)
        return np.array([])

    def get_rate_matrix(self) -> np.ndarray:
        """Return allocated rates as (n_windows, d) matrix."""
        if self.rate_log:
            return np.array(self.rate_log)
        return np.array([])

    def reset(self):
        """Reset pipeline state."""
        self.triage.reset()
        self.importance_log.clear()
        self.rate_log.clear()
        self.bandwidth_log.clear()
