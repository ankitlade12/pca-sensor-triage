"""
PCA-Triage Pipeline: End-to-end streaming sensor triage.

Combines PCATriage, RateAllocator, and reconstruction into a single
pipeline that processes sensor data window-by-window.
"""


import numpy as np

from .adaptive_k import AdaptiveKPCATriage
from .ensemble_scorer import EnsembleScorer
from .hybrid_scorer import HybridScorer
from .pca_triage import PCATriage
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
        Scoring method: "pca", "hybrid", "adaptive_k", or "ensemble".
    alpha : float
        Blending weight for hybrid scorer (1.0 = pure PCA, 0.0 = pure variance).
    sharpness : float
        Power-law exponent for rate allocation sharpening.
    variance_threshold : float
        Cumulative variance threshold for adaptive_k scorer.
    k_values : list of int
        Component counts for ensemble scorer.
    update_source : {"reconstructed", "full"}
        Data used to update PCA after acquisition. ``reconstructed`` is the
        strict constrained-sensing model. ``full`` is only appropriate when a
        local gateway sees full-rate data and the constrained link is upstream.
    hard_budget : bool
        Enforce the aggregate communication cap in every window.
    """

    def __init__(
        self,
        n_components: int = 10,
        window_size: int = 100,
        budget: float = 0.5,
        forgetting_factor: float = 0.95,
        min_rate: float = 0.05,
        reconstruction_method: str = "forward_fill",
        scorer: str = "pca",
        alpha: float = 0.7,
        sharpness: float = 1.0,
        variance_threshold: float = 0.95,
        k_values: list = None,
        update_source: str = "reconstructed",
        hard_budget: bool = True,
    ):
        self.scorer_type = scorer
        if scorer == "hybrid":
            self.triage = HybridScorer(
                n_components=n_components, alpha=alpha,
                forgetting_factor=forgetting_factor,
            )
        elif scorer == "adaptive_k":
            self.triage = AdaptiveKPCATriage(
                k_max=n_components,
                variance_threshold=variance_threshold,
                window_size=window_size,
                forgetting_factor=forgetting_factor,
            )
        elif scorer == "ensemble":
            self.triage = EnsembleScorer(
                k_values=k_values or [3, 5, 10],
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
        if update_source not in {"reconstructed", "full"}:
            raise ValueError("update_source must be 'reconstructed' or 'full'")
        if reconstruction_method == "linear":
            # Linear interpolation is supported for explicitly offline studies,
            # but the journal-default pipeline is causal forward fill.
            pass
        self.update_source = update_source
        self.hard_budget = hard_budget
        self.window_size = window_size
        self._next_rates = None

        # Logging
        self.importance_log = []
        self.rate_log = []
        self.bandwidth_log = []
        self.realized_bandwidth_log = []

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

        # Reconstruction state is reset at each explicit stream boundary.
        last_values = np.zeros(d, dtype=float)

        for w_idx in range(n_windows):
            start = w_idx * self.window_size
            end = start + self.window_size
            window = data[start:end]

            # Step 1: Use rates computed only from preceding windows. The first
            # window is a uniform-budget bootstrap.
            if self._next_rates is None:
                rates = np.full(d, self.allocator.budget)
            else:
                rates = self._next_rates.copy()
            self.rate_log.append(rates.copy())
            self.bandwidth_log.append(self.allocator.get_effective_bandwidth(rates))

            # Step 2: Acquire/transmit current samples under a hard window cap.
            triaged = self.allocator.apply_rates(
                window, rates, seed=seed + w_idx, hard_budget=self.hard_budget
            )
            self.realized_bandwidth_log.append(float(np.isfinite(triaged).mean()))

            # Step 3: Reconstruct without future observations in causal mode.
            recon = reconstruct(
                triaged,
                method=self.reconstruction_method,
                initial_values=last_values if self.reconstruction_method == "forward_fill" else None,
            )
            reconstructed[start:end] = recon
            last_values = recon[-1].copy()

            # Step 4: Update after the current window, producing rates for the
            # next window. 'full' models an uplink-constrained gateway that sees
            # local full-rate data; 'reconstructed' models constrained sensing.
            update_window = window if self.update_source == "full" else recon
            importance = self.triage.compute_importance(update_window)
            self.importance_log.append(importance.copy())
            self._next_rates = self.allocator.allocate(importance)

        # Handle remaining samples (tail < window_size)
        remaining = n % self.window_size
        if remaining > 0:
            start = n_windows * self.window_size
            # Use last known rates if available, else uniform
            if self._next_rates is not None:
                rates = self._next_rates.copy()
            else:
                rates = np.full(d, self.allocator.budget)

            self.rate_log.append(rates.copy())
            self.bandwidth_log.append(self.allocator.get_effective_bandwidth(rates))

            triaged = self.allocator.apply_rates(
                data[start:], rates, seed=seed + n_windows, hard_budget=self.hard_budget
            )
            self.realized_bandwidth_log.append(float(np.isfinite(triaged).mean()))
            recon = reconstruct(
                triaged,
                method=self.reconstruction_method,
                initial_values=last_values if self.reconstruction_method == "forward_fill" else None,
            )
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
        self._next_rates = None
        self.importance_log.clear()
        self.rate_log.clear()
        self.bandwidth_log.clear()
        self.realized_bandwidth_log.clear()
        self.allocator.observation_log.clear()
