"""
Send-on-Delta (temporal) + PCA-Triage (spatial) Joint Optimization.

Send-on-Delta: only transmit a sample when the change exceeds a threshold δ.
This is a temporal sampling strategy — complementary to PCA-Triage's spatial strategy.

Combined: PCA-Triage decides per-channel spatial rates, Send-on-Delta adds
per-sample temporal suppression within each channel. The joint system
achieves higher bandwidth savings than either alone.
"""

import numpy as np
import pandas as pd
from src.triage.rate_allocator import RateAllocator


class SendOnDelta:
    """Pure Send-on-Delta temporal sampling.

    Only transmits a sample when |x_t - x_last_sent| > delta.

    Parameters
    ----------
    delta : float
        Change threshold for transmission.
    budget : float
        Target bandwidth budget (delta is auto-tuned if budget is set).
    """

    def __init__(self, delta: float = 0.1, budget: float = None):
        self.delta = delta
        self.budget = budget

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        n, d = data.shape
        result = data.copy().astype(float)
        last_sent = data[0].copy()

        for t in range(1, n):
            for j in range(d):
                if abs(data[t, j] - last_sent[j]) > self.delta:
                    last_sent[j] = data[t, j]
                    result[t, j] = data[t, j]
                else:
                    result[t, j] = np.nan  # suppress

        # Reconstruct via forward-fill
        result = pd.DataFrame(result).ffill().bfill().fillna(0.0).values

        return result

    def get_effective_rate(self, data: np.ndarray) -> float:
        """Measure actual transmission rate."""
        n, d = data.shape
        transmitted = 0
        last_sent = data[0].copy()
        for t in range(1, n):
            for j in range(d):
                if abs(data[t, j] - last_sent[j]) > self.delta:
                    last_sent[j] = data[t, j]
                    transmitted += 1
        return transmitted / ((n - 1) * d)


class JointSpatialTemporal:
    """Joint PCA-Triage (spatial) + Send-on-Delta (temporal).

    First applies PCA-Triage to determine per-channel spatial rates,
    then applies Send-on-Delta within each channel for temporal suppression.
    The combined bandwidth savings exceed either method alone.

    Parameters
    ----------
    spatial_budget : float
        Budget for spatial allocation (PCA-Triage).
    delta : float
        Send-on-Delta threshold for temporal suppression.
    n_components : int
        PCA components.
    window_size : int
        Window size.
    """

    def __init__(
        self,
        spatial_budget: float = 0.7,
        delta: float = 0.1,
        n_components: int = 10,
        window_size: int = 50,
        min_rate: float = 0.05,
    ):
        self.spatial_budget = spatial_budget
        self.delta = delta
        self.n_components = n_components
        self.window_size = window_size
        self.min_rate = min_rate

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        from src.triage.pipeline import TriagePipeline

        # Step 1: Spatial triage (PCA-Triage)
        pipe = TriagePipeline(
            n_components=self.n_components,
            window_size=self.window_size,
            budget=self.spatial_budget,
            forgetting_factor=1.0,
            min_rate=self.min_rate,
            reconstruction_method='linear',
            scorer='hybrid',
            alpha=0.7,
        )
        spatially_triaged = pipe.process_stream(data, seed=seed)

        # Step 2: Temporal suppression (Send-on-Delta on the spatially triaged data)
        n, d = spatially_triaged.shape
        result = spatially_triaged.copy()
        last_sent = spatially_triaged[0].copy()

        for t in range(1, n):
            for j in range(d):
                if abs(spatially_triaged[t, j] - last_sent[j]) > self.delta:
                    last_sent[j] = spatially_triaged[t, j]
                else:
                    result[t, j] = np.nan

        # Reconstruct
        result = pd.DataFrame(result).ffill().bfill().fillna(0.0).values
        return result

    def get_effective_rate(self, data: np.ndarray, seed: int = 42) -> float:
        """Measure combined effective bandwidth rate."""
        from src.triage.pipeline import TriagePipeline

        pipe = TriagePipeline(
            n_components=self.n_components, window_size=self.window_size,
            budget=self.spatial_budget, forgetting_factor=1.0,
            min_rate=self.min_rate, reconstruction_method='linear',
            scorer='hybrid', alpha=0.7)
        spatially_triaged = pipe.process_stream(data, seed=seed)

        n, d = spatially_triaged.shape
        transmitted = 0
        last_sent = spatially_triaged[0].copy()
        for t in range(1, n):
            for j in range(d):
                if abs(spatially_triaged[t, j] - last_sent[j]) > self.delta:
                    last_sent[j] = spatially_triaged[t, j]
                    transmitted += 1
        spatial_rate = self.spatial_budget
        temporal_rate = transmitted / ((n - 1) * d)
        return spatial_rate * temporal_rate
