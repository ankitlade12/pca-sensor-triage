"""
Reconstruction: Recover full-rate data from triaged (sub-sampled) observations.

Supports multiple interpolation strategies for filling in missing samples
after PCA-Triage has reduced per-channel sampling rates.
"""

import numpy as np
import pandas as pd
from typing import Literal


def reconstruct(
    triaged_data: np.ndarray,
    method: Literal["forward_fill", "linear", "zero"] = "forward_fill",
) -> np.ndarray:
    """Reconstruct full-rate data from triaged observations.

    Parameters
    ----------
    triaged_data : np.ndarray, shape (n, d)
        Sub-sampled data with NaN for dropped samples.
    method : str
        Interpolation strategy:
        - "forward_fill": hold last observed value (zero-order hold)
        - "linear": linear interpolation between observed values
        - "zero": replace NaN with 0 (useful for difference features)

    Returns
    -------
    reconstructed : np.ndarray, shape (n, d)
        Reconstructed full-rate data.
    """
    if method == "zero":
        return np.nan_to_num(triaged_data, nan=0.0)

    df = pd.DataFrame(triaged_data)

    if method == "forward_fill":
        # Forward fill, then backward fill for leading NaNs
        df = df.ffill().bfill()
    elif method == "linear":
        df = df.interpolate(method="linear", axis=0, limit_direction="both")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Fill any remaining NaNs (e.g., columns with zero samples) with 0
    df = df.fillna(0.0)

    return df.values


def compute_reconstruction_error(
    original: np.ndarray, reconstructed: np.ndarray
) -> dict:
    """Compute reconstruction quality metrics.

    Parameters
    ----------
    original : np.ndarray, shape (n, d)
        Original full-rate data.
    reconstructed : np.ndarray, shape (n, d)
        Reconstructed data after triage + interpolation.

    Returns
    -------
    metrics : dict
        MSE, MAE, RMSE, and per-channel NRMSE.
    """
    diff = original - reconstructed

    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)

    # Per-channel Normalized RMSE (normalized by channel range)
    ranges = np.ptp(original, axis=0)
    ranges[ranges == 0] = 1.0  # avoid division by zero
    per_channel_rmse = np.sqrt(np.mean(diff ** 2, axis=0))
    nrmse = per_channel_rmse / ranges

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "nrmse_mean": np.mean(nrmse),
        "nrmse_per_channel": nrmse,
    }
