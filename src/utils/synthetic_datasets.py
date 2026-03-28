"""
Synthetic dataset generators that mimic real-world dataset properties.

These serve as stand-ins until real data access agreements are obtained.
Properties are calibrated to match published dataset characteristics.
"""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _generate_correlated_data(
    n_samples: int,
    n_channels: int,
    n_correlated_groups: int = 5,
    group_correlation: float = 0.7,
    anomaly_ratio: float = 0.1,
    anomaly_channels: int = 5,
    anomaly_magnitude: float = 3.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multi-channel time series with correlation structure and anomalies."""
    rng = np.random.RandomState(seed)

    # Create correlation structure via latent factors
    group_size = n_channels // n_correlated_groups
    X = np.zeros((n_samples, n_channels))

    for g in range(n_correlated_groups):
        start = g * group_size
        end = start + group_size if g < n_correlated_groups - 1 else n_channels
        n_ch = end - start

        # Shared latent signal for the group
        latent = rng.randn(n_samples)
        for ch in range(n_ch):
            noise = rng.randn(n_samples)
            X[:, start + ch] = (
                group_correlation * latent
                + (1 - group_correlation) * noise
            )

    # Add temporal structure (autoregressive)
    for ch in range(n_channels):
        for t in range(1, n_samples):
            X[t, ch] = 0.8 * X[t - 1, ch] + 0.2 * X[t, ch]

    # Generate anomaly labels
    y = np.zeros(n_samples, dtype=int)
    n_anomaly = int(n_samples * anomaly_ratio)
    # Anomalies in contiguous segments
    n_segments = max(1, n_anomaly // 100)
    segment_len = n_anomaly // n_segments

    for _ in range(n_segments):
        start_idx = rng.randint(n_samples // 4, 3 * n_samples // 4)
        end_idx = min(start_idx + segment_len, n_samples)
        y[start_idx:end_idx] = 1

        # Inject anomaly into a subset of channels
        affected = rng.choice(n_channels, size=min(anomaly_channels, n_channels), replace=False)
        for ch in affected:
            X[start_idx:end_idx, ch] += anomaly_magnitude * rng.randn(end_idx - start_idx)

    return X, y


def generate_swat_like(
    n_samples: int = 500000,
    test_size: float = 0.3,
    scale: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """Generate SWaT-like dataset (51 channels, water treatment).

    Mimics Secure Water Treatment testbed:
    - 51 sensor/actuator channels
    - ~12% attack ratio
    - 6 correlated sensor groups (water treatment stages)
    """
    X, y = _generate_correlated_data(
        n_samples=n_samples,
        n_channels=51,
        n_correlated_groups=6,
        group_correlation=0.65,
        anomaly_ratio=0.12,
        anomaly_channels=8,
        anomaly_magnitude=2.5,
        seed=seed,
    )

    sensor_cols = [f"swat_{i:02d}" for i in range(51)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, sensor_cols, scaler


def generate_wadi_like(
    n_samples: int = 500000,
    test_size: float = 0.3,
    scale: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """Generate WADI-like dataset (127 channels, water distribution).

    Mimics Water Distribution testbed:
    - 127 sensor/actuator channels (highest channel count)
    - ~5% attack ratio
    - 10 correlated groups (distribution zones)
    """
    X, y = _generate_correlated_data(
        n_samples=n_samples,
        n_channels=127,
        n_correlated_groups=10,
        group_correlation=0.6,
        anomaly_ratio=0.05,
        anomaly_channels=15,
        anomaly_magnitude=2.0,
        seed=seed,
    )

    sensor_cols = [f"wadi_{i:03d}" for i in range(127)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, sensor_cols, scaler
