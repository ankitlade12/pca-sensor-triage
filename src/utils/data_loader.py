"""
Data loading utilities for all three benchmark datasets.

Provides standardized interfaces to load, split, and preprocess
TEP, NASA Bearing, and SKAB datasets for experiments.
"""

import os
import glob
import numpy as np
import pandas as pd
import pyreadr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')


def load_tep(
    fault_types: List[int] = [0, 1, 2, 4, 5],
    train_runs: int = 20,
    test_start: int = 21,
    test_end: int = 30,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """Load Tennessee Eastman Process dataset.

    Parameters
    ----------
    fault_types : list of int
        Fault types to include. 0 = normal operation.
    train_runs : int
        Number of simulation runs for training per fault type.
    test_start, test_end : int
        Range of simulation runs for testing.
    scale : bool
        Whether to standardize features.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        Training and testing data with labels.
    sensor_cols : list of str
        Sensor column names.
    scaler : StandardScaler or None
        Fitted scaler (if scale=True).
    """
    ff_path = os.path.join(DATA_DIR, 'TEP_FaultFree_Training.RData')
    ft_path = os.path.join(DATA_DIR, 'TEP_Faulty_Training.RData')

    ff = pyreadr.read_r(ff_path)
    ff_df = ff[list(ff.keys())[0]]

    ft = pyreadr.read_r(ft_path)
    ft_df = ft[list(ft.keys())[0]]

    sensor_cols = [c for c in ff_df.columns
                   if c.startswith('xmeas_') or c.startswith('xmv_')]

    X_train_parts, y_train_parts = [], []
    X_test_parts, y_test_parts = [], []

    for fault in fault_types:
        if fault == 0:
            tr = ff_df[ff_df['simulationRun'] <= train_runs]
            te = ff_df[(ff_df['simulationRun'] > test_start) &
                       (ff_df['simulationRun'] <= test_end)]
        else:
            tr = ft_df[(ft_df['faultNumber'] == fault) &
                       (ft_df['simulationRun'] <= train_runs)]
            te = ft_df[(ft_df['faultNumber'] == fault) &
                       (ft_df['simulationRun'] > test_start) &
                       (ft_df['simulationRun'] <= test_end)]

        X_train_parts.append(tr[sensor_cols].values)
        y_train_parts.append(np.full(len(tr), fault))
        X_test_parts.append(te[sensor_cols].values)
        y_test_parts.append(np.full(len(te), fault))

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    X_test = np.vstack(X_test_parts)
    y_test = np.concatenate(y_test_parts)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, sensor_cols, scaler


def load_tep_single_fault(
    fault_num: int,
    run: int = 1,
    scale: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Load a single fault run from TEP (for adaptivity analysis)."""
    ft_path = os.path.join(DATA_DIR, 'TEP_Faulty_Training.RData')
    ft = pyreadr.read_r(ft_path)
    ft_df = ft[list(ft.keys())[0]]

    sensor_cols = [c for c in ft_df.columns
                   if c.startswith('xmeas_') or c.startswith('xmv_')]

    run_df = ft_df[(ft_df['faultNumber'] == fault_num) &
                   (ft_df['simulationRun'] == run)]
    X = run_df[sensor_cols].values

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, sensor_cols


def load_skab(
    test_size: float = 0.3,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """Load SKAB (Skoltech Anomaly Benchmark) dataset.

    Returns
    -------
    X_train, y_train, X_test, y_test, sensor_cols, scaler
    """
    skab_dir = os.path.join(DATA_DIR, 'skab_repo', 'data')

    dfs = []
    for subdir in ['anomaly-free', 'valve1', 'valve2', 'other']:
        path = os.path.join(skab_dir, subdir)
        if not os.path.isdir(path):
            continue
        for f in sorted(glob.glob(os.path.join(path, '*.csv'))):
            df = pd.read_csv(f, sep=';', parse_dates=['datetime'],
                             index_col='datetime')
            dfs.append(df)

    skab = pd.concat(dfs, ignore_index=True)
    sensor_cols = [c for c in skab.columns
                   if c not in ['anomaly', 'changepoint']]

    X = np.nan_to_num(skab[sensor_cols].values.astype(float), nan=0.0)
    y = skab['anomaly'].values.astype(int)

    split = int((1 - test_size) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, sensor_cols, scaler


def load_nasa_bearing(
    test_size: float = 0.3,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """Load NASA Bearing (synthetic) dataset.

    Returns
    -------
    X_train, y_train, X_test, y_test, sensor_cols, scaler
    """
    csv_path = os.path.join(DATA_DIR, 'nasa_bearing_synthetic.csv')
    df = pd.read_csv(csv_path)

    sensor_cols = [c for c in df.columns if c not in ['RUL', 'anomaly']]
    X = df[sensor_cols].values
    y = df['anomaly'].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, sensor_cols, scaler


def _load_csv_dataset(
    filename: str,
    label_col: str = 'anomaly',
    exclude_cols: List[str] = None,
    test_size: float = 0.3,
    scale: bool = True,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """Generic loader for CSV datasets with anomaly labels."""
    csv_path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(csv_path)

    if exclude_cols is None:
        exclude_cols = []
    exclude = set([label_col] + exclude_cols)
    sensor_cols = [c for c in df.columns if c not in exclude]

    X = np.nan_to_num(df[sensor_cols].values.astype(float), nan=0.0)
    y = df[label_col].values.astype(int)

    if stratify and len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
    else:
        split = int((1 - test_size) * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, sensor_cols, scaler


def load_smd(test_size: float = 0.3, scale: bool = True):
    """Load Server Machine Dataset (38 features, 388K samples, anomaly detection).
    Source: NetManAIOps/OmniAnomaly (KDD 2019).
    """
    return _load_csv_dataset('smd_combined.csv', test_size=test_size, scale=scale)


def load_msl(test_size: float = 0.3, scale: bool = True):
    """Load NASA Mars Science Laboratory telemetry (55 features, 132K samples).
    Source: khundman/telemanom (KDD 2018).
    """
    return _load_csv_dataset('msl_combined.csv', test_size=test_size, scale=scale)


def load_psm(test_size: float = 0.3, scale: bool = True):
    """Load Pooled Server Metrics from eBay (25 features, 220K samples).
    Source: eBay/RANSynCoders (KDD 2021).
    """
    return _load_csv_dataset('psm_combined.csv', test_size=test_size, scale=scale)


def load_hai(test_size: float = 0.3, scale: bool = True):
    """Load HIL-based Augmented ICS dataset (83 features, 259K samples).
    Source: icsdataset/hai (CSET 2020).
    """
    return _load_csv_dataset('hai_combined.csv', exclude_cols=['time'],
                             test_size=test_size, scale=scale)


# Dataset registry — all real-world datasets
DATASETS = {
    'tep': load_tep,
    'skab': load_skab,
    'nasa': load_nasa_bearing,
    'smd': load_smd,
    'msl': load_msl,
    'psm': load_psm,
    'hai': load_hai,
}


def get_dataset(name: str, **kwargs):
    """Load a dataset by name.

    Parameters
    ----------
    name : str
        One of: tep, skab, nasa, smd, msl, psm, hai
    """
    if name.lower() not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASETS.keys())}")
    return DATASETS[name.lower()](**kwargs)


def list_datasets() -> List[str]:
    """Return list of available dataset names."""
    return list(DATASETS.keys())
