"""
Data loading utilities for all three benchmark datasets.

Provides standardized interfaces to load, split, and preprocess
TEP, NASA Bearing, and SKAB datasets for experiments.
"""

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyreadr
from sklearn.preprocessing import StandardScaler

from .synthetic_datasets import generate_swat_like

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')


def load_tep(
    fault_types: List[int] = [0, 1, 2, 4, 5],
    train_runs: int = 20,
    test_start: int = 21,
    test_end: int = 30,
    scale: bool = True,
    fault_onset_sample: int = 20,
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
            te = ff_df[(ff_df['simulationRun'] >= test_start) &
                       (ff_df['simulationRun'] <= test_end)]
        else:
            tr = ft_df[(ft_df['faultNumber'] == fault) &
                       (ft_df['simulationRun'] <= train_runs)]
            te = ft_df[(ft_df['faultNumber'] == fault) &
                       (ft_df['simulationRun'] >= test_start) &
                       (ft_df['simulationRun'] <= test_end)]

        X_train_parts.append(tr[sensor_cols].values)
        if fault == 0:
            y_train_parts.append(np.zeros(len(tr), dtype=int))
        else:
            y_train_parts.append(
                np.where(tr["sample"].to_numpy() >= fault_onset_sample, fault, 0)
            )
        X_test_parts.append(te[sensor_cols].values)
        if fault == 0:
            y_test_parts.append(np.zeros(len(te), dtype=int))
        else:
            y_test_parts.append(
                np.where(te["sample"].to_numpy() >= fault_onset_sample, fault, 0)
            )

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

    split = int((1 - test_size) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

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
    split_strategy: str = "chronological",
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

    if split_strategy != "chronological":
        raise ValueError(
            "Only chronological splitting is supported for time-series data; "
            "random stratified splitting leaks temporal neighbors."
        )
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
    """Load the official SMD split for machine-1-1.

    The training stream is normal-only. The independent labeled test stream is
    kept intact and chronological.
    """
    del test_size
    root = os.path.join(DATA_DIR, "omni_temp", "ServerMachineDataset")
    train_path = os.path.join(root, "train", "machine-1-1.txt")
    test_path = os.path.join(root, "test", "machine-1-1.txt")
    label_path = os.path.join(root, "test_label", "machine-1-1.txt")
    X_train = np.loadtxt(train_path, delimiter=",")
    X_test = np.loadtxt(test_path, delimiter=",")
    y_train = np.zeros(len(X_train), dtype=int)
    y_test = np.loadtxt(label_path, delimiter=",").astype(int).reshape(-1)
    sensor_cols = [f"feature_{i}" for i in range(X_train.shape[1])]
    return _scale_official_split(X_train, y_train, X_test, y_test, sensor_cols, scale)


def load_msl(test_size: float = 0.3, scale: bool = True):
    """Load the official MSL normal-training and labeled-test arrays."""
    del test_size
    root = os.path.join(DATA_DIR, "msl_smap", "MSL")
    X_train = np.load(os.path.join(root, "MSL_train.npy"))
    X_test = np.load(os.path.join(root, "MSL_test.npy"))
    y_train = np.zeros(len(X_train), dtype=int)
    y_test = np.load(os.path.join(root, "MSL_test_label.npy")).astype(int)
    sensor_cols = [f"feature_{i}" for i in range(X_train.shape[1])]
    return _scale_official_split(X_train, y_train, X_test, y_test, sensor_cols, scale)


def load_psm(test_size: float = 0.3, scale: bool = True):
    """Load the official PSM normal-training and labeled-test CSV files."""
    del test_size
    root = os.path.join(DATA_DIR, "psm_temp", "data")
    train_df = pd.read_csv(os.path.join(root, "train.csv"))
    test_df = pd.read_csv(os.path.join(root, "test.csv"))
    label_df = pd.read_csv(os.path.join(root, "test_label.csv"))
    sensor_cols = [c for c in train_df.columns if c != "timestamp_(min)"]
    X_train = train_df[sensor_cols].to_numpy(dtype=float)
    X_test = test_df[sensor_cols].to_numpy(dtype=float)
    y_train = np.zeros(len(X_train), dtype=int)
    y_test = label_df["label"].to_numpy(dtype=int)
    return _scale_official_split(X_train, y_train, X_test, y_test, sensor_cols, scale)


def load_hai(test_size: float = 0.3, scale: bool = True):
    """Load HAI 20.07 using its official train/test file division."""
    del test_size
    root = os.path.join(DATA_DIR, "hai_temp", "hai-20.07")
    train_files = sorted(glob.glob(os.path.join(root, "train*.csv.gz")))
    test_files = sorted(glob.glob(os.path.join(root, "test*.csv.gz")))
    train_df = pd.concat([pd.read_csv(p, sep=";") for p in train_files], ignore_index=True)
    test_df = pd.concat([pd.read_csv(p, sep=";") for p in test_files], ignore_index=True)
    excluded = {"time", "attack", "attack_P1", "attack_P2", "attack_P3"}
    sensor_cols = [
        c for c in train_df.columns if c not in excluded and c in test_df.columns
    ]
    X_train = train_df[sensor_cols].to_numpy(dtype=float)
    X_test = test_df[sensor_cols].to_numpy(dtype=float)
    y_train = train_df["attack"].to_numpy(dtype=int)
    y_test = test_df["attack"].to_numpy(dtype=int)
    return _scale_official_split(X_train, y_train, X_test, y_test, sensor_cols, scale)


def _scale_official_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensor_cols: List[str],
    scale: bool,
):
    """Clean and train-scale an already separated official dataset split."""
    X_train = np.nan_to_num(X_train.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test, sensor_cols, scaler


# Dataset registry — all datasets (includes synthetic stand-ins for SWaT/WADI)
DATASETS = {
    'tep': load_tep,
    'skab': load_skab,
    'nasa': load_nasa_bearing,
    'smd': load_smd,
    'msl': load_msl,
    'psm': load_psm,
    'hai': load_hai,
    'swat': generate_swat_like,
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
