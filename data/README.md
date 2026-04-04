# Dataset Setup

This directory stores raw datasets for PCA-Triage experiments. Datasets are not
included in the repository due to size and licensing. Use `download_datasets.sh`
for automated download where possible, and follow manual steps below for the rest.

## Quick Start

```bash
bash data/download_datasets.sh
```

## Dataset Details

### TEP (Tennessee Eastman Process)
- **Source:** Harvard Dataverse (Rieth et al. 2017)
- **URL:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1
- **Files needed:** `TEP_FaultFree_Training.RData`, `TEP_Faulty_Training.RData`
- **Place in:** `data/raw/`
- **Format:** R DataFrames (loaded via `pyreadr`)
- **Size:** ~520 MB total
- **Access:** Public (Harvard Dataverse terms of use)

### SMD (Server Machine Dataset)
- **Source:** NetManAIOps/OmniAnomaly (Su et al. KDD 2019)
- **URL:** https://github.com/NetManAIOps/OmniAnomaly
- **Preprocessing:** Combine machine-1-1 train/test with labels into single CSV
- **Expected output:** `data/raw/smd_combined.csv`
- **Columns:** 38 sensor features + `anomaly` (binary)
- **Access:** Public (GitHub)

### MSL (Mars Science Laboratory)
- **Source:** khundman/telemanom (Hundman et al. KDD 2018)
- **URL:** https://github.com/khundman/telemanom
- **Data URL:** https://s3-us-west-2.amazonaws.com/telemanom/data.zip
- **Preprocessing:** Combine MSL channel files with labels
- **Expected output:** `data/raw/msl_combined.csv`
- **Columns:** 55 sensor features + `anomaly` (binary)
- **Access:** Public (GitHub/S3)

### PSM (Pooled Server Metrics)
- **Source:** eBay/RANSynCoders (Abdulaal et al. KDD 2021)
- **URL:** https://github.com/eBay/RANSynCoders
- **Preprocessing:** Combine train.csv, test.csv, test_label.csv
- **Expected output:** `data/raw/psm_combined.csv`
- **Columns:** 25 sensor features + `anomaly` (binary)
- **Access:** Public (GitHub)

### HAI (HIL-based Augmented ICS)
- **Source:** icsdataset/hai (Shin et al. CSET 2020)
- **URL:** https://github.com/icsdataset/hai
- **Preprocessing:** Combine dataset CSVs with attack labels
- **Expected output:** `data/raw/hai_combined.csv`
- **Columns:** 82 sensor features + `time` + `anomaly` (binary)
- **Access:** Public (GitHub)

### SKAB (Skoltech Anomaly Benchmark)
- **Source:** waico/SKAB (Skoltech 2020)
- **URL:** https://github.com/waico/SKAB
- **Download:** `git clone https://github.com/waico/SKAB.git data/raw/skab_repo`
- **No preprocessing needed** — loader reads CSV subdirectories directly
- **Access:** Public (GitHub)

### SWaT (Secure Water Treatment) — Synthetic
- **Generated on-the-fly** by `src/utils/synthetic_datasets.py`
- No download needed
- Used only in DL baseline experiments (Experiment 14), not main comparison

## Preprocessing Notes

All datasets undergo the same standardization:
1. Select sensor columns (exclude metadata: datetime, time, etc.)
2. Replace NaN with 0.0
3. Train/test split: 70/30, stratified by label (`random_state=42`)
4. StandardScaler: fit on train, transform both train and test

See `src/utils/data_loader.py` for exact implementation.

## Verification

After setup, verify all datasets load correctly:

```python
from src.utils.data_loader import get_dataset, list_datasets
print(list_datasets())
for ds in ['tep', 'smd', 'msl', 'psm', 'hai', 'skab']:
    X_train, y_train, X_test, y_test, cols, _ = get_dataset(ds)
    print(f"{ds}: train={X_train.shape}, test={X_test.shape}, channels={len(cols)}")
```

## Cold-Start Status

- Repository health checks are maintained and currently pass (`pytest`, paper-number verification).
- End-to-end experiment reruns still require manual dataset acquisition/preprocessing for
  TEP, SMD, MSL, PSM, and HAI.
- A full clean-machine reproduction of every paper experiment has not yet been documented in
  this repository.
- The intended source of truth for the paper is `paper/ARTIFACT_MAP.md` plus the canonical
  table files under `paper/tables/`.
