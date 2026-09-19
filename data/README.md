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
- **Files used:** official `train/machine-1-1.txt`,
  `test/machine-1-1.txt`, and `test_label/machine-1-1.txt`
- **Expected root:** `data/raw/omni_temp/ServerMachineDataset/`
- **Channels:** 38
- **Access:** Public (GitHub)

### MSL — excluded from current primary evaluation

The local `MSL_train.npy` / `MSL_test.npy` files are aggregate arrays, not
verified original single-stream sensor files. The original Telemanom data use
per-stream telemetry and encoded commands. Entity identities, concatenation order,
and inherited preprocessing were not recoverable here. No current primary
result uses MSL. See https://github.com/khundman/telemanom#data.

### PSM (Pooled Server Metrics)
- **Source:** eBay/RANSynCoders (Abdulaal et al. KDD 2021)
- **URL:** https://github.com/eBay/RANSynCoders
- **Files used:** official `train.csv`, `test.csv`, and `test_label.csv`
- **Expected root:** `data/raw/psm_temp/data/`
- **Channels:** 25 after excluding the timestamp
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

The legacy combined CSV files are retained only for reproducibility of the
rejected manuscript. New experiments must use official partitions whenever
they are available. Where only a single chronological stream is available,
the split is contiguous and no window may cross a boundary.

Preprocessing rules for revised experiments:
1. Select sensor columns (exclude metadata: datetime, time, etc.)
2. Replace NaN with 0.0
3. Preserve the source's official train/test division, machine/run identity,
   and chronological order
4. Create validation data only from the training side of the final test split
5. Fit StandardScaler on training data only and transform validation/test data
6. Reset online state at every independent machine, run, or experiment

Random stratification is prohibited for time-series experiments because it
places temporally adjacent samples and portions of the same event on both sides
of the evaluation boundary.

See `src/utils/data_loader.py` for exact implementation.

## Verification

After setup, verify all datasets load correctly:

```python
from src.utils.data_loader import get_dataset, list_datasets
print(list_datasets())
for ds in ['smd', 'psm']:
    X_train, y_train, X_test, y_test, cols, _ = get_dataset(ds)
    print(f"{ds}: train={X_train.shape}, test={X_test.shape}, channels={len(cols)}")
```

## Reproduction Status

- Repository health checks are maintained and currently pass (`pytest`, paper-number verification).
- The corrected SMD, PSM, and TEP experiments have been run from the paths
  listed above. A new machine still requires acquisition of the public source
  files because raw data are not redistributed here.
- The revised source of truth is `docs/REVISION.md`, the `causal_*.csv` files,
  and `paper/tables/causal_summary.tex`. Legacy combined CSVs and tables must not
  be substituted.

## Current TEP entry point

Use `experiments/run_causal_tep.py`, not the legacy `load_tep` loader, for the
current evaluation. Both train and held-out runs come from the released Training
RData files: runs 1–5 train; runs 21–25 test; labels use `sample >= 20` as the
fault-onset convention. This is a run-separated subset evaluation, not use of
the provider's separate Testing RData files. The exact convention is recorded
in the experiment manifest. Non-overlapping 50-sample windows reset per run.
