#!/usr/bin/env bash
# Download datasets for PCA-Triage experiments.
# Usage: bash data/download_datasets.sh
#
# Some datasets require manual download or institutional access.
# See data/README.md for details.

set -euo pipefail
RAW_DIR="$(cd "$(dirname "$0")" && pwd)/raw"
mkdir -p "$RAW_DIR"

echo "=== PCA-Triage Dataset Download Script ==="
echo "Target directory: $RAW_DIR"
echo ""

# --- TEP (Tennessee Eastman Process) ---
# Source: Harvard Dataverse (Rieth et al. 2017)
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1
echo "[1/6] TEP (Tennessee Eastman Process)"
if [ -f "$RAW_DIR/TEP_FaultFree_Training.RData" ]; then
    echo "  Already exists, skipping."
else
    echo "  MANUAL DOWNLOAD REQUIRED."
    echo "  1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1"
    echo "  2. Download TEP_FaultFree_Training.RData and TEP_Faulty_Training.RData"
    echo "  3. Place both files in: $RAW_DIR/"
fi
echo ""

# --- SMD (Server Machine Dataset) ---
# Source: NetManAIOps/OmniAnomaly (Su et al. KDD 2019)
echo "[2/6] SMD (Server Machine Dataset)"
if [ -f "$RAW_DIR/smd_combined.csv" ]; then
    echo "  Already exists, skipping."
else
    echo "  Downloading from OmniAnomaly repository..."
    echo "  NOTE: Requires combining train/test splits into smd_combined.csv"
    echo "  Source: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset"
    echo "  MANUAL STEP: Combine machine-1-1 train+test CSVs with anomaly labels."
    echo "  See data/README.md for preprocessing details."
fi
echo ""

# --- MSL (Mars Science Laboratory) ---
# Source: khundman/telemanom (Hundman et al. KDD 2018)
echo "[3/6] MSL (Mars Science Laboratory)"
if [ -f "$RAW_DIR/msl_combined.csv" ]; then
    echo "  Already exists, skipping."
else
    echo "  Source: https://github.com/khundman/telemanom"
    echo "  Also available via: https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
    echo "  MANUAL STEP: Combine MSL channels with anomaly labels into msl_combined.csv"
    echo "  See data/README.md for preprocessing details."
fi
echo ""

# --- PSM (Pooled Server Metrics) ---
# Source: eBay/RANSynCoders (Abdulaal et al. KDD 2021)
echo "[4/6] PSM (Pooled Server Metrics)"
if [ -f "$RAW_DIR/psm_combined.csv" ]; then
    echo "  Already exists, skipping."
else
    echo "  Source: https://github.com/eBay/RANSynCoders/tree/main/data"
    echo "  MANUAL STEP: Combine train.csv + test.csv + test_label.csv into psm_combined.csv"
    echo "  See data/README.md for preprocessing details."
fi
echo ""

# --- HAI (HIL-based Augmented ICS) ---
# Source: icsdataset/hai (Shin et al. CSET 2020)
echo "[5/6] HAI (HIL-based Augmented ICS Security Dataset)"
if [ -f "$RAW_DIR/hai_combined.csv" ]; then
    echo "  Already exists, skipping."
else
    echo "  Source: https://github.com/icsdataset/hai"
    echo "  MANUAL STEP: Combine HAI dataset CSVs with attack labels into hai_combined.csv"
    echo "  See data/README.md for preprocessing details."
fi
echo ""

# --- SKAB (Skoltech Anomaly Benchmark) ---
# Source: waico/SKAB (Skoltech 2020)
echo "[6/6] SKAB (Skoltech Anomaly Benchmark)"
if [ -d "$RAW_DIR/skab_repo" ]; then
    echo "  Already exists, skipping."
else
    echo "  Cloning SKAB repository..."
    git clone --depth 1 https://github.com/waico/SKAB.git "$RAW_DIR/skab_repo" 2>/dev/null || {
        echo "  Clone failed. Manual download: https://github.com/waico/SKAB"
        echo "  Place in: $RAW_DIR/skab_repo/"
    }
fi
echo ""

echo "=== Summary ==="
echo "Automated: SKAB (git clone)"
echo "Manual:    TEP (Harvard Dataverse), SMD, MSL, PSM, HAI (GitHub + preprocessing)"
echo ""
echo "After downloading, verify with: python -c 'from src.utils.data_loader import list_datasets; print(list_datasets())'"
