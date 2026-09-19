#!/usr/bin/env bash
# Fetch the current released SMD-1-1 and PSM partitions; no combined/legacy CSVs.
set -euo pipefail
PCA_RAW_DIR="$(cd "$(dirname "$0")" && pwd)/raw"
for part in train test test_label; do
  mkdir -p "$PCA_RAW_DIR/omni_temp/ServerMachineDataset/$part"
  path="$PCA_RAW_DIR/omni_temp/ServerMachineDataset/$part/machine-1-1.txt"
  if [ ! -s "$path" ]; then
    curl --fail --location --retry 3 \
      "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset/$part/machine-1-1.txt" \
      --output "$path"
  fi
  mkdir -p "$PCA_RAW_DIR/psm_temp/data"
  path="$PCA_RAW_DIR/psm_temp/data/$part.csv"
  if [ ! -s "$path" ]; then
    curl --fail --location --retry 3 \
      "https://raw.githubusercontent.com/eBay/RANSynCoders/main/data/$part.csv" \
      --output "$path"
  fi
done
cat <<'TXT'
SMD-1-1 and PSM released partitions are ready.
TEP requires TEP_FaultFree_Training.RData and TEP_Faulty_Training.RData in data/raw/.
Obtain these from https://doi.org/10.7910/DVN/6C3JR1 under the provider's terms.
MSL is excluded from the primary sensor-allocation evaluation.
TXT
