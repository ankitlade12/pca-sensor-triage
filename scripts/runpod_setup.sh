#!/bin/bash
# RunPod setup script for PCA-Triage experiments
# Usage: Run this after SSH into a RunPod instance
# Recommended pod: CPU-optimized or any GPU pod (we use CPU cores, not GPU)

set -e

echo "=== PCA-Triage RunPod Setup ==="

# Install system deps
apt-get update -qq && apt-get install -y -qq git python3-pip python3-venv > /dev/null 2>&1

# Clone repo
cd /workspace
if [ ! -d "pca-sensor-triage" ]; then
    git clone https://github.com/ankitlade12/pca-sensor-triage.git
fi
cd pca-sensor-triage

# Install Python deps
pip install -e ".[dev,stats]" -q
pip install xgboost -q

# Verify
python3 -c "from src.triage import TriagePipeline; print('Setup OK')"

echo ""
echo "=== Setup Complete ==="
echo "CPU cores available: $(nproc)"
echo ""
echo "Run experiments with:"
echo "  python3 -u experiments/run_pareto_v2_full.py"
echo ""
echo "Or run just WADI:"
echo "  python3 -u scripts/run_wadi_only.py"
