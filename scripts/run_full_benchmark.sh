#!/bin/bash
# Run the complete benchmark suite (all datasets, all experiments)
# Estimated time: 30-60 minutes depending on hardware
#
# Usage:
#   ./scripts/run_full_benchmark.sh
#   ./scripts/run_full_benchmark.sh --quick  # TEP only (~5 min)

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

QUICK=false
if [ "$1" == "--quick" ]; then
    QUICK=true
    echo "=== QUICK MODE: TEP only ==="
fi

echo "=== PCA-Triage Full Benchmark ==="
echo "Start: $(date)"
echo ""

# 1. Pareto curves
echo "--- Experiment 1: Pareto Curves ---"
python experiments/run_pareto.py
echo ""

# 2. Compute profiling
echo "--- Experiment 2: Compute Profiling ---"
python experiments/run_compute_profile.py
python experiments/run_compute_profile.py --edge
echo ""

# 3. Adaptivity
echo "--- Experiment 3: Adaptivity Analysis ---"
python experiments/run_adaptivity.py
echo ""

# 4. Ablation studies
echo "--- Experiment 4: Ablation Studies ---"
python experiments/run_ablation.py
echo ""

# 5. Scalability
echo "--- Experiment 5: Scalability ---"
python experiments/run_scalability.py
echo ""

# 6. Generate figures
echo "--- Generating Figures ---"
python paper/generate_all_figures.py
echo ""

# 7. Verify paper numbers
echo "--- Verifying Paper Numbers ---"
python scripts/verify_paper_numbers.py
echo ""

# 8. Result analysis
echo "--- Result Analysis ---"
python scripts/analyze_results.py
echo ""

echo "=== Benchmark Complete ==="
echo "End: $(date)"
echo "Results: experiments/results/"
echo "Figures: paper/figures/"
