"""Run just WADI Pareto sweep (the bottleneck dataset)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# Add experiments to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from run_pareto_v2_full import run_dataset_pareto

if __name__ == "__main__":
    print(f"CPU cores: {os.cpu_count()}")
    run_dataset_pareto('wadi')
    print("\nDone! Download results/pareto_wadi.csv")
