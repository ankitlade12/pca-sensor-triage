"""Run ALL experiments on cloud (RunPod/GCP). Full Pareto sweep for all 8 datasets."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from run_pareto_v2_full import run_dataset_pareto

if __name__ == "__main__":
    print(f"CPU cores: {os.cpu_count()}")

    datasets = ['tep', 'smd', 'msl', 'psm', 'hai', 'skab', 'swat', 'wadi']
    for ds in datasets:
        try:
            run_dataset_pareto(ds)
        except Exception as e:
            print(f"ERROR on {ds}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\nAll done! Download experiments/results/pareto_*.csv")
