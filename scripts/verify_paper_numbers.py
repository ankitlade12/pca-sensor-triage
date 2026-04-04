#!/usr/bin/env python3
"""
Verify all numerical claims in the paper against canonical experiment results.

Usage:
    python scripts/verify_paper_numbers.py

Canonical source: paper/tables/table2_results_50pct.csv (V1 base, RF-100)
NOT experiments/results/pareto_{ds}.csv (V2 tuned — not the paper's main story)

Exits with code 1 if any claim is inconsistent.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

PAPER_TABLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'tables')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results')

passed = 0
failed = 0


def check(claim, actual, expected, tol=0.005):
    global passed, failed
    ok = abs(actual - expected) <= tol
    status = "PASS" if ok else "FAIL"
    if not ok:
        failed += 1
    else:
        passed += 1
    print(f"  [{status}] {claim}: actual={actual:.4f}, expected={expected:.4f} (tol={tol})")


def check_eq(claim, actual, expected):
    global passed, failed
    ok = actual == expected
    status = "PASS" if ok else "FAIL"
    if not ok:
        failed += 1
    else:
        passed += 1
    print(f"  [{status}] {claim}: actual={actual}, expected={expected}")


print("=" * 70)
print("PAPER NUMBER VERIFICATION (canonical V1 base)")
print(f"Source: paper/tables/table2_results_50pct.csv")
print("=" * 70)

# Load canonical source
t2 = pd.read_csv(os.path.join(PAPER_TABLES_DIR, 'table2_results_50pct.csv'))

# --- Claim 1: TEP F1 at 50% ---
print("\n1. TEP F1 at 50% bandwidth (paper abstract + Table IV)")
tep = t2[t2['dataset'] == 'tep'].iloc[0]
check("PCA-Triage TEP F1 @ 50%", tep['PCA_f1_mean'], 0.961)
check("Uniform TEP F1 @ 50%", tep['Uniform_f1'], 0.924)
check("Threshold TEP F1 @ 50%", tep['Threshold_f1'], 0.958)
check("Variance TEP F1 @ 50%", tep['Variance_f1'], 0.948)

# --- Claim 2: Best unsupervised on 3/6 datasets ---
print("\n2. Win count (unsupervised methods, from canonical table2)")
col_map = {
    'PCA-Triage': 'PCA_f1_mean',
    'Uniform': 'Uniform_f1',
    'Threshold': 'Threshold_f1',
    'Variance': 'Variance_f1',
    'Random Dropout': 'Random Dropout_f1',
}
wins = 0
for _, row in t2.iterrows():
    ds = row['dataset']
    scores = {m: row[c] for m, c in col_map.items()}
    winner = max(scores, key=scores.get)
    if winner == 'PCA-Triage':
        wins += 1
    print(f"  {ds}: winner={winner} (F1={scores[winner]:.4f})")
check_eq("PCA-Triage wins (unsupervised)", wins, 3)

# --- Claim 3: Compute cost ---
print("\n3. Compute cost (edge simulation)")
profile_path = os.path.join(RESULTS_DIR, 'compute_profile_edge.json')
if os.path.exists(profile_path):
    with open(profile_path) as f:
        profile = json.load(f)
    pca_ms = profile['PCA-Triage']['ms_per_window_edge']
    check("PCA-Triage edge ms/window", pca_ms, 0.67, tol=0.05)
else:
    print("  SKIP — no edge profile data")

# --- Claim 4: Statistical tests from canonical source ---
print("\n4. Statistical tests (Friedman + Wilcoxon from table2)")
stats_path = os.path.join(RESULTS_DIR, 'friedman_ranks_v1.csv')
if os.path.exists(stats_path):
    fr = pd.read_csv(stats_path)
    pca_rank = fr[fr['method'] == 'PCA-Triage']['mean_rank'].values[0]
    check("PCA-Triage Friedman mean rank", pca_rank, 1.50, tol=0.01)
else:
    print("  SKIP — no friedman_ranks_v1.csv")

wilcox_path = os.path.join(RESULTS_DIR, 'wilcoxon_tests_v1.csv')
if os.path.exists(wilcox_path):
    wt = pd.read_csv(wilcox_path)
    thr_row = wt[wt['baseline'] == 'Threshold'].iloc[0]
    check_eq("vs Threshold W/L", f"{int(thr_row['wins'])}/{int(thr_row['losses'])}", "6/0")
else:
    print("  SKIP — no wilcoxon_tests_v1.csv")

# --- Claim 5: Adaptivity reaction time ---
print("\n5. Adaptivity reaction time")
reaction_path = os.path.join(PAPER_TABLES_DIR, 'reaction_time_comparison.csv')
if os.path.exists(reaction_path):
    rt = pd.read_csv(reaction_path)
    pca_max = rt['PCA-Triage'].max()
    check("Max reaction time at lambda=0.85 (windows)", pca_max, 19, tol=1)
else:
    print("  SKIP — no reaction time data")

# --- Claim 6: Dataset count ---
print("\n6. Dataset/benchmark counts")
check_eq("Datasets in canonical table", len(t2), 6)

# --- Summary ---
print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed")
if failed > 0:
    print("FAIL: Some paper claims do not match canonical data!")
    sys.exit(1)
else:
    print("All paper claims verified against canonical source.")
    sys.exit(0)
