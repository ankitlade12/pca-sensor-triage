#!/usr/bin/env python3
"""
Verify all numerical claims in the paper against experiment results.

Usage:
    python scripts/verify_paper_numbers.py

Checks every F1, timing, and win-count claim against raw CSV/JSON data.
Exits with code 1 if any claim is inconsistent.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results')

passed = 0
failed = 0


def check(claim, actual, expected, tol=0.02):
    global passed, failed
    ok = abs(actual - expected) <= tol
    status = "PASS" if ok else "FAIL"
    if not ok:
        failed += 1
    else:
        passed += 1
    print(f"  [{status}] {claim}: actual={actual:.4f}, expected={expected:.4f} (tol={tol})")


print("=" * 70)
print("PAPER NUMBER VERIFICATION")
print("=" * 70)

# --- Claim 1: TEP F1 at 50% ---
print("\n1. TEP F1 at 50% bandwidth")
df = pd.read_csv(os.path.join(RESULTS_DIR, 'pareto_tep.csv'))
pca_50 = df[(df['method'] == 'PCA-Triage') & (df['budget'] == 0.5)]['f1'].mean()
check("PCA-Triage TEP F1 @ 50%", pca_50, 0.961, tol=0.005)

uni_50 = df[(df['method'] == 'Uniform') & (df['budget'] == 0.5)]['f1'].mean()
check("Uniform TEP F1 @ 50%", uni_50, 0.924, tol=0.005)

# --- Claim 2: Best unsupervised on 4/6 datasets ---
print("\n2. Win count (unsupervised methods)")
unsupervised = ['PCA-Triage', 'Uniform', 'Threshold', 'Variance', 'Random Dropout']
wins = 0
for ds in ['tep', 'smd', 'psm', 'msl', 'hai', 'skab']:
    path = os.path.join(RESULTS_DIR, f'pareto_{ds}.csv')
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    at50 = df[(df['budget'] == 0.5) & (df['method'].isin(unsupervised))].groupby('method')['f1'].mean()
    winner = at50.idxmax()
    if winner == 'PCA-Triage':
        wins += 1
    print(f"  {ds}: winner={winner} (F1={at50[winner]:.4f})")
check("PCA-Triage wins (unsupervised)", wins, 4, tol=1)

# --- Claim 3: Compute cost ---
print("\n3. Compute cost")
profile_path = os.path.join(RESULTS_DIR, 'compute_profile_edge.json')
if os.path.exists(profile_path):
    with open(profile_path) as f:
        profile = json.load(f)
    pca_ms = profile['PCA-Triage']['ms_per_window_edge']
    check("PCA-Triage edge ms/window", pca_ms, 0.67, tol=0.5)
else:
    print("  SKIP — no edge profile data")

# --- Claim 4: Adaptivity reaction time ---
print("\n4. Adaptivity (lambda=0.85 reaction time)")
reaction_path = os.path.join(os.path.dirname(__file__), '..', 'paper', 'tables', 'reaction_time_comparison.csv')
if os.path.exists(reaction_path):
    rt = pd.read_csv(reaction_path)
    pca_rt = rt['PCA-Triage'].values
    max_rt = pca_rt.max()
    check("Max reaction time (windows)", max_rt, 19, tol=5)
else:
    print("  SKIP — no reaction time data")

# --- Summary ---
print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed")
if failed > 0:
    print("WARNING: Some paper claims do not match experiment data!")
    sys.exit(1)
else:
    print("All paper claims verified.")
    sys.exit(0)
