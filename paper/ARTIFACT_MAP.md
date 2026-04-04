# Paper Artifact Provenance Map

Maps every major table and figure in `paper/main.tex` to its source data,
generation script, and experiment configuration.

## Canonical Experiment Stack (Main Comparison)

- **Config:** V1 base — pure PCA scorer, forward-fill reconstruction, RF n=100
- **Tuning:** None (all methods use defaults)
- **Seeds:** 5 (42, 123, 456, 789, 1024) — averages reported
- **Canonical source CSV:** `paper/tables/table2_results_50pct.csv`
- **Generation script:** `experiments/run_pareto.py`

## Extension Experiment Stack (Ablation)

- **Config:** V2 — hybrid scorer (alpha per dataset), linear interp, sharpening, RF n=200
- **Source CSV:** `experiments/results/pareto_v2_summary.csv`
- **Generation script:** `experiments/run_pareto_v2.py`
- **Usage:** Ablation/extension results only, NOT the main comparison

## Table/Figure Provenance

| Paper Table/Figure | Source File | Script | Config | Notes |
|-------------------|------------|--------|--------|-------|
| Table IV (Results 50%) | `paper/tables/table2_results_50pct.csv` | `run_pareto.py` | V1 base | Canonical main comparison |
| Table (BW Sensitivity) | Hardcoded in tex | `run_pareto.py` | V1 base | LaTeX comment notes source |
| Table (Multi-Classifier) | `experiments/results/multi_classifier.csv` | Manual | V1 base | RF/SVM/KNN |
| Table (Component Contrib.) | `experiments/results/component_contribution.csv` | `run_component_contribution.py` | V1 base | Ablation of each component |
| Table (Ablation k,w,lambda) | `paper/tables/table3_ablation.csv` | `run_ablation.py` | V1 base | Hyperparameter sensitivity |
| Table (Reconstruction) | `experiments/results/reconstruction_comparison.csv` | `run_reconstruction_comparison.py` | V1 base + methods | Forward-fill vs linear vs zero |
| Table (Compute Cost) | `experiments/results/compute_profile.json` + `compute_profile_edge.json` | `run_compute_profile.py` | V1 base | Edge=single-thread, Laptop=multi |
| Table (Per-Fault) | `experiments/results/per_fault_breakdown.csv` | `run_per_fault.py` | V1 base | Per-fault TEP F1 |
| Table (Reaction Time) | `paper/tables/reaction_time_comparison.csv` | `run_adaptivity.py` | V1 base | Windows to 20% shift |
| Table (Friedman Ranks) | `experiments/results/friedman_ranks_v1.csv` | `run_statistical_tests_v1.py` | Reads from table2 | Canonical stats |
| Table (Wilcoxon) | `experiments/results/wilcoxon_tests_v1.csv` | `run_statistical_tests_v1.py` | Reads from table2 | Holm-corrected |
| Table (DL Baselines) | `experiments/results/dl_baselines.csv` | `run_dl_baselines.py` | V2 tuned, 20K subsample, 2 seeds | **Side experiment**, not canonical |
| Table (Robustness) | `experiments/results/realtime_simulation.csv` | `run_realtime_simulation.py` | V2 tuned | **Side experiment** |
| Table (Joint/OGD) | `experiments/results/joint_spatiotemporal.csv` | `run_joint_spatiotemporal.py` | V2 tuned | **Side experiment** |
| Table (Synth. Corr.) | `experiments/results/correlation_validation.csv` | `run_correlation_validation.py` | V1 base | Limitation check |
| Table (Adaptive k) | `experiments/results/adaptive_k_ablation.csv` | `run_new_ablations.py` | V1 base | k selection |
| Table (Ensemble) | `experiments/results/ensemble_ablation.csv` | `run_new_ablations.py` | V1 base | Ensemble scoring |
| Fig (Pareto Curves) | `paper/figures/pareto_curves.png` | `run_pareto.py` + plotting | V1 base | Pre-generated |
| Fig (Method Comparison) | `paper/figures/method_comparison_50pct.png` | Plotting script | V1 base | Bar chart |
| Fig (Adaptivity Heatmap) | `paper/figures/adaptivity_heatmap.png` | `run_adaptivity.py` | V1 base | TEP faults |
| Fig (Scalability) | `paper/figures/scalability.png` | `run_scalability.py` | V1 base | Channels vs time |

## Key Distinction

- **Tables using V1 base (canonical):** IV, BW Sensitivity, Multi-Classifier, Component,
  Ablation, Reconstruction, Compute, Per-Fault, Reaction Time, Friedman, Wilcoxon
- **Tables using V2 tuned (side experiments):** DL Baselines, Robustness, Joint/OGD
  - These are clearly labeled in the paper as using "tuned PCA-Triage configuration"
- **Extension results (0.970):** Appear only in ablation context, clearly labeled as
  "targeted extensions (hybrid scoring, linear interpolation, sharpening)"
