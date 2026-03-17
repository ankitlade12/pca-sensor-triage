# Changelog

All notable changes to PCA-Triage are documented here.

## [0.2.0] - 2026-03-16

### Added
- 4 new datasets: SMD (38 features), synth_high_correlation (30ch), synth_regime_switching (20ch), synth_scalability (200ch)
- Total datasets: 7 (3 real + 1 semi-synthetic + 3 synthetic)
- Makefile with targets: install, test, experiments, smoke, clean
- docs/DESIGN_DECISIONS.md — engineering notebook with 9 key decisions
- docs/CHANGELOG.md — this file
- configs/default.yaml, configs/ablation.yaml — experiment configuration files
- tests/test_baselines.py — 25 tests for all 6 baselines
- tests/test_integration.py — 11 end-to-end + edge case tests
- src/utils/data_loader.py — unified dataset registry with `get_dataset(name)`
- src/utils/plotting.py — publication figure generators
- experiments/run_ablation.py, run_compute_profile.py, run_adaptivity.py, run_scalability.py
- 3 recent paper citations (2024-2025): OSSFS, ML-DSRA, Low-Precision Streaming PCA

### Changed
- README.md upgraded to 548-line comprehensive format (modeled after RGTTA)
- All baselines fixed for NaN edge cases (`.fillna(0.0)` fallback)
- Table 2 now includes Wilcoxon significance markers (* p<0.05)
- All figures updated to 300 DPI

### Fixed
- Reconstruction NaN when channels have zero samples in a window
- NASA Bearing dataset with proper stratified train/test split

## [0.1.0] - 2026-03-16

### Added
- Core PCA-Triage algorithm (pca_triage.py, rate_allocator.py, reconstruction.py, pipeline.py)
- 6 baseline methods (uniform, threshold, variance, random_dropout, mutual_info, attention)
- 17 unit tests for core algorithm
- Pareto experiments on TEP + SKAB (5 seeds, 9 budgets)
- Compute profiling (laptop + edge-simulated)
- Adaptivity heatmaps for 4 TEP fault types
- Ablation studies (k, w, λ, score formula)
- Scalability experiment (10-500 channels)
- 14 publication-quality figures, 3 result tables
- Introduction, Related Work, Method section drafts
- 36 annotated paper references
- Data exploration notebook
