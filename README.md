# PCA-Driven Adaptive Sensor Triage for Edge AI Inference

A lightweight streaming algorithm that uses incremental PCA loadings to dynamically allocate per-channel sampling rates under bandwidth constraints for IoT sensor networks.

## Key Result

At **50% bandwidth**, PCA-Triage maintains full fault detection accuracy (F1 = 0.962) on the Tennessee Eastman Process benchmark — while running in **0.67 ms per decision** on a single CPU core.

## Project Structure

```
├── src/
│   ├── triage/           # Core PCA-Triage algorithm
│   │   ├── pca_triage.py       # Weighted loadings importance scorer
│   │   ├── rate_allocator.py   # Proportional bandwidth allocation
│   │   ├── reconstruction.py   # Forward-fill / interpolation
│   │   └── pipeline.py         # End-to-end streaming pipeline
│   └── baselines/        # 6 baseline methods
│       ├── uniform.py          # Uniform sampling
│       ├── threshold.py        # Threshold-based adaptive
│       ├── variance.py         # Variance-based adaptive
│       ├── random_dropout.py   # Random channel dropout
│       ├── mutual_info.py      # Mutual information (supervised)
│       └── attention.py        # Self-attention baseline
├── experiments/           # Experiment scripts and results
│   ├── run_pareto.py           # Pareto curve experiment runner
│   └── results/                # CSV/JSON result files
├── paper/
│   ├── sections/          # Paper section drafts (Introduction, Related Work, Method)
│   ├── figures/           # Publication-quality figures (300 DPI)
│   └── tables/            # Results tables (CSV)
├── references/            # 33 annotated papers
├── notebooks/             # Data exploration notebooks
├── tests/                 # Unit tests
└── data/raw/              # Datasets (not tracked in git)
```

## Datasets

1. **Tennessee Eastman Process (TEP)** — 52 sensors, 20 fault types, 250K+ samples
2. **NASA IMS Bearing** — 16 vibration features, bearing degradation cycles
3. **SKAB (Skoltech Anomaly Benchmark)** — 8 sensors, anomaly detection

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

```bash
# Run Pareto curve experiment
python experiments/run_pareto.py

# Run unit tests
python -m pytest tests/ -v
```

## Algorithm

PCA-Triage operates in three steps per window:

1. **Update** incremental PCA on the latest sensor window
2. **Score** each channel: `s_j = sum_i sigma_i * |V[i,j]|^2`
3. **Allocate** sampling rates proportional to importance under budget B

Complexity: O(wdk) per window | Memory: O(wd + kd) | Parameters: 0 (computed, not trained)

## Citation

Paper in preparation. Preprint forthcoming.

## Author

Ankit Hemant Lade
