# Contributing to PCA-Triage

Thank you for your interest in contributing to PCA-Triage.

## Setup

```bash
git clone https://github.com/ankitlade12/pca-sensor-triage.git
cd pca-sensor-triage
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

All 53 tests should pass in under 5 seconds.

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check src/ tests/ experiments/
```

## Adding a New Baseline

1. Create `src/baselines/your_method.py` implementing a class with `process_stream(data, seed) -> np.ndarray`
2. Add the import to `src/baselines/__init__.py`
3. Add unit tests in `tests/test_baselines.py`
4. Add the method to `experiments/run_pareto.py` for evaluation

## Adding a New Dataset

1. Add a loader function in `src/utils/data_loader.py` following the existing pattern
2. Add it to the `__init__.py` exports
3. Run the Pareto experiment: `python experiments/run_pareto.py`

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Run tests: `python -m pytest tests/ -v`
4. Run linting: `ruff check src/ tests/`
5. Submit a pull request with a clear description

## Experiment Reproducibility

All experiments use fixed random seeds. When adding new experiments:
- Use seeds `[42, 123, 456, 789, 1024]` for consistency
- Save results to `experiments/results/` as CSV or JSON
- Add `flush=True` to print statements for long-running experiments
