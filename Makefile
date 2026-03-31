.PHONY: install test test-quick lint format benchmark benchmark-quick clean smoke figures help

# ──────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────
install:  ## Install all dependencies
	pip install -e ".[dev,stats]"
	pre-commit install

# ──────────────────────────────────────────────────────────────
# Code Quality
# ──────────────────────────────────────────────────────────────
lint:  ## Run ruff linter
	ruff check src/ tests/

format:  ## Auto-format code with ruff
	ruff check src/ tests/ --fix
	ruff format src/ tests/

# ──────────────────────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────────────────────
test:  ## Run full test suite
	python -m pytest tests/ -v --tb=short

test-quick:  ## Run core triage tests only
	python -m pytest tests/test_triage.py -v --tb=short

smoke:  ## Quick smoke test (no data needed)
	python -c "\
		from src.triage import TriagePipeline, HybridScorer; \
		import numpy as np; \
		X = np.random.randn(500, 20); \
		p = TriagePipeline(n_components=5, window_size=50, budget=0.5, scorer='hybrid', alpha=0.7, sharpness=2.0); \
		r = p.process_stream(X, seed=42); \
		assert r.shape == X.shape and not np.isnan(r).any(); \
		print(f'OK: {X.shape} -> {r.shape}, windows={len(p.importance_log)}')"

# ──────────────────────────────────────────────────────────────
# Experiments
# ──────────────────────────────────────────────────────────────
benchmark:  ## Full Pareto sweep: 8 datasets × 6 methods × 9 budgets × 3 seeds (~2-4 hours)
	python -u experiments/run_pareto_v2_full.py

benchmark-quick:  ## Quick benchmark: budget=0.5 only (~15 min)
	python -u experiments/run_pareto_v2.py

pareto:  ## Legacy Pareto experiment
	python experiments/run_pareto.py

ablation:  ## Ablation studies (k, w, lambda)
	python experiments/run_ablation.py

# ──────────────────────────────────────────────────────────────
# Figures & Paper
# ──────────────────────────────────────────────────────────────
figures:  ## Regenerate all publication figures from results
	python paper/generate_all_figures.py

# ──────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────
clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage

clean-data:  ## Remove generated result CSVs (keeps raw data)
	rm -f experiments/results/pareto_*.csv

# ──────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
