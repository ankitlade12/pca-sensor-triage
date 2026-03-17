.PHONY: install test lint clean experiments figures all

# Setup
install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install pytest pyyaml pyreadr

# Testing
test:
	.venv/bin/python -m pytest tests/ -v

test-quick:
	.venv/bin/python -m pytest tests/test_triage.py -v

# Experiments
pareto:
	.venv/bin/python experiments/run_pareto.py

ablation:
	.venv/bin/python experiments/run_ablation.py

profile:
	.venv/bin/python experiments/run_compute_profile.py

profile-edge:
	.venv/bin/python experiments/run_compute_profile.py --edge

adaptivity:
	.venv/bin/python experiments/run_adaptivity.py

scalability:
	.venv/bin/python experiments/run_scalability.py

experiments: pareto ablation profile adaptivity scalability

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage

# Smoke test
smoke:
	.venv/bin/python -c "from src.triage import TriagePipeline; import numpy as np; \
		X = np.random.randn(500, 20); p = TriagePipeline(n_components=5, window_size=50, budget=0.5); \
		r = p.process_stream(X, seed=42); print(f'OK: {X.shape} -> {r.shape}, windows={len(p.importance_log)}')"
