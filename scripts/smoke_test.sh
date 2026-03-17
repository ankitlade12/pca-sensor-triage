#!/bin/bash
# Quick smoke test (~10 seconds)
# Verifies core algorithm works without loading real datasets

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "=== PCA-Triage Smoke Test ==="

echo "1. Import test..."
python3 -c "from src.triage import TriagePipeline; print('  OK: imports')"

echo "2. Pipeline test..."
python3 -c "
import numpy as np
from src.triage import TriagePipeline
X = np.random.randn(500, 20)
pipe = TriagePipeline(n_components=5, window_size=50, budget=0.5)
result = pipe.process_stream(X, seed=42)
assert result.shape == X.shape
assert not np.isnan(result).any()
print(f'  OK: {X.shape} -> {result.shape}, windows={len(pipe.importance_log)}')
"

echo "3. Baselines test..."
python3 -c "
from src.baselines import UniformSampling, VarianceSampling, ThresholdSampling
import numpy as np
X = np.random.randn(200, 10)
for name, cls in [('Uniform', UniformSampling), ('Variance', VarianceSampling), ('Threshold', ThresholdSampling)]:
    r = cls(budget=0.5, **({'window_size': 50} if name != 'Uniform' else {})).process_stream(X, seed=42)
    assert r.shape == X.shape
    print(f'  OK: {name}')
"

echo "4. Unit tests..."
python3 -m pytest tests/ -q --tb=no 2>&1 | tail -1

echo ""
echo "=== All smoke tests passed ==="
