# A Causal Evaluation of PCA-Guided Sensor-Rate Allocation

PCA-Triage assigns next-window sampling quotas using an incremental PCA
covariance score. This repository evaluates its limitations as well as its
utility; it does not establish universal or statistically significant superiority.

**Current manuscript:** [PDF](paper/current/manuscript.pdf) and
[editable LaTeX source](paper/current/main.tex).
**Supplementary results:** [PDF](paper/current/ESM_1.pdf).
**Revision and reproducibility record:** [docs/REVISION.md](docs/REVISION.md).

The rejected manuscript, earlier preprint, and pre-audit results are superseded.
The former README is retained only in
[docs/archive/README_before_2026-09-06.md](docs/archive/README_before_2026-09-06.md).
Do not use its numerical results, theory, or deployment claims.

## Method

1. Start each independent stream with uniform allocation under the budget.
2. Acquire the current window using integer quotas fixed before its values arrive.
3. Reconstruct missing values with causal zero-order hold.
4. Update incremental PCA using that reconstructed window.
5. Compute `s_j = sum_i explained_variance_i * loading_ij**2`, normalize to
   participation weights, then smooth. A zero-variance window uses uniform weights.
6. Convert smoothed weights into next-window rates, repeatedly redistributing
   clipped excess, then integerize the quotas to meet the aggregate cap.

The final short window uses the latest computed rates. The minimum rate is a
continuous target floor; integer rounding may reduce a realized channel rate
below it. A covariance diagonal is not a conditional redundancy or fault-relevance
score. Full-rank scores equal marginal variances of the same covariance history;
this is not identical to the rolling-variance baseline used here.

## Evaluation

Primary data: TEP (all 20 faults, separate simulation runs), SMD machine-1-1,
and PSM. MSL is excluded because the aggregate arrays do not preserve verified
entity boundaries and mix telemetry with encoded command variables.

Train/test scaling is fitted only to training observations. All controller state
resets at every independent partition or run. Both retrained and fixed downstream
detectors are evaluated. Five joint seeds vary acquisition and detector randomness.
Anomaly thresholds are fitted from training scores; a predeclared sensitivity sweep
does not select settings from test outcomes.

Communication is counted from actual acquisition masks, with per-window ledgers.
Reported payload bytes assume four bytes per value and exclude network overhead.
Event metrics preserve independent run boundaries and report missed events next
to conditional detection delay. Full results include distortion, detection,
false alarms, delay, and payload, in the manuscript and supplementary tables.

## Reproduce

Use Python 3.10+ and install this project in a virtual environment:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
```

Acquire the source files described in [data/README.md](data/README.md), then run
from the repository root:

```sh
PYTHONPATH=. python experiments/run_causal_benchmark.py
PYTHONPATH=. python experiments/run_causal_tep.py
python scripts/generate_revised_tables.py
python scripts/verify_revised_results.py
make revised-paper
# Compile the current named manuscript:
cd paper/current && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The anomaly script produces `causal_benchmark.csv`, `.windows.csv.gz`,
`.thresholds.csv`, and `.manifest.json`. TEP produces the analogous main results,
per-fault results, window ledger, and manifest. Manifests record source and data
SHA-256 hashes, package versions, fixed parameters, and seed interpretation.

`experiments/results/archive_2026-09-06/` preserves the earlier causal evaluation;
it lacks the September audit corrections and is not current submission evidence.
Other experiment scripts and manuscript variants are legacy research artifacts.

## Authors and license

Ankit Hemant Lade, Sai Krishna Jasti, Nikhil Sinha, Indar Kumar, Akanksha Tiwari.
Code is distributed under the [MIT license](LICENSE). Dataset terms remain those
of the original providers. The development repository is
https://github.com/ankitlade12/pca-sensor-triage; the accompanying source archive
identifies the exact evaluated revision independently of repository publication.
