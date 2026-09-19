# Corrected evaluation — September 2026

The current paper is **A Causal Evaluation of PCA-Guided Sensor-Rate Allocation
under Hard Communication Budgets**. This repository update supersedes the old
README and manuscript claims. It does not represent journal acceptance.

## Current evidence

- Named manuscript and all seven tables: `paper/current/main.tex` and
  `paper/current/manuscript.pdf`.
- Supplementary results: `paper/current/ESM_1.pdf`.
- Journal-neutral scientific source: `paper/main_revised.tex` and
  `paper/supplement_revised.tex`.
- Corrected experiment outputs: `experiments/results/causal_*`.
- Per-window acquisition ledgers: the adjacent `.windows.csv.gz` files.
- Experiment parameters, seeds, source/data checksums: adjacent `.manifest.json`
  files. Raw datasets must be obtained from their original providers.

At 50% bandwidth with retrained detectors, the five-seed TEP weighted F1 is
0.6752 for PCA allocation versus 0.7068 for uniform allocation. The paper makes
no general detection-superiority claim. The evaluation uses replayed process
simulation and server data, not measured hardware energy or deployed networking.

## Corrections

Sampling quotas are decided before the governed window arrives. Reconstruction
uses causal zero-order hold; controller state resets at independent partitions
and simulation runs. Scores normalize covariance participation before smoothing.
Partial windows, constant inputs and saturated rates are handled explicitly.
Communication counts reconcile against actual acquisition masks. Event metrics
preserve run boundaries; missed events accompany conditional delay. MSL aggregate
arrays are excluded from primary evidence because their boundaries were unverified.

The September 17 presentation package adds the named authors, preprint and
AI-use disclosures, and supplementary-resource identification. Its numerical
results match the September audit. The public manuscript copies match that
package byte for byte; local portal metadata and correspondence are not included.

The preprint at https://arxiv.org/abs/2604.05045 is an earlier version whose
numerical results and claims are superseded. Updating GitHub does not update arXiv
or any files already uploaded to a journal portal.

## Verification

Run `python -m pytest tests/ -q`, `ruff check src/ tests/`, and
`python scripts/verify_revised_results.py`. The latter reconciles stored outputs
with acquisition ledgers and checks experiment source hashes without rerunning
the full experiments. See the root README for reproduction commands.

Before publication of this update, 83 tests, lint, and the result/source audit
passed locally. The manuscript ZIP was previously compiled in isolation and its
extracted text matched the supplied PDF on every page.

## Historical artifacts

`docs/archive/README_before_2026-09-06.md`,
`docs/archive/main_before_2026-09-06.tex`, other non-causal experiment outputs,
and `experiments/results/archive_2026-09-06/` are historical only. Do not cite
their numbers as current evidence. `paper/main.tex` now directs readers to the
revised scientific manuscript. Existing legacy experiment scripts are retained
for provenance; use the `run_causal_*` entry points for current experiments.
