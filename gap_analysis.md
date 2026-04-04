# PCA-Triage: Comprehensive Gap Analysis for IEEE Publication

**Date:** 2026-04-03
**Last updated:** 2026-04-04
**Paper:** PCA-Driven Adaptive Sensor Triage for Edge AI Inference
**Authors:** Ankit Hemant Lade, Sai Krishna Jasti, Indar Kumar, Akanksha Tiwari, Nikhil Sinha
**Status:** Improving, but not publication-ready — main remaining blockers are result
provenance, README/paper drift, and incomplete reproducibility closure

---

## 1. Research Objective and Problem Statement

### 1.1 Problem

Multi-channel sensor networks in industrial IoT generate data volumes exceeding available
bandwidth. A chemical plant with 200 sensors at 1 Hz produces ~1.2 MB/min, but available
bandwidth may support only 50%. The question is: which sensors should get more bandwidth
and which can tolerate lower sampling rates without compromising fault detection?

### 1.2 Formal Formulation

- **Input:** d sensor channels, bandwidth budget B in (0, 1]
- **Output:** Per-channel sampling rate r_j in [r_min, 1] for each channel j = 1,...,d
- **Constraint:** (1/d) * sum(r_j) <= B
- **Goal:** Downstream fault detection trained on triaged data maintains accuracy close
  to full-rate data, updated online as new data arrives

### 1.3 Proposed Solution: PCA-Triage

A streaming algorithm that converts incremental PCA loadings into proportional per-channel
sampling rates. Per window, it:

1. Updates an IncrementalPCA model (partial_fit)
2. Computes channel importance: s_j = sum_i(sigma_i * |V_ij|^2)
3. Applies exponential smoothing: s_bar_j(t) = lambda * s_bar_j(t-1) + (1-lambda) * s_j(t)
4. Allocates rates proportional to importance under budget constraint
5. Acquires samples stochastically per-channel
6. Reconstructs missing values via linear interpolation

**Key properties:** O(wdk) time, O(wd + kd) space, zero trainable parameters, unsupervised.

### 1.4 Claimed Contributions

1. A streaming algorithm converting incremental PCA loadings to per-channel bandwidth allocation
2. Theoretical properties covering budget feasibility, convergence, correlation structure,
   reconstruction error, and adaptation behavior
3. 17 experiments on 7 benchmarks against 9 baselines; best unsupervised on 3/6 canonical
   datasets at 50% bandwidth, with targeted extensions improving TEP further
4. 0.67 ms/decision, edge-deployable, robust to packet loss and moderate sensor noise

---

## 2. Current State of the Work

### 2.1 What Exists

| Artifact                   | Status    | Details                                              |
|----------------------------|-----------|------------------------------------------------------|
| Core algorithm             | Complete  | PCATriage, HybridScorer, RateAllocator, Reconstruction |
| Baseline implementations   | Partial   | Most claimed baselines implemented; OGD missing from public baseline package |
| Paper (LaTeX)              | Draft     | ~1100 lines, IEEEtran format, all sections present   |
| Experiments                | Broad     | 17 scripts plus saved result CSVs                    |
| Unit tests                 | Good      | 64 tests, all passing                                |
| Figures                    | Complete  | 14 publication-quality figures at 300 DPI             |
| References                 | 39 cited  | Mix of foundational and recent (up to 2025)           |
| Raw data                   | Missing   | data/raw/ contains only .gitkeep                      |

### 2.2 Headline Results (current paper framing)

Note: the manuscript now uses the V1/base story as the canonical main result, but the repo
still does **not** provide a perfectly clean artifact path proving that every canonical paper
number is generated from one unambiguous source file set. Treat the table below as the
paper's current framing, not as fully closed provenance.

| Dataset | PCA-Triage F1 | Best Baseline F1 | Margin  | Current reading |
|---------|---------------|-------------------|---------|-----------------|
| TEP     | 0.961 +/- 0.001 | Threshold 0.958 | +0.3%  | Best unsupervised |
| SMD     | 0.982 +/- 0.001 | Threshold 0.981 | +0.1%  | Best unsupervised |
| MSL     | 0.921 +/- 0.004 | Variance 0.917  | +0.4%  | Best unsupervised |
| PSM     | 0.959 +/- 0.001 | Rand Drop 0.962 | -0.3%  | Narrow second     |
| HAI     | 1.000 +/- 0.000 | Variance 1.000  | Tie     | Saturated         |
| SKAB    | 0.583 +/- 0.013 | Uniform 0.586   | -0.3%  | Slight loss       |
| SWaT    | 0.986 +/- 0.000 | Rand Drop 0.998 | -1.2%  | Extension-only synthetic comparison |

---

## 3. CRITICAL GAPS (Integrity / Rejection-Level Issues)

These are issues that individually could cause desk rejection or reject decisions. They must
be resolved before any submission.

### GAP-C1: Internal Result Inconsistencies

**Severity: CRITICAL**

Three different TEP F1 numbers exist across the repo for PCA-Triage at 50% bandwidth:

| Source                         | TEP F1  | Notes                            |
|--------------------------------|---------|----------------------------------|
| paper/main.tex (abstract, L66) | 0.970   | Headline claim                   |
| paper/main.tex (Table IV, L482)| 0.970   | Matches abstract                 |
| paper/tables/table2_results_50pct.csv | 0.9614 | "Base" PCA-Triage results  |
| experiments/results/pareto_v2_summary.csv | 0.9709 | "v2" with hybrid+sharpening |
| paper/main.tex (multi-classifier, L513) | 0.962 | RF classifier table        |

The paper presents 0.970 as the main result, which comes from the v2 experiment (hybrid
scoring with alpha=0.8, sharpening gamma=1.5). But the base version (without per-dataset
tuning) scores 0.961, and the multi-classifier table also shows 0.962. A reviewer who
checks the CSVs will see three competing numbers and question which is the real result.

**Required fix:** Consolidate to ONE canonical result. Either:
- (a) Use the v2 results everywhere and clearly state the configuration (hybrid, tuned alpha), OR
- (b) Use the base results (0.961) and present tuned results as an ablation improvement

> **PARTIALLY RESOLVED** (commit a14147c, 2026-04-03; audited again 2026-04-04)
> 
> Chose option (b): V1 base (0.961) is now the canonical main result.
> 
> **What is genuinely fixed in paper/main.tex:**
> - Abstract: F1 0.970 → 0.961, "exceeding" → "matching" full-data. V2 innovations
>   reframed as "targeted extensions" that push to 0.970.
> - Table IV: Replaced all V2 numbers with V1 base from table2_results_50pct.csv.
>   Removed SWaT column (not in V1 experiments). Now 6 datasets, 5 seeds.
> - Win count corrected: 4/7 → 3/6 (TEP, SMD, MSL). PSM goes to Random Dropout
>   with V1 numbers (0.959 vs 0.962).
> - Contributions: "the first streaming algorithm" → "a streaming algorithm";
>   innovations reframed as extensions validated via ablation.
> - Figure caption, discussion, conclusion all updated consistently.
> - DL baselines caveated as subsampled conditions (20K, 2 seeds).
> - Robustness/joint experiment tables labeled as using tuned configuration.
> - Statistical section text updated to 6 datasets, flagged for recomputation.
> 
> **What remains unresolved after direct repo audit:**
> - `experiments/run_statistical_tests_v1.py` still reads from `experiments/results/pareto_{dataset}.csv`
>   as if those are the canonical V1/base artifacts.
> - But `experiments/results/pareto_tep.csv` still contains ~0.97 TEP values at 50% bandwidth,
>   which look like tuned/V2-style outputs rather than the canonical 0.961 base result.
> - So the manuscript framing is much cleaner, but the artifact provenance is still not fully closed.

### GAP-C2: Adaptivity Claim Directly Contradicts Paper's Own Table

**Severity: CRITICAL**

The paper text (line 558) states:
> "With lambda = 0.85, PCA-Triage detects importance shifts within 0-3 windows for all
> tested faults."

But the paper's own Table (lines 564-572) shows:

| Fault        | PCA-Triage | Variance | Threshold |
|--------------|------------|----------|-----------|
| IDV(1)       | **19**     | 0        | 19        |
| IDV(2)       | **19**     | 19       | 19        |
| IDV(4)       | **19**     | 19       | 19        |
| IDV(5)       | **19**     | 19       | 19        |

PCA-Triage shows 19 windows on ALL faults — not 0-3. The reaction_time_comparison.csv
file confirms the same numbers. This is a direct factual contradiction within the same
manuscript. Any reviewer who reads both the text and the table will flag this.

**Root cause:** The 0-3 window claim may be from a different lambda value or experimental
setup than what's recorded in the table. The table may have been generated with lambda=1.0
(where 19 windows = w-1 for window_size=20 with no forgetting).

**Required fix:** Re-run reaction time experiments explicitly at lambda=0.85 and update both
the table and the claim to be consistent. If the figure (reaction_time_vs_lambda.png) shows
0-3 windows at lambda=0.85 correctly, the table data needs to match that lambda value.

> **RESOLVED** (commit 1ffa1ef, 2026-04-03)
> 
> The table data (19 windows) was kept as-is — it accurately reflects what lambda=0.85
> produces at the 20% shift threshold. The text was corrected to match the table.
> 
> **What changed in paper/main.tex:**
> - Removed the unqualified "within 1-3 windows (lambda=0.85)" claim from adaptivity text
> - Reaction time analysis text now says "up to 19 windows (one full window cycle)"
>   and points to figure for lower lambda behavior
> - All "0-3 windows" claims now qualified with "lambda <= 0.80" and reference the figure
> - Edge viability contribution: "adapts to fault onset within 1-3 windows" → "with speed
>   controlled by lambda (0-3 windows at lambda <= 0.80)"
> - Discussion lambda trade-off: lambda=0.85 no longer claimed as 0-3 windows; changed
>   to "lambda <= 0.80"
> - Ablation text: same correction applied
> - Adaptivity heatmap caption: removed specific "1-3 windows" claim
> 
> **Verification:** Grepped for all "1--3 windows" and "0--3 windows" — all remaining
> instances are properly qualified with lambda <= 0.80 or reference the figure.

### GAP-C3: Theorem 1 Proof Contradicts Its Own Conclusion

**Severity: CRITICAL**

**Theorem statement (line 291):** PCA "concentrates bandwidth on the channel with higher
loading in the first principal component, freeing bandwidth for c."

**Proof derivation (lines 311-312):** With k=2 (truncated PCA):
- s_a = s_b = sigma^2 (correlated channels retain importance)
- **s_c = 0** (independent channel gets ZERO importance)

The proof shows PCA **silences** the independent channel c (gives it zero importance and
thus minimum rate r_min), which is the **opposite** of "freeing bandwidth for c." PCA
actually redirects ALL proportional budget to the correlated pair a,b — which carry
**redundant** information.

The paper then says (line 314): "This motivates our choice of moderate k" — acknowledging
the problem but not fixing the theorem statement.

**Required fix:** Either:
- (a) Restate the theorem honestly: "PCA concentrates bandwidth on channels in the dominant
  covariance structure, potentially starving independent channels when k < d. This motivates
  moderate k selection." OR
- (b) Remove the theorem and replace with an empirical observation about correlation exploitation

> **RESOLVED** (commit 1ffa1ef, 2026-04-03)
> 
> Chose option (a): Theorem restated honestly.
> 
> **What changed in paper/main.tex:**
> - Theorem title: "PCA Advantage Under Correlation" → "PCA Distinguishes Correlated
>   from Independent Channels"
> - Statement item 2 rewritten: now honestly states that with k < d, PCA assigns s_c = 0
>   to channels outside the top-k eigenspace (instead of claiming it "frees bandwidth for c")
> - Proof kept as-is (the math was always correct)
> - Added new "Interpretation" paragraph after the proof that explicitly discusses both
>   the power (detects correlation) and the risk (starves independent channels with low k)
> - Interpretation motivates two design choices: moderate k and hybrid scorer
> - Theory section intro updated: "five properties" → "four properties", removed regret
>   bound reference
> 
> **Verification:** The theorem statement now matches its own proof. No contradictions.

### GAP-C4: Regret Bound Lacks Rigorous Derivation

**Severity: HIGH**

Proposition 3 (lines 328-339) claims a regret bound:

    Regret(t) <= 2*B*d*Delta_max^2*epsilon_k / (1 - epsilon_k)

The "proof" (line 338) asserts: "The maximum per-channel allocation error is bounded by
|r_j - r_j*| <= 2B * epsilon_k / (1 - epsilon_k)" without derivation. This key inequality
is stated as fact, not proven. The oracle is defined as allocating proportional to Delta_j^2,
but PCA-Triage allocates proportional to sum(sigma_i * V_ij^2) — the connection between
these two quantities (and why the residual fraction epsilon_k bounds the allocation error)
is hand-waved.

**Required fix:** Either provide a complete derivation with all intermediate steps, or
demote this to a "heuristic bound" / "empirical observation" and remove the Proposition label.

> **RESOLVED** (commit 1ffa1ef, 2026-04-03)
> 
> Demoted from formal Proposition to informal "Regret intuition" paragraph. Removed
> the Proposition label, the unproven bound equation, and the handwaved proof. Kept the
> key insight: as k increases and epsilon_k decreases, allocation approaches oracle.
> TEP example (epsilon_k ~ 0.05) retained as empirical support.

### GAP-C5: Attention Baseline Uses Random Projections, Not Learned Attention

**Severity: CRITICAL**

The paper claims (line 405): "Transformer-Attention: A single Transformer encoder layer
where channels are tokens (12K params). Mean self-attention received per channel serves
as importance."

But src/baselines/attention.py (lines 49-55) initializes W_q, W_k, W_v with:

    rng = np.random.RandomState(42)
    self.W_q = rng.randn(d, h) / np.sqrt(h)
    self.W_k = rng.randn(d, h) / np.sqrt(h)
    self.W_v = rng.randn(d, h) / np.sqrt(h)

These matrices are **never updated or trained** — they remain fixed random projections for
the entire stream. This is NOT a learned attention mechanism. It computes attention scores
using random weights, making it essentially a random nonlinear baseline.

The paper's claim "beats attention-based methods" is therefore misleading. The basic
attention baseline in the main Pareto comparison is a random-projection model.

Note: The LSTM-Attention and Transformer-Attention in run_dl_baselines.py
(src/baselines/lstm_attention.py) ARE properly trained per-window and are separate from
this baseline. But those are only used in Experiment 14 on subsampled data (see GAP-C6).

**Required fix:** Either:
- (a) Train the attention weights per-window (SGD on reconstruction loss, like the autoencoder), OR
- (b) Rename to "Random Projection" and remove "attention-based" superiority claims from
  the main comparison, OR
- (c) Move the real LSTM-Attention/Transformer-Attention into the main comparison

> **RESOLVED** (commit 7ef2488, 2026-04-03)
> 
> Chose option (b): Renamed to "Random Projection" in the paper, clarified docstrings in code.
> 
> **What changed in paper/main.tex:**
> - Compute cost table (Table IX): "Attention" row → "Random Proj."
> - Scalability text: "Attention-based methods" → "Random-projection attention"
> - Scalability figure caption: "Attention" → "Random projection"
> 
> **What changed in code (src/baselines/attention.py):**
> - Module docstring updated: now explicitly says "FIXED random projection matrices"
>   and "NOT a learned attention model"
> - Class docstring updated: "Self-attention based" → "Random-projection attention"
> - Points readers to lstm_attention.py for trained attention baselines
> - Class name kept as AttentionSampling for backward compatibility
> 
> **Key clarification:** The attention baseline only appeared in the compute cost
> table (Table IX), never in the main F1 comparison (Table IV). The paper's DL
> attention claims (Experiment 14) refer to LSTM-Attention and Transformer-Attention
> in lstm_attention.py, which ARE properly trained per-window via PyTorch.
> 
> **Verification:** 64 tests pass. No "beats attention" claims reference the
> random-projection baseline.

### GAP-C6: Deep Learning Comparison Is Not Comparable to Main Results

**Severity: HIGH**

The DL experiment (run_dl_baselines.py) uses a fundamentally different setup:
- **Subsampled to 20,000 points** (line 60), vs full dataset in main experiments
- **Only 2 seeds** (line 30: seeds=[42, 123]), vs 3-5 in main experiments
- PCA-Triage TEP F1 in DL setting: **0.811** (dl_baselines.csv)
- PCA-Triage TEP F1 in main setting: **0.970** (paper Table IV)

The paper presents these as comparable results (Table XV, line 877: "PCA-T: .811" for TEP).
But a 16-point F1 gap between the same method in two experimental setups shows these are
measuring fundamentally different things. The claim "PCA-Triage outperforms DL baselines
on 5/7 datasets" is only valid within the subsampled context, not generalizable to the
main evaluation.

**Required fix:** Either:
- (a) Run DL baselines on full data with 5 seeds (compute-intensive but necessary for
  the claim), OR
- (b) Explicitly caveat: "Under subsampled conditions where all methods are handicapped,
  PCA-Triage degrades more gracefully than DL methods" — a weaker but honest claim, OR
- (c) Remove the "5/7 datasets" claim from the abstract and conclusion

> **RESOLVED** (via GAP-C1 fix, commit a14147c)
> 
> Chose option (b): DL baselines findings text rewritten to explicitly state
> "Under subsampled conditions (20K samples, 2 seeds)" and notes that PCA-Triage
> TEP F1 is 0.811 here vs 0.961 on full data. Framed as relative robustness to
> data scarcity, not absolute performance. Table caption adds "under subsampled
> conditions." The "5/7 datasets" claim removed from abstract and conclusion.

### GAP-C7: Reproducibility Claim Is False

**Severity: CRITICAL**

The paper states (line 1095):
> "All code, datasets, and experiment configurations are publicly available."

But data/raw/ contains only .gitkeep — zero dataset files. The data loaders expect:
- TEP: R DataFrames via pyreadr (institutional/downloaded data)
- SMD, MSL, PSM, HAI: External downloads from various sources
- SKAB: CSV files from Skoltech
- NASA: Labeled as synthetic, generated internally

A reviewer who clones the repo cannot run a single experiment. This alone can be grounds
for desk rejection at venues that require reproducibility artifacts.

**Required fix:**
- Add download scripts (data/download.sh or Makefile target) with exact URLs
- Document exact preprocessing steps and expected file checksums
- For datasets requiring institutional access (SWaT), state this clearly
- Ensure the environment (requirements.txt or pyproject.toml) pins exact versions
- Test a cold-start installation on a fresh environment

> **PARTIALLY RESOLVED** (commit 3107d94, 2026-04-03; audited again 2026-04-04)
> 
> **What was created:**
> - `uv.lock`: Lock file for the current dependency set
> - `data/download_datasets.sh`: Download script with URLs for all 6 real datasets
>   (SKAB automated via git clone; TEP, SMD, MSL, PSM, HAI require manual download
>   with documented URLs and preprocessing steps)
> - `data/README.md`: Complete dataset documentation with URLs, file formats, expected
>   columns, preprocessing steps, and verification instructions
> 
> **What changed in paper/main.tex:**
> - Reproducibility section rewritten: "All code, datasets, and experiment
>   configurations are publicly available" → honest language about code+CSVs being
>   available, datasets downloadable from original sources via provided script,
>   with lock file and download instructions
> 
> **What remains unresolved:** Full end-to-end cold-start validation has still not been
> performed, and most datasets still require manual download/manual preprocessing. The
> reproducibility claim is now much more honest, but not fully closed.

### GAP-C8: NASA Dataset Mislabeled

**Severity: MEDIUM-HIGH**

The paper's dataset table (line 380) lists NASA as a real benchmark:

    NASA | Bearing degradation | 16 | 1K | NASA IMS

No dagger (synthetic) marker is applied. However, data_loader.py (line 168) explicitly
labels it:

    def load_nasa_bearing(...):
        """Load NASA Bearing (synthetic) dataset."""

The file loaded is `nasa_bearing_synthetic.csv`. Presenting a synthetic dataset as a real
NASA benchmark misrepresents the evaluation breadth.

**Required fix:** Add dagger marker to NASA in the paper's table, or replace with genuinely
real NASA IMS data (publicly available from NASA's Prognostics Data Repository).

> **RESOLVED** (commit a14147c, 2026-04-03)
> 
> Decision: Removed NASA from the paper entirely.
> 
> Rationale: The data file (nasa_bearing_synthetic.csv) does not exist in the repo,
> no generation script exists, NASA was not in any main experiment (run_pareto_v2.py
> uses 6 datasets excluding NASA), and it only contributed one qualitative adaptivity
> figure. Real NASA IMS data is high-frequency vibration — domain-mismatched with the
> paper's process/telemetry focus.
> 
> **What changed in paper/main.tex:**
> - Removed NASA row from dataset table (was line 380)
> - Removed cross-domain adaptivity paragraph + nasa_adaptivity_heatmap figure
> - Updated all counts: "8 benchmarks" → "7 benchmarks" (abstract, contributions,
>   setup, threats, conclusion)
> - Updated "6 application domains" → "5 application domains"
> - Updated "7 real-world benchmarks" → "6 real-world benchmarks"
> 
> **Verification:** Grepped for "NASA", "bearing", "8 benchmark", "fig:nasa" — zero
> orphaned references remain.

---

## 4. MAJOR GAPS (Methodology / Major-Revision-Level Issues)

These issues won't cause immediate rejection but will likely result in "major revision"
if not addressed proactively.

### GAP-M1: Statistical Methodology Has Multiple Flaws

**Severity: HIGH**

**(a) Friedman test is not significant.** Reported chi^2 = 9.37, p = 0.053 (line 952).
This fails the conventional alpha = 0.05 threshold. Proceeding to post-hoc tests when
the omnibus test doesn't reject H0 is technically invalid. The paper still frames PCA-Triage
as the statistically validated best method.

**(b) No multiple comparison correction.** Four pairwise Wilcoxon tests are run at alpha =
0.05 each. Without Bonferroni/Holm correction, the family-wise error rate is inflated.
With Bonferroni (alpha = 0.05/4 = 0.0125):
- vs Threshold: p = 0.016 — DOES NOT SURVIVE
- vs Variance: p = 0.023 — DOES NOT SURVIVE
- vs Uniform: p = 0.039 — DOES NOT SURVIVE
- vs Random Dropout: p = 0.109 — already not significant

Under Holm correction, the smallest p-value (0.016) already exceeds the first threshold
(0.0125), so **none** of the four pairwise comparisons remain significant in a strict
family-wise-error-controlled analysis.

**(c) Only 3-5 seeds.** With 7 paired observations in Wilcoxon, you need 6/7 or 7/7 wins
for p < 0.05 (one-sided). The paper reports 6/7 — barely significant even without
correction.

**(d) No effect sizes.** Only p-values and win counts. No rank-biserial correlation,
Kendall's W, or confidence intervals. Practical significance is unclear.

**(e) Nemenyi post-hoc test is problematic.** Benavoli et al. (2016, JMLR) showed the
Nemenyi test has a flaw: comparing A vs B depends on which other algorithms are in the
pool. Pairwise Wilcoxon with Holm correction is recommended instead.

**Required fix:** Increase to 10 seeds, apply Holm correction, report effect sizes and
confidence intervals, reframe the Friedman result honestly (borderline), or drop it in
favor of only pairwise tests.

> **PARTIALLY RESOLVED** (commit 8454f3f, 2026-04-03; audited again 2026-04-04)
> 
> Recomputed all statistics from V1 base per-seed data (3 seeds × 6 datasets).
> 
> **New script:** experiments/run_statistical_tests_v1.py
> **Output CSVs:** friedman_ranks_v1.csv, wilcoxon_tests_v1.csv
> 
> **Recomputed results:**
> - Friedman: chi^2 = 8.27, p = 0.082 (not significant), Kendall's W = 0.344
> - PCA-Triage mean rank: 1.67 (best)
> - Wilcoxon (Holm-corrected): All 5/6 wins, raw p=0.031, but NO comparisons
>   survive Holm correction (min corrected threshold = 0.0125)
> - Effect sizes: rank-biserial r = 0.71 (vs Uniform) to 0.91 (vs others) = large
> 
> **Paper updates:**
> - Friedman table: updated ranks and reported chi^2, p, Kendall's W
> - Wilcoxon table: shows raw p, Holm significance (all n.s.), W/L/T, effect sizes
> - Text honestly explains: "lack of significance reflects limited n=6 datasets,
>   not inconsistent performance"
> - Abstract: removed "statistically significant" claim, replaced with "winning
>   5 of 6 datasets with large effect sizes (r=0.71-0.91)"
> - Nemenyi post-hoc removed (replaced with Holm-corrected pairwise Wilcoxon)
> 
> **Remaining limitation:** Only 3 seeds available (2 of original 5 missing from
> V1 pareto CSVs). More seeds would increase power but require re-running
> experiments with the actual datasets.
> 
> **What remains unresolved:** The statistical section is much more honest now, but the
> main 50% results table still uses significance stars and `p < 0.05` wording, which
> conflicts with the Holm-corrected interpretation.

### GAP-M2: Baseline Fairness — Unequal Hyperparameter Tuning

**Severity: HIGH**

PCA-Triage gets per-dataset tuned configurations in run_pareto_v2.py:
- TEP: n_components=10, alpha=0.8, sharpness=1.5
- SMD: n_components=12, alpha=0.75, sharpness=2.0
- PSM: n_components=8, alpha=0.4, sharpness=3.0
- HAI: n_components=15, alpha=0.05, sharpness=2.0
- SKAB: n_components=3, alpha=0.15, sharpness=2.5

That's 3 hyperparameters tuned per dataset (15+ choices total). Meanwhile, baselines
(Uniform, Variance, Threshold, Random Dropout) use fixed default parameters with no tuning.

This is an unfair comparison. A reviewer will argue PCA-Triage's advantage partly comes
from dataset-specific tuning, not the method itself.

**Required fix:** Either:
- (a) Give baselines equivalent tuning budget (e.g., tune Threshold's percentile parameter,
  Variance's window size), OR
- (b) Report PCA-Triage with default parameters as the main result, and tuned as an ablation, OR
- (c) Use nested cross-validation for all methods, clearly separating tuning from evaluation

> **PARTIALLY RESOLVED** (via GAP-C1 fix, commit a14147c; audited again 2026-04-04)
> 
> Chose option (b): V1 base PCA-Triage (no per-dataset tuning) is now the canonical
> main result. All methods use identical default parameters. The V2 innovations
> (hybrid scoring, linear interp, sharpening) are presented as ablation improvements.
> Table IV caption explicitly states: "All methods use default parameters with no
> per-dataset tuning."
> 
> **What remains unresolved:** The fairness story in the paper is better, but the repo
> still has unresolved canonical-artifact ambiguity and the manuscript still contains an
> RF-100 vs RF-200 mismatch elsewhere.

### GAP-M2b: Documentation and Experiment Protocol Mismatch

**Severity: MEDIUM-HIGH**

The README still presents a simpler, more controlled evaluation story than the paper's
current v2 results actually use:
- README says same downstream classifier is RandomForest with n=100
- README says same reconstruction is forward-fill for all methods
- v2 experiments actually use RandomForest with n=200, hybrid scoring, per-dataset tuning,
  sharpening, and linear interpolation

This matters because a reader can move between README, paper, and saved CSVs and come away
with different impressions of what the "main experiment" really was. Even if the v2 setup
is valid, the documentation inconsistency makes the evaluation protocol look unstable.

**Required fix:** Update README and manuscript together so they describe one canonical
experiment stack, clearly distinguishing legacy/base results from v2 tuned results.

> **NOT RESOLVED** (audited 2026-04-04)
> 
> The paper moved toward a cleaner V1/base canonical story, but the README still reflects
> the older repo state:
> - still leads with the `0.970` headline story,
> - still claims 4/7 wins,
> - still includes NASA,
> - still states 0-3 window reaction at `lambda = 0.85`,
> - still uses the older stronger novelty framing.
> 
> This gap remains open and should be treated as an active documentation issue.

### GAP-M3: Novelty Claim Is Overstated

**Severity: MEDIUM-HIGH**

The paper claims (line 1086): "the first streaming algorithm that converts incremental PCA
loadings into proportional per-channel bandwidth allocation."

Closely related prior work that the paper does NOT cite:

1. **Bacciu (2016)** — "Unsupervised Feature Selection for Sensor Time-Series in Pervasive
   Computing Applications." Uses cross-correlation and redundancy analysis for unsupervised
   runtime sensor selection. (NCA 2015)

2. **Ghosh et al. (2021)** — "Learning-Based Adaptive Sensor Selection Framework for
   Multi-Sensing WSN." Adaptive sensor selection using cross-correlation and prediction of
   inactive sensors. (IEEE Sensors Journal)

3. **Ghosh et al. (2021)** — "Edge Intelligence Framework for Data-Driven Dynamic Priority
   Sensing and Transmission." Dynamic priority sensing using cross-correlation plus temporal
   correlation. (IEEE TGCN)

4. **Yang et al. (2023) / FreqSense** — Adaptive sampling-rate selection under resource
   budgets for IoT sensors. (Sensors)

These papers address overlapping problems: unsupervised sensor importance via correlation,
adaptive rate allocation, and budget-constrained sensing. While none do exactly "PCA
loadings -> proportional rates," the combination of cross-correlation-based importance and
adaptive allocation is not new.

**Defensible novelty (narrower):**
- Incremental PCA specifically as the importance engine (vs. other correlation methods)
- Proportional rate allocation with budget constraint and min-rate floor
- Evaluation on anomaly/fault-detection benchmarks with edge deployment emphasis

**Required fix:** Cite these papers, soften "first" to "novel combination" or "first to use
incremental PCA specifically," and position the contribution as system-level integration
rather than a fundamentally new idea.

> **RESOLVED** (commit 1ffa1ef, 2026-04-03)
> 
> **What changed in paper/references.bib:**
> - Added 4 references: Bacciu 2016, Ghosh et al. 2021 (Sensors J), Ghosh et al. 2021
>   (TGCN), Yang et al. 2023 (FreqSense)
> 
> **What changed in paper/main.tex:**
> - Contributions: "the first streaming algorithm" → "a streaming algorithm"
> - Positioning section: Added paragraph acknowledging prior correlation-aware sensor
>   selection work, positions PCA-Triage as building on it with incremental PCA,
>   proportional (not binary) rates, and zero-parameter streaming
> - Positioning table caption: Added citation to Bacciu and Ghosh noting their batch/binary
>   approach vs our streaming/proportional approach
> - Conclusion: "the first streaming algorithm" → "a streaming algorithm"
> 
> **Verification:** "first streaming" and "first method" no longer appear in the paper.

### GAP-M4: Benchmark Quality Concerns

**Severity: MEDIUM-HIGH**

Published research (Kim et al., AAAI 2022; TSB-AD, NeurIPS 2024) has documented serious
quality issues with several benchmarks used in this paper:

| Dataset | Known Issue                                             | Source          |
|---------|---------------------------------------------------------|-----------------|
| PSM     | 28% anomaly density (unrealistically high)              | Kim et al. 2022 |
| SMD     | "Trivial" — detectable by moving standard deviation     | Kim et al. 2022 |
| MSL     | Unlabeled anomalies in training data                    | TSB-AD 2024     |
| SWaT    | Synthetic stand-in in this paper (not real SWaT data)   | Code            |
| NASA    | Labeled synthetic in code, presented as real in paper    | Code            |

Of the 7 Pareto-evaluated datasets, 5 have documented quality concerns. This doesn't
invalidate the results, but the paper should acknowledge these issues.

Additionally, newer benchmarks now exist:
- TSB-AD (NeurIPS 2024): 1,070 time series from 40 datasets — current gold standard
- TAB (VLDB 2025): 29 multivariate datasets, 40 methods evaluated
- TimeSeriesBench (2024): Industrial-grade benchmark from Alibaba

**Required fix:** Add explicit acknowledgment of benchmark limitations in the threats-to-
validity section. Consider adding 1-2 datasets from TSB-AD or TAB for stronger coverage.

> **RESOLVED** (commit bcea93f, 2026-04-03)
> 
> Added "Benchmark quality" paragraph to threats-to-validity section acknowledging
> PSM high anomaly density, SMD triviality, and MSL unlabeled anomalies. Notes that
> TEP is the best-characterized benchmark. NASA row already removed (GAP-C8).
> Adding new benchmarks (TSB-AD, TAB) deferred — requires dataset access and
> experiment re-runs.

### GAP-M5: F1 Exceeding Full-Data Performance Needs Explanation

**Severity: MEDIUM**

The paper claims PCA-Triage achieves F1 = 0.970 at 50% bandwidth, exceeding full-data
F1 = 0.962. This is cited as a headline result in the abstract, results, and conclusion.

While possible (triage can act as implicit denoising/regularization by removing noisy
low-importance channels), this is a surprising claim that will face heavy scrutiny.

The paper does NOT provide a convincing mechanistic explanation. A reviewer will ask:
- Is this an artifact of per-dataset hyperparameter tuning? (The base result is 0.961,
  which does NOT exceed full-data 0.962)
- Is this within the confidence interval of the full-data result?
- Is this driven by the specific Random Forest configuration?

**Required fix:** Provide a mechanistic analysis. Show the full-data confidence interval
alongside the triaged result. If the v2 tuned result (0.970) exceeds full-data but the
base result (0.961) does not, the explanation is the tuning, not the triage.

> **RESOLVED** (commit bcea93f, 2026-04-03)
> 
> Reframed: base PCA-Triage (0.961) is now "within 0.1% of full-data (0.962)"
> throughout. The 0.970 claim is explicitly attributed to targeted extensions.
> Added mechanistic explanation in discussion: "implicit denoising effect —
> PCA-directed sub-sampling drops low-importance channels that contribute noise,
> while linear interpolation smooths remaining signals." Confirmed that base
> without extensions does NOT exceed full-data, only extensions do.

### GAP-M6: Missing Baselines

**Severity: MEDIUM**

**(a) Compressive sensing.** A major competing paradigm for bandwidth reduction in sensor
networks. Recent work: NSPL-HCS (2025), rate-adaptive compressed sampling (2024). CS
exploits sparsity (complementary to PCA's correlation exploitation). Not including any CS
baseline leaves a visible gap for reviewers familiar with signal processing.

**(b) Online feature selection (empirical).** The paper discusses OSFS, SAOLA, OSSFS in
related work but dismisses them as "binary" without empirical comparison. Showing that
proportional allocation empirically beats binary selection would strengthen the narrative.

**(c) OGD baseline.** Claimed in the paper (line 406: "OGD: Online Gradient Descent") but
no implementation found in src/baselines/. The joint experiment (Experiment 17) reports
OGD results in the paper, but the baseline implementation is missing from the repository.

**Required fix:** At minimum, implement OGD (since it's already claimed), and add one CS
baseline. Binary feature selection baseline would be a bonus.

> **NOT RESOLVED — deferred to future work.**
> 
> Requires implementing new baselines and re-running experiments with actual datasets
> (not available in repo). OGD results exist in the paper from a prior experiment run
> but the implementation code is missing. CS and binary FS baselines would strengthen
> the paper but are not blocking for initial submission. A reviewer may request these
> in a revision.

---

## 5. MODERATE GAPS (Presentation / Strengthening Issues)

These are unlikely to cause rejection alone but weaken the paper and may accumulate into
a negative reviewer impression.

### GAP-P1: Compute Time Reporting Is Ambiguous

The paper claims 0.67 ms/decision (abstract, line 64; Table IX). Two profiles exist:

| Profile                    | ms/window | Source                      |
|----------------------------|-----------|-----------------------------|
| Edge simulation (1 thread) | 0.665     | compute_profile_edge.json   |
| Laptop (multi-thread)      | 1.461     | compute_profile.json        |

The paper uses the edge-simulated number. This is technically valid ("single CPU core") but:
- The edge simulation was NOT run on actual edge hardware (Raspberry Pi, Jetson)
- A reviewer may expect hardware measurements, not simulated single-threaded benchmarks
- The Table IX in the paper shows 0.67ms but the main compute_profile.json (which a
  reviewer would find first) shows 1.46ms

**Recommendation:** Report both numbers in the paper. State clearly: "0.67 ms (single-
threaded, simulating edge deployment) and 1.46 ms (laptop, multi-threaded overhead)."

> **RESOLVED** (2026-04-03)
> 
> Compute cost table now shows TWO columns: "ms (edge)" and "ms (laptop)" for all
> methods. Table caption clarifies: "Edge = single-threaded simulation; Laptop =
> multi-threaded measurement." Figure caption already had both numbers.
> 
> Also resolved during this pass:
> - **"Matching full-data"** language → changed to "within 0.1% of full-data" in all
>   4 instances (abstract, figure caption, regret intuition, conclusion). Factual: 
>   0.961 vs 0.962 = 0.1% relative difference.
> - **Source CSV traceability**: Added LaTeX comments linking bandwidth sensitivity
>   table to V1 base pipeline and reconstruction table to its source CSV.
>   Reconstruction CSV values verified cell-by-cell. Bandwidth sensitivity table
>   confirmed as V1 base (matches 0.961 at 50%), but pareto_tep.csv contains V2
>   tuned values — comment clarifies this.

### GAP-P2: Paper Length and Structure

At ~1100 lines of LaTeX with 17 experiments, 14 figures, and 10+ tables, this paper is
likely 14-16 pages — potentially exceeding IEEE journal limits (typically 8-12 pages for
regular papers, up to 14 for special issues).

Some experiments could move to supplementary material:
- Experiment 11 (Synthetic correlation): Inconclusive (PCA-Triage loses to Variance)
- Experiment 12 (Adaptive k): Incremental improvement
- Experiment 13 (Ensemble scoring): Marginal gains
- Experiment 17 (Joint spatial-temporal): PCA doesn't win clearly

**Recommendation:** Identify the 8-10 strongest experiments, move the rest to supplementary.

> **NOT RESOLVED — deferred.**
> 
> Requires a full LaTeX compile to assess actual page count and determine which
> experiments to move. Cannot be assessed without compiling. If overlength, candidates
> for supplementary: Experiments 11-13 (synthetic correlation, adaptive k, ensemble).

### GAP-P3: "7 Real-World + 1 Synthetic" Is Misleading

The paper states (line 364): "including 7 real-world benchmarks and 1 synthetic dataset."
But:
- SWaT: Synthetic stand-in (correctly marked with dagger)
- NASA: Synthetic (labeled in code, NOT marked in paper)
- SKAB: Real but only 8 channels (limited relevance for the correlation argument)
- HAI: Real but saturated (all methods F1 >= 0.998)

A more honest framing: "5 informative real-world benchmarks (TEP, SMD, MSL, PSM, HAI),
1 real benchmark where all methods saturate (HAI), 1 real benchmark with limited channels
(SKAB), and 2 synthetic datasets."

> **PARTIALLY RESOLVED** (via GAP-C8, commit a14147c)
> 
> NASA removed (was the most misleading "real" dataset). Paper now says "6 real-world
> benchmarks and 1 synthetic dataset" which is accurate — SWaT is marked with dagger.
> The nuances about SKAB (8 channels) and HAI (saturated) are discussed in the
> "When PCA-Triage Struggles" section.

### GAP-P4: Evaluation Metric Choice

The paper uses point-wise weighted F1 as the sole metric. Published criticism:
- Kim et al. (AAAI 2022): Random anomaly scores can achieve SOTA under point-adjust F1
- TSB-AD (NeurIPS 2024): VUS-PR identified as most reliable metric
- PATE (2024): Proximity-aware evaluation proposed

The paper's use case (comparing triage methods, not detectors) partially justifies F1 —
the detector is fixed (RF) and only the input data changes. But this argument should be
made explicitly.

**Recommendation:** Add 1-2 sentences justifying why point-wise F1 is appropriate for
triage comparison, and consider reporting VUS-PR as a supplementary metric.

> **RESOLVED** (commit bcea93f, 2026-04-03)
> 
> Added F1 metric justification in construct validity section: "the detector (RF)
> is fixed and identical across all methods — only the input data changes. F1
> therefore measures the triage strategy's impact on a fixed classifier, not
> detector quality." VUS-PR deferred as supplementary metric (would require
> reimplementation).

### GAP-P5: Theory Section Cost-Benefit

The theory section (lines 253-356) includes 4 propositions, 1 theorem, and 1 corollary.
Assessment:

| Result            | Value Added                                              |
|-------------------|----------------------------------------------------------|
| Prop 1 (Budget)   | Trivial (follows from the formula by construction)       |
| Prop 2 (Convergence) | Standard EMA convergence, nothing PCA-specific        |
| **Theorem 1**     | **Flawed** (proof contradicts conclusion — see GAP-C3)   |
| Corollary 1       | Simple bound, useful but not deep                        |
| **Prop 3 (Regret)** | **Unrigorous** derivation (see GAP-C4)                 |
| Prop 4 (Adaptation) | Standard EMA decay rate                                |

The theory section adds ~100 lines of LaTeX but provides limited insight beyond what the
algorithm description already conveys. For an empirically-driven paper, this space would
be better used for stronger experiments.

**Recommendation:** If the theory can't be made airtight, reduce to Proposition 1 (budget
feasibility) and Proposition 4 (adaptation rate) — the two that are correct and useful.
Present Theorem 1's insight as an empirical observation with the correlation matrix example.

> **PARTIALLY RESOLVED** (commits 1ffa1ef + bcea93f)
> 
> Theorem 1 restated honestly (GAP-C3). Regret bound removed (GAP-C4). Props 1, 2,
> and 4 kept — they are simple but provide formal grounding expected by IEEE reviewers.
> Theory section now has: 2 propositions (budget, convergence), 1 theorem (correlation
> distinction), 1 corollary (reconstruction error), 1 proposition (adaptation rate),
> plus informal regret intuition. More compact than before.

### GAP-P6: Per-Fault Results Weaken the Narrative

Table X (per-fault breakdown) shows PCA-Triage wins on only 2 of 10 individual TEP faults
(IDV 7, 14). Threshold dominates on 5 faults. The paper frames this positively ("balances
well across all fault types") but a skeptical reviewer could read it as "PCA-Triage wins
aggregate metrics by not being worst on any fault, not by being best at fault detection."

**Recommendation:** Frame more carefully. Explain why aggregate performance matters more
than per-fault performance for deployment (operators don't know which fault will occur).

> **RESOLVED** (commit bcea93f, 2026-04-03)
> 
> Per-fault findings rewritten: "arises not from dominating any single fault type,
> but from consistent above-average performance across all types — important in
> deployment where the fault type is unknown a priori." Stale 0.970 reference
> corrected to 0.961.

### GAP-P7: Robustness Framing Is Too Broad

The paper's robustness language suggests PCA-Triage is broadly strong under deployment
perturbations. But the saved robustness CSV shows a narrower reality:
- PCA loses to Variance/Uniform under jitter on TEP, SMD, and PSM
- PCA loses under clock drift on TEP, SMD, and PSM
- PCA also loses the combined condition on SMD

The positive claim is still defensible for packet loss and moderate noise, but the broader
"robust to deployment perturbations" framing overstates the result.

**Recommendation:** Reframe as selective robustness: strong under packet loss and moderate
noise, weaker under temporal perturbations such as jitter and clock drift.

> **RESOLVED** (commit bcea93f, 2026-04-03)
> 
> Abstract and conclusion reframed: "robust to packet loss and moderate sensor noise,
> with 3.7-4.8% degradation under combined worst-case perturbations on TEP/SMD
> (more sensitive to temporal jitter and clock drift)." The experiment findings text
> already noted temporal sensitivity honestly.

### GAP-P8: Synthetic Correlation Experiment Undermines the Theory Narrative

Experiment 11 is more problematic than merely "inconclusive." In the paper's own table,
PCA-Triage loses to Variance at **every** tested correlation level rho in the synthetic
study. The gap narrows slightly as rho increases, but PCA never wins.

That does not kill the method, but it weakens any strong theory-backed statement that
correlation structure alone explains the empirical gains. As written, the paper uses this
experiment to support the correlation argument when it actually shows only a weak, indirect
trend.

**Recommendation:** Present the synthetic study as a limitation/check on the theory claim,
not as affirmative evidence. Keep the stronger argument grounded in the real benchmark
results and correlation analysis.

> **RESOLVED** (commit bcea93f, 2026-04-03)
> 
> Experiment 11 intro reframed: "To directly test" → "To probe the limits of."
> Findings rewritten: "an important limitation check on the theory" and "Simple
> pairwise correlation alone is insufficient for PCA-Triage to outperform Variance."
> Real-world advantage attributed to "complex, multi-modal correlation structure"
> rather than theory prediction.

---

## 6. MINOR GAPS (Polish / Nice-to-Have)

### GAP-N1: Missing OGD Implementation
OGD is listed as baseline #9 in the paper but no implementation exists in src/baselines/.
Results appear in Table XVI (Experiment 17), so the experiment was run somehow — but the
code isn't in the repo.

> **NOT RESOLVED — deferred.** OGD results exist in the paper from a prior experiment
> run but the code is missing from the repository. Would need to recover or reimplement.

### GAP-N2: No Environment Lock File
Historical note: this gap is now closed. It is retained here for audit completeness.

> **RESOLVED** (commit 511a7cd, 2026-04-03)
> 
> Generated uv.lock from pyproject.toml. Removed legacy requirements.txt,
> requirements-lock.txt, and environment.yml. Project now uses uv-based dependency
> management exclusively.

### GAP-N3: Multi-Classifier Table Uses Different F1 Than Main Result
Multi-classifier table (line 513) shows TEP RF F1 = 0.962, but main Table IV shows 0.970.
This is because multi-classifier uses base PCA-Triage (not v2 hybrid). The inconsistency
will confuse readers.

> **RESOLVED** (via GAP-C1 fix, commit a14147c)
> 
> Main Table IV now uses V1 base (0.961). Multi-classifier table shows 0.962 (same
> V1 base, rounds differently due to different seed sets). These are now consistent
> within rounding — both show base PCA-Triage at ~0.961-0.962.

### GAP-N3b: Multi-Classifier Narrative Overstates Consistency
The paper says PCA-Triage "consistently" outperforms Uniform and Variance regardless of
classifier choice, but SKAB with SVM shows Variance > PCA-Triage. The table mostly supports
classifier robustness, but not a universal dominance claim.

> **RESOLVED** (commit bcea93f, 2026-04-03)
> 
> Text softened: "consistent across all classifiers" → "consistent across classifiers
> on high-correlation datasets." Added: "On SKAB, margins are minimal and Variance
> occasionally edges PCA-Triage (e.g., SVM)."

### GAP-N4: Abstract Is Too Long
The abstract (lines 63-67) is ~250 words. IEEE typically recommends 150-200 words for
journal abstracts. It tries to include every result — some trimming would improve impact.

> **RESOLVED** (commit bcea93f, 2026-04-03)
> 
> Abstract trimmed from ~250 words to 118 words. Removed step-by-step algorithm
> description (moved to method section) and excess result details.

### GAP-N5: Author Affiliations Missing
Line 57 shows only email addresses, no institutional affiliations. IEEE requires affiliations.

> **RESOLVED** (commit bcea93f, 2026-04-03)
> 
> Added "Independent Researchers" affiliation block.

### GAP-N6: Some Figures May Not Reproduce
Figure generation depends on data that's not in the repo. A reviewer running from a fresh
clone would have no figures.

> **NOT RESOLVED — inherent limitation.**
> 
> Figures were generated from experiment results that depend on datasets not in the
> repo. The pre-generated PNG figures ARE in the repo (paper/figures/), so the paper
> compiles correctly. Regenerating figures requires downloading datasets and re-running
> experiments. This is documented in data/README.md.

---

## 7. PRIOR ART NOT CITED (Must Add)

| Paper | Relevance | Why It Matters |
|-------|-----------|----------------|
| Bacciu (2016), NCA — Unsupervised feature selection for sensor time-series via redundancy | Directly competes with the "unsupervised correlation-based sensor selection" claim | Closest prior art to "first" claim |
| Ghosh et al. (2021), IEEE Sensors J — Adaptive sensor selection using cross-correlation | Per-sensor adaptive selection in WSNs | Overlaps with core contribution |
| Ghosh et al. (2021), IEEE TGCN — Dynamic priority sensing with cross-correlation | Dynamic priority under resource budgets | Overlaps with core contribution |
| Yang et al. (2023) / FreqSense — Adaptive sampling-rate selection under resource budgets | Budget-constrained adaptive sampling | Directly relevant to formulation |
| Benavoli et al. (2016), JMLR — Should We Really Use Post-Hoc Tests Based on Mean-Ranks? | Criticizes Nemenyi post-hoc test | Required for statistical methodology |
| Kim et al. (2022), AAAI — Towards a Rigorous Evaluation of TSAD | Documents benchmark quality issues | Required for benchmark discussion |

> **RESOLVED** (commit 1ffa1ef, 2026-04-03)
> 
> Added Bacciu 2016, Ghosh et al. 2021 (x2), and Yang et al. 2023 to references.bib
> and cited in positioning section. Benavoli 2016 not explicitly cited (Nemenyi
> removed, replaced with Holm-corrected Wilcoxon). Kim et al. 2022 benchmark
> concerns acknowledged in threats-to-validity paragraph.

---

## 8. VENUE ASSESSMENT

| Venue                          | IF    | Fit        | Notes                                              |
|--------------------------------|-------|------------|----------------------------------------------------|
| IEEE Sensors Journal           | ~4.3  | Best fit   | Sensor-focused, algorithmic + experimental papers   |
| IEEE Internet of Things Journal| ~8.9  | Good fit   | Higher bar, may want hardware deployment evidence   |
| IEEE Trans. Ind. Informatics   | ~13.3 | Stretch    | Expects deeper industrial validation                |
| IEEE Access                    | ~3.6  | Safe fallback | Lower selectivity (~40-45%), fast review          |

**Recommendation:** Target IEEE Sensors Journal. It matches the paper's scope, accepts
algorithm + benchmark papers without requiring hardware deployment, and the reviewer pool
will be familiar with TEP and sensor network literature.

---

## 9. HISTORICAL FIX PRIORITY MATRIX (Pre-2026-04-04 Snapshot)

This matrix reflects the original prioritization before the 2026-04-04 direct repo audit.
For the current working priority list, use Section 12 instead.

### Tier 1 — Must-fix before submission (blocks acceptance)

| ID     | Gap                                    | Effort | Impact |
|--------|----------------------------------------|--------|--------|
| GAP-C1 | Consolidate conflicting result numbers | Low    | Critical |
| GAP-C2 | Fix adaptivity claim vs table          | Medium | Critical |
| GAP-C3 | Fix Theorem 1 or reframe               | Medium | Critical |
| GAP-C5 | Fix attention baseline                 | Medium | Critical |
| GAP-C7 | Add data download scripts              | Medium | Critical |
| GAP-C8 | Correct NASA labeling                  | Low    | Medium-High |
| GAP-M3 | Soften novelty claim, add citations    | Low    | Medium-High |

### Tier 2 — Must-fix for competitiveness (prevents major revision)

| ID     | Gap                                    | Effort | Impact |
|--------|----------------------------------------|--------|--------|
| GAP-M1 | Fix statistical methodology            | High   | High   |
| GAP-M2 | Fair baseline tuning                   | High   | High   |
| GAP-C4 | Fix or remove regret bound             | Low    | Medium |
| GAP-C6 | Make DL comparison comparable          | High   | High   |
| GAP-M5 | Explain F1 > full-data                 | Medium | Medium |
| GAP-M6 | Add missing baselines (OGD at minimum) | Medium | Medium |

### Tier 3 — Should-fix for strengthening (improves score)

| ID     | Gap                                    | Effort | Impact |
|--------|----------------------------------------|--------|--------|
| GAP-M4 | Acknowledge benchmark limitations      | Low    | Medium |
| GAP-P1 | Clarify compute time reporting         | Low    | Low    |
| GAP-P2 | Trim paper to page limits              | Medium | Medium |
| GAP-P4 | Justify F1 metric or add VUS-PR        | Low    | Low    |
| GAP-P5 | Trim theory section                    | Low    | Medium |
| GAP-N5 | Add author affiliations                | Low    | Low    |

### Tier 4 — Nice-to-have (differentiates from competitors)

| ID     | Gap                                    | Effort  | Impact |
|--------|----------------------------------------|---------|--------|
| —      | Add compressive sensing baseline       | Medium  | Medium |
| —      | Add TSB-AD / TAB benchmark subset      | High    | Medium |
| —      | Hardware deployment (RPi / Jetson)     | High    | High   |
| —      | Report VUS-PR as supplementary metric  | Medium  | Low    |

---

## 10. ESTIMATED REVIEWER OUTCOME (Historical Snapshot)

This section reflects the earlier pre-fix state of the project and is left here mainly as
context for how much risk has already been reduced. It does **not** supersede the current
audit in Section 12.

If submitted as-is to IEEE Sensors Journal:

| Criterion                     | Likely Score (1-5) | Notes                              |
|-------------------------------|--------------------|------------------------------------|
| Novelty                       | 3                  | System-level, not fundamental      |
| Technical soundness            | 2                  | Theorem error, result inconsistencies |
| Experimental methodology       | 2.5                | Unfair tuning, statistical flaws    |
| Presentation                   | 3                  | Well-written but too long           |
| Reproducibility                | 1.5                | Data missing, conflicting CSVs      |
| **Overall**                    | **2.5 / 5**        | **Likely outcome: Reject**          |

If Tier 1 + Tier 2 fixes are applied:

| Criterion                     | Likely Score (1-5) | Notes                              |
|-------------------------------|--------------------|------------------------------------|
| Novelty                       | 3.5                | Honest positioning, new citations   |
| Technical soundness            | 3.5                | Clean theory, consistent results    |
| Experimental methodology       | 3.5                | Fair tuning, proper statistics      |
| Presentation                   | 3.5                | Trimmed, focused                    |
| Reproducibility                | 4                  | Download scripts, pinned deps       |
| **Overall**                    | **3.5 / 5**        | **Likely outcome: Major Revision → Accept** |

---

## 11. SUMMARY

The core research idea is solid: using incremental PCA to drive per-channel bandwidth
allocation is a practical and defensible contribution. The codebase shows significant
engineering effort and is in better shape than the earliest draft state reflected above.

Several of the most serious manuscript issues have already been improved: theorem/proof
alignment is better, the fake-attention ambiguity is clarified, the adaptivity contradiction
is fixed, NASA has been removed from the paper story, and the novelty framing is more honest.

The main remaining blockers are now narrower but still important: conflicting artifact
provenance for the canonical result story, README/paper drift, inconsistent statistical
signaling in the main table, and incomplete reproducibility closure.

**The strongest publishable version of this paper** is not "we proved new theory and we're
first in the world" but rather: "a lightweight, unsupervised, correlation-aware channel-rate
allocation method that is empirically strong across diverse benchmarks, with honest
positioning relative to prior work."

The work is worth pursuing. With focused effort on the execution checklist in Section 12,
this can still become a competitive submission to IEEE Sensors Journal.

---

## 12. CURRENT-STATE AUDIT AND EXECUTION CHECKLIST (2026-04-04)

This section is based on a direct repo audit of code, paper text, scripts, result CSVs,
README, and tests. It should be treated as more authoritative than older `RESOLVED`
comments above when the two disagree.

### 12.1 What Is Genuinely in Better Shape

- The manuscript framing is substantially improved:
  - Abstract now uses the conservative `0.961 +- 0.001` canonical TEP result and treats
    `0.970` as an extension result, not the main headline.
  - Novelty language is softened from "first" to a more defensible system-level claim.
  - NASA has been removed from the paper's benchmark story.
- The theorem section is materially safer:
  - The main PCA/variance theorem now matches the proof structure.
  - The formal regret proposition was removed and replaced with "Regret intuition."
- The adaptivity section is improved:
  - The paper no longer claims 0-3 window reaction at `lambda = 0.85`.
  - It now correctly ties the 0-3 window result to lower `lambda` values such as `0.80`.
- The synthetic-correlation discussion is more honest:
  - The paper explicitly admits PCA-Triage does not beat Variance on that simple synthetic task.
- The random-projection attention baseline is now labeled clearly as non-learned.
- Reproducibility scaffolding is stronger:
  - `uv.lock` exists.
  - `data/README.md` exists with dataset setup notes.
  - `data/download_datasets.sh` exists.
- Tests are in good shape mechanically:
  - Direct repo audit on 2026-04-04: `64 passed` via `pytest`.

### 12.2 What Is Still Not Fully Resolved (updated 2026-04-04 evening)

- ~~Canonical result source ambiguity~~ **FIXED:** `run_statistical_tests_v1.py` now reads
  from `paper/tables/table2_results_50pct.csv` (the canonical V1 base source), not from
  `experiments/results/pareto_{ds}.csv` (which contain V2 tuned results). All statistical
  tables in the paper now match this canonical source.
- ~~README contradicts paper~~ **FIXED:** README headline updated to 0.961/3 of 6, NASA
  removed from dataset table, lambda=0.85 claims corrected, dataset/test counts updated.
- ~~RF-100 vs RF-200 conflict~~ **FIXED:** Paper baselines section now says n=100 for main
  comparison, n=200 for dedicated experiments.
- ~~Significance stars contradict Holm~~ **FIXED:** All per-cell significance stars removed
  from Table IV. Caption now says "Cross-dataset Wilcoxon tests reported in Sec. Discussion."
  Statistical method description updated: Nemenyi removed, Holm correction described.
- Reproducibility is improved but not fully closed:
  - most datasets still require manual download and manual preprocessing,
  - no explicit cold-start reproduction validation is documented yet.
- `experiments/results/pareto_{ds}.csv` files still contain V2 tuned results. They are NOT
  the canonical source for any paper table but remain in the repo. A future cleanup could
  archive them to `experiments/results/v2_tuned/` to prevent confusion.

### 12.3 Priority-Ordered Execution Checklist

#### Must Fix Before Any More Reruns

1. ~~Establish one canonical result source.~~ **DONE** (commit 1a43a3f)
   - Canonical source: `paper/tables/table2_results_50pct.csv` (V1 base, RF-100)
   - `run_statistical_tests_v1.py` reads from this file, not pareto CSVs
   - Stats recomputed: Friedman chi^2=9.33 p=0.053, PCA-Triage rank 1.50

2. ~~Freeze one canonical experiment stack.~~ **DONE** (commit 1a43a3f)
   - V1 base: pure PCA scorer, forward-fill, RF-100, no per-dataset tuning
   - Paper, README, stats script all aligned to this stack
   - V2 extensions (hybrid, linear, sharpening) clearly labeled as ablation

3. ~~Resolve the RF-100 vs RF-200 conflict.~~ **DONE** (commit 1a43a3f)
   - Baselines section: "n=100 trees" for main comparison, "n=200 noted where applicable"
   - Table IV caption: "RF n=100"

4. ~~Remove or rewrite significance-star language.~~ **DONE** (commit 1a43a3f)
   - All per-cell stars removed from Table IV
   - Caption says "Cross-dataset Wilcoxon tests reported in Sec. Discussion"
   - Nemenyi reference removed from evaluation protocol; Holm described instead

5. ~~Update README to match canonical manuscript story.~~ **DONE** (commit 1a43a3f)
   - Headline: 0.961, 3/6, 7 benchmarks
   - NASA removed, SWaT* added with synthetic note
   - Lambda claims corrected (0-3 windows at <=0.80, not 0.85)
   - Test count 28→64, dataset count 8→7

6. ~~Clean this document's stale RESOLVED claims.~~ **DONE** (this update)
   - Section 12.2 updated to reflect current repo state

#### Must Fix Before Submission

1. Run and document one cold-start reproducibility pass.
   - Fresh environment.
   - Exact install steps.
   - Exact datasets successfully reproduced.
   - Any datasets still manual should be explicitly listed as such.

2. Make the paper's structure explicitly distinguish:
   - canonical main experiment,
   - tuned extensions,
   - dedicated side experiments such as DL or OGD comparisons.

3. Add provenance for each major table/figure.
   - For every main table or figure, record:
     - script used,
     - input CSV(s),
     - config or setting,
     - whether it is canonical or extension-only.

4. Decide what to do about OGD and other special baselines.
   - Either restore traceable code/artifacts for them,
   - or demote/cut claims that depend on them.

5. Harmonize documentation counts and terminology.
   - Dataset count,
   - benchmark count,
   - test count,
   - baseline count,
   - "main" vs "extension" language.

#### Nice To Fix

1. Add an artifact map document.
   - Example: `REPRODUCING.md` or `paper_artifact_map.md`
   - Map `paper table/figure -> script -> CSV -> setting`.

2. Expand tests around experiment provenance.
   - Not just algorithm behavior, but checks that canonical outputs exist and have the
     expected schema or budget/method coverage.

3. Clean README badges and top-level positioning.
   - Current badges and headline text still reflect older project state.

4. Add one machine-readable manifest for canonical outputs.
   - This can prevent future drift between paper and repo.

### 12.4 Definition of Done for the "Paper Is Internally Consistent" Milestone

This milestone should be considered complete only when all of the following are true:

- One canonical experiment stack is chosen.
- One canonical result artifact set matches the paper's main numbers.
- Statistical language is internally consistent across main tables and discussion.
- README, paper, and saved outputs tell the same story.
- Cold-start reproducibility has been actually validated and documented.

### 12.5 Recommended Instruction to a Follow-On Agent

If another agent is asked to continue from this document, the right instruction is:

"Do not assume earlier `RESOLVED` notes are correct. First verify the canonical result
source, align README/paper/scripts around one experiment stack, and remove remaining
consistency gaps before making new benchmark or paper-quality claims."
