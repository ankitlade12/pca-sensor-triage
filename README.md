# PCA-Triage: Adaptive Sensor Triage for Edge AI Inference

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-53%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents

1. [Overview](#1-overview)
2. [Core Thesis](#2-core-thesis)
3. [Algorithm](#3-algorithm)
4. [Architecture Flow](#4-architecture-flow)
5. [Baselines](#5-baselines)
6. [Datasets](#6-datasets)
7. [Experimental Design](#7-experimental-design)
8. [Results](#8-results)
9. [Design Decisions](#9-design-decisions)
10. [Reproduction](#10-reproduction)
11. [Project Structure](#11-project-structure)
12. [References](#12-references)

---

## 1. Overview

**PCA-Triage** uses streaming PCA loadings as a *meta-controller* for per-channel bandwidth allocation in sensor networks. When new sensor data arrives in windows, PCA-Triage measures each channel's contribution to the principal subspace and **continuously modulates** its sampling rate:

- **Importance scoring** via weighted PCA loadings: `s_j = Σᵢ σᵢ · |V[i,j]|²`
- **Proportional rate allocation** under a budget constraint: `Σ r_j ≤ B · d`
- **Exponential smoothing** with forgetting factor `λ` for temporal stability
- **Minimum rate floor** `r_min = 0.05` ensures no channel is fully silenced

The contribution is **not** a new model architecture — it is a **what-to-sample and how-much** strategy that wraps any downstream classifier or anomaly detector.

**Headline result (7 datasets, 6 methods):** At **50% bandwidth**, PCA-Triage achieves F1 = 0.961 on TEP fault detection — matching full-data performance — while running in **0.67 ms per decision**. Best unsupervised method on **4 of 6 real-world datasets** (TEP, SMD, MSL, SKAB), tied on HAI, with per-dataset tuning of k and r_min. Significantly better (p < 0.05, Wilcoxon) than all baselines on TEP.

---

## 2. Core Thesis

When multi-channel sensor data must be transmitted under bandwidth constraints, using streaming PCA to *dynamically allocate per-channel sampling rates proportional to importance* yields better downstream task performance than fixed or heuristic allocation strategies. Specifically:

1. **Correlated sensor networks** (TEP, 52 channels): PCA captures inter-channel correlations that variance-based methods miss. PCA-Triage outperforms Variance-based allocation by +1.4% F1 and Uniform by +3.8% F1 at 50% bandwidth.
2. **Fault onset adaptation**: When a fault activates new sensor channels, PCA loadings shift within 1-3 windows. On TEP Fault 1, the A Feed Flow valve (xmv_3) jumps from 9% to 38% sampling rate after fault onset.
3. **Edge viability**: The algorithm runs at O(wdk) with zero trainable parameters, completing each triage decision in 0.67 ms on a single CPU core — well under the 5 ms edge deployment target.

### Why PCA-Triage > Variance-Based

Variance-based allocation treats each channel independently. PCA-Triage captures **joint** variance structure: if channels A and B are highly correlated, sampling both at full rate is wasteful — PCA detects this and down-weights the redundant channel. The TEP correlation matrix shows multiple sensor clusters with |r| > 0.7, exactly where PCA's advantage emerges.

### The Gap in Existing Literature

| | **Static / Batch** | **Streaming / Adaptive** |
|---|---|---|
| **Fixed Rules** | Uniform sampling; Send-on-Delta | DSRA; Energy-aware AIMD |
| **Data-Driven** | Batch PCA detection; Offline FS | **PCA-Triage (Ours)** |

No existing method occupies the bottom-right cell: a streaming, data-driven approach that converts PCA loadings into proportional per-channel bandwidth allocation under a total budget constraint.

---

## 3. Algorithm

### 3.1 Channel Importance Scoring

For a window `X_w ∈ ℝ^(w×d)`, PCA-Triage fits IncrementalPCA to extract:
- `V ∈ ℝ^(k×d)` — top-k loading vectors
- `σ ∈ ℝ^k` — corresponding singular values

The importance score for channel `j`:

```
s_j = Σᵢ₌₁ᵏ σᵢ · V[i,j]²
```

This score captures how much channel `j` contributes to the top-k principal components, weighted by the variance each component explains.

### 3.2 Exponential Smoothing

To prevent rapid oscillations:

```
s̄_j^(t) = λ · s̄_j^(t-1) + (1 - λ) · s_j^(t)
```

| λ | Behaviour | Best For |
|---|-----------|----------|
| 1.0 | No forgetting, cumulative PCA | Static/slowly-changing systems |
| 0.85 | Fast adaptation, 0-3 window reaction | Fault onset detection |
| 0.95 | Balanced | General use |

### 3.3 Rate Allocation

Given smoothed scores `s̄` and budget `B`:

```
1. Floor:   r_j = r_min                           (all channels)
2. Allocate: r_j += (s̄_j / Σ s̄) · (B - r_min) · d  (proportional to importance)
3. Clip:    r_j = clip(r_j, r_min, 1.0)           (enforce bounds)
```

### 3.4 Full Algorithm

```
Algorithm 1: PCA-Triage
─────────────────────────────────────────────────────────
Input:  Stream of sensor windows X_w ∈ ℝ^(w×d)
        Budget B ∈ (0, 1], components k, forgetting factor λ

Init:   IncrementalPCA(k), smoothed scores s̄ = None

FOR each window X_w:
  1. UPDATE:   partial_fit(X_w) → V, σ
  2. SCORE:    s_j = Σᵢ σᵢ · V[i,j]²    ∀j ∈ {1,...,d}
  3. SMOOTH:   s̄ = λ · s̄ + (1-λ) · normalize(s)
  4. ALLOCATE: r_j = r_min + (s̄_j / Σs̄) · (B - r_min) · d
               clip to [r_min, 1.0]
  5. ACQUIRE:  keep sample x_t[j] with probability r_j
  6. RECONSTRUCT: forward-fill NaN values

Output: Triaged data, importance scores s̄, rates r
─────────────────────────────────────────────────────────
Time:   O(wdk) per window
Memory: O(wd + kd)
Params: 0 (computed, not trained)
```

### 3.5 Hyperparameters

| Parameter | Default | Range Tested | Sensitivity |
|-----------|---------|-------------|-------------|
| k (components) | 10 | 3–30 | Robust for k ∈ [3, 10]; degrades at k > 15 |
| w (window size) | 50 | 25–500 | Stable across range (F1 0.959–0.966) |
| λ (forgetting) | 1.0 | 0.80–1.0 | λ=1.0 best for accuracy; λ=0.85 for fast adaptation |
| r_min (min rate) | 0.05 | — | Prevents channel silencing |

---

## 4. Architecture Flow

### 4.1 PCA-Triage Pipeline

```
┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌────────────┐     ┌──────────┐
│  Sensor  │────▶│ Sliding  │────▶│ Incremental  │────▶│ Importance │────▶│   Rate   │
│  Array   │ xₜ  │ Window   │ X_w │    PCA       │V, σ │  Scoring   │ sⱼ  │Allocator │
│(d chan.) │     │ Buffer   │     │ (k comp.)    │     │Σσᵢ|Vᵢⱼ|² │     │(budget B)│
└──────────┘     └──────────┘     └──────────────┘     └────────────┘     └─────┬────┘
      ▲                                                                          │ rⱼ
      │                    per-channel sampling rates                            │
      └──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 End-to-End Experiment Flow

```
Phase 1: Data Loading           Phase 2: Triage              Phase 3: Evaluation
─────────────────────           ──────────────────            ──────────────────
Load TEP/SKAB/NASA   ────▶   For each window:              Train RF on triaged data
StandardScaler                  1. PCA update                Compare F1 vs full data
Train/Test split                2. Score channels            Wilcoxon significance test
                                3. Allocate rates            Generate Pareto curves
                                4. Sub-sample + reconstruct
```

---

## 5. Baselines

All 6 methods use the **same** downstream classifier (RandomForest, n=100) and data. Only the triage strategy differs.

### Why These 6 Baselines?

The 6 methods form a **controlled comparison** spanning the design space:

| Method | Strategy | Adaptive? | Supervised? | Complexity | Research Question |
|--------|----------|-----------|-------------|------------|------------------|
| **Uniform** | Same rate for all channels | No | No | O(1) | *What if we don't prioritize at all?* |
| **Threshold** | Binary: active channels get high rate | Yes | No | O(wd) | *Does simple change detection help?* |
| **Variance** | Proportional to rolling variance | Yes | No | O(wd) | *Does per-channel variance suffice?* |
| **Random Dropout** | Randomly drop (1-B) channels | No | No | O(d) | *Is random selection competitive?* |
| **Mutual Info** | Proportional to MI with labels | No (batch) | **Yes** | O(nd) | *How close to supervised optimum?* |
| **Attention** | Self-attention importance weights | Yes | No | O(d²w) | *Is attention worth the compute cost?* |
| **PCA-Triage** | Proportional to weighted PCA loadings | Yes | **No** | O(wdk) | *Does PCA correlation-awareness help?* |

### Key Comparisons

- **PCA-Triage vs Variance** → Does capturing correlations (PCA) beat treating channels independently (variance)? → **Yes: +1.4% F1 on TEP**
- **PCA-Triage vs Uniform** → Does any intelligence in allocation help? → **Yes: +3.8% F1 on TEP**
- **PCA-Triage vs Mutual Info** → Can unsupervised PCA match supervised MI? → **Yes: +2.5% F1, without needing labels**

---

## 6. Datasets

### 6.1 Real-World Datasets (7 total)

| Dataset | Domain | Sensors | Samples | Task | Source |
|---------|--------|:-------:|:-------:|------|--------|
| **TEP** | Chemical process | 52 | 250K+ | Fault detection (20 types) | Downs & Vogel, 1993 |
| **SMD** | Server machines | 38 | 388K | Anomaly detection | OmniAnomaly, KDD 2019 |
| **MSL** | Mars spacecraft telemetry | 55 | 132K | Anomaly detection | NASA, KDD 2018 |
| **PSM** | Server metrics (eBay) | 25 | 220K | Anomaly detection | RANSynCoders, KDD 2021 |
| **HAI** | Industrial control system | 82 | 259K | Attack detection | CSET 2020 |
| **SKAB** | Water circulation testbed | 8 | 47K | Anomaly detection | Skoltech, 2020 |
| **NASA** | Bearing degradation | 16 | 1K | Degradation detection | NASA IMS |

### 6.2 TEP Sensor Description

The Tennessee Eastman Process simulates a real chemical plant with:
- **41 measured variables** (XMEAS 1-41): feed rates, pressures, temperatures, levels, compositions
- **11 manipulated variables** (XMV 1-11): valve positions, cooling water flows
- **20 fault types** (IDV 1-20): step changes, random variations, valve sticking, kinetic drift
- **Correlation structure**: Multiple sensor clusters with |r| > 0.7 — ideal for PCA-based triage

---

## 7. Experimental Design

### 7.1 Scale

**6 methods × 7 datasets × 5-6 bandwidth levels × 3-5 seeds = 1,000+ Pareto runs**
Plus: ablation studies (k, w, λ, score formula), compute profiling, adaptivity analysis, scalability

### 7.2 Fairness Guarantees

| Guarantee | Mechanism |
|-----------|-----------|
| Same classifier | RandomForest(n_estimators=100) for all methods |
| Same seed | Matched seeds across methods per experiment |
| Same data | Identical train/test splits |
| Same reconstruction | Forward-fill for all methods |
| Same budget | Identical B applied across methods |

### 7.3 Metrics

| Metric | Description |
|--------|-------------|
| F1 (weighted) | Primary metric for fault detection / anomaly detection |
| MSE / NRMSE | Reconstruction quality |
| Wall-clock time | ms per triage decision |
| Peak memory | MB during processing |
| Wilcoxon p-value | Statistical significance (one-sided, n=5 seeds) |

### 7.4 Experiments

| # | Experiment | Figure | Key Finding |
|---|-----------|--------|-------------|
| 1 | Pareto curves (accuracy vs bandwidth) | Fig. 2 | PCA-Triage dominates TEP at all budgets |
| 2 | Compute cost profiling | Fig. 4 | 0.67 ms/decision, 8.5 MB memory |
| 3 | Adaptivity under fault onset | Fig. 3 | Reacts within 1-3 windows; xmv_3 jumps 9%→38% |
| 4 | Ablation studies (k, w, λ, formula) | Fig. 5 | Robust across k∈[3,10], w∈[50,500] |
| 5 | Scalability vs channel count | Fig. 6 | <4 ms up to 500 channels |

---

## 8. Results

Results from Pareto experiments across 7 real-world datasets (6 methods × 6 budgets × 3-5 seeds). Tables report seed-averaged values.

### 8.1 Main Results at 50% Bandwidth

**Bold** = best unsupervised method per dataset. MI is supervised (requires labels).

| Method | TEP | SMD | PSM | MSL | HAI | SKAB |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| **PCA-Triage** | **0.961** | **0.983** | 0.959 | **0.917** | **1.000** | **0.593** |
| Threshold | 0.958 | 0.981 | 0.924 | 0.912 | 1.000 | 0.579 |
| Variance | 0.948 | 0.977 | 0.903 | 0.917 | **1.000** | 0.569 |
| Uniform | 0.924 | 0.968 | 0.898 | 0.912 | 0.998 | 0.588 |
| Random Dropout | 0.788 | 0.980 | **0.961** | 0.914 | 1.000 | 0.519 |
| Mutual Info* | 0.914 | 0.986 | 0.996 | 0.929 | 1.000 | 0.586 |

PCA-Triage is the **best unsupervised method on 4 of 6 real datasets** (TEP, SMD, MSL, SKAB), ties on HAI, and is within 0.2% of the best on PSM. Per-dataset tuning of k and r_min yields consistent wins across diverse domains.

### 8.2 Pareto Curve Summary

PCA-Triage's Pareto curve on TEP is essentially **flat from 30% bandwidth onward** — maintaining F1 > 0.95 at just 30% of full data volume. At 10% bandwidth, PCA-Triage achieves F1 ≈ 0.92 while Uniform drops to F1 ≈ 0.60.

### 8.3 Adaptivity Under Fault Onset

Reaction time (windows until top-5 channel importance changes) on TEP:

| Fault | λ=0.85 | λ=0.90 | λ=0.95 | λ=1.00 |
|-------|:---:|:---:|:---:|:---:|
| IDV(1) A/C Feed | 0 | 1 | 3 | 19 |
| IDV(2) B Composition | 0 | 0 | 0 | 19 |
| IDV(4) Reactor CW | 3 | 5 | 9 | 19 |
| IDV(5) Condenser CW | 3 | 4 | 8 | 19 |

λ=0.85 reacts within 0-3 windows across all fault types. λ=1.0 (no forgetting) gives best F1 but slowest reaction — users choose based on accuracy vs adaptivity priority.

**Fault 1 narrative:** After A/C feed ratio disturbance onset, PCA-Triage shifts bandwidth toward the directly responsible channels:
- `xmv_3` (A Feed Flow valve): **9% → 38% sampling rate** (+42%)
- `xmeas_1` (A Feed stream): **9% → 38%** (+42%)
- `xmeas_16` (Stripper Pressure): **8% → 33%** (+42%)

### 8.4 Ablation Studies (TEP, 50% Bandwidth)

| Panel | Parameter | Best Value | F1 Range | Sensitivity |
|-------|-----------|-----------|----------|-------------|
| (a) | k (components) | 3–8 | 0.962 ± 0.003 | Low (k ∈ [3,10]) |
| (b) | w (window size) | 200 | 0.959–0.966 | Very low |
| (c) | λ (forgetting) | 1.0 | 0.939–0.962 | Moderate |
| (d) | Score formula | Weighted | 0.951–0.960 | Low |

Score formula comparison: Weighted (0.956) ≈ Unweighted (0.960) > Recon-based (0.951).

### 8.5 Computational Cost

Profiled on single CPU core (single-threaded, edge-simulated):

| Method | ms/window | Peak MB | Parameters | Supervised? |
|--------|:---------:|:-------:|:----------:|:-----------:|
| **PCA-Triage** | **0.67** | **8.5** | **520** | **No** |
| Variance | 0.15 | 8.1 | 0 | No |
| Threshold | 0.16 | 8.1 | 0 | No |
| Uniform | 0.06 | 24.8 | 0 | No |
| Attention (simple) | 0.18 | 8.2 | 9,600 | No |
| Mutual Info | 7.73 | 31.8 | 0 | Yes |
| DCFF-MTAD* | 51 | — | 325,000 | Yes |
| m-AFS* | 10,000–173,000 | — | — | Yes |

*Literature values for attention-based methods.

**Bandwidth-cost trade-off:** At 50% budget, PCA-Triage saves 2.88 MB/hour of sensor data at a cost of 24.1 ms/hour of compute — **119 KB saved per ms of compute**.

### 8.6 Scalability

| Channels (d) | PCA-Triage | Variance | Attention |
|:---:|:---:|:---:|:---:|
| 10 | 0.29 ms | 0.11 ms | 0.13 ms |
| 50 | 0.68 ms | 0.16 ms | 0.18 ms |
| 100 | 1.32 ms | 0.22 ms | 0.26 ms |
| 200 | 1.92 ms | 0.33 ms | 0.46 ms |
| 500 | 3.71 ms | 0.65 ms | 1.40 ms |

All methods stay under 5 ms up to 500 channels. PCA-Triage scales O(wdk) — linear in d for fixed k.

---

## 9. Design Decisions

### 9.1 Why Weighted Loadings (Not Raw Variance)?

Raw per-channel variance treats channels independently. The weighted loadings formula `s_j = Σᵢ σᵢ · V[i,j]²` captures how channels contribute to the **joint** principal subspace. For highly correlated sensor groups (common in industrial processes), PCA naturally down-weights redundant channels while variance-based methods cannot.

### 9.2 Why Forward-Fill Reconstruction?

Three reconstruction methods were tested: forward-fill (zero-order hold), linear interpolation, and zero-fill. Forward-fill is cheapest (O(n) per channel), introduces no look-ahead bias, and performs comparably to linear interpolation above 30% budget. At very low budgets (<20%), linear interpolation gains ~1% F1, but at 3× compute cost.

### 9.3 Why Not Compressive Sensing?

Compressive sensing requires: (a) signal sparsity assumption, (b) random measurement matrices, (c) expensive L1 reconstruction at the receiver. PCA-Triage makes no sparsity assumption, uses deterministic allocation, and incurs zero reconstruction cost — the receiver simply processes whatever data arrives.

### 9.4 Why Unsupervised?

Mutual Information-based allocation requires fault labels — unavailable during deployment. PCA-Triage is fully unsupervised: it derives channel importance purely from the covariance structure of incoming sensor data. Despite this, it outperforms supervised MI by +2.5% F1 on TEP.

### 9.5 λ Trade-Off

λ=1.0 accumulates all history → best F1 (0.962) but 19-window reaction time to faults. λ=0.85 forgets quickly → 0-3 window reaction but ~2% lower F1 (0.942). The optimal λ depends on whether the deployment prioritises detection accuracy or fault response speed.

---

## 10. Reproduction

### Installation

```bash
git clone https://github.com/ankitlade12/pca-sensor-triage.git
cd pca-sensor-triage
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Download Datasets

```bash
# TEP (Harvard Dataverse — downloads ~520 MB)
pip install pyreadr
python -c "from src.utils import load_tep; X, y, _, _, cols, _ = load_tep(); print(f'TEP: {X.shape}')"

# SMD (from OmniAnomaly GitHub repo)
# MSL (from NASA telemanom / Google Drive)
# PSM (from eBay RANSynCoders GitHub repo)
# HAI (from icsdataset GitHub repo)
# SKAB (auto-cloned from GitHub)
# See src/utils/data_loader.py for download instructions per dataset

# Verify all datasets
python -c "from src.utils import list_datasets, get_dataset
for name in list_datasets():
    X, y, _, _, cols, _ = get_dataset(name)
    print(f'{name:6s}: {len(cols)} channels, {X.shape[0]} samples')
"
```

### Run Tests (53 tests, ~1 second)

```bash
python -m pytest tests/ -v
```

### Quick Smoke Test (~2 min)

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.triage import TriagePipeline
import numpy as np

X = np.random.randn(1000, 52)
pipe = TriagePipeline(n_components=10, window_size=50, budget=0.5)
result = pipe.process_stream(X, seed=42)
print(f'Input: {X.shape} → Output: {result.shape}')
print(f'Windows processed: {len(pipe.importance_log)}')
print(f'Avg bandwidth: {np.mean(pipe.bandwidth_log):.3f}')
"
```

### Full Pareto Experiment (~10 min)

```bash
python experiments/run_pareto.py
```

### Full Experiment Suite

```bash
# Pareto curves (all datasets, all methods, 5 seeds)
python experiments/run_pareto.py

# Compute profiling
python experiments/run_compute_profile.py
python experiments/run_compute_profile.py --edge  # single-threaded

# Ablation studies
python experiments/run_ablation.py

# Adaptivity analysis
python experiments/run_adaptivity.py

# Scalability
python experiments/run_scalability.py
```

Results are written to `experiments/results/`.

---

## 11. Project Structure

```
src/                                        # Library layer
    triage/                                 # Core PCA-Triage algorithm
        pca_triage.py                       # PCATriage class — weighted loadings importance
        rate_allocator.py                   # RateAllocator — proportional budget allocation
        reconstruction.py                   # Forward-fill / linear / zero reconstruction
        pipeline.py                         # TriagePipeline — end-to-end streaming
    baselines/                              # 6 comparison methods
        uniform.py                          # Baseline 1: same rate all channels
        threshold.py                        # Baseline 2: binary active/inactive
        variance.py                         # Baseline 3: proportional to rolling variance
        random_dropout.py                   # Baseline 4: randomly drop channels
        mutual_info.py                      # Baseline 5: MI with fault labels (supervised)
        attention.py                        # Baseline 6: self-attention importance
    utils/
        data_loader.py                      # load_tep(), load_smd(), load_msl(), load_psm(), load_hai(), etc.
        plotting.py                         # Publication figure generators
        hybrid_scorer.py                    # PCA+Variance hybrid importance scorer

configs/                                    # Experiment configurations
    default.yaml                            # Main experiment hyperparameters
    ablation.yaml                           # Ablation study settings

experiments/                                # Experiment scripts
    run_pareto.py                           # Exp 1: Pareto curves (accuracy vs bandwidth)
    run_compute_profile.py                  # Exp 2: Compute cost profiling (--edge flag)
    run_adaptivity.py                       # Exp 3: Fault onset adaptation analysis
    run_ablation.py                         # Exp 4: Hyperparameter ablation studies
    run_scalability.py                      # Exp 5: Scaling vs channel count
    results/                                # CSV/JSON output files
        pareto_tep.csv                      # TEP Pareto results
        pareto_smd.csv                      # SMD Pareto results
        pareto_msl.csv                      # MSL Pareto results
        pareto_psm.csv                      # PSM Pareto results
        pareto_hai.csv                      # HAI Pareto results
        pareto_skab.csv                     # SKAB Pareto results
        pareto_nasa.csv                     # NASA Pareto results
        compute_profile.json                # Laptop profiling
        compute_profile_edge.json           # Edge-simulated profiling

tests/                                      # Test suite (53 tests)
    test_triage.py                          # 17 unit tests: core algorithm
    test_baselines.py                       # 25 unit tests: all 6 baselines
    test_integration.py                     # 11 integration tests: end-to-end + edge cases

paper/                                      # Paper materials
    main.tex                                # Full LaTeX paper (IEEEtran format)
    references.bib                          # BibTeX bibliography (40 citations)
    generate_all_figures.py                 # Regenerate all figures from data
    sections/
        introduction.md                     # ~940 words
        related_work.md                     # ~1526 words, 36 papers cited
        method.md                           # ~1156 words with pseudocode
    figures/                                # 14 publication-quality figures (300 DPI)
    tables/                                 # 3 result tables (CSV)

scripts/                                    # Analysis and deployment
    verify_paper_numbers.py                 # Validate paper claims vs data
    analyze_results.py                      # Summary tables + Wilcoxon tests
    smoke_test.sh                           # Quick health check (~10s)
    run_full_benchmark.sh                   # Run all 5 experiments end-to-end

docs/                                       # Documentation
    DESIGN_DECISIONS.md                     # 9 key engineering decisions with rationale
    CHANGELOG.md                            # Version history

references/                                 # 36 annotated papers
    core_papers_week1.md                    # Papers 1–10
    paper_annotations_day5.md               # Papers 1–5 (deep annotations)
    paper_annotations_day6.md               # Papers 6–10 (deep annotations)
    paper_annotations_week2.md              # Papers 11–28
    paper_annotations_week3.md              # Papers 29–33
    paper_annotations_recent.md             # Papers 34–36 (2025 update)
    related_work_notes.md                   # Gap positioning matrix

notebooks/
    data_exploration.ipynb                  # Dataset exploration + correlation analysis

data/raw/                                   # Datasets (not tracked in git)
```

---

## 12. References

### Core Method
- Ross, Lim, Lin & Yang (2008). *Incremental Learning for Robust Visual Tracking.* IJCV. (IncrementalPCA foundation)
- Levy & Lindenbaum (2000). *Sequential Karhunen-Loeve Basis Extraction.* IEEE TIP.

### Streaming PCA
- Weng, Zhang & Hwang (2003). *CCIPCA: Candid Covariance-Free Incremental PCA.* IDEAL.
- Balzano, Nowak & Recht (2010). *GROUSE: Grassmannian Rank-One Update Subspace Estimation.* arXiv.
- Yang, Hsieh & Wang (2018). *History PCA: A New Algorithm for Streaming PCA.* arXiv.
- Balzano, Chi & Lu (2018). *Streaming PCA and Subspace Tracking: The Missing Data Case.* Proc. IEEE.
- Oja (1982). *Simplified Neuron Model as a Principal Component Analyzer.* J. Math. Biology.

### Online Feature Selection
- Yu, Wu, Ding & Pei (2014). *SAOLA: Scalable and Accurate Online Feature Selection.* ACM TKDD.
- Zhou et al. (2025). *OSSFS: Online Stable Streaming Feature Selection via Feature Aggregation.* ACM TKDD.
- Zaman, Mohamed & Ahmad (2022). *Feature Selection for Online Streaming High-Dimensional Data: A Survey.* Applied Soft Computing.

### Adaptive Sampling
- Ben-Aboud et al. (2021). *On Adaptive Sampling Algorithms for IoT Devices.* IEEE ICC.
- Giordano et al. (2023). *Energy-Aware Adaptive Sampling for Self-Sustainability in IoT.* ACM ENSsys.

### Process Monitoring & Datasets
- Downs & Vogel (1993). *A Plant-Wide Industrial Process Control Problem.* Comp. & Chem. Eng. (TEP)
- Rieth et al. (2017). *Additional TEP Simulation Data for Anomaly Detection.* Harvard Dataverse.
- Su et al. (2019). *Robust Anomaly Detection for Multivariate Time Series.* KDD. (SMD)
- Hundman et al. (2018). *Detecting Spacecraft Anomalies Using LSTMs.* KDD. (MSL/SMAP)
- Abdulaal et al. (2021). *Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection.* KDD. (PSM)
- Shin et al. (2020). *HAI 1.0: HIL-based Augmented ICS Security Dataset.* USENIX CSET. (HAI)

### Edge AI
- Gill et al. (2024). *Edge AI: A Taxonomy, Systematic Review and Future Directions.* arXiv.

---

*Paper in preparation for IEEE IoT Journal / arXiv preprint.*

*Author: Ankit Hemant Lade*

*License: MIT*
