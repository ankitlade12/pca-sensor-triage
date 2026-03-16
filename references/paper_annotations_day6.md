# Paper Annotations — Day 6 (Detailed Reading)
## Remaining 5 Papers Read

---

## Paper 1: Adaptive Sampling for IoT Devices
**"On adaptive sampling algorithms for IoT devices"**
- **Authors:** Yassine Ben-Aboud, Daniel Bonilla Licea, Mounir Ghogho, Abdellatif Kobbane
- **Year:** 2021 | **Venue:** IEEE ICC 2021
- **Citations:** 11

### Summary
Proposes two new adaptive sampling techniques for IoT: a lightweight adaptive sampling algorithm and an optimized uniform sampling method. Uses Kalman filters to predict signal values and adjust sampling intervals. Tested on real-world air pollution and surface temperature datasets. Achieves noticeable reduction in computational load vs. state-of-the-art while maintaining data quality (measured by MAPE).

### Key Technical Details
- Sampling adapts the **temporal rate** (how often to sample) based on signal dynamics
- Uses Kalman filter predictions to decide when the next sample is needed
- Compared against DSRA (Dynamic Sampling Rate Algorithm) and exhaustive search
- Metrics: Mean Absolute Percentage Error (MAPE), data volume reduction

### Gap for Our Paper
- Adapts **temporal** sampling rate only — same rate for ALL sensors
- Does NOT differentiate between channels (which sensor matters more?)
- No PCA or multivariate analysis — treats each sensor independently
- Our method: per-channel adaptive rates based on PCA importance scores

### What We Cite This For
- Related Work §2.3: "Adaptive Sampling in IoT" subsection
- Baseline comparison: their temporal adaptation is orthogonal to our channel-level triage
- Position as: they optimize WHEN to sample, we optimize WHAT to sample

---

## Paper 2: Energy-Aware Adaptive Sampling (ETH Zurich)
**"Energy-Aware Adaptive Sampling for Self-Sustainability in Resource-Constrained IoT Devices"**
- **Authors:** Marco Giordano et al. (ETH Zurich / Hilti Corporation)
- **Year:** 2023 | **Venue:** ACM ENSsys Workshop (arXiv 2310.20331)

### Summary
Develops an energy-harvesting-aware adaptive sampling algorithm using a finite state machine inspired by TCP Reno's AIMD (additive-increase/multiplicative-decrease). Adjusts sampling rate based on battery state-of-charge as a proxy for available energy. Tested on EcoTrack IoT platform with solar harvesting across Barcelona, Munich, London.

### Key Results
- 85-95% of theoretical optimal performance across 3 cities
- 2.06x (Munich) to 8.68x (London) more daily localizations vs constant-rate
- Algorithm overhead: only 527 CPU cycles/day, O(1) memory
- 25-160x lower compute cost than Finite Horizon Control

### Key Technical Details
- FSM with 3 states: Increase (add 1 sample/hr), Hold, Decrease (halve rate)
- Decision based on daily battery charge change metric
- Completely agnostic to energy source — uses battery SoC as proxy
- Designed for single-sensor tracking (GNSS position), not multi-sensor triage

### Gap for Our Paper
- Optimizes sampling rate for **energy sustainability**, not for **information content**
- Single-sensor system — no notion of channel importance or prioritization
- Our method: allocates budget across multiple channels based on information value (PCA importance)

### What We Cite This For
- Related Work §2.3: example of practical edge IoT adaptive sampling
- Discussion: our method could be combined with energy-aware approaches (energy budget → total bandwidth, then PCA-triage allocates across channels)

---

## Paper 3: OSFSW — Online Feature Selection with Sliding Window
**"Online Feature Selection for Streaming Features with High Redundancy Using Sliding-Window Sampling"**
- **Authors:** Dianlong You, Xindong Wu, Limin Shen, Zhen Chen, Chuan Ma, Song Deng
- **Year:** 2018 | **Venue:** IEEE ICBK 2018

### Summary
Proposes OSFSW, an online feature selection algorithm that uses sliding windows to sample streaming features and conditional independence tests to discard irrelevant and redundant features. Finds an approximate Markov blanket in a smaller number of selected features. Tested on NIPS 2003 and Causality Workbench datasets.

### Key Technical Details
- **Sliding window:** prevents feature overflow in high-redundancy streams
- **Conditional independence:** G² test to assess feature relevance/redundancy
- **Markov blanket:** approximates minimal sufficient feature set
- Improves on Alpha-investing, OSFS, SAOLA in prediction accuracy with fewer selected features

### Gap for Our Paper
- **Binary selection:** features are either IN or OUT — no proportional allocation
- **Classification-oriented:** designed for supervised learning, not sensor bandwidth management
- **Per-feature statistics:** requires labeled data for conditional independence tests
- **Higher compute:** G² tests are more expensive than PCA loading computation
- Our method: proportional rate allocation (0-100% per channel), unsupervised (no labels needed), cheaper per decision

### What We Cite This For
- Related Work §2.2: closest work in streaming feature selection
- Positioning matrix: they do binary streaming selection, we do proportional streaming allocation
- Baseline consideration: could adapt OSFSW as a baseline (selected channels get full rate, others get minimum)

---

## Paper 6: GROUSE — Grassmannian Subspace Tracking
**"Online Identification and Tracking of Subspaces from Highly Incomplete Information"**
- **Authors:** Laura Balzano, Robert Nowak, Benjamin Recht (UW-Madison)
- **Year:** 2010 | **Source:** arXiv 1006.4046
- **Citations:** 250+

### Summary
GROUSE performs incremental gradient descent on the Grassmannian manifold to track d-dimensional subspaces from sparsely sampled vectors. Uses elegant rank-one updates that maintain orthonormality without reorthogonalization. Handles missing data naturally. Proven convergence to stationary points under diminishing stepsizes.

### Key Technical Details
- **Per-iteration:** O(nd + |Ω|d²) flops, O(nd) memory
- **Update:** rank-one modification using geodesic on Grassmannian
- **Missing data:** built-in — only uses observed entries Ω_t at each step
- **Convergence:** proven under diminishing stepsizes (η_t ∝ 1/t)
- **Tracking:** constant stepsizes enable tracking of time-varying subspaces
- **Matrix completion:** 2x faster than NNLS (fastest batch method) with comparable quality

### Algorithm (simplified):
```
For each new (partial) observation v_t:
  1. Solve least squares: w = argmin ||Δ_Ω(U·a - v_t)||²
  2. Predict: p = U·w
  3. Residual: r = Δ_Ω(v_t - p)
  4. Update U via rank-one geodesic step
```

### Comparison with Our Approach
| Aspect | GROUSE | Our Method |
|--------|--------|-----------|
| Goal | Subspace tracking | Channel importance → bandwidth |
| Input | Partial observations | Full sensor windows |
| Output | Updated subspace U | Per-channel sampling rates |
| Missing data | Native support | Not needed (we have full data) |
| Compute | O(nd + |Ω|d²) per sample | O(wdk) per window |

### Gap for Our Paper
- GROUSE tracks subspaces excellently but does NOT convert to channel importance scores
- No notion of bandwidth budget or rate allocation
- Designed for missing data recovery, not proactive data triage
- Our method adds the "triage layer" on top of subspace estimation

### What We Cite This For
- Method §3: alternative streaming PCA algorithm we could swap in
- Related Work §2.2: streaming subspace tracking methods
- Future Work: GROUSE + triage for scenarios with missing sensor data

---

## Paper 7: History PCA — Fast Streaming PCA
**"History PCA: A New Algorithm for Streaming PCA"**
- **Authors:** Puyudi Yang, Cho-Jui Hsieh, Jane-Ling Wang
- **Year:** 2018 | **Source:** arXiv 1802.05447

### Summary
Proposes a streaming PCA algorithm that leverages information from previously streamed data to achieve faster convergence. Uses O(Bd) memory where B≈10 is a small block size. Provides theoretical convergence guarantees with rate of convergence. Outperforms Oja's algorithm, Krasulina's method, and other streaming PCA approaches on synthetic and real-world datasets.

### Key Technical Details
- **Key insight:** don't discard past data immediately — keep a small block of B recent samples
- **Memory:** O(Bd) ≈ O(10d), can be reduced to O(d) with fewer inner iterations
- **Convergence:** provably faster than Oja's algorithm
- **Theoretical:** formal convergence guarantees with rate analysis
- **Practical:** simple to implement, minimal hyperparameters

### Comparison with sklearn IncrementalPCA
| Aspect | History PCA | IncrementalPCA |
|--------|------------|----------------|
| Memory | O(Bd) | O(batch_size × d) |
| Convergence | Proven faster | Batch-dependent |
| Implementation | Custom | Off-the-shelf |
| Adaptivity | Natural forgetting | Manual control |

### Gap for Our Paper
- Like GROUSE, focuses on subspace estimation quality — no triage step
- No bandwidth allocation or channel importance scoring
- Could be a better backbone than IncrementalPCA for our algorithm

### What We Cite This For
- Method §3: algorithmic building block option
- Ablation study: compare IncrementalPCA vs History PCA vs CCIPCA as backbone
- Related Work §2.2: streaming PCA algorithms
