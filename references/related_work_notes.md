# Related Work Notes
## PCA-Driven Adaptive Sensor Triage for Edge AI Inference

---

## 2×2 Positioning Matrix

|  | **Static / Batch** | **Streaming / Adaptive** |
|--|---------------------|--------------------------|
| **Fixed Rules** (threshold, uniform, heuristic) | Uniform sampling [baseline]; Threshold-based adaptive (SoD) | DSRA [Ben-Aboud 2021]; Energy-aware AIMD [Giordano 2023]; Send-on-Delta variants |
| **Data-Driven** (learned importance) | Batch PCA fault detection [Chen 2013]; Offline feature selection; Mutual information ranking | **OUR METHOD: PCA-Triage** — streaming PCA → per-channel bandwidth allocation |

### Reading the Matrix
- **Top-left (Static + Fixed):** Oldest approaches. Simple but waste bandwidth on uninformative channels.
- **Top-right (Streaming + Fixed):** Adapt temporal rate based on rules but don't differentiate channels by importance.
- **Bottom-left (Static + Data-driven):** Use data to identify important features but can't adapt online to changing dynamics.
- **Bottom-right (Streaming + Data-driven):** **OUR CONTRIBUTION.** Only method that uses streaming PCA to dynamically allocate per-channel rates.

### Why the Gap Exists
1. **Adaptive sampling community** focuses on temporal rate (when), not spatial selection (which channel)
2. **PCA community** uses loadings for post-hoc diagnosis, never for proactive data acquisition
3. **Feature selection community** does binary include/exclude, not proportional allocation
4. **Edge AI community** optimizes models, not data acquisition

---

## Related Work Outline (4 subsections)

### §2.1 PCA for Process Monitoring and Fault Detection
**Papers:** [8] Chen 2013, [9] Process Monitoring Survey 2024, BibMon TEP tutorial
**Narrative:** PCA is the workhorse of industrial process monitoring. T² and SPE statistics detect faults; loading analysis identifies contributing variables. Dynamic PCA (DPCA), Kernel PCA (KPCA), and deep learning variants (autoencoders, LSTM) extend the basic framework. TEP is the standard benchmark with 52 sensors and 20 fault types.
**Closing gap statement:** "However, all existing PCA-based monitoring methods assume that full-resolution data from all sensors is available. None address the practical scenario where bandwidth or energy constraints require prioritizing which channels to sample at higher rates — the precise gap our method fills."

### §2.2 Streaming PCA and Online Feature Selection
**Papers:** [5] CCIPCA (Weng 2003), [6] GROUSE (Balzano 2010), [7] History PCA (Yang 2018), [3] OSFSW (You 2018), [4] Streaming FS Survey (Zaman 2022), Streaming PCA survey (Balzano & Chi 2018)
**Narrative:** Streaming PCA algorithms (CCIPCA, GROUSE, History PCA, Oja's method) efficiently track subspaces from sequential data with O(dk) updates. Online feature selection methods (OSFS, OSFSW, SAOLA, Alpha-investing) select relevant features in streaming settings using statistical tests. Both communities produce ranked importance but neither converts rankings into proportional bandwidth allocation.
**Closing gap statement:** "Streaming PCA methods output subspace estimates; streaming feature selection methods output binary include/exclude decisions. Neither produces per-channel sampling rates under a bandwidth budget — the output our triage algorithm generates."

### §2.3 Adaptive Sampling in IoT and Sensor Networks
**Papers:** [1] Ben-Aboud 2021, [2] Giordano 2023, DSRA (2019), AdaM framework
**Narrative:** Adaptive sampling reduces energy and bandwidth consumption by adjusting sampling frequency based on signal dynamics (DSRA), energy availability (AIMD), or prediction error (Kalman-based). These methods achieve 30-80% data reduction while maintaining quality. However, they treat all sensors identically — the same temporal rate applies to every channel.
**Closing gap statement:** "Existing adaptive sampling methods answer 'how often should I sample?' but not 'which channels deserve more of my limited bandwidth?' Our PCA-triage method answers the latter question, and is orthogonal to (and combinable with) temporal adaptation."

### §2.4 Edge AI and Sensor Data Processing
**Papers:** [10] Edge AI Taxonomy (Gill 2024), Edge ML for IoT survey (2020), attention-based sensor processing papers
**Narrative:** Edge AI focuses on deploying ML models on resource-constrained devices through model compression, knowledge distillation, split computing, and early exit strategies. Attention mechanisms (self-attention, channel attention, SENet) learn sensor importance but at O(d²) compute cost, prohibitive for real-time edge deployment. The entire Edge AI literature optimizes the model side of the pipeline.
**Closing gap statement:** "While Edge AI optimizes what happens after data reaches the model, our method optimizes what data reaches the model in the first place. PCA-triage is a data-side optimization that complements existing model-side techniques, running at O(wdk) — orders of magnitude cheaper than attention-based alternatives."

---

## Key Differentiation Arguments

### vs. Adaptive Sampling (Papers 1, 2)
| Theirs | Ours |
|--------|------|
| Adapt temporal rate (when) | Adapt channel allocation (which) |
| Same rate for all sensors | Different rate per sensor |
| Signal-prediction based | PCA-importance based |
| Univariate | Multivariate (captures correlations) |

### vs. PCA Fault Detection (Papers 8, 9)
| Theirs | Ours |
|--------|------|
| Use PCA for detection (normal vs fault) | Use PCA for triage (which channels to prioritize) |
| Post-hoc diagnosis (after data collected) | Proactive acquisition (before data collected) |
| Assume full data available | Operate under bandwidth constraints |
| Static PCA on batch data | Streaming PCA with forgetting |

### vs. Streaming Feature Selection (Papers 3, 4)
| Theirs | Ours |
|--------|------|
| Binary: include or exclude | Proportional: 0-100% rate per channel |
| Supervised (needs labels) | Unsupervised (PCA is label-free) |
| Statistical tests (G², MI) | PCA loadings (algebraic, fast) |
| Designed for ML pipelines | Designed for sensor bandwidth |

### vs. Streaming PCA (Papers 5, 6, 7)
| Theirs | Ours |
|--------|------|
| Output: updated subspace U | Output: per-channel sampling rates |
| Goal: track subspace accurately | Goal: allocate bandwidth optimally |
| No downstream task | Downstream: maintain fault detection accuracy |
| No budget constraint | Budget B constrains total bandwidth |

### vs. Attention-Based Methods (Paper 10 context)
| Theirs | Ours |
|--------|------|
| O(d²) compute per step | O(wdk) per window |
| Requires training (supervised) | No training (unsupervised PCA) |
| GPU needed for real-time | CPU-only, edge-viable |
| Learns channel importance | Computes channel importance |

---

## Papers Catalogue (10/10 read)

| # | Short Title | Year | Category | Relevance |
|---|------------|------|----------|-----------|
| 1 | Adaptive Sampling IoT | 2021 | Adaptive Sampling | HIGH |
| 2 | Energy-Aware AIMD Sampling | 2023 | Adaptive Sampling | HIGH |
| 3 | OSFSW Sliding-Window FS | 2018 | Online Feature Selection | HIGH |
| 4 | Streaming FS Survey | 2022 | Online Feature Selection | CRITICAL |
| 5 | CCIPCA | 2003 | Streaming PCA | HIGH |
| 6 | GROUSE | 2010 | Streaming PCA | MEDIUM-HIGH |
| 7 | History PCA | 2018 | Streaming PCA | MEDIUM |
| 8 | PCA on TEP | 2013 | PCA Fault Detection | HIGH |
| 9 | Process Monitoring Survey | 2024 | PCA Fault Detection | CRITICAL |
| 10 | Edge AI Taxonomy | 2024 | Edge AI | HIGH |

**Total papers read & annotated: 10/10**
**Next target: 15 more in Week 2 (deep dive into each subsection)**
