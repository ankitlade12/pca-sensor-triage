# Paper Annotations — Week 2
## 18 Additional Papers (Total: 28 papers read)

---

## A. Online/Streaming Feature Selection (8 papers → Papers 11-18)

### Paper 11: SAOLA — Scalable Accurate Online Feature Selection
- **Authors:** Kui Yu, Xindong Wu, Wei Ding, Jian Pei
- **Year:** 2014 | **Venue:** IEEE ICDM 2014, ACM TKDD
- **Key idea:** Online pairwise comparison with theoretical bound on information gain gap vs optimal. Uses sequential probability ratio test (SPRT).
- **Result:** Comparable accuracy to OSFS/Alpha-investing with significantly fewer features.
- **Gap for us:** Binary include/exclude, supervised, classification-oriented. No proportional allocation.
- **Cite for:** Related Work §2.2 — key streaming FS algorithm.

### Paper 12: Alpha-Investing for Streaming Feature Selection
- **Authors:** Zhou et al.
- **Year:** 2006 | **Venue:** JMLR
- **Key idea:** Dynamically adjusts significance thresholds for testing feature relevance. Handles feature sets of unknown/infinite size.
- **Limitation:** Fails to investigate redundant features → unpredictable accuracy, large selected feature sets.
- **Gap for us:** No redundancy handling, supervised only, binary selection.
- **Cite for:** Related Work §2.2 — foundational streaming FS method.

### Paper 13: OSFS — Online Streaming Feature Selection
- **Authors:** Wu et al.
- **Year:** 2013 | **Venue:** ICML
- **Key idea:** Two-step process: (1) online relevance analysis, (2) online redundancy analysis via conditional independence. Finds approximate Markov blanket.
- **Limitation:** Running time increases exponentially with increasing features with low redundancy and high relevance.
- **Gap for us:** Supervised, binary, exponential scaling on certain data distributions.
- **Cite for:** Related Work §2.2 — most cited streaming FS algorithm.

### Paper 14: OFSVMB — Online Feature Selection via Markov Blanket
- **Authors:** Kamaru Zaman et al.
- **Year:** 2022 | **Venue:** MDPI Symmetry
- **Key idea:** Finds full Markov blanket (parents + children + spouses) online, not just parent-child (PC) set. Compared with IAMB, STMB, HITON-MB, BAMB, EEMB + OSFS, Alpha-investing, SAOLA.
- **Result:** Higher accuracy than OSFS/Alpha-investing/SAOLA on 14 real-world datasets across 12 classifiers, but slower.
- **Gap for us:** Still supervised, binary, high compute. Our PCA approach is unsupervised and O(wdk).
- **Cite for:** Related Work §2.2 — latest in Markov blanket streaming FS.

### Paper 15: Multi-Conditional Independence Streaming FS
- **Authors:** You et al.
- **Year:** 2020 | **Venue:** Int. J. Computational Intelligence Systems
- **Key idea:** Uses multi-conditional independence tests + mutual information entropy for streaming feature selection. Addresses redundancy through multiple independence conditions.
- **Gap for us:** Still classification-oriented, binary select/reject, requires labels.
- **Cite for:** Related Work §2.2 — redundancy-aware variant.

### Paper 16: Self-Adaption Sliding-Window Streaming FS
- **Authors:** You, Wu et al.
- **Year:** 2019 | **Venue:** IEEE Access
- **Key idea:** Extension of OSFSW with self-adaptive window sizing. Window adjusts to feature arrival rate and redundancy level.
- **Gap for us:** Improved windowing but same fundamental limitations (binary, supervised).
- **Cite for:** Related Work §2.2 — windowed streaming FS.

### Paper 17: Feature Selection for TEP Fault Detection
- **Authors:** Mnassri et al. (+ Galli context)
- **Year:** 2015 | **Venue:** Applied Intelligence
- **Key idea:** Benchmarks feature selection methods on TEP: mutual information (max-dependency, max-relevance, min-redundancy), STRASS (k-way correlation), PCA-based selection. Compares which sensors are most important for different fault types.
- **Key finding:** Different faults activate different sensor subsets → supports our adaptive approach.
- **Gap for us:** All methods are offline/batch — no streaming, no bandwidth allocation.
- **Cite for:** Related Work §2.1 + Experiments — validates that sensor importance varies by fault type.

### Paper 18: Galli — Feature Selection in Machine Learning (Book)
- **Authors:** Soledad Galli
- **Year:** 2022 | **Source:** Leanpub + Feature-engine library
- **Key idea:** Comprehensive treatment of feature selection: filter (variance, correlation, MI), wrapper (RFE, forward/backward), embedded (Lasso, tree importance). Feature-engine library: 150K+ monthly downloads.
- **Gap for us:** Entirely batch/offline. No streaming variant. Our streaming PCA selector could be contributed as a Feature-engine transformer.
- **Cite for:** Introduction (framing within FS literature), Future Work (Feature-engine integration).

---

## B. Attention-Based Sensor Fusion (5 papers → Papers 19-23)

### Paper 19: m-AFS — Multi-Attention Feature Selection for MTS
- **Authors:** Li Cao, Yanting Chen, Zhiyang Zhang, Ning Gui
- **Year:** 2021 | **Venue:** Computational Intelligence and Neuroscience
- **Key idea:** Parallel attention modules — Attention over Time (AoT) + Attention over Variates (AoV). Element-wise multiplication produces unified feature importance weights.
- **Result:** Correctly identifies relevant variables AND time lags on synthetic data; competitive on 6 UCI datasets. 10-173 seconds for weight generation.
- **Gap for us:** Supervised (needs labels), 10-173s per computation (too slow for edge real-time), O(d²) attention, requires training. Our PCA: unsupervised, <5ms, no training.
- **Cite for:** Related Work §2.4 — attention-based sensor importance (compute comparison target).

### Paper 20: VTT — Variable Temporal Transformer for Anomaly Detection
- **Authors:** Various (2024)
- **Year:** 2024 | **Venue:** Knowledge-Based Systems
- **Key idea:** Transformer with temporal self-attention (model time dependencies) + variable self-attention (model inter-variable correlations). Inter-variable attention reveals which sensors are correlated.
- **Gap for us:** O(d²) compute per step, requires GPU, training-dependent. Our PCA achieves similar inter-variable analysis at O(wdk).
- **Cite for:** Related Work §2.4 — transformer-based sensor importance (our compute advantage).

### Paper 21: DCFF-MTAD — Dual-Channel Feature Fusion
- **Authors:** Zheng Xu, Yumeng Yang, Xinwen Gao, Min Hu
- **Year:** 2023 | **Venue:** Sensors (MDPI)
- **Key idea:** STFT spatial channel + time-based graph attention temporal channel, fused via GRU. Achieves F1 0.89-0.94 on SMAP/SMD/MSL.
- **Compute cost:** 51ms per decision, 325K parameters. 3-28x slower than simpler methods.
- **Gap for us:** High compute (51ms vs our target <5ms), requires training, 325K params too heavy for microcontrollers. Our PCA: ~0 trainable parameters, <5ms.
- **Cite for:** Related Work §2.4 — concrete compute comparison. Table: our method vs attention-based.

### Paper 22: Graph Attention Networks for IoT Anomaly Detection
- **Authors:** Various (2024)
- **Year:** 2024 | **Source:** Multiple (DWT + dual GAT, parallel GAT)
- **Key idea:** Graph attention networks model sensor interdependencies. Multi-head attention assigns importance weights to different sensor nodes. Captures spatial correlations in sensor networks.
- **Gap for us:** Graph construction overhead, O(d²) attention per layer, requires training data. PCA captures same inter-sensor correlations algebraically.
- **Cite for:** Related Work §2.4 — graph-based sensor importance methods.

### Paper 23: Entropy-Based Feature Aggregation for Sensor Networks
- **Authors:** Various (2025)
- **Year:** 2025 | **Venue:** Entropy (MDPI)
- **Key idea:** Entropy centrality quantifies informational importance of sensor nodes. Assigns differentiated attention weights based on uncertainty/influence.
- **Gap for us:** Requires network graph structure, compute-heavy entropy estimation. PCA loadings give similar importance ranking without graph construction.
- **Cite for:** Related Work §2.4 — information-theoretic sensor importance.

---

## C. Streaming PCA / Randomized SVD Algorithms (5 papers → Papers 24-28)

### Paper 24: Halko-Martinsson-Tropp — Randomized SVD
- **Authors:** Nathan Halko, Per-Gunnar Martinsson, Joel A. Tropp
- **Year:** 2011 | **Venue:** SIAM Review (arXiv 0909.4061)
- **Citations:** 5000+
- **Key idea:** Random projection to identify subspace, then compute SVD in reduced space. Only 2(q+1) passes over matrix. Works in streaming/out-of-core mode.
- **Complexity:** O(dmn) for m×n matrix with rank-d approximation. Single-pass variant exists.
- **Gap for us:** Designed for batch/few-pass SVD, not truly online single-window update. Our IncrementalPCA (sklearn) is better suited for window-by-window streaming.
- **Cite for:** Method §3 — if IncrementalPCA is too slow, randomized SVD is a faster alternative. Also cite as theoretical foundation.

### Paper 25: Oja's Rule — Online PCA
- **Authors:** Erkki Oja
- **Year:** 1982 | **Venue:** J. Mathematical Biology
- **Key idea:** Hebbian learning rule for extracting principal component online: w_{t+1} = w_t + η(x_t x_t^T w_t - (w_t^T x_t)² w_t). O(d) per sample for rank-1.
- **Significance:** Foundational online PCA algorithm. All modern streaming PCA methods build on or compare against Oja.
- **Gap for us:** Rank-1 only in original form. Extensions exist for rank-k but convergence can be slow. IncrementalPCA is more practical.
- **Cite for:** Related Work §2.2 — foundational reference.

### Paper 26: Streaming PCA Survey (Balzano, Chi, Lu 2018)
- **Authors:** Laura Balzano, Yuejie Chi, Yue M Lu
- **Year:** 2018 | **Venue:** Proceedings of the IEEE, Vol 106(8)
- **Key idea:** Comprehensive survey of streaming PCA with missing data. Compares algebraic (ISVD, Brand's, PIMC) vs geometric (GROUSE, PAST, PETRELS) methods. Analyzes convergence (ODE method, finite-sample).
- **Key findings:**
  - No universal best algorithm — depends on missingness, dynamics, compute budget
  - GROUSE/PETRELS best for abrupt subspace changes
  - O(dk) memory for modern algorithms
  - Global finite-sample guarantees for missing data case still open
- **Gap for us:** Entire survey on subspace tracking quality — none convert to channel importance for bandwidth allocation.
- **Cite for:** Related Work §2.2 — authoritative survey on streaming PCA landscape.

### Paper 27: PAST — Projection Approximation Subspace Tracking
- **Authors:** Yang (1995)
- **Year:** 1995 | **Venue:** IEEE Trans. Signal Processing
- **Key idea:** Second-order gradient descent for subspace tracking. O(dk) per update. Faster convergence than first-order methods (Oja, GROUSE).
- **Gap for us:** Same as other streaming PCA — no triage step.
- **Cite for:** Method §3 — mention as alternative backbone.

### Paper 28: PETRELS — Parallel Subspace Tracking with Missing Data
- **Authors:** Chi et al.
- **Year:** 2013 | **Venue:** IEEE Trans. Signal Processing
- **Key idea:** Extension of PAST for missing data. Parallelizable row updates. Recursive least squares per row of subspace matrix.
- **Gap for us:** Designed for matrix completion, not sensor triage.
- **Cite for:** Related Work §2.2 — streaming PCA with missing data.

---

## Updated Paper Count: 28 total (target: 30-35 by end of Week 3)

### Category Breakdown
| Category | Papers | Count |
|----------|--------|-------|
| Adaptive Sampling (IoT) | 1, 2 | 2 |
| Online Feature Selection | 3, 4, 11-16, 17, 18 | 10 |
| Streaming PCA Algorithms | 5, 6, 7, 24-28 | 8 |
| PCA Fault Detection | 8, 9 | 2 |
| Attention-Based Sensor | 19-23 | 5 |
| Edge AI | 10 | 1 |
| **Total** | | **28** |
