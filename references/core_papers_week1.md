# Core Papers — Week 1 (Top 10)
## PCA-Driven Adaptive Sensor Triage for Edge AI Inference

**Search Strategy:** Google Scholar searches on adaptive sampling IoT, incremental PCA, sensor fusion edge AI, PCA fault detection, online feature selection, attention-based sensor processing, CCIPCA/GROUSE algorithms.

---

### Paper 1: Adaptive Sampling for IoT Devices
- **Title:** "On adaptive sampling algorithms for IoT devices"
- **Source:** ResearchGate, 2021
- **URL:** https://www.researchgate.net/publication/348758676_On_adaptive_sampling_algorithms_for_IoT_devices
- **Summary:** Surveys adaptive sampling strategies for IoT, including DSRA (Dynamic Sampling Rate Algorithm). Shows up to 79% energy savings through adaptive sampling without degrading quality.
- **Gap:** Focuses on temporal sampling rate only — does NOT adapt per-channel (which channels to prioritize). Our method fills this by using PCA to rank channel importance.
- **Baselines to note:** DSRA, Send-on-Delta (SoD)
- **Relevance:** HIGH — direct competitor in adaptive sampling space, but orthogonal approach.

---

### Paper 2: Energy-Efficient Adaptive Sampling in WSNs
- **Title:** "Energy-efficient data acquisition by adaptive sampling for wireless sensor networks"
- **Source:** ResearchGate / Sensors, ~2012
- **URL:** https://www.researchgate.net/publication/220761399_Energy-efficient_data_acquisition_by_adaptive_sampling_for_wireless_sensor_networks
- **Summary:** Proposes adaptive sampling exploiting spatial and temporal correlations in sensor data. 80% reduction in sampling frequency achievable.
- **Gap:** Uses fixed statistical models, not data-driven PCA. Does not adapt to changing system dynamics (fault onset).
- **Baselines to note:** Correlation-based sampling reduction
- **Relevance:** HIGH — foundational work in the space.

---

### Paper 3: Online Feature Selection for Streaming Features (OSFSW)
- **Title:** "Online Feature Selection for Streaming Features with High Redundancy Using Sliding-Window Sampling"
- **Source:** IEEE ICDM Workshop, 2018
- **URL:** https://ieeexplore.ieee.org/document/8588794/
- **Summary:** OSFSW uses sliding windows + conditional independence to select features online, discarding irrelevant/redundant ones. Directly addresses redundancy in streaming settings.
- **Gap:** Uses conditional independence tests (statistical), not PCA-based importance. Higher compute cost per decision. Not designed for bandwidth allocation (binary select/reject, not proportional rates).
- **Baselines to note:** OSFSW algorithm
- **Relevance:** HIGH — closest to our approach in the feature selection literature.

---

### Paper 4: Feature Selection for Streaming Data — State-of-the-Art Review
- **Title:** "Feature selection for online streaming high-dimensional data: A state-of-the-art review"
- **Source:** Applied Soft Computing, Vol 127, 2022
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S1568494622005154
- **Summary:** Comprehensive survey of streaming feature selection methods. Covers online group feature selection, multi-label streaming, and redundancy-aware approaches.
- **Gap:** Survey identifies that most methods are classification-oriented — none specifically target bandwidth allocation for sensor triage. Our PCA-based proportional rate allocation is novel.
- **Baselines to note:** Lists all major streaming FS algorithms
- **Relevance:** CRITICAL — use this to build the Related Work positioning matrix.

---

### Paper 5: CCIPCA — Candid Covariance-Free Incremental PCA
- **Title:** "A Fast Algorithm for Incremental Principal Component Analysis"
- **Source:** Springer, IDEAL 2003 (Weng et al.)
- **URL:** https://link.springer.com/chapter/10.1007/978-3-540-45080-1_122
- **Summary:** CCIPCA computes principal components incrementally without covariance matrix estimation. Very fast convergence on high-dimensional data. O(dk) per update.
- **Gap:** Original CCIPCA is designed for single-sample updates — our method uses windowed batches with IncrementalPCA for more stable estimates.
- **Baselines to note:** CCIPCA algorithm
- **Relevance:** HIGH — algorithmic building block. Consider as alternative to sklearn IncrementalPCA.

---

### Paper 6: GROUSE — Grassmannian Rank-One Update Subspace Estimation
- **Title:** "Online Identification and Tracking of Subspaces from Highly Incomplete Information"
- **Source:** Balzano et al., 2010 (arXiv 1006.4046)
- **URL:** https://ar5iv.labs.arxiv.org/html/1006.4046
- **Summary:** GROUSE tracks subspaces via gradient descent on the Grassmannian manifold. Handles missing data naturally. Proven linear convergence rate.
- **Gap:** Designed for subspace tracking, not for channel importance scoring. Missing the "triage" step that converts subspace estimates to per-channel sampling rates.
- **Baselines to note:** GROUSE algorithm, incremental SVD
- **Relevance:** MEDIUM-HIGH — alternative algorithmic approach. Could be used if missing data is an issue.

---

### Paper 7: History PCA — Streaming PCA with Fast Convergence
- **Title:** "History PCA: A New Algorithm for Streaming PCA"
- **Source:** arXiv 1802.05447, 2018
- **URL:** https://arxiv.org/abs/1802.05447
- **Summary:** Achieves faster convergence than existing streaming PCA using O(Bd) memory (B≈10 block size). Addresses the key challenge that optimal number of components changes over time in streaming data.
- **Gap:** Focuses on subspace tracking accuracy, not on downstream task (sensor triage/bandwidth allocation).
- **Baselines to note:** Oja's algorithm, Krasulina's method
- **Relevance:** MEDIUM — potential algorithmic improvement to our core PCA step.

---

### Paper 8: PCA-Based Fault Detection in Tennessee Eastman Process
- **Title:** "Research on fault detection of Tennessee Eastman Process based on PCA"
- **Source:** IEEE Conference, 2013
- **URL:** https://ieeexplore.ieee.org/document/6561084
- **Summary:** Applies standard PCA to TEP for fault detection using T² and SPE statistics. Shows PCA can detect faults quickly in complex nonlinear processes. Uses the same TEP benchmark we plan to use.
- **Gap:** Uses PCA for detection only (normal vs fault), NOT for channel prioritization/triage. Static PCA, not streaming. Does not consider bandwidth constraints.
- **Baselines to note:** T² statistic, SPE/Q statistic
- **Relevance:** HIGH — validates PCA on our primary dataset. We need to clearly differentiate from this line of work.

---

### Paper 9: Data-Driven Process Monitoring and Fault Diagnosis — Comprehensive Survey
- **Title:** "Data-Driven Process Monitoring and Fault Diagnosis: A Comprehensive Survey"
- **Source:** MDPI Processes, Vol 12, 2024
- **URL:** https://www.mdpi.com/2227-9717/12/2/251
- **Summary:** Surveys PCA, PLS, autoencoders, neural networks for process monitoring. Covers dynamic PCA (DPCA), kernel PCA, and hybrid methods. TEP is the standard benchmark throughout.
- **Gap:** All methods assume full data availability — none address the scenario where bandwidth is limited and you must choose which sensors to prioritize.
- **Baselines to note:** DPCA, KPCA, deep learning approaches
- **Relevance:** CRITICAL — positions our work relative to the entire fault detection literature. The bandwidth constraint is our unique angle.

---

### Paper 10: Edge AI for IoT — Taxonomy and Systematic Review
- **Title:** "Edge AI: A Taxonomy, Systematic Review and Future Directions"
- **Source:** arXiv 2407.04053, 2024
- **URL:** https://arxiv.org/html/2407.04053v1
- **Summary:** Comprehensive taxonomy of edge AI techniques. Covers model compression, split computing, early exit strategies. Discusses sensor fusion and on-device inference for IoT.
- **Gap:** Focuses on model-side optimization (compress the model), not data-side optimization (compress the sensor input). Our method is complementary — reduce what data reaches the model.
- **Baselines to note:** Model compression, knowledge distillation, split computing
- **Relevance:** HIGH — frames our contribution within the broader edge AI landscape.

---

## Gap Positioning Summary

| Approach | Static | Streaming/Adaptive |
|----------|--------|--------------------|
| **Fixed rules** (threshold, uniform) | Uniform sampling, SoD | DSRA, energy-aware adaptive |
| **Data-driven** (learned importance) | Batch PCA fault detection, offline feature selection | **OUR METHOD: Streaming PCA triage** |

**The gap:** No existing method uses streaming PCA to dynamically allocate per-channel sampling rates under bandwidth constraints. Existing work either:
1. Adapts sampling rate globally (not per-channel)
2. Uses PCA for fault detection (not for bandwidth triage)
3. Does feature selection (binary include/exclude, not proportional allocation)
4. Uses attention mechanisms (too compute-heavy for edge)

Our method sits uniquely at the intersection of streaming PCA + per-channel adaptive bandwidth allocation.
