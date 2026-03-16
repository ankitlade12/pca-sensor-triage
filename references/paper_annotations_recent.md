# Recent Papers Added (2024-2025 Update)
## Papers 34-36 — ensuring no gap overlap

### Paper 34: Low-Precision Streaming PCA
- **Title:** "Low-Precision Streaming PCA"
- **Authors:** Sanjoy Dasgupta, Syamantak Kumar, Shourya Pandey
- **Year:** 2025 | **Venue:** AISTATS 2025
- **Summary:** Estimates top principal component in streaming setting under quantization constraints. Proves convergence guarantees for low-precision arithmetic relevant to edge deployment.
- **Gap for us:** Algorithmic improvement to streaming PCA backbone — no triage step. Could be future work to swap in for even more constrained edge devices.
- **Cite for:** §2.2 — shows streaming PCA remains active research area.

### Paper 35: OSSFS — Online Stable Streaming Feature Selection
- **Title:** "Online Stable Streaming Feature Selection via Feature Aggregation"
- **Authors:** Peng Zhou, Qi Wang, Yunyun Zhang, Zhaolong Ling, Shu Zhao, Xindong Wu
- **Year:** 2025 | **Venue:** ACM TKDD, Vol 19(3)
- **Summary:** Uses cohesive MeanShift to partition streaming features into hyperellipsoids, selects representatives from each. Novel focus on selection *stability* (consistent results across runs).
- **Gap for us:** Still binary select/reject, supervised, classification-oriented. Does not do proportional rate allocation. Confirms that even latest 2025 FS methods don't address our problem.
- **Cite for:** §2.2 — latest in streaming FS, strengthens gap argument.

### Paper 36: ML-DSRA — Parametric ML-Based Adaptive Sampling
- **Title:** "Parametric Machine Learning-Based Adaptive Sampling Algorithm for Efficient IoT Data Collection in Environmental Monitoring"
- **Authors:** Various
- **Year:** 2024 | **Venue:** J. Network and Systems Management (Springer)
- **Summary:** Extends DSRA with ML to auto-tune temporal sampling parameters. Tested on environmental monitoring IoT sensors. Removes need for expert parameter tuning.
- **Gap for us:** Temporal rate only — same rate for all sensors. No per-channel differentiation. Confirms latest adaptive sampling still doesn't address spatial allocation.
- **Cite for:** §2.3 — latest adaptive sampling, strengthens gap.

---

## Conclusion: No competitor found (as of March 2026)
Our gap remains valid: no published work combines streaming PCA + proportional per-channel bandwidth allocation under budget constraints. Total bibliography: **36 papers**.
