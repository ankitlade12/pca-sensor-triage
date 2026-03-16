# 1. Introduction

Consider a chemical processing plant instrumented with 200 sensors monitoring temperatures, pressures, flow rates, and valve positions across reactors, separators, and distillation columns. Each sensor streams data at 1 Hz, generating a combined throughput of 1.2 MB per minute. An edge gateway at the plant floor must relay this data over a constrained industrial network—perhaps a LoRaWAN or NB-IoT link—to a centralized fault detection system. The available bandwidth supports only 50% of the full data volume. The plant operator faces a deceptively simple question: *which sensors should receive more of the limited bandwidth, and which can tolerate lower sampling rates without compromising fault detection?*

The naive answer—sample every sensor at a uniformly reduced rate—wastes bandwidth on channels that carry redundant or uninformative signals. A reactor pressure sensor and its nearby temperature sensor may be highly correlated under normal operation; sampling both at full rate adds little information. Meanwhile, a valve position sensor that has been quiescent for hours may suddenly become the earliest indicator of an emerging stiction fault—precisely when it needs higher sampling priority. The ideal allocation is *adaptive*: it shifts bandwidth toward informative channels as operating conditions change.

## The Problem

We formalize this as the *sensor triage* problem. Given $d$ sensor channels and a total bandwidth budget $B < d$ (expressed as a fraction of the uniform full-rate budget), allocate a per-channel sampling rate $r_j \in [r_{\min}, 1]$ for each channel $j = 1, \ldots, d$, subject to $\sum_j r_j \leq B \cdot d$, such that a downstream fault detection model trained on the triaged data maintains accuracy close to one trained on full-rate data. The allocation must be updated online as new data arrives, adapting to changing system dynamics without reprocessing historical data.

This problem sits at the intersection of three well-studied areas—yet none addresses it directly. *PCA-based process monitoring* [8, 9, 29] uses principal component loadings to diagnose which sensors contributed to a detected fault, but only *after* full data has been collected. *Streaming PCA algorithms* [5, 6, 7, 26] efficiently track subspaces from sequential data, but output basis matrices rather than channel-level sampling decisions. *Adaptive sampling in IoT* [1, 2] adjusts temporal sampling rates based on signal dynamics or energy constraints, but applies the same rate to every channel without differentiating by informational importance.

## The Gap

The missing piece is a mechanism that converts streaming PCA's subspace estimates into per-channel importance scores, then maps those scores to proportional sampling rates under a budget constraint. Existing streaming feature selection methods [3, 4, 11, 13] come closest, but they produce *binary* include/exclude decisions rather than proportional allocations, require *supervised* labels for relevance testing, and are designed for classification pipelines rather than sensor bandwidth management. Attention-based methods [19, 20, 21] can learn channel importance but at $O(d^2)$ computational cost with hundreds of thousands of trainable parameters—far too expensive for a microcontroller operating at the network edge.

## Our Approach

We propose **PCA-Triage**, a lightweight streaming algorithm that bridges this gap. At each time window, PCA-Triage performs three steps: (1) update an incremental PCA model from the latest sensor observations, (2) compute per-channel importance scores from the weighted loading matrix $\sum_i \sigma_i |v_{ij}|^2$, and (3) allocate sampling rates proportional to importance scores subject to the bandwidth budget $B$ and a minimum rate floor $r_{\min}$. The entire pipeline runs at $O(wdk)$ per window, where $w$ is the window size, $d$ is the number of channels, and $k$ is the number of retained components—with zero trainable parameters and constant memory.

The key insight is that PCA loadings, traditionally used for post-hoc fault diagnosis, contain exactly the information needed for proactive bandwidth allocation. Channels with high weighted loadings capture directions of maximum variance in the current operating regime; allocating more bandwidth to these channels preserves the information most useful for downstream fault detection. When operating conditions shift—e.g., a fault onset causes new sensors to become informative—the incremental PCA update naturally shifts importance scores, and PCA-Triage reallocates bandwidth within a small number of windows.

## Contributions

We make the following contributions:

1. **Algorithm.** We introduce PCA-Triage, the first streaming algorithm that converts incremental PCA loadings into proportional per-channel sampling rates under a bandwidth budget. The algorithm is unsupervised, requires no training data or labels, and runs at $O(wdk)$ per window with $O(wd)$ memory—suitable for deployment on edge hardware.

2. **Empirical validation.** We evaluate PCA-Triage on three benchmark datasets—the Tennessee Eastman Process (52 sensors, 20 fault types) [29, 32], NASA IMS Bearing (vibration data, remaining useful life prediction), and SKAB (8 sensors, anomaly detection)—against five baselines (uniform sampling, threshold-based, variance-based, random channel dropout, and mutual information). We show that PCA-Triage maintains fault detection F1 $\geq 0.90$ at 50% bandwidth, outperforming all baselines by 3–12 percentage points.

3. **Edge viability.** We demonstrate that PCA-Triage completes each triage decision in under 5 ms on a single CPU core, an order of magnitude faster than attention-based alternatives, while adapting to fault onset within 3 windows (< 15 minutes on TEP). We provide ablation studies over the number of components $k$, window size $w$, and forgetting factor $\lambda$, identifying robust default hyperparameters.

## Paper Organization

The remainder of this paper is organized as follows. Section 2 reviews related work across PCA-based monitoring, streaming PCA, online feature selection, adaptive sampling, and edge AI. Section 3 presents the PCA-Triage algorithm with formal problem statement, pseudocode, and complexity analysis. Section 4 describes our experimental setup and presents results across five experiments: Pareto curves, compute profiling, adaptivity under fault onset, ablation studies, and scalability. Section 5 discusses limitations, connections to the feature selection literature, and future directions including federated PCA and adaptive component selection. Section 6 concludes.
