# Experimental Summary
## PCA-Driven Adaptive Sensor Triage for Edge AI Inference

---

## Top 3 Claims (with supporting evidence)

### Claim 1: PCA-Triage maintains fault detection accuracy at 50% bandwidth
On TEP (52 sensors, faults 1/2/4/5), PCA-Triage achieves **F1 = 0.962 ± 0.002** at 50% bandwidth — matching or exceeding full-data performance (F1 = 0.961). This occurs because PCA-based triage acts as implicit denoising, reducing sampling on uninformative channels while preserving signal from fault-relevant sensors.

**Evidence:** Pareto curves (Figure 2), Table 2, method comparison (Figure 5).

### Claim 2: PCA-Triage adapts to changing fault conditions within 1-3 windows
When a fault onset occurs (e.g., A/C feed ratio disturbance), PCA-Triage detects the shift in channel importance within 0-3 windows depending on forgetting factor λ. For Fault 1, the A Feed Flow valve (xmv_3) jumps from 9% to 38% sampling rate after onset — correctly identifying the fault-relevant channel.

**Evidence:** Adaptivity heatmaps (Figure 3), reaction time analysis (Figure 7), narrative for 4 fault types.

### Claim 3: PCA-Triage is computationally viable for edge deployment
PCA-Triage completes each triage decision in **1.46 ms** on a single CPU core (MacBook), well under the 5ms edge target. It scales to 500 channels at < 4ms. Memory footprint is 8.5 MB with only 520 computed (not trained) parameters — orders of magnitude lighter than attention-based alternatives from the literature (DCFF-MTAD: 51ms, 325K params; m-AFS: 10-173s).

**Evidence:** Compute profiling (Figure 4), scalability plot (Figure 6).

---

## All Figures and Tables

| # | Type | Description | Status |
|---|------|-------------|--------|
| 1 | Figure | System architecture diagram | Done |
| 2 | Figure | Pareto curves (TEP + SKAB) | Done |
| 3 | Figure | Adaptivity heatmap (3 fault types) | Done |
| 4 | Figure | Compute cost bar chart | Done |
| 5 | Figure | Method comparison at 50% | Done |
| 6 | Figure | Scalability plot | Done |
| 7 | Figure | Reaction time vs forgetting factor | Done |
| 8 | Figure | Ablation 2×2 grid | Done |
| 9 | Figure | Channel importance heatmap (Fault 1) | Done |
| 10 | Figure | Allocated rates heatmap (Fault 1) | Done |
| T1 | Table | Positioning matrix (Related Work) | Done |
| T2 | Table | Results at 50% bandwidth | Done |
| T3 | Table | Ablation studies (k, w, λ) | Done |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| PCA-Triage F1 at 50% BW (TEP) | 0.962 ± 0.002 |
| Full-data F1 (TEP) | 0.961 ± 0.002 |
| Best baseline at 50% (Threshold) | 0.957 ± 0.003 |
| PCA-Triage advantage over Uniform | +4.0% F1 at 50% |
| PCA-Triage advantage at 10% BW | +53% relative improvement over Uniform |
| Time per decision | 1.46 ms |
| Peak memory | 8.5 MB |
| Parameters | 520 (computed) |
| Reaction time (λ=0.85) | 0-3 windows |
| Channels in TEP | 52 |
| Robust k range | 3-10 |
| Robust w range | 50-500 |

---

## Recommended Default Hyperparameters

| Parameter | Default | Justification |
|-----------|---------|---------------|
| k (components) | 10 | Captures ~92% variance; F1 robust for k=3-10 |
| w (window size) | 50-100 | Stable across range; w=50 for faster adaptation |
| λ (forgetting) | 1.0 for static, 0.85 for dynamic | λ=1.0 best for F1; λ=0.85 for fastest fault reaction |
| r_min (min rate) | 0.05 | Ensures no channel fully silenced |

---

## Honest Limitations

1. **SKAB performance:** On SKAB (8 sensors), PCA-Triage does not dominate — with few channels, correlation structure is limited and simpler methods perform comparably.

2. **λ trade-off:** λ=1.0 gives best F1 but slowest fault reaction (19 windows). λ=0.85 reacts within 0-3 windows but ~2% lower F1. Users must choose based on whether their application prioritizes accuracy or adaptivity.

3. **NASA Bearing:** RAR extraction issues prevented full NASA Bearing experiments. Will address in final paper with properly extracted data.

4. **Simplified attention baseline:** Our attention baseline is single-head with random weights — real trained attention models may perform better but are much heavier computationally.
