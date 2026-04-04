# Experimental Summary
## PCA-Driven Adaptive Sensor Triage for Edge AI Inference

---

## Top 3 Claims (with supporting evidence)

### Claim 1: PCA-Triage maintains fault detection accuracy at 50% bandwidth
On TEP (52 sensors, faults 1/2/4/5), PCA-Triage achieves **F1 = 0.961 ± 0.001** at 50% bandwidth — within 0.1% of full-data performance (F1 = 0.962). With targeted extensions (hybrid scoring, linear interpolation, sharpening), F1 reaches 0.970. Source: `paper/tables/table2_results_50pct.csv`.

**Evidence:** Pareto curves (Figure 2), Table 2, method comparison (Figure 5).

### Claim 2: PCA-Triage adapts to changing fault conditions
When a fault onset occurs, PCA-Triage shifts channel importance — speed controlled by lambda. At lambda<=0.80, reaction occurs within 0-3 windows. At lambda=0.85, reaction takes up to 19 windows at the 20% shift threshold. For Fault 1, xmv_3 jumps from 9% to 38% sampling rate after onset.

**Evidence:** Adaptivity heatmaps (Figure 3), reaction time analysis (Figure 7), narrative for 4 fault types.

### Claim 3: PCA-Triage is computationally viable for edge deployment
PCA-Triage completes each triage decision in **0.67 ms** under the paper's edge-simulated single-thread setting and **1.46 ms** in the laptop profile, both well under the 5 ms edge target. It scales to 500 channels at < 4ms. Memory footprint is 8.5 MB with only 520 computed (not trained) parameters — orders of magnitude lighter than attention-based alternatives from the literature (DCFF-MTAD: 51ms, 325K params; m-AFS: 10-173s).

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
| PCA-Triage F1 at 50% BW (TEP) | 0.961 ± 0.001 |
| Full-data F1 (TEP) | 0.962 |
| Best baseline at 50% (Threshold) | 0.958 |
| PCA-Triage advantage over Uniform | +3.8% F1 at 50% |
| PCA-Triage advantage at 10% BW | +53% relative improvement over Uniform |
| Time per decision | 0.67 ms (edge), 1.46 ms (laptop) |
| Peak memory | 8.5 MB |
| Parameters | 520 (computed) |
| Reaction time (λ=0.85) | up to 19 windows (0-3 at λ<=0.80) |
| Channels in TEP | 52 |
| Robust k range | 3-10 |
| Robust w range | 50-500 |

---

## Recommended Default Hyperparameters

| Parameter | Default | Justification |
|-----------|---------|---------------|
| k (components) | 10 | Captures ~92% variance; F1 robust for k=3-10 |
| w (window size) | 50-100 | Stable across range; w=50 for faster adaptation |
| λ (forgetting) | 1.0 for accuracy, <=0.80 for fast reaction | λ=1.0 best for F1; lower λ reacts faster but is noisier |
| r_min (min rate) | 0.05 | Ensures no channel fully silenced |

---

## Honest Limitations

1. **SKAB performance:** On SKAB (8 sensors), PCA-Triage does not dominate — with few channels, correlation structure is limited and simpler methods perform comparably.

2. **λ trade-off:** λ=1.0 gives best F1 but slowest fault reaction. At the 20% shift threshold, λ=0.85 can still take up to 19 windows, while lower λ (<=0.80) reacts in 0-3 windows but with noisier estimates. Users must choose based on application priority.

3. **Random-projection attention baseline:** The "Attention" baseline in the compute cost table uses fixed random projections, not trained attention. Trained LSTM-Attention and Transformer-Attention are evaluated in Experiment 14 (subsampled data).
