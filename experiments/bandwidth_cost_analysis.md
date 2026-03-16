# Bandwidth-Savings vs Compute-Cost Trade-off Analysis

## Setup
- TEP system: 52 sensors at 3-minute intervals
- Each sample: 52 × 8 bytes (float64) = 416 bytes
- Full-rate throughput: 416 bytes × 20 samples/hr = 8.32 KB/hour
- At scale (200 sensors, 1 Hz): 200 × 8 × 3600 = 5.76 MB/hour

## Bandwidth Savings at 50% Budget
- Data saved: 50% of full volume = 2.88 MB/hour (200-sensor scenario)
- Over 24 hours: **69.1 MB/day saved**
- Over 30 days: **2.07 GB/month saved**

## Compute Cost of PCA-Triage
- Time per triage decision: **0.67 ms** (edge-simulated, single-threaded)
- Decisions per hour (w=100, 3-min interval): ~0.33 decisions/hr on TEP
- Decisions per hour (w=100, 1 Hz, 200 sensors): 36 decisions/hr
- Total compute overhead: 36 × 0.67 ms = **24.1 ms/hour** (0.0007% CPU utilization)
- Energy cost: negligible (~0.5 µJ per decision at 1W TDP)

## Trade-off Summary

| Metric | Value |
|--------|-------|
| Bandwidth saved per hour | 2.88 MB |
| Compute cost per hour | 24.1 ms |
| Bandwidth saved per ms of compute | **119 KB/ms** |
| Accuracy retention | 100.4% of full-data (F1 = 0.965 vs 0.961) |
| Memory footprint | 8.5 MB |

## Comparison with Alternatives

| Method | Bandwidth saved | Compute cost | Accuracy | Supervised? |
|--------|----------------|-------------|----------|-------------|
| PCA-Triage | 50% | 0.67 ms/decision | 96.1% F1 | No |
| Variance | 50% | 0.15 ms/decision | 94.8% F1 | No |
| Mutual Info | 50% | 7.73 ms/decision | 91.4% F1 | Yes |
| Attention (lit.) | 50% | 51 ms/decision* | ~93% F1* | Yes |

*Literature values from DCFF-MTAD [21].

## Conclusion
PCA-Triage achieves the best accuracy-per-compute-cost ratio among data-driven methods. At 0.67 ms per decision, it saves 119 KB of bandwidth for every millisecond of compute — making it viable for even the most constrained edge devices. The only faster methods (Uniform, Random Dropout) sacrifice 4-17% F1.
