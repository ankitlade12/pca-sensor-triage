# PCA-Triage Design Decisions

Engineering notebook documenting key design choices with rationale and empirical evidence.

---

## 1. Why PCA Over Other Dimensionality Reduction Methods?

**Decision:** Use IncrementalPCA (sklearn) as the core importance scoring engine.

**Alternatives considered:**
- **Autoencoders:** Higher capacity but require training, non-deterministic, need GPU. Too heavy for edge.
- **Random Projections:** O(d) but importance scores are random — no data-driven channel ranking.
- **NMF:** Non-negative constraint doesn't apply to standardized sensor data (negative values).
- **ICA:** Captures independent (not principal) components — less aligned with variance-maximizing triage.

**Why PCA wins:**
- Closed-form solution: no training, no hyperparameter tuning for the decomposition itself
- IncrementalPCA supports streaming via `partial_fit()` — O(wdk) per window
- Loading vectors directly encode per-channel importance via the weighted formula
- Zero trainable parameters → deterministic, reproducible
- Proven on TEP: PCA-based monitoring is the standard in process control literature

---

## 2. Why Weighted Loadings Formula?

**Decision:** `s_j = Σᵢ σᵢ · |V[i,j]|²` instead of `Σᵢ |V[i,j]|²` (unweighted)

**Rationale:** Weighting by singular values σᵢ gives more importance to channels that contribute to high-variance principal components. A channel loading heavily on PC1 (high σ) matters more than one loading on PC10 (low σ).

**Empirical evidence (TEP, 50% bandwidth):**
- Weighted: F1 = 0.956 ± 0.001
- Unweighted: F1 = 0.960 ± 0.001
- Recon-based: F1 = 0.951 ± 0.001

Unweighted is slightly better on TEP (difference not significant), but weighted provides clearer interpretability and is the theoretically motivated choice.

---

## 3. Why Forward-Fill Reconstruction?

**Decision:** Use zero-order hold (forward-fill) as default reconstruction.

**Alternatives:**
- **Linear interpolation:** +1% F1 at <20% budget, but requires look-ahead (next observed value)
- **Zero-fill:** Worst performance — introduces artificial zero signals
- **Model-based (GP, spline):** Expensive, overkill for sensor data at >30% budget

**Forward-fill advantages:**
- O(n) per channel, trivially parallelizable
- No look-ahead bias — causal reconstruction
- At >30% budget, indistinguishable from linear interpolation
- Standard in industrial SCADA systems (last-known-good value)

---

## 4. Why λ = 1.0 Default (No Forgetting)?

**Decision:** Default forgetting factor λ = 1.0 for maximum accuracy.

**Trade-off discovered in ablation:**
- λ = 1.0 → F1 = 0.962, reaction time = 19 windows (slow)
- λ = 0.85 → F1 = 0.942, reaction time = 0-3 windows (fast)

For the paper's primary accuracy claims, λ = 1.0 is used. For adaptivity experiments, λ = 0.85 demonstrates fast reaction. Users choose based on deployment priority.

**Why not adaptive λ?** Considered but rejected — adds complexity (another hyperparameter to tune), and the current binary choice (accuracy vs speed) is clearer for practitioners.

---

## 5. Why r_min = 0.05?

**Decision:** Every channel retains at least 5% sampling rate.

**Rationale:** If a channel is fully silenced (r = 0), PCA cannot detect when that channel becomes informative again (e.g., fault onset activating a previously quiescent sensor). The 5% floor ensures at least 1 sample per 20 time steps for any channel, sufficient for PCA to detect importance shifts.

**Cost:** 5% × d channels of "wasted" bandwidth. For TEP (d=52), this reserves 2.6 channels worth of bandwidth — acceptable overhead.

---

## 6. Why k = 10 Default?

**Decision:** Retain 10 principal components by default.

**Empirical justification:**
- TEP: k=10 captures ~92% cumulative variance
- F1 is robust for k ∈ [3, 10] (F1 = 0.962 ± 0.003)
- k > 15 degrades performance (noise components hurt importance scoring)
- k = 10 balances information capture vs noise suppression

**Rule of thumb for new datasets:** Set k to capture ~90% cumulative variance.

---

## 7. Why Budget B is Expressed as Fraction (Not Absolute)?

**Decision:** Budget B ∈ (0, 1] represents fraction of full-rate volume.

**Why not absolute bytes/sec?** Fraction is hardware-independent — the same experiment (B=0.5) is meaningful regardless of sensor sampling frequency, data resolution, or network protocol. This allows direct comparison across datasets with different native data rates.

---

## 8. Why RandomForest as Downstream Classifier?

**Decision:** Use RandomForestClassifier(n=100) for all fault detection experiments.

**Rationale:** The downstream classifier is NOT our contribution — PCA-Triage is a data acquisition strategy. RF is:
- Simple, well-understood, no GPU needed
- Robust to the noise introduced by sub-sampling
- Deterministic given seed
- Fast to train (seconds, not minutes)

Using a more complex downstream model (LSTM, Transformer) would confound results — improvements could come from the model, not the triage.

---

## 9. Failed Experiments & Dead Ends

### 9.1 Adaptive λ via prediction error
**Tried:** Automatically decrease λ when reconstruction error spikes (indicating regime change).
**Result:** Oscillation — error spike → low λ → noisy importance → more error spikes. Abandoned.

### 9.2 Per-channel PCA (independent PCA per channel)
**Tried:** Run separate PCA on each channel's time series (temporal PCA).
**Result:** Loses the inter-channel correlation structure that is PCA-Triage's main advantage. F1 dropped 3%.

### 9.3 Mutual Information as secondary scorer
**Tried:** Blend PCA importance with MI importance when labels are available.
**Result:** MI requires labels (defeats unsupervised advantage). When labels are synthetic (from PCA anomaly detection), it's circular.
