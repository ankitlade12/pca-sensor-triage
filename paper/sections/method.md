# 3. Method

We present PCA-Triage, a streaming algorithm that converts incremental PCA loadings into per-channel sampling rate allocations under a bandwidth constraint. We first formalize the sensor triage problem, then describe each algorithmic component, and conclude with complexity analysis.

## 3.1 Problem Formulation

Consider a system with $d$ sensor channels producing observations $\mathbf{x}_t \in \mathbb{R}^d$ at each time step $t$. A bandwidth budget $B \in (0, 1]$ constrains the total data volume to a fraction $B$ of the uniform full-rate volume. The goal is to assign each channel $j$ a sampling rate $r_j \in [r_{\min}, 1]$ such that:

$$\frac{1}{d} \sum_{j=1}^{d} r_j \leq B$$

while maximizing the downstream task performance (e.g., fault detection F1 score) when a classifier is trained and evaluated on the triaged data. The rates $\{r_j\}$ must be updated online as new data arrives, without access to future observations or reprocessing of historical data.

**Notation.** Let $\mathbf{X}_w \in \mathbb{R}^{w \times d}$ denote a sliding window of $w$ consecutive observations. Let $\mathbf{V} \in \mathbb{R}^{k \times d}$ be the matrix of the top $k$ principal component loadings and $\boldsymbol{\sigma} \in \mathbb{R}^k$ the corresponding singular values, both obtained from an incremental PCA model updated on $\mathbf{X}_w$.

## 3.2 Channel Importance Scoring

The core insight of PCA-Triage is that the weighted PCA loadings naturally encode per-channel importance. We define the importance score for channel $j$ as:

$$s_j = \sum_{i=1}^{k} \sigma_i \cdot V_{ij}^2$$

This score captures how much channel $j$ contributes to the top-$k$ principal components, weighted by the variance each component explains. A channel that loads heavily on high-variance components receives a high score; a channel orthogonal to the principal subspace receives a low score.

**Exponential smoothing.** To avoid rapid oscillations in allocation, we smooth importance scores across windows using an exponential moving average with forgetting factor $\lambda \in (0, 1]$:

$$\bar{s}_j^{(t)} = \lambda \cdot \bar{s}_j^{(t-1)} + (1 - \lambda) \cdot s_j^{(t)}$$

The forgetting factor controls the trade-off between responsiveness and stability. Lower $\lambda$ values make the algorithm react faster to changes (e.g., fault onset) at the cost of noisier allocations; higher values provide smoother allocation but slower adaptation.

## 3.3 Rate Allocation

Given smoothed importance scores $\bar{\mathbf{s}}$ and budget $B$, we allocate sampling rates in two steps:

1. **Floor allocation.** Assign every channel the minimum rate $r_{\min}$, consuming $r_{\min} \cdot d$ of the total budget $B \cdot d$.

2. **Proportional allocation.** Distribute the remaining budget $(B - r_{\min}) \cdot d$ proportionally to normalized importance scores:

$$r_j = r_{\min} + \frac{\bar{s}_j}{\sum_{j'} \bar{s}_{j'}} \cdot (B - r_{\min}) \cdot d$$

3. **Clipping.** Clip rates to $[r_{\min}, 1]$ and redistribute any excess from clipped channels to remaining channels proportionally.

The minimum rate $r_{\min}$ ensures that no channel is completely silenced—every sensor retains at least a baseline sampling frequency. This is important because a previously quiescent channel may become informative when conditions change; without any samples, the PCA model cannot detect this change.

## 3.4 Data Acquisition and Reconstruction

**Acquisition.** Each sample from channel $j$ is independently retained with probability $r_j$. Dropped samples are marked as missing (NaN). This stochastic acquisition model is practical for IoT deployments where sensors can be instructed to report at variable rates.

**Reconstruction.** Before passing triaged data to a downstream model, missing values are filled via zero-order hold (forward fill): each NaN is replaced with the most recent observed value for that channel. We also evaluate linear interpolation as an alternative. The choice of reconstruction method affects accuracy at low bandwidth levels but has minimal impact above 30% budget in our experiments.

## 3.5 Algorithm Summary

**Algorithm 1: PCA-Triage**

```
Input: Stream of sensor windows X_w, budget B, components k,
       forgetting factor λ, minimum rate r_min

Initialize: IncrementalPCA model with k components
             s̄ ← None  (smoothed importance scores)

For each window X_w:
  1. UPDATE:  partial_fit IncrementalPCA on X_w
              → obtain V (k×d loadings), σ (k singular values)

  2. SCORE:   For each channel j = 1,...,d:
                s_j ← Σ_i σ_i · V[i,j]²

  3. SMOOTH:  If s̄ is None: s̄ ← normalize(s)
              Else: s̄ ← λ · s̄ + (1-λ) · normalize(s)
              s̄ ← s̄ / Σ s̄_j  (re-normalize)

  4. ALLOCATE: r_j ← r_min + (s̄_j / Σ s̄) · (B - r_min) · d
               Clip r_j to [r_min, 1.0]

  5. ACQUIRE:  For each sample x_t in window:
                 For each channel j:
                   Keep x_t[j] with probability r_j
                   Otherwise mark as NaN

  6. RECONSTRUCT: Forward-fill NaN values

  Output: Reconstructed window, rates r, importance scores s̄
```

## 3.6 Comparison with Baselines

We compare PCA-Triage against five baselines spanning the design space:

| Method | Allocation Strategy | Adaptive? | Supervised? | Complexity |
|--------|-------------------|-----------|-------------|------------|
| **Uniform** | Same rate for all channels | No | No | O(1) |
| **Threshold** | Binary: active channels above change threshold get high rate | Yes | No | O(wd) |
| **Variance** | Proportional to rolling variance | Yes | No | O(wd) |
| **Random Dropout** | Randomly keep B fraction of channels | No | No | O(d) |
| **Mutual Info** | Proportional to MI with fault labels | No (batch) | **Yes** | O(nd) |
| **PCA-Triage** | Proportional to weighted PCA loadings | Yes | No | O(wdk) |

Key distinctions: (1) PCA-Triage and Variance are the only methods producing proportional adaptive rates; PCA-Triage accounts for inter-channel correlations while Variance treats channels independently. (2) Mutual Info requires labels, making it inapplicable when fault types are unknown. (3) Threshold makes binary allocation decisions, wasting budget on channels that are merely active but uninformative.

## 3.7 Complexity Analysis

**Time complexity.** Each window update involves: IncrementalPCA partial_fit in $O(wdk)$, importance scoring in $O(dk)$, and rate allocation in $O(d)$. Total: $O(wdk)$ per window.

**Space complexity.** IncrementalPCA maintains a $k \times d$ components matrix and $O(wd)$ buffer for the current window. Smoothed importance scores require $O(d)$. Total: $O(wd + kd)$.

**Comparison with alternatives.** Attention-based methods require $O(d^2)$ per step for self-attention computation plus storage for trainable parameters. For TEP with $d = 52$ and $k = 10$, PCA-Triage's $O(wdk) = O(52{,}000)$ operations per window (w=100) compares favorably with self-attention's $O(d^2) = O(2{,}704)$ per sample, which over a window of 100 samples totals $O(270{,}400)$—5× more expensive per window, before accounting for backpropagation during training.

## 3.8 Hyperparameters

PCA-Triage has four hyperparameters:

- **$k$ (number of components):** Controls how much of the variance structure is captured. We recommend $k$ such that cumulative explained variance exceeds 90%. For TEP, $k = 10$ captures ~92%.

- **$w$ (window size):** Determines the temporal resolution of adaptation. Smaller windows react faster but produce noisier importance estimates. We default to $w = 100$ (5 hours on TEP at 3-minute intervals).

- **$\lambda$ (forgetting factor):** Balances responsiveness vs. stability. We default to $\lambda = 0.95$, which gives 50% weight to the last ~14 windows.

- **$r_{\min}$ (minimum rate):** Prevents complete channel silencing. We default to $r_{\min} = 0.05$ (5% of full rate), ensuring at least 1 sample per 20 time steps.

Section 4.5 presents ablation studies over each hyperparameter.
