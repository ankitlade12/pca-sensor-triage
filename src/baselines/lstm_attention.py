"""
Baseline 8: LSTM-Attention Channel Importance.

Uses a lightweight LSTM with channel-wise attention to compute importance
scores. The attention weights serve as channel importance for bandwidth
allocation — a learned, data-driven alternative to PCA-Triage.

This represents the "deep learning" approach to channel selection:
more expressive but requires training, more compute, and has trainable params.
"""

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.triage.rate_allocator import RateAllocator


class ChannelAttentionLSTM(nn.Module):
    """Lightweight LSTM with channel attention for importance scoring."""

    def __init__(self, n_channels, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden_size, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, n_channels),
            nn.Softmax(dim=-1),
        )
        self.reconstruct = nn.Linear(hidden_size, n_channels)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        # Use last hidden state for attention
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        attn_weights = self.attention(last_hidden)  # (batch, channels)
        recon = self.reconstruct(lstm_out)  # (batch, seq_len, channels)
        return attn_weights, recon


class LSTMAttentionSampling:
    """LSTM-based adaptive sampling using learned channel attention.

    Parameters
    ----------
    budget : float
        Bandwidth budget.
    window_size : int
        Window size for processing.
    hidden_size : int
        LSTM hidden dimension.
    n_epochs : int
        Training epochs per window.
    lr : float
        Learning rate.
    min_rate : float
        Minimum sampling rate.
    """

    def __init__(
        self,
        budget: float = 0.5,
        window_size: int = 50,
        hidden_size: int = 32,
        n_epochs: int = 5,
        lr: float = 0.001,
        min_rate: float = 0.05,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for LSTMAttentionSampling")

        self.budget = budget
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.min_rate = min_rate
        self.allocator = RateAllocator(budget=budget, min_rate=min_rate)
        self._model = None
        self._optimizer = None
        self._n_params = 0

    def _init_model(self, n_channels):
        self._model = ChannelAttentionLSTM(n_channels, self.hidden_size)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._n_params = sum(p.numel() for p in self._model.parameters())

    def _get_importance(self, window):
        """Train on window and extract attention weights as importance."""
        n, d = window.shape
        if self._model is None:
            self._init_model(d)

        # Reshape to (1, seq_len, channels) for LSTM
        x = torch.FloatTensor(window).unsqueeze(0)

        self._model.train()
        for _ in range(self.n_epochs):
            self._optimizer.zero_grad()
            attn, recon = self._model(x)
            # Reconstruction loss
            loss = nn.functional.mse_loss(recon, x)
            loss.backward()
            self._optimizer.step()

        # Extract attention weights as importance
        self._model.eval()
        with torch.no_grad():
            attn, _ = self._model(x)
            importance = attn.squeeze().numpy()

        # Normalize
        importance = np.maximum(importance, 0)
        imp_sum = importance.sum()
        if imp_sum > 0:
            importance = importance / imp_sum
        else:
            importance = np.ones(d) / d

        return importance

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        torch.manual_seed(seed)
        n, d = data.shape
        n_windows = n // self.window_size
        reconstructed = np.zeros_like(data, dtype=float)

        for w_idx in range(n_windows):
            start = w_idx * self.window_size
            end = start + self.window_size
            window = data[start:end]

            importance = self._get_importance(window)
            rates = self.allocator.allocate(importance)
            triaged = self.allocator.apply_rates(window, rates, seed=seed + w_idx)
            recon = pd.DataFrame(triaged).ffill().bfill().fillna(0.0).values
            reconstructed[start:end] = recon

        # Handle tail
        remaining = n % self.window_size
        if remaining > 0:
            start = n_windows * self.window_size
            rates_last = np.full(d, self.budget)
            triaged = self.allocator.apply_rates(data[start:], rates_last, seed=seed + n_windows)
            reconstructed[start:] = pd.DataFrame(triaged).ffill().bfill().fillna(0.0).values

        return reconstructed

    @property
    def n_params(self):
        return self._n_params


class TransformerAttentionSampling:
    """Transformer-based adaptive sampling using self-attention over channels.

    Uses a single Transformer encoder layer where channels are tokens.
    The attention matrix reveals channel-to-channel dependencies;
    importance = mean attention received per channel.

    Parameters
    ----------
    budget : float
        Bandwidth budget.
    window_size : int
        Window size for processing.
    d_model : int
        Transformer model dimension.
    n_heads : int
        Number of attention heads.
    n_epochs : int
        Training epochs per window.
    lr : float
        Learning rate.
    min_rate : float
        Minimum sampling rate.
    """

    def __init__(
        self,
        budget: float = 0.5,
        window_size: int = 50,
        d_model: int = 32,
        n_heads: int = 4,
        n_epochs: int = 5,
        lr: float = 0.001,
        min_rate: float = 0.05,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for TransformerAttentionSampling")

        self.budget = budget
        self.window_size = window_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_epochs = n_epochs
        self.lr = lr
        self.min_rate = min_rate
        self.allocator = RateAllocator(budget=budget, min_rate=min_rate)
        self._model = None
        self._optimizer = None
        self._n_params = 0
        self._attn_weights = None

    def _init_model(self, n_channels):
        self._embed = nn.Linear(self.window_size, self.d_model)
        self._encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads, batch_first=True,
            dim_feedforward=self.d_model * 2,
        )
        self._decoder = nn.Linear(self.d_model, self.window_size)

        self._params = list(self._embed.parameters()) + \
                       list(self._encoder_layer.parameters()) + \
                       list(self._decoder.parameters())
        self._optimizer = torch.optim.Adam(self._params, lr=self.lr)
        self._n_params = sum(p.numel() for p in self._params)

    def _get_importance(self, window):
        """Treat channels as tokens, extract attention as importance."""
        w, d = window.shape
        if self._model is None:
            self._init_model(d)
            self._model = True  # flag

        # Transpose: (1, d, w) — each channel is a token with w features
        x = torch.FloatTensor(window.T).unsqueeze(0)  # (1, d, w)

        # Train reconstruction
        for _ in range(self.n_epochs):
            self._optimizer.zero_grad()
            embedded = self._embed(x)  # (1, d, d_model)
            encoded = self._encoder_layer(embedded)  # (1, d, d_model)
            recon = self._decoder(encoded)  # (1, d, w)
            loss = nn.functional.mse_loss(recon, x)
            loss.backward()
            self._optimizer.step()

        # Extract attention weights
        with torch.no_grad():
            embedded = self._embed(x)
            # Manually compute attention from the multihead attention
            mha = self._encoder_layer.self_attn
            attn_out, attn_weights = mha(embedded, embedded, embedded,
                                          need_weights=True)
            # attn_weights: (1, d, d) — how much each channel attends to others
            # Importance = mean attention received (column sum)
            importance = attn_weights.squeeze().mean(dim=0).numpy()  # (d,)

        importance = np.maximum(importance, 0)
        imp_sum = importance.sum()
        if imp_sum > 0:
            importance = importance / imp_sum
        else:
            importance = np.ones(d) / d

        return importance

    def process_stream(self, data: np.ndarray, seed: int = 42) -> np.ndarray:
        torch.manual_seed(seed)
        n, d = data.shape
        n_windows = n // self.window_size
        reconstructed = np.zeros_like(data, dtype=float)

        for w_idx in range(n_windows):
            start = w_idx * self.window_size
            end = start + self.window_size
            window = data[start:end]

            importance = self._get_importance(window)
            rates = self.allocator.allocate(importance)
            triaged = self.allocator.apply_rates(window, rates, seed=seed + w_idx)
            recon = pd.DataFrame(triaged).ffill().bfill().fillna(0.0).values
            reconstructed[start:end] = recon

        remaining = n % self.window_size
        if remaining > 0:
            start = n_windows * self.window_size
            rates_last = np.full(d, self.budget)
            triaged = self.allocator.apply_rates(data[start:], rates_last, seed=seed + n_windows)
            reconstructed[start:] = pd.DataFrame(triaged).ffill().bfill().fillna(0.0).values

        return reconstructed

    @property
    def n_params(self):
        return self._n_params
