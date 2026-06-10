"""
Model architectures for Jane Street forecasting.

Usage:
    from models import GRUModel, build_model
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# ── GRU Model ────────────────────────────────────────────────────────
class GRUModel(nn.Module):
    """
    Seq2Seq GRU for time series forecasting.

    Architecture:
        Input:  (batch, seq=968, input_size)  — one day per symbol
        GRU:    temporal dynamics across time_ids
        Head:   per-timestep prediction
        Output: (batch, seq)

    Why GRU over LightGBM:
        LightGBM treats each row independently.
        GRU captures temporal dynamics within a day:
        how a symbol evolves across 968 time steps.

    Why GRU over Transformer (for this problem):
        Transformer adds value when cross-sectional
        dependencies are strong (e.g. Optiver: 200 stocks).
        Jane Street: 39 symbols, weaker cross-sectional signal.
        GRU alone outperformed GRU+Transformer in experiments.

    Results:
        LightGBM best: utility 1.1277
        GRU (30 epoch): utility 1.1847
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # Dropout only between layers, not after last
            dropout=dropout if num_layers > 1 else 0
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, input_size)
        Returns:
            (batch, seq) — prediction for each timestep
        """
        out, _ = self.gru(x)       # (batch, seq, hidden)
        out    = self.head(out)     # (batch, seq, 1)
        return out.squeeze(-1)      # (batch, seq)


# ── Loss functions ────────────────────────────────────────────────────
def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Weighted MSE loss with padding mask.

    Only computes loss on real (non-padded) timesteps.

    Args:
        pred:   (batch, seq)
        target: (batch, seq)
        weight: (batch, seq) — sample weights
        mask:   (batch, seq) bool — True for real timesteps
    """
    pred   = pred[mask]
    target = target[mask]
    weight = weight[mask]
    return (weight * (pred - target) ** 2).mean()


# ── Training utilities ────────────────────────────────────────────────
class EarlyStopping:
    """
    Stop training when validation metric stops improving.

    Args:
        patience: epochs to wait before stopping
        min_delta: minimum improvement to count
        mode: 'max' for utility (higher=better), 'min' for loss
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'max'):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = -np.inf if mode == 'max' else np.inf
        self.counter   = 0
        self.best_state: Optional[dict] = None

    def step(self, metric: float, model: nn.Module) -> bool:
        """
        Returns True if training should stop.
        Saves best model state automatically.
        """
        improved = (
            metric > self.best + self.min_delta
            if self.mode == 'max'
            else metric < self.best - self.min_delta
        )

        if improved:
            self.best       = metric
            self.counter    = 0
            self.best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1

        return self.counter >= self.patience

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load best saved weights into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


def build_model(
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    device: Optional[torch.device] = None
) -> GRUModel:
    """
    Build and initialize GRU model.

    Args:
        input_size: number of input features
        hidden_size: GRU hidden dimension
        num_layers: number of GRU layers
        dropout: dropout rate
        device: torch device (auto-detected if None)

    Returns:
        Initialized GRUModel on specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GRUModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"GRU model: {n_params:,} parameters | device: {device}")

    return model


# ── Online learning ───────────────────────────────────────────────────
def online_update(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    mask: torch.Tensor,
    lr: float = 3e-4,
    device: Optional[torch.device] = None
) -> float:
    """
    Single gradient step for online learning.

    Called after each day's true labels become available (via lags.parquet).
    Uses lower learning rate than training to avoid forgetting.

    Args:
        model: current model
        x, y, w, mask: today's data tensors
        lr: online learning rate (should be lower than training lr)

    Returns:
        loss value for logging

    Notes:
        Evgeniia's approach: one forward+backward pass per day
        lr=0.0003 (vs training lr=0.0005)
        This was the biggest single contributor: +0.008 CV utility

        hydantess's approach: full retrain every 12 days
        More expensive but potentially larger gains
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x, y, w, mask = (t.to(device) for t in [x, y, w, mask])

    optimizer.zero_grad()
    pred = model(x)
    loss = weighted_mse(pred, y, w, mask)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()
