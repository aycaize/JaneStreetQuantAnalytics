"""
Training pipelines for Jane Street forecasting.

Three pipelines:
    1. train_lgbm       — LightGBM with static train/val split
    2. train_gru        — GRU with static train/val split
    3. sliding_window_cv — One-step-ahead sliding window validation

Usage:
    from train import train_lgbm, train_gru, sliding_window_cv
"""

import numpy as np
import polars as pl
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Optional, Tuple
import gc
import copy

from features import load_and_feature, get_feature_cols, FEATURE_COLS
from models import GRUModel, weighted_mse, EarlyStopping, online_update, build_model
from evaluate import evaluate_predictions, utility_score


# ── LightGBM ─────────────────────────────────────────────────────────
def train_lgbm(
    path: str,
    train_range: Tuple[int, int] = (700, 1359),
    val_range: Tuple[int, int]   = (1370, 1529),
    gap: int = 10,
    feature_version: str = 'B',
    params: Optional[dict] = None
) -> Tuple[lgb.LGBMRegressor, dict]:
    """
    Train LightGBM with static train/val split.

    CV strategy:
        Train: partition 4-7 (date_id 700-1359)
        Gap:   10 days (prevents rolling feature leakage)
        Val:   partition 8  (date_id 1370-1529)

    Args:
        path:            path to train.parquet
        train_range:     (start, end) date_id for training
        val_range:       (start, end) date_id for validation
        gap:             days to skip between train end and val start
        feature_version: passed to load_and_feature
        params:          LightGBM parameters (uses defaults if None)

    Returns:
        (trained model, evaluation results dict)
    """
    if params is None:
        params = {
            'n_estimators':    1000,
            'learning_rate':   0.05,
            'num_leaves':      128,
            'min_child_samples': 50,
            'subsample':       0.8,
            'colsample_bytree': 0.8,
            'random_state':    42,
            'n_jobs':          -1,
            'verbose':         -1
        }

    print(f"Loading train ({train_range[0]}-{train_range[1]})...")
    train_df  = load_and_feature(path, *train_range, version=feature_version)
    all_feats = get_feature_cols(train_df)
    print(f"Features: {len(all_feats)} | Rows: {len(train_df):,}")

    X_tr = train_df[all_feats].to_numpy()
    y_tr = train_df['responder_6'].to_numpy()
    w_tr = train_df['weight'].to_numpy()
    del train_df
    gc.collect()

    val_start = val_range[0] + gap
    print(f"Loading val ({val_start}-{val_range[1]})...")
    val_df = load_and_feature(path, val_start, val_range[1], version=feature_version)

    X_val = val_df[all_feats].to_numpy()
    y_val = val_df['responder_6'].to_numpy()
    w_val = val_df['weight'].to_numpy()

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    del X_tr, y_tr, w_tr
    gc.collect()

    preds   = model.predict(X_val)
    results = evaluate_predictions(val_df, preds, label='LightGBM')
    results['feature_cols'] = all_feats

    del X_val, y_val, w_val, val_df
    gc.collect()

    return model, results


# ── GRU Dataset ───────────────────────────────────────────────────────
class JaneStreetDataset(Dataset):
    """
    Dataset for GRU training.

    Each sample = 1 day × 1 symbol, padded to MAX_SEQ=968 timesteps.

    Padding strategy:
        Real timesteps: actual feature values
        Padded timesteps: zeros
        Mask: True for real, False for padded

    Why per-day-per-symbol batching:
        GRU needs to see the temporal sequence within a trading day.
        Different symbols on the same day are independent sequences.
    """

    MAX_SEQ = 968

    def __init__(self, df: pl.DataFrame, feat_cols: List[str]):
        self.samples = []
        target = 'responder_6'

        groups = df.group_by(['date_id', 'symbol_id'], maintain_order=True)

        for (date_id, symbol_id), group in groups:
            group = group.sort('time_id')
            T = group.height

            X    = np.zeros((self.MAX_SEQ, len(feat_cols)), dtype=np.float32)
            y    = np.zeros(self.MAX_SEQ, dtype=np.float32)
            w    = np.zeros(self.MAX_SEQ, dtype=np.float32)
            mask = np.zeros(self.MAX_SEQ, dtype=bool)

            X[:T]    = group[feat_cols].to_numpy().astype(np.float32)
            y[:T]    = group[target].to_numpy().astype(np.float32)
            w[:T]    = group['weight'].to_numpy().astype(np.float32)
            mask[:T] = True

            self.samples.append((date_id, X, y, w, mask))

        print(f"  {len(self.samples):,} samples created")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        date_id, X, y, w, mask = self.samples[idx]
        return (
            date_id,
            torch.tensor(X),
            torch.tensor(y),
            torch.tensor(w),
            torch.tensor(mask)
        )


def _compute_gru_utility(
    preds_list, targets_list, weights_list, dates_list
) -> float:
    """Aggregate per-sample predictions to daily utility score."""
    daily = defaultdict(float)
    for pred, target, weight, date_id in zip(
        preds_list, targets_list, weights_list, dates_list
    ):
        actions = (pred > 0).astype(float)
        daily[date_id] += (weight * target * actions).sum()

    daily_pnl = np.array(list(daily.values()))
    return daily_pnl.mean() / (daily_pnl.std() + 1e-8)


# ── GRU training ─────────────────────────────────────────────────────
def train_gru(
    path: str,
    train_range: Tuple[int, int] = (700, 1359),
    val_range: Tuple[int, int]   = (1370, 1529),
    gap: int = 10,
    feature_version: str = 'B',
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 3e-4,
    n_epochs: int = 30,
    batch_size: int = 4,
    patience: int = 5
) -> Tuple[GRUModel, dict]:
    """
    Train GRU model with static train/val split.

    Architecture choices (from experiments):
        hidden_size=128, num_layers=2 > hidden_size=64, num_layers=1
        Padding to MAX_SEQ=968 for consistent batch sizes
        Weighted MSE loss (only on real timesteps, not padding)
        Gradient clipping (max_norm=1.0)

    Args:
        path:            path to train.parquet
        train_range:     (start, end) date_id
        val_range:       (start, end) date_id
        gap:             days to skip between train and val
        feature_version: feature engineering version
        hidden_size:     GRU hidden dimension
        num_layers:      number of GRU layers
        dropout:         dropout rate
        lr:              learning rate
        n_epochs:        maximum training epochs
        batch_size:      samples per batch
        patience:        early stopping patience

    Returns:
        (best model, training history dict)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"Loading train ({train_range[0]}-{train_range[1]})...")
    train_df  = load_and_feature(path, *train_range, version=feature_version)
    feat_cols = get_feature_cols(train_df)
    print(f"Features: {len(feat_cols)}")

    val_start = val_range[0] + gap
    print(f"Loading val ({val_start}-{val_range[1]})...")
    val_df = load_and_feature(path, val_start, val_range[1], version=feature_version)

    # Build datasets
    print("Building datasets...")
    train_dataset = JaneStreetDataset(train_df, feat_cols)
    val_dataset   = JaneStreetDataset(val_df,   feat_cols)
    del train_df
    gc.collect()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Build model
    model   = build_model(len(feat_cols), hidden_size, num_layers, dropout, device)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=patience, mode='max')

    history = {'train_loss': [], 'val_utility': [], 'best_epoch': 0}

    for epoch in range(n_epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0

        for _, X, y, w, mask in train_loader:
            X, y, w, mask = (t.to(device) for t in [X, y, w, mask])
            optim.zero_grad()
            pred = model(X)
            loss = weighted_mse(pred, y, w, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        all_preds, all_targets, all_weights, all_dates = [], [], [], []

        with torch.no_grad():
            for date_ids, X, y, w, mask in val_loader:
                pred = model(X.to(device)).cpu().numpy()
                for i in range(len(date_ids)):
                    m = mask[i].numpy()
                    all_preds.append(pred[i][m])
                    all_targets.append(y[i][m].numpy())
                    all_weights.append(w[i][m].numpy())
                    all_dates.append(date_ids[i].item())

        util = _compute_gru_utility(all_preds, all_targets, all_weights, all_dates)
        sched.step(-util)

        history['train_loss'].append(train_loss)
        history['val_utility'].append(util)

        marker = ""
        if stopper.step(util, model):
            marker = " ← best"
            history['best_epoch'] = epoch + 1

        print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Utility: {util:.4f}{marker}")

        if stopper.counter >= stopper.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    best_model = stopper.load_best(model)
    print(f"\nBest utility: {stopper.best:.4f} (epoch {history['best_epoch']})")

    return best_model, history


# ── Sliding window CV ─────────────────────────────────────────────────
def sliding_window_cv(
    path: str,
    model: lgb.LGBMRegressor,
    feature_version: str = 'B',
    window_size: int = 660,
    val_start: int = 1360,
    val_end: int = 1529,
    train_start: int = 700,
    retrain_params: Optional[dict] = None
) -> dict:
    """
    One-step-ahead sliding window cross-validation.

    Simulates real inference pipeline:
        For each day in [val_start, val_end]:
            1. Predict using current model (no future data)
            2. Observe true labels
            3. Retrain on sliding window ending at current day
            4. Use updated model for next day

    This is exactly how the Jane Street inference API works:
        - You predict, then lags.parquet provides true values
        - Online learning uses these true values to update model

    Args:
        path:           path to train.parquet
        model:          initial trained model
        feature_version: feature engineering version
        window_size:    number of days in sliding window
        val_start:      first day to predict
        val_end:        last day to predict
        train_start:    minimum date_id (don't go before this)
        retrain_params: LightGBM params for retraining (uses model params if None)

    Returns:
        results dict with daily PnL, utility, R²
    """
    if retrain_params is None:
        retrain_params = {
            'n_estimators':     200,
            'learning_rate':    0.05,
            'num_leaves':       128,
            'min_child_samples': 50,
            'subsample':        0.8,
            'colsample_bytree': 0.8,
            'random_state':     42,
            'n_jobs':           -1,
            'verbose':          -1
        }

    n_days    = val_end - val_start + 1
    all_daily = {}
    scanner   = pl.scan_parquet(path)

    print(f"One-step-ahead CV: {n_days} days | window={window_size}")

    for day in range(val_start, val_end + 1):

        # ── Step 1: load today's features ──
        win_start = max(train_start, day - window_size + 1)
        win_df    = load_and_feature(path, win_start, day, version=feature_version)

        feat_cols = get_feature_cols(win_df)
        today_df  = win_df.filter(pl.col('date_id') == day)

        X_today = today_df[feat_cols].to_numpy()
        y_today = today_df['responder_6'].to_numpy()
        w_today = today_df['weight'].to_numpy()

        # ── Step 2: predict (before seeing today's labels) ──
        preds   = model.predict(X_today)
        actions = (preds > 0).astype(float)
        all_daily[day] = (w_today * y_today * actions).sum()

        # ── Step 3: retrain on sliding window ──
        X_win = win_df[feat_cols].to_numpy()
        y_win = win_df['responder_6'].to_numpy()
        w_win = win_df['weight'].to_numpy()
        del win_df, today_df, X_today
        gc.collect()

        model = lgb.LGBMRegressor(**retrain_params)
        model.fit(X_win, y_win, sample_weight=w_win)
        del X_win, y_win, w_win
        gc.collect()

        # Progress every 10 days
        if (day - val_start + 1) % 10 == 0:
            so_far = np.array(list(all_daily.values()))
            util   = so_far.mean() / (so_far.std() + 1e-8)
            print(f"Day {day} ({day-val_start+1}/{n_days}) | "
                  f"Cumulative utility: {util:.4f} | "
                  f"Today PnL: {all_daily[day]:.1f}")

    # Final results
    daily_arr = np.array([all_daily[d] for d in sorted(all_daily)])
    utility   = daily_arr.mean() / (daily_arr.std() + 1e-8)
    pos_days  = (daily_arr > 0).sum()

    results = {
        'utility':    utility,
        'pos_days':   pos_days,
        'total_days': len(daily_arr),
        'win_rate':   pos_days / len(daily_arr),
        'daily_pnl':  all_daily,
        'model':      model
    }

    print(f"\n{'='*50}")
    print(f"Sliding window CV results:")
    print(f"  Utility:  {utility:.4f}")
    print(f"  Pos days: {pos_days} / {len(daily_arr)} ({pos_days/len(daily_arr):.1%})")

    return results
