"""
Feature engineering pipeline for Jane Street forecasting.

Usage:
    from features import load_and_feature, FEATURE_COLS, ALL_FEATS
"""

import polars as pl
import numpy as np
from typing import List, Optional

# ── Constants ────────────────────────────────────────────────────────
FEATURE_COLS = [f'feature_{i:02d}' for i in range(79)]
RESP_COLS    = [f'responder_{i}' for i in range(9)]
TARGET       = 'responder_6'
META_COLS    = ['date_id', 'time_id', 'symbol_id', 'weight']

# Features with structural NaN (missing before date_id ~510)
# These carry information: NaN = "old data regime"
HIGH_NAN_FEATS = ['feature_21', 'feature_26', 'feature_27', 'feature_31']

# Top features by LightGBM importance (from experiments)
TOP_FEATS = [
    'feature_01', 'feature_08', 'feature_60', 'feature_03',
    'feature_07', 'feature_61', 'feature_05', 'feature_00'
]


# ── Core loader ──────────────────────────────────────────────────────
def load_raw(
    path: str,
    date_min: int,
    date_max: int,
    include_responders: bool = False
) -> pl.DataFrame:
    """
    Load raw parquet data for a date range.

    Args:
        path: path to train.parquet
        date_min: start date_id (inclusive)
        date_max: end date_id (inclusive)
        include_responders: whether to load all responder columns

    Returns:
        Polars DataFrame sorted by [symbol_id, date_id, time_id]

    Notes:
        date_id=700 is the recommended start — before this,
        feature_21/26/27/31 are 100% NaN (data source changed ~date_id 510).
    """
    cols = FEATURE_COLS + META_COLS + [TARGET]
    if include_responders:
        cols += [r for r in RESP_COLS if r != TARGET]

    df = (
        pl.scan_parquet(path)
        .filter(pl.col('date_id').is_between(date_min, date_max))
        .select(cols)
        .collect()
    )
    return df.sort(['symbol_id', 'date_id', 'time_id'])


# ── NaN flag feature ─────────────────────────────────────────────────
def add_nan_flags(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add binary flag for structurally missing features.

    Why: feature_21/26/27/31 are NaN before date_id~510.
    The NaN pattern itself carries structural information
    (old vs new data regime), so flagging it helps the model.

    Contribution in experiments: +0.059 utility (5.9% improvement)
    """
    for f in HIGH_NAN_FEATS:
        df = df.with_columns(
            pl.col(f).is_null().cast(pl.Float32).alias(f'{f}_is_null')
        )
    return df


# ── Symbol-based rolling features ────────────────────────────────────
def add_rolling_features(
    df: pl.DataFrame,
    feats: Optional[List[str]] = None,
    windows: Optional[List[int]] = None,
    add_lags: bool = True
) -> pl.DataFrame:
    """
    Add symbol-level rolling statistics (mean, std) and lag features.

    Uses shift(1) before rolling to prevent look-ahead bias:
    at time T, only uses data from T-1 and earlier.

    Args:
        df: input DataFrame (must be sorted by [symbol_id, date_id, time_id])
        feats: feature columns to compute rolling stats for
               (defaults to TOP_FEATS)
        windows: rolling window sizes (defaults to [100])
        add_lags: whether to add lag1 and lag5 features

    Notes:
        window=100 outperformed window=500 and window=10 in experiments.
        Short windows (10) overlap with rmean100 and add noise.
        Long windows (500) lose recent signal.

    Contribution: +0.038 utility over NaN-flag baseline
    """
    if feats is None:
        feats = TOP_FEATS
    if windows is None:
        windows = [100]

    for f in feats:
        for w in windows:
            df = df.with_columns(
                pl.col(f)
                  .shift(1)
                  .rolling_mean(window_size=w, min_samples=max(2, w // 10))
                  .over('symbol_id')
                  .alias(f'{f}_rmean{w}'),

                pl.col(f)
                  .shift(1)
                  .rolling_std(window_size=w, min_samples=max(2, w // 10))
                  .over('symbol_id')
                  .alias(f'{f}_rstd{w}'),
            )

        if add_lags:
            df = df.with_columns(
                pl.col(f).shift(1).over('symbol_id').alias(f'{f}_lag1'),
                pl.col(f).shift(5).over('symbol_id').alias(f'{f}_lag5'),
            )

    return df


# ── Daily lagged responders ──────────────────────────────────────────
def add_lagged_responders(
    df: pl.DataFrame,
    resp_cols: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Add previous day's responder values as features.

    IMPORTANT: Uses daily average then shift(1) — NOT time_id-level shift.

    Why daily and not time_id-level:
        In inference, lags.parquet provides yesterday's responder values.
        Time_id-level shift(1) causes leakage: at time_id=500,
        it uses time_id=499's responder which is not available yet.

    Leakage example caught in experiments:
        Time_id shift → utility 3.69, R² 0.87 (impossible values)
        Daily shift   → utility 1.10, R² 0.016 (realistic)

    Args:
        resp_cols: responder columns to lag
                   (defaults to high-correlation ones: resp_3, 6, 7, 8)
    """
    if resp_cols is None:
        # High correlation with responder_6:
        # resp_3: 0.727, resp_8: 0.447, resp_7: 0.432
        resp_cols = ['responder_3', 'responder_6', 'responder_7', 'responder_8']

    # Step 1: compute daily average per symbol
    daily_avg = (
        df.group_by(['symbol_id', 'date_id'])
        .agg([pl.col(r).mean().alias(f'{r}_daily') for r in resp_cols])
        .sort(['symbol_id', 'date_id'])
    )

    # Step 2: shift by 1 day (yesterday's value)
    daily_avg = daily_avg.with_columns([
        pl.col(f'{r}_daily')
          .shift(1)
          .over('symbol_id')
          .alias(f'{r}_prev_day')
        for r in resp_cols
    ]).drop([f'{r}_daily' for r in resp_cols])

    # Step 3: join back to main df
    df = df.join(daily_avg, on=['symbol_id', 'date_id'], how='left')

    return df


# ── Full pipeline ─────────────────────────────────────────────────────
def load_and_feature(
    path: str,
    date_min: int,
    date_max: int,
    version: str = 'B'
) -> pl.DataFrame:
    """
    Full feature engineering pipeline.

    Versions:
        'baseline': NaN flags only
        'A':        NaN flags + rolling mean/std (w=500, top10 feats)
        'B':        NaN flags + rolling mean (w=100) + lag1 + lag5  ← recommended
        'full':     B + daily lagged responders

    Args:
        path: path to train.parquet
        date_min: start date_id
        date_max: end date_id
        version: feature set version

    Returns:
        DataFrame with engineered features, filled nulls = 0

    Recommended usage:
        train_df = load_and_feature(path, 700,  1359, version='B')
        val_df   = load_and_feature(path, 1370, 1529, version='B')
    """
    df = load_raw(path, date_min, date_max,
                  include_responders=(version == 'full'))
    df = add_nan_flags(df)
    df = df.fill_null(0)

    if version == 'baseline':
        pass

    elif version == 'A':
        df = add_rolling_features(df, feats=TOP_FEATS, windows=[500], add_lags=False)

    elif version == 'B':
        df = add_rolling_features(df, feats=TOP_FEATS, windows=[100], add_lags=True)

    elif version == 'full':
        df = add_rolling_features(df, feats=TOP_FEATS, windows=[100], add_lags=True)
        df = add_lagged_responders(df)
        # Drop raw responders (keep only lagged versions)
        drop = [r for r in RESP_COLS if r != TARGET]
        df   = df.drop([c for c in drop if c in df.columns])

    return df.fill_null(0)


def get_feature_cols(df: pl.DataFrame) -> List[str]:
    """Return model input feature columns (excludes meta and target)."""
    exclude = set(META_COLS + [TARGET] + RESP_COLS)
    return [c for c in df.columns if c not in exclude]


# ── Utility ───────────────────────────────────────────────────────────
def check_leakage(df: pl.DataFrame, feat_col: str) -> dict:
    """
    Basic leakage check: compare feature stats across date_id boundaries.
    Sudden jumps at date boundaries suggest leakage.
    """
    daily = (
        df.group_by('date_id')
        .agg(pl.col(feat_col).mean().alias('mean'))
        .sort('date_id')
    )
    diffs = daily['mean'].diff().abs()
    return {
        'max_jump': diffs.max(),
        'mean_jump': diffs.mean(),
        'suspicious': diffs.max() > diffs.mean() * 10
    }
