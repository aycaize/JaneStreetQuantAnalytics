"""
Evaluation metrics for Jane Street forecasting.

Two metrics tracked throughout:
    1. Utility score  — trading performance (Sharpe-like)
    2. Weighted R²    — official competition metric

Usage:
    from evaluate import utility_score, weighted_r2, evaluate_predictions
"""

import numpy as np
import polars as pl
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# ── Utility Score ────────────────────────────────────────────────────
def utility_score(
    dates: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute utility score (Sharpe-like metric).

    Formula:
        action     = 1 if pred > 0 else 0
        daily_pnl  = sum(weight * target * action) per day
        utility    = mean(daily_pnl) / std(daily_pnl)

    Why utility and not just accuracy:
        The competition rewards consistent positive returns,
        not just correct direction. A model that's right 55%
        of the time but loses big on the 45% is penalized.

    Args:
        dates:       date_id for each row
        weights:     sample weights (larger = more important trade)
        targets:     true responder_6 values
        predictions: model predictions

    Returns:
        (utility, daily_pnl_array)
    """
    actions = (predictions > 0).astype(float)

    daily: Dict[int, float] = defaultdict(float)
    for i in range(len(predictions)):
        daily[dates[i]] += weights[i] * targets[i] * actions[i]

    daily_pnl = np.array(list(daily.values()))
    util = daily_pnl.mean() / (daily_pnl.std() + 1e-8)

    return util, daily_pnl


def utility_score_df(df: pl.DataFrame, predictions: np.ndarray) -> Tuple[float, dict]:
    """
    Compute utility score from a Polars DataFrame.

    Args:
        df: must contain columns [date_id, weight, responder_6]
        predictions: model predictions aligned with df rows

    Returns:
        (utility, stats_dict)
    """
    dates   = df['date_id'].to_numpy()
    weights = df['weight'].to_numpy()
    targets = df['responder_6'].to_numpy()

    util, daily_pnl = utility_score(dates, weights, targets, predictions)

    stats = {
        'utility':    util,
        'mean_pnl':   daily_pnl.mean(),
        'std_pnl':    daily_pnl.std(),
        'pos_days':   (daily_pnl > 0).sum(),
        'total_days': len(daily_pnl),
        'win_rate':   (daily_pnl > 0).mean()
    }

    return util, stats


# ── Weighted R² ──────────────────────────────────────────────────────
def weighted_r2(
    targets: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Weighted zero-mean R² — official competition metric.

    Formula:
        ss_res = sum(weight * (target - pred)²)
        ss_tot = sum(weight * target²)   ← zero-mean: no mean subtracted
        R²     = 1 - ss_res / ss_tot

    "Zero-mean" means the baseline is predicting zero for all,
    not the mean of targets. This makes sense for financial returns
    which are expected to be near zero.

    Args:
        targets:     true responder_6 values
        predictions: model predictions
        weights:     sample weights

    Returns:
        R² value (negative means worse than predicting zero)

    Notes:
        R² for this competition is typically very small (~0.015)
        because the signal-to-noise ratio is very low.
        Utility score is more informative for trading performance.
    """
    ss_res = (weights * (targets - predictions) ** 2).sum()
    ss_tot = (weights * targets ** 2).sum()

    if ss_tot == 0:
        return 0.0

    return float(1 - ss_res / ss_tot)


# ── Leakage detection ─────────────────────────────────────────────────
def detect_leakage(
    targets: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    threshold_r2: float = 0.1,
    threshold_utility: float = 2.0
) -> dict:
    """
    Heuristic check for data leakage.

    Red flags:
        R² > 0.1    → unrealistically high (typical is ~0.015)
        utility > 2 → unrealistically high (typical is ~1.1)
        win_rate > 0.95 → unrealistically high

    Leakage caught in experiments:
        Time_id-level responder lag → R²=0.87, utility=3.69
        (should be: R²=0.016, utility=1.10)

    Returns:
        dict with leakage flags and values
    """
    util, daily_pnl = utility_score(
        np.arange(len(targets)),  # dummy dates
        weights, targets, predictions
    )
    r2       = weighted_r2(targets, predictions, weights)
    win_rate = (daily_pnl > 0).mean()

    flags = {
        'r2':              r2,
        'utility':         util,
        'win_rate':        win_rate,
        'r2_suspicious':   r2 > threshold_r2,
        'util_suspicious': util > threshold_utility,
        'likely_leakage':  r2 > threshold_r2 or util > threshold_utility
    }

    if flags['likely_leakage']:
        print(f"⚠️  LEAKAGE SUSPECTED:")
        print(f"   R²={r2:.4f} (threshold={threshold_r2})")
        print(f"   utility={util:.4f} (threshold={threshold_utility})")
    else:
        print(f"✓ No leakage detected (R²={r2:.4f}, utility={util:.4f})")

    return flags


# ── Full evaluation ───────────────────────────────────────────────────
def evaluate_predictions(
    df: pl.DataFrame,
    predictions: np.ndarray,
    label: str = "",
    check_leakage: bool = True
) -> dict:
    """
    Full evaluation: utility + R² + leakage check.

    Args:
        df:           DataFrame with [date_id, weight, responder_6]
        predictions:  model predictions
        label:        experiment label for printing
        check_leakage: whether to run leakage check

    Returns:
        results dict
    """
    targets = df['responder_6'].to_numpy()
    weights = df['weight'].to_numpy()
    dates   = df['date_id'].to_numpy()

    util, daily_pnl = utility_score(dates, weights, targets, predictions)
    r2              = weighted_r2(targets, predictions, weights)

    results = {
        'label':      label,
        'utility':    util,
        'r2':         r2,
        'pos_days':   (daily_pnl > 0).sum(),
        'total_days': len(daily_pnl),
        'win_rate':   (daily_pnl > 0).mean(),
        'mean_pnl':   daily_pnl.mean(),
        'std_pnl':    daily_pnl.std(),
    }

    header = f"[{label}] " if label else ""
    print(f"\n{header}Results:")
    print(f"  Utility:     {util:.4f}")
    print(f"  Weighted R²: {r2:.6f}")
    print(f"  Pos days:    {results['pos_days']} / {results['total_days']} "
          f"({results['win_rate']:.1%})")

    if check_leakage:
        detect_leakage(targets, predictions, weights)

    return results


# ── Experiment tracker ────────────────────────────────────────────────
class ExperimentTracker:
    """
    Track and compare experiment results.

    Usage:
        tracker = ExperimentTracker()
        tracker.log('baseline', utility=1.02, r2=0.014, notes='79 features')
        tracker.log('nan_flag', utility=1.08, r2=0.015, notes='+NaN flag')
        tracker.summary()
    """

    def __init__(self):
        self.experiments: List[dict] = []

    def log(
        self,
        name: str,
        utility: float,
        r2: float,
        n_features: int = 0,
        early_stop: int = 0,
        notes: str = ""
    ):
        prev_util = self.experiments[-1]['utility'] if self.experiments else utility
        self.experiments.append({
            'name':       name,
            'utility':    utility,
            'r2':         r2,
            'delta':      utility - prev_util,
            'n_features': n_features,
            'early_stop': early_stop,
            'notes':      notes
        })
        print(f"Logged: {name} | utility={utility:.4f} "
              f"(delta={utility-prev_util:+.4f})")

    def summary(self):
        print(f"\n{'='*70}")
        print(f"{'#':<3} {'Name':<30} {'Utility':>8} {'Delta':>8} {'R²':>10}")
        print(f"{'='*70}")
        for i, exp in enumerate(self.experiments):
            marker = " ←best" if exp['utility'] == self.best_utility else ""
            print(f"{i+1:<3} {exp['name']:<30} "
                  f"{exp['utility']:>8.4f} "
                  f"{exp['delta']:>+8.4f} "
                  f"{exp['r2']:>10.6f}"
                  f"{marker}")
        print(f"{'='*70}")
        print(f"Best: {self.best_name} (utility={self.best_utility:.4f})")

    @property
    def best_utility(self) -> float:
        return max(e['utility'] for e in self.experiments) if self.experiments else 0

    @property
    def best_name(self) -> str:
        return max(self.experiments, key=lambda e: e['utility'])['name'] \
               if self.experiments else ""
