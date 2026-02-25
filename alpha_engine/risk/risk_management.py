"""
Risk Management — volatility regime detection, drawdown controls, and
risk metric computation.

Provides KMeans-based volatility regime classification, standard risk
metrics (VaR, CVaR, max drawdown), and circuit-breaker logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Volatility Regime Detection                                         #
# ------------------------------------------------------------------ #


def detect_volatility_regime(
    returns: pd.Series,
    n_regimes: int = 3,
    vol_window: int = 60,
) -> pd.Series:
    """Classify each date into a volatility regime via KMeans.

    Features used for clustering:
      - Rolling realised volatility
      - Change in rolling volatility (vol momentum)
      - Rolling vol-of-vol

    Regime labels are ordered by mean volatility:
      0 = low-vol, 1 = mid-vol, 2 = high-vol.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    n_regimes : int
        Number of clusters (default 3).
    vol_window : int
        Lookback for rolling statistics.

    Returns
    -------
    pd.Series
        Integer regime label per date.
    """
    vol = returns.rolling(vol_window).std() * np.sqrt(252)
    vol_chg = vol.diff(5)
    vol_of_vol = vol.rolling(vol_window).std()

    feat_df = pd.DataFrame({
        "vol": vol,
        "vol_chg": vol_chg,
        "vol_of_vol": vol_of_vol,
    }).dropna()

    if len(feat_df) < n_regimes * 10:
        logger.warning("Too few observations for regime detection")
        return pd.Series(0, index=returns.index, name="regime")

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values)

    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Reorder labels by ascending mean vol
    regime_s = pd.Series(labels, index=feat_df.index)
    mean_vols = feat_df["vol"].groupby(regime_s).mean().sort_values()
    label_map = {old: new for new, old in enumerate(mean_vols.index)}
    regime_s = regime_s.map(label_map).astype(int)
    regime_s.name = "regime"

    # Reindex to full series
    regime_full = regime_s.reindex(returns.index)
    regime_full = regime_full.ffill().fillna(0).astype(int)

    logger.info(
        "Regime detection: %s",
        {k: int(v) for k, v in regime_full.value_counts().items()},
    )
    return regime_full


# ------------------------------------------------------------------ #
#  Risk Metrics                                                        #
# ------------------------------------------------------------------ #


def compute_risk_metrics(
    returns: pd.Series,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Compute standard risk metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    confidence : float
        Confidence level for VaR / CVaR.

    Returns
    -------
    dict
        Risk metric → value.
    """
    r = returns.dropna()
    if r.empty:
        return {}

    var = float(np.percentile(r, (1 - confidence) * 100))
    cvar = float(r[r <= var].mean()) if (r <= var).any() else var

    dd = _drawdown_series(r)
    max_dd = float(dd.min())
    dd_duration = _max_drawdown_duration(dd)

    return {
        f"VaR_{confidence:.0%}": round(var, 6),
        f"CVaR_{confidence:.0%}": round(cvar, 6),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_duration_days": dd_duration,
    }


def check_drawdown_breach(
    returns: pd.Series,
    threshold: float = 0.20,
) -> bool:
    """Return True if current drawdown exceeds ``threshold``.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    threshold : float
        Maximum allowable drawdown (positive, e.g. 0.20 = 20%).

    Returns
    -------
    bool
    """
    dd = _drawdown_series(returns)
    if dd.empty:
        return False
    return bool(dd.iloc[-1] < -threshold)


# ------------------------------------------------------------------ #
#  Global Risk-Off Switch                                              #
# ------------------------------------------------------------------ #


def compute_risk_off_signal(
    returns: pd.Series,
    vol_window: int = 60,
    vol_percentile_threshold: int = 90,
    sma_slope_window: int = 200,
) -> pd.Series:
    """Compute a binary risk-off signal.

    Risk-off is triggered when **either** condition is true:
      1. Rolling realised vol exceeds its own historical 90th percentile.
      2. The slope of the ``sma_slope_window``-day SMA of the equity
         curve is negative (i.e. the trend is down).

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.
    vol_window : int
        Lookback for rolling volatility (default 60).
    vol_percentile_threshold : int
        Percentile threshold for vol spike detection (default 90).
    sma_slope_window : int
        Window for the trend filter SMA (default 200).

    Returns
    -------
    pd.Series[bool]
        True = risk-off (go to cash), False = risk-on (normal trading).
    """
    # Condition 1: Vol spike
    vol = returns.rolling(vol_window).std() * np.sqrt(252)
    expanding_pctl = vol.expanding(min_periods=vol_window).apply(
        lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False,
    )
    vol_spike = expanding_pctl > (vol_percentile_threshold / 100.0)

    # Condition 2: Equity-curve trend filter
    equity = (1 + returns).cumprod()
    sma = equity.rolling(sma_slope_window, min_periods=sma_slope_window // 2).mean()
    sma_slope = sma.diff(5)  # 5-day change in SMA as proxy for slope
    trend_down = sma_slope < 0

    risk_off = (vol_spike | trend_down).fillna(False)
    risk_off.name = "risk_off"

    n_off = int(risk_off.sum())
    pct_off = n_off / max(len(risk_off), 1) * 100
    logger.info(
        "Risk-off signal: %d / %d days (%.1f%%) flagged",
        n_off, len(risk_off), pct_off,
    )
    return risk_off


def apply_regime_filter(
    returns: pd.Series,
    risk_off: pd.Series,
) -> pd.Series:
    """Zero-out returns during risk-off periods (shift to cash).

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.
    risk_off : pd.Series[bool]
        True = risk-off day.

    Returns
    -------
    pd.Series
        Filtered returns (zero on risk-off days).
    """
    aligned = risk_off.reindex(returns.index, fill_value=False)
    filtered = returns.where(~aligned, 0.0)
    logger.info(
        "Regime filter applied: zeroed %d of %d days",
        int(aligned.sum()), len(returns),
    )
    return filtered


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the drawdown time-series from returns."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def _max_drawdown_duration(dd: pd.Series) -> int:
    """Length (in trading days) of the longest drawdown."""
    is_dd = dd < 0
    if not is_dd.any():
        return 0

    # Run-length encoding
    groups = (~is_dd).cumsum()
    durations = is_dd.groupby(groups).sum()
    return int(durations.max())
