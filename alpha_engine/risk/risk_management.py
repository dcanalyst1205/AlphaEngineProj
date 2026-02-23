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
