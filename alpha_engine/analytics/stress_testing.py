"""
Stress Testing — walk-forward analysis, Monte Carlo simulation,
and beta leakage quantification for portfolio robustness.

Provides:
  - ``run_walk_forward_analysis``: k-stage expanding-window walk-forward
    with OOS Sharpe per stage.
  - ``run_monte_carlo``: Bootstrap resampling of daily returns to produce
    confidence intervals for key metrics.
  - ``compute_beta_leakage``: OLS regression of strategy on benchmark to
    quantify residual market beta.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Walk-Forward Robustness                                             #
# ------------------------------------------------------------------ #


def run_walk_forward_analysis(
    returns: pd.Series,
    n_stages: int = 5,
    oos_fraction: float = 0.20,
) -> pd.DataFrame:
    """Split the return series into k expanding-window stages and
    compute OOS Sharpe for each.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    n_stages : int
        Number of walk-forward stages (default 5).
    oos_fraction : float
        Fraction of each period used as OOS (default 0.20).

    Returns
    -------
    pd.DataFrame
        Columns: stage, oos_start, oos_end, oos_sharpe, oos_return,
        oos_vol, oos_max_dd.
    """
    r = returns.dropna()
    n = len(r)

    if n < n_stages * 50:
        logger.warning("Insufficient data for %d-stage walk-forward", n_stages)
        return pd.DataFrame()

    stage_size = n // n_stages
    results = []

    for i in range(n_stages):
        end_idx = (i + 1) * stage_size if i < n_stages - 1 else n
        oos_len = max(int(stage_size * oos_fraction), 20)
        oos_start_idx = end_idx - oos_len

        oos = r.iloc[oos_start_idx:end_idx]
        if len(oos) < 10:
            continue

        oos_ret = float(oos.mean() * 252)
        oos_vol = float(oos.std() * np.sqrt(252))
        oos_sharpe = oos_ret / oos_vol if oos_vol > 0 else 0.0

        # Max drawdown in OOS window
        cum = (1 + oos).cumprod()
        peak = cum.cummax()
        dd = ((cum - peak) / peak)
        oos_max_dd = float(dd.min())

        results.append({
            "stage": i + 1,
            "oos_start": str(oos.index[0].date()),
            "oos_end": str(oos.index[-1].date()),
            "oos_sharpe": round(oos_sharpe, 4),
            "oos_return": round(oos_ret, 4),
            "oos_vol": round(oos_vol, 4),
            "oos_max_dd": round(oos_max_dd, 4),
        })

    df = pd.DataFrame(results)
    logger.info("Walk-forward analysis (%d stages):\n%s", n_stages, df.to_string())
    return df


# ------------------------------------------------------------------ #
#  Monte Carlo Simulation                                              #
# ------------------------------------------------------------------ #


def run_monte_carlo(
    returns: pd.Series,
    n_simulations: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Bootstrap resampling of daily returns to estimate metric CIs.

    Each simulation draws ``len(returns)`` days *with replacement* from the
    realised return distribution. We compute terminal wealth, CAGR,
    Sharpe, and max drawdown for each path.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    n_simulations : int
        Number of bootstrap paths (default 2000).
    random_state : int
        Numpy random seed.

    Returns
    -------
    dict
        Keys: ``percentile_5``, ``percentile_50``, ``percentile_95``,
        each containing {terminal_wealth, cagr, sharpe, max_drawdown}.
        Also includes ``all_terminal_wealth`` (list) for plotting.
    """
    rng = np.random.RandomState(random_state)
    r = returns.dropna().values
    n = len(r)
    trading_days = 252

    if n < 50:
        logger.warning("Too few observations for Monte Carlo (%d)", n)
        return {}

    terminal_wealths = []
    cagrs = []
    sharpes = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Bootstrap sample (with replacement)
        sampled = rng.choice(r, size=n, replace=True)
        cum = np.cumprod(1 + sampled)
        tw = cum[-1]
        terminal_wealths.append(tw)

        # CAGR
        years = n / trading_days
        cagr = tw ** (1 / max(years, 0.5)) - 1 if tw > 0 else -1.0
        cagrs.append(cagr)

        # Sharpe
        ann_ret = sampled.mean() * trading_days
        ann_vol = sampled.std() * np.sqrt(trading_days)
        sharpes.append(ann_ret / ann_vol if ann_vol > 0 else 0.0)

        # Max drawdown
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_drawdowns.append(dd.min())

    # Compile percentiles
    def _pctl(arr, p):
        return round(float(np.percentile(arr, p)), 6)

    result = {
        "n_simulations": n_simulations,
        "percentile_5": {
            "terminal_wealth": _pctl(terminal_wealths, 5),
            "cagr": _pctl(cagrs, 5),
            "sharpe": _pctl(sharpes, 5),
            "max_drawdown": _pctl(max_drawdowns, 5),
        },
        "percentile_50": {
            "terminal_wealth": _pctl(terminal_wealths, 50),
            "cagr": _pctl(cagrs, 50),
            "sharpe": _pctl(sharpes, 50),
            "max_drawdown": _pctl(max_drawdowns, 50),
        },
        "percentile_95": {
            "terminal_wealth": _pctl(terminal_wealths, 95),
            "cagr": _pctl(cagrs, 95),
            "sharpe": _pctl(sharpes, 95),
            "max_drawdown": _pctl(max_drawdowns, 95),
        },
        "all_terminal_wealth": [round(float(tw), 4) for tw in terminal_wealths],
    }

    logger.info(
        "Monte Carlo (%d sims): median TW=%.3f, 5th=%.3f, 95th=%.3f",
        n_simulations,
        result["percentile_50"]["terminal_wealth"],
        result["percentile_5"]["terminal_wealth"],
        result["percentile_95"]["terminal_wealth"],
    )
    return result


# ------------------------------------------------------------------ #
#  Beta Leakage                                                        #
# ------------------------------------------------------------------ #


def compute_beta_leakage(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, float]:
    """OLS regression of strategy on benchmark returns.

    A truly market-neutral strategy should have beta ≈ 0. Significant
    beta indicates unintended market exposure.

    Parameters
    ----------
    strategy_returns : pd.Series
    benchmark_returns : pd.Series

    Returns
    -------
    dict
        Keys: ``beta``, ``alpha`` (annualised), ``r_squared``,
        ``residual_vol``.
    """
    aligned = pd.DataFrame({
        "strat": strategy_returns,
        "bench": benchmark_returns,
    }).dropna()

    if len(aligned) < 30:
        logger.warning("Too few aligned observations for beta analysis")
        return {"beta": 0.0, "alpha": 0.0, "r_squared": 0.0, "residual_vol": 0.0}

    x = aligned["bench"].values
    y = aligned["strat"].values

    # OLS via normal equation: y = alpha + beta * x
    x_mean = x.mean()
    y_mean = y.mean()
    cov_xy = ((x - x_mean) * (y - y_mean)).mean()
    var_x = ((x - x_mean) ** 2).mean()

    beta = cov_xy / var_x if var_x > 0 else 0.0
    alpha_daily = y_mean - beta * x_mean
    alpha_ann = alpha_daily * 252

    # R²
    ss_res = ((y - alpha_daily - beta * x) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Residual vol
    residuals = y - alpha_daily - beta * x
    res_vol = float(residuals.std() * np.sqrt(252))

    result = {
        "beta": round(float(beta), 4),
        "alpha": round(float(alpha_ann), 4),
        "r_squared": round(float(r_sq), 4),
        "residual_vol": round(res_vol, 4),
    }

    logger.info(
        "Beta leakage: beta=%.4f, alpha(ann)=%.4f, R²=%.4f",
        result["beta"], result["alpha"], result["r_squared"],
    )
    return result
