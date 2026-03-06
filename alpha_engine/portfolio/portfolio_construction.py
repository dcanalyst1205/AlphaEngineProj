"""
Portfolio Construction — long-short portfolios with vol-targeting and costs.

Ranks cross-sectional predictions each rebalance day, constructs a
dollar-neutral long-short portfolio (top/bottom quintiles), applies
position caps and volatility targeting, models transaction costs,
and enforces turnover constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Result container                                                    #
# ------------------------------------------------------------------ #


@dataclass
class PortfolioResult:
    """Container for portfolio-level outputs."""

    # Daily portfolio returns (gross)
    gross_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # Daily portfolio returns (net of costs)
    net_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # Transaction costs per rebalance
    costs_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # Weights at each rebalance: date → {ticker: weight}
    weights_history: Dict[pd.Timestamp, Dict[str, float]] = field(
        default_factory=dict
    )
    # Turnover per rebalance
    turnover_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    # Vol-scaling applied
    vol_scale_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #


def construct_portfolio(
    predictions: Dict[str, pd.Series],
    realised: Dict[str, pd.Series],
    universe: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    rebalance_dates: List[pd.Timestamp],
) -> PortfolioResult:
    """Build a long-short portfolio from model predictions.

    Parameters
    ----------
    predictions : Dict[str, pd.Series]
        Ticker → predicted returns (indexed by date).
    realised : Dict[str, pd.Series]
        Ticker → realised forward returns.
    universe : Dict[str, pd.DataFrame]
        Ticker → OHLCV (used for vol calculations).
    config : dict
        Full config dictionary.
    rebalance_dates : List[pd.Timestamp]
        Dates on which to rebalance.

    Returns
    -------
    PortfolioResult
    """
    port_cfg = config["portfolio"]
    cost_cfg = config["costs"]
    long_pct = port_cfg["long_percentile"]
    short_pct = port_cfg["short_percentile"]
    max_pos = port_cfg["max_position_weight"]
    vol_target = port_cfg["volatility_target"]
    vol_lookback = port_cfg["vol_lookback"]
    max_turnover = port_cfg["max_turnover"]
    weighting = port_cfg.get("weighting", "equal")

    # --- Build daily returns matrix -------------------------------- #
    daily_ret = _build_returns_matrix(universe)

    result = PortfolioResult()
    prev_weights: Dict[str, float] = {}
    port_ret_list: List[Tuple[pd.Timestamp, float]] = []
    cost_list: List[Tuple[pd.Timestamp, float]] = []
    turnover_list: List[Tuple[pd.Timestamp, float]] = []
    vol_scale_list: List[Tuple[pd.Timestamp, float]] = []

    # Apply signal smoothing to reduce turnover
    # Default halflife=5 days if not specified
    smooth_halflife = port_cfg.get("signal_smoothing_halflife", 5)
    if smooth_halflife > 0:
        logger.info("Smoothing predictions with halflife=%d days", smooth_halflife)
        predictions = {
            t: p.ewm(halflife=smooth_halflife).mean() 
            for t, p in predictions.items() 
            if not p.empty
        }

    # Determine full set of trading days in the out-of-sample range
    if not rebalance_dates:
        return result

    all_trading_days = daily_ret.index.sort_values()
    oos_start = rebalance_dates[0]
    oos_days = all_trading_days[all_trading_days >= oos_start]

    current_weights: Dict[str, float] = {}
    next_reb_idx = 0

    for day in oos_days:
        # ---- Check for rebalance ---------------------------------- #
        if next_reb_idx < len(rebalance_dates) and day >= rebalance_dates[next_reb_idx]:
            reb_date = rebalance_dates[next_reb_idx]
            next_reb_idx += 1

            # Collect cross-sectional predictions for this date
            pred_cs: Dict[str, float] = {}
            for ticker, pred_s in predictions.items():
                if reb_date in pred_s.index:
                    pred_cs[ticker] = pred_s.loc[reb_date]
                elif len(pred_s) > 0:
                    # Use the latest prediction before reb_date
                    valid = pred_s.index[pred_s.index <= reb_date]
                    if len(valid) > 0:
                        pred_cs[ticker] = pred_s.loc[valid[-1]]

            if len(pred_cs) < 10:
                continue

            # Rank and select long/short with 5th percentile Confidence Threshold
            raw_weights = _rank_and_select(
                pred_cs, 95.0, 5.0, # Top/Bottom 5th percentile
                weighting=weighting,
                daily_ret=daily_ret,
                current_date=reb_date,
                vol_lookback=vol_lookback,
            )

            # Position caps
            raw_weights = _apply_position_caps(raw_weights, max_pos)

            # Volatility targeting
            vol_scale = _compute_vol_scale(
                current_weights, daily_ret, day, vol_target, vol_lookback
            )
            scaled_weights = {
                t: w * vol_scale for t, w in raw_weights.items()
            }

            # Turnover constraint
            final_weights = _apply_turnover_constraint(
                scaled_weights, current_weights, max_turnover
            )

            # --- STRICT MARKET NEUTRAL CONSTRAINT ---
            # Force the dollar-amount of Longs to equal Shorts exactly at every rebalance
            l_sum = sum(w for w in final_weights.values() if w > 0)
            s_sum = abs(sum(w for w in final_weights.values() if w < 0))
            if l_sum > 0 and s_sum > 0:
                target_leg = (l_sum + s_sum) / 2.0
                final_weights = {
                    t: (w * (target_leg / l_sum) if w > 0 else w * (target_leg / s_sum))
                    for t, w in final_weights.items()
                }
            elif l_sum > 0 or s_sum > 0:
                final_weights = {t: 0.0 for t in final_weights.keys()}

            # Transaction costs with Square-Root Market Impact
            tc = _compute_transaction_costs(
                current_weights, final_weights, daily_ret, universe, day, cost_cfg
            )

            # Record
            turnover = _compute_turnover(current_weights, final_weights)
            turnover_list.append((day, turnover))
            cost_list.append((day, tc))
            vol_scale_list.append((day, vol_scale))
            result.weights_history[day] = final_weights.copy()

            current_weights = final_weights

        # ---- Compute daily return --------------------------------- #
        if day in daily_ret.index and current_weights:
            day_ret = 0.0
            for ticker, w in current_weights.items():
                if ticker in daily_ret.columns and not np.isnan(
                    daily_ret.loc[day, ticker]
                ):
                    day_ret += w * daily_ret.loc[day, ticker]
            port_ret_list.append((day, day_ret))

    # ---- Assemble result ------------------------------------------ #
    if port_ret_list:
        ret_df = pd.DataFrame(port_ret_list, columns=["date", "return"]).set_index(
            "date"
        )
        result.gross_returns = ret_df["return"]

        # Spread costs across trading days between rebalances
        cost_s = pd.Series(
            dict(cost_list), name="costs"
        ).reindex(result.gross_returns.index, fill_value=0.0)
        result.costs_series = cost_s
        result.net_returns = result.gross_returns - cost_s

    if turnover_list:
        result.turnover_series = pd.Series(dict(turnover_list), name="turnover")
    if vol_scale_list:
        result.vol_scale_series = pd.Series(dict(vol_scale_list), name="vol_scale")

    logger.info(
        "Portfolio built: %d trading days, %d rebalances",
        len(result.gross_returns),
        len(result.weights_history),
    )
    return result


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #


def _build_returns_matrix(
    universe: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a (dates × tickers) matrix of daily simple returns."""
    series_list = {}
    for ticker, df in universe.items():
        if "Adj Close" in df.columns:
            series_list[ticker] = df["Adj Close"].pct_change()
        elif "Close" in df.columns:
            series_list[ticker] = df["Close"].pct_change()
    return pd.DataFrame(series_list).sort_index()


def _rank_and_select(
    pred_cs: Dict[str, float],
    long_pct: float,
    short_pct: float,
    weighting: str = "equal",
    daily_ret: Optional[pd.DataFrame] = None,
    current_date: Optional[pd.Timestamp] = None,
    vol_lookback: int = 63,
) -> Dict[str, float]:
    """Rank predictions and assign weights to long/short buckets.

    Long top ``(100 - long_pct)``% and short bottom ``short_pct``%.
    Weights are dollar-neutral (sum to ~0).

    Parameters
    ----------
    weighting : str
        ``"equal"`` for equal weighting, ``"inverse_volatility"`` for
        inverse-vol weighting (lower vol → higher weight).
    """
    s = pd.Series(pred_cs).dropna().sort_values()

    short_cutoff = np.percentile(s.values, short_pct)
    long_cutoff = np.percentile(s.values, long_pct)

    shorts = s[s <= short_cutoff].index.tolist()
    longs = s[s >= long_cutoff].index.tolist()

    weights: Dict[str, float] = {}

    if weighting == "inverse_volatility" and daily_ret is not None and current_date is not None:
        weights = _inverse_volatility_weights(
            longs, shorts, daily_ret, current_date, vol_lookback,
        )
    else:
        # Equal weighting (original)
        if longs:
            w_long = 1.0 / len(longs)
            for t in longs:
                weights[t] = w_long
        if shorts:
            w_short = -1.0 / len(shorts)
            for t in shorts:
                weights[t] = w_short

    return weights


def _inverse_volatility_weights(
    longs: List[str],
    shorts: List[str],
    daily_ret: pd.DataFrame,
    current_date: pd.Timestamp,
    vol_lookback: int = 63,
) -> Dict[str, float]:
    """Compute inverse-volatility weights for long and short buckets.

    Lower realised volatility → higher weight (more stable assets
    get larger allocations). Falls back to equal weighting if vol
    data is insufficient.
    """
    dates_before = daily_ret.index[daily_ret.index < current_date]
    lookback = dates_before[-vol_lookback:] if len(dates_before) >= vol_lookback else dates_before

    def _get_inv_vols(tickers: List[str]) -> Dict[str, float]:
        inv_vols: Dict[str, float] = {}
        for t in tickers:
            if t in daily_ret.columns and len(lookback) > 10:
                vol = daily_ret.loc[lookback, t].std() * np.sqrt(252)
                inv_vols[t] = 1.0 / max(vol, 0.01)  # floor to avoid div-by-zero
            else:
                inv_vols[t] = 1.0  # fallback to equal
        return inv_vols

    weights: Dict[str, float] = {}

    if longs:
        long_ivols = _get_inv_vols(longs)
        total_long = sum(long_ivols.values())
        for t in longs:
            weights[t] = long_ivols[t] / total_long

    if shorts:
        short_ivols = _get_inv_vols(shorts)
        total_short = sum(short_ivols.values())
        for t in shorts:
            weights[t] = -(short_ivols[t] / total_short)

    return weights


def _apply_position_caps(
    weights: Dict[str, float],
    max_weight: float,
) -> Dict[str, float]:
    """Cap absolute position weights and renormalise."""
    capped = {t: np.clip(w, -max_weight, max_weight) for t, w in weights.items()}
    # Renormalise longs and shorts separately to maintain dollar neutrality
    long_sum = sum(w for w in capped.values() if w > 0)
    short_sum = abs(sum(w for w in capped.values() if w < 0))

    out: Dict[str, float] = {}
    for t, w in capped.items():
        if w > 0 and long_sum > 0:
            out[t] = w / long_sum
        elif w < 0 and short_sum > 0:
            out[t] = -abs(w) / short_sum
        else:
            out[t] = 0.0

    return out


def _compute_vol_scale(
    current_weights: Dict[str, float],
    daily_ret: pd.DataFrame,
    current_date: pd.Timestamp,
    vol_target: float,
    vol_lookback: int,
) -> float:
    """Compute the scalar to apply to portfolio weights for vol-targeting.

    Uses trailing realised volatility of the portfolio.
    """
    if not current_weights:
        return 1.0

    # Compute trailing portfolio returns
    dates_before = daily_ret.index[daily_ret.index < current_date]
    if len(dates_before) < vol_lookback:
        return 1.0

    lookback_dates = dates_before[-vol_lookback:]
    port_rets = pd.Series(0.0, index=lookback_dates)

    for ticker, w in current_weights.items():
        if ticker in daily_ret.columns:
            port_rets += w * daily_ret.loc[lookback_dates, ticker].fillna(0)

    realised_vol = port_rets.std() * np.sqrt(252)
    if realised_vol <= 0 or np.isnan(realised_vol):
        return 1.0

    scale = vol_target / realised_vol
    # Clamp to avoid extreme leverage
    return float(np.clip(scale, 0.25, 3.0))


def _apply_turnover_constraint(
    new_weights: Dict[str, float],
    old_weights: Dict[str, float],
    max_turnover: float,
) -> Dict[str, float]:
    """Limit one-way turnover to ``max_turnover``."""
    if not old_weights:
        return new_weights

    turnover = _compute_turnover(old_weights, new_weights)
    if turnover <= max_turnover:
        return new_weights

    # Blend old and new to hit turnover target
    blend = max_turnover / max(turnover, 1e-8)
    blended: Dict[str, float] = {}
    all_tickers = set(old_weights) | set(new_weights)

    for t in all_tickers:
        w_old = old_weights.get(t, 0.0)
        w_new = new_weights.get(t, 0.0)
        blended[t] = w_old + blend * (w_new - w_old)

    return blended


def _compute_turnover(
    old: Dict[str, float],
    new: Dict[str, float],
) -> float:
    """One-way turnover: half the sum of absolute weight changes."""
    all_tickers = set(old) | set(new)
    return 0.5 * sum(abs(new.get(t, 0) - old.get(t, 0)) for t in all_tickers)


def _compute_transaction_costs(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
    daily_ret: pd.DataFrame,
    universe: Dict[str, pd.DataFrame],
    date: pd.Timestamp,
    cost_cfg: Dict[str, Any],
) -> float:
    """Model transaction costs for a single rebalance.

    Costs = commission (bps) + spread (bps) + Square-Root Market Impact.
    Market Impact = sigma * sqrt(OrderSize / DailyVolume)
    """
    comm_bps = cost_cfg.get("commission_bps", 2.0) / 10_000
    spread_bps = cost_cfg.get("spread_bps", 1.0) / 10_000
    
    all_tickers = set(old_weights) | set(new_weights)
    total_cost = 0.0

    for ticker in all_tickers:
        w_old = old_weights.get(ticker, 0.0)
        w_new = new_weights.get(ticker, 0.0)
        delta = abs(w_new - w_old)
        
        if delta < 1e-8:
            continue

        # 1. Fixed costs: Commission + spread
        fixed_cost = delta * (comm_bps + spread_bps)
        
        # 2. Square-Root Market Impact: sigma * sqrt(delta / avg_volume)
        # We use a simplified version for research: sigma * sqrt(delta) * multiplier
        impact_cost = 0.0
        if ticker in universe:
            df = universe[ticker]
            # Get 20-day average dollar volume
            if "Volume" in df.columns and "Adj Close" in df.columns:
                dol_vol = (df["Volume"] * df["Adj Close"]).rolling(20).mean()
                if date in dol_vol.index:
                    avg_dol_vol = dol_vol.loc[date]
                    # We assume a portfolio size for the purpose of the sqrt model relative impact
                    # Let's say $100M AUM for impact scaling if not specified
                    aum = cost_cfg.get("assumed_aum", 100_000_000)
                    order_size = delta * aum
                    
                    if avg_dol_vol > 0:
                        vol_20d = daily_ret[ticker].rolling(20).std() if ticker in daily_ret.columns else 0.02
                        sigma = vol_20d.loc[date] if date in vol_20d.index else 0.02
                        # Square-root model: sigma * sqrt(OrderSize / DailyVolume)
                        impact_cost = delta * sigma * np.sqrt(order_size / (avg_dol_vol + 1e-9))
        
        total_cost += fixed_cost + impact_cost

    return total_cost
