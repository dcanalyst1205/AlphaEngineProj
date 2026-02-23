"""
Feature Engineering — 30+ alpha factors with strict leakage prevention.

Every feature is computed using data available *before* the prediction
date.  The ``feature_lag`` parameter (default 1 day) controls the
minimum shift applied to raw price data before feature computation,
guaranteeing that no same-day information leaks into the model.

Feature categories
------------------
- Momentum (1m / 3m / 6m returns, 12-1m momentum)
- Volatility (20d / 60d realised vol, ATR, vol-of-vol)
- Technical (RSI, MACD line / signal / histogram)
- Volume (z-score, 5d/20d ratio, OBV slope)
- Statistical (rolling skewness, kurtosis, autocorrelation)
- Cross-sectional (rolling beta vs benchmark, idiosyncratic vol)
- Regime (vol-clustering ratio, high-vol flag)
- Mean reversion (distance from SMA, Bollinger Band position)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #


def compute_all_features(
    universe: Dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
Feature Engineering — v2.0 Cross-Sectional Alpha Factors.

Implements a robust set of relative-value features designed for learning-to-rank:
- Momentum Hierarchy: 12-1m, 6-1m, 3-1m, and Acceleration
- Liquidity & Quality: Dollar Volume Rank, Amihud Illiquidity, Turnover
- Trend Quality: Efficiency Ratio, Stability, Breakout
- Regime Context: Market Dispersion, Correlation

All features are Z-scored cross-sectionally per day to ensure stationarity.
"""

    """
    Compute features for the entire universe and normalize cross-sectionally.
    """
    features: Dict[str, pd.DataFrame] = {}
    
    # Pre-compute Benchmark returns for beta/correlation
    bench_ret = benchmark_df["Adj Close"].pct_change()
    
    logger.info("Computing features for %d tickers...", len(universe))
    
    for idx, (ticker, df) in enumerate(universe.items(), 1):
        if idx % 25 == 0:
            logger.info("Computing features: %d / %d", idx, len(universe))
            
        try:
            # Enforce data quality
            if df.empty or len(df) < 252:
                continue
                
            feat = _compute_single_ticker(ticker, df, bench_ret, config)
            if not feat.empty:
                features[ticker] = feat
                
        except Exception:
            logger.warning("Feature computation failed for %s", ticker, exc_info=True)

    logger.info("Features computed for %d tickers", len(features))
    
    # CRITICAL: Cross-Sectional Normalization
    # This transforms time-series values into relative rankings
    features = standardize_features_cross_sectional(features)
    
    return features


def compute_target_rank(
    price_series: pd.Series,
    horizon: int = 21,
) -> pd.Series:
    """
    Compute forward returns for ranking.
    Actually, we compute raw forward returns here.
    Ranking happens inside the backtester or DataHandler before training.
    
    The target is simply: Return(t to t+21).
    """
    # Forward return: (Price_{t+h} / Price_t) - 1
    fwd_ret = price_series.shift(-horizon) / price_series - 1.0
    fwd_ret.name = f"fwd_ret_{horizon}d"
    return fwd_ret


# ------------------------------------------------------------------ #
#  Internal: single-ticker feature computation                         #
# ------------------------------------------------------------------ #


def _compute_single_ticker(
    ticker: str,
    df: pd.DataFrame,
    bench_ret: pd.Series,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Compute raw features for a single ticker (before normalization)."""
    # Configuration
    lag = config["features"].get("feature_lag", 1)
    
    # Data Setup
    closes = df["Adj Close"]
    opens = df["Open"]
    highs = df["High"]
    lows = df["Low"]
    volumes = df["Volume"]
    
    # Returns
    ret_1d = closes.pct_change()
    
    feats = {}
    
    # ---- 1. Momentum Hierarchy (Lagged by definition involves shifts) ----
    
    # Standard: exclude most recent month (21d) for reversal avoidance
    # 12-1 Month Momentum: Return from t-252 to t-21
    ret_12m = closes.shift(21) / closes.shift(252) - 1
    feats["mom_12m_1m"] = ret_12m.shift(lag)
    
    # 6-1 Month
    ret_6m = closes.shift(21) / closes.shift(126) - 1
    feats["mom_6m_1m"] = ret_6m.shift(lag)
    
    # 3-1 Month
    ret_3m = closes.shift(21) / closes.shift(63) - 1
    feats["mom_3m_1m"] = ret_3m.shift(lag)
    
    # Short-Term Reversal (1 month)
    ret_1m = closes / closes.shift(21) - 1
    feats["mom_1m"] = ret_1m.shift(lag)  # Often negatively correlated
    
    # Momentum Acceleration: (6m-1m) - (12m-1m) ? Or just 3m - 6m?
    # Let's use 3m-1m minus 6m-1m (is it accelerating?)
    feats["mom_accel"] = (feats["mom_3m_1m"] - feats["mom_6m_1m"])

    # ---- 2. Liquidity & Quality --------------------------------------
    
    # Dollar Volume (Log) - We will Rank this cross-sectionally later
    # But raw log dollar vol is good
    dol_vol = closes * volumes
    feats["log_dol_vol"] = np.log1p(dol_vol.rolling(21).mean()).shift(lag)
    
    # Amihud Illiquidity Proxy: |Ret| / (Price * Vol)
    # High value = Illiquid. 
    amihud = ret_1d.abs() / (dol_vol + 1e-9)
    feats["amihud_illiq"] = np.log1p(amihud.rolling(63).mean()).shift(lag)
    
    # Turnover Stability: Std of Volume / Mean Volume
    vol_stability = volumes.rolling(63).std() / (volumes.rolling(63).mean() + 1)
    feats["vol_stability"] = vol_stability.shift(lag)

    # ---- 3. Trend Quality & Volatility -------------------------------
    
    # Efficiency Ratio (Kaufman): Net Move / Sum of Abs Moves
    window = 21
    net_move = (closes - closes.shift(window)).abs()
    path_len = ret_1d.abs().rolling(window).sum() * closes.shift(window) # approx sum of abs changes
    # More precise: sum of |diff|
    path_len_precise = closes.diff().abs().rolling(window).sum()
    feats["efficiency_ratio"] = (net_move / (path_len_precise + 1e-9)).shift(lag) 
    
    # Volatility (63d)
    vol_63 = ret_1d.rolling(63).std()
    feats["vol_63d"] = vol_63.shift(lag)
    
    # Idiosyncratic Volatility (vs SPY)
    # We can approximate by residual of regression, or simpler: 
    # Var(Stock) - Beta^2 * Var(Market). 
    # Let's compute Rolling Beta first.
    
    # Align dates for correlation
    stock_ret_aligned = ret_1d
    bench_ret_aligned = bench_ret.reindex(stock_ret_aligned.index).fillna(0)
    
    # Rolling Beta (63d)
    cov = stock_ret_aligned.rolling(63).cov(bench_ret_aligned)
    var_b = bench_ret_aligned.rolling(63).var()
    beta = cov / (var_b + 1e-9)
    feats["beta_63d"] = beta.shift(lag)
    
    # Idiosyncratic Vol calculation
    # Resids = StockRet - Beta * MarketRet
    # IdioVol = Std(Resids)
    # Approximation: Sqrt(Var(Stock) - Beta^2 * Var(Market))
    var_s = stock_ret_aligned.rolling(63).var()
    idio_var = var_s - (beta**2 * var_b)
    feats["idio_vol_63d"] = np.sqrt(idio_var.clip(lower=0)).shift(lag)
    
    # Downside Deviation (Semi-variance)
    # rolling apply is slow, skipping for speed or approximate 
    # by taking rolling min returns
    feats["downside_skew"] = (ret_1d.rolling(63).min() / vol_63).shift(lag)

    # ---- 4. Relative Value -------------------------------------------
    
    # Price vs 52-week High
    high_252 = highs.rolling(252).max()
    feats["price_to_high"] = (closes / high_252).shift(lag)
    
    # Price vs 50d SMA (Trend)
    sma_50 = closes.rolling(50).mean()
    feats["price_to_sma50"] = (closes / sma_50).shift(lag)
    
    # ---- Assemble ----------------------------------------------------
    features_df = pd.DataFrame(feats, index=df.index)
    
    # Drop rows with NaNs (warmup period)
    # Grouped standardization handles NaNs gracefully, but models don't.
    # We'll drop later or here.
    # Let's drop initial NaNs but keep index aligned.
    
    return features_df


def standardize_features_cross_sectional(
    features: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Z-score all features cross-sectionally per date.
    Vectorized for performance to avoid gateway timeouts.
    """
    if not features:
        return features
        
    logger.info("Standardizing features cross-sectionally (vectorized)...")
    
    # 1. Stack: (Date, Ticker) -> Features
    # Use a faster approach to build the panel
    all_dfs = []
    for ticker, df in features.items():
        if df.empty: continue
        d = df.copy()
        d["_ticker"] = ticker
        all_dfs.append(d)
        
    full = pd.concat(all_dfs)
    
    # 2. Vectorized Z-Score
    cols = [c for c in full.columns if c != "_ticker"]
    
    # Group by Date (index level 0)
    # Using groupby.mean() and groupby.std() is faster than transform(zscore)
    # for large number of groups/columns
    group_means = full[cols].groupby(level=0).mean()
    group_stds = full[cols].groupby(level=0).std()
    
    # Handle zero std
    group_stds = group_stds.replace(0, 1.0).fillna(1.0)
    
    # Align and broadcast
    # reindex(full.index, level=0) broadcasts the group stats to every row
    full_z = (full[cols] - group_means.reindex(full.index, level=0)) / group_stds.reindex(full.index, level=0)
    
    # 3. Clip and Fill
    full_z = full_z.clip(-3.0, 3.0).fillna(0.0)
    
    # 4. Unstack back to dictionary
    full_z["_ticker"] = full["_ticker"]
    
    output = {}
    for ticker, group in full_z.groupby("_ticker"):
        output[ticker] = group.drop(columns=["_ticker"])
        
    return output



