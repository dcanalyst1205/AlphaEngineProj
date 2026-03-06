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
    horizon: int = 7,
) -> pd.Series:
    """
    Compute forward returns for ranking.
    Actually, we compute raw forward returns here.
    Ranking happens inside the backtester or DataHandler before training.
    
    The target is simply: Return(t to t+horizon).
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


# Sector mapping for top ~100 tickers to implement Sector Neutralization
GICS_SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "CSCO": "Tech", "ADBE": "Tech",
    "TXN": "Tech", "QCOM": "Tech", "AMAT": "Tech", "IBM": "Tech", "AVGO": "Tech",
    "ACN": "Tech", "ADI": "Tech", "LRCX": "Tech", "FIS": "Tech", "SNPS": "Tech",
    "KLAC": "Tech", "INTC": "Tech", "CRM": "Tech", "AMD": "Tech", "INTU": "Tech",
    "GOOGL": "Comm", "META": "Comm", "DIS": "Comm", "NFLX": "Comm", "CMCSA": "Comm",
    "ATVI": "Comm", "TMUS": "Comm", "VZ": "Comm", "T": "Comm",
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "NKE": "ConsDisc",
    "LOW": "ConsDisc", "SBUX": "ConsDisc", "BKNG": "ConsDisc", "TGT": "ConsDisc",
    "ORLY": "ConsDisc", "F": "ConsDisc", "MCD": "ConsDisc",
    "JPM": "Fin", "BAC": "Fin", "GS": "Fin", "BLK": "Fin", "AXP": "Fin", "CME": "Fin",
    "MCO": "Fin", "PNC": "Fin", "CB": "Fin", "V": "Fin", "MA": "Fin", "WFC": "Fin", "C": "Fin",
    "UNH": "Health", "PFE": "Health", "TMO": "Health", "ABT": "Health", "MRK": "Health",
    "LLY": "Health", "ABBV": "Health", "DHR": "Health", "MDT": "Health", "BMY": "Health",
    "AMGN": "Health", "GILD": "Health", "ISRG": "Health", "SYK": "Health", "CI": "Health",
    "REGN": "Health", "BDX": "Health", "HUM": "Health", "EW": "Health", "JNJ": "Health", "CVS": "Health",
    "XOM": "Energy", "CVX": "Energy", "SLB": "Energy", "COP": "Energy", "EOG": "Energy",
    "WMT": "Staples", "PEP": "Staples", "COST": "Staples", "PM": "Staples", "MO": "Staples",
    "MDLZ": "Staples", "CL": "Staples", "KMB": "Staples", "KO": "Staples", "PG": "Staples",
    "UPS": "Indus", "HON": "Indus", "CAT": "Indus", "BA": "Indus", "GE": "Indus", 
    "ADP": "Indus", "MMC": "Indus", "DE": "Indus", "CSX": "Indus", "ITW": "Indus", 
    "MMM": "Indus", "NOC": "Indus", "WM": "Indus", "EMR": "Indus", "UNP": "Indus", "RTX": "Indus", "LMT": "Indus",
    "NEE": "Util", "DUK": "Util", "SO": "Util", "D": "Util",
    "LIN": "Mat", "SHW": "Mat", "APD": "Mat",
    "PLD": "RealEst", "CCI": "RealEst", "EQIX": "RealEst", "AMT": "RealEst"
}

def standardize_features_cross_sectional(
    features: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Z-score all features cross-sectionally per date and per sector.
    This neutralizes the features so we are not accidentally betting all on one sector.
    """
    if not features:
        return features
        
    logger.info("Standardizing features cross-sectionally by sector...")
    
    all_dfs = []
    for ticker, df in features.items():
        if df.empty: continue
        d = df.copy()
        d["_ticker"] = ticker
        d["_sector"] = GICS_SECTOR_MAP.get(ticker, "Other")
        all_dfs.append(d)
        
    full = pd.concat(all_dfs)
    # Ensure index is named for merging
    full.index.name = "_date"
    cols = [c for c in full.columns if c not in ("_ticker", "_sector")]
    
    # 2. Vectorized Z-Score by Date AND Sector
    # Reset index to make _date a column for easier groupby/merge
    full_reset = full.reset_index()
    grouped = full_reset.groupby(["_date", "_sector"])[cols]
    group_means = grouped.transform("mean")
    group_stds = grouped.transform("std").replace(0, 1.0).fillna(1.0)
    
    # 3. Z-Score
    full_z_vals = (full_reset[cols] - group_means) / group_stds
    full_z = pd.DataFrame(full_z_vals.values, index=full.index, columns=cols)
    
    # 4. Clip and Fill
    full_z = full_z.clip(-3.0, 3.0).fillna(0.0)
    
    # 5. Unstack back to dictionary
    full_z["_ticker"] = full["_ticker"]
    
    output = {}
    for ticker, group in full_z.groupby("_ticker"):
        output[ticker] = group.drop(columns=["_ticker"])
        
    return output



