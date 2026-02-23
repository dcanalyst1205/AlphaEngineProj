"""
Data Loader — OHLCV data acquisition and parquet-based storage.

Downloads daily OHLCV data for a configurable equity universe plus a
benchmark (SPY) via yfinance, stores locally as parquet via pyarrow,
and provides deterministic load-or-download semantics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #


def load_universe(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load (or download) OHLCV data for every ticker in the universe.

    Data is cached on disk as per-ticker parquet files.  If a parquet
    already exists for a ticker it is loaded directly; otherwise yfinance
    is queried and the result is persisted.

    Parameters
    ----------
    config : dict
        Full configuration dictionary (parsed from config.yaml).

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame, DatetimeIndex-ed.
    """
    tickers: list[str] = config["universe"]["tickers"]
    start: str = config["universe"]["start_date"]
    end: str = config["universe"]["end_date"]
    parquet_dir = Path(config["data"]["parquet_dir"])
    parquet_dir.mkdir(parents=True, exist_ok=True)

    data: Dict[str, pd.DataFrame] = {}
    missing_tickers: list[str] = []

    # --- attempt to load from cache first ---
    for ticker in tickers:
        path = parquet_dir / f"{ticker}.parquet"
        if path.exists():
            df = _read_parquet(path)
            data[ticker] = df
            logger.debug("Loaded %s from cache (%d rows)", ticker, len(df))
        else:
            missing_tickers.append(ticker)

    # --- download whatever is missing ---
    if missing_tickers:
        logger.info(
            "Downloading %d tickers from yfinance …", len(missing_tickers)
        )
        downloaded = download_ohlcv(missing_tickers, start, end)
        for ticker, df in downloaded.items():
            path = parquet_dir / f"{ticker}.parquet"
            _write_parquet(df, path)
            data[ticker] = df

    # --- filter to only tickers that have valid data ---
    valid: Dict[str, pd.DataFrame] = {}
    for ticker, df in data.items():
        if df.empty or df["Close"].isna().all():
            logger.warning("Dropping %s — no valid price data", ticker)
            continue
        valid[ticker] = df

    logger.info(
        "Universe loaded: %d / %d tickers with data", len(valid), len(tickers)
    )
    return valid


def get_benchmark(config: Dict[str, Any]) -> pd.DataFrame:
    """Load (or download) the benchmark OHLCV data (default: SPY).

    Parameters
    ----------
    config : dict
        Full configuration dictionary.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame for the benchmark, DatetimeIndex-ed.
    """
    ticker: str = config["universe"]["benchmark"]
    start: str = config["universe"]["start_date"]
    end: str = config["universe"]["end_date"]
    parquet_dir = Path(config["data"]["parquet_dir"])
    parquet_dir.mkdir(parents=True, exist_ok=True)

    path = parquet_dir / f"{ticker}.parquet"
    if path.exists():
        logger.debug("Loaded benchmark %s from cache", ticker)
        return _read_parquet(path)

    logger.info("Downloading benchmark %s …", ticker)
    downloaded = download_ohlcv([ticker], start, end)
    if ticker not in downloaded or downloaded[ticker].empty:
        raise RuntimeError(f"Failed to download benchmark {ticker}")

    df = downloaded[ticker]
    _write_parquet(df, path)
    return df


# ------------------------------------------------------------------ #
#  Download helpers                                                    #
# ------------------------------------------------------------------ #


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    batch_size: int = 50,
) -> Dict[str, pd.DataFrame]:
    """Download daily OHLCV data from yfinance.

    Downloads in batches for efficiency.  Tickers that fail are logged
    and skipped rather than raising.

    Parameters
    ----------
    tickers : list[str]
        List of Yahoo Finance tickers.
    start, end : str
        Date range strings (``YYYY-MM-DD``).
    batch_size : int
        Number of tickers to request per yfinance call.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.
    """
    result: Dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logger.info(
            "Downloading batch %d–%d of %d",
            i + 1,
            min(i + batch_size, len(tickers)),
            len(tickers),
        )
        try:
            raw: pd.DataFrame = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            logger.exception("yfinance batch download failed for batch %d", i)
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = raw.copy()
                else:
                    df = raw[ticker].copy()

                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)

                df = _clean_ohlcv(df, ticker)
                if not df.empty:
                    result[ticker] = df
            except Exception:
                logger.warning("Failed to parse data for %s", ticker, exc_info=True)

    logger.info("Downloaded %d / %d tickers", len(result), len(tickers))
    return result


# ------------------------------------------------------------------ #
#  Parquet I/O                                                         #
# ------------------------------------------------------------------ #


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet with pyarrow engine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=True)
    logger.debug("Saved parquet: %s (%d rows)", path, len(df))


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file and ensure DatetimeIndex."""
    df = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


# ------------------------------------------------------------------ #
#  Cleaning                                                            #
# ------------------------------------------------------------------ #

_EXPECTED_COLS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


def _clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate and clean a raw OHLCV DataFrame.

    - Ensures required columns exist
    - Drops rows where *all* OHLCV values are NaN
    - Forward-fills small gaps (≤ 5 days)
    - Ensures a sorted DatetimeIndex
    """
    if df.empty:
        logger.warning("%s: empty DataFrame received", ticker)
        return df

    # Normalise column names (yfinance can be inconsistent)
    col_map: Dict[str, str] = {}
    for col in df.columns:
        clean = str(col).strip().title()
        if clean == "Adj Close" or str(col).strip().lower() in (
            "adj close",
            "adj_close",
            "adjclose",
        ):
            col_map[col] = "Adj Close"
        else:
            col_map[col] = clean
    df = df.rename(columns=col_map)

    # Check for required columns
    present = set(df.columns) & _EXPECTED_COLS
    if len(present) < 4:
        logger.warning(
            "%s: only %d of 6 expected columns found — skipping", ticker, len(present)
        )
        return pd.DataFrame()

    # If "Adj Close" is missing, copy Close
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    # Keep only expected columns
    df = df[[c for c in df.columns if c in _EXPECTED_COLS]]

    # Sort index, drop full-NaN rows, forward-fill small gaps
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna(how="all")
    df = df.ffill(limit=5)

    return df
