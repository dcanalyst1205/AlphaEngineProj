"""
Performance Analytics — comprehensive strategy evaluation and reporting.

Computes institutional-grade performance metrics, generates equity curve
and drawdown plots, compares against a benchmark, and produces a
professional summary report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ------------------------------------------------------------------ #
#  Core metrics                                                        #
# ------------------------------------------------------------------ #


def compute_performance(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Compute a comprehensive set of performance metrics.

    Metrics
    -------
    CAGR, annualised volatility, Sharpe, Sortino, max drawdown,
    Calmar ratio, information ratio (vs benchmark), hit rate,
    skewness, kurtosis, best/worst day.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns (gross or net).
    benchmark_returns : pd.Series, optional
        Daily benchmark returns for IR calculation.

    Returns
    -------
    dict
        Metric name → value.
    """
    r = returns.dropna()
    if r.empty:
        return {}

    n_years = len(r) / TRADING_DAYS

    # Cumulative
    cum = (1 + r).cumprod()
    total_ret = float(cum.iloc[-1] - 1)

    # CAGR
    cagr = float(cum.iloc[-1] ** (1 / max(n_years, 0.01)) - 1)

    # Volatility
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))

    # Sharpe
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    # Sortino
    downside = r[r < 0]
    downside_vol = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 0 else 1e-8
    sortino = cagr / downside_vol

    # Max drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Hit rate
    hit_rate = float((r > 0).mean())

    metrics: Dict[str, float] = {
        "total_return": round(total_ret, 4),
        "cagr": round(cagr, 4),
        "ann_volatility": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "hit_rate": round(hit_rate, 4),
        "skewness": round(float(r.skew()), 4),
        "kurtosis": round(float(r.kurt()), 4),
        "best_day": round(float(r.max()), 6),
        "worst_day": round(float(r.min()), 6),
        "n_trading_days": len(r),
    }

    # Information ratio
    if benchmark_returns is not None:
        ir = _information_ratio(r, benchmark_returns)
        metrics["information_ratio"] = round(ir, 4)

    return metrics


def compare_vs_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Side-by-side comparison of strategy vs benchmark.

    Returns
    -------
    dict
        ``{"strategy": {...}, "benchmark": {...}, "excess": {...}}``.
    """
    strat_metrics = compute_performance(strategy_returns, benchmark_returns)
    bench_metrics = compute_performance(benchmark_returns)

    excess = strategy_returns.subtract(
        benchmark_returns.reindex(strategy_returns.index), fill_value=0
    )
    excess_metrics = compute_performance(excess)
    excess_metrics.pop("information_ratio", None)

    return {
        "strategy": strat_metrics,
        "benchmark": bench_metrics,
        "excess": excess_metrics,
    }


# ------------------------------------------------------------------ #
#  Feature importance stability                                        #
# ------------------------------------------------------------------ #


def feature_importance_stability(
    fold_importances: List[pd.Series],
) -> pd.DataFrame:
    """Analyse stability of feature importance across folds.

    Returns a DataFrame with mean, std, and rank-correlation stability
    for each feature.
    """
    if not fold_importances:
        return pd.DataFrame()

    from scipy.stats import spearmanr

    imp_df = pd.concat(fold_importances, axis=1).fillna(0)
    imp_df.columns = [f"fold_{i}" for i in range(imp_df.shape[1])]

    result = pd.DataFrame({
        "mean_importance": imp_df.mean(axis=1),
        "std_importance": imp_df.std(axis=1),
        "cv": imp_df.std(axis=1) / imp_df.mean(axis=1).replace(0, np.nan),
    }).sort_values("mean_importance", ascending=False)

    # Pairwise rank correlation of importance vectors across folds
    if imp_df.shape[1] >= 2:
        rank_corrs = []
        for i in range(imp_df.shape[1] - 1):
            corr, _ = spearmanr(imp_df.iloc[:, i], imp_df.iloc[:, i + 1])
            rank_corrs.append(corr)
        result.attrs["avg_rank_correlation"] = float(np.nanmean(rank_corrs))

    return result


# ------------------------------------------------------------------ #
#  Information Coefficient (IC) Analytics                             #
# ------------------------------------------------------------------ #


def compute_daily_ic(
    predictions: Dict[str, pd.Series],
    realised: Dict[str, pd.Series],
) -> pd.Series:
    """Compute cross-sectional Spearman Rank IC for each day.

    Returns a Series of IC values indexed by date.
    """
    from scipy.stats import spearmanr

    # Align predictions and realised into a single DataFrame
    dfs = []
    for ticker in predictions:
        if ticker not in realised:
            continue
        p = predictions[ticker]
        r = realised[ticker]
        df = pd.DataFrame({"pred": p, "real": r})
        df["_ticker"] = ticker
        dfs.append(df)
    
    if not dfs:
        return pd.Series(dtype=float)

    full = pd.concat(dfs)

    # Group by date and compute Spearman corr
    def group_ic(group):
        if len(group) < 5:
            return np.nan
        corr, _ = spearmanr(group["pred"], group["real"])
        return corr

    daily_ic = full.groupby(level=0).apply(group_ic)
    return daily_ic.dropna()


def evaluate_ic_stability(daily_ic: pd.Series) -> Dict[str, float]:
    """Compute stability metrics for the IC series.

    Metrics:
    - Mean IC
    - IC Standard Deviation
    - IC IR (Mean / Std)
    - IC t-stat (Mean / (Std/sqrt(N)))
    """
    if daily_ic.empty:
        return {}
    
    mean_ic = daily_ic.mean()
    std_ic = daily_ic.std()
    n = len(daily_ic)
    
    ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 and n > 1 else 0.0
    
    return {
        "mean_ic": round(float(mean_ic), 6),
        "std_ic": round(float(std_ic), 6),
        "ic_ir": round(float(ic_ir), 4),
        "ic_t_stat": round(float(t_stat), 4),
        "n_periods": n,
    }


# ------------------------------------------------------------------ #
#  Signal decay                                                        #
# ------------------------------------------------------------------ #


def evaluate_signal_decay(
    predictions: Dict[str, pd.Series],
    universe: Dict[str, pd.DataFrame],
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Measure information coefficient at multiple forward horizons.

    Parameters
    ----------
    predictions : Dict[str, pd.Series]
        Ticker → predicted return.
    universe : Dict[str, pd.DataFrame]
        Ticker → OHLCV (for computing forward returns at each horizon).
    horizons : list[int]
        Forward horizons in trading days (default 1–10).

    Returns
    -------
    pd.DataFrame
        Horizon → (IC, rank_IC).
    """
    if horizons is None:
        horizons = list(range(1, 11))

    from scipy.stats import spearmanr

    results = []

    for h in horizons:
        ic_list = []
        for ticker in predictions:
            if ticker not in universe:
                continue
            pred = predictions[ticker]
            adj_close = universe[ticker]["Adj Close"]
            fwd = np.log(adj_close.shift(-h) / adj_close)
            common = pred.index.intersection(fwd.dropna().index)
            if len(common) < 20:
                continue
            p = pred.loc[common].values
            f = fwd.loc[common].values
            mask = np.isfinite(p) & np.isfinite(f)
            if mask.sum() < 20:
                continue
            ic_list.append(float(np.corrcoef(p[mask], f[mask])[0, 1]))

        avg_ic = float(np.nanmean(ic_list)) if ic_list else np.nan
        results.append({"horizon": h, "avg_ic": round(avg_ic, 6), "n_tickers": len(ic_list)})

    return pd.DataFrame(results).set_index("horizon")


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #


def plot_equity_curves(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    path: Path,
    net_returns: Optional[pd.Series] = None,
) -> None:
    """Save an equity curve chart as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    strat_cum = (1 + strategy_returns).cumprod()
    bench_cum = (1 + benchmark_returns.reindex(strategy_returns.index, fill_value=0)).cumprod()

    ax.plot(strat_cum.index, strat_cum.values, label="Strategy (gross)", linewidth=1.5)
    if net_returns is not None:
        net_cum = (1 + net_returns).cumprod()
        ax.plot(net_cum.index, net_cum.values, label="Strategy (net)", linewidth=1.5, linestyle="--")
    ax.plot(bench_cum.index, bench_cum.values, label="SPY Buy & Hold", linewidth=1.2, alpha=0.7)

    ax.set_title("Equity Curves", fontsize=14, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved equity curve: %s", path)


def plot_drawdown(
    returns: pd.Series,
    path: Path,
) -> None:
    """Save a drawdown chart as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax.plot(dd.index, dd.values, color="crimson", linewidth=0.8)
    ax.set_title("Underwater (Drawdown) Chart", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved drawdown chart: %s", path)


def plot_rolling_sharpe(
    returns: pd.Series,
    path: Path,
    window: int = 63,
) -> None:
    """Save a rolling Sharpe ratio chart as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rolling_mean = returns.rolling(window).mean() * TRADING_DAYS
    rolling_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    rolling_sharpe = rolling_mean / rolling_vol.replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.2, color="teal")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved rolling Sharpe: %s", path)


def plot_feature_importance(
    importance: pd.Series,
    path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of feature importances."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importance.head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top.index, top.values, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance chart: %s", path)


# ------------------------------------------------------------------ #
#  Report generation                                                   #
# ------------------------------------------------------------------ #


def generate_report(
    strategy_metrics: Dict[str, float],
    benchmark_metrics: Dict[str, float],
    fold_metrics: List[Dict[str, Any]],
    feature_importance: Optional[pd.Series],
    config: Dict[str, Any],
    output_dir: Path,
    ic_stability: Optional[Dict[str, float]] = None,
) -> None:
    """Write a text summary report to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "summary_report.txt"

    lines = [
        "=" * 72,
        "  ALPHA ENGINE — PERFORMANCE SUMMARY REPORT",
        "=" * 72,
        "",
        "CONFIGURATION",
        f"  Universe     : {len(config['universe']['tickers'])} tickers",
        f"  Benchmark    : {config['universe']['benchmark']}",
        f"  Date range   : {config['universe']['start_date']} -> {config['universe']['end_date']}",
        f"  Model        : {config['model']['primary']}",
        f"  Rebalance    : every {config['walk_forward']['rebalance_every']} days",
        f"  Vol target   : {config['portfolio']['volatility_target']:.0%}",
        "",
        "-" * 72,
        "  STRATEGY PERFORMANCE (NET OF COSTS)",
        "-" * 72,
    ]

    for k, v in strategy_metrics.items():
        lines.append(f"  {k:<28s} : {v}")

    lines += [
        "",
        "-" * 72,
        "  BENCHMARK PERFORMANCE (SPY BUY & HOLD)",
        "-" * 72,
    ]
    for k, v in benchmark_metrics.items():
        lines.append(f"  {k:<28s} : {v}")

    if fold_metrics:
        avg_ic = np.nanmean([m.get("ic", np.nan) for m in fold_metrics])
        avg_hit = np.nanmean([m.get("hit_rate", np.nan) for m in fold_metrics])
        lines += [
            "",
            "-" * 72,
            "  MODEL DIAGNOSTICS (WALK-FORWARD)",
            "-" * 72,
            f"  Total folds          : {len(fold_metrics)}",
            f"  Avg fold IC          : {avg_ic:.4f}",
            f"  Avg fold hit rate    : {avg_hit:.4f}",
        ]

        if ic_stability:
            lines += [
                f"  Mean Daily IC        : {ic_stability.get('mean_ic', 0):.4f}",
                f"  Daily IC IR          : {ic_stability.get('ic_ir', 0):.4f}",
                f"  Daily IC t-stat      : {ic_stability.get('ic_t_stat', 0):.4f}",
                f"  In-sample periods    : {ic_stability.get('n_periods', 0)}",
            ]

    if feature_importance is not None and not feature_importance.empty:
        lines += [
            "",
            "-" * 72,
            "  TOP 10 FEATURES (by importance)",
            "-" * 72,
        ]
        for feat, imp in feature_importance.head(10).items():
            lines.append(f"  {feat:<28s} : {imp:.6f}")

    lines += ["", "=" * 72, ""]

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Report saved to %s", report_path)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Annualised information ratio."""
    excess = strategy_returns.subtract(
        benchmark_returns.reindex(strategy_returns.index), fill_value=0
    )
    te = excess.std() * np.sqrt(TRADING_DAYS)
    if te <= 0:
        return 0.0
    return float(excess.mean() * TRADING_DAYS / te)
