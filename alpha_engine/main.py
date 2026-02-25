"""
Alpha Engine — main entry point and orchestrator.

Run via CLI::

    python -m alpha_engine.main --config alpha_engine/config/config.yaml

The pipeline:
    1. Load configuration
    2. Download / load OHLCV data + benchmark
    3. Engineer alpha features & compute targets
    4. Run walk-forward expanding-window backtest
    5. Construct portfolio (long-short, vol-targeted, net of costs)
    6. Detect volatility regimes
    7. Compute performance analytics & generate report
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# ---- internal imports ------------------------------------------------ #
from alpha_engine.data.data_loader import get_benchmark, load_universe
from alpha_engine.features.feature_engineering import (
    compute_all_features,
    compute_target_rank,
)
from alpha_engine.backtest.walk_forward import run_walk_forward
from alpha_engine.portfolio.portfolio_construction import construct_portfolio
from alpha_engine.risk.risk_management import (
    apply_regime_filter,
    check_drawdown_breach,
    compute_risk_metrics,
    compute_risk_off_signal,
    detect_volatility_regime,
)
from alpha_engine.analytics.performance_metrics import (
    compare_vs_benchmark,
    compute_performance,
    evaluate_signal_decay,
    feature_importance_stability,
    generate_report,
    plot_drawdown,
    plot_equity_curves,
    plot_feature_importance,
    plot_rolling_sharpe,
)
from alpha_engine.analytics.stress_testing import (
    compute_beta_leakage,
    run_monte_carlo,
    run_walk_forward_analysis,
)


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha Engine - ML Multi-Factor Equity Alpha Engine",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="alpha_engine/config/config.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Pipeline                                                            #
# ------------------------------------------------------------------ #


def run_pipeline(config: Dict[str, Any], config_path: str = "Config"):
    """Orchestrate the full Alpha Engine pipeline."""
    t0 = time.time()

    # Logging is set up by checking config or caller
    if not logging.getLogger().handlers:
        _setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Alpha Engine v2.0 starting ...")
    logger.info("Config Details: %s", config_path)

    try:
        # ---- 1. Data ------------------------------------------------- #
        logger.info("=" * 60)
        logger.info("STEP 1 / 7 - Loading data")
        logger.info("=" * 60)

        universe = load_universe(config)
        benchmark_df = get_benchmark(config)
        benchmark_ret = benchmark_df["Adj Close"].pct_change().dropna()

        logger.info(
            "Loaded %d tickers + benchmark (%s)",
            len(universe),
            config["universe"]["benchmark"],
        )

        # ---- 2. Features --------------------------------------------- #
        logger.info("=" * 60)
        logger.info("STEP 2 / 7 - Engineering features (v2.0 Cross-Sectional)")
        logger.info("=" * 60)
        
        # Checkpoint logic
        parquet_dir = Path(config["data"].get("parquet_dir", "data/parquet"))
        checkpoint_path = parquet_dir / "features_v2_checkpoint.parquet"
        
        if checkpoint_path.exists():
            logger.info("Found feature checkpoint: %s. Loading...", checkpoint_path)
            panel_features = pd.read_parquet(checkpoint_path)
            # Convert panel back to Dict[ticker, DataFrame] for downstream compatibility
            features = {
                ticker: group.droplevel("ticker") 
                for ticker, group in panel_features.groupby(level="ticker")
            }
        else:
            features = compute_all_features(universe, benchmark_df, config)
            # Save checkpoint
            if features:
                logger.info("Saving feature checkpoint to %s", checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                # Convert to panel for saving
                dfs = []
                for t, df in features.items():
                    d = df.copy()
                    d["ticker"] = t
                    dfs.append(d.set_index("ticker", append=True).swaplevel())
                panel_features = pd.concat(dfs).sort_index()
                panel_features.to_parquet(checkpoint_path)

        horizon = config.get("target", {}).get("horizon", 21)

        # Use simple forward return target for Ranking (LambdaRank)
        # The ranking happens inside the model training (grouping by date)
        logger.info("Computing forward return targets (horizon=%dd)...", horizon)
        targets: Dict[str, pd.Series] = {}
        for ticker, df in universe.items():
            if ticker in features:
                targets[ticker] = compute_target_rank(
                    price_series=df["Adj Close"],
                    horizon=horizon,
                )

        logger.info(
            "Computed features & targets for %d tickers",
            len(features),
        )

        # ---- 3. Walk-forward backtest -------------------------------- #
        logger.info("=" * 60)
        logger.info("STEP 3 / 7 — Walk-forward backtest")
        logger.info("=" * 60)

        wf_result = run_walk_forward(features, targets, config)

        if not wf_result.predictions:
            logger.error("Walk-forward produced no predictions — aborting")
            return

        # ---- 4. Portfolio construction -------------------------------- #
        logger.info("=" * 60)
        logger.info("STEP 4 / 7 — Portfolio construction")
        logger.info("=" * 60)

        port_result = construct_portfolio(
            predictions=wf_result.predictions,
            realised=wf_result.realised,
            universe=universe,
            config=config,
            rebalance_dates=wf_result.rebalance_dates,
        )

        if port_result.gross_returns.empty:
            logger.error("Portfolio returned no data — aborting")
            return

        # ---- 5. Risk ------------------------------------------------- #
        logger.info("=" * 60)
        logger.info("STEP 5 / 7 — Risk analysis")
        logger.info("=" * 60)

        risk_cfg = config.get("risk", {})
        regimes = detect_volatility_regime(
            port_result.net_returns,
            n_regimes=risk_cfg.get("n_regimes", 3),
            vol_window=risk_cfg.get("regime_vol_window", 60),
        )

        risk_metrics = compute_risk_metrics(port_result.net_returns)
        logger.info("Risk metrics: %s", risk_metrics)

        if check_drawdown_breach(
            port_result.net_returns,
            risk_cfg.get("max_drawdown_threshold", 0.20),
        ):
            logger.warning("⚠ Max drawdown threshold breached!")

        # ---- 5a. Global Risk-Off Filter (v2.0) ----------------------- #
        strategy_returns = port_result.net_returns  # baseline (unfiltered)
        if risk_cfg.get("regime_filter_enabled", False):
            logger.info("Computing Global Risk-Off signal ...")
            risk_off = compute_risk_off_signal(
                port_result.net_returns,
                vol_window=risk_cfg.get("regime_vol_window", 60),
                vol_percentile_threshold=risk_cfg.get("vol_percentile_threshold", 90),
                sma_slope_window=risk_cfg.get("sma_slope_window", 200),
            )
            strategy_returns = apply_regime_filter(port_result.net_returns, risk_off)
            logger.info("Filtered strategy returns available")
        else:
            strategy_returns = port_result.net_returns

        # ---- 5b. Stress Testing (v2.0) ------------------------------- #
        logger.info("=" * 60)
        logger.info("STEP 5b — Stress testing & robustness")
        logger.info("=" * 60)

        # Align benchmark returns to strategy period (needed for beta leakage)
        strat_dates = strategy_returns.index
        bench_aligned = benchmark_ret.reindex(strat_dates).fillna(0)

        wf_analysis = run_walk_forward_analysis(strategy_returns, n_stages=5)
        mc_result = run_monte_carlo(strategy_returns, n_simulations=2000)
        beta_result = compute_beta_leakage(strategy_returns, bench_aligned)

        # ---- 6. Performance analytics -------------------------------- #
        logger.info("=" * 60)
        logger.info("STEP 6 / 7 — Performance analytics")
        logger.info("=" * 60)

        comparison = compare_vs_benchmark(strategy_returns, bench_aligned)

        logger.info("Strategy (net): %s", comparison["strategy"])
        logger.info("Benchmark     : %s", comparison["benchmark"])

        # Turnover
        if not port_result.turnover_series.empty:
            avg_turnover = float(port_result.turnover_series.mean())
            logger.info("Avg one-way turnover per rebalance: %.2f%%", avg_turnover * 100)
            comparison["strategy"]["avg_turnover"] = round(avg_turnover, 4)

        # Signal decay
        logger.info("Evaluating signal decay …")
        decay = evaluate_signal_decay(wf_result.predictions, universe)
        if not decay.empty:
            logger.info("\nSignal decay:\n%s", decay.to_string())

        # Feature importance stability
        if wf_result.feature_importance is not None:
            logger.info("\nTop features:\n%s", wf_result.feature_importance.head(15).to_string())

        # v2.0: Information Coefficient (IC) Stability
        logger.info("Computing Information Coefficient (IC) stability ...")
        from alpha_engine.analytics.performance_metrics import compute_daily_ic, evaluate_ic_stability
        daily_ic = compute_daily_ic(wf_result.predictions, wf_result.realised)
        ic_stability = evaluate_ic_stability(daily_ic)
        logger.info("IC Stability: %s", ic_stability)

        # ---- 7. Report & plots --------------------------------------- #
        logger.info("=" * 60)
        logger.info("STEP 7 / 7 — Generating report & plots")
        logger.info("=" * 60)

        results_dir = Path(config["output"]["results_dir"])
        plots_dir = Path(config["output"]["plots_dir"])

        generate_report(
            strategy_metrics=comparison["strategy"],
            benchmark_metrics=comparison["benchmark"],
            fold_metrics=wf_result.fold_metrics,
            feature_importance=wf_result.feature_importance,
            config=config,
            output_dir=results_dir,
            ic_stability=ic_stability,
        )

        # Save daily IC
        if not daily_ic.empty:
            daily_ic.to_csv(results_dir / "daily_ic.csv")

        plot_equity_curves(
            port_result.gross_returns,
            bench_aligned,
            plots_dir / "equity_curves.png",
            net_returns=strategy_returns,
        )
        plot_drawdown(strategy_returns, plots_dir / "drawdown.png")
        plot_rolling_sharpe(strategy_returns, plots_dir / "rolling_sharpe.png")

        # Save stress test results
        if not wf_analysis.empty:
            wf_analysis.to_csv(results_dir / "walk_forward_analysis.csv", index=False)
        if mc_result:
            import json
            with open(results_dir / "monte_carlo.json", "w") as f_mc:
                json.dump(mc_result, f_mc, indent=2)
        if beta_result:
            pd.Series(beta_result).to_csv(results_dir / "beta_leakage.csv")

        if wf_result.feature_importance is not None:
            plot_feature_importance(
                wf_result.feature_importance, plots_dir / "feature_importance.png"
            )

        # Save fold metrics
        if wf_result.fold_metrics:
            fold_df = pd.DataFrame(wf_result.fold_metrics)
            fold_df.to_csv(results_dir / "fold_metrics.csv", index=False)

        # Save signal decay
        if not decay.empty:
            decay.to_csv(results_dir / "signal_decay.csv")

        # Save regime series
        regimes.to_csv(results_dir / "regimes.csv")

    except Exception:
        logger.exception("Pipeline failed with an unhandled exception")
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("Results -> %s", results_dir.resolve())
    logger.info("Plots   -> %s", plots_dir.resolve())
    logger.info("=" * 60)


# ------------------------------------------------------------------ #
#  Logging setup                                                       #
# ------------------------------------------------------------------ #


def _setup_logging(config: Dict[str, Any]) -> None:
    """Configure structured logging from config."""
    import io

    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "output/alpha_engine.log")

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reconfigure stdout to UTF-8 to avoid cp1252 issues on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)

    logging.basicConfig(
        level=level,
        handlers=[stream_handler, file_handler],
        force=True,
    )


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Alpha Engine v2.0 starting ...")
    logger.info("Config Path: %s", config_path.resolve())

    try:
        run_pipeline(config, str(config_path))
    except Exception:
        logger.exception("Pipeline failed with an unhandled exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
