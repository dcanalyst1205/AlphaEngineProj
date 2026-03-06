import traceback
import sys
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import os
import shutil

# Set PYTHONPATH
sys.path.append(os.getcwd())

# Configuration setup
with open("alpha_engine/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# USE TEMPORARY DIRECTORY TO FORCE DOWNLOAD OF 2026 DATA
qa_data_dir = Path("tmp/qa_data")
if qa_data_dir.exists():
    shutil.rmtree(qa_data_dir)
qa_data_dir.mkdir(parents=True, exist_ok=True)

config["data"]["parquet_dir"] = str(qa_data_dir)
config["universe"]["start_date"] = "2025-06-01"
config["universe"]["end_date"] = "2026-03-05"
config["walk_forward"]["min_train_days"] = 90 # ~3 months training
config["walk_forward"]["rebalance_every"] = 10 # More rebalances for speedier verification
config["walk_forward"]["embargo_days"] = 2    # Minimal embargo for QA

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QA")

from alpha_engine.data.data_loader import load_universe, get_benchmark
from alpha_engine.features.feature_engineering import compute_all_features, compute_target_rank
from alpha_engine.backtest.walk_forward import run_walk_forward
from alpha_engine.portfolio.portfolio_construction import construct_portfolio
from alpha_engine.risk.risk_management import compute_risk_metrics

try:
    print("Step 1: Downloading fresh data (including 2026)...")
    universe = load_universe(config)
    benchmark = get_benchmark(config)
    
    # Check max date
    max_date = pd.Timestamp("1900-01-01")
    for t, df in universe.items():
        if not df.empty:
            max_date = max(max_date, df.index[-1])
    print(f"Data audit: Max date in universe is {max_date.date()}")
    
    if max_date < pd.Timestamp("2026-03-01"):
        print("WARNING: Data still does not reach March 2026. Check yfinance connectivity.")

    print("Step 2: Computing Features & Targets...")
    feats = compute_all_features(universe, benchmark, config)
    targets = {}
    for ticker, df in universe.items():
        if ticker in feats:
            targets[ticker] = compute_target_rank(df["Adj Close"], config["target"]["horizon"])

    print("Step 3: Running Walk-Forward...")
    wf_res = run_walk_forward(feats, targets, config)
    
    print("Step 4: Constructing Portfolio...")
    port_res = construct_portfolio(wf_res.predictions, wf_res.realised, universe, config, wf_res.rebalance_dates)

    print("\n--- Neutrality Check ---")
    from alpha_engine.features.feature_engineering import GICS_SECTOR_MAP
    if port_res.weights_history:
        last_date = sorted(port_res.weights_history.keys())[-1]
        last_weights = port_res.weights_history[last_date]
        sector_exposure = {}
        for t, w in last_weights.items():
            sec = GICS_SECTOR_MAP.get(t, "Other")
            sector_exposure[sec] = sector_exposure.get(sec, 0.0) + w

        neutrality_passed = True
        for sec, exp in sector_exposure.items():
            if abs(exp) > 0.05:
                print(f"FLAG! Sector {sec} exposure is {exp*100:.2f}% (Limit is 5%)")
                neutrality_passed = False
        if neutrality_passed:
            print("Neutrality check passed: All sector net exposures < 5%.")

    from alpha_engine.analytics.performance_metrics import compute_performance
    strat_net = port_res.net_returns
    if not strat_net.empty:
        print(f"Backtest result dates: {strat_net.index[0].date()} to {strat_net.index[-1].date()}")
        strat_2026 = strat_net[strat_net.index >= "2026-01-01"]
        if not strat_2026.empty:
            risk = compute_risk_metrics(strat_2026)
            perf = compute_performance(strat_2026)
            win_rate = (strat_2026 > 0).mean() * 100
            sharpe = perf.get("sharpe", 0.0)
            max_dd = risk.get("max_drawdown", 0.0) * 100
            t_2026 = port_res.turnover_series[port_res.turnover_series.index >= "2026-01-01"]
            turnover = t_2026.mean() * 100 if not t_2026.empty else 0.0

            print("\n=== SUMMARY METRICS (Jan-March 2026) ===")
            print(f"New Win Rate (%): {win_rate:.2f}%")
            print(f"New Sharpe Ratio: {sharpe:.2f}")
            print(f"Max Drawdown (%): {max_dd:.2f}%")
            print(f"Average Daily Turnover: {turnover:.2f}%")
        else:
            print("No 2026 data in result. Adjusting rebalance dates might be needed.")
    else:
        print("No strategy returns generated.")

except Exception:
    traceback.print_exc()
finally:
    # Cleanup
    # shutil.rmtree(qa_data_dir) 
    pass
