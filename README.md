# Alpha Engine — ML Multi-Factor Equity Alpha Engine

A **production-grade, research-oriented framework** for predicting equity returns and
constructing systematic long-short portfolios using machine learning.

Built to resemble institutional quant infrastructure — modular, config-driven,
walk-forward validated, and net-of-costs — not a toy trading bot.

---

## Architecture

```
alpha_engine/
├── config/
│   └── config.yaml              ← All tunable parameters
├── data/
│   └── data_loader.py           ← OHLCV fetch (yfinance) + parquet I/O
├── features/
│   └── feature_engineering.py   ← 30+ alpha features, leakage-safe
├── models/
│   └── train_model.py           ← RF / XGBoost / LightGBM factory
├── backtest/
│   └── walk_forward.py          ← Expanding-window walk-forward engine
├── portfolio/
│   └── portfolio_construction.py ← Long-short, vol-targeting, costs
├── risk/
│   └── risk_management.py       ← Vol regime detection, drawdown controls
├── analytics/
│   └── performance_metrics.py   ← Sharpe, CAGR, drawdowns, plots, report
└── main.py                      ← CLI entry point
```

---

## Methodology

### Data Pipeline
- Daily OHLCV for 100 US equities + SPY benchmark via **yfinance**
- Stored and cached as **Parquet** (pyarrow) for fast reload
- Configurable universe, date range, and benchmark

### Alpha Features (30+)
| Category | Features |
|---|---|
| Momentum | 1m, 3m, 6m log returns; 12m−1m momentum |
| Volatility | 20d/60d realised vol, ATR(14), vol-of-vol |
| Technical | RSI(14), MACD (line/signal/histogram) |
| Volume | Volume z-score, 5d/20d ratio, OBV slope |
| Statistical | Rolling skewness, kurtosis, autocorrelation (5d/20d) |
| Cross-sectional | Beta vs SPY, idiosyncratic vol |
| Regime | Vol-clustering ratio, high-vol flag |
| Mean reversion | Distance from SMA(20/50), Bollinger Band position |

All features are **shifted by 1+ days** to prevent lookahead bias.

### ML Models
- **LightGBM**, **XGBoost**, **RandomForest** — configurable via YAML
- Target: **5-day forward log return**
- Out-of-sample metrics: IC, Rank IC, Hit Rate, R²

### Walk-Forward Validation
- **Expanding training window** (min 2 years)
- **Embargo gap** (5 days) between train/test
- Rebalance every 21 trading days
- **Zero random splits** — strict temporal ordering

### Portfolio Construction
- Rank predictions → **Long top 20% / Short bottom 20%**
- Equal-weight within buckets
- **Volatility targeting** (12% annualised via trailing realised vol)
- **Position caps** (5% per name)
- **Turnover constraint** (60% one-way cap with blending)

### Transaction Cost Model
- Fixed commission: 2 bps
- Spread: 1 bp
- Slippage: 10% × daily volatility (vol-proportional)
- All performance reported **net of costs**

### Risk Management
- **KMeans volatility regime detection** (3 regimes: low/mid/high)
- Max drawdown circuit breaker (20%)
- VaR & CVaR (95%)

### Performance Analytics
- CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Information Ratio, Hit Rate
- SPY buy-and-hold comparison
- Equity curves, drawdown charts, rolling Sharpe, feature importance plots
- Signal decay analysis (1–10 day horizons)
- Feature importance stability across folds

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the engine
```bash
python -m alpha_engine.main --config alpha_engine/config/config.yaml
```

### 3. Check results
```
output/
├── results/
│   ├── summary_report.txt     ← Performance summary
│   ├── fold_metrics.csv       ← Per-fold model diagnostics
│   ├── signal_decay.csv       ← IC at horizons 1–10d
│   └── regimes.csv            ← Vol regime labels
└── plots/
    ├── equity_curves.png      ← Strategy vs benchmark
    ├── drawdown.png           ← Underwater chart
    ├── rolling_sharpe.png     ← Rolling 63d Sharpe
    └── feature_importance.png ← Top 20 features
```

---

## Configuration

All parameters live in `alpha_engine/config/config.yaml`:

| Section | Key parameters |
|---|---|
| `universe` | Ticker list, benchmark, date range |
| `features` | Lookback windows, indicator periods |
| `model` | Model type, hyperparameters |
| `walk_forward` | Min train days, rebalance frequency, embargo |
| `portfolio` | Percentiles, position caps, vol target, turnover limit |
| `costs` | Commission, spread, slippage multiplier |
| `risk` | Drawdown threshold, regime count |

---

## Key Design Principles

1. **No lookahead bias** — features lagged, strict temporal walk-forward, embargo gap
2. **Transaction costs always applied** — commission + spread + vol-proportional slippage
3. **Config-driven** — change any parameter from YAML without touching code
4. **Modular** — every component is a standalone module with typed signatures
5. **Robustness over performance** — realistic assumptions, no cherry-picked results

---

## Limitations

- **Data**: yfinance is non-commercial; institutional work uses Bloomberg/Refinitiv
- **Execution**: no real-time order routing or market impact modeling
- **Universe**: survivorship bias not fully addressed (uses current tickers)
- **Costs**: simplified model (no market impact, borrow costs for shorts)
- **Single asset class**: equities only (no cross-asset signals)

---

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for full dependency list
