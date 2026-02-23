# Alpha Engine v2.0 — ML Multi-Factor Equity Alpha Engine

A **production-grade, research-oriented framework** for predicting equity returns and constructing systematic long-short portfolios using modern Machine Learning.

**v2.0 Update**: Transitioned to a **Learning-to-Rank (LTR)** framework using **LightGBM LambdaRank**, significantly improving cross-sectional IC and alpha stability. Now includes a **premium visual dashboard** for performance analytics.

---

## Architecture

```
AlphaEngineProj/
├── alpha_engine/             ← Core engine (Python)
│   ├── config/               ← All tunable parameters (.yaml)
│   ├── data/                 ← OHLCV fetch (yfinance) + parquet I/O
│   ├── features/             ← 30+ alpha features, leakage-safe
│   ├── models/               ← LTR (LightGBM) & Regression factory
│   ├── backtest/             ← Cross-sectional walk-forward engine
│   ├── portfolio/            ← Long-short, vol-targeting, costs
│   ├── risk/                 ← Vol regime detection, drawdown controls
│   └── analytics/            ← Performance metrics (Sharpe, IC, NDCG)
├── dashboard/                ← Premium Analytics UI (Next.js, Tailwind)
├── export_for_dashboard.py   ← Bridge: Engine results → Dashboard JSON
└── main.py                   ← CLI entry point
```

---

## Methodology (v2.0: Learning-to-Rank)

### Cross-Sectional Ranking
Unlike v1.0 which used point-wise regression, v2.0 utilizes a **Learning-to-Rank** framework.
- **Model**: LightGBM `lambdarank` (NDCG optimized).
- **Target**: Cross-sectional ranking of 5-day forward returns.
- **Advantage**: Better captures the relative performance of stocks within a universe, which is the primary driver of long-short alpha.

### Alpha Features (30+)
| Category | Features |
|---|---|
| Momentum | 1m, 3m, 6m log returns; 12m−1m momentum |
| Volatility | 20d/60d realised vol, ATR(14), vol-of-vol |
| Technical | RSI(14), MACD (line/signal/histogram) |
| Volume | Volume z-score, 5d/20d ratio, OBV slope |
| Statistical | Rolling skewness, kurtosis, autocorrelation (5d/20d) |
| Cross-sectional | Beta vs SPY, idiosyncratic vol, cross-sectional ranking |
| Regime | Vol-clustering ratio, KMeans vol-regime detection |
| Mean reversion | Distance from SMA(20/50), Bollinger Band position |

---

## Visual Analytics (Next.js Dashboard)

Alpha Engine now includes a **premium, responsive dashboard** built with **Next.js 14** and **Tailwind CSS**.

- **Modern UI**: Sleek dark mode, glassmorphism, and responsive charts.
- **Key Metrics**: CAGR, Sharpe, Max Drawdown, and Hit Rate.
- **Interactive Charts**: Equity curves (Strategy vs Benchmark), Drawdown charts, and Feature Importance.
- **Live Bridge**: Run `python export_for_dashboard.py` to sync engine results with the UI.

---

## Quick Start

### 1. Core Engine (Python)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the engine
python -m alpha_engine.main --config alpha_engine/config/config.yaml

# Export results for dashboard
python export_for_dashboard.py
```

### 2. Dashboard (Next.js)
```bash
cd dashboard
npm install
npm run dev
# Go to http://localhost:3000
```

---

## Configuration

All parameters live in `alpha_engine/config/config.yaml`. Key v2.0 updates:
- `model.primary`: `lightgbm` with `objective: "lambdarank"`
- `model.lightgbm.metric`: `ndcg`
- `portfolio.long_percentile`: Configurable for top/bottom buckets.

---

## Performance Analytics

- **Metrics**: CAGR, Sharpe, Sortino, Max Drawdown, Information Ratio, Hit Rate.
- **Rank IC**: Daily Information Coefficient stability.
- **NDCG**: Normalized Discounted Cumulative Gain for ranking quality.
- **Signal Decay**: IC decay across 1–10 day horizons.

---

## Key Design Principles

1. **No Lookahead Bias**: Shifted features, temporal splits, embargo gaps.
2. **Institutional Methodology**: Transaction costs (bps + slippage) applied.
3. **Cross-Sectional Focus**: Optimized for relative stock selection.
4. **Visual Excellence**: Professional-grade dashboard for research presentation.

---

## Requirements

- Python ≥ 3.10
- Node.js ≥ 18.x (for Dashboard)
- See `requirements.txt` and `dashboard/package.json`
