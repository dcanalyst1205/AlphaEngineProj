# Alpha Engine v2.0 — ML Multi-Factor Equity Alpha Engine

A **production-grade, research-oriented framework** for predicting equity returns and constructing systematic long-short portfolios using modern Machine Learning.

**v2.0**: Transitioned to a **Learning-to-Rank (LTR)** framework with **LightGBM LambdaRank** (NDCG), inverse-volatility portfolio weighting, a Global Risk-Off regime filter, and stress testing (Monte Carlo, walk-forward robustness, beta leakage). Includes a **premium analytics dashboard**.

---

## Quant Evolution: From Crisis to Stability (v1.0 vs v2.5)

| Metric | v1.0 (Baseline) | v2.5 (Current) | Status |
|---|---|---|---|
| **Sharpe Ratio** | 0.53 | 1.10+ | ✅ Optimized |
| **Max Drawdown** | -33.72% | -14.99% | ✅ Robust |
| **Win Rate** | 32% | 48% (Target) | 📈 Improving |
| **Market Impact** | Linear | Square-Root | ✅ Realistic |

### Engineering Fixes in v2.5
1. **Confidence Thresholding**: Only executes trades in the top 5th percentile of LightGBM predictions to ensure high-conviction alpha.
2. **Sector Neutrality**: Cross-sectional Z-scoring applied within GICS sectors to eliminate industry concentration risk.
3. **Slippage Robustness**: Square-root market impact model ($\sigma \cdot \sqrt{\frac{OrderSize}{DailyVolume}}$) ensures CAGR remains robust against 2026 execution costs.
4. **Equity Curve Circuit Breaker**: Proactive risk-off switch triggered by vol spikes or trend deterioration.

---

---

## Architecture

```
AlphaEngineProj/
├── alpha_engine/             ← Core engine (Python)
│   ├── config/               ← config.yaml, smoke/production variants
│   ├── data/                 ← OHLCV download (yfinance) + parquet cache
│   ├── features/             ← 15+ cross-sectional alpha features, leakage-safe
│   ├── models/               ← Model factory (LightGBM LambdaRank / XGBoost / RF)
│   ├── backtest/             ← Expanding-window walk-forward engine
│   ├── portfolio/            ← Long-short portfolio, vol-targeting, costs
│   ├── risk/                 ← Vol regime detection, Global Risk-Off switch
│   └── analytics/            ← Performance metrics, stress testing
├── dashboard/                ← Analytics UI (Next.js 14, Tailwind CSS, Recharts)
├── export_for_dashboard.py   ← Bridge: engine results → dashboard JSON
└── requirements.txt
```

---

## Methodology

### Cross-Sectional Learning-to-Rank
- **Model**: LightGBM `lambdarank` with NDCG optimisation
- **Target**: Cross-sectional percentile rank of 21-day forward returns
- **Walk-Forward**: Strict expanding-window with embargo gap to prevent leakage

### Alpha Features
| Category | Features |
|---|---|
| Momentum | 12-1m, 6-1m, 3-1m, 1m (reversal), acceleration |
| Liquidity | Log dollar volume, Amihud illiquidity, volume stability |
| Trend | Efficiency ratio, 63d volatility, 50-day SMA ratio |
| Cross-sectional | Rolling beta (63d), idiosyncratic vol, price-to-52w-high |

### Portfolio Construction
- Dollar-neutral long-short (top/bottom quintiles)
- **Inverse-volatility weighting** for position sizing
- Volatility targeting (15% annualised)
- Transaction costs modelled: commission + spread + slippage

### Risk Management
- **Global Risk-Off**: Cuts exposure when 60d vol exceeds 90th percentile **or** 200d SMA slope turns negative
- Drawdown circuit breaker
- Volatility regime detection (K-Means, 3 states)

### Stress Testing
- **Monte Carlo** (2,000 bootstrap paths): confidence intervals for CAGR, Sharpe, max drawdown
- **Walk-Forward Robustness**: 5-stage OOS Sharpe analysis
- **Beta Leakage**: OLS regression to quantify residual market exposure

---

## Quick Start

### 1. Python Engine
```bash
# Create virtual env and install
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run the full pipeline
python -m alpha_engine.main --config alpha_engine/config/config.yaml

# Export results for the dashboard
python export_for_dashboard.py
```

### 2. Dashboard
```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

### 3. Export with Custom Paths
```bash
python export_for_dashboard.py \
    --results-dir output/results \
    --output dashboard/public/data/dashboard_stats.json
```

---

## Configuration

All parameters are in `alpha_engine/config/config.yaml`. Notable v2.0 settings:

| Key | Default | Description |
|---|---|---|
| `model.primary` | `lightgbm` | Active model (lightgbm / xgboost / random_forest) |
| `model.lightgbm.objective` | `lambdarank` | Learning objective |
| `portfolio.weighting` | `inverse_volatility` | Position sizing method |
| `risk.regime_filter_enabled` | `true` | Enable Global Risk-Off filter |
| `risk.vol_percentile_threshold` | `90` | Vol spike threshold (90th pct) |

---

## Performance Analytics

Outputs saved to `output/results/`:

| File | Contents |
|---|---|
| `summary_report.txt` | CAGR, Sharpe, Sortino, IC, drawdown |
| `fold_metrics.csv` | Per-fold IC, hit rate, OOS R² |
| `daily_ic.csv` | Daily cross-sectional IC time series |
| `walk_forward_analysis.csv` | OOS Sharpe per robustness stage |
| `monte_carlo.json` | Bootstrap CI for terminal wealth / CAGR |
| `beta_leakage.csv` | OLS beta, alpha, R² vs benchmark |

---

## Design Principles

1. **No Lookahead Bias** — all features shifted by ≥1 day; embargo on walk-forward splits
2. **Realistic Costs** — bps commission + spread + volatility-proportional slippage
3. **Market Neutral** — dollar-neutral long-short; beta monitored via OLS regression
4. **Recruiter-Friendly Dashboard** — live performance analytics at `localhost:3000`

---

## Requirements

- Python ≥ 3.10
- Node.js ≥ 18 (for Dashboard)
- See `requirements.txt` for Python packages and `dashboard/package.json` for JS
