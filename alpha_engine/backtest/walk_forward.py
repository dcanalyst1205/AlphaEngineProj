"""
Walk-Forward Engine — expanding-window time-series validation.

Implements strict temporal walk-forward backtesting:
- Training window expands from ``min_train_days`` to the current
  rebalance date.
- An embargo gap separates train from test to prevent information
  leakage from overlapping target windows.
- Predictions are collected for each out-of-sample period and
  returned alongside realised returns and feature importances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from alpha_engine.models.train_model import (
    evaluate_model,
    get_feature_importance,
    predict,
    train_model,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Result container                                                    #
# ------------------------------------------------------------------ #


@dataclass
class WalkForwardResult:
    """Container for walk-forward backtest outputs."""

    # Ticker → DatetimeIndex → predicted return
    predictions: Dict[str, pd.Series] = field(default_factory=dict)
    # Ticker → DatetimeIndex → realised forward return
    realised: Dict[str, pd.Series] = field(default_factory=dict)
    # Per-fold evaluation metrics
    fold_metrics: List[Dict[str, Any]] = field(default_factory=list)
    # Aggregated feature importance (averaged across folds)
    feature_importance: Optional[pd.Series] = None
    # Rebalance dates used
    rebalance_dates: List[pd.Timestamp] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #


def run_walk_forward(
    features: Dict[str, pd.DataFrame],
    targets: Dict[str, pd.Series],
    config: Dict[str, Any],
) -> WalkForwardResult:
    """Execute the full walk-forward backtest.

    Parameters
    ----------
    features : Dict[str, pd.DataFrame]
        Ticker → feature DataFrame.
    targets : Dict[str, pd.Series]
        Ticker → forward-return Series.
    config : dict
        Full config dictionary.

    Returns
    -------
    WalkForwardResult
        Predictions, realised returns, fold metrics, feature importances.
    """
    wf_cfg = config["walk_forward"]
    model_cfg = config["model"]
    model_type: str = model_cfg["primary"]
    model_params: Dict[str, Any] = model_cfg.get(model_type, {})
    min_train = wf_cfg["min_train_days"]
    rebalance_every = wf_cfg["rebalance_every"]
    embargo = wf_cfg["embargo_days"]

    # --- Build pooled panel ---------------------------------------- #
    panel_X, panel_y, panel_ticker = _build_pooled_panel(features, targets)
    if panel_X.empty:
        logger.error("Empty panel after pooling — check data")
        return WalkForwardResult()

    all_dates = panel_X.index.get_level_values("date").unique().sort_values()
    logger.info(
        "Panel: %d rows, %d features, date range %s → %s",
        len(panel_X),
        panel_X.shape[1],
        all_dates[0].date(),
        all_dates[-1].date(),
    )

    # --- Determine rebalance schedule ------------------------------ #
    rebalance_dates = all_dates[min_train::rebalance_every]
    logger.info("Walk-forward: %d rebalance dates", len(rebalance_dates))

    result = WalkForwardResult(rebalance_dates=list(rebalance_dates))
    importances_list: List[pd.Series] = []

    for i, reb_date in enumerate(rebalance_dates):
        # ---- Train / test split ---------------------------------- #
        train_end = reb_date
        test_start_idx = all_dates.get_loc(reb_date) + embargo
        if test_start_idx >= len(all_dates):
            break
        test_start = all_dates[test_start_idx]

        test_end_idx = min(
            test_start_idx + rebalance_every - 1, len(all_dates) - 1
        )
        test_end = all_dates[test_end_idx]

        train_mask = panel_X.index.get_level_values("date") <= train_end
        test_mask = (
            (panel_X.index.get_level_values("date") >= test_start)
            & (panel_X.index.get_level_values("date") <= test_end)
        )

        X_train = panel_X.loc[train_mask]
        y_train = panel_y.loc[train_mask]
        X_test = panel_X.loc[test_mask]
        y_test = panel_y.loc[test_mask]

        if len(X_train) < 100 or len(X_test) == 0:
            continue

        # ---- 1. Sort by Date for LambdaRank (Query = Date) ----------- #
        # LightGBM LambdaRank requires usage of groups, and groups must be
        # contiguous. We group by Date.
        
        X_train = X_train.sort_index(level="date")
        y_train = y_train.reindex(X_train.index)
        
        X_test = X_test.sort_index(level="date")
        y_test = y_test.reindex(X_test.index)

        # ---- 2. Cross-Sectional Target Ranking -------------------- #
        # v2.0: Convert raw forward returns into 10 relevance levels (0-10)
        # per day. This makes the target suitable for LambdaRank (LTR).
        y_train_rank = (y_train.groupby(level="date").rank(pct=True) * 10).astype(int)
        # We don't necessarily rank y_test for evaluation (we'll compute IC vs raw),
        # but the model is trained on ranks.
        
        train_groups = X_train.index.get_level_values("date")
        
        # ---- 3. Train -------------------------------------------- #
        model = train_model(
            X_train, 
            y_train_rank,  # Pass ranked targets
            config, 
            groups=train_groups
        )
        y_pred = predict(model, X_test)

        # ---- Collect predictions per ticker ----------------------- #
        test_idx = X_test.index
        pred_series = pd.Series(y_pred, index=test_idx, name="prediction")

        for ticker in test_idx.get_level_values("ticker").unique():
            ticker_mask = test_idx.get_level_values("ticker") == ticker
            dates = test_idx.get_level_values("date")[ticker_mask]
            preds = pred_series.loc[ticker_mask].values
            reals = y_test.loc[ticker_mask].values

            if ticker not in result.predictions:
                result.predictions[ticker] = pd.Series(dtype=float)
                result.realised[ticker] = pd.Series(dtype=float)

            result.predictions[ticker] = pd.concat([
                result.predictions[ticker],
                pd.Series(preds, index=dates),
            ])
            result.realised[ticker] = pd.concat([
                result.realised[ticker],
                pd.Series(reals, index=dates),
            ])

        # ---- Fold metrics ----------------------------------------- #
        metrics = evaluate_model(y_test, y_pred)
        metrics["fold"] = i
        metrics["train_end"] = str(train_end.date())
        metrics["test_start"] = str(test_start.date())
        metrics["test_end"] = str(test_end.date())
        metrics["n_train"] = len(X_train)
        metrics["n_test"] = len(X_test)
        result.fold_metrics.append(metrics)

        # ---- Feature importance ----------------------------------- #
        imp = get_feature_importance(model, list(X_train.columns))
        importances_list.append(imp)

        if (i + 1) % 10 == 0 or i == len(rebalance_dates) - 1:
            logger.info(
                "Fold %d/%d | train→%s | test %s→%s | IC=%.4f",
                i + 1,
                len(rebalance_dates),
                train_end.date(),
                test_start.date(),
                test_end.date(),
                metrics.get("ic", float("nan")),
            )

    # ---- Aggregate feature importance ----------------------------- #
    if importances_list:
        result.feature_importance = (
            pd.concat(importances_list, axis=1)
            .mean(axis=1)
            .sort_values(ascending=False)
        )

    # ---- Summary -------------------------------------------------- #
    if result.fold_metrics:
        avg_ic = np.nanmean([m["ic"] for m in result.fold_metrics])
        avg_hit = np.nanmean([m["hit_rate"] for m in result.fold_metrics])
        logger.info(
            "Walk-forward complete: %d folds, avg IC=%.4f, avg hit=%.3f",
            len(result.fold_metrics),
            avg_ic,
            avg_hit,
        )

    return result


# ------------------------------------------------------------------ #
#  Panel construction                                                  #
# ------------------------------------------------------------------ #


def _build_pooled_panel(
    features: Dict[str, pd.DataFrame],
    targets: Dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Stack per-ticker feature/target into a pooled panel.

    Returns a MultiIndex (ticker, date) DataFrame for features and
    matching Series for targets.
    """
    rows_X: list[pd.DataFrame] = []
    rows_y: list[pd.Series] = []

    for ticker in features:
        if ticker not in targets:
            continue
        feat = features[ticker]
        tgt = targets[ticker]

        # Align on common dates and drop NaN rows
        common = feat.index.intersection(tgt.index)
        f = feat.loc[common]
        t = tgt.loc[common]

        valid = f.notna().all(axis=1) & t.notna()
        f = f.loc[valid]
        t = t.loc[valid]

        if f.empty:
            continue

        # Add ticker level to index
        idx = pd.MultiIndex.from_arrays(
            [np.full(len(f), ticker), f.index],
            names=["ticker", "date"],
        )
        f.index = idx
        t.index = idx

        rows_X.append(f)
        rows_y.append(t)

    if not rows_X:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=str)

    panel_X = pd.concat(rows_X).sort_index()
    panel_y = pd.concat(rows_y).sort_index()
    panel_ticker = panel_X.index.get_level_values("ticker")

    return panel_X, panel_y, panel_ticker
