"""
Model Training — factory for RandomForest, XGBoost, and LightGBM.

Provides a unified interface for training, prediction, feature
importance extraction, and out-of-sample evaluation of regression models
used for alpha prediction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Type alias for any of the supported models
ModelType = Any


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
    groups: Optional[pd.Series] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    val_groups: Optional[pd.Series] = None,
) -> Any:
    """Train a model (LightGBM LambdaRank preferred)."""
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("primary", "lightgbm")
    params = model_cfg.get(model_type, {})

    if model_type == "lightgbm":
        return _train_lightgbm(
            X_train, y_train, params, groups, X_val, y_val, val_groups
        )
    elif model_type == "xgboost":
        return _train_xgboost(X_train, y_train, params)
    elif model_type == "random_forest":
        return _train_rf(X_train, y_train, params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Generate predictions."""
    if hasattr(model, "predict"):
        return model.predict(X)
    return np.zeros(len(X))


def get_feature_importance(model: Any, flist: List[str]) -> pd.Series:
    """Extract feature importance."""
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "feature_importance"):
        # LightGBM Booster
        imps = model.feature_importance(importance_type="gain")
    else:
        return pd.Series(0, index=flist)
        
    return pd.Series(imps, index=flist).sort_values(ascending=False)


def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute out-of-sample evaluation metrics.

    Metrics
    -------
    - IC (information coefficient) : Pearson correlation
    - Rank IC : Spearman rank correlation
    - Hit rate : fraction where sign(pred) == sign(actual)
    - OOS R2 : out-of-sample R-squared
    - RMSE : root mean squared error
    """
    y_t = y_true.values if isinstance(y_true, pd.Series) else y_true
    mask = np.isfinite(y_t) & np.isfinite(y_pred)
    y_t = y_t[mask]
    y_p = y_pred[mask]

    if len(y_t) < 10:
        return {"ic": np.nan, "rank_ic": np.nan, "hit_rate": np.nan,
                "r2": np.nan, "rmse": np.nan}

    ic = float(np.corrcoef(y_t, y_p)[0, 1])

    # Rank IC (Spearman)
    from scipy.stats import spearmanr
    rank_ic, _ = spearmanr(y_t, y_p)

    # Hit rate
    hit = float(np.mean(np.sign(y_t) == np.sign(y_p)))

    # R2 and RMSE
    r2 = float(r2_score(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    return {
        "ic": round(ic, 6),
        "rank_ic": round(float(rank_ic), 6),
        "hit_rate": round(hit, 4),
        "r2": round(r2, 6),
        "rmse": round(rmse, 8),
    }


# ------------------------------------------------------------------ #
#  Internal Training Logic                                             #
# ------------------------------------------------------------------ #


def _train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    groups: Optional[pd.Series],
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    val_groups: pd.Series = None,
) -> Any:
    """Train LightGBM with LambdaRank support."""
    import lightgbm as lgb
    
    # 1. Prepare Datasets
    # LightGBM requires data to be sorted by group for LambdaRank
    pass_groups = False
    val_group_sizes = None  # initialised here to prevent UnboundLocalError
    objective = params.get("objective", "regression")

    if objective == "lambdarank":
        if groups is None:
            raise ValueError("LambdaRank requires 'groups' (e.g. Date)")

        # LightGBM expects a list of group sizes (counts of items per group).
        # Data must be sorted by group (date) before calling.
        train_group_sizes = pd.Series(groups).groupby(groups, sort=False).count().values
        pass_groups = True

        # Validation groups
        if X_val is not None and val_groups is not None:
            val_group_sizes = pd.Series(val_groups).groupby(val_groups, sort=False).count().values
            
    # Create LGBM Dataset
    dtrain = lgb.Dataset(X_train, label=y_train)
    if pass_groups:
        dtrain.set_group(train_group_sizes)
        
    valid_sets = [dtrain]
    valid_names = ["train"]
    
    if X_val is not None and y_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        if pass_groups:
            # Check if lambda rank and val_groups exist
            if objective == "lambdarank" and val_groups is not None:
                dval.set_group(val_group_sizes)
        valid_sets.append(dval)
        valid_names.append("valid")
        
    # 2. Train
    # Default params for Ranking if not in config
    if objective == "lambdarank":
        # Ensure metric is ndcg
        if "metric" not in params:
            params["metric"] = "ndcg"
        # Ensure label_gain is set (optional but good)
        if "label_gain" not in params:
             # Default gains for relevance 0, 1, ...
             # Since our targets are continuous returns, LightGBM handles this.
             pass

    # Verbosity
    params["verbosity"] = -1
    
    # Train
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=params.get("n_estimators", 1000),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[
            lgb.early_stopping(stopping_rounds=params.get("early_stopping_rounds", 50)),
            lgb.log_evaluation(period=0), # Silent
        ],
    )
    
    return model


def _train_xgboost(X, y, params):
    from xgboost import XGBRegressor
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model


def _train_rf(X, y, params):
    model = RandomForestRegressor(**params)
    model.fit(X, y)
    return model
