# tradesense/modeling.py
"""Phase 3 probabilistic modeling for TradeSense."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


def _validate_inputs(features: pd.DataFrame, close_prices: pd.Series) -> None:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas.DataFrame")
    if not isinstance(close_prices, pd.Series):
        raise TypeError("close_prices must be a pandas.Series")
    if features.empty:
        raise ValueError("features must not be empty")
    if close_prices.empty:
        raise ValueError("close_prices must not be empty")


def create_target(close_prices: pd.Series, horizon: int = 5) -> pd.Series:
    """Create the binary target label for continuation after a fixed horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    future_close = close_prices.shift(-horizon)
    target = (future_close > close_prices).astype("float64")
    target[future_close.isna()] = np.nan
    return target


def prepare_model_data(
    features: pd.DataFrame, close_prices: pd.Series, horizon: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """Align features and close prices, then create a clean target series."""
    _validate_inputs(features, close_prices)

    features = features.sort_index()
    close_prices = close_prices.sort_index()

    common_index = features.index.intersection(close_prices.index)
    if common_index.empty:
        raise ValueError("features and close_prices must share at least one index value")

    features = features.loc[common_index]
    close_prices = close_prices.loc[common_index]

    target = create_target(close_prices, horizon=horizon)
    valid_mask = target.notna() & features.notna().all(axis=1)

    features = features.loc[valid_mask]
    target = target.loc[valid_mask].astype("int64")

    unique_values = set(target.unique())
    if not unique_values.issubset({0, 1}):
        raise ValueError("target must be binary")

    if features.empty:
        raise ValueError("No valid rows available after alignment and target creation")

    return features, target


def time_train_test_split(
    features: pd.DataFrame, target: pd.Series, train_size: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset by time order without shuffling."""
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1")
    if len(features) != len(target):
        raise ValueError("features and target must be the same length")

    split_index = int(len(features) * train_size)
    if split_index <= 0 or split_index >= len(features):
        raise ValueError("Not enough data to perform the requested split")

    X_train = features.iloc[:split_index]
    y_train = target.iloc[:split_index]
    X_test = features.iloc[split_index:]
    y_test = target.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[XGBClassifier, LogisticRegression]:
    """Train the XGBoost and baseline Logistic Regression models."""
    if y_train.nunique() < 2:
        raise ValueError("Training data must contain at least two classes")

    xgboost_model = XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
    )
    xgboost_model.fit(X_train, y_train)

    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)

    return xgboost_model, baseline_model


def _compute_metrics(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def run_modeling_pipeline(
    features: pd.DataFrame, close_prices: pd.Series, horizon: int = 5
) -> Dict[str, Dict[str, object]]:
    """Train, evaluate, and report probabilistic models for continuation."""
    X, y = prepare_model_data(features, close_prices, horizon=horizon)
    X_train, X_test, y_train, y_test = time_train_test_split(X, y, train_size=0.7)

    if y_test.nunique() < 2:
        raise ValueError("Test data must contain at least two classes")

    xgb_model, baseline_model = train_models(X_train, y_train)

    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    baseline_prob = baseline_model.predict_proba(X_test)[:, 1]

    xgb_metrics = _compute_metrics(y_test, xgb_prob)
    baseline_metrics = _compute_metrics(y_test, baseline_prob)

    feature_importance = pd.Series(
        xgb_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    return {
        "xgboost": {
            "precision": xgb_metrics["precision"],
            "recall": xgb_metrics["recall"],
            "f1": xgb_metrics["f1"],
            "roc_auc": xgb_metrics["roc_auc"],
            "feature_importance": feature_importance,
        },
        "baseline": {
            "precision": baseline_metrics["precision"],
            "recall": baseline_metrics["recall"],
            "f1": baseline_metrics["f1"],
            "roc_auc": baseline_metrics["roc_auc"],
        },
    }
