# tests/test_phase3.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.modeling import (  # noqa: E402
    create_target,
    prepare_model_data,
    run_modeling_pipeline,
    time_train_test_split,
    train_models,
)


def _sample_data(rows: int = 200):
    index = pd.date_range("2021-01-01", periods=rows, freq="D")
    close = pd.Series(
        100 + np.sin(np.linspace(0, 12, rows)) * 5 + np.linspace(0, 3, rows),
        index=index,
    )
    features = pd.DataFrame(
        {
            "f1": close.pct_change().fillna(0),
            "f2": close.diff().fillna(0),
            "f3": np.cos(np.linspace(0, 8, rows)),
            "f4": np.sin(np.linspace(0, 4, rows)),
            "f5": close - close.rolling(5, min_periods=1).mean(),
        },
        index=index,
    )
    return features, close


def test_target_is_binary():
    _, close = _sample_data()
    target = create_target(close)
    values = set(target.dropna().unique())
    assert values.issubset({0, 1})
    assert len(values) == 2


def test_time_split_preserves_order():
    features, close = _sample_data()
    X, y = prepare_model_data(features, close)
    X_train, X_test, y_train, y_test = time_train_test_split(X, y, train_size=0.7)

    assert X_train.index.max() < X_test.index.min()
    assert y_train.index.equals(X_train.index)
    assert y_test.index.equals(X_test.index)


def test_models_output_probabilities():
    features, close = _sample_data()
    X, y = prepare_model_data(features, close)
    X_train, X_test, y_train, _ = time_train_test_split(X, y, train_size=0.7)
    xgb_model, baseline_model = train_models(X_train, y_train)

    xgb_proba = xgb_model.predict_proba(X_test)
    baseline_proba = baseline_model.predict_proba(X_test)

    assert xgb_proba.shape[0] == X_test.shape[0]
    assert xgb_proba.shape[1] == 2
    assert baseline_proba.shape[0] == X_test.shape[0]
    assert baseline_proba.shape[1] == 2
    assert np.all((xgb_proba >= 0) & (xgb_proba <= 1))
    assert np.all((baseline_proba >= 0) & (baseline_proba <= 1))


def test_metrics_and_feature_importance():
    features, close = _sample_data()
    results = run_modeling_pipeline(features, close)

    for model_key in ["xgboost", "baseline"]:
        assert model_key in results
        metrics = results[model_key]
        for key in ["precision", "recall", "f1", "roc_auc"]:
            assert key in metrics
            assert isinstance(metrics[key], float)
            assert np.isfinite(metrics[key])

    feature_importance = results["xgboost"]["feature_importance"]
    assert isinstance(feature_importance, pd.Series)
    assert not feature_importance.empty
