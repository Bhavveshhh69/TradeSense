from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .market import DEFAULT_TIMEFRAME_MIN, STRATEGY_FAMILY, get_market_profile, resolve_market
from .provider import YahooIntradayProvider
from .quality import DataQualityValidator
from .sentiment import IntradaySentimentEngine
from .strategy import FEATURE_COLUMNS, ORBSessionVWAPStrategy


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_SCHEMA_VERSION = 4
LOOKBACK_DAYS = 90
US_BOOTSTRAP_SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AMD", "NFLX", "JPM", "TSLA")
IN_BOOTSTRAP_SYMBOLS = ("RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "HCLTECH.NS", "WIPRO.NS", "LT.NS")
ROUND_TRIP_COST_R = 0.02
STRESS_COST_MULTIPLIER = 1.75
MIN_TRADE_COUNT_FLOOR = 12
MIN_ELIGIBLE_SESSION_COVERAGE = 0.30
MIN_PROFIT_FACTOR = 1.1
MIN_WIN_RATE_LOWER_BOUND = 0.5
EXECUTION_BPS = {
    "US": {"entry": 4.0, "exit": 5.0, "borrow": 1.0},
    "IN": {"entry": 6.0, "exit": 7.0, "borrow": 1.0},
}
POLICY_GRID = (
    {"threshold_delta": 0.0, "min_breakout_strength": 0.0, "min_relative_volume": 0.8, "min_body_wick": 0.0, "max_range_expansion": 3.5, "min_abs_vwap_distance": 0.0, "max_session_progress": 0.65, "use_sentiment_gate": False},
    {"threshold_delta": 0.02, "min_breakout_strength": 0.1, "min_relative_volume": 0.9, "min_body_wick": 0.25, "max_range_expansion": 3.0, "min_abs_vwap_distance": 0.003, "max_session_progress": 0.55, "use_sentiment_gate": False},
    {"threshold_delta": 0.04, "min_breakout_strength": 0.12, "min_relative_volume": 1.0, "min_body_wick": 0.35, "max_range_expansion": 2.8, "min_abs_vwap_distance": 0.004, "max_session_progress": 0.5, "use_sentiment_gate": False},
    {"threshold_delta": 0.06, "min_breakout_strength": 0.18, "min_relative_volume": 1.1, "min_body_wick": 0.45, "max_range_expansion": 2.5, "min_abs_vwap_distance": 0.005, "max_session_progress": 0.42, "use_sentiment_gate": False},
    {"threshold_delta": 0.02, "min_breakout_strength": 0.1, "min_relative_volume": 0.9, "min_body_wick": 0.25, "max_range_expansion": 3.0, "min_abs_vwap_distance": 0.003, "max_session_progress": 0.55, "use_sentiment_gate": True},
    {"threshold_delta": 0.04, "min_breakout_strength": 0.12, "min_relative_volume": 1.0, "min_body_wick": 0.35, "max_range_expansion": 2.8, "min_abs_vwap_distance": 0.004, "max_session_progress": 0.5, "use_sentiment_gate": True},
    {"threshold_delta": 0.06, "min_breakout_strength": 0.18, "min_relative_volume": 1.1, "min_body_wick": 0.45, "max_range_expansion": 2.5, "min_abs_vwap_distance": 0.005, "max_session_progress": 0.42, "use_sentiment_gate": True},
    {"threshold_delta": 0.08, "min_breakout_strength": 0.22, "min_relative_volume": 1.2, "min_body_wick": 0.55, "max_range_expansion": 2.2, "min_abs_vwap_distance": 0.006, "max_session_progress": 0.35, "use_sentiment_gate": True},
)


@dataclass(frozen=True)
class ModelArtifact:
    model: object | None
    calibrator: object | None
    threshold: float
    feature_names: tuple[str, ...]
    model_name: str
    model_type: str
    metadata: dict[str, Any]
    model_bench_summary: dict[str, Any]


@dataclass(frozen=True)
class CandidateBundle:
    model: object
    calibrator: object | None
    probabilities: np.ndarray
    threshold: float
    validation_summary: dict[str, Any]
    holdout_summary: dict[str, Any]


class ModelRegistry:
    def __init__(
        self,
        provider: YahooIntradayProvider | None = None,
        sentiment_engine: IntradaySentimentEngine | None = None,
    ) -> None:
        self.provider = provider or YahooIntradayProvider()
        self.validator = DataQualityValidator()
        self.strategy = ORBSessionVWAPStrategy()
        self.sentiment_engine = sentiment_engine or IntradaySentimentEngine()

    def artifact_path(self, market: str, timeframe_min: int = DEFAULT_TIMEFRAME_MIN) -> Path:
        return MODEL_DIR / f"intraday_{market}_{STRATEGY_FAMILY}_{timeframe_min}m.joblib"

    def load_or_train(self, market: str, timeframe_min: int = DEFAULT_TIMEFRAME_MIN) -> ModelArtifact:
        path = self.artifact_path(market, timeframe_min=timeframe_min)
        if path.exists():
            payload = joblib.load(path)
            metadata = dict(payload.get("metadata", {}))
            if metadata.get("registry_version") == MODEL_SCHEMA_VERSION:
                return ModelArtifact(
                    model=payload.get("model"),
                    calibrator=payload.get("calibrator"),
                    threshold=float(payload.get("threshold", 0.55)),
                    feature_names=tuple(payload.get("feature_names", FEATURE_COLUMNS)),
                    model_name=str(payload.get("model_name", payload.get("model_type", "heuristic"))),
                    model_type=str(payload.get("model_type", payload.get("model_name", "heuristic"))),
                    metadata=metadata,
                    model_bench_summary=dict(payload.get("model_bench_summary", {})),
                )
        artifact = self._train_market(market, timeframe_min=timeframe_min)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": artifact.model,
                "calibrator": artifact.calibrator,
                "threshold": artifact.threshold,
                "feature_names": list(artifact.feature_names),
                "model_name": artifact.model_name,
                "model_type": artifact.model_type,
                "metadata": artifact.metadata,
                "model_bench_summary": artifact.model_bench_summary,
            },
            path,
        )
        return artifact

    def predict_probability(self, artifact: ModelArtifact, feature_frame: pd.DataFrame) -> float:
        X = feature_frame.loc[:, list(artifact.feature_names)].fillna(0.0)
        if artifact.model is None:
            score = float(
                0.35
                + 0.18 * float(X.iloc[0]["breakout_strength"])
                + 0.22 * float(X.iloc[0]["vwap_distance"])
                + 0.08 * float(X.iloc[0]["continuation_2"])
                + 0.05 * float(X.iloc[0]["relative_volume"] - 1.0)
            )
            return float(max(0.0, min(1.0, score)))
        raw = artifact.model.predict_proba(X)[:, 1]
        if artifact.calibrator is not None:
            return float(artifact.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1][0])
        return float(raw[0])

    def grouped_walk_forward_report(self, market: str, timeframe_min: int = DEFAULT_TIMEFRAME_MIN) -> dict[str, Any]:
        dataset = self._build_training_dataset(market, timeframe_min)
        if dataset is None or dataset.empty:
            return {
                "market": market,
                "timeframe": f"{timeframe_min}m",
                "models": {},
                "promotion_gate": {
                    "passed": False,
                    "reason": "No training dataset available for grouped walk-forward validation.",
                },
            }

        sessions = sorted(dataset["session_date"].unique())
        min_train = 12
        val_size = 6
        test_size = 3
        if len(sessions) < (min_train + val_size + test_size):
            return {
                "market": market,
                "timeframe": f"{timeframe_min}m",
                "models": {},
                "promotion_gate": {
                    "passed": False,
                    "reason": "Not enough session groups for walk-forward validation.",
                },
            }

        per_model_rows: dict[str, list[dict[str, Any]]] = {"xgboost": [], "logistic_regression": [], "random_forest": []}
        for train_end in range(min_train, len(sessions) - val_size - test_size + 1, test_size):
            train_sessions = set(sessions[:train_end])
            val_sessions = set(sessions[train_end : train_end + val_size])
            test_sessions = set(sessions[train_end + val_size : train_end + val_size + test_size])
            fold_candidates = self._fit_candidates(dataset, train_sessions, val_sessions, test_sessions)
            for model_name, bundle in fold_candidates.items():
                test_frame = dataset.loc[dataset["session_date"].isin(test_sessions)].copy()
                for idx, probability in zip(test_frame.index, bundle.probabilities, strict=False):
                    per_model_rows[model_name].append(
                        {
                            "session_date": dataset.at[idx, "session_date"],
                            "probability": float(probability),
                            "threshold": float(bundle.threshold),
                            "r_multiple": float(dataset.at[idx, "r_multiple"]),
                            "target": int(dataset.at[idx, "target"]),
                            "contextual_sentiment_score": float(dataset.at[idx, "contextual_sentiment_score"]),
                            "sentiment_confidence": float(dataset.at[idx, "sentiment_confidence"]),
                            "setup_side": str(dataset.at[idx, "setup_side"]),
                            "breakout_strength": float(dataset.at[idx, "breakout_strength"]),
                            "relative_volume": float(dataset.at[idx, "relative_volume"]),
                            "body_wick_imbalance": float(dataset.at[idx, "body_wick_imbalance"]),
                            "range_expansion": float(dataset.at[idx, "range_expansion"]),
                            "vwap_distance": float(dataset.at[idx, "vwap_distance"]),
                            "session_progress": float(dataset.at[idx, "session_progress"]),
                        }
                    )

        model_reports: dict[str, Any] = {}
        for model_name, rows in per_model_rows.items():
            frame = pd.DataFrame(rows)
            if frame.empty:
                continue
            base_policy = dict(POLICY_GRID[0])
            base_policy["use_sentiment_gate"] = False
            tuned_policy, with_sentiment = self._select_policy(frame)
            without_sentiment = self._trade_metrics(frame, policy=base_policy)
            model_reports[model_name] = {
                "without_sentiment": without_sentiment,
                "with_sentiment": with_sentiment,
                "sentiment_uplift": round(
                    with_sentiment["net_expectancy"] - without_sentiment["net_expectancy"],
                    4,
                ),
                "selected_policy": tuned_policy,
            }

        live_candidate = self.load_or_train(market, timeframe_min=timeframe_min)
        live_report = model_reports.get(live_candidate.model_name, {})
        live_with_sentiment = live_report.get("with_sentiment", {})
        passed = bool(
            live_with_sentiment
            and live_with_sentiment.get("net_expectancy", 0.0) > 0
            and live_with_sentiment.get("profit_factor", 0.0) >= MIN_PROFIT_FACTOR
            and live_with_sentiment.get("trade_count", 0) >= MIN_TRADE_COUNT_FLOOR
            and live_with_sentiment.get("eligible_session_coverage", 0.0) >= MIN_ELIGIBLE_SESSION_COVERAGE
            and live_with_sentiment.get("win_rate_lower_bound", 0.0) >= MIN_WIN_RATE_LOWER_BOUND
        )
        return {
            "market": market,
            "timeframe": f"{timeframe_min}m",
            "models": model_reports,
            "promotion_gate": {
                "passed": passed,
                "model_name": live_candidate.model_name,
                "reason": "Promotion gate passed." if passed else "Live model did not clear expectancy, profit factor, trade-count, coverage, or confidence floors.",
            },
        }

    def _train_market(self, market: str, timeframe_min: int) -> ModelArtifact:
        dataset = self._build_training_dataset(market, timeframe_min)
        if dataset is None or len(dataset) < 20 or dataset["target"].nunique() < 2:
            return ModelArtifact(
                model=None,
                calibrator=None,
                threshold=0.55,
                feature_names=tuple(FEATURE_COLUMNS),
                model_name="heuristic",
                model_type="heuristic",
                metadata={
                    "market": market,
                    "trained_at": datetime.now(tz=UTC).isoformat(),
                    "registry_version": MODEL_SCHEMA_VERSION,
                    "reason": "insufficient_intraday_dataset",
                },
                model_bench_summary={},
            )

        sessions = sorted(dataset["session_date"].unique())
        train_cut = max(12, int(len(sessions) * 0.6))
        val_cut = max(train_cut + 6, int(len(sessions) * 0.8))
        train_sessions = set(sessions[:train_cut])
        val_sessions = set(sessions[train_cut:val_cut])
        test_sessions = set(sessions[val_cut:])
        if not val_sessions or not test_sessions:
            return ModelArtifact(
                model=None,
                calibrator=None,
                threshold=0.55,
                feature_names=tuple(FEATURE_COLUMNS),
                model_name="heuristic",
                model_type="heuristic",
                metadata={
                    "market": market,
                    "trained_at": datetime.now(tz=UTC).isoformat(),
                    "registry_version": MODEL_SCHEMA_VERSION,
                    "reason": "insufficient_walk_forward_sessions",
                },
                model_bench_summary={},
            )

        candidates = self._fit_candidates(dataset, train_sessions, val_sessions, test_sessions)
        if not candidates:
            return ModelArtifact(
                model=None,
                calibrator=None,
                threshold=0.55,
                feature_names=tuple(FEATURE_COLUMNS),
                model_name="heuristic",
                model_type="heuristic",
                metadata={
                    "market": market,
                    "trained_at": datetime.now(tz=UTC).isoformat(),
                    "registry_version": MODEL_SCHEMA_VERSION,
                    "reason": "candidate_training_failed",
                },
                model_bench_summary={},
            )

        best_name = max(candidates, key=lambda name: (self._candidate_score(candidates[name]), 1 if name == "xgboost" else 0))
        best_candidate = candidates[best_name]
        bench_summary = {
            model_name: {
                "validation": bundle.validation_summary,
                "holdout": bundle.holdout_summary,
                "threshold": bundle.threshold,
            }
            for model_name, bundle in candidates.items()
        }
        return ModelArtifact(
            model=best_candidate.model,
            calibrator=best_candidate.calibrator,
            threshold=best_candidate.threshold,
            feature_names=tuple(FEATURE_COLUMNS),
            model_name=best_name,
            model_type=best_name,
            metadata={
                "market": market,
                "trained_at": datetime.now(tz=UTC).isoformat(),
                "registry_version": MODEL_SCHEMA_VERSION,
                "validation_expectancy": best_candidate.validation_summary["net_expectancy"],
                "validation_rows": int(sum(dataset["session_date"].isin(val_sessions))),
                "holdout_expectancy": best_candidate.holdout_summary["net_expectancy"],
                "selected_policy": best_candidate.validation_summary.get("selected_policy"),
            },
            model_bench_summary=bench_summary,
        )

    def _fit_candidates(
        self,
        dataset: pd.DataFrame,
        train_sessions: set,
        val_sessions: set,
        test_sessions: set,
    ) -> dict[str, CandidateBundle]:
        X_train = dataset.loc[dataset["session_date"].isin(train_sessions), FEATURE_COLUMNS]
        y_train = dataset.loc[dataset["session_date"].isin(train_sessions), "target"]
        X_val = dataset.loc[dataset["session_date"].isin(val_sessions), FEATURE_COLUMNS]
        y_val = dataset.loc[dataset["session_date"].isin(val_sessions), "target"]
        X_test = dataset.loc[dataset["session_date"].isin(test_sessions), FEATURE_COLUMNS]
        if X_train.empty or X_val.empty or X_test.empty or y_train.nunique() < 2:
            return {}

        model_specs: dict[str, object] = {
            "xgboost": XGBClassifier(
                n_estimators=220,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                eval_metric="logloss",
            ),
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=4,
                random_state=42,
            ),
        }

        candidates: dict[str, CandidateBundle] = {}
        val_r = dataset.loc[dataset["session_date"].isin(val_sessions), "r_multiple"].to_numpy()
        test_frame = dataset.loc[dataset["session_date"].isin(test_sessions)].copy()
        for model_name, model in model_specs.items():
            fitted = model.fit(X_train, y_train)
            val_raw = fitted.predict_proba(X_val)[:, 1]
            calibrator = self._fit_calibrator(val_raw, y_val.to_numpy())
            val_prob = self._apply_calibrator(calibrator, val_raw)
            threshold, _ = self._best_threshold(val_prob, val_r)
            test_raw = fitted.predict_proba(X_test)[:, 1]
            test_prob = self._apply_calibrator(calibrator, test_raw)

            val_eval = dataset.loc[dataset["session_date"].isin(val_sessions)].copy()
            val_eval["probability"] = val_prob
            val_eval["threshold"] = threshold
            test_eval = test_frame.copy()
            test_eval["probability"] = test_prob
            test_eval["threshold"] = threshold
            best_policy, validation_summary = self._select_policy(val_eval)
            test_summary = self._trade_metrics(test_eval, policy=best_policy)
            candidates[model_name] = CandidateBundle(
                model=fitted,
                calibrator=calibrator,
                probabilities=test_prob,
                threshold=threshold,
                validation_summary=validation_summary,
                holdout_summary=test_summary,
            )
        return candidates

    def _fit_calibrator(self, probabilities: np.ndarray, y_true: np.ndarray) -> LogisticRegression | None:
        if len(np.unique(y_true)) < 2:
            return None
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(probabilities.reshape(-1, 1), y_true)
        return calibrator

    def _apply_calibrator(self, calibrator: LogisticRegression | None, probabilities: np.ndarray) -> np.ndarray:
        if calibrator is None:
            return probabilities
        return calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]

    def _best_threshold(self, probabilities: np.ndarray, r_values: np.ndarray) -> tuple[float, float]:
        best_threshold = 0.55
        best_expectancy = -999.0
        for threshold in np.linspace(0.5, 0.75, 11):
            mask = probabilities >= threshold
            if mask.sum() < 3:
                continue
            expectancy = float(np.mean(r_values[mask] - ROUND_TRIP_COST_R))
            if expectancy > best_expectancy:
                best_expectancy = expectancy
                best_threshold = float(round(threshold, 2))
        return best_threshold, best_expectancy if best_expectancy > -999.0 else 0.0

    def _build_training_dataset(self, market: str, timeframe_min: int) -> pd.DataFrame | None:
        rows: list[dict[str, Any]] = []
        symbols = US_BOOTSTRAP_SYMBOLS if market == "US" else IN_BOOTSTRAP_SYMBOLS
        for symbol in symbols:
            symbol_market, _ = resolve_market(symbol)
            if symbol_market != market:
                continue
            profile = get_market_profile(symbol, timeframe_min=timeframe_min)
            request = type(
                "Req",
                (),
                {
                    "symbol": symbol,
                    "market": profile.market,
                    "exchange": profile.exchange,
                    "timezone": profile.timezone,
                    "currency": profile.currency,
                    "timeframe_min": timeframe_min,
                    "lookback_days": LOOKBACK_DAYS,
                    "source": "yfinance",
                },
            )()
            bars = self.provider.fetch_bars(request)
            quality = self.validator.validate(bars, profile, timeframe_min)
            context = self.strategy.build_context(symbol, bars, profile, quality)
            frame = context.feature_frame
            if frame.empty:
                continue
            rows.extend(self._extract_proposals(frame, symbol, profile))
        if not rows:
            return None
        dataset = pd.DataFrame(rows)
        return dataset.dropna(subset=["target"]).reset_index(drop=True)

    def _extract_proposals(self, frame: pd.DataFrame, symbol: str, profile) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        grouped = frame.groupby("session_date", sort=True)
        for session_date, session in grouped:
            session = session.copy().reset_index(drop=True)
            if session.empty:
                continue
            entry_rows = session[(session["bar_index"] >= 2)].copy()
            signal_idx = None
            side = None
            for idx, row in entry_rows.iterrows():
                if row["close"] > row["opening_range_high"] and row["close"] > row["session_vwap"] and row["vwap_slope"] > 0:
                    signal_idx = int(idx)
                    side = "LONG"
                    break
                if row["close"] < row["opening_range_low"] and row["close"] < row["session_vwap"] and row["vwap_slope"] < 0:
                    signal_idx = int(idx)
                    side = "SHORT"
                    break
            if signal_idx is None or side is None or signal_idx >= len(session) - 1:
                continue
            signal_row = session.iloc[signal_idx]
            entry_row = session.iloc[signal_idx + 1]
            risk_unit = max(float(signal_row["opening_range_width"]), float(entry_row["open"]) * 0.0035)
            entry_price = self._apply_entry_slippage(float(entry_row["open"]), side, profile.market)
            if side == "LONG":
                stop_price = entry_price - risk_unit
                take_profit_price = entry_price + risk_unit * 1.5
            else:
                stop_price = entry_price + risk_unit
                take_profit_price = entry_price - risk_unit * 1.5
            future = session[session["timestamp"] >= entry_row["timestamp"]]
            r_multiple = self._label_trade(future, side, entry_price, stop_price, take_profit_price, risk_unit, profile.market)
            sentiment_snapshot = self.sentiment_engine.snapshot(
                symbol,
                profile,
                signal_row["timestamp"].to_pydatetime() if hasattr(signal_row["timestamp"], "to_pydatetime") else signal_row["timestamp"],
            )
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": session_date,
                    **{name: float(signal_row[name]) for name in FEATURE_COLUMNS},
                    "setup_side": side,
                    "target": 1 if r_multiple > 0 else 0,
                    "r_multiple": r_multiple,
                    "stock_sentiment_score": sentiment_snapshot.stock_sentiment_score,
                    "sector_sentiment_score": sentiment_snapshot.sector_sentiment_score or 0.0,
                    "contextual_sentiment_score": sentiment_snapshot.contextual_sentiment_score,
                    "sentiment_confidence": sentiment_snapshot.sentiment_confidence,
                }
            )
        return rows

    def _select_policy(self, frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
        best_policy = dict(POLICY_GRID[0])
        best_summary = self._trade_metrics(frame, policy=best_policy)
        best_score = self._policy_score(best_summary)
        for policy in POLICY_GRID[1:]:
            summary = self._trade_metrics(frame, policy=policy)
            score = self._policy_score(summary)
            if score > best_score:
                best_policy = dict(policy)
                best_summary = summary
                best_score = score
        best_summary["selected_policy"] = dict(best_policy)
        return best_policy, best_summary

    def _policy_score(self, summary: dict[str, Any]) -> tuple[float, float, float, float]:
        trade_count = float(summary.get("trade_count", 0))
        if trade_count < MIN_TRADE_COUNT_FLOOR or float(summary.get("eligible_session_coverage", 0.0)) < MIN_ELIGIBLE_SESSION_COVERAGE:
            return (-999.0, -999.0, -999.0, trade_count)
        return (
            float(summary.get("win_rate_lower_bound", 0.0)),
            float(summary.get("net_expectancy", 0.0)) - (0.004 * float(summary.get("complexity", 0.0))),
            float(summary.get("profit_factor", 0.0)),
            float(summary.get("accuracy", 0.0)),
        )

    def _candidate_score(self, bundle: CandidateBundle) -> tuple[float, float, float, float, float]:
        holdout = bundle.holdout_summary
        validation = bundle.validation_summary
        return (
            float(holdout.get("win_rate_lower_bound", 0.0)),
            float(holdout.get("net_expectancy", 0.0)),
            float(holdout.get("profit_factor", 0.0)),
            float(validation.get("win_rate_lower_bound", 0.0)),
            float(validation.get("net_expectancy", 0.0)),
        )

    def _trade_metrics(self, frame: pd.DataFrame, *, policy: dict[str, Any]) -> dict[str, Any]:
        eval_frame = frame.copy().sort_values("session_date").reset_index(drop=True)
        if eval_frame.empty:
            return {
                "win_rate": 0.0,
                "accuracy": 0.0,
                "trade_count": 0,
                "eligible_session_coverage": 0.0,
                "average_r_multiple": 0.0,
                "net_expectancy": 0.0,
                "profit_factor": 0.0,
                "win_rate_lower_bound": 0.0,
                "max_drawdown": 0.0,
                "calibration": {"brier_score": 0.0},
                "confidence_distribution": {},
                "complexity": 0,
            }

        effective_thresholds: list[float] = []
        traded_mask: list[bool] = []
        for row in eval_frame.itertuples(index=False):
            threshold = float(row.threshold) + float(policy.get("threshold_delta", 0.0))
            if bool(policy.get("use_sentiment_gate", False)):
                threshold, _ = self.sentiment_engine.gate_threshold(
                    threshold,
                    str(row.setup_side),
                    type(
                        "Snapshot",
                        (),
                        {
                            "contextual_sentiment_score": float(row.contextual_sentiment_score),
                            "sentiment_confidence": float(row.sentiment_confidence),
                        },
                    )(),
                )
            passes_filters = (
                abs(float(row.breakout_strength)) >= float(policy.get("min_breakout_strength", 0.0))
                and float(row.relative_volume) >= float(policy.get("min_relative_volume", 0.0))
                and float(row.body_wick_imbalance) >= float(policy.get("min_body_wick", 0.0))
                and float(row.range_expansion) <= float(policy.get("max_range_expansion", 99.0))
                and abs(float(row.vwap_distance)) >= float(policy.get("min_abs_vwap_distance", 0.0))
                and float(row.session_progress) <= float(policy.get("max_session_progress", 1.0))
            )
            effective_thresholds.append(threshold)
            traded_mask.append(passes_filters and float(row.probability) >= threshold)

        eval_frame["effective_threshold"] = effective_thresholds
        eval_frame["traded"] = traded_mask
        traded = eval_frame[eval_frame["traded"]].copy()
        accuracy = float(((eval_frame["traded"].astype(int) == eval_frame["target"].astype(int)).mean()))
        base_policy = POLICY_GRID[0]
        complexity = sum(
            1 for key, value in policy.items() if key != "use_sentiment_gate" and value != base_policy.get(key)
        ) + int(bool(policy.get("use_sentiment_gate", False)))
        total_sessions = max(int(eval_frame["session_date"].nunique()), 1)
        if traded.empty:
            return {
                "win_rate": 0.0,
                "accuracy": round(accuracy, 4),
                "trade_count": 0,
                "eligible_session_coverage": 0.0,
                "average_r_multiple": 0.0,
                "net_expectancy": 0.0,
                "profit_factor": 0.0,
                "win_rate_lower_bound": 0.0,
                "max_drawdown": 0.0,
                "calibration": {
                    "brier_score": round(float(np.mean((eval_frame["probability"] - eval_frame["target"]) ** 2)), 4),
                },
                "confidence_distribution": {
                    "above_threshold": 0,
                    "below_threshold": int((~eval_frame["traded"]).sum()),
                },
                "complexity": complexity,
            }

        net_r = traded["r_multiple"] - (ROUND_TRIP_COST_R * STRESS_COST_MULTIPLIER)
        gross_wins = traded.loc[traded["r_multiple"] > 0, "r_multiple"].sum()
        gross_losses = abs(traded.loc[traded["r_multiple"] < 0, "r_multiple"].sum())
        equity = np.cumsum(net_r.to_numpy())
        running_peak = np.maximum.accumulate(np.r_[0.0, equity])
        drawdown = running_peak[1:] - equity
        win_count = int((traded["r_multiple"] > 0).sum())
        trade_count = int(len(traded))
        win_rate = float(win_count / trade_count)
        return {
            "win_rate": round(win_rate, 4),
            "accuracy": round(accuracy, 4),
            "trade_count": trade_count,
            "eligible_session_coverage": round(float(traded["session_date"].nunique() / total_sessions), 4),
            "average_r_multiple": round(float(traded["r_multiple"].mean()), 4),
            "net_expectancy": round(float(net_r.mean()), 4),
            "profit_factor": round(float(gross_wins / gross_losses) if gross_losses else float(gross_wins), 4),
            "win_rate_lower_bound": round(self._wilson_lower_bound(win_count, trade_count), 4),
            "max_drawdown": round(float(drawdown.max()) if len(drawdown) else 0.0, 4),
            "calibration": {
                "brier_score": round(float(np.mean((eval_frame["probability"] - eval_frame["target"]) ** 2)), 4),
            },
            "confidence_distribution": {
                "above_threshold": int(len(traded)),
                "below_threshold": int((~eval_frame["traded"]).sum()),
            },
            "complexity": complexity,
        }

    def _label_trade(
        self,
        future: pd.DataFrame,
        side: str,
        entry_price: float,
        stop_price: float,
        take_profit_price: float,
        risk_unit: float,
        market: str,
    ) -> float:
        for row in future.itertuples(index=False):
            if side == "LONG":
                stop_hit = row.low <= stop_price
                take_hit = row.high >= take_profit_price
                if stop_hit and take_hit:
                    return self._net_r_multiple(stop_price, entry_price, risk_unit, side, market)
                if stop_hit:
                    return self._net_r_multiple(stop_price, entry_price, risk_unit, side, market)
                if take_hit:
                    return self._net_r_multiple(take_profit_price, entry_price, risk_unit, side, market)
            else:
                stop_hit = row.high >= stop_price
                take_hit = row.low <= take_profit_price
                if stop_hit and take_hit:
                    return self._net_r_multiple(stop_price, entry_price, risk_unit, side, market)
                if stop_hit:
                    return self._net_r_multiple(stop_price, entry_price, risk_unit, side, market)
                if take_hit:
                    return self._net_r_multiple(take_profit_price, entry_price, risk_unit, side, market)
        if future.empty:
            return 0.0
        final_close = float(future.iloc[-1]["close"])
        return self._net_r_multiple(final_close, entry_price, risk_unit, side, market)

    def _apply_entry_slippage(self, price: float, side: str, market: str) -> float:
        bps = EXECUTION_BPS[market]["entry"] / 10_000.0
        return price * (1 + bps) if side == "LONG" else price * (1 - bps)

    def _apply_exit_slippage(self, price: float, side: str, market: str) -> float:
        bps = EXECUTION_BPS[market]["exit"] / 10_000.0
        return price * (1 - bps) if side == "LONG" else price * (1 + bps)

    def _net_r_multiple(self, exit_price: float, entry_price: float, risk_unit: float, side: str, market: str) -> float:
        adjusted_exit = self._apply_exit_slippage(exit_price, side, market)
        signed = (adjusted_exit - entry_price) / risk_unit if side == "LONG" else (entry_price - adjusted_exit) / risk_unit
        extra_cost = (EXECUTION_BPS[market]["borrow"] / 10_000.0) * (entry_price / max(risk_unit, 1e-6)) if side == "SHORT" else 0.0
        return float(round(signed - extra_cost, 4))

    def _wilson_lower_bound(self, wins: int, total: int, z: float = 1.96) -> float:
        if total <= 0:
            return 0.0
        phat = wins / total
        denominator = 1 + (z * z) / total
        centre = phat + (z * z) / (2 * total)
        margin = z * np.sqrt((phat * (1 - phat) / total) + (z * z) / (4 * total * total))
        return float((centre - margin) / denominator)
