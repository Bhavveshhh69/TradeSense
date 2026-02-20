"""Deterministic post-decision context and reasoning engine."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Mapping
import logging


logger = logging.getLogger(__name__)

_REQUIRED_PREDICTION_FIELDS = ("probability", "confidence", "model_version", "timestamp")
_REQUIRED_DECISION_FIELDS = (
    "decision",
    "probability",
    "confidence",
    "confidence_level",
    "strength",
    "model_version",
    "timestamp",
)
_REQUIRED_FEATURE_FIELDS = (
    "price_vs_ema20",
    "price_vs_ema50",
    "ema20_vs_ema50",
    "volatility_regime",
    "range_expansion",
    "risk_state",
)


class ContextInputError(ValueError):
    """Raised when context engine inputs are invalid."""


class TradeSenseContextEngine:
    """Build deterministic reasoning context from prediction, decision, and features."""

    def build_context(
        self,
        prediction: Mapping[str, object],
        decision: Mapping[str, object],
        features: Mapping[str, object],
    ) -> dict:
        """Return structured reasoning context for UI/reporting consumers."""

        validated_prediction = self._validate_prediction(prediction)
        validated_decision = self._validate_decision(decision)
        validated_features = self._validate_features(features)

        pred_model_version = str(validated_prediction["model_version"])
        dec_model_version = str(validated_decision["model_version"])
        if pred_model_version != dec_model_version:
            raise ContextInputError(
                "prediction.model_version and decision.model_version must match"
            )

        decision_value = str(validated_decision["decision"])
        confidence_level = str(validated_decision["confidence_level"])
        strength = float(validated_decision["strength"])

        context = {
            "decision_summary": self._build_decision_summary(decision_value, confidence_level),
            "confidence_summary": self._build_confidence_summary(confidence_level),
            "strength_summary": self._build_strength_summary(strength),
            "trend_summary": self._build_trend_summary(validated_features),
            "risk_summary": self._build_risk_summary(validated_features),
            "model_summary": (
                "Prediction generated using calibrated XGBoost model version "
                f"{pred_model_version}."
            ),
            "generated_at": datetime.now(tz=UTC).isoformat(),
        }

        logger.info(
            "Context generated: decision=%s model_version=%s",
            decision_value,
            pred_model_version,
        )

        return context

    @staticmethod
    def _validate_prediction(prediction: Mapping[str, object]) -> Mapping[str, object]:
        if not isinstance(prediction, Mapping):
            raise ContextInputError("prediction must be a mapping/dict")
        missing = [field for field in _REQUIRED_PREDICTION_FIELDS if field not in prediction]
        if missing:
            raise ContextInputError(f"prediction missing required fields: {missing}")

        probability = prediction["probability"]
        confidence = prediction["confidence"]
        model_version = prediction["model_version"]
        timestamp = prediction["timestamp"]

        if not isinstance(probability, (int, float)):
            raise ContextInputError("prediction.probability must be numeric")
        if not 0.0 <= float(probability) <= 1.0:
            raise ContextInputError("prediction.probability must be between 0 and 1")

        if not isinstance(confidence, (int, float)):
            raise ContextInputError("prediction.confidence must be numeric")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ContextInputError("prediction.confidence must be between 0 and 1")

        if not isinstance(model_version, str) or not model_version.strip():
            raise ContextInputError("prediction.model_version must be a non-empty string")
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ContextInputError("prediction.timestamp must be a non-empty string")

        return prediction

    @staticmethod
    def _validate_decision(decision: Mapping[str, object]) -> Mapping[str, object]:
        if not isinstance(decision, Mapping):
            raise ContextInputError("decision must be a mapping/dict")

        missing = [field for field in _REQUIRED_DECISION_FIELDS if field not in decision]
        if missing:
            raise ContextInputError(f"decision missing required fields: {missing}")

        decision_value = decision["decision"]
        if decision_value not in {"BUY", "SELL", "HOLD"}:
            raise ContextInputError("decision.decision must be one of BUY/SELL/HOLD")

        confidence_level = decision["confidence_level"]
        if confidence_level not in {"very_low", "low", "moderate", "high", "very_high"}:
            raise ContextInputError("decision.confidence_level is invalid")

        strength = decision["strength"]
        if (
            not isinstance(strength, (int, float))
            or isinstance(strength, bool)
            or not 0.0 <= float(strength) <= 1.0
        ):
            raise ContextInputError("decision.strength must be between 0 and 1")

        probability = decision["probability"]
        confidence = decision["confidence"]
        timestamp = decision["timestamp"]
        if not isinstance(probability, (int, float)):
            raise ContextInputError("decision.probability must be numeric")
        if not 0.0 <= float(probability) <= 1.0:
            raise ContextInputError("decision.probability must be between 0 and 1")
        if not isinstance(confidence, (int, float)):
            raise ContextInputError("decision.confidence must be numeric")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ContextInputError("decision.confidence must be between 0 and 1")
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ContextInputError("decision.timestamp must be a non-empty string")

        return decision

    @staticmethod
    def _validate_features(features: Mapping[str, object]) -> Mapping[str, object]:
        if not isinstance(features, Mapping):
            raise ContextInputError("features must be a mapping/dict")
        missing = [field for field in _REQUIRED_FEATURE_FIELDS if field not in features]
        if missing:
            raise ContextInputError(f"features missing required fields: {missing}")

        for field in ("price_vs_ema20", "price_vs_ema50", "ema20_vs_ema50"):
            value = features[field]
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise ContextInputError(f"features.{field} must be numeric") from exc

        return features

    @staticmethod
    def _build_decision_summary(decision: str, confidence_level: str) -> str:
        if decision == "BUY":
            return (
                "Model predicts upward price movement with "
                f"{confidence_level} confidence."
            )
        if decision == "SELL":
            return (
                "Model predicts downward price movement with "
                f"{confidence_level} confidence."
            )
        return "Model prediction is within neutral zone. No directional edge detected."

    @staticmethod
    def _build_confidence_summary(confidence_level: str) -> str:
        mapping = {
            "very_high": "Prediction confidence is extremely strong. Signal reliability is high.",
            "high": "Prediction confidence is strong. Signal reliability is favorable.",
            "moderate": "Prediction confidence is moderate. Signal reliability is acceptable.",
            "low": "Prediction confidence is weak. Signal reliability is limited.",
            "very_low": "Prediction confidence is extremely weak. Signal reliability is poor.",
        }
        return mapping[confidence_level]

    @staticmethod
    def _build_strength_summary(strength: float) -> str:
        if strength >= 0.8:
            return "Signal strength is very strong."
        if strength >= 0.6:
            return "Signal strength is strong."
        if strength >= 0.4:
            return "Signal strength is moderate."
        if strength >= 0.2:
            return "Signal strength is weak."
        return "Signal strength is very weak."

    @staticmethod
    def _build_trend_summary(features: Mapping[str, object]) -> str:
        price_vs_ema20 = float(features["price_vs_ema20"])
        price_vs_ema50 = float(features["price_vs_ema50"])
        ema20_vs_ema50 = float(features["ema20_vs_ema50"])

        if price_vs_ema20 > 0.0 and price_vs_ema50 > 0.0 and ema20_vs_ema50 >= 0.0:
            return "Market is in short and medium term uptrend."
        if price_vs_ema20 < 0.0 and price_vs_ema50 < 0.0 and ema20_vs_ema50 <= 0.0:
            return "Market is in short and medium term downtrend."
        return "Market trend is mixed or transitional."

    def _build_risk_summary(self, features: Mapping[str, object]) -> str:
        volatility_regime_high = self._is_high_value(features["volatility_regime"])
        range_expansion_high = self._is_high_range_expansion(features["range_expansion"])
        risk_state_high = self._is_high_value(features["risk_state"])

        if volatility_regime_high or range_expansion_high:
            return "Market volatility is elevated. Risk is higher than normal."
        if risk_state_high:
            return "Market risk conditions are elevated."
        return "Market risk conditions are normal."

    @staticmethod
    def _is_high_value(value: object) -> bool:
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized in {"high", "elevated", "2"}
        if isinstance(value, (int, float)):
            return float(value) >= 2.0
        return False

    @staticmethod
    def _is_high_range_expansion(value: object) -> bool:
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized in {"high", "elevated"}
        if isinstance(value, (int, float)):
            return float(value) >= 1.5
        return False
