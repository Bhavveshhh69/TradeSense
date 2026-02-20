"""Post-inference decision engine for calibrated TradeSense predictions."""

from __future__ import annotations

from typing import Mapping
import logging


logger = logging.getLogger(__name__)

_REQUIRED_INPUT_FIELDS = ("probability", "confidence", "model_version", "timestamp")


class DecisionInputError(ValueError):
    """Raised when predictor output is invalid for decision generation."""


class TradeSenseDecisionEngine:
    """Convert calibrated predictor output into structured trade decisions."""

    def decide_from_prediction(self, predictor_output: Mapping[str, object]) -> dict:
        """Build decision output from validated predictor output."""

        validated = self._validate_predictor_output(predictor_output)
        probability = float(validated["probability"])

        distance = abs(probability - 0.5)
        strength = min(1.0, max(0.0, distance * 2.0))
        confidence_level = self._map_confidence_level(distance)
        decision = self._map_decision(probability)

        result = {
            "decision": decision,
            "probability": probability,
            "confidence": float(validated["confidence"]),
            "confidence_level": confidence_level,
            "strength": strength,
            "model_version": str(validated["model_version"]),
            "timestamp": str(validated["timestamp"]),
        }

        logger.info(
            "Decision generated: decision=%s model_version=%s confidence_level=%s",
            decision,
            result["model_version"],
            confidence_level,
        )

        return result

    @staticmethod
    def _validate_predictor_output(predictor_output: Mapping[str, object]) -> Mapping[str, object]:
        if not isinstance(predictor_output, Mapping):
            raise DecisionInputError("predictor_output must be a mapping/dict")

        missing = [field for field in _REQUIRED_INPUT_FIELDS if field not in predictor_output]
        if missing:
            raise DecisionInputError(
                f"predictor_output missing required fields: {missing}"
            )

        probability = predictor_output["probability"]
        confidence = predictor_output["confidence"]
        model_version = predictor_output["model_version"]
        timestamp = predictor_output["timestamp"]

        if not isinstance(probability, (int, float)):
            raise DecisionInputError("probability must be numeric")
        if not 0.0 <= float(probability) <= 1.0:
            raise DecisionInputError("probability must be between 0 and 1")

        if not isinstance(confidence, (int, float)):
            raise DecisionInputError("confidence must be numeric")
        if not 0.0 <= float(confidence) <= 1.0:
            raise DecisionInputError("confidence must be between 0 and 1")

        if not isinstance(model_version, str) or not model_version.strip():
            raise DecisionInputError("model_version must be a non-empty string")
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise DecisionInputError("timestamp must be a non-empty string")

        return predictor_output

    @staticmethod
    def _map_decision(probability: float) -> str:
        if 0.45 <= probability <= 0.55:
            return "HOLD"
        if probability > 0.55:
            return "BUY"
        return "SELL"

    @staticmethod
    def _map_confidence_level(distance: float) -> str:
        if distance < 0.05:
            return "very_low"
        if distance < 0.10:
            return "low"
        if distance < 0.20:
            return "moderate"
        if distance < 0.30:
            return "high"
        return "very_high"
