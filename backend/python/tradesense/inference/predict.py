"""Production inference engine for calibrated model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
import logging
import time

import joblib
import numpy as np
import pandas as pd

from tradesense.inference.decision_engine import TradeSenseDecisionEngine

logger = logging.getLogger(__name__)

_REQUIRED_BUNDLE_KEYS = ("calibrated_model", "feature_names", "metadata")
_LEGACY_BUNDLE_KEYS = ("model", "calibrator", "feature_names", "calibration_meta")
_MAX_PREDICTION_LATENCY_MS = 50.0


class BundleIntegrityError(RuntimeError):
    """Raised when a persisted model bundle is incomplete or invalid."""


class FeatureSchemaError(ValueError):
    """Raised when inference features fail strict schema validation."""


@dataclass(frozen=True)
class _LegacyCalibratedModelAdapter:
    """Adapts legacy (model + calibrator) bundles to calibrated_model semantics."""

    model: Any
    calibrator: Any

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw_probabilities = self.model.predict_proba(features)
        raw = np.asarray(raw_probabilities, dtype=float)

        if raw.ndim != 2 or raw.shape[1] < 2:
            raise BundleIntegrityError(
                "Legacy model.predict_proba must return a 2-column probability array"
            )

        calibrated_pos = self.calibrator.predict_proba(raw[:, 1].reshape(-1, 1))[:, 1]
        calibrated_neg = 1.0 - calibrated_pos
        return np.column_stack([calibrated_neg, calibrated_pos])


class TradeSensePredictor:
    """Load a trained model bundle and produce calibrated production predictions."""

    def __init__(self, bundle_path: str | Path | None = None) -> None:
        self.bundle_path = self._resolve_bundle_path(bundle_path)
        bundle = joblib.load(self.bundle_path)
        normalized_bundle = self._normalize_bundle(bundle)

        self.calibrated_model = normalized_bundle["calibrated_model"]
        self.feature_names = normalized_bundle["feature_names"]
        self.metadata = normalized_bundle["metadata"]
        self.model_version = str(
            self.metadata.get("model_version")
            or self.metadata.get("version")
            or "unknown"
        )

        logger.info("Model loaded successfully from %s", self.bundle_path)

    def predict_from_features(self, feature_row: pd.DataFrame) -> dict:
        """Run calibrated inference for a single feature row."""

        self._validate_feature_row(feature_row)

        start = time.perf_counter()
        probabilities = self.calibrated_model.predict_proba(feature_row)
        calibrated_probability = self._extract_class_1_probability(probabilities)

        prediction = 1 if calibrated_probability >= 0.5 else 0
        confidence = (
            calibrated_probability if prediction == 1 else 1.0 - calibrated_probability
        )

        latency_ms = (time.perf_counter() - start) * 1000.0
        if latency_ms > _MAX_PREDICTION_LATENCY_MS:
            logger.warning(
                "Prediction latency %.3f ms exceeded %.1f ms threshold",
                latency_ms,
                _MAX_PREDICTION_LATENCY_MS,
            )

        timestamp = datetime.now(tz=UTC).isoformat()

        result = {
            "probability": calibrated_probability,
            "prediction": prediction,
            "confidence": confidence,
            "model_version": self.model_version,
            "timestamp": timestamp,
        }

        logger.info(
            "Prediction generated successfully for model_version=%s",
            self.model_version,
        )

        return result

    def predict_and_decide(
        self,
        feature_row: pd.DataFrame,
        decision_engine: TradeSenseDecisionEngine | None = None,
    ) -> dict:
        """Run calibrated inference and convert it to a trade decision."""

        prediction = self.predict_from_features(feature_row)
        engine = decision_engine or TradeSenseDecisionEngine()
        return engine.decide_from_prediction(prediction)

    @staticmethod
    def _resolve_bundle_path(bundle_path: str | Path | None) -> Path:
        if bundle_path is not None:
            path = Path(bundle_path)
            if not path.exists():
                raise FileNotFoundError(f"Model bundle not found at: {path}")
            return path

        models_dir = Path(__file__).resolve().parents[1] / "models"
        candidates = [
            models_dir / "model_bundle.pkl",
            models_dir / "xgboost.joblib",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Model bundle not found. Expected one of: "
            f"{[str(path) for path in candidates]}"
        )

    @staticmethod
    def _normalize_bundle(bundle: Any) -> dict:
        if not isinstance(bundle, Mapping):
            raise BundleIntegrityError("Model bundle must be a mapping/dict")

        if all(key in bundle for key in _REQUIRED_BUNDLE_KEYS):
            normalized = {
                "calibrated_model": bundle["calibrated_model"],
                "feature_names": bundle["feature_names"],
                "metadata": bundle["metadata"],
            }
        elif all(key in bundle for key in _LEGACY_BUNDLE_KEYS):
            normalized = {
                "calibrated_model": _LegacyCalibratedModelAdapter(
                    model=bundle["model"],
                    calibrator=bundle["calibrator"],
                ),
                "feature_names": bundle["feature_names"],
                "metadata": bundle["calibration_meta"],
            }
        else:
            missing = [key for key in _REQUIRED_BUNDLE_KEYS if key not in bundle]
            raise BundleIntegrityError(
                "Bundle missing required keys "
                f"{missing}. Required keys: {list(_REQUIRED_BUNDLE_KEYS)}"
            )

        calibrated_model = normalized["calibrated_model"]
        feature_names = normalized["feature_names"]
        metadata = normalized["metadata"]

        if not hasattr(calibrated_model, "predict_proba"):
            raise BundleIntegrityError(
                "calibrated_model is invalid: missing predict_proba"
            )

        if not isinstance(feature_names, list) or not feature_names:
            raise BundleIntegrityError(
                "feature_names is invalid: expected non-empty list"
            )

        invalid_names = [name for name in feature_names if not isinstance(name, str) or not name]
        if invalid_names:
            raise BundleIntegrityError("feature_names contains invalid entries")

        if not isinstance(metadata, Mapping):
            raise BundleIntegrityError("metadata is invalid: expected mapping/dict")

        return {
            "calibrated_model": calibrated_model,
            "feature_names": list(feature_names),
            "metadata": dict(metadata),
        }

    def _validate_feature_row(self, feature_row: pd.DataFrame) -> None:
        if not isinstance(feature_row, pd.DataFrame):
            raise TypeError("feature_row must be a pandas DataFrame")
        if feature_row.shape[0] != 1:
            raise FeatureSchemaError("feature_row must contain exactly one row")

        expected = self.feature_names
        received = list(feature_row.columns)

        if received != expected:
            missing = [name for name in expected if name not in received]
            extra = [name for name in received if name not in expected]

            if missing or extra:
                logger.error("Schema validation failed")
                raise FeatureSchemaError(
                    "Feature schema mismatch: "
                    f"missing={missing}, extra={extra}"
                )

            logger.error("Schema validation failed")
            raise FeatureSchemaError(
                "Feature column order mismatch: input columns contain same names "
                "but in a different order"
            )

        if feature_row.isna().any(axis=None):
            raise FeatureSchemaError("feature_row contains NaN values")

        logger.info("Schema validation passed")

    @staticmethod
    def _extract_class_1_probability(probabilities: Any) -> float:
        array = np.asarray(probabilities, dtype=float)

        if array.ndim == 1 and array.size == 2:
            class_1_probability = float(array[1])
        elif array.ndim == 2 and array.shape[0] == 1 and array.shape[1] >= 2:
            class_1_probability = float(array[0, 1])
        else:
            raise BundleIntegrityError(
                "calibrated_model.predict_proba returned invalid shape; "
                "expected (1, 2) probabilities"
            )

        if not 0.0 <= class_1_probability <= 1.0:
            raise BundleIntegrityError("Calibrated probability is out of [0, 1] bounds")

        return class_1_probability
