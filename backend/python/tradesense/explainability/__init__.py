# tradesense/explainability/__init__.py
"""Deterministic explainability utilities for TradeSense (Phase 10)."""

from tradesense.explainability.attribution import (
    compute_attributions_from_importance,
    compute_attributions_from_model,
    normalize_attributions,
)
from tradesense.explainability.rules import (
    build_structured_explanation,
    build_confidence_modifiers,
)

__all__ = [
    "compute_attributions_from_importance",
    "compute_attributions_from_model",
    "normalize_attributions",
    "build_structured_explanation",
    "build_confidence_modifiers",
]
