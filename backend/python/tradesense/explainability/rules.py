# tradesense/explainability/rules.py
"""Rule-based explanation builder for TradeSense."""

from __future__ import annotations

from typing import Dict, List

from tradesense.explainability.templates import explain_feature


def build_structured_explanation(
    feature_attributions: Dict[str, float],
    feature_values: Dict[str, float],
    confidence_reason: str,
    trend_state: int,
    momentum_state: int,
    risk_state: int,
    max_drivers: int = 3,
) -> Dict[str, List[str]]:
    if not isinstance(feature_attributions, dict):
        raise ValueError("feature_attributions must be a dict")
    if not isinstance(feature_values, dict):
        raise ValueError("feature_values must be a dict")

    sorted_items = sorted(
        feature_attributions.items(),
        key=lambda item: (-abs(item[1]), item[0]),
    )

    positive: List[str] = []
    negative: List[str] = []
    for name, contribution in sorted_items:
        if contribution > 0 and len(positive) < max_drivers:
            positive.append(_explain(name, feature_values))
        elif contribution < 0 and len(negative) < max_drivers:
            negative.append(_explain(name, feature_values))
        if len(positive) >= max_drivers and len(negative) >= max_drivers:
            break

    confidence_modifiers = build_confidence_modifiers(
        confidence_reason=confidence_reason,
        trend_state=trend_state,
        momentum_state=momentum_state,
        risk_state=risk_state,
    )

    return {
        "key_drivers": positive,
        "negative_factors": negative,
        "confidence_modifiers": confidence_modifiers,
    }


def build_confidence_modifiers(
    confidence_reason: str,
    trend_state: int,
    momentum_state: int,
    risk_state: int,
) -> List[str]:
    modifiers: List[str] = []
    if isinstance(confidence_reason, str) and confidence_reason.strip():
        modifiers.append(confidence_reason.strip())

    if risk_state == 2:
        modifiers.append("Elevated risk regime")
    if signals_conflict(trend_state, momentum_state):
        modifiers.append("Trend and momentum conflict")

    return _dedupe(modifiers)


def build_risk_notes(trend_state: int, momentum_state: int, risk_state: int) -> List[str]:
    notes: List[str] = []
    if risk_state == 2:
        notes.append("Elevated risk regime")
    if signals_conflict(trend_state, momentum_state):
        notes.append("Trend and momentum conflict")
    return notes


def build_model_honesty(confidence_level: str, conflict: bool) -> str:
    parts: List[str] = []
    if confidence_level == "low":
        parts.append("Confidence is low based on probability strength.")
    if conflict:
        parts.append("Signals conflict, reducing confidence.")
    if not parts:
        parts.append("Confidence is aligned with probability strength.")
    return " ".join(parts)


def signals_conflict(trend_state: int, momentum_state: int) -> bool:
    return trend_state != 0 and trend_state * momentum_state == -1


def _explain(name: str, feature_values: Dict[str, float]) -> str:
    value = feature_values.get(name, 0.0)
    return explain_feature(name, float(value))


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered
