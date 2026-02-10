# tradesense/explainer/prompt_builder.py
"""Prompt construction for LLM explanations (interpretive only)."""

from __future__ import annotations

import json
from typing import Optional


def build_explanation_prompt(insight: dict, context_summary: Optional[str]) -> str:
    """Build a structured prompt for the explanation LLM."""
    if not isinstance(insight, dict):
        raise ValueError("insight must be a dict")

    structured_explanation = insight.get("structured_explanation")
    sentiment = insight.get("sentiment") if isinstance(insight.get("sentiment"), dict) else None
    history = context_summary.strip() if isinstance(context_summary, str) and context_summary.strip() else None

    payload = {
        "structured_explanation": structured_explanation,
        "sentiment": sentiment,
        "historical_context": history,
    }

    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)

    return (
        "You are an analyst explaining model outputs in plain language.\n"
        "Use ONLY the provided structured explanation. Do NOT add new signals or predictions.\n"
        "Do NOT predict prices. Do NOT provide financial advice.\n"
        "Output must be valid JSON with keys: summary, narrative, disclaimer.\n"
        "Style: neutral, analytical, concise.\n"
        "If a field is null, state that it is unavailable.\n"
        "<INSIGHT>\n"
        f"{serialized}\n"
        "</INSIGHT>\n"
    )

