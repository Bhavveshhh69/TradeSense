# tradesense/explainer/__init__.py
"""LLM-assisted explanation layer for TradeSense."""

from tradesense.explainer.llm_client import generate_explanation
from tradesense.explainer.prompt_builder import build_explanation_prompt

__all__ = ["build_explanation_prompt", "generate_explanation"]
