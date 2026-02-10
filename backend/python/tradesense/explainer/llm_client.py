# tradesense/explainer/llm_client.py
"""LLM client for generating interpretive explanations (OpenAI)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import httpx


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 20.0
DEFAULT_DISCLAIMER = (
    "This explanation is informational only and is not financial advice."
)


def generate_explanation(prompt: str) -> Dict[str, str]:
    """Call OpenAI Chat Completions and return a structured explanation."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to generate explanations")

    api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    max_tokens = _clamp_int(os.getenv("OPENAI_MAX_TOKENS"), DEFAULT_MAX_TOKENS, 1, 1024)
    temperature = _clamp_float(os.getenv("OPENAI_TEMPERATURE"), DEFAULT_TEMPERATURE, 0.0, 0.3)
    timeout = _clamp_float(os.getenv("OPENAI_TIMEOUT"), DEFAULT_TIMEOUT, 1.0, 60.0)

    payload = {
        "model": model,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": "You explain model outputs without making predictions or offering advice.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    content = _extract_content(data)
    parsed = _parse_json(content)
    return _normalize_explanation(parsed)


def _extract_content(payload: Dict[str, Any]) -> str:
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected LLM response format") from exc


def _parse_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("LLM response content is not text")
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = _strip_code_fences(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response was not valid JSON") from exc


def _strip_code_fences(text: str) -> str:
    parts = text.split("```")
    if len(parts) < 3:
        return text
    return parts[1].strip()


def _normalize_explanation(payload: Dict[str, Any]) -> Dict[str, str]:
    summary = _to_string(payload.get("summary"))
    narrative = _to_string(payload.get("narrative"))
    disclaimer = _to_string(payload.get("disclaimer")) or DEFAULT_DISCLAIMER
    if not summary:
        summary = "No summary provided."
    if not narrative:
        narrative = "No narrative provided."
    return {
        "summary": summary,
        "narrative": narrative,
        "disclaimer": disclaimer,
    }


def _to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _clamp_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))
