"""FinBERT sentiment analysis (CPU-only).

Loads the ProsusAI/finbert model once at module import time.
"""

from __future__ import annotations

from typing import List

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_NAME = "ProsusAI/finbert"

# Load model and tokenizer once at module import time (CPU-only).
_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
_MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
_PIPELINE = pipeline(
    "text-classification",
    model=_MODEL,
    tokenizer=_TOKENIZER,
    device=-1,
    return_all_scores=False,
)


def analyze_texts(texts: List[str]) -> List[dict]:
    """Analyze a list of news texts and return FinBERT sentiment results.

    Args:
        texts: List of non-empty strings.

    Returns:
        List of dicts: {"label": "positive|neutral|negative", "score": float}
    """
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be a non-empty list of strings")

    results: List[dict] = []
    for text in texts:
        if not isinstance(text, str):
            raise ValueError("texts must be a non-empty list of strings")
        stripped = text.strip()
        if not stripped:
            raise ValueError("texts must be a non-empty list of strings")

        prediction = _PIPELINE(stripped, truncation=True)
        if isinstance(prediction, list):
            prediction = prediction[0]

        label = str(prediction.get("label", "neutral")).lower()
        score = float(prediction.get("score", 0.0))
        results.append({"label": label, "score": score})

    return results
