# tradesense/rag/store.py
"""Local RAG memory store backed by a vector index.

Stores structured insight summaries only (no raw market data).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


_EMBEDDING_DIM = 384
_VECTOR = HashingVectorizer(
    n_features=_EMBEDDING_DIM,
    alternate_sign=False,
    norm="l2",
)


class _NumpyIndex:
    def __init__(self, vectors: Optional[np.ndarray] = None):
        if vectors is None:
            vectors = np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
        self.vectors = vectors.astype(np.float32, copy=False)

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        if vectors.size == 0:
            return
        vectors = vectors.astype(np.float32, copy=False)
        if self.vectors.size == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query: np.ndarray, k: int):
        if self.ntotal == 0:
            distances = np.zeros((1, k), dtype=np.float32)
            indices = -np.ones((1, k), dtype=np.int64)
            return distances, indices

        sims = self.vectors @ query.T
        sims = sims.reshape(-1)
        k = min(k, sims.size)
        order = np.argsort(-sims)[:k]

        distances = np.zeros((1, k), dtype=np.float32)
        indices = -np.ones((1, k), dtype=np.int64)
        distances[0, :k] = sims[order]
        indices[0, :k] = order
        return distances, indices


class RagStore:
    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = (root_dir or _default_store_dir()).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def store_insight(self, insight: dict) -> None:
        record = _normalize_insight(insight)
        symbol = record["symbol"]
        records = self._load_records(symbol)
        records.append(record)

        vectors = self._load_vectors(symbol, records[:-1])
        vector = _embed_texts([_record_to_text(record)])
        vectors = np.vstack([vectors, vector]) if vectors.size else vector

        self._save_records(symbol, records)
        self._save_index(symbol, vectors)

    def retrieve_context(self, symbol: str, limit: int = 5) -> List[dict]:
        symbol = _normalize_symbol(symbol)
        if not symbol:
            return []

        records = self._load_records(symbol)
        if not records:
            return []

        query = _embed_texts([f"symbol {symbol}"])
        index = self._load_index(symbol, records)
        distances, indices = index.search(query, min(limit, len(records)))

        candidates = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(records):
                continue
            record = records[int(idx)]
            candidates.append((float(score), _timestamp_key(record), int(idx), record))

        candidates.sort(key=lambda item: (item[1], item[0], item[2]), reverse=True)
        return [item[3] for item in candidates[:limit]]

    def _load_records(self, symbol: str) -> List[dict]:
        path = self._metadata_path(symbol)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def _save_records(self, symbol: str, records: List[dict]) -> None:
        path = self._metadata_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, ensure_ascii=True, sort_keys=True), encoding="utf-8")

    def _load_index(self, symbol: str, records: List[dict]):
        if faiss is not None:
            index_path = self._index_path(symbol)
            if index_path.exists():
                index = faiss.read_index(str(index_path))
                if index.ntotal == len(records):
                    return index
            vectors = _embed_texts([_record_to_text(r) for r in records]) if records else np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
            return _build_faiss_index(vectors)

        vectors = self._load_vectors(symbol, records)
        return _NumpyIndex(vectors)

    def _load_vectors(self, symbol: str, records: List[dict]) -> np.ndarray:
        if faiss is not None:
            index_path = self._index_path(symbol)
            if index_path.exists():
                index = faiss.read_index(str(index_path))
                if index.ntotal == len(records):
                    return _extract_vectors(index)
            vectors = _embed_texts([_record_to_text(r) for r in records]) if records else np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
            return vectors

        vector_path = self._vector_path(symbol)
        if vector_path.exists():
            vectors = np.load(vector_path)
            if vectors.shape[0] == len(records):
                return vectors.astype(np.float32, copy=False)
        vectors = _embed_texts([_record_to_text(r) for r in records]) if records else np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
        return vectors

    def _save_index(self, symbol: str, vectors: np.ndarray) -> None:
        vectors = vectors.astype(np.float32, copy=False)
        if faiss is not None:
            index = _build_faiss_index(vectors)
            index_path = self._index_path(symbol)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(index_path))
            return

        vector_path = self._vector_path(symbol)
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vector_path, vectors)

    def _metadata_path(self, symbol: str) -> Path:
        return self.root_dir / symbol / "metadata.json"

    def _index_path(self, symbol: str) -> Path:
        return self.root_dir / symbol / "index.faiss"

    def _vector_path(self, symbol: str) -> Path:
        return self.root_dir / symbol / "vectors.npy"


_STORE: Optional[RagStore] = None


def _get_store() -> RagStore:
    global _STORE
    root = _default_store_dir().resolve()
    if _STORE is None or _STORE.root_dir != root:
        _STORE = RagStore(root)
    return _STORE


def store_insight(insight: dict) -> None:
    """Persist a structured insight summary for later retrieval."""
    _get_store().store_insight(insight)


def _default_store_dir() -> Path:
    base = Path(__file__).resolve().parents[1] / "rag_store"
    return Path(os.getenv("TRADESENSE_RAG_DIR", base)).resolve()


def _normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        return ""
    return symbol.strip().upper()


def _normalize_list(value) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    cleaned: List[str] = []
    for item in value:
        if not isinstance(item, str):
            item = str(item)
        stripped = item.strip()
        if stripped:
            cleaned.append(stripped)
    return cleaned


def _normalize_insight(insight: dict) -> dict:
    if not isinstance(insight, dict):
        raise ValueError("insight must be a dict")

    symbol = _normalize_symbol(insight.get("symbol", ""))
    if not symbol:
        raise ValueError("insight.symbol must be a non-empty string")

    timestamp = insight.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        timestamp = datetime.now(timezone.utc).isoformat()

    probability = float(insight.get("probability", 0.0))
    confidence_level = str(insight.get("confidence_level", "unknown"))
    key_drivers = _normalize_list(insight.get("key_drivers"))
    risk_notes = _normalize_list(insight.get("risk_notes"))

    sentiment = insight.get("sentiment")
    if sentiment is not None and not isinstance(sentiment, dict):
        sentiment = None

    news_headlines = _normalize_list(insight.get("news_headlines"))
    if not news_headlines:
        news_headlines = None

    record = {
        "symbol": symbol,
        "timestamp": timestamp,
        "probability": probability,
        "confidence_level": confidence_level,
        "sentiment": sentiment,
        "key_drivers": key_drivers,
        "risk_notes": risk_notes,
        "news_headlines": news_headlines,
        "summary": insight.get("summary"),
        "market_context": insight.get("market_context"),
    }

    return record


def _record_to_text(record: dict) -> str:
    parts = [
        f"symbol {record.get('symbol', '')}",
        f"probability {record.get('probability', '')}",
        f"confidence {record.get('confidence_level', '')}",
    ]
    summary = record.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts.append(f"summary {summary.strip()}")

    market_context = record.get("market_context")
    if isinstance(market_context, dict):
        context_bits = [
            str(market_context.get("trend", "")),
            str(market_context.get("momentum", "")),
            str(market_context.get("volatility", "")),
        ]
        context_bits = [bit for bit in context_bits if bit and bit != "None"]
        if context_bits:
            parts.append("market_context " + " ".join(context_bits))

    key_drivers = record.get("key_drivers") or []
    if key_drivers:
        parts.append("drivers " + " ".join(key_drivers))

    risk_notes = record.get("risk_notes") or []
    if risk_notes:
        parts.append("risks " + " ".join(risk_notes))

    sentiment = record.get("sentiment")
    if isinstance(sentiment, dict):
        bias = sentiment.get("sentiment_bias") or ""
        score = sentiment.get("sentiment_score")
        strength = sentiment.get("sentiment_strength") or ""
        parts.append(f"sentiment {bias} {strength} {score}")

    news_headlines = record.get("news_headlines") or []
    if news_headlines:
        parts.append("news " + " ".join(news_headlines))

    return " | ".join(parts)


def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
    matrix = _VECTOR.transform(texts)
    vectors = matrix.astype(np.float32).toarray()
    return vectors


def _build_faiss_index(vectors: np.ndarray):
    index = faiss.IndexFlatIP(_EMBEDDING_DIM)
    if vectors.size:
        index.add(vectors.astype(np.float32, copy=False))
    return index


def _extract_vectors(index) -> np.ndarray:
    if index.ntotal == 0:
        return np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
    vectors = np.zeros((index.ntotal, _EMBEDDING_DIM), dtype=np.float32)
    if hasattr(index, "reconstruct_n"):
        index.reconstruct_n(0, index.ntotal, vectors)
        return vectors
    for i in range(index.ntotal):
        vectors[i, :] = index.reconstruct(i)
    return vectors


def _timestamp_key(record: dict) -> float:
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        return 0.0
    cleaned = timestamp.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()
