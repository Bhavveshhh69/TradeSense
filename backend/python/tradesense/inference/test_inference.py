"""Phase 13A inference smoke test for production predictor."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tradesense.data_provider import get_market_data
from tradesense.features import build_feature_matrix
from tradesense.inference.predict import TradeSensePredictor


def load_latest_feature_row(symbol: str, lookback_days: int = 220):
    end_date = datetime.now(tz=UTC).date()
    start_date = end_date - timedelta(days=lookback_days)

    market_data = get_market_data(
        [symbol],
        start_date.isoformat(),
        end_date.isoformat(),
        interval="1d",
    )

    symbol_data = market_data.get(symbol)
    if symbol_data is None or symbol_data.empty:
        raise RuntimeError(f"No market data available for symbol: {symbol}")

    feature_matrix = build_feature_matrix(symbol_data)
    if feature_matrix.empty:
        raise RuntimeError(f"Feature pipeline returned no rows for symbol: {symbol}")

    return feature_matrix.tail(1)


def main() -> None:
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"

    predictor = TradeSensePredictor()
    feature_row = load_latest_feature_row(symbol)
    prediction = predictor.predict_from_features(feature_row)

    output = {
        "symbol": symbol,
        **prediction,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
