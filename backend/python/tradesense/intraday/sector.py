from __future__ import annotations

from dataclasses import dataclass

from .contracts import SectorResolution
from .market import resolve_market


US_SECTOR_MAP: dict[str, tuple[str, tuple[str, ...]]] = {
    "AAPL": ("Technology", ("MSFT", "NVDA", "META")),
    "MSFT": ("Technology", ("AAPL", "NVDA", "META")),
    "NVDA": ("Technology", ("AAPL", "MSFT", "AMD")),
    "META": ("Technology", ("AAPL", "MSFT", "GOOGL")),
    "AMZN": ("Consumer Discretionary", ("TSLA", "HD", "LOW")),
}

IN_SECTOR_MAP: dict[str, tuple[str, tuple[str, ...]]] = {
    "RELIANCE": ("Energy", ("ONGC.NS", "IOC.NS", "BPCL.NS")),
    "TCS": ("Information Technology", ("INFY.NS", "WIPRO.NS", "HCLTECH.NS")),
    "INFY": ("Information Technology", ("TCS.NS", "WIPRO.NS", "HCLTECH.NS")),
    "HDFCBANK": ("Financial Services", ("ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS")),
    "ICICIBANK": ("Financial Services", ("HDFCBANK.NS", "SBIN.NS", "KOTAKBANK.NS")),
}


def _canonical_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    for suffix in (".NS", ".BO"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


@dataclass
class SectorResolver:
    def resolve(self, symbol: str) -> SectorResolution:
        market, _ = resolve_market(symbol)
        canonical = _canonical_symbol(symbol)
        mapping = US_SECTOR_MAP if market == "US" else IN_SECTOR_MAP
        resolved = mapping.get(canonical)
        if resolved is None:
            return SectorResolution(
                symbol=symbol,
                market=market,
                sector=None,
                peer_symbols=(),
                sector_available=False,
            )
        sector, peers = resolved
        filtered_peers = tuple(peer for peer in peers if peer.upper() != symbol.strip().upper())
        return SectorResolution(
            symbol=symbol,
            market=market,
            sector=sector,
            peer_symbols=filtered_peers,
            sector_available=True,
        )
