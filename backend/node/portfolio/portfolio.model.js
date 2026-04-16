const { randomUUID } = require('crypto');

const MAX_TICKER_LENGTH = 20;
const TICKER_PATTERN = /^[A-Z0-9][A-Z0-9.\-]{0,19}$/;
const TRADE_SIDES = new Set(['BUY', 'SELL', 'SHORT', 'COVER']);

function createHttpError(status, message) {
  const error = new Error(message);
  error.status = status;
  return error;
}

function normalizeTicker(value) {
  if (typeof value !== 'string') {
    throw createHttpError(400, 'ticker is required');
  }

  const normalized = value.trim().toUpperCase();
  if (!normalized) {
    throw createHttpError(400, 'ticker is required');
  }

  if (normalized.length > MAX_TICKER_LENGTH || !TICKER_PATTERN.test(normalized)) {
    throw createHttpError(400, 'ticker format is invalid');
  }

  return normalized;
}

function normalizePositiveNumber(value, fieldName) {
  const normalized = Number(value);
  if (!Number.isFinite(normalized) || normalized <= 0) {
    throw createHttpError(400, `${fieldName} must be a positive number`);
  }
  return normalized;
}

function normalizeOptionalPositiveNumber(value) {
  if (value === undefined || value === null || value === '') {
    return null;
  }

  const normalized = Number(value);
  if (!Number.isFinite(normalized) || normalized <= 0) {
    return null;
  }

  return normalized;
}

function resolveInstrumentMetadata(ticker) {
  const normalizedTicker = normalizeTicker(ticker);

  if (normalizedTicker.endsWith('.NS')) {
    return {
      market: 'IN',
      exchange: 'NSE',
      instrument_currency: 'INR',
      instrument_type: 'Equity',
    };
  }

  if (normalizedTicker.endsWith('.BO')) {
    return {
      market: 'IN',
      exchange: 'BSE',
      instrument_currency: 'INR',
      instrument_type: 'Equity',
    };
  }

  if (normalizedTicker.startsWith('^')) {
    return {
      market: normalizedTicker === '^GSPC' ? 'US' : 'IN',
      exchange:
        normalizedTicker === '^NSEI'
          ? 'NSE'
          : normalizedTicker === '^BSESN'
            ? 'BSE'
            : 'INDEX',
      instrument_currency: normalizedTicker === '^GSPC' ? 'USD' : 'INR',
      instrument_type: 'Index',
    };
  }

  return {
    market: 'US',
    exchange: 'US',
    instrument_currency: 'USD',
    instrument_type: 'Equity',
  };
}

function buildHolding(input) {
  const payload = input || {};
  const ticker = normalizeTicker(payload.ticker);
  const metadata = resolveInstrumentMetadata(ticker);

  return {
    id: randomUUID(),
    ticker,
    symbol: ticker,
    display_name: payload.display_name || ticker,
    market: metadata.market,
    shares: normalizePositiveNumber(payload.shares, 'shares'),
    buy_price: normalizePositiveNumber(payload.buy_price, 'buy_price'),
    exchange: metadata.exchange,
    instrument_type: metadata.instrument_type,
    instrument_currency: metadata.instrument_currency,
    price_native: null,
    fx_rate_to_base: null,
    price_base: null,
    market_value_base: null,
    added_at: new Date().toISOString(),
  };
}

function normalizeStoredHolding(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null;
  }

  const id = typeof raw.id === 'string' ? raw.id.trim() : '';
  if (!id) {
    return null;
  }

  let ticker;
  try {
    ticker = normalizeTicker(raw.ticker);
  } catch (error) {
    return null;
  }

  const shares = Number(raw.shares);
  const buyPrice = Number(raw.buy_price);
  if (!Number.isFinite(shares) || shares <= 0 || !Number.isFinite(buyPrice) || buyPrice <= 0) {
    return null;
  }

  const addedAt =
    typeof raw.added_at === 'string' && raw.added_at.trim()
      ? raw.added_at
      : new Date().toISOString();
  const metadata = resolveInstrumentMetadata(ticker);

  const exchange =
    typeof raw.exchange === 'string' && raw.exchange.trim()
      ? raw.exchange.trim().toUpperCase()
      : metadata.exchange;
  const instrumentCurrency =
    typeof raw.instrument_currency === 'string' && raw.instrument_currency.trim()
      ? raw.instrument_currency.trim().toUpperCase()
      : metadata.instrument_currency;

  const market =
    typeof raw.market === 'string' && raw.market.trim()
      ? raw.market.trim().toUpperCase()
      : metadata.market;
  const instrumentType =
    typeof raw.instrument_type === 'string' && raw.instrument_type.trim()
      ? raw.instrument_type.trim()
      : metadata.instrument_type;
  const displayName =
    typeof raw.display_name === 'string' && raw.display_name.trim()
      ? raw.display_name.trim()
      : ticker;

  return {
    id,
    ticker,
    symbol:
      typeof raw.symbol === 'string' && raw.symbol.trim() ? raw.symbol.trim().toUpperCase() : ticker,
    display_name: displayName,
    market,
    shares,
    buy_price: buyPrice,
    exchange,
    instrument_type: instrumentType,
    instrument_currency: instrumentCurrency,
    price_native: normalizeOptionalPositiveNumber(raw.price_native),
    fx_rate_to_base: normalizeOptionalPositiveNumber(raw.fx_rate_to_base),
    price_base: normalizeOptionalPositiveNumber(raw.price_base),
    market_value_base: normalizeOptionalPositiveNumber(raw.market_value_base),
    added_at: addedAt,
  };
}

function normalizeTradeSide(value) {
  if (typeof value !== 'string' || !value.trim()) {
    throw createHttpError(400, 'trade side is required');
  }

  const normalized = value.trim().toUpperCase();
  if (!TRADE_SIDES.has(normalized)) {
    throw createHttpError(400, 'trade side must be BUY, SELL, SHORT, or COVER');
  }

  return normalized;
}

function buildTrade(input) {
  const payload = input || {};
  const ticker = normalizeTicker(payload.ticker ?? payload.normalized ?? payload.symbol);
  const metadata = resolveInstrumentMetadata(ticker);
  const side = normalizeTradeSide(payload.side);
  const occurredAt =
    typeof payload.occurred_at === 'string' && payload.occurred_at.trim()
      ? payload.occurred_at.trim()
      : new Date().toISOString();

  return {
    id: randomUUID(),
    ticker,
    symbol:
      typeof payload.symbol === 'string' && payload.symbol.trim()
        ? payload.symbol.trim().toUpperCase()
        : ticker.replace(/\.(NS|BO)$/i, ''),
    normalized: ticker,
    display_name:
      typeof payload.display_name === 'string' && payload.display_name.trim()
        ? payload.display_name.trim()
        : ticker,
    market:
      typeof payload.market === 'string' && payload.market.trim()
        ? payload.market.trim().toUpperCase()
        : metadata.market,
    exchange:
      typeof payload.exchange === 'string' && payload.exchange.trim()
        ? payload.exchange.trim().toUpperCase()
        : metadata.exchange,
    instrument_type:
      typeof payload.instrument_type === 'string' && payload.instrument_type.trim()
        ? payload.instrument_type.trim()
        : metadata.instrument_type,
    instrument_currency:
      typeof payload.instrument_currency === 'string' && payload.instrument_currency.trim()
        ? payload.instrument_currency.trim().toUpperCase()
        : metadata.instrument_currency,
    side,
    quantity: normalizePositiveNumber(payload.quantity, 'quantity'),
    price: normalizePositiveNumber(payload.price, 'price'),
    note:
      typeof payload.note === 'string' && payload.note.trim() ? payload.note.trim() : null,
    source:
      typeof payload.source === 'string' && payload.source.trim() ? payload.source.trim() : 'manual',
    occurred_at: occurredAt,
  };
}

function normalizeStoredTrade(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null;
  }

  const id = typeof raw.id === 'string' && raw.id.trim() ? raw.id.trim() : '';
  if (!id) {
    return null;
  }

  try {
    const trade = buildTrade({
      ...raw,
      occurred_at:
        typeof raw.occurred_at === 'string' && raw.occurred_at.trim()
          ? raw.occurred_at.trim()
          : new Date().toISOString(),
    });

    return {
      ...trade,
      id,
    };
  } catch (error) {
    return null;
  }
}

module.exports = {
  TRADE_SIDES,
  buildTrade,
  buildHolding,
  createHttpError,
  normalizeStoredTrade,
  normalizeTradeSide,
  normalizeStoredHolding,
  normalizeTicker,
  resolveInstrumentMetadata,
};
