const { randomUUID } = require('crypto');

const MAX_TICKER_LENGTH = 20;
const TICKER_PATTERN = /^[A-Z0-9][A-Z0-9.\-]{0,19}$/;

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
      exchange: 'NSE',
      instrument_currency: 'INR',
    };
  }

  if (normalizedTicker.endsWith('.BO')) {
    return {
      exchange: 'BSE',
      instrument_currency: 'INR',
    };
  }

  return {
    exchange: 'US',
    instrument_currency: 'USD',
  };
}

function buildHolding(input) {
  const payload = input || {};
  const ticker = normalizeTicker(payload.ticker);
  const metadata = resolveInstrumentMetadata(ticker);

  return {
    id: randomUUID(),
    ticker,
    shares: normalizePositiveNumber(payload.shares, 'shares'),
    buy_price: normalizePositiveNumber(payload.buy_price, 'buy_price'),
    exchange: metadata.exchange,
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

  return {
    id,
    ticker,
    shares,
    buy_price: buyPrice,
    exchange,
    instrument_currency: instrumentCurrency,
    price_native: normalizeOptionalPositiveNumber(raw.price_native),
    fx_rate_to_base: normalizeOptionalPositiveNumber(raw.fx_rate_to_base),
    price_base: normalizeOptionalPositiveNumber(raw.price_base),
    market_value_base: normalizeOptionalPositiveNumber(raw.market_value_base),
    added_at: addedAt,
  };
}

module.exports = {
  buildHolding,
  createHttpError,
  normalizeStoredHolding,
  normalizeTicker,
  resolveInstrumentMetadata,
};
