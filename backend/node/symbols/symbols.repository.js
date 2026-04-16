const fs = require('fs/promises');
const path = require('path');

const DATA_DIR = path.resolve(__dirname, '..', 'data', 'symbols');
const MARKET_MASTER_FILE = path.join(DATA_DIR, 'market_master.json');

let cachedPayload = null;
let cachedMtimeMs = null;

function createHttpError(status, message) {
  const error = new Error(message);
  error.status = status;
  return error;
}

function normalizeInstrument(rawInstrument) {
  if (!rawInstrument || typeof rawInstrument !== 'object' || Array.isArray(rawInstrument)) {
    return null;
  }

  const normalized = String(rawInstrument.normalized || '').trim().toUpperCase();
  const symbol = String(rawInstrument.symbol || '').trim().toUpperCase();
  const displayName = String(rawInstrument.display_name || '').trim();

  if (!normalized || !symbol || !displayName) {
    return null;
  }

  return {
    id: String(rawInstrument.id || `${rawInstrument.market || 'UNKNOWN'}:${normalized}`),
    symbol,
    normalized,
    display_name: displayName,
    market: String(rawInstrument.market || '').trim().toUpperCase(),
    exchange: String(rawInstrument.exchange || '').trim().toUpperCase(),
    instrument_type: String(rawInstrument.instrument_type || '').trim() || 'Equity',
    country: String(rawInstrument.country || '').trim().toUpperCase(),
    search_terms: Array.isArray(rawInstrument.search_terms)
      ? [...new Set(
          rawInstrument.search_terms
            .filter((term) => typeof term === 'string')
            .map((term) => term.trim().toUpperCase())
            .filter(Boolean)
        )]
      : [],
    source: typeof rawInstrument.source === 'string' ? rawInstrument.source : null,
  };
}

async function loadMarketMasterFromDisk() {
  let raw;
  try {
    raw = await fs.readFile(MARKET_MASTER_FILE, 'utf8');
  } catch (error) {
    if (error?.code === 'ENOENT') {
      throw createHttpError(
        500,
        'Market master is missing. Run `npm run build:market-master` in backend/node.'
      );
    }
    throw error;
  }

  let parsed;
  try {
    parsed = JSON.parse(raw || '{}');
  } catch (error) {
    throw createHttpError(500, 'Market master is invalid JSON.');
  }

  const instruments = Array.isArray(parsed?.instruments)
    ? parsed.instruments.map(normalizeInstrument).filter(Boolean)
    : [];

  if (instruments.length === 0) {
    throw createHttpError(500, 'Market master is empty.');
  }

  return {
    generated_at: parsed.generated_at || null,
    counts: parsed.counts || {},
    instruments,
  };
}

async function getMarketMaster() {
  const stats = await fs.stat(MARKET_MASTER_FILE).catch((error) => {
    if (error?.code === 'ENOENT') {
      throw createHttpError(
        500,
        'Market master is missing. Run `npm run build:market-master` in backend/node.'
      );
    }
    throw error;
  });

  if (cachedPayload && cachedMtimeMs === stats.mtimeMs) {
    return cachedPayload;
  }

  const payload = await loadMarketMasterFromDisk();
  cachedPayload = payload;
  cachedMtimeMs = stats.mtimeMs;
  return payload;
}

module.exports = {
  getMarketMaster,
};
