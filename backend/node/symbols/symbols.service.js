const repository = require('./symbols.repository');

function createHttpError(status, message, extra = {}) {
  const error = new Error(message);
  error.status = status;
  Object.assign(error, extra);
  return error;
}

function normalizeInputSymbol(inputSymbol) {
  if (typeof inputSymbol !== 'string') {
    throw createHttpError(400, 'symbol is required');
  }

  const normalized = inputSymbol.trim().toUpperCase();
  if (!normalized) {
    throw createHttpError(400, 'symbol is required');
  }

  return normalized;
}

function normalizeMarketFilter(inputValue) {
  if (typeof inputValue !== 'string' || !inputValue.trim()) {
    return null;
  }

  const normalized = inputValue.trim().toUpperCase();
  if (!['US', 'IN'].includes(normalized)) {
    throw createHttpError(400, 'market must be one of US or IN');
  }

  return normalized;
}

function normalizeKindFilter(inputValue) {
  if (typeof inputValue !== 'string' || !inputValue.trim()) {
    return null;
  }

  const normalized = inputValue.trim().toUpperCase();
  if (['EQUITY', 'EQUITIES', 'STOCK', 'STOCKS'].includes(normalized)) {
    return 'Equity';
  }
  if (['INDEX', 'INDICES'].includes(normalized)) {
    return 'Index';
  }

  throw createHttpError(400, 'kind must be one of equity or index');
}

function normalizeLimit(limitValue, defaultValue = 40) {
  if (limitValue === undefined || limitValue === null || limitValue === '') {
    return defaultValue;
  }

  const numericValue = Number(limitValue);
  if (!Number.isFinite(numericValue)) {
    throw createHttpError(400, 'limit must be a number');
  }

  const normalized = Math.trunc(numericValue);
  if (normalized < 1 || normalized > 100) {
    throw createHttpError(400, 'limit must be between 1 and 100');
  }

  return normalized;
}

function buildSearchBlob(instrument) {
  return [
    instrument.symbol,
    instrument.normalized,
    instrument.display_name,
    instrument.exchange,
    instrument.market,
    instrument.instrument_type,
    ...(Array.isArray(instrument.search_terms) ? instrument.search_terms : []),
  ]
    .filter((value) => typeof value === 'string' && value.trim())
    .join(' ')
    .toUpperCase();
}

function scoreInstrument(instrument, query) {
  const normalizedQuery = query.trim().toUpperCase();
  const searchBlob = buildSearchBlob(instrument);

  if (instrument.normalized === normalizedQuery) {
    return 140;
  }
  if (instrument.symbol === normalizedQuery) {
    return 130;
  }
  if (instrument.display_name.toUpperCase() === normalizedQuery) {
    return 120;
  }
  if (instrument.symbol.startsWith(normalizedQuery)) {
    return 100;
  }
  if (instrument.normalized.startsWith(normalizedQuery)) {
    return 95;
  }
  if (instrument.display_name.toUpperCase().startsWith(normalizedQuery)) {
    return 85;
  }
  if (searchBlob.includes(normalizedQuery)) {
    return 60;
  }
  return 0;
}

function groupLabelForInstrument(instrument) {
  if (instrument.instrument_type === 'Index') {
    return instrument.market === 'IN' ? 'India Indices' : 'US Indices';
  }

  return instrument.market === 'IN' ? 'India Equities' : 'US Equities';
}

async function getInstrumentCatalog() {
  const payload = await repository.getMarketMaster();
  return payload.instruments;
}

async function buildIndexes() {
  const instruments = await getInstrumentCatalog();
  const byNormalized = new Map();
  const bySymbol = new Map();
  const byDisplayName = new Map();

  for (const instrument of instruments) {
    byNormalized.set(instrument.normalized, instrument);

    const bySymbolList = bySymbol.get(instrument.symbol) || [];
    bySymbolList.push(instrument);
    bySymbol.set(instrument.symbol, bySymbolList);

    const displayKey = instrument.display_name.toUpperCase();
    const byDisplayNameList = byDisplayName.get(displayKey) || [];
    byDisplayNameList.push(instrument);
    byDisplayName.set(displayKey, byDisplayNameList);
  }

  return { instruments, byNormalized, bySymbol, byDisplayName };
}

async function resolveInstrument(inputSymbol) {
  const symbol = normalizeInputSymbol(inputSymbol);
  const { byNormalized, bySymbol, byDisplayName } = await buildIndexes();

  const exactNormalized = byNormalized.get(symbol);
  if (exactNormalized) {
    return {
      ...exactNormalized,
      changed: exactNormalized.normalized !== symbol,
      ambiguous: false,
    };
  }

  const rawSymbolMatches = bySymbol.get(symbol) || [];
  if (rawSymbolMatches.length === 1) {
    const instrument = rawSymbolMatches[0];
    return {
      ...instrument,
      changed: instrument.normalized !== symbol,
      ambiguous: false,
    };
  }

  if (rawSymbolMatches.length > 1) {
    throw createHttpError(409, `symbol ${symbol} is ambiguous; select a specific market instrument`, {
      matches: rawSymbolMatches,
    });
  }

  const displayNameMatches = byDisplayName.get(symbol) || [];
  if (displayNameMatches.length === 1) {
    const instrument = displayNameMatches[0];
    return {
      ...instrument,
      changed: true,
      ambiguous: false,
    };
  }

  if (displayNameMatches.length > 1) {
    throw createHttpError(409, `instrument ${symbol} is ambiguous; select a specific market instrument`, {
      matches: displayNameMatches,
    });
  }

  throw createHttpError(404, `symbol ${symbol} is unsupported in the current market master`);
}

async function normalizeSymbol(inputSymbol) {
  const instrument = await resolveInstrument(inputSymbol);
  return instrument.normalized;
}

async function validateSymbol(inputSymbol) {
  try {
    await resolveInstrument(inputSymbol);
    return true;
  } catch (error) {
    if (error?.status === 404 || error?.status === 409 || error?.status === 400) {
      return false;
    }
    throw error;
  }
}

async function searchSymbols({ query, market, kind, limit } = {}) {
  const normalizedQuery = typeof query === 'string' ? query.trim().toUpperCase() : '';
  if (!normalizedQuery) {
    return [];
  }

  const marketFilter = normalizeMarketFilter(market);
  const kindFilter = normalizeKindFilter(kind);
  const resultLimit = normalizeLimit(limit, 40);
  const catalog = await getInstrumentCatalog();

  return catalog
    .filter((instrument) => (marketFilter ? instrument.market === marketFilter : true))
    .filter((instrument) => (kindFilter ? instrument.instrument_type === kindFilter : true))
    .map((instrument) => ({
      instrument,
      score: scoreInstrument(instrument, normalizedQuery),
    }))
    .filter((item) => item.score > 0)
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      return left.instrument.normalized.localeCompare(right.instrument.normalized);
    })
    .slice(0, resultLimit)
    .map(({ instrument }) => ({
      ...instrument,
      group_label: groupLabelForInstrument(instrument),
    }));
}

module.exports = {
  getInstrumentCatalog,
  normalizeKindFilter,
  normalizeLimit,
  normalizeMarketFilter,
  normalizeSymbol,
  resolveInstrument,
  searchSymbols,
  validateSymbol,
};
