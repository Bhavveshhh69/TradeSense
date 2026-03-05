const axios = require('axios');

const FX_TIMEOUT_MS = Number(process.env.FX_TIMEOUT_MS || 5000);
const FX_CACHE_TTL_MS = Number(process.env.FX_CACHE_TTL_MS || 5 * 60 * 1000);

const PAIR_TO_YAHOO_SYMBOL = {
  USD_INR: 'USDINR=X',
  INR_USD: 'INRUSD=X',
};

const rateCache = new Map();

function normalizeCurrency(currency) {
  if (typeof currency !== 'string' || !currency.trim()) {
    throw new Error('currency is required');
  }

  return currency.trim().toUpperCase();
}

function getCacheKey(fromCurrency, toCurrency) {
  return `${fromCurrency}_${toCurrency}`;
}

function getRateFromChartResponse(data) {
  const result = data?.chart?.result?.[0];
  const candidates = [
    result?.meta?.regularMarketPrice,
    result?.meta?.previousClose,
    result?.indicators?.quote?.[0]?.close?.at(-1),
  ];

  for (const value of candidates) {
    const numericValue = Number(value);
    if (Number.isFinite(numericValue) && numericValue > 0) {
      return numericValue;
    }
  }

  return null;
}

async function fetchYahooFxRate(pairSymbol, fromCurrency, toCurrency) {
  const endpoint = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(
    pairSymbol
  )}?interval=1d&range=1d`;
  const response = await axios.get(endpoint, {
    timeout: FX_TIMEOUT_MS,
  });

  const rate = getRateFromChartResponse(response?.data);
  if (!Number.isFinite(rate) || rate <= 0) {
    throw new Error(`Invalid FX response for ${fromCurrency}->${toCurrency}`);
  }

  return rate;
}

async function getFxRate(fromCurrencyInput, toCurrencyInput) {
  const fromCurrency = normalizeCurrency(fromCurrencyInput);
  const toCurrency = normalizeCurrency(toCurrencyInput);

  if (fromCurrency === toCurrency) {
    return 1;
  }

  const cacheKey = getCacheKey(fromCurrency, toCurrency);
  const cached = rateCache.get(cacheKey);
  if (cached && cached.expires_at > Date.now()) {
    return cached.rate;
  }

  const pairSymbol = PAIR_TO_YAHOO_SYMBOL[cacheKey];
  if (!pairSymbol) {
    throw new Error(`Unsupported FX pair ${fromCurrency}->${toCurrency}`);
  }

  const rate = await fetchYahooFxRate(pairSymbol, fromCurrency, toCurrency);
  rateCache.set(cacheKey, {
    rate,
    expires_at: Date.now() + FX_CACHE_TTL_MS,
  });

  return rate;
}

module.exports = {
  getFxRate,
};
