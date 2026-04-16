const axios = require('axios');

const REASONING_URL = process.env.REASONING_URL || 'http://localhost:8000/predict';
const REASONING_TIMEOUT_MS = Number(process.env.REASONING_TIMEOUT_MS || 5000);
const PYTHON_API_BASE_URL =
  (process.env.PYTHON_API_BASE_URL || REASONING_URL.replace(/\/predict\/?$/, '')).replace(
    /\/$/,
    ''
  );

function normalizeProviderBaseUrl() {
  if (!PYTHON_API_BASE_URL) {
    return 'http://localhost:8000';
  }

  return PYTHON_API_BASE_URL;
}

function extractPriceErrorMessage(error, symbol) {
  if (error?.response?.data?.detail?.error) {
    return `${error.response.data.detail.error} (${symbol})`;
  }

  if (typeof error?.response?.data?.detail === 'string') {
    return `${error.response.data.detail} (${symbol})`;
  }

  if (error?.code === 'ECONNABORTED') {
    return `Latest price request timed out (${symbol})`;
  }

  if (typeof error?.message === 'string' && error.message.trim()) {
    return `${error.message} (${symbol})`;
  }

  return `Unable to fetch current price (${symbol})`;
}

function extractProviderErrorPayload(error, fallbackStatus, fallbackMessage) {
  if (error?.response) {
    const providerError = new Error(fallbackMessage);
    providerError.status = error.response.status;
    providerError.data = error.response.data;
    throw providerError;
  }

  if (error?.code === 'ECONNABORTED') {
    const timeoutError = new Error(`${fallbackMessage} timeout`);
    timeoutError.status = 504;
    timeoutError.data = { error: `${fallbackMessage} timeout` };
    throw timeoutError;
  }

  const providerError = new Error(fallbackMessage);
  providerError.status = fallbackStatus;
  providerError.data = { error: fallbackMessage };
  throw providerError;
}

async function callReasoning(symbol) {
  const normalizedSymbol =
    typeof symbol === 'string' ? symbol.trim().toUpperCase() : '';
  if (!normalizedSymbol) {
    const error = new Error('Invalid analyze request');
    error.status = 400;
    error.data = { error: 'symbol is required' };
    throw error;
  }

  const predictRequest = { symbol: normalizedSymbol };

  try {
    const response = await axios.post(REASONING_URL, predictRequest, {
      timeout: REASONING_TIMEOUT_MS,
    });
    return response.data;
  } catch (err) {
    if (err && err.response) {
      const error = new Error('Reasoning service error');
      error.status = err.response.status;
      error.data = err.response.data;
      throw error;
    }

    if (err && err.code === 'ECONNABORTED') {
      const error = new Error('Reasoning service timeout');
      error.status = 504;
      error.data = { error: 'Reasoning service timeout' };
      throw error;
    }

    const error = new Error('Reasoning service unavailable');
    error.status = 502;
    error.data = { error: 'Reasoning service unavailable' };
    throw error;
  }
}

async function callLatestPrice(symbol) {
  const normalizedSymbol =
    typeof symbol === 'string' ? symbol.trim().toUpperCase() : '';
  if (!normalizedSymbol) {
    return {
      symbol: null,
      current_price: null,
      price_error: true,
      price_error_message: 'symbol is required',
    };
  }

  const endpoint = `${normalizeProviderBaseUrl()}/market/latest-price/${encodeURIComponent(
    normalizedSymbol
  )}`;

  try {
    const response = await axios.get(endpoint, {
      timeout: REASONING_TIMEOUT_MS,
    });

    const responseSymbolRaw =
      typeof response?.data?.symbol === 'string' ? response.data.symbol : normalizedSymbol;
    const responseSymbol = responseSymbolRaw.trim().toUpperCase();
    const latestPrice = Number(response?.data?.price);

    if (responseSymbol !== normalizedSymbol) {
      return {
        symbol: normalizedSymbol,
        current_price: null,
        price_error: true,
        price_error_message: `Unexpected symbol returned from latest price endpoint (${normalizedSymbol})`,
      };
    }

    if (!Number.isFinite(latestPrice) || latestPrice <= 0) {
      return {
        symbol: normalizedSymbol,
        current_price: null,
        price_error: true,
        price_error_message: `Invalid price response (${normalizedSymbol})`,
      };
    }

    return {
      symbol: normalizedSymbol,
      current_price: latestPrice,
      price_error: false,
      price_error_message: null,
      market: response?.data?.market || null,
      timeframe: response?.data?.timeframe || null,
      as_of: response?.data?.timestamp || null,
    };
  } catch (error) {
    return {
      symbol: normalizedSymbol,
      current_price: null,
      price_error: true,
      price_error_message: extractPriceErrorMessage(error, normalizedSymbol),
      market: null,
      timeframe: null,
      as_of: null,
    };
  }
}

async function callMarketHistory(symbol, days = 30) {
  const normalizedSymbol =
    typeof symbol === 'string' ? symbol.trim().toUpperCase() : '';
  if (!normalizedSymbol) {
    const error = new Error('Invalid history request');
    error.status = 400;
    error.data = { error: 'symbol is required' };
    throw error;
  }

  const endpoint = `${normalizeProviderBaseUrl()}/market/history/${encodeURIComponent(
    normalizedSymbol
  )}`;

  try {
    const response = await axios.get(endpoint, {
      timeout: REASONING_TIMEOUT_MS,
      params: { days },
    });
    return response.data;
  } catch (error) {
    extractProviderErrorPayload(error, 502, 'Market history unavailable');
  }
}

async function callValidation(symbol, payload = {}) {
  const normalizedSymbol =
    typeof symbol === 'string' ? symbol.trim().toUpperCase() : '';
  if (!normalizedSymbol) {
    const error = new Error('Invalid validation request');
    error.status = 400;
    error.data = { error: 'symbol is required' };
    throw error;
  }

  const endpoint = `${normalizeProviderBaseUrl()}/analyze/validate`;
  try {
    const response = await axios.post(
      endpoint,
      {
        symbol: normalizedSymbol,
        ...payload,
      },
      { timeout: Math.max(REASONING_TIMEOUT_MS, 30000) }
    );
    return response.data;
  } catch (error) {
    extractProviderErrorPayload(error, 502, 'Validation service unavailable');
  }
}

module.exports = {
  callMarketHistory,
  callLatestPrice,
  callReasoning,
  callValidation,
};
