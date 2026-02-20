const axios = require('axios');

const REASONING_URL = process.env.REASONING_URL || 'http://localhost:8000/predict';
const REASONING_TIMEOUT_MS = Number(process.env.REASONING_TIMEOUT_MS || 5000);

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

module.exports = {
  callReasoning,
};
