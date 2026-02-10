const axios = require('axios');

const REASONING_URL = process.env.REASONING_URL || 'http://localhost:8000/reason';
const REASONING_TIMEOUT_MS = Number(process.env.REASONING_TIMEOUT_MS || 5000);

async function callReasoning(payload) {
  try {
    const response = await axios.post(REASONING_URL, payload, {
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
