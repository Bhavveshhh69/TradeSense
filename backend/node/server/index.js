const express = require('express');
const axios = require('axios');

const analyzeRoutes = require('./routes/analyze');

const app = express();

app.use(express.json());
app.use('/api', analyzeRoutes);

const PORT = process.env.PORT || 3000;
const REASONING_URL = process.env.REASONING_URL || 'http://localhost:8000/reason';
const REASONING_TIMEOUT_MS = Number(process.env.REASONING_TIMEOUT_MS || 5000);
const PREWARM_ENABLED = ['1', 'true', 'yes'].includes(
  String(process.env.REASONING_PREWARM || '').toLowerCase()
);

const PREWARM_PAYLOAD = {
  symbol: 'WARM',
  probability: 0.5,
  feature_importance: { rsi: 0.1 },
  feature_values: { rsi: 0.1 },
  trend_state: 0,
  momentum_state: 1,
  risk_state: 1,
};

async function prewarmReasoning() {
  if (!PREWARM_ENABLED) {
    return;
  }

  try {
    await axios.post(REASONING_URL, PREWARM_PAYLOAD, { timeout: REASONING_TIMEOUT_MS });
    console.log('prewarm reasoning ok');
  } catch (err) {
    console.log(`prewarm reasoning failed: ${err.message}`);
  }
}

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Express server listening on port ${PORT}`);
    prewarmReasoning();
  });
}

module.exports = app;
