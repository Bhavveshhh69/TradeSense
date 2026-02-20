const express = require('express');
const axios = require('axios');

const analyzeRoutes = require('./routes/analyze');

const app = express();

app.use(express.json());
app.use((err, req, res, next) => {
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({ error: 'Invalid JSON body' });
  }
  return next(err);
});
app.use('/api', analyzeRoutes);

const PORT = process.env.PORT || 3000;
const REASONING_URL = process.env.REASONING_URL || 'http://localhost:8000/predict';
const REASONING_TIMEOUT_MS = Number(process.env.REASONING_TIMEOUT_MS || 5000);
const PREWARM_ENABLED = ['1', 'true', 'yes'].includes(
  String(process.env.REASONING_PREWARM || '').toLowerCase()
);

const PREWARM_PAYLOAD = {
  symbol: 'WARM',
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
