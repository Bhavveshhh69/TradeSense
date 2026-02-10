const express = require('express');

const cache = require('../cache/memoryCache');
const { callReasoning } = require('../services/reasoning');
const validateAnalyzeRequest = require('../middleware/validate');

const router = express.Router();

function elapsedMs(start) {
  return Number(process.hrtime.bigint() - start) / 1e6;
}

router.post('/analyze', validateAnalyzeRequest, async (req, res) => {
  const requestStart = process.hrtime.bigint();
  const cacheKey = cache.hashBody(req.body);
  const cached = cache.get(cacheKey);

  if (cached) {
    const totalMs = elapsedMs(requestStart);
    console.log(`timing total_ms=${totalMs.toFixed(1)} python_ms=0.0`);
    return res.status(200).json(cached);
  }

  const pythonStart = process.hrtime.bigint();
  try {
    const data = await callReasoning(req.body.payload);
    const pythonMs = elapsedMs(pythonStart);
    cache.set(cacheKey, data);
    const totalMs = elapsedMs(requestStart);
    console.log(`timing total_ms=${totalMs.toFixed(1)} python_ms=${pythonMs.toFixed(1)}`);
    return res.status(200).json(data);
  } catch (err) {
    const pythonMs = elapsedMs(pythonStart);
    const totalMs = elapsedMs(requestStart);
    console.log(`timing total_ms=${totalMs.toFixed(1)} python_ms=${pythonMs.toFixed(1)}`);
    const status = err.status || 502;
    const body = err.data || { error: err.message || 'Reasoning service error' };
    return res.status(status).json(body);
  }
});

module.exports = router;
