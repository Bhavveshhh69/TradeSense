function validateAnalyzeRequest(req, res, next) {
  const body = req.body || {};
  const symbol = body.symbol;
  const payload = body.payload;

  if (typeof symbol !== 'string' || symbol.trim().length === 0) {
    return res.status(400).json({ error: 'symbol is required' });
  }

  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    return res.status(400).json({ error: 'payload must be an object' });
  }

  return next();
}

module.exports = validateAnalyzeRequest;
