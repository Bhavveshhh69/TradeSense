function validateAnalyzeRequest(req, res, next) {
  const body = req.body || {};
  const symbol = body.symbol;

  if (typeof symbol !== 'string') {
    return res.status(400).json({ error: 'symbol is required' });
  }

  const normalizedSymbol = symbol.trim().toUpperCase();
  if (normalizedSymbol.length === 0) {
    return res.status(400).json({ error: 'symbol is required' });
  }

  req.body.symbol = normalizedSymbol;
  return next();
}

module.exports = validateAnalyzeRequest;
