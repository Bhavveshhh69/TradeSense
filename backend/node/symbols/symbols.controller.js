const symbolsService = require('./symbols.service');

function handleError(res, error) {
  const status =
    typeof error?.status === 'number' && Number.isInteger(error.status)
      ? error.status
      : 500;

  return res.status(status).json({
    error: error?.message || 'Internal server error',
  });
}

async function searchSymbols(req, res) {
  try {
    const query = typeof req.query?.q === 'string' ? req.query.q : '';
    const results = await symbolsService.searchSymbols(query);
    return res.status(200).json({ results });
  } catch (error) {
    return handleError(res, error);
  }
}

async function normalizeSymbol(req, res) {
  try {
    const rawSymbol = req.params.symbol;
    const normalized = await symbolsService.normalizeSymbol(rawSymbol);
    return res.status(200).json({
      input: rawSymbol,
      normalized,
    });
  } catch (error) {
    return handleError(res, error);
  }
}

module.exports = {
  normalizeSymbol,
  searchSymbols,
};
