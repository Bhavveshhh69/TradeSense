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
    const market = typeof req.query?.market === 'string' ? req.query.market : undefined;
    const kind = typeof req.query?.kind === 'string' ? req.query.kind : undefined;
    const limit = req.query?.limit;
    const results = await symbolsService.searchSymbols({ query, market, kind, limit });
    return res.status(200).json({
      results,
      query,
      market: market ? market.toUpperCase() : null,
      kind: kind || null,
      limit: results.length,
    });
  } catch (error) {
    return handleError(res, error);
  }
}

async function normalizeSymbol(req, res) {
  try {
    const rawSymbol = req.params.symbol;
    let resolved;
    if (typeof symbolsService.resolveInstrument === 'function') {
      resolved = await symbolsService.resolveInstrument(rawSymbol);
    } else {
      const normalized = await symbolsService.normalizeSymbol(rawSymbol);
      resolved = {
        normalized,
        changed: normalized !== rawSymbol,
        symbol: normalized,
        display_name: normalized,
        market: null,
        exchange: null,
        instrument_type: null,
      };
    }
    return res.status(200).json({
      id: resolved.id,
      input: rawSymbol,
      normalized: resolved.normalized,
      changed: resolved.changed,
      symbol: resolved.symbol,
      display_name: resolved.display_name,
      market: resolved.market,
      exchange: resolved.exchange,
      instrument_type: resolved.instrument_type,
      country: resolved.country,
      search_terms: resolved.search_terms,
    });
  } catch (error) {
    return handleError(res, error);
  }
}

module.exports = {
  normalizeSymbol,
  searchSymbols,
};
