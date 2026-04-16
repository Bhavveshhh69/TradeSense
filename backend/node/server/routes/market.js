const express = require('express');

const symbolsService = require('../../symbols/symbols.service');
const { callLatestPrice, callMarketHistory } = require('../services/reasoning');

const router = express.Router();

function handleError(res, error, fallbackMessage) {
  const status =
    typeof error?.status === 'number' && Number.isInteger(error.status) ? error.status : 500;

  return res.status(status).json(error?.data || { error: error?.message || fallbackMessage });
}

function resolveCurrencyCode(instrument) {
  return instrument?.market === 'IN' ? 'INR' : 'USD';
}

function calculateTrendPct(current, reference) {
  const currentValue = Number(current);
  const referenceValue = Number(reference);
  if (!Number.isFinite(currentValue) || !Number.isFinite(referenceValue) || referenceValue <= 0) {
    return null;
  }

  return ((currentValue - referenceValue) / referenceValue) * 100;
}

router.get('/market/history/:symbol', async (req, res) => {
  try {
    const instrument = await symbolsService.resolveInstrument(req.params.symbol);
    const payload = await callMarketHistory(instrument.normalized, req.query?.days || 30);
    return res.status(200).json({
      ...payload,
      symbol: instrument.normalized,
      display_name: instrument.display_name,
      market: instrument.market,
      exchange: instrument.exchange,
      instrument_type: instrument.instrument_type,
      country: instrument.country,
    });
  } catch (error) {
    return handleError(res, error, 'Unable to load market history');
  }
});

router.get('/market/quote/:symbol', async (req, res) => {
  try {
    const instrument = await symbolsService.resolveInstrument(req.params.symbol);
    const [latestPrice, historyPayload] = await Promise.all([
      callLatestPrice(instrument.normalized),
      callMarketHistory(instrument.normalized, 30),
    ]);

    if (latestPrice.price_error || !Number.isFinite(Number(latestPrice.current_price))) {
      const error = new Error(latestPrice.price_error_message || 'Quote unavailable');
      error.status = 502;
      throw error;
    }

    const history = Array.isArray(historyPayload?.history) ? historyPayload.history : [];
    const closes = history
      .map((point) => ({
        date: point?.date,
        close: Number(point?.close),
      }))
      .filter((point) => Number.isFinite(point.close) && point.close > 0);

    const latestClose = Number(latestPrice.current_price);
    const previousClose = closes.length >= 2 ? closes[closes.length - 2].close : null;
    const trend5Reference = closes.length >= 6 ? closes[closes.length - 6].close : null;
    const trend30Reference = closes.length >= 1 ? closes[0].close : null;

    return res.status(200).json({
      symbol: instrument.normalized,
      display_name: instrument.display_name,
      market: instrument.market,
      exchange: instrument.exchange,
      instrument_type: instrument.instrument_type,
      country: instrument.country,
      current_price: latestClose,
      previous_close: previousClose,
      day_change:
        previousClose !== null ? Number((latestClose - previousClose).toFixed(4)) : null,
      day_change_pct:
        previousClose !== null ? calculateTrendPct(latestClose, previousClose) : null,
      trend_5d_pct: calculateTrendPct(latestClose, trend5Reference),
      trend_30d_pct: calculateTrendPct(latestClose, trend30Reference),
      currency: resolveCurrencyCode(instrument),
      as_of: latestPrice.as_of,
    });
  } catch (error) {
    return handleError(res, error, 'Unable to load quote snapshot');
  }
});

module.exports = router;
