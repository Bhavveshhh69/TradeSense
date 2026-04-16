const express = require('express');

const cache = require('../cache/memoryCache');
const { callLatestPrice, callReasoning, callValidation } = require('../services/reasoning');
const symbolsService = require('../../symbols/symbols.service');
const aiExplainer = require('../../services/ai_explainer.service');
const recentAnalysisService = require('../services/recent_analysis.service');
const validateAnalyzeRequest = require('../middleware/validate');

const router = express.Router();

function elapsedMs(start) {
  return Number(process.hrtime.bigint() - start) / 1e6;
}

function normalizeProbability(probabilityValue) {
  const probability = Number(probabilityValue);
  if (!Number.isFinite(probability)) {
    return null;
  }

  return Math.max(0, Math.min(1, probability));
}

function titleCase(value) {
  if (typeof value !== 'string' || !value.trim()) {
    return null;
  }

  return value
    .toLowerCase()
    .split(/[_\s]+/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

async function resolveAnalyzeInstrument(rawSymbol) {
  return symbolsService.resolveInstrument(rawSymbol);
}

function buildRecentAnalysisEntry(analysis, instrument) {
  return {
    symbol: instrument.symbol,
    normalized: instrument.normalized,
    display_name: instrument.display_name,
    market: instrument.market,
    exchange: instrument.exchange,
    instrument_type: instrument.instrument_type,
    signal: analysis.signal,
    decision_label: analysis.decision_label,
    confidence_level: analysis.confidence_level,
    current_price: analysis.current_price,
    price_error: analysis.price_error,
    price_error_message: analysis.price_error_message,
    trend_summary: analysis.trend_summary,
    risk_summary: analysis.risk_summary,
    signal_explanation: analysis.signal_explanation,
    trade_actionable: analysis.trade_actionable,
    recorded_at: new Date().toISOString(),
  };
}

function mapAnalyzeResponse(instrument, predictionData, latestPrice) {
  const context = predictionData && predictionData.context && typeof predictionData.context === 'object'
    ? predictionData.context
    : {};
  const trendSummary =
    typeof predictionData?.trend_summary === 'string'
      ? predictionData.trend_summary
      : typeof context.trend_summary === 'string'
        ? context.trend_summary
        : null;
  const riskSummary =
    typeof predictionData?.risk_summary === 'string'
      ? predictionData.risk_summary
      : typeof context.risk_summary === 'string'
        ? context.risk_summary
        : null;
  const rawProbability = normalizeProbability(predictionData?.probability);
  const displayProbability = rawProbability === null ? null : Number(rawProbability.toFixed(4));
  const decision = typeof predictionData?.decision === 'string' ? predictionData.decision : 'NO_TRADE';
  const setupSide =
    typeof predictionData?.setup_side === 'string' ? predictionData.setup_side : null;
  const confidenceLevel = titleCase(predictionData?.confidence_level) || 'Low';
  const summary =
    typeof predictionData?.summary === 'string'
      ? predictionData.summary
      : 'No intraday summary was returned by the Python engine.';

  return {
    ...predictionData,
    id: instrument.id,
    symbol: instrument.normalized,
    raw_symbol: instrument.symbol,
    normalized: instrument.normalized,
    display_name: instrument.display_name,
    market: instrument.market,
    exchange: instrument.exchange,
    instrument_type: instrument.instrument_type,
    country: instrument.country,
    probability: displayProbability,
    current_price: latestPrice.current_price,
    price_error: latestPrice.price_error,
    price_error_message: latestPrice.price_error ? latestPrice.price_error_message : null,
    trend_summary: trendSummary,
    risk_summary: riskSummary,
    signal: decision,
    decision_label: titleCase(decision),
    model_confidence_level:
      typeof predictionData?.confidence_level === 'string' ? predictionData.confidence_level : null,
    confidence_level: confidenceLevel,
    setup_side: setupSide,
    trade_actionable: decision === 'LONG' || decision === 'SHORT',
    signal_explanation: summary,
  };
}

router.post('/analyze', validateAnalyzeRequest, async (req, res) => {
  const requestStart = process.hrtime.bigint();
  let instrument;
  try {
    instrument = await resolveAnalyzeInstrument(req.body.symbol);
  } catch (error) {
    const status = error.status || 400;
    return res.status(status).json({
      error: error.message || 'Unable to resolve symbol',
      matches: Array.isArray(error.matches) ? error.matches : undefined,
    });
  }

  const normalizedSymbol = instrument.normalized;
  const cachePayload = {
    ...req.body,
    symbol: normalizedSymbol,
  };
  const cacheKey = cache.hashBody(cachePayload);
  const cached = cache.get(cacheKey);

  if (cached) {
    const totalMs = elapsedMs(requestStart);
    console.log(`timing total_ms=${totalMs.toFixed(1)} python_ms=0.0`);
    return res.status(200).json({
      ...cached,
      explanation: Object.prototype.hasOwnProperty.call(cached, 'explanation')
        ? cached.explanation
        : aiExplainer.FALLBACK_EXPLANATION,
      market_insight: Object.prototype.hasOwnProperty.call(cached, 'market_insight')
        ? cached.market_insight
        : null,
      explanation_is_fallback:
        cached.explanation_is_fallback === true ||
        !Object.prototype.hasOwnProperty.call(cached, 'explanation'),
    });
  }

  const pythonStart = process.hrtime.bigint();
  try {
    const latestPrice = await callLatestPrice(normalizedSymbol);
    const predictionData = await callReasoning(normalizedSymbol);
    const analysis = mapAnalyzeResponse(instrument, predictionData, latestPrice);
    const narratives = await aiExplainer.generateNarratives(analysis);
    const data = {
      ...analysis,
      explanation: narratives.explanation,
      market_insight: narratives.marketInsight,
      explanation_is_fallback: narratives.explanationIsFallback === true,
    };
    try {
      const recentEntry = buildRecentAnalysisEntry(data, instrument);
      await recentAnalysisService.recordAnalysis(recentEntry);
    } catch (persistError) {
      console.warn(
        `[analyze] unable to persist recent analysis for ${normalizedSymbol}: ${
          persistError?.message || 'unknown error'
        }`
      );
    }
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

router.post('/analyze/validate', async (req, res) => {
  try {
    const instrument = await resolveAnalyzeInstrument(req.body?.symbol);
    const validation = await callValidation(instrument.normalized, {
      start_date: req.body?.start_date,
      end_date: req.body?.end_date,
      interval: req.body?.interval,
      horizon: req.body?.horizon,
    });

    return res.status(200).json({
      ...validation,
      id: instrument.id,
      symbol: instrument.normalized,
      raw_symbol: instrument.symbol,
      display_name: instrument.display_name,
      market: instrument.market,
      exchange: instrument.exchange,
      instrument_type: instrument.instrument_type,
      country: instrument.country,
    });
  } catch (error) {
    const status =
      typeof error?.status === 'number' && Number.isInteger(error.status) ? error.status : 500;
    return res.status(status).json(error?.data || {
      error: error?.message || 'Unable to run validation',
      matches: Array.isArray(error?.matches) ? error.matches : undefined,
    });
  }
});

router.get('/analyze/recent', async (req, res) => {
  try {
    const limit = req.query?.limit;
    const results = await recentAnalysisService.listRecentAnalyses(limit);
    return res.status(200).json({ results });
  } catch (error) {
    const status =
      typeof error?.status === 'number' && Number.isInteger(error.status) ? error.status : 500;
    return res.status(status).json({
      error: error?.message || 'Unable to load recent analyses',
    });
  }
});

module.exports = router;
