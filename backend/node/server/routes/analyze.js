const express = require('express');

const cache = require('../cache/memoryCache');
const { callLatestPrice, callReasoning } = require('../services/reasoning');
const symbolsService = require('../../symbols/symbols.service');
const validateAnalyzeRequest = require('../middleware/validate');

const router = express.Router();

function elapsedMs(start) {
  return Number(process.hrtime.bigint() - start) / 1e6;
}

function normalizeDecision(decisionValue) {
  const normalized = typeof decisionValue === 'string' ? decisionValue.trim().toUpperCase() : '';
  if (normalized === 'BUY' || normalized === 'SELL' || normalized === 'HOLD') {
    return normalized;
  }

  return null;
}

function computeSignalLabel(probabilityValue) {
  const probability = Number(probabilityValue);
  if (!Number.isFinite(probability)) {
    return 'NEUTRAL';
  }

  if (probability > 0.65) {
    return 'STRONG_BUY';
  }

  if (probability >= 0.55) {
    return 'BUY';
  }

  if (probability >= 0.45) {
    return 'NEUTRAL';
  }

  if (probability >= 0.35) {
    return 'SELL';
  }

  return 'STRONG_SELL';
}

function computeSignalDirection(signalLabel, probabilityValue, decisionValue) {
  if (signalLabel === 'BUY' || signalLabel === 'STRONG_BUY') {
    return 'BULLISH';
  }

  if (signalLabel === 'SELL' || signalLabel === 'STRONG_SELL') {
    return 'BEARISH';
  }

  const decision = normalizeDecision(decisionValue);
  if (decision === 'BUY') {
    return 'BULLISH';
  }

  if (decision === 'SELL') {
    return 'BEARISH';
  }

  const probability = Number(probabilityValue);
  if (Number.isFinite(probability) && probability < 0.5) {
    return 'BEARISH';
  }

  return 'BULLISH';
}

function computeSignalStrength(signalLabel) {
  if (signalLabel === 'STRONG_BUY' || signalLabel === 'STRONG_SELL') {
    return 'STRONG';
  }

  if (signalLabel === 'BUY' || signalLabel === 'SELL') {
    return 'MODERATE';
  }

  return 'WEAK';
}

function computeMarketCondition(trendSummary) {
  const trendText = typeof trendSummary === 'string' ? trendSummary.toLowerCase() : '';
  if (trendText.includes('uptrend')) {
    return 'BULLISH';
  }

  if (trendText.includes('downtrend')) {
    return 'BEARISH';
  }

  return 'NEUTRAL';
}

function computeRecommendation(decisionValue, signalStrength) {
  const decision = normalizeDecision(decisionValue);

  if (decision === 'HOLD') {
    return 'WAIT';
  }

  if (decision === 'BUY') {
    if (signalStrength === 'STRONG') {
      return 'BUY';
    }

    if (signalStrength === 'MODERATE') {
      return 'BUY_BIAS';
    }

    return 'WATCH';
  }

  if (decision === 'SELL') {
    if (signalStrength === 'STRONG') {
      return 'SELL';
    }

    if (signalStrength === 'MODERATE') {
      return 'SELL_BIAS';
    }

    return 'WATCH';
  }

  return 'WAIT';
}

function computeSignalExplanation(signalLabel, signalStrength, signalDirection) {
  if (signalLabel === 'NEUTRAL') {
    return `Model probability is near neutral; ${signalDirection.toLowerCase()} direction has weak conviction.`;
  }

  return `${signalStrength.toLowerCase()} ${signalDirection.toLowerCase()} signal derived from model probability band ${signalLabel}.`;
}

async function normalizeAnalyzeSymbol(rawSymbol) {
  try {
    return await symbolsService.normalizeSymbol(rawSymbol);
  } catch (error) {
    console.warn(
      `[analyze] symbol normalization failed for ${rawSymbol}: ${error?.message || 'unknown error'}`
    );
    return rawSymbol;
  }
}

function mapAnalyzeResponse(symbol, predictionData, latestPrice) {
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
  const signalLabel = computeSignalLabel(predictionData?.probability);
  const signalDirection = computeSignalDirection(
    signalLabel,
    predictionData?.probability,
    predictionData?.decision
  );
  const signalStrength = computeSignalStrength(signalLabel);
  const marketCondition = computeMarketCondition(trendSummary);
  const recommendation = computeRecommendation(predictionData?.decision, signalStrength);
  const signalExplanation = computeSignalExplanation(
    signalLabel,
    signalStrength,
    signalDirection
  );

  return {
    ...predictionData,
    symbol,
    current_price: latestPrice.current_price,
    price_error: latestPrice.price_error,
    price_error_message: latestPrice.price_error ? latestPrice.price_error_message : null,
    trend_summary: trendSummary,
    risk_summary: riskSummary,
    signal: signalLabel,
    signal_direction: signalDirection,
    signal_strength: signalStrength,
    market_condition: marketCondition,
    recommendation,
    signal_explanation: signalExplanation,
  };
}

router.post('/analyze', validateAnalyzeRequest, async (req, res) => {
  const requestStart = process.hrtime.bigint();
  const normalizedSymbol = await normalizeAnalyzeSymbol(req.body.symbol);
  const cachePayload = {
    ...req.body,
    symbol: normalizedSymbol,
  };
  const cacheKey = cache.hashBody(cachePayload);
  const cached = cache.get(cacheKey);

  if (cached) {
    const totalMs = elapsedMs(requestStart);
    console.log(`timing total_ms=${totalMs.toFixed(1)} python_ms=0.0`);
    return res.status(200).json(cached);
  }

  const pythonStart = process.hrtime.bigint();
  try {
    const latestPrice = await callLatestPrice(normalizedSymbol);
    const predictionData = await callReasoning(normalizedSymbol);
    const data = mapAnalyzeResponse(normalizedSymbol, predictionData, latestPrice);
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
