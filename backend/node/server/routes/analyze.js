const express = require('express');

const cache = require('../cache/memoryCache');
const { callLatestPrice, callReasoning } = require('../services/reasoning');
const symbolsService = require('../../symbols/symbols.service');
const aiExplainer = require('../../services/ai_explainer.service');
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

function computeSignalLabel(probabilityValue) {
  const probability = normalizeProbability(probabilityValue);
  if (probability === null) {
    return 'HOLD';
  }

  if (probability >= 0.7) {
    return 'STRONG_BUY';
  }

  if (probability >= 0.58) {
    return 'BUY';
  }

  if (probability > 0.42) {
    return 'HOLD';
  }

  if (probability > 0.3) {
    return 'SELL';
  }

  return 'STRONG_SELL';
}

function computeConfidenceTier(probabilityValue) {
  const probability = normalizeProbability(probabilityValue);
  if (probability === null) {
    return 'LOW';
  }

  const distanceFromNeutral = Math.abs(probability - 0.5);
  if (distanceFromNeutral >= 0.3) {
    return 'STRONG';
  }

  if (distanceFromNeutral >= 0.2) {
    return 'HIGH';
  }

  if (distanceFromNeutral >= 0.1) {
    return 'MODERATE';
  }

  return 'LOW';
}

function formatSignalLabel(signalLabel) {
  const labels = {
    STRONG_BUY: 'Strong Buy',
    BUY: 'Buy',
    HOLD: 'Hold',
    SELL: 'Sell',
    STRONG_SELL: 'Strong Sell',
  };
  return labels[signalLabel] || 'Hold';
}

function formatConfidenceTier(confidenceTier) {
  const tiers = {
    LOW: 'Low',
    MODERATE: 'Moderate',
    HIGH: 'High',
    STRONG: 'Strong',
  };
  return tiers[confidenceTier] || 'Low';
}

function computeProbabilityBand(signalLabel) {
  const bands = {
    STRONG_BUY: '70%-100%',
    BUY: '58%-69%',
    HOLD: '43%-57%',
    SELL: '31%-42%',
    STRONG_SELL: '0%-30%',
  };
  return bands[signalLabel] || '43%-57%';
}

function computeSignalDirection(signalLabel, probabilityValue) {
  if (signalLabel === 'BUY' || signalLabel === 'STRONG_BUY') {
    return 'BULLISH';
  }

  if (signalLabel === 'SELL' || signalLabel === 'STRONG_SELL') {
    return 'BEARISH';
  }

  const probability = normalizeProbability(probabilityValue);
  if (probability !== null && probability < 0.5) {
    return 'BEARISH';
  }

  return 'BULLISH';
}

function computeSignalStrength(signalLabel, confidenceTier) {
  if (signalLabel === 'HOLD') {
    return 'WEAK';
  }

  if (confidenceTier === 'STRONG') {
    return 'STRONG';
  }

  if (
    (signalLabel === 'STRONG_BUY' || signalLabel === 'STRONG_SELL') &&
    confidenceTier === 'HIGH'
  ) {
    return 'STRONG';
  }

  if (confidenceTier === 'HIGH' || confidenceTier === 'MODERATE') {
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

function computeRecommendation(signalLabel) {
  if (signalLabel === 'STRONG_BUY') {
    return 'BUY';
  }

  if (signalLabel === 'BUY') {
    return 'BUY_BIAS';
  }

  if (signalLabel === 'SELL') {
    return 'SELL_BIAS';
  }

  if (signalLabel === 'STRONG_SELL') {
    return 'SELL';
  }

  return 'WAIT';
}

function computeSignalExplanation(signalLabel, signalStrength, signalDirection, confidenceTier) {
  if (signalLabel === 'HOLD') {
    return `Model probability is near neutral, so the current stance is hold with ${confidenceTier.toLowerCase()} conviction.`;
  }

  return `${signalStrength.toLowerCase()} ${signalDirection.toLowerCase()} signal in the ${formatSignalLabel(signalLabel)} category with ${confidenceTier.toLowerCase()} confidence.`;
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
  const rawProbability = normalizeProbability(predictionData?.probability);
  const signalLabel = computeSignalLabel(rawProbability);
  const confidenceTier = computeConfidenceTier(rawProbability);
  const signalDirection = computeSignalDirection(signalLabel, rawProbability);
  const signalStrength = computeSignalStrength(signalLabel, confidenceTier);
  const marketCondition = computeMarketCondition(trendSummary);
  const recommendation = computeRecommendation(signalLabel);
  const signalExplanation = computeSignalExplanation(
    signalLabel,
    signalStrength,
    signalDirection,
    confidenceTier
  );
  const displayProbability = rawProbability === null ? null : Number(rawProbability.toFixed(2));

  return {
    ...predictionData,
    symbol,
    probability: displayProbability,
    raw_probability: predictionData?.probability,
    current_price: latestPrice.current_price,
    price_error: latestPrice.price_error,
    price_error_message: latestPrice.price_error ? latestPrice.price_error_message : null,
    trend_summary: trendSummary,
    risk_summary: riskSummary,
    signal: signalLabel,
    prediction_category: formatSignalLabel(signalLabel),
    probability_band: computeProbabilityBand(signalLabel),
    confidence_tier: formatConfidenceTier(confidenceTier),
    model_confidence_level:
      typeof predictionData?.confidence_level === 'string' ? predictionData.confidence_level : null,
    confidence_level: formatConfidenceTier(confidenceTier),
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
    const analysis = mapAnalyzeResponse(normalizedSymbol, predictionData, latestPrice);
    const narratives = await aiExplainer.generateNarratives(analysis);
    const data = {
      ...analysis,
      explanation: narratives.explanation,
      market_insight: narratives.marketInsight,
      explanation_is_fallback: narratives.explanationIsFallback === true,
    };
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
