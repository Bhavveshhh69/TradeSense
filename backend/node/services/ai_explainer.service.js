const OpenAI = require('openai');
const marketIntelligence = require('./market_intelligence.service');

const DEFAULT_GROQ_MODEL = 'llama3-70b-8192';
const DEFAULT_GROQ_BASE_URL = 'https://api.groq.com/openai/v1';
const DEFAULT_TIMEOUT_MS = 8000;
const EXPLANATION_CACHE_TTL_MS = Number(process.env.AI_EXPLANATION_CACHE_TTL_MS || 5 * 60 * 1000);
const MIN_REQUEST_INTERVAL_MS = 60000;
const MAX_CACHE_ENTRIES = 500;
const FALLBACK_EXPLANATION =
  'TradeSense AI explanation is temporarily unavailable due to API limits. Based on current model signals, the system suggests a cautious stance with weak momentum and moderate market risk.';
const explanationCache = new Map();
const inFlightRequests = new Map();
const lastSymbolRequestTimestamps = new Map();
const recentSymbolNarratives = new Map();

function normalizeSymbol(value) {
  if (typeof value !== 'string') {
    return 'UNKNOWN';
  }

  const symbol = value.trim().toUpperCase();
  return symbol || 'UNKNOWN';
}

function normalizePrediction(analysisResult) {
  const candidates = [
    analysisResult?.prediction_category,
    analysisResult?.prediction_label,
    analysisResult?.prediction,
    analysisResult?.decision,
    analysisResult?.signal,
    analysisResult?.recommendation,
  ];

  for (const value of candidates) {
    if (typeof value === 'string' && value.trim()) {
      const normalized = value.trim().toUpperCase();
      if (normalized === '0') {
        return 'SELL';
      }

      if (normalized === '1') {
        return 'HOLD';
      }

      if (normalized === '2') {
        return 'BUY';
      }

      return normalized;
    }

    if (typeof value === 'number' && Number.isFinite(value)) {
      if (value === 0) {
        return 'SELL';
      }

      if (value === 1) {
        return 'HOLD';
      }

      if (value === 2) {
        return 'BUY';
      }
    }
  }

  return 'HOLD';
}

function formatPredictionCategory(value) {
  if (typeof value !== 'string' || !value.trim()) {
    return 'Hold';
  }

  const normalized = value
    .trim()
    .toUpperCase()
    .replace(/\s+/g, '_');

  const labels = {
    STRONG_BUY: 'Strong Buy',
    BUY: 'Buy',
    HOLD: 'Hold',
    SELL: 'Sell',
    STRONG_SELL: 'Strong Sell',
  };

  return labels[normalized] || 'Hold';
}

function normalizeProbability(value) {
  const probability = Number(value);
  if (!Number.isFinite(probability)) {
    return 'unknown';
  }

  return probability.toFixed(2);
}

function normalizeTrend(analysisResult) {
  if (typeof analysisResult?.trend === 'string' && analysisResult.trend.trim()) {
    return analysisResult.trend.trim().toLowerCase();
  }

  const marketCondition =
    typeof analysisResult?.market_condition === 'string'
      ? analysisResult.market_condition.trim().toLowerCase()
      : '';
  if (marketCondition === 'bullish' || marketCondition === 'bearish') {
    return marketCondition;
  }

  return 'mixed';
}

function normalizeSentiment(analysisResult) {
  if (typeof analysisResult?.sentiment === 'string' && analysisResult.sentiment.trim()) {
    return analysisResult.sentiment.trim().toLowerCase();
  }

  return 'neutral';
}

function normalizeConfidence(analysisResult) {
  if (typeof analysisResult?.confidence_tier === 'string' && analysisResult.confidence_tier.trim()) {
    return analysisResult.confidence_tier.trim().toLowerCase();
  }

  if (typeof analysisResult?.confidence === 'string' && analysisResult.confidence.trim()) {
    return analysisResult.confidence.trim().toLowerCase();
  }

  if (
    typeof analysisResult?.confidence_level === 'string' &&
    analysisResult.confidence_level.trim()
  ) {
    const raw = analysisResult.confidence_level.trim().toLowerCase();
    if (raw.includes('high')) {
      return 'high';
    }

    if (raw.includes('strong')) {
      return 'strong';
    }

    if (raw.includes('moderate')) {
      return 'moderate';
    }

    if (raw.includes('medium')) {
      return 'medium';
    }

    if (raw.includes('low')) {
      return 'low';
    }
  }

  const numericConfidence = Number(analysisResult?.confidence_score ?? analysisResult?.confidence);
  if (Number.isFinite(numericConfidence)) {
    if (numericConfidence >= 0.75) {
      return 'high';
    }

    if (numericConfidence >= 0.5) {
      return 'medium';
    }
  }

  return 'low';
}

function buildCompanyNewsSection(newsContext) {
  const headlines = Array.isArray(newsContext?.companyHeadlines) ? newsContext.companyHeadlines : [];
  if (!headlines.length) {
    return ['Recent Company News:', 'None available from the last 7 days.'].join('\n');
  }

  return ['Recent Company News:', ...headlines.map((headline) => `- ${headline}`)].join('\n');
}

function buildMacroNewsSection(newsContext) {
  const headlines = Array.isArray(newsContext?.macroHeadlines) ? newsContext.macroHeadlines : [];
  if (!headlines.length) {
    return ['Recent Macro Market Events:', 'None available from the last 7 days.'].join('\n');
  }

  return ['Recent Macro Market Events:', ...headlines.map((headline) => `- ${headline}`)].join('\n');
}

function buildNarrativesPrompt(analysisResult, newsContext) {
  const symbol = normalizeSymbol(analysisResult?.symbol);
  const prediction = formatPredictionCategory(normalizePrediction(analysisResult));
  const probability = normalizeProbability(analysisResult?.probability);
  const trend = normalizeTrend(analysisResult);
  const sentiment = normalizeSentiment(analysisResult);
  const confidence = normalizeConfidence(analysisResult);
  const marketCondition =
    typeof analysisResult?.market_condition === 'string' && analysisResult.market_condition.trim()
      ? analysisResult.market_condition.trim().toLowerCase()
      : 'neutral';

  return [
    'Explain this stock analysis in concise language for beginner investors.',
    '',
    `Stock: ${symbol}`,
    `Prediction Category: ${prediction}`,
    `Confidence Tier: ${confidence}`,
    `Probability: ${probability}`,
    `Trend: ${trend}`,
    `Sentiment: ${sentiment}`,
    `Market condition: ${marketCondition}`,
    '',
    buildCompanyNewsSection(newsContext),
    '',
    buildMacroNewsSection(newsContext),
    '',
    'Explain why the model arrived at this prediction and keep language beginner-friendly.',
    'Only reference headlines when they are clearly relevant to this company or sector.',
    'If headlines are unrelated, ignore them completely.',
    'Never force connections between unrelated companies or sectors.',
    'Avoid repetitive generic statements about uncertainty.',
    'Describe how macro market conditions could influence this stock in the near term.',
    '',
    'Return valid JSON only with this exact shape:',
    '{"explanation":"1-2 concise paragraphs explaining the prediction, key drivers, and risk posture","market_insight":"1-2 concise sentences summarizing relevant external market factors"}',
  ].join('\n');
}

function buildBaseCacheKey(analysisResult) {
  const symbol = normalizeSymbol(analysisResult?.symbol);
  const prediction = normalizePrediction(analysisResult);
  const sentiment = normalizeSentiment(analysisResult);
  const trend = normalizeTrend(analysisResult);
  return `${symbol}_${prediction}_${sentiment}_${trend}`;
}

function extractText(responseData) {
  const content = responseData?.choices?.[0]?.message?.content;
  if (typeof content === 'string' && content.trim()) {
    return content.trim();
  }

  if (!Array.isArray(content)) {
    return null;
  }

  const text = content
    .map((part) => {
      if (typeof part === 'string') {
        return part.trim();
      }

      return typeof part?.text === 'string' ? part.text.trim() : '';
    })
    .filter(Boolean)
    .join('\n')
    .trim();

  return text || null;
}

function stripCodeFences(value) {
  if (typeof value !== 'string') {
    return '';
  }

  return value
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
}

function buildFallbackMarketInsight(analysisResult) {
  const trend = normalizeTrend(analysisResult);
  const prediction = normalizePrediction(analysisResult);

  if (trend === 'bullish' && prediction === 'BUY') {
    return 'Market conditions are constructive, but confirmation and position sizing still matter.';
  }

  if (trend === 'bearish' || prediction === 'SELL') {
    return 'Broader market pressure still looks fragile, so preserving capital may be more important than chasing trades.';
  }

  return 'Market context remains mixed, so waiting for stronger confirmation may help manage risk.';
}

function normalizeNarrativesValue(value, analysisResult) {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const explanation =
    typeof value.explanation === 'string' && value.explanation.trim()
      ? value.explanation.trim()
      : null;

  if (!explanation) {
    return null;
  }

  const marketInsightCandidate = value.marketInsight ?? value.market_insight;
  const marketInsight =
    typeof marketInsightCandidate === 'string' && marketInsightCandidate.trim()
      ? marketInsightCandidate.trim()
      : buildFallbackMarketInsight(analysisResult);

  return {
    explanation,
    marketInsight,
    explanationIsFallback:
      value.explanationIsFallback === true || value.explanation_is_fallback === true,
  };
}

function extractJsonObject(value) {
  if (typeof value !== 'string') {
    return null;
  }

  const firstBrace = value.indexOf('{');
  const lastBrace = value.lastIndexOf('}');
  if (firstBrace === -1 || lastBrace === -1 || lastBrace <= firstBrace) {
    return null;
  }

  return value.slice(firstBrace, lastBrace + 1);
}

function decodeLooseJsonString(value) {
  if (typeof value !== 'string') {
    return '';
  }

  return value
    .replace(/\\"/g, '"')
    .replace(/\\n/g, '\n')
    .replace(/\\r/g, '\r')
    .replace(/\\t/g, '\t')
    .replace(/\\\\/g, '\\')
    .trim();
}

function looksLikeStructuredPayload(value) {
  if (typeof value !== 'string') {
    return false;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return false;
  }

  if (
    (trimmed.startsWith('{') && trimmed.endsWith('}')) ||
    (trimmed.startsWith('[') && trimmed.endsWith(']'))
  ) {
    return true;
  }

  const normalized = trimmed.toLowerCase();
  return (
    normalized.includes('"explanation"') ||
    normalized.includes('"market_insight"') ||
    normalized.includes('"marketinsight"')
  );
}

function extractLabeledNarratives(rawText, analysisResult) {
  if (typeof rawText !== 'string' || !rawText.trim()) {
    return null;
  }

  const explanationRegex =
    /(?:^|\n)\s*explanation\s*:\s*([\s\S]*?)(?=(?:^|\n)\s*(?:market[_\s-]?insight)\s*:|$)/i;
  const marketInsightRegex = /(?:^|\n)\s*market[_\s-]?insight\s*:\s*([\s\S]*?)$/i;
  const explanationMatch = rawText.match(explanationRegex);
  if (!explanationMatch || typeof explanationMatch[1] !== 'string') {
    return null;
  }

  const explanation = explanationMatch[1].trim();
  const marketInsightMatch = rawText.match(marketInsightRegex);
  const marketInsight =
    marketInsightMatch && typeof marketInsightMatch[1] === 'string'
      ? marketInsightMatch[1].trim()
      : buildFallbackMarketInsight(analysisResult);

  return normalizeNarrativesValue(
    {
      explanation,
      marketInsight,
    },
    analysisResult
  );
}

function extractLooseNarrativesValue(rawText, analysisResult) {
  if (typeof rawText !== 'string' || !rawText.trim()) {
    return null;
  }

  const explanationMatch = rawText.match(
    /"explanation"\s*:\s*"([\s\S]*?)"\s*,\s*"(?:marketInsight|market_insight)"/
  );
  const marketInsightMatch = rawText.match(
    /"(?:marketInsight|market_insight)"\s*:\s*"([\s\S]*?)"\s*}/
  );

  if (!explanationMatch) {
    return null;
  }

  return normalizeNarrativesValue(
    {
      explanation: decodeLooseJsonString(explanationMatch[1]),
      marketInsight: marketInsightMatch
        ? decodeLooseJsonString(marketInsightMatch[1])
        : buildFallbackMarketInsight(analysisResult),
    },
    analysisResult
  );
}

function parseNarrativesValue(rawText, analysisResult) {
  const cleanedText = stripCodeFences(rawText);
  if (!cleanedText) {
    return null;
  }

  const parseTargets = [cleanedText, extractJsonObject(cleanedText)].filter(Boolean);
  for (const candidate of parseTargets) {
    try {
      const parsed = JSON.parse(candidate);
      const narratives = normalizeNarrativesValue(parsed, analysisResult);
      if (narratives) {
        return narratives;
      }
    } catch (error) {
      continue;
    }
  }

  const looselyParsedNarratives = extractLooseNarrativesValue(cleanedText, analysisResult);
  if (looselyParsedNarratives) {
    return looselyParsedNarratives;
  }

  const labeledNarratives = extractLabeledNarratives(cleanedText, analysisResult);
  if (labeledNarratives) {
    return labeledNarratives;
  }

  if (looksLikeStructuredPayload(cleanedText)) {
    return null;
  }

  return normalizeNarrativesValue(
    {
      explanation: cleanedText,
      marketInsight: buildFallbackMarketInsight(analysisResult),
    },
    analysisResult
  );
}

function readCacheEntry(cacheKey) {
  const cachedEntry = explanationCache.get(cacheKey);
  if (!cachedEntry || typeof cachedEntry !== 'object') {
    return null;
  }

  const normalizedValue = normalizeNarrativesValue(cachedEntry.value, null);
  if (!normalizedValue) {
    explanationCache.delete(cacheKey);
    return null;
  }

  const timestamp = Number(cachedEntry.timestamp);
  if (!Number.isFinite(timestamp)) {
    explanationCache.delete(cacheKey);
    return null;
  }

  return {
    value: normalizedValue,
    timestamp,
    expired: Date.now() - timestamp >= EXPLANATION_CACHE_TTL_MS,
  };
}

function writeCacheEntry(cacheKey, value) {
  const normalizedValue = normalizeNarrativesValue(value, null);
  if (!normalizedValue) {
    return;
  }

  if (explanationCache.has(cacheKey)) {
    explanationCache.delete(cacheKey);
  }

  explanationCache.set(cacheKey, {
    value: normalizedValue,
    timestamp: Date.now(),
  });

  trimMapSize(explanationCache);
}

function trimMapSize(map) {
  while (map.size > MAX_CACHE_ENTRIES) {
    const oldestKey = map.keys().next().value;
    if (!oldestKey) {
      break;
    }
    map.delete(oldestKey);
  }
}

function readRecentSymbolNarratives(symbol) {
  const entry = recentSymbolNarratives.get(symbol);
  if (!entry || typeof entry !== 'object') {
    return null;
  }

  const normalizedValue = normalizeNarrativesValue(entry.value, null);
  if (!normalizedValue) {
    recentSymbolNarratives.delete(symbol);
    return null;
  }

  const timestamp = Number(entry.timestamp);
  if (!Number.isFinite(timestamp)) {
    recentSymbolNarratives.delete(symbol);
    return null;
  }

  return {
    value: normalizedValue,
    timestamp,
  };
}

function writeRecentSymbolNarratives(symbol, value) {
  const normalizedValue = normalizeNarrativesValue(value, null);
  if (!normalizedValue) {
    return;
  }

  if (recentSymbolNarratives.has(symbol)) {
    recentSymbolNarratives.delete(symbol);
  }

  recentSymbolNarratives.set(symbol, {
    value: normalizedValue,
    timestamp: Date.now(),
  });

  trimMapSize(recentSymbolNarratives);
}

function createFallbackNarratives(analysisResult) {
  return {
    explanation: FALLBACK_EXPLANATION,
    marketInsight: buildFallbackMarketInsight(analysisResult),
    explanationIsFallback: true,
  };
}

function getGroqConfig() {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return null;
  }

  const model = process.env.GROQ_MODEL || DEFAULT_GROQ_MODEL;
  const baseUrl = (process.env.GROQ_API_BASE_URL || DEFAULT_GROQ_BASE_URL).replace(/\/$/, '');
  const timeoutMs = Number(process.env.GROQ_TIMEOUT_MS || DEFAULT_TIMEOUT_MS);

  return {
    client: new OpenAI({
      apiKey,
      baseURL: baseUrl,
      maxRetries: 0,
      timeout: Number.isFinite(timeoutMs) ? timeoutMs : DEFAULT_TIMEOUT_MS,
    }),
    model,
    timeoutMs: Number.isFinite(timeoutMs) ? timeoutMs : DEFAULT_TIMEOUT_MS,
  };
}

async function generateNarratives(analysisResult) {
  const cacheKey = buildBaseCacheKey(analysisResult);
  const symbol = normalizeSymbol(analysisResult?.symbol);
  const cachedEntry = readCacheEntry(cacheKey);
  if (cachedEntry && !cachedEntry.expired) {
    console.log('[ai-explainer] cache hit', cacheKey);
    writeRecentSymbolNarratives(symbol, cachedEntry.value);
    return cachedEntry.value;
  }

  const activeRequest = inFlightRequests.get(cacheKey);
  if (activeRequest) {
    console.log('[ai-explainer] cache hit', cacheKey);
    return activeRequest;
  }

  console.log('[ai-explainer] cache miss', cacheKey);

  const lastRequestTimestamp = Number(lastSymbolRequestTimestamps.get(symbol));
  const recentNarratives = readRecentSymbolNarratives(symbol);
  if (
    Number.isFinite(lastRequestTimestamp) &&
    Date.now() - lastRequestTimestamp < MIN_REQUEST_INTERVAL_MS &&
    recentNarratives
  ) {
    console.log('[ai-explainer] cooldown active', symbol);
    return recentNarratives.value;
  }

  const groqConfig = getGroqConfig();
  if (!groqConfig) {
    const fallbackNarratives = createFallbackNarratives(analysisResult);
    console.log('[ai-explainer] AI fallback explanation used', cacheKey);
    writeCacheEntry(cacheKey, fallbackNarratives);
    writeRecentSymbolNarratives(symbol, fallbackNarratives);
    return fallbackNarratives;
  }

  lastSymbolRequestTimestamps.set(symbol, Date.now());
  trimMapSize(lastSymbolRequestTimestamps);

  const requestPromise = (async () => {
    try {
      const newsContext = await marketIntelligence.fetchRecentNewsContext(symbol);
      const prompt = buildNarrativesPrompt(analysisResult, newsContext);
      console.log('[ai-explainer] groq request', cacheKey);
      const response = await groqConfig.client.chat.completions.create(
        {
          model: groqConfig.model,
          messages: [
            {
              role: 'system',
              content:
                'You are a financial analysis assistant explaining stock predictions in simple language. Return valid JSON only with keys explanation and market_insight.',
            },
            {
              role: 'user',
              content: prompt,
            },
          ],
          temperature: 0.4,
        },
        {
          timeout: groqConfig.timeoutMs,
        }
      );

      const narratives = parseNarrativesValue(extractText(response), analysisResult);
      if (narratives) {
        console.log('[ai-explainer] AI explanation generated', cacheKey);
        writeCacheEntry(cacheKey, narratives);
        writeRecentSymbolNarratives(symbol, narratives);
        return narratives;
      }

      const fallbackNarratives = createFallbackNarratives(analysisResult);
      console.log('[ai-explainer] AI fallback explanation used', cacheKey);
      writeCacheEntry(cacheKey, fallbackNarratives);
      writeRecentSymbolNarratives(symbol, fallbackNarratives);
      return fallbackNarratives;
    } catch (error) {
      console.warn(
        `[ai-explainer] explanation generation failed: ${error?.message || 'unknown error'}`
      );
      const fallbackNarratives = createFallbackNarratives(analysisResult);
      console.log('[ai-explainer] AI fallback explanation used', cacheKey);
      writeCacheEntry(cacheKey, fallbackNarratives);
      writeRecentSymbolNarratives(symbol, fallbackNarratives);
      return fallbackNarratives;
    } finally {
      inFlightRequests.delete(cacheKey);
    }
  })();

  inFlightRequests.set(cacheKey, requestPromise);
  return requestPromise;
}

async function generateExplanation(analysisResult) {
  const narratives = await generateNarratives(analysisResult);
  return narratives.explanation;
}

async function generateMarketInsight(analysisResult) {
  const narratives = await generateNarratives(analysisResult);
  return narratives.marketInsight;
}

function __clearState() {
  explanationCache.clear();
  inFlightRequests.clear();
  lastSymbolRequestTimestamps.clear();
  recentSymbolNarratives.clear();
}

module.exports = {
  FALLBACK_EXPLANATION,
  MIN_REQUEST_INTERVAL_MS,
  __clearState,
  generateNarratives,
  generateExplanation,
  generateMarketInsight,
};
