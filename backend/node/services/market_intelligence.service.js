const axios = require('axios');

const DEFAULT_FINNHUB_BASE_URL = 'https://finnhub.io/api/v1/company-news';
const DEFAULT_ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query';
const DEFAULT_TIMEOUT_MS = 5000;
const NEWS_CACHE_TTL_MS = Number(process.env.MARKET_INTELLIGENCE_CACHE_TTL_MS || 10 * 60 * 1000);
const MACRO_CACHE_TTL_MS = Number(
  process.env.MARKET_INTELLIGENCE_MACRO_CACHE_TTL_MS || NEWS_CACHE_TTL_MS
);
const ALPHA_VANTAGE_MIN_INTERVAL_MS = Number(process.env.ALPHA_VANTAGE_MIN_INTERVAL_MS || 15000);
const MAX_CACHE_ENTRIES = 500;
const MAX_COMPANY_HEADLINES = Number(process.env.MAX_COMPANY_NEWS_HEADLINES || 3);
const MAX_MACRO_HEADLINES = Number(process.env.MAX_MACRO_NEWS_HEADLINES || 3);
const MAX_FILTER_LOGS = 12;

const UNIVERSAL_MACRO_KEYWORDS = [
  'global market',
  'inflation',
  'interest rate',
  'federal reserve',
  'central bank',
  'economic outlook',
  'recession',
  'gdp',
  'unemployment',
  'currency',
  'bond yield',
];

const SECTOR_CONFIGS = {
  banks: {
    keywords: ['interest rates', 'inflation', 'rbi policy', 'rbi', 'credit growth', 'npa'],
    alphaTopics: ['financial_markets', 'economy_macro'],
  },
  technology: {
    keywords: ['ai', 'artificial intelligence', 'semiconductor', 'chip', 'tariff', 'export control'],
    alphaTopics: ['technology', 'financial_markets'],
  },
  energy: {
    keywords: ['oil prices', 'opec', 'geopolitical tension', 'crude', 'energy demand', 'gas prices'],
    alphaTopics: ['energy_transportation', 'economy_macro'],
  },
  consumer: {
    keywords: ['retail demand', 'consumer spending', 'discretionary spending', 'household demand', 'fmcg'],
    alphaTopics: ['retail_wholesale', 'economy_macro'],
  },
  general: {
    keywords: [],
    alphaTopics: ['financial_markets', 'economy_macro'],
  },
};

const SYMBOL_SECTOR_OVERRIDES = {
  AAPL: 'technology',
  NVDA: 'technology',
  MSFT: 'technology',
  GOOGL: 'technology',
  META: 'technology',
  AMD: 'technology',
  INTC: 'technology',
  TSM: 'technology',
  RELIANCE: 'energy',
  'RELIANCE.NS': 'energy',
  ONGC: 'energy',
  'ONGC.NS': 'energy',
  HDFCBANK: 'banks',
  'HDFCBANK.NS': 'banks',
  ICICIBANK: 'banks',
  'ICICIBANK.NS': 'banks',
  SBIN: 'banks',
  'SBIN.NS': 'banks',
  KOTAKBANK: 'banks',
  'KOTAKBANK.NS': 'banks',
  AXISBANK: 'banks',
  'AXISBANK.NS': 'banks',
  HINDUNILVR: 'consumer',
  'HINDUNILVR.NS': 'consumer',
  ITC: 'consumer',
  'ITC.NS': 'consumer',
  WMT: 'consumer',
  COST: 'consumer',
  PG: 'consumer',
};

const symbolNewsCache = new Map();
const inFlightRequests = new Map();
const macroNewsCacheBySector = new Map();
const macroInFlightRequestsBySector = new Map();
let lastAlphaVantageRequestTimestamp = 0;
let missingFinnhubKeyWarningLogged = false;
let missingAlphaVantageKeyWarningLogged = false;

function normalizeSymbol(value) {
  if (typeof value !== 'string') {
    return 'UNKNOWN';
  }

  const symbol = value.trim().toUpperCase();
  return symbol || 'UNKNOWN';
}

function normalizePositiveNumber(value, fallbackValue) {
  const numberValue = Number(value);
  if (Number.isFinite(numberValue) && numberValue > 0) {
    return numberValue;
  }

  return fallbackValue;
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

function formatFinnhubDate(date) {
  return date.toISOString().slice(0, 10);
}

function buildDateRange(now = new Date()) {
  const endDate = new Date(now);
  const startDate = new Date(now);
  startDate.setUTCDate(startDate.getUTCDate() - 7);

  return {
    from: formatFinnhubDate(startDate),
    to: formatFinnhubDate(endDate),
  };
}

function createEmptyNewsContext() {
  return {
    companyHeadlines: [],
    macroHeadlines: [],
  };
}

function normalizeHeadlines(headlines, maxHeadlines) {
  if (!Array.isArray(headlines)) {
    return [];
  }

  const limit = normalizePositiveNumber(maxHeadlines, 3);
  const seen = new Set();
  const normalizedHeadlines = [];
  for (const candidate of headlines) {
    if (typeof candidate !== 'string') {
      continue;
    }

    const headline = candidate.trim();
    if (!headline) {
      continue;
    }

    const dedupeKey = headline.toLowerCase();
    if (seen.has(dedupeKey)) {
      continue;
    }

    seen.add(dedupeKey);
    normalizedHeadlines.push(headline);
    if (normalizedHeadlines.length >= limit) {
      break;
    }
  }

  return normalizedHeadlines;
}

function normalizeNewsContext(value) {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const companyHeadlines = normalizeHeadlines(value.companyHeadlines, MAX_COMPANY_HEADLINES);
  const macroHeadlines = normalizeHeadlines(value.macroHeadlines, MAX_MACRO_HEADLINES);
  return { companyHeadlines, macroHeadlines };
}

function isCacheEntryExpired(timestamp, ttlMs) {
  const numericTimestamp = Number(timestamp);
  if (!Number.isFinite(numericTimestamp)) {
    return true;
  }

  return Date.now() - numericTimestamp >= normalizePositiveNumber(ttlMs, 1);
}

function readSymbolCacheEntry(cacheKey) {
  const cachedEntry = symbolNewsCache.get(cacheKey);
  if (!cachedEntry || typeof cachedEntry !== 'object') {
    return null;
  }

  const value = normalizeNewsContext(cachedEntry.value);
  if (!value || isCacheEntryExpired(cachedEntry.timestamp, NEWS_CACHE_TTL_MS)) {
    symbolNewsCache.delete(cacheKey);
    return null;
  }

  return value;
}

function writeSymbolCacheEntry(cacheKey, value) {
  const normalizedValue = normalizeNewsContext(value);
  if (!normalizedValue) {
    return;
  }

  if (symbolNewsCache.has(cacheKey)) {
    symbolNewsCache.delete(cacheKey);
  }

  symbolNewsCache.set(cacheKey, {
    value: normalizedValue,
    timestamp: Date.now(),
  });
  trimMapSize(symbolNewsCache);
}

function readMacroCacheEntry(sector) {
  const cachedEntry = macroNewsCacheBySector.get(sector);
  if (!cachedEntry || typeof cachedEntry !== 'object') {
    return null;
  }

  const macroHeadlines = normalizeHeadlines(cachedEntry.value, MAX_MACRO_HEADLINES);
  if (!macroHeadlines.length || isCacheEntryExpired(cachedEntry.timestamp, MACRO_CACHE_TTL_MS)) {
    macroNewsCacheBySector.delete(sector);
    return null;
  }

  return macroHeadlines;
}

function writeMacroCacheEntry(sector, macroHeadlines) {
  const normalizedHeadlines = normalizeHeadlines(macroHeadlines, MAX_MACRO_HEADLINES);
  if (!normalizedHeadlines.length) {
    return;
  }

  macroNewsCacheBySector.set(sector, {
    value: normalizedHeadlines,
    timestamp: Date.now(),
  });
  trimMapSize(macroNewsCacheBySector);
}

function sleep(ms) {
  if (!Number.isFinite(ms) || ms <= 0) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function extractFinnhubHeadlines(payload) {
  if (!Array.isArray(payload)) {
    return [];
  }

  const sortedNews = [...payload].sort(
    (left, right) => Number(right?.datetime || 0) - Number(left?.datetime || 0)
  );
  return normalizeHeadlines(
    sortedNews.map((item) => item?.headline),
    MAX_COMPANY_HEADLINES
  );
}

function normalizeSymbolRoot(symbol) {
  return symbol.replace(/\.(NS|BO)$/i, '');
}

function deriveSectorFromSymbol(symbol) {
  const normalized = normalizeSymbol(symbol);
  const symbolRoot = normalizeSymbolRoot(normalized);

  if (SYMBOL_SECTOR_OVERRIDES[normalized]) {
    return SYMBOL_SECTOR_OVERRIDES[normalized];
  }

  if (SYMBOL_SECTOR_OVERRIDES[symbolRoot]) {
    return SYMBOL_SECTOR_OVERRIDES[symbolRoot];
  }

  if (symbolRoot.includes('BANK') || symbolRoot.startsWith('HDFC') || symbolRoot.startsWith('ICICI')) {
    return 'banks';
  }

  if (
    symbolRoot.includes('TECH') ||
    symbolRoot.includes('SOFT') ||
    symbolRoot.includes('SEMI') ||
    symbolRoot.includes('CHIP')
  ) {
    return 'technology';
  }

  if (symbolRoot.includes('OIL') || symbolRoot.includes('ENERGY') || symbolRoot.includes('GAS')) {
    return 'energy';
  }

  if (symbolRoot.includes('RETAIL') || symbolRoot.includes('CONSUMER') || symbolRoot.includes('FMCG')) {
    return 'consumer';
  }

  return 'general';
}

function getSectorConfig(sector) {
  return SECTOR_CONFIGS[sector] || SECTOR_CONFIGS.general;
}

function extractAlphaTopicNames(item) {
  const topics = Array.isArray(item?.topics) ? item.topics : [];
  return topics
    .map((topic) => {
      if (typeof topic === 'string') {
        return topic.trim().toLowerCase();
      }
      return typeof topic?.topic === 'string' ? topic.topic.trim().toLowerCase() : '';
    })
    .filter(Boolean);
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function keywordMatches(text, keyword) {
  if (typeof text !== 'string' || typeof keyword !== 'string') {
    return false;
  }

  const normalizedKeyword = keyword.trim().toLowerCase();
  if (!normalizedKeyword) {
    return false;
  }

  const escapedKeyword = escapeRegex(normalizedKeyword).replace(/\s+/g, '\\s+');
  const pattern = new RegExp(`\\b${escapedKeyword}\\b`, 'i');
  return pattern.test(text);
}

function scoreMacroHeadline(item, sectorConfig) {
  const title = typeof item?.title === 'string' ? item.title.trim() : '';
  const summary = typeof item?.summary === 'string' ? item.summary.trim() : '';
  const combined = `${title} ${summary}`.toLowerCase();
  if (!combined) {
    return {
      score: 0,
      sectorKeywordHits: 0,
      universalKeywordHits: 0,
      topicHits: 0,
    };
  }

  const sectorKeywordHits = sectorConfig.keywords.filter((keyword) =>
    keywordMatches(combined, keyword)
  ).length;
  const universalKeywordHits = UNIVERSAL_MACRO_KEYWORDS.filter((keyword) =>
    keywordMatches(combined, keyword)
  ).length;
  const topicNames = extractAlphaTopicNames(item);
  const topicHits = sectorConfig.alphaTopics.filter((topic) => topicNames.includes(topic)).length;

  return {
    score: sectorKeywordHits * 2 + topicHits * 2 + universalKeywordHits,
    sectorKeywordHits,
    universalKeywordHits,
    topicHits,
  };
}

function logMacroHeadlineDecision(sector, isRelevant, title, scoring, index) {
  if (index >= MAX_FILTER_LOGS) {
    return;
  }

  const reason = `sector_hits=${scoring.sectorKeywordHits} topic_hits=${scoring.topicHits} macro_hits=${scoring.universalKeywordHits} score=${scoring.score}`;
  if (isRelevant) {
    console.log(`[market-intelligence] macro relevant [${sector}] ${title} (${reason})`);
    return;
  }

  console.log(`[market-intelligence] macro discarded [${sector}] ${title} (${reason})`);
}

function extractMacroHeadlines(payload, sector, sectorConfig) {
  const feed = Array.isArray(payload?.feed) ? payload.feed : [];
  if (!feed.length) {
    return [];
  }

  const minScore = sector === 'general' ? 1 : 2;
  const relevantItems = [];
  let relevantLogCount = 0;
  let discardedLogCount = 0;

  for (const item of feed) {
    const title = typeof item?.title === 'string' ? item.title.trim() : '';
    if (!title) {
      continue;
    }

    const scoring = scoreMacroHeadline(item, sectorConfig);
    const isRelevant = scoring.score >= minScore;
    if (isRelevant) {
      logMacroHeadlineDecision(sector, true, title, scoring, relevantLogCount);
      relevantLogCount += 1;
      relevantItems.push({
        title,
        score: scoring.score,
        timePublished: typeof item?.time_published === 'string' ? item.time_published : '',
      });
    } else {
      logMacroHeadlineDecision(sector, false, title, scoring, discardedLogCount);
      discardedLogCount += 1;
    }
  }

  const sortedRelevantItems = relevantItems.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }
    return right.timePublished.localeCompare(left.timePublished);
  });

  const macroHeadlines = normalizeHeadlines(
    sortedRelevantItems.map((item) => item.title),
    MAX_MACRO_HEADLINES
  );
  console.log(
    `[market-intelligence] macro headlines filtered for sector=${sector} considered=${feed.length} relevant=${relevantItems.length} discarded=${Math.max(0, feed.length - relevantItems.length)} returned=${macroHeadlines.length}`
  );
  return macroHeadlines;
}

async function fetchCompanyNewsHeadlines(symbol, options = {}) {
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) {
    if (!missingFinnhubKeyWarningLogged) {
      console.warn('[market-intelligence] FINNHUB_API_KEY is not configured; company news disabled.');
      missingFinnhubKeyWarningLogged = true;
    }
    return [];
  }

  const endpoint = process.env.FINNHUB_COMPANY_NEWS_URL || DEFAULT_FINNHUB_BASE_URL;
  const timeoutMs = normalizePositiveNumber(process.env.FINNHUB_TIMEOUT_MS, DEFAULT_TIMEOUT_MS);
  const { from, to } = buildDateRange(options.now);
  console.log(`[market-intelligence] fetching company news for ${symbol}`);

  try {
    const response = await axios.get(endpoint, {
      params: {
        symbol,
        from,
        to,
        token: apiKey,
      },
      timeout: timeoutMs,
    });
    const companyHeadlines = extractFinnhubHeadlines(response?.data);
    console.log(
      `[market-intelligence] company news fetched for ${symbol} (${companyHeadlines.length} headlines)`
    );
    return companyHeadlines;
  } catch (error) {
    console.warn(
      `[market-intelligence] company news fetch failed for ${symbol}: ${error?.message || 'unknown error'}`
    );
    return [];
  }
}

async function fetchMacroNewsHeadlines(symbol) {
  const sector = deriveSectorFromSymbol(symbol);
  const sectorConfig = getSectorConfig(sector);
  const cachedMacroHeadlines = readMacroCacheEntry(sector);
  if (cachedMacroHeadlines) {
    console.log(`[market-intelligence] using cached macro market news for sector=${sector}`);
    return cachedMacroHeadlines;
  }

  const activeRequest = macroInFlightRequestsBySector.get(sector);
  if (activeRequest) {
    return activeRequest;
  }

  const apiKey = process.env.ALPHA_VANTAGE_API_KEY;
  if (!apiKey) {
    if (!missingAlphaVantageKeyWarningLogged) {
      console.warn('[market-intelligence] ALPHA_VANTAGE_API_KEY is not configured; macro news disabled.');
      missingAlphaVantageKeyWarningLogged = true;
    }
    return [];
  }

  const endpoint = process.env.ALPHA_VANTAGE_NEWS_URL || DEFAULT_ALPHA_VANTAGE_BASE_URL;
  const timeoutMs = normalizePositiveNumber(process.env.ALPHA_VANTAGE_TIMEOUT_MS, DEFAULT_TIMEOUT_MS);
  const minIntervalMs = normalizePositiveNumber(
    process.env.ALPHA_VANTAGE_MIN_INTERVAL_MS,
    ALPHA_VANTAGE_MIN_INTERVAL_MS
  );
  const keywordParam = sectorConfig.keywords.join(',');
  const topicsParam = sectorConfig.alphaTopics.join(',');

  const requestPromise = (async () => {
    try {
      const sinceLastRequestMs = Date.now() - Number(lastAlphaVantageRequestTimestamp || 0);
      const waitMs = Math.max(0, minIntervalMs - sinceLastRequestMs);
      if (waitMs > 0) {
        await sleep(waitMs);
      }

      console.log(
        `[market-intelligence] fetching macro market news for sector=${sector} keywords=${keywordParam || 'none'}`
      );
      const response = await axios.get(endpoint, {
        params: {
          function: 'NEWS_SENTIMENT',
          sort: 'LATEST',
          limit: 50,
          apikey: apiKey,
          ...(keywordParam ? { keywords: keywordParam } : {}),
          ...(topicsParam ? { topics: topicsParam } : {}),
        },
        timeout: timeoutMs,
      });
      lastAlphaVantageRequestTimestamp = Date.now();

      const macroHeadlines = extractMacroHeadlines(response?.data, sector, sectorConfig);
      if (macroHeadlines.length) {
        writeMacroCacheEntry(sector, macroHeadlines);
      }
      console.log(
        `[market-intelligence] macro news fetched for sector=${sector} (${macroHeadlines.length} headlines)`
      );
      return macroHeadlines;
    } catch (error) {
      console.warn(
        `[market-intelligence] macro news fetch failed for sector=${sector}: ${error?.message || 'unknown error'}`
      );
      return [];
    } finally {
      macroInFlightRequestsBySector.delete(sector);
    }
  })();

  macroInFlightRequestsBySector.set(sector, requestPromise);
  return requestPromise;
}

async function fetchRecentNewsContext(symbol, options = {}) {
  const normalizedSymbol = normalizeSymbol(symbol);
  const cacheKey = normalizedSymbol;
  const cachedEntry = readSymbolCacheEntry(cacheKey);
  if (cachedEntry) {
    console.log(`[market-intelligence] using cached news context for ${normalizedSymbol}`);
    return cachedEntry;
  }

  const activeRequest = inFlightRequests.get(cacheKey);
  if (activeRequest) {
    return activeRequest;
  }

  const requestPromise = (async () => {
    try {
      const [companyHeadlines, macroHeadlines] = await Promise.all([
        fetchCompanyNewsHeadlines(normalizedSymbol, options),
        fetchMacroNewsHeadlines(normalizedSymbol),
      ]);

      const newsContext =
        normalizeNewsContext({
          companyHeadlines,
          macroHeadlines,
        }) || createEmptyNewsContext();

      writeSymbolCacheEntry(cacheKey, newsContext);
      return newsContext;
    } catch (error) {
      console.warn(
        `[market-intelligence] failed to build context for ${normalizedSymbol}: ${error?.message || 'unknown error'}`
      );
      return createEmptyNewsContext();
    } finally {
      inFlightRequests.delete(cacheKey);
    }
  })();

  inFlightRequests.set(cacheKey, requestPromise);
  return requestPromise;
}

function __clearState() {
  symbolNewsCache.clear();
  inFlightRequests.clear();
  macroNewsCacheBySector.clear();
  macroInFlightRequestsBySector.clear();
  lastAlphaVantageRequestTimestamp = 0;
  missingFinnhubKeyWarningLogged = false;
  missingAlphaVantageKeyWarningLogged = false;
}

module.exports = {
  ALPHA_VANTAGE_MIN_INTERVAL_MS,
  MACRO_CACHE_TTL_MS,
  NEWS_CACHE_TTL_MS,
  __clearState,
  buildDateRange,
  deriveSectorFromSymbol,
  fetchRecentNewsContext,
};
