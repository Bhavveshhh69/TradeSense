const axios = require('axios');

const repository = require('./portfolio.repository');
const {
  buildTrade,
  buildHolding,
  createHttpError,
  normalizeTradeSide,
  normalizeTicker,
  resolveInstrumentMetadata,
} = require('./portfolio.model');
const symbolsService = require('../symbols/symbols.service');
const fxService = require('../services/fx.service');
const {
  generatePortfolioInsights,
} = require('../services/portfolio_intelligence.service');

const REASONING_URL = process.env.REASONING_URL || 'http://localhost:8000/predict';
const REASONING_TIMEOUT_MS = Number(process.env.REASONING_TIMEOUT_MS || 5000);
const PYTHON_API_BASE_URL =
  (process.env.PYTHON_API_BASE_URL || REASONING_URL.replace(/\/predict\/?$/, '')).replace(
    /\/$/,
    ''
  );
const DEFAULT_HISTORY_DAYS = 30;
const MAX_HISTORY_DAYS = 90;
const DEFAULT_BASE_CURRENCY = 'INR';
const SYMBOL_PRICE_CACHE_TTL_MS = 10000;
const symbolPriceCache = Object.create(null);

function roundMoney(value) {
  return Number(value.toFixed(2));
}

function roundPercent(value) {
  return Number(value.toFixed(2));
}

function roundMetric(value, decimals = 4) {
  return Number(value.toFixed(decimals));
}

function normalizeCurrencyCode(value, fallback = DEFAULT_BASE_CURRENCY) {
  if (typeof value !== 'string' || !value.trim()) {
    return fallback;
  }

  return value.trim().toUpperCase();
}

function getBaseCurrency() {
  return normalizeCurrencyCode(process.env.PORTFOLIO_BASE_CURRENCY, DEFAULT_BASE_CURRENCY);
}

function normalizeHistoryDays(daysInput) {
  if (daysInput === undefined || daysInput === null || daysInput === '') {
    return DEFAULT_HISTORY_DAYS;
  }

  const parsed = Number(daysInput);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw createHttpError(400, 'days must be a positive integer');
  }

  return Math.min(parsed, MAX_HISTORY_DAYS);
}

function extractFxErrorMessage(error, fromCurrency, toCurrency) {
  const pairLabel = `${fromCurrency}->${toCurrency}`;

  if (error?.code === 'ECONNABORTED') {
    return `FX rate request timed out (${pairLabel})`;
  }

  if (typeof error?.message === 'string' && error.message.trim()) {
    return `${error.message} (${pairLabel})`;
  }

  return `Unable to fetch FX rate (${pairLabel})`;
}

function normalizeProviderBaseUrl() {
  if (!PYTHON_API_BASE_URL) {
    return 'http://localhost:8000';
  }

  return PYTHON_API_BASE_URL;
}

function extractPriceErrorMessage(error, symbol) {
  if (error?.response?.data?.detail?.error) {
    return `${error.response.data.detail.error} (${symbol})`;
  }

  if (typeof error?.response?.data?.detail === 'string') {
    return `${error.response.data.detail} (${symbol})`;
  }

  if (error?.code === 'ECONNABORTED') {
    return `Latest price request timed out (${symbol})`;
  }

  if (typeof error?.message === 'string' && error.message.trim()) {
    return `${error.message} (${symbol})`;
  }

  return `Unable to fetch current price (${symbol})`;
}

function extractHistoryErrorMessage(error, symbol) {
  if (error?.response?.data?.detail?.error) {
    return `${error.response.data.detail.error} (${symbol})`;
  }

  if (typeof error?.response?.data?.detail === 'string') {
    return `${error.response.data.detail} (${symbol})`;
  }

  if (error?.code === 'ECONNABORTED') {
    return `Price history request timed out (${symbol})`;
  }

  if (typeof error?.message === 'string' && error.message.trim()) {
    return `${error.message} (${symbol})`;
  }

  return `Unable to fetch price history (${symbol})`;
}

async function fetchLatestPrice(symbol) {
  const requestSymbol = symbol;
  const endpoint = `${normalizeProviderBaseUrl()}/market/latest-price/${encodeURIComponent(
    requestSymbol
  )}`;

  try {
    const response = await axios.get(endpoint, {
      timeout: REASONING_TIMEOUT_MS,
    });

    const responseSymbolRaw =
      typeof response?.data?.symbol === 'string' ? response.data.symbol : requestSymbol;
    const responseSymbol = normalizeTicker(responseSymbolRaw);
    const latestPrice = Number(response?.data?.price);
    if (!Number.isFinite(latestPrice) || latestPrice <= 0) {
      return {
        requested_symbol: requestSymbol,
        response_symbol: responseSymbol,
        price: null,
        error: true,
        error_message: `Invalid price response (${requestSymbol})`,
      };
    }

    return {
      requested_symbol: requestSymbol,
      response_symbol: responseSymbol,
      price: latestPrice,
      error: false,
      error_message: null,
    };
  } catch (error) {
    return {
      requested_symbol: requestSymbol,
      response_symbol: null,
      price: null,
      error: true,
      error_message: extractPriceErrorMessage(error, requestSymbol),
    };
  }
}

function readCachedSymbolPrice(symbol) {
  const cacheEntry = symbolPriceCache[symbol];
  if (!cacheEntry || typeof cacheEntry !== 'object') {
    return null;
  }

  const ageMs = Date.now() - Number(cacheEntry.timestamp);
  if (ageMs >= SYMBOL_PRICE_CACHE_TTL_MS) {
    delete symbolPriceCache[symbol];
    return null;
  }

  const cachedPrice = Number(cacheEntry.price);
  if (!Number.isFinite(cachedPrice) || cachedPrice <= 0) {
    delete symbolPriceCache[symbol];
    return null;
  }

  return cachedPrice;
}

function writeCachedSymbolPrice(symbol, price) {
  symbolPriceCache[symbol] = {
    price,
    timestamp: Date.now(),
  };
}

function clearCachedSymbolPrices() {
  Object.keys(symbolPriceCache).forEach((symbol) => {
    delete symbolPriceCache[symbol];
  });
}

async function fetchLatestPriceWithCache(symbol) {
  const cachedPrice = readCachedSymbolPrice(symbol);
  if (cachedPrice !== null) {
    return {
      requested_symbol: symbol,
      response_symbol: symbol,
      price: cachedPrice,
      error: false,
      error_message: null,
    };
  }

  const latestPrice = await fetchLatestPrice(symbol);

  if (!latestPrice?.error) {
    const livePrice = Number(latestPrice.price);
    if (Number.isFinite(livePrice) && livePrice > 0) {
      writeCachedSymbolPrice(symbol, livePrice);
    }
  }

  return latestPrice;
}

async function normalizePortfolioTicker(rawTicker) {
  try {
    const resolvedSymbol = await symbolsService.normalizeSymbol(rawTicker);
    return normalizeTicker(resolvedSymbol);
  } catch (error) {
    return normalizeTicker(rawTicker);
  }
}

async function resolveFxRateToBase(instrumentCurrency, baseCurrency) {
  const sourceCurrency = normalizeCurrencyCode(instrumentCurrency);
  const targetCurrency = normalizeCurrencyCode(baseCurrency);

  if (sourceCurrency === targetCurrency) {
    return 1;
  }

  const fxRate = await fxService.getFxRate(sourceCurrency, targetCurrency);
  const numericRate = Number(fxRate);
  if (!Number.isFinite(numericRate) || numericRate <= 0) {
    throw new Error(`Invalid FX rate ${sourceCurrency}->${targetCurrency}`);
  }

  return numericRate;
}

function formatUtcDate(date) {
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, '0');
  const day = String(date.getUTCDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function buildHistoryDateRange(days) {
  const todayUtc = new Date();
  const end = new Date(
    Date.UTC(todayUtc.getUTCFullYear(), todayUtc.getUTCMonth(), todayUtc.getUTCDate())
  );
  const dates = [];

  for (let offset = days - 1; offset >= 0; offset -= 1) {
    const pointDate = new Date(end);
    pointDate.setUTCDate(end.getUTCDate() - offset);
    dates.push(formatUtcDate(pointDate));
  }

  return dates;
}

function normalizeHistoryPrices(rawPrices) {
  if (!Array.isArray(rawPrices)) {
    return [];
  }

  const result = [];
  for (const point of rawPrices) {
    const date = typeof point?.date === 'string' ? point.date.trim() : '';
    const close = Number(point?.close);
    if (!date || !Number.isFinite(close) || close <= 0) {
      continue;
    }
    result.push({ date, close });
  }

  result.sort((a, b) => a.date.localeCompare(b.date));
  return result;
}

async function fetchHistoricalPrices(symbol, days) {
  const requestSymbol = symbol;
  const endpoint = `${normalizeProviderBaseUrl()}/market/history/${encodeURIComponent(
    requestSymbol
  )}?days=${days}`;

  try {
    const response = await axios.get(endpoint, {
      timeout: REASONING_TIMEOUT_MS,
    });
    const responseSymbolRaw =
      typeof response?.data?.symbol === 'string' ? response.data.symbol : requestSymbol;
    const responseSymbol = normalizeTicker(responseSymbolRaw);
    const prices = normalizeHistoryPrices(response?.data?.history);

    if (!prices.length) {
      return {
        requested_symbol: requestSymbol,
        response_symbol: responseSymbol,
        prices: [],
        error: true,
        error_message: `Invalid price history response (${requestSymbol})`,
      };
    }

    return {
      requested_symbol: requestSymbol,
      response_symbol: responseSymbol,
      prices,
      error: false,
      error_message: null,
    };
  } catch (error) {
    return {
      requested_symbol: requestSymbol,
      response_symbol: null,
      prices: [],
      error: true,
      error_message: extractHistoryErrorMessage(error, requestSymbol),
    };
  }
}

function classifyConcentrationRisk(largestWeight) {
  if (largestWeight > 0.6) {
    return 'HIGH';
  }

  if (largestWeight > 0.35) {
    return 'MODERATE';
  }

  return 'LOW';
}

function classifyDiversificationRisk(score) {
  if (score < 2) {
    return 'HIGH';
  }

  if (score < 4) {
    return 'MODERATE';
  }

  return 'LOW';
}

function calculateStandardDeviation(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return 0;
  }

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function classifyVolatility(volatility) {
  if (volatility < 0.01) {
    return 'LOW';
  }

  if (volatility <= 0.02) {
    return 'MODERATE';
  }

  return 'HIGH';
}

function getDisplayTicker(ticker) {
  if (typeof ticker !== 'string' || !ticker.trim()) {
    return 'the largest position';
  }

  const [baseSymbol] = ticker.split('.');
  return baseSymbol || ticker;
}

function normalizePercentValue(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return null;
  }

  return numericValue <= 1 ? numericValue * 100 : numericValue;
}

function generatePortfolioRecommendations(metrics) {
  const payload = metrics && typeof metrics === 'object' ? metrics : {};
  const recommendations = [];

  const concentrationRisk =
    typeof payload.concentration_risk === 'string' ? payload.concentration_risk : 'LOW';
  const largestPosition =
    payload.largest_position && typeof payload.largest_position === 'object'
      ? payload.largest_position
      : null;
  const largestTicker =
    largestPosition && typeof largestPosition.ticker === 'string' ? largestPosition.ticker : null;
  const diversificationScore = Number(payload.diversification_score);
  const volatilityLevel =
    typeof payload.volatility_level === 'string' ? payload.volatility_level : 'LOW';
  const bestPerformer =
    payload.best_performer && typeof payload.best_performer === 'object'
      ? payload.best_performer
      : null;
  const bestPerformerWeightPercent = normalizePercentValue(bestPerformer?.weight);

  if (concentrationRisk === 'HIGH') {
    recommendations.push(
      largestTicker
        ? `Reduce ${getDisplayTicker(largestTicker)} exposure`
        : 'Reduce concentration in the largest position'
    );
  }

  if (Number.isFinite(diversificationScore) && diversificationScore < 2) {
    recommendations.push('Add more assets to diversify portfolio');
  }

  if (volatilityLevel === 'HIGH') {
    recommendations.push('Consider rebalancing high-risk positions');
  }

  if (bestPerformer && Number.isFinite(bestPerformerWeightPercent) && bestPerformerWeightPercent > 60) {
    recommendations.push(`Book partial profits in ${getDisplayTicker(bestPerformer.ticker)}`);
  }

  if (recommendations.length === 0) {
    recommendations.push('Portfolio allocation looks balanced. Continue periodic rebalancing.');
  }

  return { recommendations };
}

function sortTradesChronologically(trades) {
  return [...trades].sort((left, right) => {
    const dateCompare = String(left.occurred_at || '').localeCompare(String(right.occurred_at || ''));
    if (dateCompare !== 0) {
      return dateCompare;
    }
    return String(left.id || '').localeCompare(String(right.id || ''));
  });
}

function signedQuantityDelta(side, quantity) {
  return side === 'BUY' || side === 'COVER' ? quantity : -quantity;
}

function createPositionState(seed) {
  return {
    ticker: seed.ticker,
    symbol: seed.symbol || seed.ticker,
    display_name: seed.display_name || seed.ticker,
    market: seed.market || null,
    exchange: seed.exchange || null,
    instrument_type: seed.instrument_type || null,
    instrument_currency: seed.instrument_currency || resolveInstrumentMetadata(seed.ticker).instrument_currency,
    quantity_signed: 0,
    average_price: 0,
    realized_pnl_native: 0,
    last_trade_at: null,
  };
}

function applyTradeToPositionState(state, trade, options = {}) {
  const validate = options.validate === true;
  const quantity = Number(trade.quantity);
  const price = Number(trade.price);
  const side = normalizeTradeSide(trade.side);

  if (!Number.isFinite(quantity) || quantity <= 0) {
    throw createHttpError(400, 'quantity must be a positive number');
  }
  if (!Number.isFinite(price) || price <= 0) {
    throw createHttpError(400, 'price must be a positive number');
  }

  if (side === 'BUY') {
    if (validate && state.quantity_signed < 0) {
      throw createHttpError(400, 'Use COVER to reduce an existing short position');
    }

    const existingQuantity = Math.max(state.quantity_signed, 0);
    const totalQuantity = existingQuantity + quantity;
    state.average_price =
      existingQuantity <= 0
        ? price
        : ((state.average_price * existingQuantity) + price * quantity) / totalQuantity;
    state.quantity_signed = existingQuantity + quantity;
  }

  if (side === 'SELL') {
    if (validate && state.quantity_signed <= 0) {
      throw createHttpError(400, 'Use SHORT to open a new short position');
    }
    if (validate && quantity > state.quantity_signed) {
      throw createHttpError(400, 'SELL quantity exceeds the current long position');
    }

    const closingQuantity = Math.min(quantity, Math.max(state.quantity_signed, 0));
    state.realized_pnl_native += (price - state.average_price) * closingQuantity;
    state.quantity_signed -= closingQuantity;
    if (state.quantity_signed === 0) {
      state.average_price = 0;
    }
  }

  if (side === 'SHORT') {
    if (validate && state.quantity_signed > 0) {
      throw createHttpError(400, 'Use SELL to reduce an existing long position');
    }

    const existingQuantity = Math.abs(Math.min(state.quantity_signed, 0));
    const totalQuantity = existingQuantity + quantity;
    state.average_price =
      existingQuantity <= 0
        ? price
        : ((state.average_price * existingQuantity) + price * quantity) / totalQuantity;
    state.quantity_signed = -(existingQuantity + quantity);
  }

  if (side === 'COVER') {
    if (validate && state.quantity_signed >= 0) {
      throw createHttpError(400, 'Use BUY to open a new long position');
    }
    if (validate && quantity > Math.abs(state.quantity_signed)) {
      throw createHttpError(400, 'COVER quantity exceeds the current short position');
    }

    const closingQuantity = Math.min(quantity, Math.abs(Math.min(state.quantity_signed, 0)));
    state.realized_pnl_native += (state.average_price - price) * closingQuantity;
    state.quantity_signed += closingQuantity;
    if (state.quantity_signed === 0) {
      state.average_price = 0;
    }
  }

  state.average_price = Number.isFinite(state.average_price) ? roundMetric(state.average_price, 6) : 0;
  state.realized_pnl_native = roundMoney(state.realized_pnl_native);
  state.last_trade_at = trade.occurred_at;
}

function derivePositionStates(trades) {
  const states = new Map();

  for (const trade of sortTradesChronologically(trades)) {
    const ticker = normalizeTicker(trade.ticker);
    const existingState = states.get(ticker) || createPositionState(trade);
    existingState.symbol = trade.symbol || existingState.symbol;
    existingState.display_name = trade.display_name || existingState.display_name;
    existingState.market = trade.market || existingState.market;
    existingState.exchange = trade.exchange || existingState.exchange;
    existingState.instrument_type = trade.instrument_type || existingState.instrument_type;
    existingState.instrument_currency =
      trade.instrument_currency || existingState.instrument_currency;
    applyTradeToPositionState(existingState, trade, { validate: false });
    states.set(ticker, existingState);
  }

  return states;
}

function buildQuantityTimeline(dates, tradesForSymbol) {
  const timeline = [];
  const sortedTrades = sortTradesChronologically(tradesForSymbol);
  let signedQuantity = 0;
  let tradeIndex = 0;

  for (const date of dates) {
    while (tradeIndex < sortedTrades.length) {
      const trade = sortedTrades[tradeIndex];
      const tradeDate = String(trade.occurred_at || '').slice(0, 10);
      if (!tradeDate || tradeDate > date) {
        break;
      }
      signedQuantity += signedQuantityDelta(trade.side, Number(trade.quantity));
      tradeIndex += 1;
    }
    timeline.push(signedQuantity);
  }

  return timeline;
}

async function buildTradeFromInput(input, options = {}) {
  const payload = input || {};
  const rawTicker = payload.ticker ?? payload.symbol ?? payload.normalized;
  if (typeof rawTicker !== 'string' || !rawTicker.trim()) {
    throw createHttpError(400, 'ticker is required');
  }

  let instrument;
  if (typeof symbolsService.resolveInstrument === 'function') {
    instrument = await symbolsService.resolveInstrument(rawTicker);
  } else {
    const normalized = await normalizePortfolioTicker(rawTicker);
    const metadata = resolveInstrumentMetadata(normalized);
    instrument = {
      symbol: normalized.replace(/\.(NS|BO)$/i, ''),
      normalized,
      display_name: normalized,
      market: metadata.market,
      exchange: metadata.exchange,
      instrument_type: metadata.instrument_type,
    };
  }
  const side = normalizeTradeSide(options.sideOverride || payload.side);

  return buildTrade({
    ...payload,
    ticker: instrument.normalized,
    symbol: instrument.symbol,
    normalized: instrument.normalized,
    display_name: instrument.display_name,
    market: instrument.market,
    exchange: instrument.exchange,
    instrument_type: instrument.instrument_type,
    instrument_currency: resolveInstrumentMetadata(instrument.normalized).instrument_currency,
    side,
  });
}

async function migrateLegacyHoldingsToTrades() {
  const existingTrades =
    typeof repository.getAllTrades === 'function' ? await repository.getAllTrades() : [];
  if (Array.isArray(existingTrades) && existingTrades.length > 0) {
    return existingTrades;
  }

  const legacyHoldings = await repository.getAllHoldings();
  if (!Array.isArray(legacyHoldings) || legacyHoldings.length === 0) {
    return [];
  }

  const backfilledTrades = legacyHoldings.map((holding) =>
    buildTrade({
      ticker: holding.ticker,
      symbol: holding.symbol,
      display_name: holding.display_name,
      market: holding.market,
      exchange: holding.exchange,
      instrument_type: holding.instrument_type,
      instrument_currency: holding.instrument_currency,
      side: 'BUY',
      quantity: holding.shares,
      price: holding.buy_price,
      source: 'legacy_backfill',
      note: 'Backfilled from legacy holdings',
      occurred_at: holding.added_at,
    })
  );

  if (typeof repository.replaceAllTrades === 'function') {
    await repository.replaceAllTrades(backfilledTrades);
  }
  return backfilledTrades;
}

async function getLedgerTrades() {
  const existingTrades =
    typeof repository.getAllTrades === 'function' ? await repository.getAllTrades() : [];
  if (Array.isArray(existingTrades) && existingTrades.length > 0) {
    return sortTradesChronologically(existingTrades);
  }

  const migratedTrades = await migrateLegacyHoldingsToTrades();
  return sortTradesChronologically(migratedTrades);
}

async function getPositionStateForSymbol(symbol) {
  const normalizedTicker = await normalizePortfolioTicker(symbol);
  const states = derivePositionStates(await getLedgerTrades());
  return states.get(normalizedTicker) || createPositionState({ ticker: normalizedTicker });
}

function buildPriceErrorPayload(symbol) {
  return {
    requested_symbol: symbol,
    response_symbol: null,
    price: null,
    error: true,
    error_message: `Unable to fetch current price (${symbol})`,
  };
}

async function enrichTransactions(trades, baseCurrency) {
  return Promise.all(
    trades.map(async (trade) => {
      let fxRateToBase = null;
      let fxErrorMessage = null;

      try {
        fxRateToBase = await resolveFxRateToBase(trade.instrument_currency, baseCurrency);
      } catch (error) {
        fxErrorMessage = extractFxErrorMessage(error, trade.instrument_currency, baseCurrency);
      }

      return {
        ...trade,
        signed_quantity: signedQuantityDelta(trade.side, Number(trade.quantity)),
        fx_rate_to_base:
          Number.isFinite(Number(fxRateToBase)) && Number(fxRateToBase) > 0
            ? roundMetric(Number(fxRateToBase), 6)
            : null,
        price_base:
          Number.isFinite(Number(fxRateToBase)) && Number(fxRateToBase) > 0
            ? roundMoney(Number(trade.price) * Number(fxRateToBase))
            : null,
        base_currency: baseCurrency,
        price_error: fxRateToBase === null,
        price_error_message: fxErrorMessage,
      };
    })
  );
}

async function buildPortfolioSnapshot() {
  const baseCurrency = getBaseCurrency();
  const trades = await getLedgerTrades();
  const positionStates = derivePositionStates(trades);
  const activeStates = [...positionStates.values()].filter(
    (state) => Number.isFinite(Number(state.quantity_signed)) && Number(state.quantity_signed) !== 0
  );

  if (activeStates.length === 0) {
    const emptyPayload = {
      holdings: [],
      positions: [],
      summary: {
        total_portfolio_value: 0,
        total_gross_exposure: 0,
        total_net_exposure: 0,
        total_invested_value: 0,
        total_unrealized_pnl: 0,
        total_realized_pnl: roundMoney(
          [...positionStates.values()].reduce((sum, state) => sum + Number(state.realized_pnl_native || 0), 0)
        ),
        total_profit_loss: roundMoney(
          [...positionStates.values()].reduce((sum, state) => sum + Number(state.realized_pnl_native || 0), 0)
        ),
        total_profit_loss_percent: 0,
        active_positions: 0,
        long_positions: 0,
        short_positions: 0,
        winners_count: 0,
        losers_count: 0,
        has_price_errors: false,
        base_currency: baseCurrency,
      },
    };

    return {
      ...emptyPayload,
      portfolio_intelligence: generatePortfolioInsights(emptyPayload),
    };
  }

  const fxRatesByCurrency = new Map();
  for (const state of positionStates.values()) {
    const currency = normalizeCurrencyCode(state.instrument_currency);
    if (fxRatesByCurrency.has(currency)) {
      continue;
    }

    try {
      fxRatesByCurrency.set(currency, {
        rate: await resolveFxRateToBase(currency, baseCurrency),
        error: null,
      });
    } catch (error) {
      fxRatesByCurrency.set(currency, {
        rate: null,
        error: extractFxErrorMessage(error, currency, baseCurrency),
      });
    }
  }

  const latestPriceEntries = await Promise.all(
    activeStates.map(async (state) => [state.ticker, await fetchLatestPriceWithCache(state.ticker)])
  );
  const latestPriceBySymbol = new Map(latestPriceEntries);

  const holdings = [];
  let totalGrossExposure = 0;
  let totalNetExposure = 0;
  let totalInvestedValue = 0;
  let totalUnrealizedPnl = 0;
  let totalRealizedPnl = 0;
  let winnersCount = 0;
  let losersCount = 0;
  let hasPriceErrors = false;

  for (const state of [...activeStates].sort((left, right) =>
    left.ticker.localeCompare(right.ticker)
  )) {
    const latestPrice = latestPriceBySymbol.get(state.ticker) || buildPriceErrorPayload(state.ticker);
    const fxInfo =
      fxRatesByCurrency.get(normalizeCurrencyCode(state.instrument_currency)) || {
        rate: null,
        error: null,
      };
    const fxRateToBase =
      Number.isFinite(Number(fxInfo.rate)) && Number(fxInfo.rate) > 0
        ? roundMetric(Number(fxInfo.rate), 6)
        : null;
    const currentPriceNative =
      Number.isFinite(Number(latestPrice.price)) && Number(latestPrice.price) > 0
        ? roundMoney(Number(latestPrice.price))
        : null;
    const currentPriceBase =
      currentPriceNative !== null && fxRateToBase !== null
        ? roundMoney(currentPriceNative * fxRateToBase)
        : null;
    const quantitySigned = Number(state.quantity_signed);
    const quantityAbs = Math.abs(quantitySigned);
    const grossExposureBase =
      currentPriceBase === null ? null : roundMoney(quantityAbs * currentPriceBase);
    const netMarketValueBase =
      currentPriceBase === null ? null : roundMoney(quantitySigned * currentPriceBase);
    const investedValueBase =
      fxRateToBase === null ? null : roundMoney(quantityAbs * Number(state.average_price) * fxRateToBase);
    const realizedPnlBase =
      fxRateToBase === null
        ? roundMoney(Number(state.realized_pnl_native))
        : roundMoney(Number(state.realized_pnl_native) * fxRateToBase);

    let unrealizedPnlBase = null;
    if (currentPriceBase !== null && fxRateToBase !== null) {
      const avgPriceBase = Number(state.average_price) * fxRateToBase;
      if (quantitySigned > 0) {
        unrealizedPnlBase = roundMoney((currentPriceBase - avgPriceBase) * quantityAbs);
      } else {
        unrealizedPnlBase = roundMoney((avgPriceBase - currentPriceBase) * quantityAbs);
      }
    }

    const totalPnlBase =
      unrealizedPnlBase === null ? null : roundMoney(realizedPnlBase + unrealizedPnlBase);
    const totalPnlPercent =
      totalPnlBase === null || !Number.isFinite(Number(investedValueBase)) || Number(investedValueBase) <= 0
        ? null
        : roundPercent((totalPnlBase / Number(investedValueBase)) * 100);
    const priceError = Boolean(latestPrice.error) || fxRateToBase === null;
    const priceErrorMessage = latestPrice.error ? latestPrice.error_message : fxInfo.error;

    if (grossExposureBase === null || currentPriceBase === null) {
      hasPriceErrors = true;
    } else {
      totalGrossExposure += grossExposureBase;
      totalNetExposure += Number(netMarketValueBase || 0);
    }

    if (investedValueBase === null) {
      hasPriceErrors = true;
    } else {
      totalInvestedValue += investedValueBase;
    }

    if (unrealizedPnlBase === null) {
      hasPriceErrors = true;
    } else {
      totalUnrealizedPnl += unrealizedPnlBase;
    }

    totalRealizedPnl += realizedPnlBase;

    if (Number.isFinite(Number(totalPnlBase))) {
      if (Number(totalPnlBase) > 0) {
        winnersCount += 1;
      } else if (Number(totalPnlBase) < 0) {
        losersCount += 1;
      }
    }

    holdings.push({
      id: state.ticker,
      ticker: state.ticker,
      symbol: state.symbol,
      normalized: state.ticker,
      display_name: state.display_name,
      market: state.market,
      exchange: state.exchange,
      instrument_type: state.instrument_type,
      instrument_currency: state.instrument_currency,
      base_currency: baseCurrency,
      side: quantitySigned > 0 ? 'LONG' : 'SHORT',
      quantity: quantityAbs,
      quantity_signed: quantitySigned,
      shares: quantityAbs,
      avg_price: roundMoney(Number(state.average_price)),
      buy_price: roundMoney(Number(state.average_price)),
      price_native: currentPriceNative,
      fx_rate_to_base: fxRateToBase,
      price_base: currentPriceBase,
      current_price: currentPriceBase,
      current_value: grossExposureBase,
      gross_exposure_base: grossExposureBase,
      net_market_value_base: netMarketValueBase,
      invested_value: investedValueBase,
      unrealized_pnl: unrealizedPnlBase,
      realized_pnl: realizedPnlBase,
      profit_loss: totalPnlBase,
      profit_loss_percent: totalPnlPercent,
      price_error: priceError,
      price_error_message: priceError ? priceErrorMessage : null,
      last_trade_at: state.last_trade_at,
    });
  }

  const summary = {
    total_portfolio_value: roundMoney(totalGrossExposure),
    total_gross_exposure: roundMoney(totalGrossExposure),
    total_net_exposure: roundMoney(totalNetExposure),
    total_invested_value: roundMoney(totalInvestedValue),
    total_unrealized_pnl: roundMoney(totalUnrealizedPnl),
    total_realized_pnl: roundMoney(totalRealizedPnl),
    total_profit_loss: roundMoney(totalRealizedPnl + totalUnrealizedPnl),
    total_profit_loss_percent:
      totalInvestedValue > 0
        ? roundPercent(((totalRealizedPnl + totalUnrealizedPnl) / totalInvestedValue) * 100)
        : 0,
    active_positions: holdings.length,
    long_positions: holdings.filter((holding) => holding.side === 'LONG').length,
    short_positions: holdings.filter((holding) => holding.side === 'SHORT').length,
    winners_count: winnersCount,
    losers_count: losersCount,
    has_price_errors: hasPriceErrors,
    base_currency: baseCurrency,
  };

  const payload = {
    holdings,
    positions: holdings,
    summary,
  };

  return {
    ...payload,
    portfolio_intelligence: generatePortfolioInsights(payload),
  };
}

async function getPortfolioHistory(daysInput) {
  const days = normalizeHistoryDays(daysInput);
  const baseCurrency = getBaseCurrency();
  const trades = await getLedgerTrades();
  const dates = buildHistoryDateRange(days);

  if (!Array.isArray(trades) || trades.length === 0) {
    return {
      symbol_count: 0,
      days,
      equity_curve: dates.map((date) => ({
        date,
        portfolio_value: 0,
      })),
    };
  }

  const tradesBySymbol = trades.reduce((map, trade) => {
    const symbolTrades = map.get(trade.ticker) || [];
    symbolTrades.push(trade);
    map.set(trade.ticker, symbolTrades);
    return map;
  }, new Map());
  const symbols = [...tradesBySymbol.keys()];

  const historyResults = await Promise.all(
    symbols.map(async (symbol) => [symbol, await fetchHistoricalPrices(symbol, days)])
  );
  const fxRateResults = await Promise.all(
    symbols.map(async (symbol) => {
      const instrumentMetadata = resolveInstrumentMetadata(symbol);
      try {
        const fxRateToBase = await resolveFxRateToBase(
          instrumentMetadata.instrument_currency,
          baseCurrency
        );
        return [symbol, fxRateToBase, null];
      } catch (error) {
        return [
          symbol,
          null,
          extractFxErrorMessage(error, instrumentMetadata.instrument_currency, baseCurrency),
        ];
      }
    })
  );

  const symbolHistoryMap = new Map();
  const symbolFxRateMap = new Map();
  const fxRateBySymbol = new Map(
    fxRateResults.map(([symbol, fxRate, errorMessage]) => [symbol, { fxRate, errorMessage }])
  );
  let successfulSymbolCount = 0;

  for (const [symbol, result] of historyResults) {
    const fxResult = fxRateBySymbol.get(symbol);
    const fxRateToBase = Number(fxResult?.fxRate);
    const symbolMatches = result.response_symbol === symbol;
    if (
      !result.error &&
      symbolMatches &&
      result.prices.length > 0 &&
      Number.isFinite(fxRateToBase) &&
      fxRateToBase > 0
    ) {
      symbolHistoryMap.set(symbol, result.prices);
      symbolFxRateMap.set(symbol, fxRateToBase);
      successfulSymbolCount += 1;
      continue;
    }

    symbolHistoryMap.set(symbol, []);
    symbolFxRateMap.set(symbol, 0);
  }

  if (successfulSymbolCount === 0) {
    return {
      symbol_count: symbols.length,
      days,
      equity_curve: dates.map((date) => ({
        date,
        portfolio_value: 0,
      })),
    };
  }

  const symbolTimelineMap = new Map(
    symbols.map((symbol) => [symbol, buildQuantityTimeline(dates, tradesBySymbol.get(symbol) || [])])
  );

  const historyStates = new Map();
  for (const [symbol, prices] of symbolHistoryMap.entries()) {
    historyStates.set(symbol, {
      index: 0,
      lastKnownPrice: null,
      prices,
    });
  }

  const equityCurve = dates.map((date, dateIndex) => {
    let portfolioValue = 0;

    for (const symbol of symbols) {
      const timeline = symbolTimelineMap.get(symbol) || [];
      const signedQuantity = Number(timeline[dateIndex] || 0);
      const quantityAbs = Math.abs(signedQuantity);
      if (quantityAbs <= 0) {
        continue;
      }

      const historyState = historyStates.get(symbol);
      const fxRateToBase = Number(symbolFxRateMap.get(symbol) || 0);
      if (!historyState || !Array.isArray(historyState.prices) || historyState.prices.length === 0) {
        continue;
      }
      if (!Number.isFinite(fxRateToBase) || fxRateToBase <= 0) {
        continue;
      }

      while (
        historyState.index < historyState.prices.length &&
        historyState.prices[historyState.index].date <= date
      ) {
        historyState.lastKnownPrice = historyState.prices[historyState.index].close;
        historyState.index += 1;
      }

      if (
        typeof historyState.lastKnownPrice === 'number' &&
        Number.isFinite(historyState.lastKnownPrice)
      ) {
        portfolioValue += quantityAbs * historyState.lastKnownPrice * fxRateToBase;
      }
    }

    return {
      date,
      portfolio_value: roundMoney(portfolioValue),
    };
  });

  return {
    symbol_count: symbols.length,
    days,
    equity_curve: equityCurve,
  };
}

async function getPortfolioInsights(daysInput) {
  const holdingsPayload = await getHoldings();
  const historyPayload = await getPortfolioHistory(daysInput);

  const holdings = Array.isArray(holdingsPayload?.holdings) ? holdingsPayload.holdings : [];
  const holdingsWithValue = holdings.filter((holding) => {
    const currentValue = Number(holding?.current_value);
    return Number.isFinite(currentValue) && currentValue > 0;
  });

  const totalPortfolioValue = holdingsWithValue.reduce(
    (sum, holding) => sum + Number(holding.current_value),
    0
  );
  const hasTotalValue = Number.isFinite(totalPortfolioValue) && totalPortfolioValue > 0;

  const weightedHoldings = hasTotalValue
    ? holdingsWithValue.map((holding) => {
        const currentValue = Number(holding.current_value);
        const weightRatio = currentValue / totalPortfolioValue;
        return {
          ticker: holding.ticker,
          current_value: roundMoney(currentValue),
          weight_ratio: weightRatio,
          weight_percent: roundPercent(weightRatio * 100),
        };
      })
    : [];
  const weightByTicker = weightedHoldings.reduce((map, holding) => {
    const previousWeight = map.get(holding.ticker) || 0;
    map.set(holding.ticker, previousWeight + holding.weight_percent);
    return map;
  }, new Map());

  const largestHolding =
    weightedHoldings.length > 0
      ? [...weightedHoldings].sort((a, b) => b.weight_ratio - a.weight_ratio)[0]
      : null;
  const largestWeight = largestHolding ? largestHolding.weight_ratio : 0;
  const concentrationRisk = classifyConcentrationRisk(largestWeight);

  const validPerformers = holdingsWithValue.filter((holding) => {
    const pnlPercent = Number(holding?.profit_loss_percent);
    return Number.isFinite(pnlPercent);
  });
  const bestPerformer =
    validPerformers.length > 0
      ? [...validPerformers].sort(
          (a, b) => Number(b.profit_loss_percent) - Number(a.profit_loss_percent)
        )[0]
      : null;
  const worstPerformer =
    validPerformers.length > 0
      ? [...validPerformers].sort(
          (a, b) => Number(a.profit_loss_percent) - Number(b.profit_loss_percent)
        )[0]
      : null;

  const weights = weightedHoldings.map((holding) => holding.weight_ratio);
  const sumOfSquares = weights.reduce((sum, weight) => sum + weight ** 2, 0);
  const diversificationScore = sumOfSquares > 0 ? 1 / sumOfSquares : 0;
  const diversificationRisk = classifyDiversificationRisk(diversificationScore);

  const equityCurve = Array.isArray(historyPayload?.equity_curve) ? historyPayload.equity_curve : [];
  const dailyReturns = [];
  for (let i = 1; i < equityCurve.length; i += 1) {
    const today = Number(equityCurve[i]?.portfolio_value);
    const yesterday = Number(equityCurve[i - 1]?.portfolio_value);

    if (!Number.isFinite(today) || !Number.isFinite(yesterday) || yesterday <= 0) {
      continue;
    }

    const dailyReturn = (today - yesterday) / yesterday;
    if (Number.isFinite(dailyReturn)) {
      dailyReturns.push(dailyReturn);
    }
  }

  const volatility = calculateStandardDeviation(dailyReturns);
  const volatilityLevel = classifyVolatility(volatility);

  const insights = [];
  if (largestHolding && concentrationRisk === 'HIGH') {
    insights.push(`Portfolio is highly concentrated in ${largestHolding.ticker}.`);
  } else if (largestHolding && concentrationRisk === 'MODERATE') {
    insights.push(`Portfolio has moderate concentration in ${largestHolding.ticker}.`);
  } else if (weightedHoldings.length > 0) {
    insights.push('Portfolio concentration is relatively balanced.');
  } else {
    insights.push('No active positions available to assess concentration.');
  }

  if (diversificationRisk === 'HIGH') {
    insights.push('Diversification risk is high.');
  } else if (diversificationRisk === 'MODERATE') {
    insights.push('Diversification can be improved across more uncorrelated positions.');
  } else {
    insights.push('Diversification is healthy at the current allocation.');
  }

  if (largestHolding && concentrationRisk !== 'LOW') {
    insights.push(`Consider reducing exposure to ${getDisplayTicker(largestHolding.ticker)}.`);
  }

  if (bestPerformer) {
    insights.push(
      `Best performer is ${bestPerformer.ticker} (${roundPercent(
        Number(bestPerformer.profit_loss_percent)
      )}%).`
    );
  }

  if (worstPerformer && worstPerformer.id !== bestPerformer?.id) {
    insights.push(
      `Weakest performer is ${worstPerformer.ticker} (${roundPercent(
        Number(worstPerformer.profit_loss_percent)
      )}%).`
    );
  }

  if (volatilityLevel === 'HIGH') {
    insights.push('Portfolio volatility is high based on recent daily equity swings.');
  } else if (volatilityLevel === 'MODERATE') {
    insights.push('Portfolio volatility is moderate over recent sessions.');
  } else {
    insights.push('Portfolio volatility remains low over recent sessions.');
  }

  return {
    concentration_risk: concentrationRisk,
    largest_position: largestHolding
      ? {
          ticker: largestHolding.ticker,
          weight: largestHolding.weight_percent,
          current_value: largestHolding.current_value,
        }
      : null,
    best_performer: bestPerformer
      ? {
          ticker: bestPerformer.ticker,
          profit_loss_percent: roundPercent(Number(bestPerformer.profit_loss_percent)),
          weight: roundPercent(weightByTicker.get(bestPerformer.ticker) || 0),
        }
      : null,
    worst_performer: worstPerformer
      ? {
          ticker: worstPerformer.ticker,
          profit_loss_percent: roundPercent(Number(worstPerformer.profit_loss_percent)),
        }
      : null,
    diversification_score: roundMetric(diversificationScore, 2),
    volatility_level: volatilityLevel,
    insights,
  };
}

async function getPortfolioAdvisor(daysInput) {
  const metrics = await getPortfolioInsights(daysInput);
  return generatePortfolioRecommendations(metrics);
}

async function createTrade(input) {
  const trade = await buildTradeFromInput(input);
  const currentState = await getPositionStateForSymbol(trade.ticker);
  applyTradeToPositionState(currentState, trade, { validate: true });
  const [createdTrade] = await repository.appendTrades([trade]);
  return createdTrade;
}

async function adjustPosition(symbol, input) {
  const payload = input || {};
  const normalizedTicker = await normalizePortfolioTicker(symbol);
  const targetQuantityRaw = Number(payload.target_quantity);
  if (!Number.isFinite(targetQuantityRaw)) {
    throw createHttpError(400, 'target_quantity must be a valid number');
  }

  const targetQuantity = roundMetric(targetQuantityRaw, 6);
  const currentState = await getPositionStateForSymbol(normalizedTicker);
  const currentQuantity = roundMetric(Number(currentState.quantity_signed || 0), 6);

  if (currentQuantity === targetQuantity) {
    return [];
  }

  let referencePrice = null;
  if (payload.price !== undefined && payload.price !== null && payload.price !== '') {
    const numericPrice = Number(payload.price);
    if (!Number.isFinite(numericPrice) || numericPrice <= 0) {
      throw createHttpError(400, 'price must be a positive number');
    }
    referencePrice = numericPrice;
  }

  if (referencePrice === null && Number(currentState.average_price) > 0) {
    referencePrice = Number(currentState.average_price);
  }

  if (referencePrice === null) {
    const latestPrice = await fetchLatestPriceWithCache(normalizedTicker);
    if (!latestPrice.error && Number(latestPrice.price) > 0) {
      referencePrice = Number(latestPrice.price);
    }
  }

  if (referencePrice === null) {
    throw createHttpError(400, 'price is required to adjust a position without an existing cost basis');
  }

  const adjustmentNote =
    typeof payload.note === 'string' && payload.note.trim()
      ? payload.note.trim()
      : `Adjustment to target quantity ${targetQuantity}`;

  const generatedTrades = [];
  let workingQuantity = currentQuantity;

  const enqueueTrade = async (side, quantity) => {
    if (!Number.isFinite(quantity) || quantity <= 0) {
      return;
    }
    const trade = await buildTradeFromInput(
      {
        ticker: normalizedTicker,
        quantity,
        price: referencePrice,
        note: adjustmentNote,
        source: 'adjustment',
      },
      { sideOverride: side }
    );
    generatedTrades.push(trade);
    workingQuantity += signedQuantityDelta(side, quantity);
  };

  if (targetQuantity === 0) {
    if (workingQuantity > 0) {
      await enqueueTrade('SELL', workingQuantity);
    } else if (workingQuantity < 0) {
      await enqueueTrade('COVER', Math.abs(workingQuantity));
    }
  } else if (targetQuantity > 0) {
    if (workingQuantity < 0) {
      await enqueueTrade('COVER', Math.abs(workingQuantity));
    }
    if (workingQuantity < targetQuantity) {
      await enqueueTrade('BUY', targetQuantity - workingQuantity);
    } else if (workingQuantity > targetQuantity) {
      await enqueueTrade('SELL', workingQuantity - targetQuantity);
    }
  } else if (targetQuantity < 0) {
    if (workingQuantity > 0) {
      await enqueueTrade('SELL', workingQuantity);
    }
    if (workingQuantity > targetQuantity) {
      await enqueueTrade('SHORT', Math.abs(targetQuantity - workingQuantity));
    } else if (workingQuantity < targetQuantity) {
      await enqueueTrade('COVER', Math.abs(workingQuantity - targetQuantity));
    }
  }

  if (generatedTrades.length === 0) {
    return [];
  }

  return repository.appendTrades(generatedTrades);
}

async function getTransactions() {
  const baseCurrency = getBaseCurrency();
  const trades = await getLedgerTrades();
  const enrichedTrades = await enrichTransactions([...trades].reverse(), baseCurrency);
  return {
    transactions: enrichedTrades,
    summary: {
      count: enrichedTrades.length,
      base_currency: baseCurrency,
    },
  };
}

async function addHolding(input) {
  const payload = input || {};
  const legacyHolding = buildHolding({
    ticker: await normalizePortfolioTicker(payload.ticker),
    shares: payload.shares,
    buy_price: payload.buy_price,
    display_name: payload.display_name,
  });
  const createdTrade = await createTrade({
    ticker: legacyHolding.ticker,
    quantity: legacyHolding.shares,
    price: legacyHolding.buy_price,
    side: 'BUY',
    display_name: legacyHolding.display_name,
    source: 'legacy_add',
    note: 'Legacy add-holding compatibility path',
  });

  return {
    id: createdTrade.id,
    ticker: createdTrade.ticker,
    shares: createdTrade.quantity,
    buy_price: createdTrade.price,
    added_at: createdTrade.occurred_at,
  };
}

async function getHoldings() {
  return buildPortfolioSnapshot();
}

async function deleteHolding(id) {
  if (typeof id !== 'string' || !id.trim()) {
    throw createHttpError(400, 'id is required');
  }

  throw createHttpError(
    410,
    'Direct holding deletion is no longer supported. Use the position adjustment flow instead.'
  );
}

async function calculateMetrics() {
  return getHoldings();
}

module.exports = {
  __clearCachedSymbolPrices: clearCachedSymbolPrices,
  addHolding,
  adjustPosition,
  calculateMetrics,
  createTrade,
  deleteHolding,
  generatePortfolioRecommendations,
  getPortfolioAdvisor,
  getPortfolioHistory,
  getPortfolioInsights,
  getHoldings,
  getTransactions,
  normalizeHistoryDays,
};
