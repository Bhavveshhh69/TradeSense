const axios = require('axios');

const repository = require('./portfolio.repository');
const {
  buildHolding,
  createHttpError,
  normalizeTicker,
  resolveInstrumentMetadata,
} = require('./portfolio.model');
const symbolsService = require('../symbols/symbols.service');
const fxService = require('../services/fx.service');

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

async function normalizeHoldingTicker(rawTicker) {
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
  const end = new Date(Date.UTC(todayUtc.getUTCFullYear(), todayUtc.getUTCMonth(), todayUtc.getUTCDate()));
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

function groupHoldingsBySymbol(holdings) {
  const grouped = new Map();

  for (const holding of holdings) {
    const key = holding.ticker;
    const shares = Number(holding.shares);
    if (!Number.isFinite(shares) || shares <= 0) {
      continue;
    }

    const previousShares = grouped.get(key) || 0;
    grouped.set(key, previousShares + shares);
  }

  return grouped;
}

function buildEquityCurve(dates, symbolSharesMap, symbolHistoryMap, symbolFxRateMap) {
  const symbolStates = new Map();

  for (const [symbol, prices] of symbolHistoryMap.entries()) {
    symbolStates.set(symbol, {
      index: 0,
      lastKnownPrice: null,
      prices,
    });
  }

  return dates.map((date) => {
    let portfolioValue = 0;

    for (const [symbol, shares] of symbolSharesMap.entries()) {
      const state = symbolStates.get(symbol);
      if (!state || !Array.isArray(state.prices) || state.prices.length === 0) {
        continue;
      }
      const fxRateToBase = Number(symbolFxRateMap.get(symbol) || 0);
      if (!Number.isFinite(fxRateToBase) || fxRateToBase <= 0) {
        continue;
      }

      while (state.index < state.prices.length && state.prices[state.index].date <= date) {
        state.lastKnownPrice = state.prices[state.index].close;
        state.index += 1;
      }

      if (typeof state.lastKnownPrice === 'number' && Number.isFinite(state.lastKnownPrice)) {
        portfolioValue += shares * state.lastKnownPrice * fxRateToBase;
      }
    }

    return {
      date,
      portfolio_value: roundMoney(portfolioValue),
    };
  });
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

async function calculateMetrics(holdings) {
  const baseCurrency = getBaseCurrency();

  if (!Array.isArray(holdings) || holdings.length === 0) {
    return {
      holdings: [],
      summary: {
        total_portfolio_value: 0,
        total_invested_value: 0,
        total_profit_loss: 0,
        total_profit_loss_percent: 0,
        base_currency: baseCurrency,
      },
    };
  }

  const normalizedHoldings = [];
  for (const holding of holdings) {
    const normalizedSymbol = await normalizeHoldingTicker(holding.ticker);
    normalizedHoldings.push({
      ...holding,
      ticker: normalizedSymbol,
    });
  }

  const latestPriceResults = await Promise.all(
    normalizedHoldings.map((holding) => fetchLatestPrice(holding.ticker))
  );

  const enrichedHoldings = [];
  for (let index = 0; index < normalizedHoldings.length; index += 1) {
    const holding = normalizedHoldings[index];
    const holdingSymbol = holding.ticker;
    const instrumentMetadata = resolveInstrumentMetadata(holdingSymbol);
    const instrumentCurrency = instrumentMetadata.instrument_currency;
    const exchange = instrumentMetadata.exchange;

    let fxRateToBase;
    let fxErrorMessage = null;

    try {
      fxRateToBase = await resolveFxRateToBase(instrumentCurrency, baseCurrency);
    } catch (error) {
      fxRateToBase = null;
      fxErrorMessage = extractFxErrorMessage(error, instrumentCurrency, baseCurrency);
    }

    const latestPrice =
      latestPriceResults[index] || {
        requested_symbol: holdingSymbol,
        response_symbol: null,
        price: null,
        error: true,
        error_message: `Unable to fetch current price (${holdingSymbol})`,
      };
    const responseSymbol =
      typeof latestPrice.response_symbol === 'string' ? latestPrice.response_symbol : null;

    console.log(
      'Price fetched:',
      responseSymbol || latestPrice.requested_symbol || holdingSymbol,
      latestPrice.price
    );

    if (responseSymbol && responseSymbol !== holdingSymbol) {
      console.warn(
        `[portfolio:price-enrichment] symbol mismatch expected=${holdingSymbol} received=${responseSymbol}`
      );
    }

    const priceResult = latestPrice;

    const priceNative =
      typeof priceResult.price === 'number' && Number.isFinite(priceResult.price)
        ? roundMoney(priceResult.price)
        : null;
    const roundedFxRateToBase =
      typeof fxRateToBase === 'number' && Number.isFinite(fxRateToBase) && fxRateToBase > 0
        ? roundMetric(fxRateToBase, 6)
        : null;
    const priceBase =
      priceNative !== null && roundedFxRateToBase !== null
        ? roundMoney(priceNative * roundedFxRateToBase)
        : null;
    const marketValueBase =
      priceBase === null ? null : roundMoney(Number(holding.shares) * priceBase);
    const investedValueNative = roundMoney(Number(holding.shares) * Number(holding.buy_price));
    const investedValue =
      roundedFxRateToBase === null
        ? null
        : roundMoney(investedValueNative * roundedFxRateToBase);
    const profitLoss =
      marketValueBase === null || investedValue === null
        ? null
        : roundMoney(marketValueBase - investedValue);
    const profitLossPercent =
      profitLoss === null || investedValue === 0
        ? null
        : roundPercent((profitLoss / investedValue) * 100);
    const priceError = Boolean(priceResult.error) || roundedFxRateToBase === null;
    const priceErrorMessage = priceResult.error
      ? priceResult.error_message
      : fxErrorMessage;

    console.log(
      `[portfolio:price-enrichment] requested_symbol=${holdingSymbol} returned_price=${String(
        latestPrice.price
      )} returned_symbol=${String(responseSymbol)} assigned_symbol=${holdingSymbol} instrument_currency=${instrumentCurrency} base_currency=${baseCurrency} fx_rate=${String(
        roundedFxRateToBase
      )}`
    );

    enrichedHoldings.push({
      id: holding.id,
      ticker: holdingSymbol,
      shares: holding.shares,
      buy_price: holding.buy_price,
      exchange,
      instrument_currency: instrumentCurrency,
      base_currency: baseCurrency,
      price_native: priceNative,
      fx_rate_to_base: roundedFxRateToBase,
      price_base: priceBase,
      market_value_base: marketValueBase,
      current_price: priceBase,
      current_value: marketValueBase,
      invested_value: investedValue,
      profit_loss: profitLoss,
      profit_loss_percent: profitLossPercent,
      price_error: priceError,
      price_error_message: priceError ? priceErrorMessage : null,
    });
  }

  let totalPortfolioValue = 0;
  let totalInvestedValue = 0;
  let hasPriceErrors = false;

  enrichedHoldings.forEach((holding) => {
    if (holding.current_value === null) {
      hasPriceErrors = true;
    } else {
      totalPortfolioValue += holding.current_value;
    }
    if (typeof holding.invested_value === 'number' && Number.isFinite(holding.invested_value)) {
      totalInvestedValue += holding.invested_value;
    } else {
      hasPriceErrors = true;
    }
  });

  const totalProfitLoss = hasPriceErrors
    ? null
    : roundMoney(totalPortfolioValue - totalInvestedValue);
  const totalProfitLossPercent =
    totalProfitLoss === null || totalInvestedValue === 0
      ? null
      : roundPercent((totalProfitLoss / totalInvestedValue) * 100);

  return {
    holdings: enrichedHoldings,
    summary: {
      total_portfolio_value: roundMoney(totalPortfolioValue),
      total_invested_value: roundMoney(totalInvestedValue),
      total_profit_loss: totalProfitLoss,
      total_profit_loss_percent: totalProfitLossPercent,
      has_price_errors: hasPriceErrors,
      base_currency: baseCurrency,
    },
  };
}

async function getPortfolioHistory(daysInput) {
  const days = normalizeHistoryDays(daysInput);
  const baseCurrency = getBaseCurrency();
  const holdings = await repository.getAllHoldings();
  const dates = buildHistoryDateRange(days);

  if (!Array.isArray(holdings) || holdings.length === 0) {
    return {
      symbol_count: 0,
      days,
      equity_curve: dates.map((date) => ({
        date,
        portfolio_value: 0,
      })),
    };
  }

  const normalizedHoldings = [];
  for (const holding of holdings) {
    const normalizedSymbol = await normalizeHoldingTicker(holding.ticker);
    normalizedHoldings.push({
      ...holding,
      ticker: normalizedSymbol,
    });
  }

  const symbolSharesMap = groupHoldingsBySymbol(normalizedHoldings);
  const symbols = [...symbolSharesMap.keys()];
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
  const fxRateBySymbol = new Map(fxRateResults.map(([symbol, fxRate, errorMessage]) => [symbol, { fxRate, errorMessage }]));
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

    if (fxResult?.errorMessage) {
      console.error(
        `[portfolio:history-fx] symbol=${symbol} error=${fxResult.errorMessage}`
      );
    }
    symbolHistoryMap.set(symbol, []);
    symbolFxRateMap.set(symbol, 0);
  }

  if (symbols.length > 0 && successfulSymbolCount === 0) {
    throw createHttpError(502, 'Unable to fetch portfolio history');
  }

  return {
    symbol_count: symbols.length,
    days,
    equity_curve: buildEquityCurve(dates, symbolSharesMap, symbolHistoryMap, symbolFxRateMap),
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

async function addHolding(input) {
  const payload = input || {};
  const normalizedSymbol = await symbolsService.normalizeSymbol(payload.ticker);
  const item = buildHolding({
    ...payload,
    ticker: normalizedSymbol,
  });
  return repository.addHolding(item);
}

async function getHoldings() {
  const holdings = await repository.getAllHoldings();
  return calculateMetrics(holdings);
}

async function deleteHolding(id) {
  if (typeof id !== 'string' || !id.trim()) {
    throw createHttpError(400, 'id is required');
  }

  const deleted = await repository.deleteHoldingById(id.trim());
  if (!deleted) {
    throw createHttpError(404, 'Holding not found');
  }
}

module.exports = {
  addHolding,
  calculateMetrics,
  deleteHolding,
  generatePortfolioRecommendations,
  getPortfolioAdvisor,
  getPortfolioHistory,
  getPortfolioInsights,
  getHoldings,
  normalizeHistoryDays,
};
