const request = require('supertest');

jest.mock('axios');
jest.mock('../../symbols/symbols.service', () => ({
  normalizeSymbol: jest.fn(),
  resolveInstrument: jest.fn(),
  searchSymbols: jest.fn(),
}));
jest.mock('../../services/ai_explainer.service', () => ({
  FALLBACK_EXPLANATION: 'Fallback explanation',
  generateNarratives: jest.fn(async () => ({
    explanation: 'Generated explanation',
    marketInsight: 'Generated market insight',
    explanationIsFallback: false,
  })),
}));
jest.mock('../services/recent_analysis.service', () => ({
  listRecentAnalyses: jest.fn(),
  recordAnalysis: jest.fn(),
}));

const axios = require('axios');
const app = require('../index');
const cache = require('../cache/memoryCache');
const symbolsService = require('../../symbols/symbols.service');
const aiExplainer = require('../../services/ai_explainer.service');
const recentAnalysisService = require('../services/recent_analysis.service');
const actualSymbolsService = jest.requireActual('../../symbols/symbols.service');

function mockPredictPayload(overrides = {}) {
  return {
    symbol: 'AAPL',
    market: 'US',
    exchange: 'NASDAQ',
    timeframe: '15m',
    strategy_family: 'orb_vwap_continuation',
    prediction: 1,
    probability: 0.6384,
    confidence: 0.0884,
    decision: 'LONG',
    decision_reason_type: null,
    actionability_state: 'actionable',
    confidence_level: 'moderate',
    strength: 0.0884,
    context: {
      trend_summary: 'Intraday setup is evaluated against opening-range direction and session VWAP alignment.',
      risk_summary: 'Quality gates and bracket sizing are session-aware and market-aware.',
    },
    model_version: 'intraday-xgboost',
    model_name: 'xgboost',
    model_threshold: 0.52,
    model_bench_summary: {
      xgboost: {
        validation: { net_expectancy: 0.21 },
        holdout: { net_expectancy: 0.17 },
        threshold: 0.52,
      },
    },
    timestamp: '2026-04-15T14:15:00+00:00',
    generated_at: '2026-04-15T14:16:00+00:00',
    setup_side: 'LONG',
    entry_price: 194.25,
    stop_price: 192.75,
    take_profit_price: 196.5,
    forced_exit_time: '2026-04-15T19:45:00+00:00',
    no_trade_reason: null,
    promotion_gate: {
      passed: true,
      reason: 'Promotion gate passed.',
      market: 'US',
      artifact_timestamp: '2026-04-15T14:00:00+00:00',
    },
    data_quality: {
      missing_bar_count: 0,
      expected_bar_count: 25,
      completeness_score: 1,
      stale_data: false,
      timezone_valid: true,
      session_valid: true,
      usable_for_live: true,
      usable_for_backtest: true,
      warnings: [],
    },
    summary: 'Long intraday setup detected.',
    market_context: {
      market: 'US',
      exchange: 'NASDAQ',
      session_window: { start: '10:00', end: '11:00', opening_range_bars: 2 },
    },
    key_drivers: ['breakout_strength', 'vwap_distance'],
    risk_notes: [],
    model_honesty: 'The probability estimates a same-session bracket outcome.',
    current_price: 194.25,
    trade_window: { start: '10:00', end: '11:00', opening_range_bars: 2 },
    threshold: 0.55,
    base_threshold: 0.55,
    effective_threshold: 0.52,
    threshold_adjustment_reason: 'Supportive sentiment slightly lowered the long-entry threshold.',
    threshold_gap: 0.1184,
    stock_sentiment_score: 0.31,
    sector_sentiment_score: 0.12,
    contextual_sentiment_score: 0.253,
    sentiment_confidence: 0.73,
    sentiment_gate_reason: 'Company and Technology news are supportive.',
    stock_article_count: 2,
    sector_article_count: 3,
    ...overrides,
  };
}

beforeEach(() => {
  cache.clear();
  axios.post.mockReset();
  axios.get.mockReset();
  symbolsService.normalizeSymbol.mockReset();
  symbolsService.resolveInstrument.mockReset();
  symbolsService.normalizeSymbol.mockImplementation(async (ticker) => ticker);
  symbolsService.resolveInstrument.mockImplementation(async (ticker) => {
    const normalized = await symbolsService.normalizeSymbol(ticker);
    return {
      symbol: normalized.replace(/\.(NS|BO)$/i, ''),
      normalized,
      display_name: normalized,
      market: normalized.endsWith('.NS') ? 'IN' : 'US',
      exchange: normalized.endsWith('.NS') ? 'NSE' : 'US',
      instrument_type: 'Equity',
    };
  });
  aiExplainer.generateNarratives.mockReset();
  aiExplainer.generateNarratives.mockResolvedValue({
    explanation: 'Generated explanation',
    marketInsight: 'Generated market insight',
    explanationIsFallback: false,
  });
  recentAnalysisService.listRecentAnalyses.mockReset();
  recentAnalysisService.recordAnalysis.mockReset();
  recentAnalysisService.recordAnalysis.mockResolvedValue(undefined);
});

test.each([
  ['RELIANCE', 'RELIANCE.NS'],
  ['RELIANCE.NS', 'RELIANCE.NS'],
  ['AAPL', 'AAPL'],
])('POST /api/analyze normalizes %s to %s and preserves Python decision output', async (inputSymbol, normalizedSymbol) => {
  symbolsService.normalizeSymbol.mockImplementation(actualSymbolsService.normalizeSymbol);
  axios.get.mockResolvedValue({
    data: {
      symbol: normalizedSymbol,
      market: normalizedSymbol.endsWith('.NS') ? 'IN' : 'US',
      timeframe: '15m',
      price: 1419.4,
    },
  });
  axios.post.mockResolvedValue({
    data: mockPredictPayload({
      symbol: normalizedSymbol,
      market: normalizedSymbol.endsWith('.NS') ? 'IN' : 'US',
    }),
  });

  const response = await request(app).post('/api/analyze').send({ symbol: inputSymbol });

  expect(response.status).toBe(200);
  expect(response.body).toMatchObject({
    normalized: normalizedSymbol,
    symbol: normalizedSymbol,
    current_price: 1419.4,
    decision_label: 'Long',
    signal: 'LONG',
    trade_actionable: true,
    actionability_state: 'actionable',
    confidence_level: 'Moderate',
    setup_side: 'LONG',
    timeframe: '15m',
    model_name: 'xgboost',
    model_threshold: 0.52,
    contextual_sentiment_score: 0.253,
  });
  expect(response.body.signal_explanation).toContain('Long intraday setup');
  expect(axios.get).toHaveBeenCalledWith(
    `http://localhost:8000/market/latest-price/${encodeURIComponent(normalizedSymbol)}`,
    expect.objectContaining({ timeout: expect.any(Number) })
  );
  expect(axios.post).toHaveBeenCalledWith(
    'http://localhost:8000/predict',
    { symbol: normalizedSymbol },
    expect.objectContaining({ timeout: expect.any(Number) })
  );
});

test('POST /api/analyze reuses cached response for normalized symbols', async () => {
  symbolsService.normalizeSymbol.mockImplementation(actualSymbolsService.normalizeSymbol);
  axios.get.mockResolvedValue({
    data: {
      symbol: 'RELIANCE.NS',
      market: 'IN',
      timeframe: '15m',
      price: 2500.35,
    },
  });
  axios.post.mockResolvedValue({
    data: mockPredictPayload({ symbol: 'RELIANCE.NS', market: 'IN' }),
  });

  const first = await request(app).post('/api/analyze').send({ symbol: 'RELIANCE' });
  const second = await request(app).post('/api/analyze').send({ symbol: 'RELIANCE.NS' });

  expect(first.status).toBe(200);
  expect(second.status).toBe(200);
  expect(second.body).toEqual(first.body);
  expect(axios.get).toHaveBeenCalledTimes(1);
  expect(axios.post).toHaveBeenCalledTimes(1);
});

test('POST /api/analyze keeps price null when latest price fetch fails', async () => {
  symbolsService.normalizeSymbol.mockResolvedValue('AAPL');
  axios.get.mockRejectedValue({
    response: {
      data: {
        detail: {
          error: 'Price unavailable for symbol',
        },
      },
    },
  });
  axios.post.mockResolvedValue({
    data: mockPredictPayload({ symbol: 'AAPL' }),
  });

  const response = await request(app).post('/api/analyze').send({ symbol: 'aapl' });

  expect(response.status).toBe(200);
  expect(response.body).toMatchObject({
    symbol: 'AAPL',
    current_price: null,
    price_error: true,
    signal: 'LONG',
    trade_actionable: true,
  });
  expect(response.body.price_error_message).toContain('Price unavailable for symbol (AAPL)');
});

test('POST /api/analyze returns strict resolution errors for unsupported symbols', async () => {
  symbolsService.resolveInstrument.mockRejectedValueOnce(
    Object.assign(new Error('symbol UNKNOWN is unsupported in the current market master'), {
      status: 404,
    })
  );

  const response = await request(app).post('/api/analyze').send({ symbol: 'unknown' });

  expect(response.status).toBe(404);
  expect(response.body).toEqual({
    error: 'symbol UNKNOWN is unsupported in the current market master',
    matches: undefined,
  });
});

test('POST /api/analyze returns no-trade output without Node-side remapping', async () => {
  symbolsService.normalizeSymbol.mockResolvedValue('AAPL');
  axios.get.mockResolvedValue({
    data: {
      symbol: 'AAPL',
      market: 'US',
      timeframe: '15m',
      price: 193.75,
    },
  });
  axios.post.mockResolvedValue({
    data: mockPredictPayload({
      decision: 'NO_TRADE',
      decision_reason_type: 'hard_blocker',
      actionability_state: 'blocked',
      probability: 0,
      confidence_level: 'low',
      setup_side: null,
      entry_price: null,
      stop_price: null,
      take_profit_price: null,
      no_trade_reason: 'Price has not broken the opening range',
      summary: 'No intraday trade is being taken.',
    }),
  });

  const response = await request(app).post('/api/analyze').send({ symbol: 'AAPL' });

  expect(response.status).toBe(200);
  expect(response.body).toMatchObject({
    signal: 'NO_TRADE',
    decision_label: 'No Trade',
    trade_actionable: false,
    signal_explanation: 'No intraday trade is being taken.',
    no_trade_reason: 'Price has not broken the opening range',
  });
});

test('POST /api/analyze preserves watchlist semantics from Python', async () => {
  symbolsService.normalizeSymbol.mockResolvedValue('AAPL');
  axios.get.mockResolvedValue({
    data: {
      symbol: 'AAPL',
      market: 'US',
      timeframe: '15m',
      price: 193.75,
    },
  });
  axios.post.mockResolvedValue({
    data: mockPredictPayload({
      decision: 'WATCHLIST',
      decision_reason_type: 'threshold_miss',
      actionability_state: 'monitor',
      probability: 0.6091,
      summary: 'Watchlist only for the current US session. The long setup is valid, but the estimated win probability is 61% and remains 1% below the live threshold.',
      no_trade_reason: 'Model probability did not clear the live expectancy threshold.',
      threshold_gap: -0.0109,
    }),
  });

  const response = await request(app).post('/api/analyze').send({ symbol: 'AAPL' });

  expect(response.status).toBe(200);
  expect(response.body).toMatchObject({
    signal: 'WATCHLIST',
    decision_label: 'Watchlist',
    trade_actionable: false,
    actionability_state: 'monitor',
    decision_reason_type: 'threshold_miss',
  });
  expect(response.body.signal_explanation).toContain('Watchlist only');
});

test('POST /api/analyze returns 400 for invalid input', async () => {
  const missingSymbol = await request(app).post('/api/analyze').send({});
  const emptySymbol = await request(app).post('/api/analyze').send({ symbol: '   ' });
  const nonStringSymbol = await request(app).post('/api/analyze').send({ symbol: 123 });

  expect(missingSymbol.status).toBe(400);
  expect(emptySymbol.status).toBe(400);
  expect(nonStringSymbol.status).toBe(400);
});

test('GET /api/analyze/recent returns persisted recent analyses', async () => {
  recentAnalysisService.listRecentAnalyses.mockResolvedValue([
    {
      id: 'recent-1',
      normalized: 'AAPL',
      display_name: 'Apple Inc.',
      signal: 'LONG',
      recorded_at: '2026-04-15T14:16:00+00:00',
    },
  ]);

  const response = await request(app).get('/api/analyze/recent?limit=5');

  expect(response.status).toBe(200);
  expect(response.body).toEqual({
    results: [
      {
        id: 'recent-1',
        normalized: 'AAPL',
        display_name: 'Apple Inc.',
        signal: 'LONG',
        recorded_at: '2026-04-15T14:16:00+00:00',
      },
    ],
  });
  expect(recentAnalysisService.listRecentAnalyses).toHaveBeenCalledWith('5');
});

test('POST /api/analyze/validate returns a flattened validation report', async () => {
  symbolsService.resolveInstrument.mockResolvedValueOnce({
    id: 'US:AAPL',
    symbol: 'AAPL',
    normalized: 'AAPL',
    display_name: 'Apple Inc.',
    market: 'US',
    exchange: 'NASDAQ',
    instrument_type: 'Equity',
    country: 'US',
  });
  axios.post.mockResolvedValueOnce({
    data: {
      symbol: 'AAPL',
      market: 'US',
      timeframe: '15m',
      period: { start_date: '2026-03-02', end_date: '2026-04-02', horizon: 1 },
      total_predictions: 243,
      accuracy: 0.5185,
      ece: 0.2026,
      brier_score: 0.2917,
      trade_metrics: { trade_count: 18, net_expectancy: 0.09, profit_factor: 1.33, wilson_lower_bound: 0.51 },
      regime_breakdown: { volatility: { normal: { sessions: 12, trade_count: 8, net_expectancy: 0.11 } } },
      cost_assumptions: { stress_cost_multiplier: 1.75, round_trip_cost_r: 0.02, stressed_round_trip_cost_r: 0.035 },
      sample_quality: { total_sessions: 30, traded_sessions: 18, skipped_sessions: 12 },
      promotion_gate: { passed: true, reason: 'Promotion gate passed.' },
      accuracy_by_confidence: { low: 0.5, moderate: 0.53 },
      reliability_curve: [{ probability_mean: 0.55, accuracy: 0.5, count: 30 }],
    },
  });

  const response = await request(app).post('/api/analyze/validate').send({ symbol: 'aapl' });

  expect(response.status).toBe(200);
  expect(response.body).toMatchObject({
    id: 'US:AAPL',
    symbol: 'AAPL',
    raw_symbol: 'AAPL',
    display_name: 'Apple Inc.',
    market: 'US',
    exchange: 'NASDAQ',
    instrument_type: 'Equity',
    country: 'US',
    timeframe: '15m',
    total_predictions: 243,
    accuracy: 0.5185,
    ece: 0.2026,
    brier_score: 0.2917,
    trade_metrics: { trade_count: 18, net_expectancy: 0.09, profit_factor: 1.33, wilson_lower_bound: 0.51 },
  });
  expect(axios.post).toHaveBeenCalledWith(
    'http://localhost:8000/analyze/validate',
    {
      symbol: 'AAPL',
      start_date: undefined,
      end_date: undefined,
      interval: undefined,
      horizon: undefined,
    },
    expect.objectContaining({ timeout: expect.any(Number) })
  );
});
