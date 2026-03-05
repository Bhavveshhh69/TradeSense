const request = require('supertest');

jest.mock('axios');
jest.mock('../../symbols/symbols.service', () => ({
  normalizeSymbol: jest.fn(),
}));

const axios = require('axios');
const app = require('../index');
const cache = require('../cache/memoryCache');
const symbolsService = require('../../symbols/symbols.service');
const actualSymbolsService = jest.requireActual('../../symbols/symbols.service');

function mockPredictPayload(overrides = {}) {
  return {
    symbol: 'AAPL',
    prediction: 0,
    probability: 0.53,
    confidence: 0.51,
    decision: 'HOLD',
    confidence_level: 'very_low',
    strength: 0.03,
    context: {
      trend_summary: 'Market trend is mixed or transitional.',
      risk_summary: 'Market risk conditions are normal.',
    },
    model_version: 'phase14-v1',
    timestamp: '2026-02-20T15:19:04.753559+00:00',
    generated_at: '2026-02-20T15:19:04.753559+00:00',
    ...overrides,
  };
}

beforeEach(() => {
  cache.clear();
  axios.post.mockReset();
  axios.get.mockReset();
  symbolsService.normalizeSymbol.mockReset();
  symbolsService.normalizeSymbol.mockImplementation(async (ticker) => ticker);
});

test.each([
  ['RELIANCE', 'RELIANCE.NS'],
  ['RELIANCE.NS', 'RELIANCE.NS'],
  ['TCS', 'TCS.NS'],
  ['INFY', 'INFY.NS'],
  ['AAPL', 'AAPL'],
  ['NVDA', 'NVDA'],
])(
  'POST /api/analyze normalizes %s to %s and enriches with latest price',
  async (inputSymbol, normalizedSymbol) => {
    symbolsService.normalizeSymbol.mockImplementation(actualSymbolsService.normalizeSymbol);
    axios.get.mockResolvedValue({
      data: {
        symbol: normalizedSymbol,
        price: 1419.4,
      },
    });
    axios.post.mockResolvedValue({
      data: mockPredictPayload({ symbol: normalizedSymbol }),
    });

    const response = await request(app)
      .post('/api/analyze')
      .send({ symbol: inputSymbol });

    expect(response.status).toBe(200);
    expect(response.body).toMatchObject({
      symbol: normalizedSymbol,
      current_price: 1419.4,
      prediction: 0,
      probability: 0.53,
      confidence_level: 'very_low',
      trend_summary: 'Market trend is mixed or transitional.',
      risk_summary: 'Market risk conditions are normal.',
      signal: 'NEUTRAL',
      signal_direction: 'BULLISH',
      signal_strength: 'WEAK',
      market_condition: 'NEUTRAL',
      recommendation: 'WAIT',
      price_error: false,
      price_error_message: null,
    });
    expect(response.body.signal_explanation).toContain('near neutral');
    expect(axios.get).toHaveBeenCalledWith(
      `http://localhost:8000/market/latest-price/${encodeURIComponent(normalizedSymbol)}`,
      expect.objectContaining({ timeout: expect.any(Number) })
    );
    expect(axios.post).toHaveBeenCalledWith(
      'http://localhost:8000/predict',
      { symbol: normalizedSymbol },
      expect.objectContaining({ timeout: expect.any(Number) })
    );
  }
);

test('POST /api/analyze reuses cached response for symbols normalized to same value', async () => {
  symbolsService.normalizeSymbol.mockImplementation(actualSymbolsService.normalizeSymbol);
  axios.get.mockResolvedValue({
    data: {
      symbol: 'RELIANCE.NS',
      price: 2500.35,
    },
  });
  axios.post.mockResolvedValue({
    data: mockPredictPayload({ symbol: 'RELIANCE.NS' }),
  });

  const first = await request(app).post('/api/analyze').send({ symbol: 'RELIANCE' });
  const second = await request(app).post('/api/analyze').send({ symbol: 'RELIANCE.NS' });

  expect(first.status).toBe(200);
  expect(second.status).toBe(200);
  expect(second.body).toEqual(first.body);
  expect(axios.get).toHaveBeenCalledTimes(1);
  expect(axios.post).toHaveBeenCalledTimes(1);
});

test('POST /api/analyze returns prediction with null price when latest price fetch fails', async () => {
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

  const response = await request(app)
    .post('/api/analyze')
    .send({ symbol: 'aapl' });

  expect(response.status).toBe(200);
  expect(response.body).toMatchObject({
    symbol: 'AAPL',
    current_price: null,
    prediction: 0,
    probability: 0.53,
    signal: 'NEUTRAL',
    signal_direction: 'BULLISH',
    signal_strength: 'WEAK',
    recommendation: 'WAIT',
    price_error: true,
  });
  expect(response.body.price_error_message).toContain('Price unavailable for symbol (AAPL)');
  expect(axios.post).toHaveBeenCalledTimes(1);
});

test('POST /api/analyze falls back to raw symbol when normalization fails', async () => {
  const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
  symbolsService.normalizeSymbol.mockRejectedValue(new Error('normalization unavailable'));
  axios.get.mockResolvedValue({
    data: {
      symbol: 'AAPL',
      price: 123.45,
    },
  });
  axios.post.mockResolvedValue({
    data: mockPredictPayload({ symbol: 'AAPL' }),
  });

  const response = await request(app)
    .post('/api/analyze')
    .send({ symbol: 'aapl' });

  expect(response.status).toBe(200);
  expect(response.body.symbol).toBe('AAPL');
  expect(axios.get).toHaveBeenCalledWith(
    'http://localhost:8000/market/latest-price/AAPL',
    expect.objectContaining({ timeout: expect.any(Number) })
  );
  expect(axios.post).toHaveBeenCalledWith(
    'http://localhost:8000/predict',
    { symbol: 'AAPL' },
    expect.objectContaining({ timeout: expect.any(Number) })
  );
  expect(warnSpy).toHaveBeenCalledWith(
    '[analyze] symbol normalization failed for AAPL: normalization unavailable'
  );

  warnSpy.mockRestore();
});

test('POST /api/analyze returns 400 for invalid input', async () => {
  const missingSymbol = await request(app).post('/api/analyze').send({});
  const emptySymbol = await request(app).post('/api/analyze').send({ symbol: '   ' });
  const nonStringSymbol = await request(app).post('/api/analyze').send({ symbol: 123 });

  expect(missingSymbol.status).toBe(400);
  expect(emptySymbol.status).toBe(400);
  expect(nonStringSymbol.status).toBe(400);
  expect(axios.post).not.toHaveBeenCalled();
  expect(axios.get).not.toHaveBeenCalled();
});

test('POST /api/analyze returns 400 for malformed JSON body', async () => {
  const response = await request(app)
    .post('/api/analyze')
    .set('Content-Type', 'application/json')
    .send('{"symbol":');

  expect(response.status).toBe(400);
  expect(response.body).toEqual({ error: 'Invalid JSON body' });
  expect(axios.post).not.toHaveBeenCalled();
  expect(axios.get).not.toHaveBeenCalled();
});

test('POST /api/analyze maps probability 0.53 to NEUTRAL/WEAK and WAIT recommendation', async () => {
  symbolsService.normalizeSymbol.mockResolvedValue('AAPL');
  axios.get.mockResolvedValue({
    data: {
      symbol: 'AAPL',
      price: 263.75,
    },
  });
  axios.post.mockResolvedValue({
    data: mockPredictPayload({
      symbol: 'AAPL',
      probability: 0.53,
      context: {
        trend_summary: 'Market trend is mixed or transitional.',
        risk_summary: 'Market risk conditions are normal.',
      },
    }),
  });

  const response = await request(app).post('/api/analyze').send({ symbol: 'AAPL' });

  expect(response.status).toBe(200);
  expect(response.body).toMatchObject({
    probability: 0.53,
    signal: 'NEUTRAL',
    signal_direction: 'BULLISH',
    signal_strength: 'WEAK',
    market_condition: 'NEUTRAL',
    recommendation: 'WAIT',
  });
});

test('POST /api/analyze maps probability 0.62 with bullish trend to BUY_BIAS', async () => {
  symbolsService.normalizeSymbol.mockResolvedValue('AAPL');
  axios.get.mockResolvedValue({
    data: {
      symbol: 'AAPL',
      price: 263.75,
    },
  });
  axios.post.mockResolvedValue({
    data: mockPredictPayload({
      symbol: 'AAPL',
      probability: 0.62,
      decision: 'BUY',
      confidence_level: 'medium',
      context: {
        trend_summary: 'Short-term uptrend remains intact with higher highs.',
        risk_summary: 'Market risk conditions are normal.',
      },
    }),
  });

  const response = await request(app).post('/api/analyze').send({ symbol: 'AAPL' });

  expect(response.status).toBe(200);
  expect(response.body).toMatchObject({
    probability: 0.62,
    signal: 'BUY',
    signal_direction: 'BULLISH',
    signal_strength: 'MODERATE',
    market_condition: 'BULLISH',
    recommendation: 'BUY_BIAS',
  });
});

test('POST /api/analyze maps probability 0.30 to strong bearish signal and SELL recommendation', async () => {
  symbolsService.normalizeSymbol.mockResolvedValue('AAPL');
  axios.get.mockResolvedValue({
    data: {
      symbol: 'AAPL',
      price: 263.75,
    },
  });
  axios.post.mockResolvedValue({
    data: mockPredictPayload({
      symbol: 'AAPL',
      probability: 0.30,
      decision: 'SELL',
      confidence_level: 'high',
      context: {
        trend_summary: 'Persistent downtrend with lower highs and lower lows.',
        risk_summary: 'Market risk conditions are elevated.',
      },
    }),
  });

  const response = await request(app).post('/api/analyze').send({ symbol: 'AAPL' });

  expect(response.status).toBe(200);
  expect(response.body).toMatchObject({
    probability: 0.30,
    signal: 'STRONG_SELL',
    signal_direction: 'BEARISH',
    signal_strength: 'STRONG',
    market_condition: 'BEARISH',
    recommendation: 'SELL',
  });
});

test('POST /api/analyze maps probability 0.35 to SELL band with moderate bearish strength', async () => {
  symbolsService.normalizeSymbol.mockResolvedValue('AAPL');
  axios.get.mockResolvedValue({
    data: {
      symbol: 'AAPL',
      price: 263.75,
    },
  });
  axios.post.mockResolvedValue({
    data: mockPredictPayload({
      symbol: 'AAPL',
      probability: 0.35,
      decision: 'SELL',
      context: {
        trend_summary: 'Persistent downtrend with lower highs and lower lows.',
        risk_summary: 'Market risk conditions are elevated.',
      },
    }),
  });

  const response = await request(app).post('/api/analyze').send({ symbol: 'AAPL' });

  expect(response.status).toBe(200);
  expect(response.body).toMatchObject({
    probability: 0.35,
    signal: 'SELL',
    signal_direction: 'BEARISH',
    signal_strength: 'MODERATE',
    recommendation: 'SELL_BIAS',
  });
});
