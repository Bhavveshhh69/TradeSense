const request = require('supertest');

jest.mock('../../symbols/symbols.service', () => ({
  resolveInstrument: jest.fn(),
}));

jest.mock('../services/reasoning', () => ({
  callLatestPrice: jest.fn(),
  callMarketHistory: jest.fn(),
}));

const app = require('../index');
const symbolsService = require('../../symbols/symbols.service');
const reasoningService = require('../services/reasoning');

beforeEach(() => {
  jest.clearAllMocks();
  symbolsService.resolveInstrument.mockResolvedValue({
    id: 'US:AAPL',
    symbol: 'AAPL',
    normalized: 'AAPL',
    display_name: 'Apple Inc.',
    market: 'US',
    exchange: 'NASDAQ',
    instrument_type: 'Equity',
    country: 'US',
  });
});

test('GET /api/market/quote/:symbol returns a normalized quote snapshot', async () => {
  reasoningService.callLatestPrice.mockResolvedValue({
    symbol: 'AAPL',
    current_price: 200,
    price_error: false,
    price_error_message: null,
    as_of: '2026-04-16T10:00:00.000Z',
  });
  reasoningService.callMarketHistory.mockResolvedValue({
    symbol: 'AAPL',
    history: [
      { date: '2026-03-17T00:00:00.000Z', close: 180 },
      { date: '2026-04-09T00:00:00.000Z', close: 190 },
      { date: '2026-04-15T00:00:00.000Z', close: 195 },
    ],
  });

  const response = await request(app).get('/api/market/quote/aapl');

  expect(response.status).toBe(200);
  expect(response.body).toEqual({
    symbol: 'AAPL',
    display_name: 'Apple Inc.',
    market: 'US',
    exchange: 'NASDAQ',
    instrument_type: 'Equity',
    country: 'US',
    current_price: 200,
    previous_close: 190,
    day_change: 10,
    day_change_pct: expect.any(Number),
    trend_5d_pct: null,
    trend_30d_pct: expect.any(Number),
    currency: 'USD',
    as_of: '2026-04-16T10:00:00.000Z',
  });
});

test('GET /api/market/history/:symbol returns normalized market history', async () => {
  reasoningService.callMarketHistory.mockResolvedValue({
    symbol: 'AAPL',
    market: 'US',
    timeframe: '15m',
    history: [{ date: '2026-04-15T00:00:00.000Z', close: 195 }],
  });

  const response = await request(app).get('/api/market/history/aapl?days=20');

  expect(response.status).toBe(200);
  expect(response.body).toMatchObject({
    symbol: 'AAPL',
    display_name: 'Apple Inc.',
    market: 'US',
    exchange: 'NASDAQ',
    instrument_type: 'Equity',
    country: 'US',
    history: [{ date: '2026-04-15T00:00:00.000Z', close: 195 }],
  });
  expect(reasoningService.callMarketHistory).toHaveBeenCalledWith('AAPL', '20');
});
