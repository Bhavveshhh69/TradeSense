const request = require('supertest');

jest.mock('../../portfolio/portfolio.service', () => ({
  addHolding: jest.fn(),
  adjustPosition: jest.fn(),
  createTrade: jest.fn(),
  getPortfolioAdvisor: jest.fn(),
  getPortfolioInsights: jest.fn(),
  getPortfolioHistory: jest.fn(),
  getHoldings: jest.fn(),
  getTransactions: jest.fn(),
  deleteHolding: jest.fn(),
}));

const app = require('../index');
const portfolioService = require('../../portfolio/portfolio.service');

beforeEach(() => {
  jest.clearAllMocks();
});

test('POST /api/portfolio/add stores and returns holding', async () => {
  const item = {
    id: 'abc-123',
    ticker: 'AAPL',
    shares: 5,
    buy_price: 150.25,
    added_at: '2026-02-21T00:00:00.000Z',
  };
  portfolioService.addHolding.mockResolvedValue(item);

  const response = await request(app).post('/api/portfolio/add').send({
    ticker: 'aapl',
    shares: 5,
    buy_price: 150.25,
  });

  expect(response.status).toBe(201);
  expect(response.body).toEqual({
    success: true,
    item,
  });
  expect(portfolioService.addHolding).toHaveBeenCalledWith({
    ticker: 'aapl',
    shares: 5,
    buy_price: 150.25,
  });
});

test('GET /api/portfolio returns holdings and summary', async () => {
  const payload = {
    holdings: [],
    summary: {
      total_portfolio_value: 0,
      total_invested_value: 0,
      total_profit_loss: 0,
      total_profit_loss_percent: 0,
    },
  };
  portfolioService.getHoldings.mockResolvedValue(payload);

  const response = await request(app).get('/api/portfolio');

  expect(response.status).toBe(200);
  expect(response.body).toEqual(payload);
});

test('POST /api/portfolio/trades stores a ledger trade and returns portfolio snapshot', async () => {
  const trade = {
    id: 'trade-1',
    ticker: 'AAPL',
    side: 'BUY',
    quantity: 2,
    price: 150,
  };
  const portfolio = {
    holdings: [{ ticker: 'AAPL', quantity: 2, side: 'LONG' }],
    summary: { total_portfolio_value: 24000 },
  };
  portfolioService.createTrade.mockResolvedValue(trade);
  portfolioService.getHoldings.mockResolvedValue(portfolio);

  const response = await request(app).post('/api/portfolio/trades').send({
    ticker: 'AAPL',
    side: 'BUY',
    quantity: 2,
    price: 150,
  });

  expect(response.status).toBe(201);
  expect(response.body).toEqual({
    success: true,
    trade,
    portfolio,
  });
  expect(portfolioService.createTrade).toHaveBeenCalledWith({
    ticker: 'AAPL',
    side: 'BUY',
    quantity: 2,
    price: 150,
  });
});

test('GET /api/portfolio/transactions returns ledger history', async () => {
  const payload = {
    transactions: [{ id: 'trade-1', ticker: 'AAPL', signed_quantity: 2 }],
    summary: { count: 1, base_currency: 'INR' },
  };
  portfolioService.getTransactions.mockResolvedValue(payload);

  const response = await request(app).get('/api/portfolio/transactions');

  expect(response.status).toBe(200);
  expect(response.body).toEqual(payload);
});

test('POST /api/portfolio/positions/:symbol/adjust creates explicit adjustment trades', async () => {
  const trades = [
    { id: 'sell-1', ticker: 'AAPL', side: 'SELL', quantity: 5 },
    { id: 'short-1', ticker: 'AAPL', side: 'SHORT', quantity: 2 },
  ];
  const portfolio = {
    holdings: [{ ticker: 'AAPL', quantity: 2, side: 'SHORT' }],
    summary: { total_portfolio_value: 19200 },
  };
  portfolioService.adjustPosition.mockResolvedValue(trades);
  portfolioService.getHoldings.mockResolvedValue(portfolio);

  const response = await request(app)
    .post('/api/portfolio/positions/AAPL/adjust')
    .send({ target_quantity: -2, price: 120 });

  expect(response.status).toBe(201);
  expect(response.body).toEqual({
    success: true,
    trades,
    portfolio,
  });
  expect(portfolioService.adjustPosition).toHaveBeenCalledWith('AAPL', {
    target_quantity: -2,
    price: 120,
  });
});

test('GET /api/portfolio/history returns equity curve payload', async () => {
  const payload = {
    symbol_count: 2,
    days: 30,
    equity_curve: [
      { date: '2026-03-01', portfolio_value: 124000.5 },
      { date: '2026-03-02', portfolio_value: 124532.1 },
    ],
  };
  portfolioService.getPortfolioHistory.mockResolvedValue(payload);

  const response = await request(app).get('/api/portfolio/history?days=30');

  expect(response.status).toBe(200);
  expect(response.body).toEqual(payload);
  expect(portfolioService.getPortfolioHistory).toHaveBeenCalledWith('30');
});

test('GET /api/portfolio/insights returns insights payload', async () => {
  const payload = {
    concentration_risk: 'HIGH',
    largest_position: { ticker: 'RELIANCE.NS', weight: 0.62, current_value: 62000 },
    best_performer: { ticker: 'INFY.NS', profit_loss_percent: 11.2 },
    worst_performer: { ticker: 'TCS.NS', profit_loss_percent: -4.1 },
    diversification_score: 1.78,
    volatility_level: 'MODERATE',
    insights: ['Portfolio is highly concentrated in RELIANCE.NS.'],
  };
  portfolioService.getPortfolioInsights.mockResolvedValue(payload);

  const response = await request(app).get('/api/portfolio/insights?days=45');

  expect(response.status).toBe(200);
  expect(response.body).toEqual(payload);
  expect(portfolioService.getPortfolioInsights).toHaveBeenCalledWith('45');
});

test('GET /api/portfolio/advisor returns recommendations payload', async () => {
  const payload = {
    recommendations: [
      'Reduce RELIANCE exposure',
      'Add more assets to diversify portfolio',
      'Consider rebalancing high-risk positions',
    ],
  };
  portfolioService.getPortfolioAdvisor.mockResolvedValue(payload);

  const response = await request(app).get('/api/portfolio/advisor?days=30');

  expect(response.status).toBe(200);
  expect(response.body).toEqual(payload);
  expect(portfolioService.getPortfolioAdvisor).toHaveBeenCalledWith('30');
});

test('DELETE /api/portfolio/:id removes holding', async () => {
  portfolioService.deleteHolding.mockResolvedValue(undefined);

  const response = await request(app).delete('/api/portfolio/abc-123');

  expect(response.status).toBe(200);
  expect(response.body).toEqual({ success: true });
  expect(portfolioService.deleteHolding).toHaveBeenCalledWith('abc-123');
});

test('DELETE /api/portfolio/:id returns service error', async () => {
  const error = new Error('Holding not found');
  error.status = 404;
  portfolioService.deleteHolding.mockRejectedValue(error);

  const response = await request(app).delete('/api/portfolio/unknown');

  expect(response.status).toBe(404);
  expect(response.body).toEqual({ error: 'Holding not found' });
});
