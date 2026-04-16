const request = require('supertest');

jest.mock('../services/command_center.service', () => ({
  getCommandCenter: jest.fn(),
}));

const app = require('../index');
const commandCenterService = require('../services/command_center.service');

beforeEach(() => {
  jest.clearAllMocks();
});

test('GET /api/command-center returns aggregated command-center payload', async () => {
  const payload = {
    generated_at: '2026-04-16T10:00:00.000Z',
    risk_headline: 'AAPL is carrying 42.0% of gross exposure.',
    top_portfolio_action: 'Reduce AAPL exposure',
    recent_signals: [{ normalized: 'AAPL', signal: 'LONG' }],
    daily_brief: {
      headline: 'AAPL is carrying 42.0% of gross exposure.',
      bullets: ['3 active positions are live across 2 longs and 1 shorts.'],
    },
  };
  commandCenterService.getCommandCenter.mockResolvedValue(payload);

  const response = await request(app).get('/api/command-center');

  expect(response.status).toBe(200);
  expect(response.body).toEqual(payload);
});
