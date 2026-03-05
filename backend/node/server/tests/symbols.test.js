const request = require('supertest');

jest.mock('../../symbols/symbols.service', () => ({
  normalizeSymbol: jest.fn(),
  searchSymbols: jest.fn(),
  validateSymbol: jest.fn(),
}));

const app = require('../index');
const symbolsService = require('../../symbols/symbols.service');

beforeEach(() => {
  jest.clearAllMocks();
});

test('GET /api/symbols/search returns matching results', async () => {
  symbolsService.searchSymbols.mockResolvedValue(['RELIANCE', 'RELIANCE.NS']);

  const response = await request(app).get('/api/symbols/search?q=REL');

  expect(response.status).toBe(200);
  expect(response.body).toEqual({
    results: ['RELIANCE', 'RELIANCE.NS'],
  });
  expect(symbolsService.searchSymbols).toHaveBeenCalledWith('REL');
});

test('GET /api/symbols/normalize/:symbol returns normalized symbol', async () => {
  symbolsService.normalizeSymbol.mockResolvedValue('RELIANCE.NS');

  const response = await request(app).get('/api/symbols/normalize/reliance');

  expect(response.status).toBe(200);
  expect(response.body).toEqual({
    input: 'reliance',
    normalized: 'RELIANCE.NS',
  });
  expect(symbolsService.normalizeSymbol).toHaveBeenCalledWith('reliance');
});
