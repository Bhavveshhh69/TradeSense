const request = require('supertest');

jest.mock('../../symbols/symbols.service', () => ({
  normalizeSymbol: jest.fn(),
  resolveInstrument: jest.fn(),
  searchSymbols: jest.fn(),
  validateSymbol: jest.fn(),
}));

const app = require('../index');
const symbolsService = require('../../symbols/symbols.service');

beforeEach(() => {
  jest.clearAllMocks();
});

test('GET /api/symbols/search returns matching results', async () => {
  symbolsService.searchSymbols.mockResolvedValue([
    {
      id: 'IN:RELIANCE.NS',
      symbol: 'RELIANCE',
      normalized: 'RELIANCE.NS',
      display_name: 'Reliance Industries Ltd',
      market: 'IN',
      exchange: 'NSE',
      instrument_type: 'Equity',
      country: 'IN',
      search_terms: ['RELIANCE', 'RELIANCE.NS', 'RELIANCE INDUSTRIES LTD'],
      group_label: 'India',
    },
  ]);

  const response = await request(app).get('/api/symbols/search?q=REL&market=IN&kind=equity&limit=10');

  expect(response.status).toBe(200);
  expect(response.body).toEqual({
    results: [
      {
        id: 'IN:RELIANCE.NS',
        symbol: 'RELIANCE',
        normalized: 'RELIANCE.NS',
        display_name: 'Reliance Industries Ltd',
        market: 'IN',
        exchange: 'NSE',
        instrument_type: 'Equity',
        country: 'IN',
        search_terms: ['RELIANCE', 'RELIANCE.NS', 'RELIANCE INDUSTRIES LTD'],
        group_label: 'India',
      },
    ],
    query: 'REL',
    market: 'IN',
    kind: 'equity',
    limit: 1,
  });
  expect(symbolsService.searchSymbols).toHaveBeenCalledWith({
    query: 'REL',
    market: 'IN',
    kind: 'equity',
    limit: '10',
  });
});

test('GET /api/symbols/normalize/:symbol returns normalized symbol', async () => {
  symbolsService.resolveInstrument.mockResolvedValue({
    id: 'IN:RELIANCE.NS',
    symbol: 'RELIANCE',
    normalized: 'RELIANCE.NS',
    changed: true,
    display_name: 'Reliance Industries Ltd',
    market: 'IN',
    exchange: 'NSE',
    instrument_type: 'Equity',
    country: 'IN',
    search_terms: ['RELIANCE', 'RELIANCE.NS', 'RELIANCE INDUSTRIES LTD'],
  });

  const response = await request(app).get('/api/symbols/normalize/reliance');

  expect(response.status).toBe(200);
  expect(response.body).toEqual({
    id: 'IN:RELIANCE.NS',
    input: 'reliance',
    normalized: 'RELIANCE.NS',
    changed: true,
    symbol: 'RELIANCE',
    display_name: 'Reliance Industries Ltd',
    market: 'IN',
    exchange: 'NSE',
    instrument_type: 'Equity',
    country: 'IN',
    search_terms: ['RELIANCE', 'RELIANCE.NS', 'RELIANCE INDUSTRIES LTD'],
  });
  expect(symbolsService.resolveInstrument).toHaveBeenCalledWith('reliance');
});
