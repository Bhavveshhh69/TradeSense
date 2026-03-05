const axios = require('axios');

jest.mock('axios');
jest.mock('../portfolio.repository', () => ({
  addHolding: jest.fn(),
  deleteHoldingById: jest.fn(),
  getAllHoldings: jest.fn(),
}));
jest.mock('../../symbols/symbols.service', () => ({
  normalizeSymbol: jest.fn(),
}));
jest.mock('../../services/fx.service', () => ({
  getFxRate: jest.fn(),
}));

const repository = require('../portfolio.repository');
const symbolsService = require('../../symbols/symbols.service');
const fxService = require('../../services/fx.service');
const portfolioService = require('../portfolio.service');

beforeEach(() => {
  jest.clearAllMocks();
  fxService.getFxRate.mockImplementation(async (fromCurrency, toCurrency) => {
    if (fromCurrency === 'USD' && toCurrency === 'INR') {
      return 80;
    }
    if (fromCurrency === 'INR' && toCurrency === 'USD') {
      return 0.0125;
    }
    throw new Error(`Unsupported FX pair ${fromCurrency}->${toCurrency}`);
  });
});

test('getHoldings returns null price with error flag when latest price fetch fails', async () => {
  repository.getAllHoldings.mockResolvedValue([
    {
      id: '1',
      ticker: 'AAPL',
      shares: 2,
      buy_price: 100,
      added_at: '2026-02-21T00:00:00.000Z',
    },
    {
      id: '2',
      ticker: 'INVALID',
      shares: 1,
      buy_price: 50,
      added_at: '2026-02-21T00:00:00.000Z',
    },
  ]);

  symbolsService.normalizeSymbol.mockImplementation(async (ticker) => ticker);

  axios.get.mockImplementation(async (url) => {
    if (url.includes('/AAPL')) {
      return { data: { symbol: 'AAPL', price: 150.5 } };
    }

    const error = new Error('not found');
    error.response = {
      data: {
        detail: {
          error: 'Price unavailable for symbol',
          symbol: 'INVALID',
        },
      },
    };
    throw error;
  });

  const payload = await portfolioService.getHoldings();

  expect(payload.holdings).toHaveLength(2);
  expect(payload.holdings[0]).toEqual(
    expect.objectContaining({
      ticker: 'AAPL',
      instrument_currency: 'USD',
      base_currency: 'INR',
      price_native: 150.5,
      price_base: 12040,
      current_price: 12040,
      price_error: false,
    })
  );
  expect(payload.holdings[1]).toEqual(
    expect.objectContaining({
      ticker: 'INVALID',
      current_price: null,
      current_value: null,
      profit_loss: null,
      price_error: true,
    })
  );
  expect(payload.summary).toEqual(
    expect.objectContaining({
      total_portfolio_value: 24080,
      total_invested_value: 20000,
      total_profit_loss: null,
      has_price_errors: true,
      base_currency: 'INR',
    })
  );
});

test('addHolding normalizes symbol before persistence', async () => {
  symbolsService.normalizeSymbol.mockResolvedValue('RELIANCE.NS');
  repository.addHolding.mockImplementation(async (item) => item);

  const result = await portfolioService.addHolding({
    ticker: 'reliance',
    shares: 5,
    buy_price: 100,
  });

  expect(symbolsService.normalizeSymbol).toHaveBeenCalledWith('reliance');
  expect(repository.addHolding).toHaveBeenCalledWith(
    expect.objectContaining({
      ticker: 'RELIANCE.NS',
      shares: 5,
      buy_price: 100,
    })
  );
  expect(result).toEqual(
    expect.objectContaining({
      ticker: 'RELIANCE.NS',
    })
  );
});

test('getPortfolioHistory returns calendar curve with carry-forward pricing', async () => {
  jest.useFakeTimers().setSystemTime(new Date('2026-03-04T12:00:00.000Z'));

  try {
    repository.getAllHoldings.mockResolvedValue([
      {
        id: '1',
        ticker: 'AAPL',
        shares: 2,
        buy_price: 100,
        added_at: '2026-02-21T00:00:00.000Z',
      },
      {
        id: '2',
        ticker: 'RELIANCE',
        shares: 3,
        buy_price: 200,
        added_at: '2026-02-21T00:00:00.000Z',
      },
    ]);

    symbolsService.normalizeSymbol.mockImplementation(async (ticker) =>
      ticker === 'RELIANCE' ? 'RELIANCE.NS' : ticker
    );

    axios.get.mockImplementation(async (url) => {
      if (url.includes('/market/history/AAPL')) {
        return {
          data: {
            symbol: 'AAPL',
            history: [
              { date: '2026-03-01', close: 100 },
              { date: '2026-03-03', close: 120 },
            ],
          },
        };
      }

      if (url.includes('/market/history/RELIANCE.NS')) {
        return {
          data: {
            symbol: 'RELIANCE.NS',
            history: [
              { date: '2026-03-02', close: 200 },
              { date: '2026-03-03', close: 210 },
            ],
          },
        };
      }

      throw new Error('Unexpected URL');
    });

    const result = await portfolioService.getPortfolioHistory(3);

    expect(result.symbol_count).toBe(2);
    expect(result.days).toBe(3);
    expect(result.equity_curve).toEqual([
      { date: '2026-03-02', portfolio_value: 16600 },
      { date: '2026-03-03', portfolio_value: 19830 },
      { date: '2026-03-04', portfolio_value: 19830 },
    ]);
    expect(axios.get).toHaveBeenCalledWith(
      expect.stringContaining('/market/history/AAPL?days=3'),
      expect.any(Object)
    );
    expect(axios.get).toHaveBeenCalledWith(
      expect.stringContaining('/market/history/RELIANCE.NS?days=3'),
      expect.any(Object)
    );
  } finally {
    jest.useRealTimers();
  }
});

test('getPortfolioHistory skips failed symbols and succeeds when at least one symbol has history', async () => {
  jest.useFakeTimers().setSystemTime(new Date('2026-03-04T12:00:00.000Z'));

  try {
    repository.getAllHoldings.mockResolvedValue([
      {
        id: '1',
        ticker: 'AAPL',
        shares: 2,
        buy_price: 100,
        added_at: '2026-02-21T00:00:00.000Z',
      },
      {
        id: '2',
        ticker: 'NVDA',
        shares: 1,
        buy_price: 200,
        added_at: '2026-02-21T00:00:00.000Z',
      },
    ]);

    symbolsService.normalizeSymbol.mockImplementation(async (ticker) => ticker);

    axios.get.mockImplementation(async (url) => {
      if (url.includes('/market/history/AAPL')) {
        return {
          data: {
            symbol: 'AAPL',
            history: [
              { date: '2026-03-02', close: 100 },
              { date: '2026-03-03', close: 120 },
            ],
          },
        };
      }

      const error = new Error('not found');
      error.response = {
        data: {
          detail: {
            error: 'Price history unavailable for symbol',
            symbol: 'NVDA',
          },
        },
      };
      throw error;
    });

    const result = await portfolioService.getPortfolioHistory(3);

    expect(result.days).toBe(3);
    expect(result.equity_curve).toEqual([
      { date: '2026-03-02', portfolio_value: 16000 },
      { date: '2026-03-03', portfolio_value: 19200 },
      { date: '2026-03-04', portfolio_value: 19200 },
    ]);
  } finally {
    jest.useRealTimers();
  }
});

test('getPortfolioHistory caps days to 90 and returns empty-portfolio baseline', async () => {
  repository.getAllHoldings.mockResolvedValue([]);

  const result = await portfolioService.getPortfolioHistory(120);

  expect(result.symbol_count).toBe(0);
  expect(result.days).toBe(90);
  expect(result.equity_curve).toHaveLength(90);
  expect(result.equity_curve.every((point) => point.portfolio_value === 0)).toBe(true);
  expect(axios.get).not.toHaveBeenCalled();
});

test('getPortfolioInsights computes concentration, diversification, and volatility insights', async () => {
  jest.useFakeTimers().setSystemTime(new Date('2026-03-04T12:00:00.000Z'));

  try {
    repository.getAllHoldings.mockResolvedValue([
      {
        id: '1',
        ticker: 'RELIANCE.NS',
        shares: 10,
        buy_price: 100,
        added_at: '2026-02-21T00:00:00.000Z',
      },
      {
        id: '2',
        ticker: 'INFY.NS',
        shares: 2,
        buy_price: 300,
        added_at: '2026-02-21T00:00:00.000Z',
      },
    ]);

    symbolsService.normalizeSymbol.mockImplementation(async (ticker) => ticker);

    axios.get.mockImplementation(async (url) => {
      if (url.includes('/market/latest-price/RELIANCE.NS')) {
        return { data: { symbol: 'RELIANCE.NS', price: 500 } };
      }
      if (url.includes('/market/latest-price/INFY.NS')) {
        return { data: { symbol: 'INFY.NS', price: 250 } };
      }
      if (url.includes('/market/history/RELIANCE.NS')) {
        return {
          data: {
            symbol: 'RELIANCE.NS',
            history: [
              { date: '2026-03-02', close: 100 },
              { date: '2026-03-03', close: 110 },
              { date: '2026-03-04', close: 100 },
            ],
          },
        };
      }
      if (url.includes('/market/history/INFY.NS')) {
        return {
          data: {
            symbol: 'INFY.NS',
            history: [
              { date: '2026-03-02', close: 200 },
              { date: '2026-03-03', close: 200 },
              { date: '2026-03-04', close: 200 },
            ],
          },
        };
      }

      throw new Error('Unexpected URL');
    });

    const result = await portfolioService.getPortfolioInsights(3);

    expect(result).toEqual(
      expect.objectContaining({
        concentration_risk: 'HIGH',
        largest_position: {
          ticker: 'RELIANCE.NS',
          weight: 90.91,
          current_value: 5000,
        },
        best_performer: {
          ticker: 'RELIANCE.NS',
          profit_loss_percent: 400,
          weight: 90.91,
        },
        worst_performer: {
          ticker: 'INFY.NS',
          profit_loss_percent: -16.67,
        },
        diversification_score: 1.2,
        volatility_level: 'HIGH',
      })
    );
    expect(result.insights).toEqual(
      expect.arrayContaining([
        'Portfolio is highly concentrated in RELIANCE.NS.',
        'Diversification risk is high.',
        'Consider reducing exposure to RELIANCE.',
      ])
    );
  } finally {
    jest.useRealTimers();
  }
});

test('getPortfolioInsights ranks performers by profit_loss_percent and computes value-based weights', async () => {
  jest.useFakeTimers().setSystemTime(new Date('2026-03-04T12:00:00.000Z'));

  try {
    repository.getAllHoldings.mockResolvedValue([
      {
        id: 'aapl-1',
        ticker: 'AAPL',
        shares: 1,
        buy_price: 260,
        added_at: '2026-02-21T00:00:00.000Z',
      },
      {
        id: 'rel-1',
        ticker: 'RELIANCE.NS',
        shares: 2,
        buy_price: 650,
        added_at: '2026-02-21T00:00:00.000Z',
      },
    ]);

    symbolsService.normalizeSymbol.mockImplementation(async (ticker) => ticker);

    axios.get.mockImplementation(async (url) => {
      if (url.includes('/market/latest-price/AAPL')) {
        return { data: { symbol: 'AAPL', price: 263.75 } };
      }
      if (url.includes('/market/latest-price/RELIANCE.NS')) {
        return { data: { symbol: 'RELIANCE.NS', price: 1358 } };
      }
      if (url.includes('/market/history/AAPL')) {
        return {
          data: {
            symbol: 'AAPL',
            history: [
              { date: '2026-03-02', close: 260 },
              { date: '2026-03-03', close: 262 },
              { date: '2026-03-04', close: 263.75 },
            ],
          },
        };
      }
      if (url.includes('/market/history/RELIANCE.NS')) {
        return {
          data: {
            symbol: 'RELIANCE.NS',
            history: [
              { date: '2026-03-02', close: 1300 },
              { date: '2026-03-03', close: 1330 },
              { date: '2026-03-04', close: 1358 },
            ],
          },
        };
      }

      throw new Error('Unexpected URL');
    });

    const result = await portfolioService.getPortfolioInsights(3);

    expect(result.largest_position).toEqual({
      ticker: 'AAPL',
      weight: 88.6,
      current_value: 21100,
    });
    expect(result.best_performer).toEqual({
      ticker: 'RELIANCE.NS',
      profit_loss_percent: 108.92,
      weight: 11.4,
    });
    expect(result.worst_performer).toEqual({
      ticker: 'AAPL',
      profit_loss_percent: 1.44,
    });
  } finally {
    jest.useRealTimers();
  }
});

test('getPortfolioInsights returns baseline insight payload for empty portfolio', async () => {
  repository.getAllHoldings.mockResolvedValue([]);

  const result = await portfolioService.getPortfolioInsights(30);

  expect(result).toEqual(
    expect.objectContaining({
      concentration_risk: 'LOW',
      largest_position: null,
      best_performer: null,
      worst_performer: null,
      diversification_score: 0,
      volatility_level: 'LOW',
    })
  );
  expect(result.insights).toContain('No active positions available to assess concentration.');
  expect(axios.get).not.toHaveBeenCalled();
});

test('generatePortfolioRecommendations applies all risk rules', () => {
  const result = portfolioService.generatePortfolioRecommendations({
    concentration_risk: 'HIGH',
    diversification_score: 1.4,
    volatility_level: 'HIGH',
    largest_position: { ticker: 'RELIANCE.NS', weight: 68.25, current_value: 65220 },
    best_performer: { ticker: 'RELIANCE.NS', profit_loss_percent: 24.8, weight: 68.25 },
    worst_performer: { ticker: 'INFY.NS', profit_loss_percent: -5.2 },
  });

  expect(result).toEqual({
    recommendations: [
      'Reduce RELIANCE exposure',
      'Add more assets to diversify portfolio',
      'Consider rebalancing high-risk positions',
      'Book partial profits in RELIANCE',
    ],
  });
});

test('generatePortfolioRecommendations returns stable fallback when no rule triggers', () => {
  const result = portfolioService.generatePortfolioRecommendations({
    concentration_risk: 'LOW',
    diversification_score: 4.8,
    volatility_level: 'LOW',
    largest_position: { ticker: 'ITC.NS', weight: 22.15, current_value: 14500 },
    best_performer: { ticker: 'ITC.NS', profit_loss_percent: 4.5, weight: 22.15 },
    worst_performer: { ticker: 'HDFCBANK.NS', profit_loss_percent: 1.1 },
  });

  expect(result).toEqual({
    recommendations: ['Portfolio allocation looks balanced. Continue periodic rebalancing.'],
  });
});
