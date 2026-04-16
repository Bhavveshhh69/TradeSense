const axios = require('axios');

jest.mock('axios');
jest.mock('../portfolio.repository', () => ({
  appendTrades: jest.fn(),
  getAllHoldings: jest.fn(),
  getAllTrades: jest.fn(),
  replaceAllTrades: jest.fn(),
}));
jest.mock('../../symbols/symbols.service', () => ({
  normalizeSymbol: jest.fn(),
  resolveInstrument: jest.fn(),
}));
jest.mock('../../services/fx.service', () => ({
  getFxRate: jest.fn(),
}));

const repository = require('../portfolio.repository');
const symbolsService = require('../../symbols/symbols.service');
const fxService = require('../../services/fx.service');
const portfolioService = require('../portfolio.service');

function makeTrade(overrides = {}) {
  return {
    id: overrides.id || `${overrides.side || 'BUY'}-${overrides.ticker || 'AAPL'}-${overrides.quantity || 1}`,
    ticker: overrides.ticker || 'AAPL',
    symbol: overrides.symbol || 'AAPL',
    normalized: overrides.normalized || overrides.ticker || 'AAPL',
    display_name: overrides.display_name || overrides.ticker || 'AAPL',
    market: overrides.market || 'US',
    exchange: overrides.exchange || 'US',
    instrument_type: overrides.instrument_type || 'Equity',
    instrument_currency: overrides.instrument_currency || 'USD',
    side: overrides.side || 'BUY',
    quantity: overrides.quantity || 1,
    price: overrides.price || 100,
    note: overrides.note || null,
    source: overrides.source || 'manual',
    occurred_at: overrides.occurred_at || '2026-03-02T00:00:00.000Z',
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  portfolioService.__clearCachedSymbolPrices();
  repository.getAllTrades.mockResolvedValue([]);
  repository.getAllHoldings.mockResolvedValue([]);
  repository.appendTrades.mockImplementation(async (items) => items);
  repository.replaceAllTrades.mockResolvedValue(undefined);
  symbolsService.normalizeSymbol.mockImplementation(async (ticker) => String(ticker).toUpperCase());
  symbolsService.resolveInstrument.mockImplementation(async (ticker) => {
    const normalized = String(ticker).trim().toUpperCase();
    const isIndia = normalized.includes('RELIANCE') || normalized.endsWith('.NS');
    return {
      symbol: normalized.replace(/\.(NS|BO)$/i, ''),
      normalized: isIndia && !normalized.endsWith('.NS') ? `${normalized}.NS` : normalized,
      display_name: normalized.replace(/\.(NS|BO)$/i, ''),
      market: isIndia ? 'IN' : 'US',
      exchange: isIndia ? 'NSE' : 'US',
      instrument_type: 'Equity',
    };
  });
  fxService.getFxRate.mockImplementation(async (fromCurrency, toCurrency) => {
    if (fromCurrency === toCurrency) {
      return 1;
    }
    if (fromCurrency === 'USD' && toCurrency === 'INR') {
      return 80;
    }
    if (fromCurrency === 'INR' && toCurrency === 'USD') {
      return 0.0125;
    }
    throw new Error(`Unsupported FX pair ${fromCurrency}->${toCurrency}`);
  });
});

test('getHoldings backfills legacy holdings into synthetic ledger trades', async () => {
  repository.getAllHoldings.mockResolvedValue([
    {
      id: 'legacy-aapl',
      ticker: 'AAPL',
      shares: 2,
      buy_price: 100,
      added_at: '2026-03-01T00:00:00.000Z',
    },
  ]);

  axios.get.mockImplementation(async (url) => {
    if (url.includes('/market/latest-price/AAPL')) {
      return { data: { symbol: 'AAPL', price: 150 } };
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  const payload = await portfolioService.getHoldings();

  expect(repository.replaceAllTrades).toHaveBeenCalledWith(
    expect.arrayContaining([
      expect.objectContaining({
        ticker: 'AAPL',
        side: 'BUY',
        quantity: 2,
        price: 100,
        source: 'legacy_backfill',
      }),
    ])
  );
  expect(payload.holdings).toEqual([
    expect.objectContaining({
      ticker: 'AAPL',
      side: 'LONG',
      quantity: 2,
      avg_price: 100,
      current_value: 24000,
      unrealized_pnl: 8000,
      realized_pnl: 0,
    }),
  ]);
  expect(payload.summary).toEqual(
    expect.objectContaining({
      total_portfolio_value: 24000,
      total_invested_value: 16000,
      total_unrealized_pnl: 8000,
      total_realized_pnl: 0,
      total_profit_loss: 8000,
      long_positions: 1,
      short_positions: 0,
    })
  );
});

test('createTrade appends normalized trades for the ledger', async () => {
  const result = await portfolioService.createTrade({
    ticker: 'reliance',
    side: 'SHORT',
    quantity: 3,
    price: 2500,
  });

  expect(repository.appendTrades).toHaveBeenCalledWith([
    expect.objectContaining({
      ticker: 'RELIANCE.NS',
      symbol: 'RELIANCE',
      side: 'SHORT',
      quantity: 3,
      price: 2500,
      market: 'IN',
      exchange: 'NSE',
    }),
  ]);
  expect(result).toEqual(
    expect.objectContaining({
      ticker: 'RELIANCE.NS',
      side: 'SHORT',
      quantity: 3,
    })
  );
});

test('adjustPosition creates explicit trades instead of mutating positions', async () => {
  repository.getAllTrades.mockResolvedValue([
    makeTrade({
      id: 'buy-aapl',
      ticker: 'AAPL',
      side: 'BUY',
      quantity: 5,
      price: 100,
      occurred_at: '2026-03-01T00:00:00.000Z',
    }),
  ]);

  const result = await portfolioService.adjustPosition('AAPL', {
    target_quantity: -2,
    price: 120,
    note: 'rebalance',
  });

  expect(result).toHaveLength(2);
  expect(result.map((trade) => trade.side)).toEqual(['SELL', 'SHORT']);
  expect(result.map((trade) => trade.quantity)).toEqual([5, 2]);
  expect(repository.appendTrades).toHaveBeenCalledWith(
    expect.arrayContaining([
      expect.objectContaining({ side: 'SELL', price: 120, source: 'adjustment' }),
      expect.objectContaining({ side: 'SHORT', price: 120, source: 'adjustment' }),
    ])
  );
});

test('getTransactions returns enriched signed quantities', async () => {
  repository.getAllTrades.mockResolvedValue([
    makeTrade({
      id: 'buy-aapl',
      ticker: 'AAPL',
      side: 'BUY',
      quantity: 1,
      price: 100,
      occurred_at: '2026-03-01T00:00:00.000Z',
    }),
    makeTrade({
      id: 'short-nvda',
      ticker: 'NVDA',
      side: 'SHORT',
      quantity: 2,
      price: 90,
      occurred_at: '2026-03-03T00:00:00.000Z',
    }),
  ]);

  const payload = await portfolioService.getTransactions();

  expect(payload.summary).toEqual({ count: 2, base_currency: 'INR' });
  expect(payload.transactions[0]).toEqual(
    expect.objectContaining({
      ticker: 'NVDA',
      signed_quantity: -2,
      price_base: 7200,
      base_currency: 'INR',
    })
  );
  expect(payload.transactions[1]).toEqual(
    expect.objectContaining({
      ticker: 'AAPL',
      signed_quantity: 1,
      price_base: 8000,
    })
  );
});

test('getPortfolioHistory builds gross-exposure curve from the transaction ledger', async () => {
  jest.useFakeTimers().setSystemTime(new Date('2026-03-04T12:00:00.000Z'));

  try {
    repository.getAllTrades.mockResolvedValue([
      makeTrade({
        id: 'buy-aapl',
        ticker: 'AAPL',
        side: 'BUY',
        quantity: 2,
        price: 100,
        occurred_at: '2026-03-02T00:00:00.000Z',
      }),
      makeTrade({
        id: 'sell-aapl',
        ticker: 'AAPL',
        side: 'SELL',
        quantity: 1,
        price: 120,
        occurred_at: '2026-03-04T00:00:00.000Z',
      }),
    ]);

    axios.get.mockImplementation(async (url) => {
      if (url.includes('/market/history/AAPL')) {
        return {
          data: {
            symbol: 'AAPL',
            history: [
              { date: '2026-03-02', close: 100 },
              { date: '2026-03-03', close: 110 },
              { date: '2026-03-04', close: 120 },
            ],
          },
        };
      }

      throw new Error(`Unexpected URL ${url}`);
    });

    const result = await portfolioService.getPortfolioHistory(3);

    expect(result.symbol_count).toBe(1);
    expect(result.days).toBe(3);
    expect(result.equity_curve).toEqual([
      { date: '2026-03-02', portfolio_value: 16000 },
      { date: '2026-03-03', portfolio_value: 17600 },
      { date: '2026-03-04', portfolio_value: 9600 },
    ]);
  } finally {
    jest.useRealTimers();
  }
});

test('getPortfolioInsights computes weights and performer ranking from active ledger positions', async () => {
  jest.useFakeTimers().setSystemTime(new Date('2026-03-04T12:00:00.000Z'));

  try {
    repository.getAllTrades.mockResolvedValue([
      makeTrade({
        id: 'buy-rel',
        ticker: 'RELIANCE.NS',
        symbol: 'RELIANCE',
        display_name: 'Reliance Industries Ltd',
        market: 'IN',
        exchange: 'NSE',
        instrument_currency: 'INR',
        side: 'BUY',
        quantity: 10,
        price: 100,
        occurred_at: '2026-03-01T00:00:00.000Z',
      }),
      makeTrade({
        id: 'short-infy',
        ticker: 'INFY.NS',
        symbol: 'INFY',
        display_name: 'Infosys Ltd',
        market: 'IN',
        exchange: 'NSE',
        instrument_currency: 'INR',
        side: 'SHORT',
        quantity: 2,
        price: 300,
        occurred_at: '2026-03-01T00:00:00.000Z',
      }),
    ]);

    axios.get.mockImplementation(async (url) => {
      if (url.includes('/market/latest-price/RELIANCE.NS')) {
        return { data: { symbol: 'RELIANCE.NS', price: 500 } };
      }
      if (url.includes('/market/latest-price/INFY.NS')) {
        return { data: { symbol: 'INFY.NS', price: 200 } };
      }
      if (url.includes('/market/history/RELIANCE.NS')) {
        return {
          data: {
            symbol: 'RELIANCE.NS',
            history: [
              { date: '2026-03-02', close: 100 },
              { date: '2026-03-03', close: 120 },
              { date: '2026-03-04', close: 140 },
            ],
          },
        };
      }
      if (url.includes('/market/history/INFY.NS')) {
        return {
          data: {
            symbol: 'INFY.NS',
            history: [
              { date: '2026-03-02', close: 250 },
              { date: '2026-03-03', close: 240 },
              { date: '2026-03-04', close: 200 },
            ],
          },
        };
      }
      throw new Error(`Unexpected URL ${url}`);
    });

    const result = await portfolioService.getPortfolioInsights(3);

    expect(result).toEqual(
      expect.objectContaining({
        concentration_risk: 'HIGH',
        largest_position: {
          ticker: 'RELIANCE.NS',
          weight: 92.59,
          current_value: 5000,
        },
        best_performer: {
          ticker: 'RELIANCE.NS',
          profit_loss_percent: 400,
          weight: 92.59,
        },
        worst_performer: {
          ticker: 'INFY.NS',
          profit_loss_percent: 33.33,
        },
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
