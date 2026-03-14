const { generatePortfolioInsights } = require('../portfolio_intelligence.service');

test('generatePortfolioInsights detects high concentration risk for largest holding above 50%', () => {
  const result = generatePortfolioInsights({
    holdings: [
      {
        ticker: 'AAPL',
        market_value_base: 5500,
        instrument_currency: 'USD',
      },
      {
        ticker: 'XOM',
        market_value_base: 4500,
        instrument_currency: 'USD',
      },
    ],
    summary: {
      base_currency: 'USD',
    },
  });

  const concentrationInsight = result.insights.find(
    (insight) => insight.type === 'concentration_risk'
  );

  expect(concentrationInsight).toEqual(
    expect.objectContaining({
      severity: 'high',
    })
  );
  expect(concentrationInsight.message).toContain('AAPL');
  expect(concentrationInsight.message).toContain('55');
  expect(result.diversification_score).toBe(60);
});

test('generatePortfolioInsights flags sector dominance when a sector exceeds 60%', () => {
  const result = generatePortfolioInsights({
    holdings: [
      {
        ticker: 'AAPL',
        market_value_base: 4000,
        instrument_currency: 'USD',
      },
      {
        ticker: 'NVDA',
        market_value_base: 3000,
        instrument_currency: 'USD',
      },
      {
        ticker: 'XOM',
        market_value_base: 3000,
        instrument_currency: 'USD',
      },
    ],
    summary: {
      base_currency: 'USD',
    },
  });

  const sectorInsight = result.insights.find((insight) => insight.type === 'sector_exposure');

  expect(sectorInsight).toEqual(
    expect.objectContaining({
      severity: 'high',
    })
  );
  expect(sectorInsight.message).toContain('Technology');
  expect(sectorInsight.message).toContain('70');
  expect(result.diversification_score).toBe(55);
});

test('generatePortfolioInsights highlights foreign currency exposure above 50%', () => {
  const result = generatePortfolioInsights({
    holdings: [
      {
        ticker: 'AAPL',
        market_value_base: 3000,
        instrument_currency: 'USD',
      },
      {
        ticker: 'MSFT',
        market_value_base: 2500,
        instrument_currency: 'USD',
      },
      {
        ticker: 'RELIANCE.NS',
        market_value_base: 4500,
        instrument_currency: 'INR',
      },
    ],
    summary: {
      base_currency: 'INR',
    },
  });

  const currencyInsight = result.insights.find((insight) => insight.type === 'currency_exposure');

  expect(currencyInsight).toEqual(
    expect.objectContaining({
      severity: 'medium',
    })
  );
  expect(currencyInsight.message).toContain('55');
  expect(currencyInsight.message).toContain('INR');
});

test('generatePortfolioInsights returns perfect diversification score for balanced multi-sector portfolio', () => {
  const result = generatePortfolioInsights({
    holdings: [
      {
        ticker: 'AAPL',
        market_value_base: 2500,
        instrument_currency: 'USD',
      },
      {
        ticker: 'HDFCBANK.NS',
        market_value_base: 2500,
        instrument_currency: 'INR',
      },
      {
        ticker: 'RELIANCE.NS',
        market_value_base: 2500,
        instrument_currency: 'INR',
      },
      {
        ticker: 'AMZN',
        market_value_base: 2500,
        instrument_currency: 'USD',
      },
    ],
    summary: {
      base_currency: 'USD',
    },
  });

  expect(result).toEqual({
    diversification_score: 100,
    insights: [],
  });
});
