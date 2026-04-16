import { expect, test } from '@playwright/test'

const instruments = {
  RELIANCE: {
    id: 'IN:RELIANCE.NS',
    symbol: 'RELIANCE',
    normalized: 'RELIANCE.NS',
    display_name: 'Reliance Industries',
    market: 'IN',
    exchange: 'NSE',
    instrument_type: 'equity',
    country: 'IN',
  },
  TCS: {
    id: 'IN:TCS.NS',
    symbol: 'TCS',
    normalized: 'TCS.NS',
    display_name: 'Tata Consultancy Services',
    market: 'IN',
    exchange: 'NSE',
    instrument_type: 'equity',
    country: 'IN',
  },
  NVDA: {
    id: 'US:NVDA',
    symbol: 'NVDA',
    normalized: 'NVDA',
    display_name: 'NVIDIA Corporation',
    market: 'US',
    exchange: 'NASDAQ',
    instrument_type: 'equity',
    country: 'US',
  },
  SPX: {
    id: 'US:^GSPC',
    symbol: '^GSPC',
    normalized: '^GSPC',
    display_name: 'S&P 500',
    market: 'US',
    exchange: 'SP',
    instrument_type: 'index',
    country: 'US',
  },
  NIFTY: {
    id: 'IN:^NSEI',
    symbol: '^NSEI',
    normalized: '^NSEI',
    display_name: 'Nifty 50',
    market: 'IN',
    exchange: 'NSE',
    instrument_type: 'index',
    country: 'IN',
  },
}

function quoteFor(symbol) {
  if (symbol === 'NVDA') {
    return {
      symbol,
      current_price: 912.42,
      previous_close: 901.04,
      day_change: 11.38,
      day_change_pct: 1.26,
      trend_5d_pct: 4.8,
      trend_30d_pct: 11.6,
      currency: 'USD',
      as_of: '2026-04-16T10:00:00Z',
    }
  }

  if (symbol === 'TCS.NS') {
    return {
      symbol,
      current_price: 4128.2,
      previous_close: 4099.35,
      day_change: 28.85,
      day_change_pct: 0.7,
      trend_5d_pct: 1.9,
      trend_30d_pct: 7.2,
      currency: 'INR',
      as_of: '2026-04-16T10:00:00Z',
    }
  }

  if (symbol === '^GSPC') {
    return {
      symbol,
      current_price: 5422.31,
      previous_close: 5398.45,
      day_change: 23.86,
      day_change_pct: 0.44,
      trend_5d_pct: 1.2,
      trend_30d_pct: 3.9,
      currency: 'USD',
      as_of: '2026-04-16T10:00:00Z',
    }
  }

  if (symbol === '^NSEI') {
    return {
      symbol,
      current_price: 23850.55,
      previous_close: 23790.1,
      day_change: 60.45,
      day_change_pct: 0.25,
      trend_5d_pct: 1.5,
      trend_30d_pct: 5.4,
      currency: 'INR',
      as_of: '2026-04-16T10:00:00Z',
    }
  }

  return {
    symbol,
    current_price: 2885.5,
    previous_close: 2850,
    day_change: 35.5,
    day_change_pct: 1.25,
    trend_5d_pct: 2.6,
    trend_30d_pct: 6.4,
    currency: 'INR',
    as_of: '2026-04-16T10:00:00Z',
  }
}

function validationFor(symbol) {
  if (symbol === 'NVDA') {
    return {
      symbol,
      period: {
        start_date: '2025-04-02',
        end_date: '2026-04-02',
        horizon: 5,
      },
      total_predictions: 104,
      accuracy: 0.6282,
      ece: 0.0614,
      brier_score: 0.2177,
      accuracy_by_confidence: {
        '0.50-0.60': 0.57,
        '0.60-0.70': 0.66,
        '0.70-0.80': 0.72,
      },
      reliability_curve: [
        { probability_mean: 0.56, accuracy: 0.57, count: 28 },
        { probability_mean: 0.66, accuracy: 0.66, count: 44 },
        { probability_mean: 0.74, accuracy: 0.72, count: 32 },
      ],
    }
  }

  return {
    symbol,
    period: {
      start_date: '2025-04-02',
      end_date: '2026-04-02',
      horizon: 5,
    },
    total_predictions: 104,
    accuracy: symbol === '^GSPC' ? 0.6038 : 0.6154,
    ece: symbol === '^GSPC' ? 0.0681 : 0.0732,
    brier_score: symbol === '^GSPC' ? 0.2177 : 0.2214,
    accuracy_by_confidence: {
      '0.50-0.60': 0.55,
      '0.60-0.70': 0.63,
      '0.70-0.80': 0.7,
    },
    reliability_curve: [
      { probability_mean: 0.56, accuracy: 0.55, count: 30 },
      { probability_mean: 0.66, accuracy: 0.63, count: 42 },
      { probability_mean: 0.74, accuracy: 0.7, count: 32 },
    ],
  }
}

function buildSummary(holdings) {
  const totalValue = holdings.reduce((sum, holding) => sum + Number(holding.current_value || 0), 0)
  const totalPnl = holdings.reduce((sum, holding) => sum + Number(holding.profit_loss || 0), 0)
  const longPositions = holdings.filter((holding) => holding.side === 'LONG').length
  const shortPositions = holdings.filter((holding) => holding.side === 'SHORT').length

  return {
    total_portfolio_value: totalValue,
    total_gross_exposure: totalValue,
    total_net_exposure: totalValue,
    total_invested_value: totalValue,
    total_unrealized_pnl: totalPnl,
    total_realized_pnl: 0,
    total_profit_loss: totalPnl,
    total_profit_loss_percent: totalValue ? (totalPnl / totalValue) * 100 : 0,
    active_positions: holdings.length,
    long_positions: longPositions,
    short_positions: shortPositions,
    winners_count: holdings.filter((holding) => Number(holding.profit_loss) >= 0).length,
    losers_count: holdings.filter((holding) => Number(holding.profit_loss) < 0).length,
    base_currency: 'INR',
  }
}

async function registerMockApi(page) {
  const state = {
    holdings: [
      {
        id: 'holding-msft',
        ticker: 'MSFT',
        display_name: 'Microsoft Corporation',
        side: 'LONG',
        quantity: 4,
        avg_price: 412,
        current_value: 1680,
        profit_loss: 48,
        profit_loss_percent: 2.94,
        instrument_currency: 'USD',
        market_value_base: 1680,
      },
    ],
    transactions: [],
  }

  const buildPortfolio = () => ({
    holdings: state.holdings,
    positions: [],
    summary: buildSummary(state.holdings),
  })

  await page.route('**/api/symbols/search**', async (route) => {
    const query = new URL(route.request().url()).searchParams.get('q')?.toUpperCase() || ''
    const results = Object.values(instruments).filter((instrument) => {
      return (
        instrument.normalized.includes(query) ||
        instrument.display_name.toUpperCase().includes(query) ||
        instrument.symbol.includes(query)
      )
    })

    await route.fulfill({ json: { results } })
  })

  await page.route('**/api/symbols/normalize/**', async (route) => {
    const input = decodeURIComponent(route.request().url().split('/').pop() || '').toUpperCase()
    const instrument =
      Object.values(instruments).find((candidate) => candidate.normalized === input) ||
      Object.values(instruments).find((candidate) => candidate.symbol === input)

    await route.fulfill({
      status: instrument ? 200 : 404,
      json: instrument
        ? {
            input,
            normalized: instrument.normalized,
            changed: input !== instrument.normalized,
            ...instrument,
          }
        : { error: 'Unsupported symbol' },
    })
  })

  await page.route('**/api/market/quote/**', async (route) => {
    const symbol = decodeURIComponent(route.request().url().split('/').pop() || '').toUpperCase()
    await route.fulfill({ json: quoteFor(symbol) })
  })

  await page.route('**/api/market/history/**', async (route) => {
    await route.fulfill({
      json: {
        history: [
          { date: '2026-04-10', close: 2820 },
          { date: '2026-04-11', close: 2835 },
          { date: '2026-04-12', close: 2850 },
          { date: '2026-04-13', close: 2868 },
          { date: '2026-04-14', close: 2885.5 },
        ],
      },
    })
  })

  await page.route('**/api/command-center', async (route) => {
    await route.fulfill({
      json: {
        recent_signals: [
          {
            ...instruments.SPX,
            signal: 'LONG',
            confidence_level: 'High confidence',
          },
        ],
        market_sessions: [
          {
            market: 'IN',
            label: 'India',
            session_status: 'open',
            local_time: '11:15',
            opens_at: '09:15',
            closes_at: '15:30',
          },
          {
            market: 'US',
            label: 'US',
            session_status: 'pre-open',
            local_time: '07:45',
            opens_at: '09:30',
            closes_at: '16:00',
          },
        ],
        portfolio_summary: buildSummary(state.holdings),
        risk_headline: 'Concentration is controlled.',
        top_portfolio_action: 'Wait for a validated setup before sizing up.',
        daily_brief: {
          headline: 'Operator ready',
          bullets: ['One live position. New paper trades will refresh the portfolio workspace.'],
        },
        market_intelligence: {
          companyHeadlines: [{ title: 'Large-cap momentum remains constructive.' }],
          macroHeadlines: [{ title: 'Rates are stable into the India session.' }],
        },
      },
    })
  })

  await page.route('**/api/analyze/recent**', async (route) => {
    await route.fulfill({
      json: {
        results: [
          {
            ...instruments.NIFTY,
            signal: 'LONG',
            confidence_level: 'High confidence',
          },
          {
            ...instruments.NVDA,
            signal: 'LONG',
            confidence_level: 'High confidence',
          },
        ],
      },
    })
  })

  await page.route('**/api/portfolio', async (route) => {
    await route.fulfill({ json: buildPortfolio() })
  })

  await page.route('**/api/portfolio/history**', async (route) => {
    await route.fulfill({
      json: {
        days: 30,
        equity_curve: [{ date: '2026-04-15', portfolio_value: buildSummary(state.holdings).total_portfolio_value }],
      },
    })
  })

  await page.route('**/api/portfolio/insights**', async (route) => {
    await route.fulfill({
      json: {
        concentration_risk: 'LOW',
        largest_position: { ticker: state.holdings[0]?.ticker || 'MSFT', weight: 0.48 },
        best_performer: { ticker: 'MSFT', profit_loss_percent: 2.94 },
        worst_performer: { ticker: 'MSFT', profit_loss_percent: 2.94 },
        diversification_score: 4.1,
        volatility_level: 'LOW',
        insights: ['Risk remains in a manageable band.'],
      },
    })
  })

  await page.route('**/api/portfolio/advisor**', async (route) => {
    await route.fulfill({
      json: {
        recommendations: ['Keep position sizing conservative until more validations are run.'],
      },
    })
  })

  await page.route('**/api/portfolio/transactions**', async (route) => {
    await route.fulfill({
      json: {
        transactions: state.transactions,
        summary: { count: state.transactions.length, base_currency: 'INR' },
      },
    })
  })

  await page.route('**/api/analyze/validate', async (route) => {
    const payload = route.request().postDataJSON()
    await route.fulfill({ json: validationFor(payload.symbol) })
  })

  await page.route('**/api/analyze', async (route) => {
    const payload = route.request().postDataJSON()
    const instrument =
      Object.values(instruments).find((candidate) => candidate.normalized === payload.symbol) ||
      instruments.RELIANCE

    await route.fulfill({
      json: {
        ...instrument,
        symbol: instrument.normalized,
        decision_label: 'Long',
        signal: 'LONG',
        signal_explanation: 'Long intraday setup detected.',
        confidence_level: 'High confidence',
        probability: 0.67,
        entry_price: 2880,
        stop_price: 2840,
        take_profit_price: 2945,
        forced_exit_time: '15:15 IST',
        model_name: 'xgboost',
        trend_summary: 'Momentum and breadth remain constructive.',
        risk_summary: 'Risk stays inside the session bracket.',
        model_honesty: 'This is directional quality, not proof of profitability.',
        sentiment_gate_reason: 'Company and sector news are supportive.',
      },
    })
  })

  await page.route('**/api/portfolio/trades', async (route) => {
    const payload = route.request().postDataJSON()
    const instrument =
      Object.values(instruments).find((candidate) => candidate.normalized === payload.ticker) ||
      instruments.RELIANCE

    state.holdings.unshift({
      id: `holding-${payload.ticker}`,
      ticker: payload.ticker,
      display_name: instrument.display_name,
      side: payload.side === 'SHORT' ? 'SHORT' : 'LONG',
      quantity: payload.quantity,
      avg_price: payload.price,
      current_value: payload.quantity * payload.price,
      profit_loss: 0,
      profit_loss_percent: 0,
      instrument_currency: quoteFor(payload.ticker).currency,
      market_value_base: payload.quantity * payload.price,
    })

    state.transactions.unshift({
      id: `trade-${Date.now()}`,
      ticker: payload.ticker,
      display_name: instrument.display_name,
      side: payload.side,
      quantity: payload.quantity,
      price: payload.price,
      price_base: payload.price,
      occurred_at: new Date().toISOString(),
      source: 'manual',
    })

    await route.fulfill({ json: { ok: true } })
  })

  await page.route('**/api/portfolio/positions/**/adjust', async (route) => {
    await route.fulfill({ json: { ok: true } })
  })
}

async function lockInstrument(page, query, name) {
  const input = page.getByPlaceholder('Search US or India stocks and indices')
  await input.click()
  await input.fill(query)
  const option = page.getByRole('option', { name: new RegExp(name, 'i') })
  await expect(option).toBeVisible()
  await option.click()
}

function activeInstrumentBar(page) {
  return page.locator('.instrument-context-bar')
}

async function expectActiveInstrument(page, instrument, priceText) {
  const bar = activeInstrumentBar(page)
  await expect(bar.getByText(instrument.display_name)).toBeVisible()
  await expect(bar.getByText(instrument.normalized, { exact: true })).toBeVisible()
  await expect(bar.getByText(priceText, { exact: true })).toBeVisible()
}

async function expectNoHorizontalOverflow(page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
  expect(overflow).toBeLessThanOrEqual(1)
}

test('picker resolves a US equity, India equity, US index, and India index', async ({ page }) => {
  await registerMockApi(page)
  await page.goto('/')

  await expect(page.getByRole('heading', { name: /Pick symbol, inspect quote/i })).toBeVisible()

  const coverageSet = [
    { query: 'NVIDIA', instrument: instruments.NVDA, price: '$912.42' },
    { query: 'TCS', instrument: instruments.TCS, price: '₹4,128.20' },
    { query: 'S&P 500', instrument: instruments.SPX, price: '$5,422.31' },
    { query: 'Nifty', instrument: instruments.NIFTY, price: '₹23,850.55' },
  ]

  for (const item of coverageSet) {
    await lockInstrument(page, item.query, item.instrument.display_name)
    await expectActiveInstrument(page, item.instrument, item.price)
    await expect(page.getByRole('button', { name: 'Analyze' })).toBeEnabled()
    await expect(page.getByRole('button', { name: 'Paper Trade' }).first()).toBeEnabled()
  }
})

test('analysis and validation run from the UI for India and US symbols', async ({ page }) => {
  await registerMockApi(page)
  await page.goto('/analysis')

  await lockInstrument(page, 'Reliance', 'Reliance Industries')
  await expectActiveInstrument(page, instruments.RELIANCE, '₹2,885.50')
  await page.getByRole('button', { name: 'Analyze' }).click()
  await expect(page.getByRole('heading', { name: 'Long' })).toBeVisible()
  await expect(page.getByText('This is directional quality, not proof of profitability.')).toBeVisible()
  await page.getByRole('button', { name: 'Run validation' }).click()
  await expect(page.getByText('Empirical check')).toBeVisible()
  await expect(page.getByText('Accuracy', { exact: true }).first()).toBeVisible()
  await expect(page.getByText('0.2214')).toBeVisible()

  await lockInstrument(page, 'NVIDIA', 'NVIDIA Corporation')
  await expectActiveInstrument(page, instruments.NVDA, '$912.42')
  await page.getByRole('button', { name: 'Analyze' }).click()
  await expect(page.getByRole('heading', { name: 'Long' })).toBeVisible()
  await page.getByRole('button', { name: 'Run validation' }).click()
  await expect(page.getByText('0.2177')).toBeVisible()
})

test('global paper trade drawer books a trade and refreshes the portfolio workspace', async ({ page }) => {
  await registerMockApi(page)
  await page.goto('/analysis')

  await lockInstrument(page, 'Reliance', 'Reliance Industries')
  await expectActiveInstrument(page, instruments.RELIANCE, '₹2,885.50')

  await page.getByRole('button', { name: 'Paper Trade' }).first().click()
  await expect(page.getByRole('heading', { name: 'Book the order' })).toBeVisible()
  await page.getByLabel('Quantity').fill('5')
  await page.getByLabel('Price').fill('2800')
  await page.getByRole('button', { name: 'Book paper trade' }).click()

  await expect(page.getByText('Paper trade booked for RELIANCE.NS.')).toBeVisible()
  await page.getByRole('link', { name: 'Portfolio' }).click()
  await expect(page.getByText('Live book')).toBeVisible()
  await expect(page.getByText('Transaction history')).toBeVisible()
  const transactionTable = page.getByRole('table').nth(1)
  await expect(transactionTable.getByRole('cell', { name: 'manual' })).toBeVisible()
  await expect(transactionTable.getByText('RELIANCE.NS', { exact: true })).toBeVisible()
})

test('responsive layouts keep the picker and trade drawer usable', async ({ page }) => {
  await registerMockApi(page)

  const viewports = [
    { width: 1440, height: 1200, label: 'desktop' },
    { width: 1024, height: 1180, label: 'tablet' },
    { width: 390, height: 844, label: 'mobile' },
  ]

  for (const viewport of viewports) {
    await page.setViewportSize({ width: viewport.width, height: viewport.height })
    await page.goto('/')
    await lockInstrument(page, 'Nifty', 'Nifty 50')

    await expectActiveInstrument(page, instruments.NIFTY, '₹23,850.55')
    await expect(page.getByRole('button', { name: 'Analyze' })).toBeVisible()
    await expectNoHorizontalOverflow(page)

    await page.screenshot({ path: `test-results/${viewport.label}-today.png`, fullPage: true })

    await page.getByRole('link', { name: 'Analysis' }).click()
    await expectNoHorizontalOverflow(page)
    await page.screenshot({ path: `test-results/${viewport.label}-analysis.png`, fullPage: true })

    await page.getByRole('link', { name: 'Portfolio' }).click()
    await expectNoHorizontalOverflow(page)
    await page.screenshot({ path: `test-results/${viewport.label}-portfolio.png`, fullPage: true })

    await page.getByPlaceholder('Search US or India stocks and indices').click()
    await page.screenshot({ path: `test-results/${viewport.label}-picker-open.png`, fullPage: true })

    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: 'Paper Trade' }).first().click()
    await expect(page.getByRole('heading', { name: 'Book the order' })).toBeVisible()
    await page.screenshot({ path: `test-results/${viewport.label}-trade-drawer.png`, fullPage: true })
  }
})
