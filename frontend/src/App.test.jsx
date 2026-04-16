import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'

const mockInstrument = {
  id: 'IN:RELIANCE.NS',
  symbol: 'RELIANCE',
  normalized: 'RELIANCE.NS',
  display_name: 'Reliance Industries',
  market: 'IN',
  exchange: 'NSE',
  instrument_type: 'equity',
  country: 'IN',
}

const mockQuote = {
  symbol: 'RELIANCE.NS',
  current_price: 2885.5,
  previous_close: 2850,
  day_change: 35.5,
  day_change_pct: 1.25,
  trend_5d_pct: 2.6,
  trend_30d_pct: 6.4,
  currency: 'INR',
  as_of: '2026-04-16T09:20:00Z',
}

const mockValidation = {
  symbol: 'RELIANCE.NS',
  period: {
    start_date: '2025-04-02',
    end_date: '2026-04-02',
    horizon: 5,
  },
  total_predictions: 104,
  accuracy: 0.6154,
  ece: 0.0732,
  brier_score: 0.2214,
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

const mockAnalysis = {
  ...mockInstrument,
  symbol: 'RELIANCE.NS',
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
}

const fetchCommandCenter = vi.fn()
const analyzeMarket = vi.fn()
const fetchQuoteSnapshot = vi.fn()
const fetchRecentAnalyses = vi.fn()
const fetchValidationReport = vi.fn()
const fetchPortfolio = vi.fn()
const fetchPortfolioHistory = vi.fn()
const fetchPortfolioInsights = vi.fn()
const fetchPortfolioAdvisor = vi.fn()
const fetchPortfolioTransactions = vi.fn()
const createPortfolioTrade = vi.fn()
const adjustPortfolioPosition = vi.fn()
const normalizeSymbol = vi.fn()

vi.mock('./api/commandCenter', () => ({
  fetchCommandCenter: (...args) => fetchCommandCenter(...args),
}))

vi.mock('./api/analysis', () => ({
  analyzeMarket: (...args) => analyzeMarket(...args),
  fetchQuoteSnapshot: (...args) => fetchQuoteSnapshot(...args),
  fetchRecentAnalyses: (...args) => fetchRecentAnalyses(...args),
  fetchValidationReport: (...args) => fetchValidationReport(...args),
}))

vi.mock('./api/portfolio', () => ({
  fetchPortfolio: (...args) => fetchPortfolio(...args),
  fetchPortfolioHistory: (...args) => fetchPortfolioHistory(...args),
  fetchPortfolioInsights: (...args) => fetchPortfolioInsights(...args),
  fetchPortfolioAdvisor: (...args) => fetchPortfolioAdvisor(...args),
  fetchPortfolioTransactions: (...args) => fetchPortfolioTransactions(...args),
  createPortfolioTrade: (...args) => createPortfolioTrade(...args),
  adjustPortfolioPosition: (...args) => adjustPortfolioPosition(...args),
}))

vi.mock('./api/symbols', () => ({
  normalizeSymbol: (...args) => normalizeSymbol(...args),
}))

vi.mock('./components/InstrumentPicker', () => ({
  default: function MockInstrumentPicker({ value, onChange }) {
    return (
      <button type="button" onClick={() => onChange(mockInstrument)}>
        {value?.normalized || 'Lock Reliance'}
      </button>
    )
  },
}))

vi.mock('./components/analysis/StockPriceChart', () => ({
  default: function MockStockPriceChart({ symbol }) {
    return <div>Chart for {symbol}</div>
  },
}))

vi.mock('./components/portfolio/PortfolioEquityChart', () => ({
  default: function MockPortfolioEquityChart() {
    return <div>Portfolio equity chart</div>
  },
}))

vi.mock('./components/portfolio/PortfolioAllocationChart', () => ({
  default: function MockPortfolioAllocationChart() {
    return <div>Portfolio allocation chart</div>
  },
}))

vi.mock('./components/portfolio/PortfolioInsights', () => ({
  default: function MockPortfolioInsights() {
    return <div>Portfolio insights block</div>
  },
}))

vi.mock('./components/portfolio/PortfolioAdvisor', () => ({
  default: function MockPortfolioAdvisor() {
    return <div>Portfolio advisor block</div>
  },
}))

let holdingsState
let transactionsState

function resetPortfolioState() {
  holdingsState = [
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
  ]
  transactionsState = []
}

function buildPortfolioSummary() {
  const totalValue = holdingsState.reduce(
    (sum, holding) => sum + Number(holding.current_value || 0),
    0,
  )
  const totalPnl = holdingsState.reduce(
    (sum, holding) => sum + Number(holding.profit_loss || 0),
    0,
  )

  return {
    total_portfolio_value: totalValue,
    total_gross_exposure: totalValue,
    total_net_exposure: totalValue,
    total_invested_value: totalValue,
    total_unrealized_pnl: totalPnl,
    total_realized_pnl: 0,
    total_profit_loss: totalPnl,
    total_profit_loss_percent: totalValue ? (totalPnl / totalValue) * 100 : 0,
    active_positions: holdingsState.length,
    long_positions: holdingsState.filter((holding) => holding.side === 'LONG').length,
    short_positions: holdingsState.filter((holding) => holding.side === 'SHORT').length,
    winners_count: holdingsState.filter((holding) => Number(holding.profit_loss) >= 0).length,
    losers_count: holdingsState.filter((holding) => Number(holding.profit_loss) < 0).length,
    base_currency: 'INR',
  }
}

function installDefaultMocks() {
  fetchCommandCenter.mockResolvedValue({
    recent_signals: [],
    market_sessions: [
      {
        market: 'IN',
        label: 'India',
        session_status: 'open',
        local_time: '11:15',
        opens_at: '09:15',
        closes_at: '15:30',
      },
    ],
    portfolio_summary: buildPortfolioSummary(),
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
  })

  fetchRecentAnalyses.mockResolvedValue({
    results: [
      {
        id: 'recent-msft',
        symbol: 'MSFT',
        normalized: 'MSFT',
        display_name: 'Microsoft Corporation',
        market: 'US',
        exchange: 'NASDAQ',
        instrument_type: 'equity',
        signal: 'LONG',
        confidence_level: 'High confidence',
      },
    ],
  })

  fetchPortfolio.mockImplementation(async () => ({
    holdings: [...holdingsState],
    positions: [],
    summary: buildPortfolioSummary(),
  }))

  fetchPortfolioHistory.mockResolvedValue({
    days: 30,
    equity_curve: [{ date: '2026-04-15', portfolio_value: 1680 }],
  })

  fetchPortfolioInsights.mockResolvedValue({
    concentration_risk: 'LOW',
    largest_position: { ticker: 'MSFT', weight: 0.58 },
    best_performer: { ticker: 'MSFT', profit_loss_percent: 2.94 },
    worst_performer: { ticker: 'MSFT', profit_loss_percent: 2.94 },
    diversification_score: 4.2,
    volatility_level: 'LOW',
    insights: ['Risk is concentrated but manageable.'],
  })

  fetchPortfolioAdvisor.mockResolvedValue({
    recommendations: ['Keep gross exposure controlled while testing new setups.'],
  })

  fetchPortfolioTransactions.mockImplementation(async () => ({
    transactions: [...transactionsState],
    summary: { count: transactionsState.length, base_currency: 'INR' },
  }))

  normalizeSymbol.mockResolvedValue(mockInstrument)
  fetchQuoteSnapshot.mockResolvedValue(mockQuote)
  analyzeMarket.mockResolvedValue(mockAnalysis)
  fetchValidationReport.mockResolvedValue(mockValidation)

  createPortfolioTrade.mockImplementation(async (payload) => {
    holdingsState = [
      {
        id: 'holding-reliance',
        ticker: payload.ticker,
        display_name: mockInstrument.display_name,
        side: payload.side === 'SHORT' ? 'SHORT' : 'LONG',
        quantity: payload.quantity,
        avg_price: payload.price,
        current_value: payload.quantity * payload.price,
        profit_loss: 0,
        profit_loss_percent: 0,
        instrument_currency: 'INR',
        market_value_base: payload.quantity * payload.price,
      },
      ...holdingsState,
    ]

    transactionsState = [
      {
        id: 'trade-1',
        ticker: payload.ticker,
        display_name: mockInstrument.display_name,
        side: payload.side,
        quantity: payload.quantity,
        price: payload.price,
        price_base: payload.price,
        occurred_at: '2026-04-16T09:30:00Z',
        source: 'manual',
      },
      ...transactionsState,
    ]

    return { ok: true }
  })

  adjustPortfolioPosition.mockResolvedValue({ ok: true })
}

describe('TradeSense active trader shell', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    resetPortfolioState()
    installDefaultMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders only the core top-level workspaces and empty active-instrument guidance', async () => {
    render(<App />)

    expect(await screen.findByText('Pick symbol, inspect quote, analyze, place paper trade, review the book.')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Today' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Analysis' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Portfolio' })).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Holdings' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Intelligence' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'No Trade' })).not.toBeInTheDocument()
    expect(screen.getByText('No instrument locked')).toBeInTheDocument()
  })

  it('locks an instrument, runs analysis, and exposes validation in the same workspace', async () => {
    render(<App />)

    fireEvent.click((await screen.findAllByRole('button', { name: 'Lock Reliance' }))[0])

    expect((await screen.findAllByText('Reliance Industries')).length).toBeGreaterThan(0)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Analyze' })).toBeEnabled())

    fireEvent.click(screen.getByRole('button', { name: 'Analyze' }))

    expect(await screen.findByRole('heading', { name: 'Long' })).toBeInTheDocument()
    expect(screen.getByText('This is directional quality, not proof of profitability.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Run validation' })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Run validation' }))

    expect(await screen.findByText('Empirical check')).toBeInTheDocument()
    expect(screen.getByText('Accuracy')).toBeInTheDocument()
    expect(screen.getByText('Brier score')).toBeInTheDocument()
    expect(screen.getByText('0.2214')).toBeInTheDocument()
  })

  it('books a global paper trade and refreshes the portfolio workspace', async () => {
    render(<App />)

    fireEvent.click((await screen.findAllByRole('button', { name: 'Lock Reliance' }))[0])
    expect((await screen.findAllByText('Reliance Industries')).length).toBeGreaterThan(0)

    fireEvent.click(screen.getAllByRole('button', { name: 'Paper Trade' })[0])

    expect(await screen.findByRole('heading', { name: 'Book the order' })).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Quantity'), { target: { value: '5' } })
    fireEvent.change(screen.getByLabelText('Price'), { target: { value: '2800' } })
    fireEvent.click(screen.getByRole('button', { name: 'Book paper trade' }))

    expect(await screen.findByText('Paper trade booked for RELIANCE.NS.')).toBeInTheDocument()
    expect(createPortfolioTrade).toHaveBeenCalledWith(
      expect.objectContaining({
        ticker: 'RELIANCE.NS',
        quantity: 5,
        price: 2800,
      }),
    )

    fireEvent.click(screen.getByRole('link', { name: 'Portfolio' }))

    expect(await screen.findByText('Live book')).toBeInTheDocument()
    expect(screen.getAllByText('Reliance Industries').length).toBeGreaterThan(0)
    expect(screen.getByText('Transaction history')).toBeInTheDocument()

    const transactionTable = screen.getAllByRole('table')[1]
    expect(within(transactionTable).getAllByText('manual').length).toBeGreaterThan(0)
  })
})
