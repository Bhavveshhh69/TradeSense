import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  BrowserRouter,
  Navigate,
  NavLink,
  Route,
  Routes,
  useNavigate,
} from 'react-router-dom'

import './App.css'
import {
  analyzeMarket,
  fetchQuoteSnapshot,
  fetchRecentAnalyses,
  fetchValidationReport,
} from './api/analysis'
import { fetchCommandCenter } from './api/commandCenter'
import {
  adjustPortfolioPosition,
  createPortfolioTrade,
  fetchPortfolio,
  fetchPortfolioAdvisor,
  fetchPortfolioHistory,
  fetchPortfolioInsights,
  fetchPortfolioTransactions,
} from './api/portfolio'
import { normalizeSymbol } from './api/symbols'
import StockPriceChart from './components/analysis/StockPriceChart'
import InstrumentPicker from './components/InstrumentPicker'
import PortfolioAdvisor from './components/portfolio/PortfolioAdvisor'
import PortfolioAllocationChart from './components/portfolio/PortfolioAllocationChart'
import PortfolioEquityChart from './components/portfolio/PortfolioEquityChart'
import PortfolioInsights from './components/portfolio/PortfolioInsights'
import {
  buildPortfolioAdvisorInsights,
  computePortfolioAnalytics,
  formatMoney,
  formatPercent,
  formatSignedMoney,
} from './utils/dashboard'

const PORTFOLIO_DAYS = 30
const RECENT_SELECTIONS_KEY = 'tradesense.recent-instruments.v2'

const NAV_ITEMS = [
  { to: '/', label: 'Today' },
  { to: '/analysis', label: 'Analysis' },
  { to: '/portfolio', label: 'Portfolio' },
]

const EMPTY_PORTFOLIO = {
  holdings: [],
  positions: [],
  summary: {
    total_portfolio_value: 0,
    total_gross_exposure: 0,
    total_net_exposure: 0,
    total_invested_value: 0,
    total_unrealized_pnl: 0,
    total_realized_pnl: 0,
    total_profit_loss: 0,
    total_profit_loss_percent: 0,
    active_positions: 0,
    long_positions: 0,
    short_positions: 0,
    winners_count: 0,
    losers_count: 0,
    has_price_errors: false,
    base_currency: 'INR',
  },
}

const EMPTY_HISTORY = {
  days: PORTFOLIO_DAYS,
  equity_curve: [],
}

const EMPTY_INSIGHTS = {
  concentration_risk: 'LOW',
  largest_position: null,
  best_performer: null,
  worst_performer: null,
  diversification_score: 0,
  volatility_level: 'LOW',
  insights: [],
}

const EMPTY_ADVISOR = {
  recommendations: [],
}

const EMPTY_TRANSACTIONS = {
  transactions: [],
  summary: {
    count: 0,
    base_currency: 'INR',
  },
}

const EMPTY_COMMAND_CENTER = {
  recent_signals: [],
  market_sessions: [],
  portfolio_summary: EMPTY_PORTFOLIO.summary,
  risk_headline: '',
  top_portfolio_action: '',
  daily_brief: {
    headline: '',
    bullets: [],
  },
  market_intelligence: {
    companyHeadlines: [],
    macroHeadlines: [],
  },
}

function normalizeInstrument(value) {
  if (!value || typeof value !== 'object') {
    return null
  }

  const normalized =
    typeof value.normalized === 'string' && value.normalized.trim()
      ? value.normalized.trim().toUpperCase()
      : typeof value.symbol === 'string' && value.symbol.trim()
        ? value.symbol.trim().toUpperCase()
        : ''

  if (!normalized) {
    return null
  }

  return {
    id:
      typeof value.id === 'string' && value.id.trim()
        ? value.id.trim()
        : `${value.market || 'UNKNOWN'}:${normalized}`,
    symbol:
      typeof value.symbol === 'string' && value.symbol.trim()
        ? value.symbol.trim().toUpperCase()
        : normalized,
    normalized,
    display_name:
      typeof value.display_name === 'string' && value.display_name.trim()
        ? value.display_name.trim()
        : normalized,
    market:
      typeof value.market === 'string' && value.market.trim()
        ? value.market.trim().toUpperCase()
        : '',
    exchange:
      typeof value.exchange === 'string' && value.exchange.trim()
        ? value.exchange.trim().toUpperCase()
        : '',
    instrument_type:
      typeof value.instrument_type === 'string' && value.instrument_type.trim()
        ? value.instrument_type.trim()
        : '',
    country:
      typeof value.country === 'string' && value.country.trim()
        ? value.country.trim().toUpperCase()
        : '',
  }
}

function loadRecentSelections() {
  if (typeof window === 'undefined') {
    return []
  }

  try {
    const raw = window.localStorage.getItem(RECENT_SELECTIONS_KEY)
    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.map(normalizeInstrument).filter(Boolean) : []
  } catch {
    return []
  }
}

function persistRecentSelections(items) {
  if (typeof window === 'undefined') {
    return
  }

  window.localStorage.setItem(RECENT_SELECTIONS_KEY, JSON.stringify(items))
}

function formatApiError(error, fallback) {
  if (typeof error?.response?.data?.error === 'string' && error.response.data.error.trim()) {
    return error.response.data.error
  }

  if (typeof error?.response?.data?.detail === 'string' && error.response.data.detail.trim()) {
    return error.response.data.detail
  }

  if (typeof error?.message === 'string' && error.message.trim()) {
    return error.message
  }

  return fallback
}

function getCurrencyCode(input) {
  const instrument = normalizeInstrument(input)
  if (instrument?.market === 'IN') {
    return 'INR'
  }
  if (instrument?.market === 'US') {
    return 'USD'
  }

  const symbol = typeof input === 'string' ? input.trim().toUpperCase() : ''
  if (
    symbol.endsWith('.NS') ||
    symbol.endsWith('.BO') ||
    symbol.startsWith('^NSE') ||
    symbol.startsWith('^BSE') ||
    symbol.startsWith('^CNX')
  ) {
    return 'INR'
  }

  return 'USD'
}

function toPositiveNumber(value) {
  const numericValue = Number(value)
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return null
  }

  return numericValue
}

function toSignedNumber(value) {
  const numericValue = Number(value)
  return Number.isFinite(numericValue) ? numericValue : null
}

function dedupeRecentAnalyses(...collections) {
  const byNormalized = new Map()

  for (const collection of collections) {
    for (const row of Array.isArray(collection) ? collection : []) {
      const instrument = normalizeInstrument(row)
      if (!instrument?.normalized || byNormalized.has(instrument.normalized)) {
        continue
      }

      byNormalized.set(instrument.normalized, {
        ...row,
        ...instrument,
      })
    }
  }

  return [...byNormalized.values()]
}

function StatePanel({ title, message, action = null, compact = false }) {
  return (
    <section className={`state-panel${compact ? ' is-compact' : ''}`}>
      <div>
        <strong>{title}</strong>
        <p>{message}</p>
      </div>
      {action}
    </section>
  )
}

function MetricCard({ eyebrow, label, value, tone = 'neutral', note }) {
  return (
    <article className={`metric-card tone-${tone}`}>
      {eyebrow ? <span className="metric-card-eyebrow">{eyebrow}</span> : null}
      <span className="metric-card-label">{label}</span>
      <strong>{value}</strong>
      {note ? <small>{note}</small> : null}
    </article>
  )
}

function SectionCard({ kicker, title, caption, action = null, children }) {
  return (
    <section className="section-card">
      <header className="section-card-header">
        <div>
          {kicker ? <p className="section-card-kicker">{kicker}</p> : null}
          {title ? <h2>{title}</h2> : null}
          {caption ? <p className="section-card-caption">{caption}</p> : null}
        </div>
        {action}
      </header>
      {children}
    </section>
  )
}

function SessionBadge({ session }) {
  const status = String(session?.session_status || 'closed').replace(/-/g, ' ')

  return (
    <div className={`session-badge status-${String(session?.session_status || 'closed')}`}>
      <strong>{session?.label || session?.market || 'Market'}</strong>
      <span>{status}</span>
      <small>
        {session?.local_time || '--:--'} · {session?.opens_at || '--:--'} to{' '}
        {session?.closes_at || '--:--'}
      </small>
    </div>
  )
}

function InstrumentContextBar({ instrument, quote, quoteLoading, quoteError, onAnalyze, onTrade, analyzing }) {
  if (!instrument) {
    return (
      <section className="instrument-context-bar is-empty">
        <div>
          <span className="section-card-kicker">Active Instrument</span>
          <strong>No instrument locked</strong>
          <p>Use the search bar to lock a US or India stock/index. Quote context then stays pinned on every screen.</p>
        </div>
      </section>
    )
  }

  const changeTone =
    Number(quote?.day_change_pct) > 0 ? 'positive' : Number(quote?.day_change_pct) < 0 ? 'negative' : 'neutral'

  return (
    <section className="instrument-context-bar">
      <div className="instrument-context-main">
        <span className="section-card-kicker">Active Instrument</span>
        <div className="instrument-context-heading">
          <strong>{instrument.display_name}</strong>
          <span>{instrument.normalized}</span>
        </div>
        <p>
          {instrument.market === 'IN' ? 'India' : 'US'} · {instrument.exchange} ·{' '}
          {instrument.instrument_type || 'Instrument'}
        </p>
      </div>

      <div className="instrument-context-metrics">
        <MetricCard
          label="Current price"
          value={
            quoteLoading
              ? 'Loading...'
              : formatMoney(quote?.current_price, quote?.currency || getCurrencyCode(instrument))
          }
        />
        <MetricCard
          label="Day change"
          value={quoteLoading ? 'Loading...' : formatPercent(quote?.day_change_pct, 2, { signed: true })}
          note={
            quoteLoading || quote?.day_change == null
              ? undefined
              : formatSignedMoney(quote.day_change, quote.currency || getCurrencyCode(instrument))
          }
          tone={changeTone}
        />
        <MetricCard
          label="5-day trend"
          value={quoteLoading ? 'Loading...' : formatPercent(quote?.trend_5d_pct, 2, { signed: true })}
          tone={Number(quote?.trend_5d_pct) > 0 ? 'positive' : Number(quote?.trend_5d_pct) < 0 ? 'negative' : 'neutral'}
        />
        <MetricCard
          label="30-day trend"
          value={quoteLoading ? 'Loading...' : formatPercent(quote?.trend_30d_pct, 2, { signed: true })}
          tone={Number(quote?.trend_30d_pct) > 0 ? 'positive' : Number(quote?.trend_30d_pct) < 0 ? 'negative' : 'neutral'}
        />
      </div>

      <div className="instrument-context-actions">
        <button type="button" className="secondary-button" onClick={onAnalyze} disabled={analyzing}>
          {analyzing ? 'Running...' : 'Run analysis'}
        </button>
        <button type="button" className="primary-button" onClick={onTrade}>
          Paper Trade
        </button>
      </div>

      {quoteError ? <p className="inline-error">{quoteError}</p> : null}
    </section>
  )
}

function PositionsTable({ holdings, baseCurrency, onUseInstrument }) {
  const rows = Array.isArray(holdings) ? holdings : []

  if (rows.length === 0) {
    return (
      <StatePanel
        title="No positions yet"
        message="Book a paper trade to create the first ledger-backed position."
        compact
      />
    )
  }

  return (
    <div className="table-shell">
      <table className="data-table">
        <thead>
          <tr>
            <th>Instrument</th>
            <th>Side</th>
            <th>Qty</th>
            <th>Avg price</th>
            <th>Market value</th>
            <th>P&amp;L</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((holding) => (
            <tr key={holding.id || holding.ticker}>
              <td>
                <div className="table-primary-cell">
                  <strong>{holding.display_name || holding.ticker}</strong>
                  <span>{holding.ticker}</span>
                </div>
              </td>
              <td>
                <span className={`side-pill side-${String(holding.side || '').toLowerCase()}`}>
                  {holding.side || 'N/A'}
                </span>
              </td>
              <td>{holding.quantity ?? 'N/A'}</td>
              <td>{formatMoney(holding.avg_price, holding.instrument_currency || baseCurrency)}</td>
              <td>{formatMoney(holding.current_value, baseCurrency)}</td>
              <td className={Number(holding.profit_loss) >= 0 ? 'value-positive' : 'value-negative'}>
                {formatSignedMoney(holding.profit_loss, baseCurrency)} ·{' '}
                {formatPercent(holding.profit_loss_percent, 2, { signed: true })}
              </td>
              <td>
                <button
                  type="button"
                  className="table-button"
                  onClick={() => onUseInstrument(normalizeInstrument(holding))}
                >
                  Lock
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TransactionsTable({ transactions, baseCurrency }) {
  const rows = Array.isArray(transactions) ? transactions : []

  if (rows.length === 0) {
    return (
      <StatePanel
        title="No ledger entries yet"
        message="New paper trades and portfolio adjustments will show up here."
        compact
      />
    )
  }

  return (
    <div className="table-shell">
      <table className="data-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Instrument</th>
            <th>Side</th>
            <th>Qty</th>
            <th>Price</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((trade) => (
            <tr key={trade.id}>
              <td>{new Date(trade.occurred_at).toLocaleString()}</td>
              <td>
                <div className="table-primary-cell">
                  <strong>{trade.display_name || trade.ticker}</strong>
                  <span>{trade.ticker}</span>
                </div>
              </td>
              <td>
                <span className={`side-pill side-${String(trade.side || '').toLowerCase()}`}>
                  {trade.side}
                </span>
              </td>
              <td>{trade.quantity}</td>
              <td>{formatMoney(trade.price_base || trade.price, baseCurrency)}</td>
              <td>{trade.source || 'manual'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TradeTicketForm({ instrument, quote, onSubmit, submitting }) {
  const [side, setSide] = useState('BUY')
  const [quantity, setQuantity] = useState('')
  const [manualPrice, setManualPrice] = useState('')
  const [note, setNote] = useState('')
  const [error, setError] = useState(null)
  const quotedPrice =
    Number.isFinite(Number(quote?.current_price)) && Number(quote.current_price) > 0
      ? String(Number(quote.current_price))
      : ''
  const effectivePrice = manualPrice || quotedPrice

  async function handleSubmit(event) {
    event.preventDefault()

    if (!instrument?.normalized) {
      setError('Choose an instrument before booking a paper trade.')
      return
    }

    const normalizedQuantity = toPositiveNumber(quantity)
    const normalizedPrice = toPositiveNumber(effectivePrice)

    if (normalizedQuantity === null) {
      setError('Quantity must be a positive number.')
      return
    }

    if (normalizedPrice === null) {
      setError('Price must be a positive number.')
      return
    }

    setError(null)

    try {
      await onSubmit({
        ticker: instrument.normalized,
        side,
        quantity: normalizedQuantity,
        price: normalizedPrice,
        note,
      })
      setQuantity('')
      setManualPrice('')
      setNote('')
    } catch (submitError) {
      setError(submitError?.message || 'Unable to book the paper trade.')
    }
  }

  return (
    <form className="ticket-form" onSubmit={handleSubmit}>
      <div className="ticket-form-header">
        <div>
          <span className="section-card-kicker">Paper Trade</span>
          <h3>{instrument?.display_name || 'No instrument selected'}</h3>
        </div>
        <span>{instrument?.normalized || 'Lock an instrument first'}</span>
      </div>

      <div className="ticket-grid">
        <label>
          <span>Side</span>
          <select value={side} onChange={(event) => setSide(event.target.value)} disabled={submitting}>
            <option value="BUY">BUY</option>
            <option value="SELL">SELL</option>
            <option value="SHORT">SHORT</option>
            <option value="COVER">COVER</option>
          </select>
        </label>
        <label>
          <span>Quantity</span>
          <input
            type="number"
            min="0"
            step="any"
            value={quantity}
            onChange={(event) => setQuantity(event.target.value)}
            placeholder="10"
            disabled={submitting}
          />
        </label>
        <label>
          <span>Price</span>
          <input
            type="number"
            min="0"
            step="any"
            value={effectivePrice}
            onChange={(event) => setManualPrice(event.target.value)}
            placeholder="Market reference"
            disabled={submitting}
          />
        </label>
        <label className="ticket-grid-span">
          <span>Execution note</span>
          <input
            type="text"
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder="Why this paper trade is being placed"
            disabled={submitting}
          />
        </label>
      </div>

      {error ? <p className="inline-error">{error}</p> : null}

      <button type="submit" className="primary-button" disabled={submitting}>
        {submitting ? 'Booking...' : 'Book paper trade'}
      </button>
    </form>
  )
}

function AdjustmentTicket({ instrument, onSubmit, submitting }) {
  const [targetQuantity, setTargetQuantity] = useState('')
  const [price, setPrice] = useState('')
  const [note, setNote] = useState('')
  const [error, setError] = useState(null)

  async function handleSubmit(event) {
    event.preventDefault()

    if (!instrument?.normalized) {
      setError('Lock an instrument before adjusting the position.')
      return
    }

    const normalizedTargetQuantity = toSignedNumber(targetQuantity)
    if (normalizedTargetQuantity === null) {
      setError('Target quantity must be a valid signed number.')
      return
    }

    const payload = {
      target_quantity: normalizedTargetQuantity,
      note,
    }

    const normalizedPrice = toPositiveNumber(price)
    if (normalizedPrice !== null) {
      payload.price = normalizedPrice
    }

    setError(null)

    try {
      await onSubmit(instrument.normalized, payload)
      setTargetQuantity('')
      setPrice('')
      setNote('')
    } catch (submitError) {
      setError(submitError?.message || 'Unable to create adjustment entries.')
    }
  }

  return (
    <form className="ticket-form is-secondary" onSubmit={handleSubmit}>
      <div className="ticket-form-header">
        <div>
          <span className="section-card-kicker">Position Adjustment</span>
          <h3>{instrument?.display_name || 'No instrument selected'}</h3>
        </div>
        <span>{instrument?.normalized || 'Lock an instrument first'}</span>
      </div>

      <div className="ticket-grid">
        <label>
          <span>Target signed qty</span>
          <input
            type="number"
            step="any"
            value={targetQuantity}
            onChange={(event) => setTargetQuantity(event.target.value)}
            placeholder="0 to flatten, -5 to go short"
            disabled={submitting}
          />
        </label>
        <label>
          <span>Reference price</span>
          <input
            type="number"
            min="0"
            step="any"
            value={price}
            onChange={(event) => setPrice(event.target.value)}
            placeholder="Optional"
            disabled={submitting}
          />
        </label>
        <label className="ticket-grid-span">
          <span>Reason</span>
          <input
            type="text"
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder="Why the ledger needs a correction"
            disabled={submitting}
          />
        </label>
      </div>

      {error ? <p className="inline-error">{error}</p> : null}

      <button type="submit" className="secondary-button" disabled={submitting}>
        {submitting ? 'Applying...' : 'Create adjustment entries'}
      </button>
    </form>
  )
}

function PaperTradeDrawer({ open, instrument, quote, submitting, onClose, onSubmit }) {
  useEffect(() => {
    if (!open) {
      return undefined
    }

    function handleKeyDown(event) {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [onClose, open])

  if (!open) {
    return null
  }

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="trade-drawer" onClick={(event) => event.stopPropagation()}>
        <div className="trade-drawer-header">
          <div>
            <span className="section-card-kicker">Global Paper Trade</span>
            <h2>Book the order</h2>
            <p>The ticket is prefilled from the active instrument. Successful orders refresh the portfolio and command center immediately.</p>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close trade drawer">
            ×
          </button>
        </div>

        <div className="trade-drawer-context">
          <MetricCard label="Instrument" value={instrument?.normalized || 'None'} />
          <MetricCard
            label="Current price"
            value={formatMoney(quote?.current_price, quote?.currency || getCurrencyCode(instrument))}
          />
          <MetricCard
            label="Day change"
            value={formatPercent(quote?.day_change_pct, 2, { signed: true })}
            tone={Number(quote?.day_change_pct) > 0 ? 'positive' : Number(quote?.day_change_pct) < 0 ? 'negative' : 'neutral'}
          />
        </div>

        <TradeTicketForm instrument={instrument} quote={quote} onSubmit={onSubmit} submitting={submitting} />
      </aside>
    </div>
  )
}

function ValidationPanel({ instrument, validation, loading, error, onRunValidation }) {
  if (!instrument) {
    return (
      <StatePanel
        title="Validation needs an instrument"
        message="Lock a supported instrument first. Validation then runs the intraday walk-forward replay for the active market model from the product UI."
        compact
      />
    )
  }

  return (
    <SectionCard
      kicker="Validation"
      title="Empirical check"
      caption="This reports intraday trade evidence first, then calibration diagnostics as secondary evidence. Promotion status, cost stress, and sample quality are shown explicitly."
      action={
        <button type="button" className="secondary-button" onClick={onRunValidation} disabled={loading}>
          {loading ? 'Running...' : 'Run validation'}
        </button>
      }
    >
      {error ? <p className="inline-error">{error}</p> : null}

      {validation ? (
        <div className="validation-stack">
          <div className="metric-grid metric-grid-four">
            <MetricCard label="Trade count" value={String(validation.trade_metrics?.trade_count || 0)} />
            <MetricCard
              label="Net expectancy"
              value={Number(validation.trade_metrics?.net_expectancy ?? 0).toFixed(4)}
            />
            <MetricCard label="Profit factor" value={Number(validation.trade_metrics?.profit_factor ?? 0).toFixed(2)} />
            <MetricCard label="Wilson lower bound" value={formatPercent(validation.trade_metrics?.wilson_lower_bound, 2, { scale: 100 })} />
          </div>

          <div className="validation-period">
            <strong>
              {validation.period?.start_date} to {validation.period?.end_date}
            </strong>
            <span>{validation.sample_quality?.execution_assumption || `Horizon: ${validation.period?.horizon || 1}`}</span>
          </div>

          <div className="content-grid content-grid-two">
            <div className="validation-card">
              <h3>Promotion gate</h3>
              <div className="validation-list">
                <div className="validation-list-row">
                  <span>Status</span>
                  <strong>{validation.promotion_gate?.passed ? 'Passed' : 'Blocked'}</strong>
                </div>
                <div className="validation-list-row">
                  <span>Reason</span>
                  <strong>{validation.promotion_gate?.reason || 'Not returned'}</strong>
                </div>
                <div className="validation-list-row">
                  <span>Artifact time</span>
                  <strong>{validation.promotion_gate?.artifact_timestamp || 'Not returned'}</strong>
                </div>
              </div>
            </div>

            <div className="validation-card">
              <h3>Sample quality</h3>
              <div className="validation-list">
                <div className="validation-list-row">
                  <span>Total sessions</span>
                  <strong>{validation.sample_quality?.total_sessions ?? 0}</strong>
                </div>
                <div className="validation-list-row">
                  <span>Eligible sessions</span>
                  <strong>{validation.sample_quality?.eligible_sessions ?? 0}</strong>
                </div>
                <div className="validation-list-row">
                  <span>Traded sessions</span>
                  <strong>{validation.sample_quality?.traded_sessions ?? 0}</strong>
                </div>
                <div className="validation-list-row">
                  <span>Skipped sessions</span>
                  <strong>{validation.sample_quality?.skipped_sessions ?? 0}</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="content-grid content-grid-two">
            <div className="validation-card">
              <h3>Cost assumptions</h3>
              <div className="validation-list">
                <div className="validation-list-row">
                  <span>Stress multiplier</span>
                  <strong>{Number(validation.cost_assumptions?.stress_cost_multiplier ?? 0).toFixed(2)}x</strong>
                </div>
                <div className="validation-list-row">
                  <span>Round-trip cost</span>
                  <strong>{Number(validation.cost_assumptions?.round_trip_cost_r ?? 0).toFixed(4)}R</strong>
                </div>
                <div className="validation-list-row">
                  <span>Stressed cost</span>
                  <strong>{Number(validation.cost_assumptions?.stressed_round_trip_cost_r ?? 0).toFixed(4)}R</strong>
                </div>
                <div className="validation-list-row">
                  <span>Survivorship note</span>
                  <strong>{validation.sample_quality?.survivorship_note || 'Not returned'}</strong>
                </div>
              </div>
            </div>

            <div className="validation-card">
              <h3>Regime breakdown</h3>
              <div className="validation-list">
                {Object.entries(validation.regime_breakdown?.volatility || {}).map(([bucket, summary]) => (
                  <div key={bucket} className="validation-list-row">
                    <span>{bucket}</span>
                    <strong>
                      {summary.trade_count || 0} trades · {Number(summary.net_expectancy ?? 0).toFixed(4)}R
                    </strong>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="content-grid content-grid-two">
            <div className="validation-card">
              <h3>Accuracy by confidence</h3>
              <div className="validation-list">
                {Object.entries(validation.accuracy_by_confidence || {}).map(([bucket, score]) => (
                  <div key={bucket} className="validation-list-row">
                    <span>{bucket}</span>
                    <strong>{formatPercent(score, 2, { scale: 100 })}</strong>
                  </div>
                ))}
              </div>
            </div>

            <div className="validation-card">
              <h3>Reliability curve</h3>
              <div className="validation-list">
                {(Array.isArray(validation.reliability_curve) ? validation.reliability_curve : []).map((point) => (
                  <div
                    key={`${point.probability_mean}-${point.count}`}
                    className="validation-list-row"
                  >
                    <span>
                      Pred {formatPercent(point.probability_mean, 1, { scale: 100 })}
                    </span>
                    <strong>
                      Actual {formatPercent(point.accuracy, 1, { scale: 100 })} · n={point.count}
                    </strong>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="metric-grid metric-grid-four">
            <MetricCard label="Accuracy" value={formatPercent(validation.accuracy, 2, { scale: 100 })} />
            <MetricCard label="ECE" value={Number(validation.ece ?? 0).toFixed(4)} />
            <MetricCard label="Brier score" value={Number(validation.brier_score ?? 0).toFixed(4)} />
            <MetricCard label="Total predictions" value={String(validation.total_predictions || 0)} />
          </div>
        </div>
      ) : (
        <StatePanel
          title="No validation run yet"
          message="Run the validation pass to see intraday trade evidence, promotion status, stressed-cost assumptions, and calibration diagnostics for the active instrument."
          compact
        />
      )}
    </SectionCard>
  )
}

function TodayPage({
  commandCenter,
  recentAnalyses,
  selectedInstrument,
  quote,
  onAnalyze,
  analyzing,
  onUseInstrument,
  onOpenTradeDrawer,
  baseCurrency,
}) {
  const summary = commandCenter?.portfolio_summary || EMPTY_PORTFOLIO.summary
  const dailyBrief = commandCenter?.daily_brief || EMPTY_COMMAND_CENTER.daily_brief
  const topAction = commandCenter?.top_portfolio_action || 'No urgent portfolio action'
  const marketIntelligence = commandCenter?.market_intelligence || EMPTY_COMMAND_CENTER.market_intelligence

  return (
    <div className="page-stack">
      <section className="hero-card">
        <div>
          <span className="section-card-kicker">Operator workflow</span>
          <h1>Pick symbol, inspect quote, analyze, place paper trade, review the book.</h1>
          <p>
            TradeSense is now built for one job: fast decision support for an active trader. The
            active instrument stays pinned, the quote stays visible, and the paper-trade ticket is
            reachable from every screen.
          </p>
        </div>
        <div className="hero-actions">
          <button type="button" className="primary-button" onClick={onAnalyze} disabled={!selectedInstrument || analyzing}>
            {analyzing ? 'Running analysis...' : 'Run analysis'}
          </button>
          <button type="button" className="secondary-button" onClick={onOpenTradeDrawer} disabled={!selectedInstrument}>
            Paper Trade
          </button>
        </div>
      </section>

      <div className="metric-grid metric-grid-four">
        <MetricCard label="Portfolio value" value={formatMoney(summary.total_portfolio_value, baseCurrency)} />
        <MetricCard label="Gross exposure" value={formatMoney(summary.total_gross_exposure, baseCurrency)} />
        <MetricCard label="Net exposure" value={formatMoney(summary.total_net_exposure, baseCurrency)} />
        <MetricCard label="Open positions" value={String(summary.active_positions || 0)} />
      </div>

      <div className="content-grid content-grid-two">
        <SectionCard
          kicker="Daily brief"
          title={dailyBrief.headline || 'Awaiting session brief'}
          caption={commandCenter?.risk_headline || 'Portfolio and session risk posture will appear here as data refreshes.'}
        >
          <ul className="bullet-list">
            {(Array.isArray(dailyBrief.bullets) ? dailyBrief.bullets : []).map((bullet) => (
              <li key={bullet}>{bullet}</li>
            ))}
          </ul>
          <div className="top-action-banner">
            <span>Top portfolio action</span>
            <strong>{topAction}</strong>
          </div>
        </SectionCard>

        <SectionCard
          kicker="Market sessions"
          title="What is open right now?"
          caption="This stays practical: open/closed state, local time, and session windows."
        >
          <div className="session-badge-grid">
            {(Array.isArray(commandCenter.market_sessions) ? commandCenter.market_sessions : []).map((session) => (
              <SessionBadge key={session.market || session.label} session={session} />
            ))}
            {!commandCenter.market_sessions?.length ? (
              <StatePanel title="No session data" message="Session status will populate when the command center refresh completes." compact />
            ) : null}
          </div>
        </SectionCard>
      </div>

      <div className="content-grid content-grid-two">
        <SectionCard
          kicker="Recent analysis"
          title="Reopen a symbol fast"
          caption="No fake defaults. Only your recent analyses and selected symbols appear here."
        >
          {recentAnalyses.length > 0 ? (
            <div className="signal-list">
              {recentAnalyses.slice(0, 8).map((signal) => (
                <button
                  key={signal.id || signal.normalized}
                  type="button"
                  className="signal-card"
                  onClick={() => onUseInstrument(signal)}
                >
                  <div>
                    <strong>{signal.display_name || signal.normalized}</strong>
                    <span>{signal.normalized}</span>
                  </div>
                  <div className="signal-card-meta">
                    <span className={`side-pill side-${String(signal.signal || '').toLowerCase()}`}>
                      {signal.signal || 'Stored'}
                    </span>
                    <small>{signal.confidence_level || signal.exchange || 'Ready to reopen'}</small>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <StatePanel
              title="No recent analysis"
              message="Run your first symbol through Analysis and it will show up here for quick reopen."
              compact
            />
          )}
        </SectionCard>

        <SectionCard
          kicker="Live context"
          title="Pinned quote posture"
          caption="The active instrument price context is always visible, so operations do not depend on remembering a prior screen."
        >
          <div className="metric-grid metric-grid-two">
            <MetricCard
              label="Current price"
              value={formatMoney(quote?.current_price, quote?.currency || getCurrencyCode(selectedInstrument))}
            />
            <MetricCard
              label="Day change"
              value={formatPercent(quote?.day_change_pct, 2, { signed: true })}
              note={formatSignedMoney(quote?.day_change, quote?.currency || getCurrencyCode(selectedInstrument))}
              tone={Number(quote?.day_change_pct) > 0 ? 'positive' : Number(quote?.day_change_pct) < 0 ? 'negative' : 'neutral'}
            />
            <MetricCard
              label="5-day trend"
              value={formatPercent(quote?.trend_5d_pct, 2, { signed: true })}
              tone={Number(quote?.trend_5d_pct) > 0 ? 'positive' : Number(quote?.trend_5d_pct) < 0 ? 'negative' : 'neutral'}
            />
            <MetricCard
              label="30-day trend"
              value={formatPercent(quote?.trend_30d_pct, 2, { signed: true })}
              tone={Number(quote?.trend_30d_pct) > 0 ? 'positive' : Number(quote?.trend_30d_pct) < 0 ? 'negative' : 'neutral'}
            />
          </div>
        </SectionCard>
      </div>

      <SectionCard
        kicker="Intelligence"
        title="Company and macro headlines"
        caption="Still honest: this is only what the command center has, not invented editorial filler."
      >
        <div className="headline-grid">
          {[
            ...(Array.isArray(marketIntelligence.companyHeadlines) ? marketIntelligence.companyHeadlines : []),
            ...(Array.isArray(marketIntelligence.macroHeadlines) ? marketIntelligence.macroHeadlines : []),
          ]
            .slice(0, 6)
            .map((headline, index) => (
              <article key={`${headline.title || headline}-${index}`} className="headline-card">
                <strong>{headline.title || headline}</strong>
                {headline.summary ? <p>{headline.summary}</p> : null}
              </article>
            ))}
          {!marketIntelligence.companyHeadlines?.length && !marketIntelligence.macroHeadlines?.length ? (
            <StatePanel
              title="No headline stack yet"
              message="Headline context will appear here once the upstream feeds populate."
              compact
            />
          ) : null}
        </div>
      </SectionCard>
    </div>
  )
}

function AnalysisPage({
  selectedInstrument,
  quote,
  analysis,
  analysisError,
  analyzing,
  validation,
  validationError,
  validationLoading,
  onAnalyze,
  onRunValidation,
  recentAnalyses,
  onUseInstrument,
}) {
  if (!selectedInstrument) {
    return (
      <StatePanel
        title="Start with a symbol"
        message="Search for any supported US or India stock/index in the top bar. Once locked, Analysis shows the quote, decision state, and validation panel in one workspace."
      />
    )
  }

  const currencyCode = quote?.currency || getCurrencyCode(selectedInstrument)
  const noTrade = analysis?.signal === 'NO_TRADE' || analysis?.actionability_state === 'blocked'
  const watchlist = analysis?.signal === 'WATCHLIST' || analysis?.actionability_state === 'monitor'

  return (
    <div className="page-stack">
      <div className="content-grid content-grid-two">
        <SectionCard
          kicker="Decision"
          title={analysis?.decision_label || 'No analysis run yet'}
          caption={
            noTrade
              ? analysis?.no_trade_reason || analysis?.signal_explanation || 'The model blocked the setup.'
              : watchlist
                ? analysis?.signal_explanation || analysis?.no_trade_reason || 'The setup is on watchlist until one more condition clears.'
              : analysis?.signal_explanation || 'Run analysis to build the decision state for this symbol.'
          }
          action={
            <button type="button" className="primary-button" onClick={onAnalyze} disabled={analyzing}>
              {analyzing ? 'Running...' : 'Run analysis'}
            </button>
          }
        >
          {analysisError ? <p className="inline-error">{analysisError}</p> : null}
          <div className="metric-grid metric-grid-four">
            <MetricCard
              label="Current price"
              value={formatMoney(quote?.current_price, currencyCode)}
            />
            <MetricCard
              label="Probability"
              value={analysis ? formatPercent(analysis.probability, 2, { scale: 100 }) : 'N/A'}
            />
            <MetricCard label="Confidence" value={analysis?.confidence_level || 'N/A'} />
            <MetricCard label="State" value={analysis?.actionability_state || analysis?.model_name || 'N/A'} />
          </div>

          {analysis ? (
            <div className="content-grid content-grid-two">
              <div className="analysis-card">
                <h3>{noTrade ? 'Execution blockers' : watchlist ? 'Watchlist trigger' : 'Trade plan'}</h3>
                <div className="validation-list">
                  <div className="validation-list-row">
                    <span>Entry</span>
                    <strong>{formatMoney(analysis.entry_price, currencyCode)}</strong>
                  </div>
                  <div className="validation-list-row">
                    <span>Stop</span>
                    <strong>{formatMoney(analysis.stop_price, currencyCode)}</strong>
                  </div>
                  <div className="validation-list-row">
                    <span>Take profit</span>
                    <strong>{formatMoney(analysis.take_profit_price, currencyCode)}</strong>
                  </div>
                  <div className="validation-list-row">
                    <span>Forced exit</span>
                    <strong>{analysis.forced_exit_time || 'N/A'}</strong>
                  </div>
                  <div className="validation-list-row">
                    <span>Live threshold</span>
                    <strong>{Number(analysis.effective_threshold ?? analysis.threshold ?? 0).toFixed(4)}</strong>
                  </div>
                  <div className="validation-list-row">
                    <span>Threshold gap</span>
                    <strong>
                      {analysis.threshold_gap === null || analysis.threshold_gap === undefined
                        ? 'N/A'
                        : Number(analysis.threshold_gap).toFixed(4)}
                    </strong>
                  </div>
                </div>
              </div>

              <div className="analysis-card">
                <h3>{noTrade ? 'Observed context' : watchlist ? 'What unlocks the trade' : 'Why the setup exists'}</h3>
                <div className="analysis-note-list">
                  <article>
                    <span>Trend summary</span>
                    <strong>{analysis.trend_summary || 'Not returned'}</strong>
                  </article>
                  <article>
                    <span>Risk summary</span>
                    <strong>{analysis.risk_summary || 'Not returned'}</strong>
                  </article>
                  <article>
                    <span>Model honesty</span>
                    <strong>{analysis.model_honesty || 'Not returned'}</strong>
                  </article>
                  <article>
                    <span>Sentiment gate</span>
                    <strong>{analysis.sentiment_gate_reason || 'Not returned'}</strong>
                  </article>
                  <article>
                    <span>Decision reason</span>
                    <strong>{analysis.decision_reason_type || 'Actionable setup'}</strong>
                  </article>
                  <article>
                    <span>Promotion gate</span>
                    <strong>{analysis.promotion_gate?.reason || 'Not returned'}</strong>
                  </article>
                </div>
              </div>
            </div>
          ) : (
            <StatePanel
              title="Analysis is ready"
              message="Run this symbol to populate the decision state. If the model rejects the setup, the blockers stay on this same screen rather than pushing you into a dead-end route."
              compact
            />
          )}
        </SectionCard>

        <SectionCard
          kicker="Price action"
          title="Thirty-day chart"
          caption="The chart stays in analysis, while the pinned quote above keeps the latest price visible across the whole app."
        >
          <StockPriceChart symbol={selectedInstrument.normalized} />
        </SectionCard>
      </div>

      <ValidationPanel
        instrument={selectedInstrument}
        validation={validation}
        loading={validationLoading}
        error={validationError}
        onRunValidation={onRunValidation}
      />

      <SectionCard
        kicker="Recent symbols"
        title="Reopen recent analysis"
        caption="Use this to jump between validated symbols without losing the operator flow."
      >
        {recentAnalyses.length > 0 ? (
          <div className="signal-list">
            {recentAnalyses.slice(0, 8).map((signal) => (
              <button
                key={signal.id || signal.normalized}
                type="button"
                className="signal-card"
                onClick={() => onUseInstrument(signal)}
              >
                <div>
                  <strong>{signal.display_name || signal.normalized}</strong>
                  <span>{signal.normalized}</span>
                </div>
                <div className="signal-card-meta">
                  <span className={`side-pill side-${String(signal.signal || '').toLowerCase()}`}>
                    {signal.signal || 'Stored'}
                  </span>
                  <small>{signal.confidence_level || signal.exchange || 'Ready to reopen'}</small>
                </div>
              </button>
            ))}
          </div>
        ) : (
          <StatePanel title="No recent analysis" message="Recent symbols will appear here after the first run." compact />
        )}
      </SectionCard>
    </div>
  )
}

function PortfolioPage({
  portfolio,
  history,
  insights,
  advisor,
  transactions,
  selectedInstrument,
  onUseInstrument,
  onAdjustSubmit,
  adjustSubmitting,
  onOpenTradeDrawer,
  baseCurrency,
  loading,
  error,
}) {
  const analytics = useMemo(
    () => computePortfolioAnalytics(portfolio.holdings, history, insights),
    [portfolio.holdings, history, insights]
  )
  const advisorMessages = useMemo(
    () => buildPortfolioAdvisorInsights(analytics, advisor.recommendations),
    [advisor.recommendations, analytics]
  )

  if (loading) {
    return <StatePanel title="Loading portfolio" message="Refreshing positions, ledger, and risk views." />
  }

  if (error) {
    return <StatePanel title="Portfolio unavailable" message={error} />
  }

  return (
    <div className="page-stack">
      <section className="hero-card compact">
        <div>
          <span className="section-card-kicker">Portfolio workspace</span>
          <h1>{formatMoney(portfolio.summary.total_portfolio_value, baseCurrency)}</h1>
          <p>
            One workspace for positions, ledger, risk, allocation, and adjustments. No duplicate
            holdings screen and no separate intelligence route.
          </p>
        </div>
        <div className="hero-actions">
          <button type="button" className="primary-button" onClick={onOpenTradeDrawer}>
            Paper Trade
          </button>
        </div>
      </section>

      <div className="metric-grid metric-grid-six">
        <MetricCard label="Gross exposure" value={formatMoney(portfolio.summary.total_gross_exposure, baseCurrency)} />
        <MetricCard label="Net exposure" value={formatMoney(portfolio.summary.total_net_exposure, baseCurrency)} />
        <MetricCard label="Open positions" value={String(portfolio.summary.active_positions || 0)} />
        <MetricCard
          label="Largest position"
          value={insights?.largest_position?.ticker || 'N/A'}
          note={insights?.largest_position?.weight ? formatPercent(insights.largest_position.weight, 1) : undefined}
        />
        <MetricCard label="Concentration" value={insights?.concentration_risk || 'LOW'} />
        <MetricCard label="Volatility" value={insights?.volatility_level || 'LOW'} />
      </div>

      <div className="content-grid content-grid-two">
        <PortfolioEquityChart history={history} baseCurrency={baseCurrency} />
        <PortfolioAllocationChart holdings={portfolio.holdings} baseCurrency={baseCurrency} />
      </div>

      <div className="content-grid content-grid-two">
        <PortfolioInsights data={insights} metrics={analytics} baseCurrency={baseCurrency} />
        <PortfolioAdvisor data={{ recommendations: advisorMessages }} title="Portfolio advisor" />
      </div>

      <div className="content-grid content-grid-two">
        <SectionCard
          kicker="Positions"
          title="Live book"
          caption="Lock any row into the active instrument bar, then analyze or adjust it without changing screens."
        >
          <PositionsTable holdings={portfolio.holdings} baseCurrency={baseCurrency} onUseInstrument={onUseInstrument} />
        </SectionCard>

        <SectionCard
          kicker="Ledger"
          title="Transaction history"
          caption="The paper-trade ledger is the source of truth for positions and portfolio analytics."
        >
          <TransactionsTable transactions={transactions.transactions} baseCurrency={baseCurrency} />
        </SectionCard>
      </div>

      <SectionCard
        kicker="Adjustments"
        title="Reconcile the book"
        caption="Use signed target quantity updates when the ledger needs a correction rather than a new directional trade."
      >
        <AdjustmentTicket
          instrument={selectedInstrument}
          onSubmit={onAdjustSubmit}
          submitting={adjustSubmitting}
        />
      </SectionCard>
    </div>
  )
}

function AppShell() {
  const navigate = useNavigate()
  const [selectedInstrument, setSelectedInstrument] = useState(null)
  const [recentSelections, setRecentSelections] = useState(loadRecentSelections)
  const [quote, setQuote] = useState(null)
  const [quoteLoading, setQuoteLoading] = useState(false)
  const [quoteError, setQuoteError] = useState(null)
  const [commandCenter, setCommandCenter] = useState(EMPTY_COMMAND_CENTER)
  const [recentAnalyses, setRecentAnalyses] = useState([])
  const [portfolio, setPortfolio] = useState(EMPTY_PORTFOLIO)
  const [history, setHistory] = useState(EMPTY_HISTORY)
  const [insights, setInsights] = useState(EMPTY_INSIGHTS)
  const [advisor, setAdvisor] = useState(EMPTY_ADVISOR)
  const [transactions, setTransactions] = useState(EMPTY_TRANSACTIONS)
  const [analysis, setAnalysis] = useState(null)
  const [analysisError, setAnalysisError] = useState(null)
  const [validation, setValidation] = useState(null)
  const [validationError, setValidationError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [globalError, setGlobalError] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [validationLoading, setValidationLoading] = useState(false)
  const [tradeSubmitting, setTradeSubmitting] = useState(false)
  const [adjustSubmitting, setAdjustSubmitting] = useState(false)
  const [tradeDrawerOpen, setTradeDrawerOpen] = useState(false)
  const [flashMessage, setFlashMessage] = useState(null)

  const baseCurrency = portfolio.summary.base_currency || 'INR'
  const combinedRecentAnalyses = useMemo(
    () => dedupeRecentAnalyses(recentAnalyses, commandCenter?.recent_signals, recentSelections),
    [commandCenter?.recent_signals, recentAnalyses, recentSelections]
  )

  const rememberInstrument = useCallback((instrument) => {
    const normalizedInstrument = normalizeInstrument(instrument)
    if (!normalizedInstrument) {
      return
    }

    setRecentSelections((current) => {
      const next = [
        normalizedInstrument,
        ...current.filter((item) => item.normalized !== normalizedInstrument.normalized),
      ].slice(0, 8)
      persistRecentSelections(next)
      return next
    })
  }, [])

  const refreshPortfolioBundle = useCallback(async () => {
    const [portfolioPayload, historyPayload, insightsPayload, advisorPayload, transactionPayload] =
      await Promise.all([
        fetchPortfolio(),
        fetchPortfolioHistory(PORTFOLIO_DAYS),
        fetchPortfolioInsights(PORTFOLIO_DAYS),
        fetchPortfolioAdvisor(PORTFOLIO_DAYS),
        fetchPortfolioTransactions(),
      ])

    setPortfolio({
      ...EMPTY_PORTFOLIO,
      ...portfolioPayload,
      holdings: Array.isArray(portfolioPayload?.holdings) ? portfolioPayload.holdings : [],
      positions: Array.isArray(portfolioPayload?.positions) ? portfolioPayload.positions : [],
      summary: { ...EMPTY_PORTFOLIO.summary, ...(portfolioPayload?.summary || {}) },
    })
    setHistory({
      days: historyPayload?.days || PORTFOLIO_DAYS,
      equity_curve: Array.isArray(historyPayload?.equity_curve) ? historyPayload.equity_curve : [],
    })
    setInsights({ ...EMPTY_INSIGHTS, ...(insightsPayload || {}) })
    setAdvisor({
      recommendations: Array.isArray(advisorPayload?.recommendations)
        ? advisorPayload.recommendations
        : [],
    })
    setTransactions({
      transactions: Array.isArray(transactionPayload?.transactions)
        ? transactionPayload.transactions
        : [],
      summary: { ...EMPTY_TRANSACTIONS.summary, ...(transactionPayload?.summary || {}) },
    })
  }, [])

  const refreshCommandCenter = useCallback(async () => {
    const payload = await fetchCommandCenter()
    setCommandCenter({
      ...EMPTY_COMMAND_CENTER,
      ...payload,
      portfolio_summary: {
        ...EMPTY_COMMAND_CENTER.portfolio_summary,
        ...(payload?.portfolio_summary || {}),
      },
      daily_brief: { ...EMPTY_COMMAND_CENTER.daily_brief, ...(payload?.daily_brief || {}) },
      recent_signals: Array.isArray(payload?.recent_signals) ? payload.recent_signals : [],
      market_sessions: Array.isArray(payload?.market_sessions) ? payload.market_sessions : [],
      market_intelligence: {
        companyHeadlines: Array.isArray(payload?.market_intelligence?.companyHeadlines)
          ? payload.market_intelligence.companyHeadlines
          : [],
        macroHeadlines: Array.isArray(payload?.market_intelligence?.macroHeadlines)
          ? payload.market_intelligence.macroHeadlines
          : [],
      },
    })
  }, [])

  const refreshRecentAnalyses = useCallback(async () => {
    const payload = await fetchRecentAnalyses(8)
    setRecentAnalyses(Array.isArray(payload?.results) ? payload.results : [])
  }, [])

  const refreshAll = useCallback(async () => {
    setLoading(true)
    try {
      await Promise.all([refreshPortfolioBundle(), refreshCommandCenter(), refreshRecentAnalyses()])
      setGlobalError(null)
    } catch (error) {
      setGlobalError(formatApiError(error, 'Unable to load TradeSense data.'))
    } finally {
      setLoading(false)
    }
  }, [refreshCommandCenter, refreshPortfolioBundle, refreshRecentAnalyses])

  useEffect(() => {
    refreshAll()
  }, [refreshAll])

  useEffect(() => {
    if (!selectedInstrument?.normalized) {
      setQuote(null)
      setQuoteError(null)
      setQuoteLoading(false)
      return
    }

    let cancelled = false
    setQuoteLoading(true)
    setQuoteError(null)

    fetchQuoteSnapshot(selectedInstrument.normalized)
      .then((payload) => {
        if (!cancelled) {
          setQuote(payload)
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setQuote(null)
          setQuoteError(formatApiError(error, 'Unable to load the active quote snapshot.'))
        }
      })
      .finally(() => {
        if (!cancelled) {
          setQuoteLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [selectedInstrument?.normalized])

  const resolveInstrument = useCallback(
    async (instrument) => {
      const candidate = normalizeInstrument(instrument)
      const lookupValue =
        candidate?.normalized ||
        (typeof instrument === 'string' && instrument.trim() ? instrument.trim() : '')

      if (!lookupValue) {
        setSelectedInstrument(null)
        setQuote(null)
        setAnalysis(null)
        setValidation(null)
        return null
      }

      try {
        const payload = await normalizeSymbol(lookupValue)
        const resolved = normalizeInstrument(payload)

        if (!resolved) {
          throw new Error('Resolved instrument payload was empty.')
        }

        setSelectedInstrument(resolved)
        setAnalysis((current) =>
          current?.normalized === resolved.normalized || current?.symbol === resolved.normalized
            ? current
            : null
        )
        setValidation((current) => (current?.symbol === resolved.normalized ? current : null))
        setAnalysisError(null)
        setValidationError(null)
        rememberInstrument(resolved)
        return resolved
      } catch (error) {
        const message = formatApiError(error, 'Unable to lock the selected instrument.')
        setGlobalError(message)
        return null
      }
    },
    [rememberInstrument]
  )

  const runAnalysis = useCallback(async () => {
    const instrument = await resolveInstrument(selectedInstrument)
    if (!instrument?.normalized) {
      setAnalysisError('Choose a supported instrument before running analysis.')
      return
    }

    setAnalyzing(true)
    setAnalysisError(null)
    navigate('/analysis')

    try {
      const payload = await analyzeMarket(instrument.normalized)
      const normalizedPayload = normalizeInstrument(payload) || instrument
      const nextAnalysis = {
        ...payload,
        ...normalizedPayload,
      }
      setAnalysis(nextAnalysis)
      setSelectedInstrument(normalizedPayload)
      rememberInstrument(normalizedPayload)
      await Promise.all([refreshCommandCenter(), refreshRecentAnalyses()])
    } catch (error) {
      setAnalysisError(formatApiError(error, 'Unable to run analysis.'))
    } finally {
      setAnalyzing(false)
    }
  }, [navigate, refreshCommandCenter, refreshRecentAnalyses, rememberInstrument, resolveInstrument, selectedInstrument])

  const runValidation = useCallback(async () => {
    const instrument = await resolveInstrument(selectedInstrument)
    if (!instrument?.normalized) {
      setValidationError('Choose a supported instrument before running validation.')
      return
    }

    setValidationLoading(true)
    setValidationError(null)

    try {
      const payload = await fetchValidationReport(instrument.normalized)
      setValidation(payload)
      setSelectedInstrument(normalizeInstrument(payload) || instrument)
    } catch (error) {
      setValidationError(formatApiError(error, 'Unable to run validation.'))
    } finally {
      setValidationLoading(false)
    }
  }, [resolveInstrument, selectedInstrument])

  const handleTradeSubmit = useCallback(
    async (payload) => {
      setTradeSubmitting(true)
      try {
        await createPortfolioTrade(payload)
        await Promise.all([refreshPortfolioBundle(), refreshCommandCenter()])
        setTradeDrawerOpen(false)
        setFlashMessage(`Paper trade booked for ${payload.ticker}.`)
      } catch (error) {
        throw new Error(formatApiError(error, 'Unable to book the paper trade.'))
      } finally {
        setTradeSubmitting(false)
      }
    },
    [refreshCommandCenter, refreshPortfolioBundle]
  )

  const handleAdjustSubmit = useCallback(
    async (symbol, payload) => {
      setAdjustSubmitting(true)
      try {
        await adjustPortfolioPosition(symbol, payload)
        await Promise.all([refreshPortfolioBundle(), refreshCommandCenter()])
        setFlashMessage(`Portfolio adjustment posted for ${symbol}.`)
      } catch (error) {
        throw new Error(formatApiError(error, 'Unable to create adjustment entries.'))
      } finally {
        setAdjustSubmitting(false)
      }
    },
    [refreshCommandCenter, refreshPortfolioBundle]
  )

  useEffect(() => {
    if (!flashMessage) {
      return undefined
    }

    const timer = window.setTimeout(() => setFlashMessage(null), 3500)
    return () => window.clearTimeout(timer)
  }, [flashMessage])

  return (
    <div className="app-shell">
      <header className="shell-topbar">
        <div className="shell-topbar-left">
          <div className="brand-lockup">
            <strong>TradeSense</strong>
            <span>Active trader workspace</span>
          </div>
          <nav className="primary-nav" aria-label="Primary">
            {NAV_ITEMS.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.to === '/'} className="primary-nav-link">
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>

        <div className="shell-topbar-search">
          <InstrumentPicker
            value={selectedInstrument}
            onChange={resolveInstrument}
            recentSelections={recentSelections}
          />
        </div>

        <div className="shell-topbar-actions">
          <button type="button" className="secondary-button" onClick={runAnalysis} disabled={!selectedInstrument || analyzing}>
            {analyzing ? 'Running...' : 'Analyze'}
          </button>
          <button
            type="button"
            className="primary-button"
            onClick={() => setTradeDrawerOpen(true)}
            disabled={!selectedInstrument}
          >
            Paper Trade
          </button>
        </div>
      </header>

      <div className="shell-body">
        <InstrumentContextBar
          instrument={selectedInstrument}
          quote={quote}
          quoteLoading={quoteLoading}
          quoteError={quoteError}
          onAnalyze={runAnalysis}
          onTrade={() => setTradeDrawerOpen(true)}
          analyzing={analyzing}
        />

        {flashMessage ? <div className="flash-banner">{flashMessage}</div> : null}
        {globalError ? <div className="global-error">{globalError}</div> : null}

        <main className="shell-content">
          <Routes>
            <Route
              path="/"
              element={
                <TodayPage
                  commandCenter={commandCenter}
                  recentAnalyses={combinedRecentAnalyses}
                  selectedInstrument={selectedInstrument}
                  quote={quote}
                  onAnalyze={runAnalysis}
                  analyzing={analyzing}
                  onUseInstrument={resolveInstrument}
                  onOpenTradeDrawer={() => setTradeDrawerOpen(true)}
                  baseCurrency={baseCurrency}
                />
              }
            />
            <Route
              path="/analysis"
              element={
                <AnalysisPage
                  selectedInstrument={selectedInstrument}
                  quote={quote}
                  analysis={analysis}
                  analysisError={analysisError}
                  analyzing={analyzing}
                  validation={validation}
                  validationError={validationError}
                  validationLoading={validationLoading}
                  onAnalyze={runAnalysis}
                  onRunValidation={runValidation}
                  recentAnalyses={combinedRecentAnalyses}
                  onUseInstrument={resolveInstrument}
                />
              }
            />
            <Route
              path="/portfolio"
              element={
                <PortfolioPage
                  portfolio={portfolio}
                  history={history}
                  insights={insights}
                  advisor={advisor}
                  transactions={transactions}
                  selectedInstrument={selectedInstrument}
                  onUseInstrument={resolveInstrument}
                  onAdjustSubmit={handleAdjustSubmit}
                  adjustSubmitting={adjustSubmitting}
                  onOpenTradeDrawer={() => setTradeDrawerOpen(true)}
                  baseCurrency={baseCurrency}
                  loading={loading}
                  error={globalError}
                />
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>

      <PaperTradeDrawer
        open={tradeDrawerOpen}
        instrument={selectedInstrument}
        quote={quote}
        submitting={tradeSubmitting}
        onClose={() => setTradeDrawerOpen(false)}
        onSubmit={handleTradeSubmit}
      />
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  )
}
