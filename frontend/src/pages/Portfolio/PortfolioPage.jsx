import { useCallback, useEffect, useRef, useState } from 'react'
import {
  addPortfolioHolding,
  deletePortfolioHolding,
  fetchPortfolio,
  fetchPortfolioAdvisor,
  fetchPortfolioHistory,
  fetchPortfolioInsights,
} from '../../api/portfolio'
import AddHoldingForm from './AddHoldingForm'
import PortfolioEquityChart from '../../components/portfolio/PortfolioEquityChart'
import PortfolioAllocationChart from '../../components/portfolio/PortfolioAllocationChart'
import PortfolioInsights from '../../components/portfolio/PortfolioInsights'
import PortfolioAdvisor from '../../components/portfolio/PortfolioAdvisor'
import PortfolioTable from './PortfolioTable'

const PORTFOLIO_DAYS = 30

const EMPTY_SUMMARY = {
  total_portfolio_value: 0,
  total_invested_value: 0,
  total_profit_loss: 0,
  total_profit_loss_percent: 0,
  base_currency: 'INR',
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

function normalizePortfolioPayload(portfolioPayload) {
  const holdings = Array.isArray(portfolioPayload?.holdings)
    ? portfolioPayload.holdings.map((holding) => ({ ...holding }))
    : []
  const summary =
    portfolioPayload?.summary && typeof portfolioPayload.summary === 'object'
      ? { ...EMPTY_SUMMARY, ...portfolioPayload.summary }
      : EMPTY_SUMMARY

  return { holdings, summary }
}

function formatApiError(error) {
  if (typeof error?.response?.data?.error === 'string') {
    return error.response.data.error
  }

  if (typeof error?.message === 'string' && error.message.trim()) {
    return error.message
  }

  return 'Unable to load portfolio data'
}

function formatNumber(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  return value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

function getCurrencySymbol(currencyCode) {
  const code = typeof currencyCode === 'string' ? currencyCode.trim().toUpperCase() : ''
  if (code === 'USD') {
    return '$'
  }

  if (code === 'INR') {
    return '\u20B9'
  }

  return ''
}

function formatMoney(value, currencyCode) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  const symbol = getCurrencySymbol(currencyCode)
  if (symbol) {
    return `${symbol}${formatNumber(value)}`
  }

  const fallbackCode = typeof currencyCode === 'string' && currencyCode.trim()
    ? currencyCode.trim().toUpperCase()
    : ''
  return fallbackCode ? `${fallbackCode} ${formatNumber(value)}` : formatNumber(value)
}

function formatSignedMoney(value, currencyCode) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  const prefix = value > 0 ? '+' : value < 0 ? '-' : ''
  const absolute = Math.abs(value)
  return `${prefix}${formatMoney(absolute, currencyCode)}`
}

function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  const prefix = value > 0 ? '+' : ''
  return `${prefix}${value.toFixed(2)}%`
}

export default function PortfolioPage() {
  const [portfolio, setPortfolio] = useState({
    holdings: [],
    summary: EMPTY_SUMMARY,
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [deletingId, setDeletingId] = useState(null)
  const [portfolioHistory, setPortfolioHistory] = useState(EMPTY_HISTORY)
  const [portfolioInsights, setPortfolioInsights] = useState(EMPTY_INSIGHTS)
  const [portfolioAdvisor, setPortfolioAdvisor] = useState(EMPTY_ADVISOR)
  const latestReloadVersionRef = useRef(0)

  const reloadPortfolioData = useCallback(async ({ withLoading } = { withLoading: true }) => {
    const reloadVersion = latestReloadVersionRef.current + 1
    latestReloadVersionRef.current = reloadVersion

    if (withLoading) {
      setLoading(true)
    }

    try {
      const [portfolioPayload, historyPayload, insightsPayload, advisorPayload] = await Promise.all([
        fetchPortfolio(),
        fetchPortfolioHistory(PORTFOLIO_DAYS),
        fetchPortfolioInsights(PORTFOLIO_DAYS),
        fetchPortfolioAdvisor(PORTFOLIO_DAYS),
      ])

      if (reloadVersion !== latestReloadVersionRef.current) {
        return
      }

      const normalizedPortfolio = normalizePortfolioPayload(portfolioPayload)
      setPortfolio(normalizedPortfolio)
      setPortfolioHistory({
        days:
          typeof historyPayload?.days === 'number' && Number.isFinite(historyPayload.days)
            ? historyPayload.days
            : PORTFOLIO_DAYS,
        equity_curve: Array.isArray(historyPayload?.equity_curve)
          ? historyPayload.equity_curve.map((point) => ({ ...point }))
          : [],
      })
      setPortfolioInsights({
        concentration_risk: insightsPayload?.concentration_risk || 'LOW',
        largest_position: insightsPayload?.largest_position || null,
        best_performer: insightsPayload?.best_performer || null,
        worst_performer: insightsPayload?.worst_performer || null,
        diversification_score:
          typeof insightsPayload?.diversification_score === 'number' &&
          Number.isFinite(insightsPayload.diversification_score)
            ? insightsPayload.diversification_score
            : 0,
        volatility_level: insightsPayload?.volatility_level || 'LOW',
        insights: Array.isArray(insightsPayload?.insights) ? [...insightsPayload.insights] : [],
      })
      setPortfolioAdvisor({
        recommendations: Array.isArray(advisorPayload?.recommendations)
          ? advisorPayload.recommendations.filter((item) => typeof item === 'string' && item.trim())
          : [],
      })
      setError(null)
    } catch (requestError) {
      if (reloadVersion === latestReloadVersionRef.current) {
        setError(formatApiError(requestError))
      }
    } finally {
      if (withLoading && reloadVersion === latestReloadVersionRef.current) {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    reloadPortfolioData()
  }, [reloadPortfolioData])

  const handleAdd = async (payload) => {
    setSubmitting(true)
    setError(null)

    try {
      await addPortfolioHolding(payload)
      await reloadPortfolioData({ withLoading: false })
    } catch (requestError) {
      throw new Error(formatApiError(requestError))
    } finally {
      setSubmitting(false)
    }
  }

  const handleDelete = async (id) => {
    setDeletingId(id)
    setError(null)

    try {
      await deletePortfolioHolding(id)
      await reloadPortfolioData({ withLoading: false })
    } catch (requestError) {
      setError(formatApiError(requestError))
    } finally {
      setDeletingId(null)
    }
  }

  const summary = portfolio.summary || EMPTY_SUMMARY
  const baseCurrency = summary?.base_currency || 'INR'

  return (
    <section className="portfolio">
      <div className="portfolio-header">
        <h2>Portfolio Tracker</h2>
        <p>Decision support only. TradeSense does not execute trades.</p>
      </div>

      <div className="portfolio-summary-grid">
        <article className="portfolio-summary-card">
          <span>Total Value</span>
          <strong>{formatMoney(summary.total_portfolio_value, baseCurrency)}</strong>
        </article>
        <article className="portfolio-summary-card">
          <span>Total Invested</span>
          <strong>{formatMoney(summary.total_invested_value, baseCurrency)}</strong>
        </article>
        <article className="portfolio-summary-card">
          <span>Total Profit/Loss</span>
          <strong
            className={
              summary.total_profit_loss > 0
                ? 'profit-loss-positive'
                : summary.total_profit_loss < 0
                  ? 'profit-loss-negative'
                  : 'profit-loss-neutral'
            }
          >
            {formatSignedMoney(summary.total_profit_loss, baseCurrency)} ({formatPercent(summary.total_profit_loss_percent)})
          </strong>
        </article>
      </div>
      <p className="status">
        Portfolio values converted to base currency ({baseCurrency}).
      </p>

      <AddHoldingForm onAdd={handleAdd} submitting={submitting} />
      <PortfolioEquityChart history={portfolioHistory} baseCurrency={baseCurrency} />
      <PortfolioAllocationChart holdings={portfolio.holdings} baseCurrency={baseCurrency} />
      <PortfolioInsights data={portfolioInsights} />
      <PortfolioAdvisor data={portfolioAdvisor} />

      {loading && <p className="status">Loading portfolio...</p>}
      {error && (
        <div className="error">
          <strong>Portfolio Error</strong>
          <p>{error}</p>
        </div>
      )}

      {!loading && !error && (
        <PortfolioTable
          holdings={portfolio.holdings}
          onDelete={handleDelete}
          deletingId={deletingId}
        />
      )}
    </section>
  )
}
