import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { analyzeMarket } from '../../api/analysis'
import {
  addPortfolioHolding,
  deletePortfolioHolding,
  fetchPortfolio,
  fetchPortfolioAdvisor,
  fetchPortfolioHistory,
  fetchPortfolioInsights,
} from '../../api/portfolio'
import StockPriceChart from '../../components/analysis/StockPriceChart'
import PortfolioAllocationChart from '../../components/portfolio/PortfolioAllocationChart'
import PortfolioAdvisor from '../../components/portfolio/PortfolioAdvisor'
import PortfolioEquityChart from '../../components/portfolio/PortfolioEquityChart'
import PortfolioInsights from '../../components/portfolio/PortfolioInsights'
import AddHoldingForm from '../Portfolio/AddHoldingForm'
import PortfolioTable from '../Portfolio/PortfolioTable'
import {
  buildPortfolioAdvisorInsights,
  computePortfolioAnalytics,
  formatMoney,
  formatPercent,
  formatSignedMoney,
  getPredictionLabel,
} from '../../utils/dashboard'

const PORTFOLIO_DAYS = 30
const PORTFOLIO_REFRESH_THROTTLE_MS = 30 * 1000
const FALLBACK_AI_EXPLANATION =
  'TradeSense AI explanation is temporarily unavailable due to API limits. Based on current model signals, the system suggests a cautious stance with weak momentum and moderate market risk.'

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

function formatApiError(error, fallback) {
  if (typeof error?.response?.data?.detail === 'string' && error.response.data.detail.trim()) {
    return error.response.data.detail
  }

  if (typeof error?.response?.data?.error === 'string' && error.response.data.error.trim()) {
    return error.response.data.error
  }

  if (typeof error?.response?.data === 'string' && error.response.data.trim()) {
    return error.response.data
  }

  if (typeof error?.message === 'string' && error.message.trim()) {
    return error.message
  }

  return fallback
}

function normalizeRecommendation(value) {
  if (typeof value !== 'string') {
    return 'WAIT'
  }

  const normalized = value.trim().toUpperCase()
  if (!normalized) {
    return 'WAIT'
  }

  if (normalized === 'BUY_BIAS') {
    return 'BUY'
  }

  if (normalized === 'SELL_BIAS') {
    return 'SELL'
  }

  if (normalized === 'WATCH') {
    return 'HOLD'
  }

  return normalized
}

function getRecommendationSummary(recommendation) {
  const normalized = normalizeRecommendation(recommendation)

  if (normalized === 'BUY') {
    return 'Momentum and sentiment are aligned enough to support a constructive bias.'
  }

  if (normalized === 'SELL') {
    return 'The model is detecting downside pressure and a weaker near-term setup.'
  }

  return 'The signal is still mixed, so waiting for clearer confirmation may reduce risk.'
}

function getAnalysisCurrencyCode(symbolValue) {
  const normalizedSymbol =
    typeof symbolValue === 'string' && symbolValue.trim() ? symbolValue.trim().toUpperCase() : ''

  if (normalizedSymbol.endsWith('.NS') || normalizedSymbol.endsWith('.BO')) {
    return 'INR'
  }

  return 'USD'
}

export default function DashboardPage({ initialSection = 'dashboard' }) {
  const [portfolio, setPortfolio] = useState({
    holdings: [],
    summary: EMPTY_SUMMARY,
  })
  const [portfolioLoading, setPortfolioLoading] = useState(true)
  const [portfolioError, setPortfolioError] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [deletingId, setDeletingId] = useState(null)
  const [portfolioHistory, setPortfolioHistory] = useState(EMPTY_HISTORY)
  const [portfolioInsights, setPortfolioInsights] = useState(EMPTY_INSIGHTS)
  const [portfolioAdvisor, setPortfolioAdvisor] = useState(EMPTY_ADVISOR)
  const latestReloadVersionRef = useRef(0)
  const lastPortfolioRefreshAtRef = useRef(0)
  const activePortfolioReloadRef = useRef(null)

  const [predictionData, setPredictionData] = useState(null)
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [analysisError, setAnalysisError] = useState(null)
  const [symbol, setSymbol] = useState('')

  const dashboardSectionRef = useRef(null)
  const portfolioSectionRef = useRef(null)
  const analysisSectionRef = useRef(null)
  const holdingsSectionRef = useRef(null)

  const reloadPortfolioData = useCallback(async ({ withLoading = true, force = false } = {}) => {
    if (
      !force &&
      lastPortfolioRefreshAtRef.current > 0 &&
      Date.now() - lastPortfolioRefreshAtRef.current < PORTFOLIO_REFRESH_THROTTLE_MS
    ) {
      return activePortfolioReloadRef.current
    }

    if (activePortfolioReloadRef.current) {
      return activePortfolioReloadRef.current
    }

    const reloadVersion = latestReloadVersionRef.current + 1
    latestReloadVersionRef.current = reloadVersion

    const requestPromise = (async () => {
      if (withLoading) {
        setPortfolioLoading(true)
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

        setPortfolio(normalizePortfolioPayload(portfolioPayload))
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
        lastPortfolioRefreshAtRef.current = Date.now()
        setPortfolioError(null)
      } catch (requestError) {
        if (reloadVersion === latestReloadVersionRef.current) {
          setPortfolioError(formatApiError(requestError, 'Unable to load portfolio data'))
        }
      } finally {
        if (withLoading && reloadVersion === latestReloadVersionRef.current) {
          setPortfolioLoading(false)
        }
      }
    })()

    activePortfolioReloadRef.current = requestPromise

    try {
      await requestPromise
    } finally {
      if (activePortfolioReloadRef.current === requestPromise) {
        activePortfolioReloadRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    reloadPortfolioData({ withLoading: true })
  }, [reloadPortfolioData])

  useEffect(() => {
    const targetMap = {
      dashboard: dashboardSectionRef,
      portfolio: portfolioSectionRef,
      analyze: analysisSectionRef,
      analysis: analysisSectionRef,
      holdings: holdingsSectionRef,
    }

    const target = targetMap[initialSection]?.current
    if (!target) {
      return
    }

    const timer = window.setTimeout(() => {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 120)

    return () => window.clearTimeout(timer)
  }, [initialSection])

  const handleAddHolding = async (payload) => {
    setSubmitting(true)
    setPortfolioError(null)

    try {
      await addPortfolioHolding(payload)
      await reloadPortfolioData({ withLoading: false, force: true })
    } catch (requestError) {
      throw new Error(formatApiError(requestError, 'Unable to add holding'))
    } finally {
      setSubmitting(false)
    }
  }

  const handleDeleteHolding = async (id) => {
    setDeletingId(id)
    setPortfolioError(null)

    try {
      await deletePortfolioHolding(id)
      await reloadPortfolioData({ withLoading: false, force: true })
    } catch (requestError) {
      setPortfolioError(formatApiError(requestError, 'Unable to delete holding'))
    } finally {
      setDeletingId(null)
    }
  }

  const handleRunAnalysis = async () => {
    const normalizedSymbol = symbol.trim().toUpperCase()
    if (!normalizedSymbol) {
      return
    }

    setAnalysisLoading(true)
    setAnalysisError(null)
    setPredictionData(null)

    try {
      const data = await analyzeMarket(normalizedSymbol)
      setPredictionData(data)
    } catch (requestError) {
      setAnalysisError(
        `Unable to analyze ${normalizedSymbol}. ${formatApiError(
          requestError,
          'Unable to analyze right now. Please try again.'
        )}`
      )
    } finally {
      setAnalysisLoading(false)
    }
  }

  const summary = portfolio.summary || EMPTY_SUMMARY
  const baseCurrency = summary?.base_currency || 'INR'
  const normalizedSymbol = symbol.trim().toUpperCase()
  const resolvedSymbol = predictionData?.symbol ?? normalizedSymbol
  const analysisCurrencyCode = getAnalysisCurrencyCode(resolvedSymbol)
  const recommendation = normalizeRecommendation(
    predictionData?.recommendation ?? getPredictionLabel(predictionData?.prediction, 'WAIT')
  )
  const predictionLabel = getPredictionLabel(
    predictionData?.prediction_label ?? predictionData?.prediction ?? predictionData?.decision,
    predictionData?.decision ?? 'N/A'
  )
  const aiExplanation =
    typeof predictionData?.explanation === 'string' && predictionData.explanation.trim()
      ? predictionData.explanation.trim()
      : FALLBACK_AI_EXPLANATION
  const explanationIsFallback =
    Boolean(predictionData) &&
    (predictionData?.explanation_is_fallback === true ||
      predictionData?.explanation !== aiExplanation)
  const marketInsight =
    typeof predictionData?.market_insight === 'string' && predictionData.market_insight.trim()
      ? predictionData.market_insight.trim()
      : null
  const analytics = useMemo(
    () => computePortfolioAnalytics(portfolio.holdings, portfolioHistory, portfolioInsights),
    [portfolio.holdings, portfolioHistory, portfolioInsights]
  )
  const advisorRecommendations = useMemo(
    () =>
      buildPortfolioAdvisorInsights(
        analytics,
        Array.isArray(portfolioAdvisor?.recommendations) ? portfolioAdvisor.recommendations : []
      ),
    [analytics, portfolioAdvisor]
  )

  return (
    <section id="dashboard" ref={dashboardSectionRef} className="dashboard-page">
      <div className="dashboard-hero">
        <div>
          <p className="dashboard-eyebrow">TradeSense Dashboard</p>
          <h1>One place to track portfolio risk and run live stock analysis.</h1>
          <p className="dashboard-hero-copy">
            Portfolio monitoring, AI-assisted explanations, and prediction context now live in a
            single workflow.
          </p>
        </div>
        <div className="dashboard-hero-stats">
          <article className="hero-stat-card">
            <span>Portfolio Risk Score</span>
            <strong className={`risk-score-${analytics.riskScore.toLowerCase()}`}>
              {analytics.riskScore}
            </strong>
          </article>
          <article className="hero-stat-card">
            <span>Largest Holding</span>
            <strong>
              {analytics.topHolding ? analytics.topHolding.ticker : 'N/A'}{' '}
              {analytics.topHolding ? `(${formatPercent(analytics.largestHoldingPercent, 1)})` : ''}
            </strong>
          </article>
        </div>
      </div>

      <div className="dashboard-grid">
        <section
          id="portfolio"
          ref={portfolioSectionRef}
          className="dashboard-section dashboard-panel"
        >
          <div className="section-header">
            <div>
              <p className="section-kicker">Portfolio Overview</p>
              <h2>Allocation, returns, and portfolio health at a glance.</h2>
            </div>
            <p className="section-caption">
              Portfolio values converted to base currency ({baseCurrency}).
            </p>
          </div>

          <div className="portfolio-summary-grid dashboard-summary-grid">
            <article className="portfolio-summary-card">
              <span>Total Value</span>
              <strong>{formatMoney(summary.total_portfolio_value, baseCurrency)}</strong>
            </article>
            <article className="portfolio-summary-card">
              <span>Total Invested</span>
              <strong>{formatMoney(summary.total_invested_value, baseCurrency)}</strong>
            </article>
            <article className="portfolio-summary-card">
              <span>Profit/Loss</span>
              <strong
                className={
                  summary.total_profit_loss > 0
                    ? 'profit-loss-positive'
                    : summary.total_profit_loss < 0
                      ? 'profit-loss-negative'
                      : 'profit-loss-neutral'
                }
              >
                {formatSignedMoney(summary.total_profit_loss, baseCurrency)} (
                {formatPercent(summary.total_profit_loss_percent, 2, { signed: true })})
              </strong>
            </article>
            <article className="portfolio-summary-card risk-score-card">
              <span>Portfolio Risk Score</span>
              <strong className={`risk-score-${analytics.riskScore.toLowerCase()}`}>
                {analytics.riskScore}
              </strong>
              <small>
                Volatility {formatPercent(analytics.volatilityPercent, 2)}, max drawdown{' '}
                {formatPercent(analytics.maxDrawdownPct, 2)}
              </small>
            </article>
          </div>

          <div className="dashboard-two-column">
            <PortfolioAllocationChart holdings={portfolio.holdings} baseCurrency={baseCurrency} />
            <PortfolioInsights data={portfolioInsights} metrics={analytics} baseCurrency={baseCurrency} />
          </div>

          {(portfolioLoading || portfolioError) && (
            <div className="dashboard-inline-state">
              {portfolioLoading && <p className="status">Loading portfolio...</p>}
              {portfolioError && (
                <div className="error">
                  <strong>Portfolio Error</strong>
                  <p>{portfolioError}</p>
                </div>
              )}
            </div>
          )}
        </section>

        <section className="dashboard-section dashboard-panel">
          <PortfolioEquityChart history={portfolioHistory} baseCurrency={baseCurrency} />
        </section>

        <section className="dashboard-section dashboard-panel">
          <PortfolioAdvisor
            title="Portfolio AI Advisor"
            data={{ recommendations: advisorRecommendations }}
          />
        </section>

        <section ref={holdingsSectionRef} className="dashboard-section dashboard-panel">
          <div className="section-header">
            <div>
              <p className="section-kicker">Holdings</p>
              <h2>Update positions without leaving the dashboard.</h2>
            </div>
          </div>
          <AddHoldingForm onAdd={handleAddHolding} submitting={submitting} />
          {!portfolioLoading && !portfolioError && (
            <PortfolioTable
              holdings={portfolio.holdings}
              onDelete={handleDeleteHolding}
              deletingId={deletingId}
            />
          )}
        </section>

        <section
          id="analysis"
          ref={analysisSectionRef}
          className="dashboard-section dashboard-panel"
        >
          <div className="section-header">
            <div>
              <p className="section-kicker">Stock Analysis</p>
              <h2>Run a live analysis for any supported symbol.</h2>
            </div>
          </div>

          <form
            className="dashboard-actions"
            onSubmit={(event) => {
              event.preventDefault()
              handleRunAnalysis()
            }}
          >
            <label>
              <span>Symbol</span>
              <input
                type="text"
                value={symbol}
                onChange={(event) => setSymbol(event.target.value.toUpperCase())}
                placeholder="AAPL"
              />
            </label>
            <button
              type="submit"
              className="button"
              disabled={analysisLoading || normalizedSymbol.length === 0}
            >
              {analysisLoading ? 'Analyzing...' : 'Run Analysis'}
            </button>
            {analysisLoading && <span className="status">Analyzing live market context...</span>}
          </form>

          {analysisError && (
            <div className="error">
              <strong>Analysis Failed</strong>
              <p>{analysisError}</p>
            </div>
          )}

          {predictionData && (
            <>
              <div className="analysis-summary-grid">
                <article className="analysis-summary-card">
                  <span>Symbol</span>
                  <strong>{resolvedSymbol || 'N/A'}</strong>
                </article>
                <article className="analysis-summary-card">
                  <span>Current Price</span>
                  <strong>{formatMoney(predictionData.current_price, analysisCurrencyCode)}</strong>
                </article>
              </div>

              {predictionData.price_error && (
                <p className="analysis-inline-note">
                  Price update note:{' '}
                  {predictionData.price_error_message || 'Current price is unavailable.'}
                </p>
              )}

              <StockPriceChart symbol={resolvedSymbol} />

              <section className="analysis-signal-card">
                <div className="analysis-signal-header">
                  <h3>Trading Signal</h3>
                </div>
                <div className="analysis-signal-grid">
                  <article className="analysis-signal-item">
                    <span>Signal Strength</span>
                    <strong>{predictionData?.signal_strength ?? 'N/A'}</strong>
                  </article>
                  <article className="analysis-signal-item">
                    <span>Market Condition</span>
                    <strong>{predictionData?.market_condition ?? 'N/A'}</strong>
                  </article>
                  <article className="analysis-signal-item">
                    <span>Recommendation</span>
                    <strong>{recommendation}</strong>
                  </article>
                </div>
                {predictionData?.signal_explanation && (
                  <p className="analysis-signal-explanation">{predictionData.signal_explanation}</p>
                )}
                <p className="analysis-signal-summary">{getRecommendationSummary(recommendation)}</p>
              </section>

              <section className="analysis-stats-card">
                <div className="analysis-info-header">
                  <h3>Prediction Statistics</h3>
                </div>
                <div className="analysis-result-grid">
                  <article className="analysis-result-card">
                    <span>Prediction</span>
                    <strong>{predictionLabel}</strong>
                  </article>
                  <article className="analysis-result-card">
                    <span>Probability</span>
                    <strong>{formatPercent(predictionData?.probability, 2, { scale: 100 })}</strong>
                  </article>
                  <article className="analysis-result-card">
                    <span>Confidence</span>
                    <strong>{predictionData?.confidence ?? predictionData?.confidence_level ?? 'N/A'}</strong>
                  </article>
                  <article className="analysis-result-card">
                    <span>Trend</span>
                    <strong>{predictionData?.trend_summary ?? predictionData?.context?.trend_summary ?? 'N/A'}</strong>
                  </article>
                  <article className="analysis-result-card">
                    <span>Risk</span>
                    <strong>{predictionData?.risk_summary ?? predictionData?.context?.risk_summary ?? 'N/A'}</strong>
                  </article>
                </div>
              </section>

              <section className="analysis-info-card">
                <div className="analysis-info-header">
                  <h3>TradeSense AI Explanation</h3>
                  {explanationIsFallback && (
                    <span className="analysis-info-badge">Fallback explanation</span>
                  )}
                </div>
                <p className="analysis-info-text">{aiExplanation}</p>
              </section>

              {marketInsight && (
                <section className="analysis-info-card market-insights-card">
                  <div className="analysis-info-header">
                    <h3>Market Insights</h3>
                  </div>
                  <p className="analysis-info-text">{marketInsight}</p>
                </section>
              )}
            </>
          )}
        </section>
      </div>
    </section>
  )
}
