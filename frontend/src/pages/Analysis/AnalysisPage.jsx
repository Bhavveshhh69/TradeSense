import { useState } from 'react'

import { analyzeMarket } from '../../api/analysis'
import StockPriceChart from '../../components/analysis/StockPriceChart'

function formatError(err) {
  if (typeof err?.response?.data?.detail === 'string' && err.response.data.detail.trim()) {
    return err.response.data.detail
  }

  if (typeof err?.response?.data?.error === 'string' && err.response.data.error.trim()) {
    return err.response.data.error
  }

  if (typeof err?.response?.data === 'string' && err.response.data.trim()) {
    return err.response.data
  }

  if (typeof err?.message === 'string' && err.message.trim()) {
    return err.message
  }

  return 'Unable to analyze right now. Please try again.'
}

function formatPercent(value, digits = 2) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  return `${(value * 100).toFixed(digits)}%`
}

function formatPrice(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  return value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

function getSignalClass(signalStrength) {
  if (signalStrength === 'STRONG') {
    return 'signal-strong'
  }

  if (signalStrength === 'MODERATE') {
    return 'signal-moderate'
  }

  if (signalStrength === 'WEAK') {
    return 'signal-weak'
  }

  return 'signal-noedge'
}

export default function AnalysisPage() {
  const [predictionData, setPredictionData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [symbol, setSymbol] = useState('')

  const normalizedSymbol = symbol.trim().toUpperCase()
  const resolvedSymbol = predictionData?.symbol ?? normalizedSymbol
  const trendSummary = predictionData?.trend_summary ?? predictionData?.context?.trend_summary ?? 'N/A'
  const riskSummary = predictionData?.risk_summary ?? predictionData?.context?.risk_summary ?? 'N/A'
  const signalStrength = predictionData?.signal_strength ?? 'NO_EDGE'
  const marketCondition = predictionData?.market_condition ?? 'NEUTRAL'
  const recommendation = predictionData?.recommendation ?? 'WAIT'
  const signalExplanation = predictionData?.signal_explanation ?? 'Signal explanation is unavailable.'

  const handleRun = async () => {
    if (!normalizedSymbol) {
      return
    }

    setLoading(true)
    setError(null)
    setPredictionData(null)

    try {
      const data = await analyzeMarket(normalizedSymbol)
      setPredictionData(data)
    } catch (requestError) {
      setError(`Unable to analyze ${normalizedSymbol}. ${formatError(requestError)}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="dashboard analysis-page">
      <p>
        Enter a symbol to run live prediction analysis via the production model pipeline.
      </p>

      <form
        className="dashboard-actions"
        onSubmit={(event) => {
          event.preventDefault()
          handleRun()
        }}
      >
        <label>
          <span>Symbol</span>
          <input
            type="text"
            value={symbol}
            onChange={(event) => setSymbol(event.target.value.toUpperCase())}
            placeholder="Enter symbol"
          />
        </label>
        <button
          type="submit"
          className="button"
          disabled={loading || normalizedSymbol.length === 0}
        >
          {loading ? 'Analyzing...' : 'Run Analysis'}
        </button>
        {loading && <span className="status">Analyzing...</span>}
      </form>

      {error && (
        <div className="error">
          <strong>Analysis Failed</strong>
          <p>{error}</p>
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
              <strong>{formatPrice(predictionData.current_price)}</strong>
            </article>
          </div>

          {predictionData.price_error && (
            <p className="analysis-inline-note">
              Price update note: {predictionData.price_error_message || 'Current price is unavailable.'}
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
                <strong className={getSignalClass(signalStrength)}>{signalStrength}</strong>
              </article>
              <article className="analysis-signal-item">
                <span>Market Condition</span>
                <strong>{marketCondition}</strong>
              </article>
              <article className="analysis-signal-item">
                <span>Recommendation</span>
                <strong>{recommendation}</strong>
              </article>
            </div>
            <p className="analysis-signal-explanation">{signalExplanation}</p>
          </section>

          <div className="analysis-result-grid">
            <article className="analysis-result-card">
              <span>Prediction</span>
              <strong>{predictionData.decision ?? 'N/A'}</strong>
            </article>
            <article className="analysis-result-card">
              <span>Probability</span>
              <strong>{formatPercent(predictionData.probability, 2)}</strong>
            </article>
            <article className="analysis-result-card">
              <span>Confidence</span>
              <strong>{predictionData.confidence_level ?? 'N/A'}</strong>
            </article>
            <article className="analysis-result-card">
              <span>Trend</span>
              <strong>{trendSummary}</strong>
            </article>
            <article className="analysis-result-card">
              <span>Risk</span>
              <strong>{riskSummary}</strong>
            </article>
          </div>
        </>
      )}
    </section>
  )
}
