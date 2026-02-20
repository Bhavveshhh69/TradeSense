import { useState } from 'react'
import { analyzeMarket } from '../api/analyze'

function formatError(err) {
  if (err?.response?.data?.detail) {
    return err.response.data.detail
  }

  if (typeof err?.response?.data === 'string') {
    return err.response.data
  }

  if (err?.message) {
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

export default function Dashboard() {
  const [predictionData, setPredictionData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [symbol, setSymbol] = useState('')

  const normalizedSymbol = symbol.trim().toUpperCase()

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
    } catch (err) {
      setError(`Unable to analyze ${normalizedSymbol}. ${formatError(err)}`)
    } finally {
      setLoading(false)
    }
  }

  const context = predictionData?.context ?? {}

  return (
    <section className="dashboard">
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
        <div className="result">
          <div>
            Symbol: <strong>{predictionData.symbol ?? normalizedSymbol}</strong>
          </div>
          <div>
            Prediction: <strong>{predictionData.decision ?? 'N/A'}</strong>
          </div>
          <div>
            Probability: <strong>{formatPercent(predictionData.probability, 2)}</strong>
          </div>
          <div>
            Confidence Level: <strong>{predictionData.confidence_level ?? 'N/A'}</strong>
          </div>
          <div>
            Strength: <strong>{formatPercent(predictionData.strength, 1)}</strong>
          </div>
          <div>
            Raw Prediction: <strong>{predictionData.prediction ?? 'N/A'}</strong>
          </div>
          <div>
            Decision Summary: <strong>{context.decision_summary ?? 'N/A'}</strong>
          </div>
          <div>
            Confidence Summary: <strong>{context.confidence_summary ?? 'N/A'}</strong>
          </div>
          <div>
            Trend Summary: <strong>{context.trend_summary ?? 'N/A'}</strong>
          </div>
          <div>
            Risk Summary: <strong>{context.risk_summary ?? 'N/A'}</strong>
          </div>
        </div>
      )}
    </section>
  )
}
