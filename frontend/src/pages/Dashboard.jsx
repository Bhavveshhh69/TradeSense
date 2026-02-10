import { useState } from 'react'
import { analyzeMarket } from '../api/analyze'

function formatError(err) {
  if (err?.response?.data) {
    const data = err.response.data
    return typeof data === 'string' ? data : JSON.stringify(data, null, 2)
  }

  if (err?.message) {
    return err.message
  }

  return 'Unknown error'
}

export default function Dashboard() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [symbol, setSymbol] = useState('')

  const normalizedSymbol = symbol.trim().toUpperCase()

  const handleRun = async () => {
    if (!normalizedSymbol) {
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await analyzeMarket({
        symbol: normalizedSymbol,
        payload: {
          symbol: normalizedSymbol,
          probability: 0.62,
          feature_importance: {
            price_vs_ema20: 0.25,
            rsi_slope_3: 0.2,
            macd_hist_accel: 0.15,
            volume_ratio: 0.1,
            price_vs_ema50: 0.05,
          },
          feature_values: {
            price_vs_ema20: 0.02,
            rsi_slope_3: -0.1,
            macd_hist_accel: 0.0,
            volume_ratio: 1.2,
            price_vs_ema50: -0.01,
          },
          trend_state: 1,
          momentum_state: 1,
          risk_state: 1,
        },
      })
      setResult(data)
    } catch (err) {
      setError(formatError(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="dashboard">
      <p>
        Run a deterministic market analysis using the existing Node API. The response
        below is the direct reasoning output from the backend.
      </p>

      <div className="dashboard-actions">
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
          className="button"
          onClick={handleRun}
          disabled={loading || normalizedSymbol.length === 0}
        >
          {loading ? 'Running...' : 'Run Analysis'}
        </button>
        {loading && <span className="status">Waiting for /api/analyze...</span>}
      </div>

      {error && (
        <div className="error">
          <strong>Error</strong>
          <pre>{error}</pre>
        </div>
      )}

      {result && (
        <>
          <div>
            Analyzed Symbol: <strong>{result.symbol || normalizedSymbol}</strong>
          </div>
          <pre className="result">{JSON.stringify(result, null, 2)}</pre>
        </>
      )}
    </section>
  )
}
