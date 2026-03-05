import { useState } from 'react'
import { normalizeSymbol } from '../../api/symbols'

function normalizePositiveNumber(value) {
  const normalized = Number(value)
  if (!Number.isFinite(normalized) || normalized <= 0) {
    return null
  }
  return normalized
}

export default function AddHoldingForm({ onAdd, submitting }) {
  const [ticker, setTicker] = useState('')
  const [shares, setShares] = useState('')
  const [buyPrice, setBuyPrice] = useState('')
  const [error, setError] = useState(null)
  const [normalizedTicker, setNormalizedTicker] = useState('')
  const [normalizing, setNormalizing] = useState(false)

  const resolveNormalizedTicker = async (value) => {
    const normalizedInput = typeof value === 'string' ? value.trim().toUpperCase() : ''
    if (!normalizedInput) {
      setNormalizedTicker('')
      return ''
    }

    setNormalizing(true)
    try {
      const response = await normalizeSymbol(normalizedInput)
      const normalized = typeof response?.normalized === 'string'
        ? response.normalized.trim().toUpperCase()
        : normalizedInput
      setNormalizedTicker(normalized)
      return normalized
    } catch {
      setNormalizedTicker(normalizedInput)
      return normalizedInput
    } finally {
      setNormalizing(false)
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()

    const tickerInput = ticker.trim().toUpperCase()
    const resolvedTicker = await resolveNormalizedTicker(tickerInput)
    const normalizedShares = normalizePositiveNumber(shares)
    const normalizedBuyPrice = normalizePositiveNumber(buyPrice)

    if (!resolvedTicker) {
      setError('Ticker is required')
      return
    }

    if (normalizedShares === null) {
      setError('Shares must be a positive number')
      return
    }

    if (normalizedBuyPrice === null) {
      setError('Buy price must be a positive number')
      return
    }

    setError(null)

    try {
      await onAdd({
        ticker: resolvedTicker,
        shares: normalizedShares,
        buy_price: normalizedBuyPrice,
      })
      setTicker('')
      setShares('')
      setBuyPrice('')
      setNormalizedTicker('')
    } catch (submitError) {
      setError(submitError?.message || 'Unable to add holding')
    }
  }

  return (
    <form className="portfolio-form" onSubmit={handleSubmit}>
      <h3>Add Holding</h3>
      <div className="portfolio-form-grid">
        <label>
          <span>Ticker</span>
          <input
            type="text"
            value={ticker}
            onChange={(event) => {
              setTicker(event.target.value.toUpperCase())
              setNormalizedTicker('')
            }}
            onBlur={async () => {
              await resolveNormalizedTicker(ticker)
            }}
            placeholder="AAPL or RELIANCE.NS"
            disabled={submitting || normalizing}
          />
        </label>
        <label>
          <span>Shares</span>
          <input
            type="number"
            min="0"
            step="any"
            value={shares}
            onChange={(event) => setShares(event.target.value)}
            placeholder="10"
            disabled={submitting}
          />
        </label>
        <label>
          <span>Buy Price</span>
          <input
            type="number"
            min="0"
            step="any"
            value={buyPrice}
            onChange={(event) => setBuyPrice(event.target.value)}
            placeholder="150.25"
            disabled={submitting}
          />
        </label>
      </div>

      <button type="submit" className="button" disabled={submitting || normalizing}>
        {submitting || normalizing ? 'Adding...' : 'Add Holding'}
      </button>

      {normalizing && <p className="portfolio-inline-note">Normalizing symbol...</p>}
      {!normalizing && normalizedTicker && (
        <p className="portfolio-inline-note">
          Normalized symbol: <strong>{normalizedTicker}</strong>
        </p>
      )}

      {error && <p className="portfolio-inline-error">{error}</p>}
    </form>
  )
}
