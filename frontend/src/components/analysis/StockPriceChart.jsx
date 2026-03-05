import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { fetchStockHistory } from '../../api/analysis'

function formatHistoryError(error, symbol) {
  if (typeof error?.response?.data?.detail === 'string' && error.response.data.detail.trim()) {
    return error.response.data.detail
  }

  if (typeof error?.response?.data?.error === 'string' && error.response.data.error.trim()) {
    return error.response.data.error
  }

  if (typeof error?.message === 'string' && error.message.trim()) {
    return error.message
  }

  return `Unable to load 30-day price history for ${symbol}.`
}

function formatAxisDate(date) {
  if (typeof date !== 'string') {
    return ''
  }

  const [year, month, day] = date.split('-')
  if (!year || !month || !day) {
    return date
  }

  return `${month}/${day}`
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

function normalizeHistoryPoints(history) {
  if (!Array.isArray(history)) {
    return []
  }

  return history
    .map((point) => {
      const date = typeof point?.date === 'string' ? point.date.trim() : ''
      const close = Number(point?.close)
      if (!date || !Number.isFinite(close)) {
        return null
      }

      return {
        date,
        close,
      }
    })
    .filter(Boolean)
    .sort((a, b) => a.date.localeCompare(b.date))
}

export default function StockPriceChart({ symbol }) {
  const normalizedSymbol = useMemo(
    () => (typeof symbol === 'string' ? symbol.trim().toUpperCase() : ''),
    [symbol]
  )
  const [history, setHistory] = useState([])
  const [resolvedSymbol, setResolvedSymbol] = useState(normalizedSymbol)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false

    async function loadHistory() {
      if (!normalizedSymbol) {
        setHistory([])
        setResolvedSymbol('')
        setError(null)
        setLoading(false)
        return
      }

      setLoading(true)
      setError(null)

      try {
        const payload = await fetchStockHistory(normalizedSymbol)
        if (cancelled) {
          return
        }

        const points = normalizeHistoryPoints(payload?.history)
        const responseSymbol =
          typeof payload?.symbol === 'string' && payload.symbol.trim()
            ? payload.symbol.trim().toUpperCase()
            : normalizedSymbol

        setHistory(points)
        setResolvedSymbol(responseSymbol)
      } catch (requestError) {
        if (!cancelled) {
          setHistory([])
          setResolvedSymbol(normalizedSymbol)
          setError(formatHistoryError(requestError, normalizedSymbol))
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    loadHistory()

    return () => {
      cancelled = true
    }
  }, [normalizedSymbol])

  if (!normalizedSymbol) {
    return null
  }

  if (loading) {
    return (
      <section className="analysis-chart-card">
        <p className="status">Loading 30-day price history...</p>
      </section>
    )
  }

  if (error) {
    return (
      <section className="analysis-chart-card">
        <div className="analysis-chart-header">
          <h3>Price Chart</h3>
          <span>{resolvedSymbol || normalizedSymbol}</span>
        </div>
        <p className="analysis-chart-message">{error}</p>
      </section>
    )
  }

  if (history.length === 0) {
    return (
      <section className="analysis-chart-card">
        <div className="analysis-chart-header">
          <h3>Price Chart</h3>
          <span>{resolvedSymbol || normalizedSymbol}</span>
        </div>
        <p className="analysis-chart-message">
          No recent price history is available for this symbol.
        </p>
      </section>
    )
  }

  return (
    <section className="analysis-chart-card">
      <div className="analysis-chart-header">
        <h3>Price Chart</h3>
        <span>Last 30 days</span>
      </div>
      <div className="analysis-chart-wrap">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={history} margin={{ top: 10, right: 12, left: 4, bottom: 6 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="date" tickFormatter={formatAxisDate} minTickGap={24} />
            <YAxis
              dataKey="close"
              tickFormatter={(value) => formatPrice(Number(value))}
              width={92}
            />
            <Tooltip
              labelFormatter={(label) => `Date: ${label}`}
              formatter={(value) => [formatPrice(Number(value)), 'Close']}
            />
            <Line
              type="monotone"
              dataKey="close"
              stroke="#1d4ed8"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
