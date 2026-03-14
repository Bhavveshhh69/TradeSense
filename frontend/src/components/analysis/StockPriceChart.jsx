import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { fetchStockHistory } from '../../api/analysis'
import {
  formatAxisDate,
  formatLongDate,
  formatPercent,
  normalizePriceHistory,
} from '../../utils/dashboard'

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

function formatPrice(value) {
  const numericValue = Number(value)
  if (!Number.isFinite(numericValue)) {
    return 'N/A'
  }

  return numericValue.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

function formatPriceWithCurrency(value, currencyCode = 'USD') {
  const numericValue = Number(value)
  if (!Number.isFinite(numericValue)) {
    return 'N/A'
  }

  return numericValue.toLocaleString(undefined, {
    style: 'currency',
    currency: currencyCode,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

function PriceTooltip({ active, label, payload, currencyCode }) {
  if (!active || !Array.isArray(payload) || payload.length === 0) {
    return null
  }

  const point = payload[0]?.payload
  const pointDate = typeof point?.date === 'string' ? point.date : label

  return (
    <div className="analysis-chart-tooltip">
      <p>Date: {formatLongDate(pointDate)}</p>
      <p>Price: {formatPriceWithCurrency(Number(point?.close), currencyCode)}</p>
      <p>Daily change: {formatPercent(point?.dailyChangePct, 2, { signed: true })}</p>
    </div>
  )
}

function resolveCurrencyCode(symbol) {
  if (typeof symbol === 'string' && (symbol.endsWith('.NS') || symbol.endsWith('.BO'))) {
    return 'INR'
  }

  return 'USD'
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

        const points = normalizePriceHistory(payload?.history)
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

  const latestPoint = history[history.length - 1]
  const hasLargeMove = history.some((point) => point.hasLargeMove)
  const currencyCode = resolveCurrencyCode(resolvedSymbol || normalizedSymbol)

  return (
    <section className="analysis-chart-card">
      <div className="analysis-chart-header">
        <h3>Price Chart</h3>
        <span>Last 30 days</span>
      </div>
      <div className="chart-kpi-row">
        <span>Latest price: {formatPriceWithCurrency(latestPoint?.close, currencyCode)}</span>
        <span>Daily move: {formatPercent(latestPoint?.dailyChangePct, 2, { signed: true })}</span>
        <span className={hasLargeMove ? 'chart-alert chart-alert-hot' : 'chart-alert'}>
          {hasLargeMove ? 'Large move detected (>2%)' : 'Price action within normal range'}
        </span>
      </div>
      <div className="analysis-chart-wrap">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={history} margin={{ top: 10, right: 12, left: 4, bottom: 6 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="date" tickFormatter={formatAxisDate} minTickGap={24} />
            <YAxis dataKey="close" tickFormatter={(value) => formatPrice(Number(value))} width={92} />
            <Tooltip content={<PriceTooltip currencyCode={currencyCode} />} />
            <Line
              type="monotone"
              dataKey="close"
              stroke={hasLargeMove ? '#ea580c' : '#1d4ed8'}
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 4 }}
            />
            {history
              .filter((point) => point.hasLargeMove)
              .map((point) => (
                <ReferenceDot
                  key={`${point.date}-${point.close}`}
                  x={point.date}
                  y={point.close}
                  r={4}
                  fill="#dc2626"
                  stroke="#ffffff"
                />
              ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
