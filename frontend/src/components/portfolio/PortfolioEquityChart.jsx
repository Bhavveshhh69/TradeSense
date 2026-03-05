import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const DEFAULT_DAYS = 30

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

function formatValue(value, currencyCode) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  const symbol = getCurrencySymbol(currencyCode)
  const formatted = formatNumber(value)
  return symbol ? `${symbol}${formatted}` : formatted
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

export default function PortfolioEquityChart({ history, days = DEFAULT_DAYS, baseCurrency = 'INR' }) {
  const curve = Array.isArray(history?.equity_curve) ? history.equity_curve : []
  const resolvedDays =
    typeof history?.days === 'number' && Number.isFinite(history.days)
      ? history.days
      : days

  return (
    <section className="portfolio-history-card">
      <div className="portfolio-history-header">
        <h3>Portfolio Equity Curve</h3>
        <span>Last {resolvedDays} days</span>
      </div>
      <div className="portfolio-history-chart">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={curve} margin={{ top: 10, right: 12, left: 4, bottom: 6 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="date" tickFormatter={formatAxisDate} minTickGap={24} />
            <YAxis
              dataKey="portfolio_value"
              tickFormatter={(value) => formatValue(Number(value), baseCurrency)}
              width={84}
            />
            <Tooltip
              labelFormatter={(label) => `Date: ${label}`}
              formatter={(value) => [formatValue(Number(value), baseCurrency), 'Portfolio Value']}
            />
            <Line
              type="monotone"
              dataKey="portfolio_value"
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
