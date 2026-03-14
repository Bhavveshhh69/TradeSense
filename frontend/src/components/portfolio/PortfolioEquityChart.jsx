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

import {
  formatAxisDate,
  formatLongDate,
  formatMoney,
  formatPercent,
  normalizeEquityCurve,
} from '../../utils/dashboard'

const DEFAULT_DAYS = 30

function EquityTooltip({ active, label, payload, baseCurrency }) {
  if (!active || !Array.isArray(payload) || payload.length === 0) {
    return null
  }

  const point = payload[0]?.payload
  const pointDate = typeof point?.date === 'string' ? point.date : label

  return (
    <div className="analysis-chart-tooltip">
      <p>Date: {formatLongDate(pointDate)}</p>
      <p>Value: {formatMoney(point?.portfolio_value, baseCurrency)}</p>
      <p>Daily change: {formatPercent(point?.dailyChangePct, 2, { signed: true })}</p>
      <p>Drawdown: {formatPercent(point?.drawdownPct, 2)}</p>
    </div>
  )
}

export default function PortfolioEquityChart({
  history,
  days = DEFAULT_DAYS,
  baseCurrency = 'INR',
}) {
  const curve = normalizeEquityCurve(history)
  const resolvedDays =
    typeof history?.days === 'number' && Number.isFinite(history.days) ? history.days : days

  if (curve.length === 0) {
    return (
      <section className="portfolio-history-card">
        <div className="portfolio-history-header">
          <h3>Portfolio Equity Curve</h3>
          <span>Last {resolvedDays} days</span>
        </div>
        <p className="analysis-chart-message">No portfolio history is available yet.</p>
      </section>
    )
  }

  const latestPoint = curve[curve.length - 1]
  const hasLargeMove = curve.some((point) => point.hasLargeMove)
  const maxDrawdownPct = curve.reduce(
    (minimumValue, point) =>
      typeof point.drawdownPct === 'number' && point.drawdownPct < minimumValue
        ? point.drawdownPct
        : minimumValue,
    0
  )

  return (
    <section className="portfolio-history-card">
      <div className="portfolio-history-header">
        <h3>Portfolio Equity Curve</h3>
        <span>Last {resolvedDays} days</span>
      </div>
      <div className="chart-kpi-row">
        <span>Current value: {formatMoney(latestPoint?.portfolio_value, baseCurrency)}</span>
        <span>Daily change: {formatPercent(latestPoint?.dailyChangePct, 2, { signed: true })}</span>
        <span>Max drawdown: {formatPercent(maxDrawdownPct, 2)}</span>
      </div>
      <div className="portfolio-history-chart">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={curve} margin={{ top: 10, right: 12, left: 4, bottom: 6 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="date" tickFormatter={formatAxisDate} minTickGap={24} />
            <YAxis
              dataKey="portfolio_value"
              tickFormatter={(value) => formatMoney(Number(value), baseCurrency)}
              width={92}
            />
            <Tooltip content={<EquityTooltip baseCurrency={baseCurrency} />} />
            <Line
              type="monotone"
              dataKey="portfolio_value"
              stroke={hasLargeMove ? '#ea580c' : '#1d4ed8'}
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 4 }}
            />
            {curve
              .filter((point) => point.hasLargeMove)
              .map((point) => (
                <ReferenceDot
                  key={`${point.date}-${point.portfolio_value}`}
                  x={point.date}
                  y={point.portfolio_value}
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
