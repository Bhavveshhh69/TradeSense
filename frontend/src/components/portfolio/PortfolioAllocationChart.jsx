import { useMemo } from 'react'
import {
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'

const ALLOCATION_COLORS = [
  '#1d4ed8',
  '#059669',
  '#c2410c',
  '#7c3aed',
  '#be123c',
  '#0f766e',
  '#4338ca',
  '#b45309',
]

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
  const formatted = formatNumber(value)
  return symbol ? `${symbol}${formatted}` : formatted
}

function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  return `${value.toFixed(2)}%`
}

function buildAllocation(holdings) {
  const normalized = Array.isArray(holdings)
    ? holdings
        .map((holding) => ({
          name:
            typeof holding?.ticker === 'string' && holding.ticker.trim()
              ? holding.ticker.trim().toUpperCase()
              : 'UNKNOWN',
          value: Number(holding?.market_value_base ?? holding?.current_value),
        }))
        .filter((entry) => Number.isFinite(entry.value) && entry.value > 0)
    : []

  const total = normalized.reduce((sum, entry) => sum + entry.value, 0)
  if (!Number.isFinite(total) || total <= 0) {
    return { total: 0, data: [] }
  }

  const data = normalized.map((entry) => {
    const weight = entry.value / total
    return {
      ...entry,
      weight,
      percentage: weight * 100,
    }
  })

  return { total, data }
}

function labelFormatter({ name, percent }) {
  const percentage = Number(percent) * 100
  return `${name} ${percentage.toFixed(1)}%`
}

function legendFormatter(value, entry) {
  const percentage = entry?.payload?.payload?.percentage ?? entry?.payload?.percentage
  return `${value} (${formatPercent(Number(percentage))})`
}

function tooltipFormatter(value, _name, item, baseCurrency) {
  const amount = formatMoney(Number(value), baseCurrency)
  const percentage = formatPercent(Number(item?.payload?.percentage))
  return [`${amount} (${percentage})`, 'Current Value']
}

export default function PortfolioAllocationChart({ holdings, baseCurrency = 'INR' }) {
  const { total, data } = useMemo(() => buildAllocation(holdings), [holdings])

  return (
    <section className="allocation-card">
      <div className="allocation-header">
        <h3>Portfolio Allocation</h3>
        {data.length > 0 && (
          <span className="allocation-total">Total: {formatMoney(total, baseCurrency)}</span>
        )}
      </div>

      {data.length === 0 ? (
        <p className="allocation-empty">No holdings with valid current value available.</p>
      ) : (
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie
                data={data}
                dataKey="value"
                nameKey="name"
                innerRadius={72}
                outerRadius={112}
                paddingAngle={1.5}
                minAngle={1}
                labelLine={false}
                label={labelFormatter}
              >
                {data.map((entry, index) => (
                  <Cell
                    key={`${entry.name}-${index}`}
                    fill={ALLOCATION_COLORS[index % ALLOCATION_COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip formatter={(value, name, item) => tooltipFormatter(value, name, item, baseCurrency)} />
              <Legend formatter={legendFormatter} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  )
}
