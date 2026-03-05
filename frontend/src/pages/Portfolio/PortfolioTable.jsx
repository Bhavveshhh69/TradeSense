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

function formatCurrencyValue(value, currencyCode) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  const symbol = getCurrencySymbol(currencyCode)
  if (symbol) {
    return `${symbol}${formatNumber(value)}`
  }

  const code = typeof currencyCode === 'string' && currencyCode.trim()
    ? currencyCode.trim().toUpperCase()
    : ''
  return code ? `${code} ${formatNumber(value)}` : formatNumber(value)
}

function toFiniteNumber(value) {
  if (value === null || value === undefined || value === '') {
    return Number.NaN
  }

  const normalized = Number(value)
  return Number.isFinite(normalized) ? normalized : Number.NaN
}

function formatSignedNumber(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  const prefix = value > 0 ? '+' : ''
  return `${prefix}${formatNumber(value)}`
}

function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  const prefix = value > 0 ? '+' : ''
  return `${prefix}${value.toFixed(2)}%`
}

function getProfitLossClass(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'profit-loss-neutral'
  }

  if (value > 0) {
    return 'profit-loss-positive'
  }

  if (value < 0) {
    return 'profit-loss-negative'
  }

  return 'profit-loss-neutral'
}

export default function PortfolioTable({ holdings, onDelete, deletingId }) {
  const holdingsSnapshot = Array.isArray(holdings) ? holdings.map((holding) => ({ ...holding })) : []

  if (!holdingsSnapshot.length) {
    return <div className="portfolio-empty">No holdings added yet.</div>
  }

  return (
    <div className="portfolio-table-wrap">
      <table className="portfolio-table">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Shares</th>
            <th>Buy Price</th>
            <th>Current Price</th>
            <th>Current Value</th>
            <th>Profit/Loss</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {holdingsSnapshot.map((holding) => {
            const {
              id,
              ticker,
              shares,
              buy_price: buyPrice,
              price_native: priceNative,
              current_price: currentPrice,
              market_value_base: marketValueBase,
              current_value: currentValue,
              pnl,
              profit_loss: profitLoss,
              pnl_percent,
              profit_loss_percent: profitLossPercent,
              instrument_currency: instrumentCurrency,
            } = holding || {}

            console.log('render row', ticker, priceNative)

            const sharesValue = toFiniteNumber(shares)
            const buyPriceValue = toFiniteNumber(buyPrice)
            const currentPriceValue = toFiniteNumber(priceNative ?? currentPrice)
            const currentValueValue = toFiniteNumber(marketValueBase ?? currentValue)
            const profitLossValue = toFiniteNumber(pnl ?? profitLoss)
            const profitLossPercentValue = toFiniteNumber(pnl_percent ?? profitLossPercent)

            return (
              <tr key={ticker}>
                <td>{ticker || 'N/A'}</td>
                <td>{formatNumber(sharesValue)}</td>
                <td>{formatCurrencyValue(buyPriceValue, instrumentCurrency)}</td>
                <td>{formatCurrencyValue(currentPriceValue, instrumentCurrency)}</td>
                <td>{formatNumber(currentValueValue)}</td>
                <td className={getProfitLossClass(profitLossValue)}>
                  {formatSignedNumber(profitLossValue)} ({formatPercent(profitLossPercentValue)})
                </td>
                <td>
                  <button
                    type="button"
                    className="button button-danger"
                    onClick={() => onDelete(id)}
                    disabled={deletingId === id}
                  >
                    {deletingId === id ? 'Deleting...' : 'Delete'}
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
