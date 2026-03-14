const PREDICTION_LABELS = {
  0: 'SELL',
  1: 'HOLD',
  2: 'BUY',
}

const KNOWN_SECTOR_MAP = {
  AAPL: 'Technology',
  MSFT: 'Technology',
  NVDA: 'Technology',
  AMD: 'Technology',
  META: 'Technology',
  GOOGL: 'Technology',
  GOOG: 'Technology',
  ORCL: 'Technology',
  INFY: 'Technology',
  TCS: 'Technology',
  WIPRO: 'Technology',
  HCLTECH: 'Technology',
  HDFCBANK: 'Financials',
  ICICIBANK: 'Financials',
  SBIN: 'Financials',
  AXISBANK: 'Financials',
  KOTAKBANK: 'Financials',
  JPM: 'Financials',
  BAC: 'Financials',
  GS: 'Financials',
  XOM: 'Energy',
  CVX: 'Energy',
  BP: 'Energy',
  COP: 'Energy',
  RELIANCE: 'Energy',
  ONGC: 'Energy',
  TSLA: 'Automotive',
  GM: 'Automotive',
  F: 'Automotive',
  PFE: 'Healthcare',
  JNJ: 'Healthcare',
  ABBV: 'Healthcare',
  SUNPHARMA: 'Healthcare',
  DRREDDY: 'Healthcare',
  CIPLA: 'Healthcare',
  AMZN: 'Consumer',
  WMT: 'Consumer',
  COST: 'Consumer',
  ITC: 'Consumer',
  HINDUNILVR: 'Consumer',
}

const SUGGESTED_SECTORS = [
  'Technology',
  'Financials',
  'Energy',
  'Healthcare',
  'Consumer',
  'Industrials',
]

function toFiniteNumber(value) {
  const numericValue = Number(value)
  return Number.isFinite(numericValue) ? numericValue : null
}

export function getPredictionLabel(value, fallback = 'N/A') {
  if (typeof value === 'string' && value.trim()) {
    const normalized = value.trim().toUpperCase()
    if (normalized === '0' || normalized === '1' || normalized === '2') {
      return PREDICTION_LABELS[normalized] ?? fallback
    }

    return normalized
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    return PREDICTION_LABELS[value] ?? fallback
  }

  return fallback
}

export function formatPercent(value, digits = 2, { signed = false, scale = 1 } = {}) {
  const numericValue = toFiniteNumber(value)
  if (numericValue === null) {
    return 'N/A'
  }

  const scaledValue = numericValue * scale
  const prefix = signed && scaledValue > 0 ? '+' : ''
  return `${prefix}${scaledValue.toFixed(digits)}%`
}

export function formatNumber(value, digits = 2) {
  const numericValue = toFiniteNumber(value)
  if (numericValue === null) {
    return 'N/A'
  }

  return numericValue.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

export function getCurrencySymbol(currencyCode) {
  const code = typeof currencyCode === 'string' ? currencyCode.trim().toUpperCase() : ''

  if (code === 'USD') {
    return '$'
  }

  if (code === 'INR') {
    return '\u20B9'
  }

  return ''
}

export function formatMoney(value, currencyCode, digits = 2) {
  const numericValue = toFiniteNumber(value)
  if (numericValue === null) {
    return 'N/A'
  }

  const symbol = getCurrencySymbol(currencyCode)
  const formatted = formatNumber(numericValue, digits)
  if (symbol) {
    return `${symbol}${formatted}`
  }

  const fallbackCode =
    typeof currencyCode === 'string' && currencyCode.trim()
      ? currencyCode.trim().toUpperCase()
      : ''
  return fallbackCode ? `${fallbackCode} ${formatted}` : formatted
}

export function formatSignedMoney(value, currencyCode, digits = 2) {
  const numericValue = toFiniteNumber(value)
  if (numericValue === null) {
    return 'N/A'
  }

  const prefix = numericValue > 0 ? '+' : numericValue < 0 ? '-' : ''
  return `${prefix}${formatMoney(Math.abs(numericValue), currencyCode, digits)}`
}

export function formatAxisDate(date) {
  if (typeof date !== 'string') {
    return ''
  }

  const [year, month, day] = date.split('-')
  if (!year || !month || !day) {
    return date
  }

  return `${month}/${day}`
}

export function formatLongDate(date) {
  if (typeof date !== 'string') {
    return 'N/A'
  }

  const parsedDate = new Date(`${date}T00:00:00`)
  if (Number.isNaN(parsedDate.getTime())) {
    return date
  }

  return parsedDate.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function normalizePriceHistory(history) {
  if (!Array.isArray(history)) {
    return []
  }

  const normalized = history
    .map((point) => {
      const date = typeof point?.date === 'string' ? point.date.trim() : ''
      const close = toFiniteNumber(point?.close)
      if (!date || close === null) {
        return null
      }

      return {
        date,
        close,
      }
    })
    .filter(Boolean)
    .sort((left, right) => left.date.localeCompare(right.date))

  return normalized.map((point, index) => {
    const previousClose = index > 0 ? normalized[index - 1]?.close : null
    const dailyChangePct =
      previousClose !== null && previousClose !== 0
        ? ((point.close - previousClose) / previousClose) * 100
        : null

    return {
      ...point,
      dailyChangePct,
      hasLargeMove: dailyChangePct !== null && Math.abs(dailyChangePct) > 2,
      largeMoveClose: dailyChangePct !== null && Math.abs(dailyChangePct) > 2 ? point.close : null,
    }
  })
}

export function normalizeEquityCurve(history) {
  const rawCurve = Array.isArray(history?.equity_curve) ? history.equity_curve : []
  let rollingPeak = 0
  let maxDrawdownPct = 0

  const curve = rawCurve
    .map((point) => {
      const date = typeof point?.date === 'string' ? point.date.trim() : ''
      const portfolioValue = toFiniteNumber(point?.portfolio_value)
      if (!date || portfolioValue === null) {
        return null
      }

      rollingPeak = Math.max(rollingPeak, portfolioValue)
      const drawdownPct =
        rollingPeak > 0 ? ((portfolioValue - rollingPeak) / rollingPeak) * 100 : 0
      maxDrawdownPct = Math.min(maxDrawdownPct, drawdownPct)

      return {
        date,
        portfolio_value: portfolioValue,
        drawdownPct,
      }
    })
    .filter(Boolean)

  return curve.map((point, index) => {
    const previousValue = index > 0 ? curve[index - 1]?.portfolio_value : null
    const dailyChangePct =
      previousValue !== null && previousValue !== 0
        ? ((point.portfolio_value - previousValue) / previousValue) * 100
        : null

    return {
      ...point,
      dailyChangePct,
      hasLargeMove: dailyChangePct !== null && Math.abs(dailyChangePct) > 2,
      largeMoveValue:
        dailyChangePct !== null && Math.abs(dailyChangePct) > 2 ? point.portfolio_value : null,
      maxDrawdownPct,
    }
  })
}

function inferSectorFromTicker(ticker) {
  const normalizedTicker =
    typeof ticker === 'string' && ticker.trim() ? ticker.trim().toUpperCase() : 'UNKNOWN'
  const [baseTicker] = normalizedTicker.split('.')

  if (KNOWN_SECTOR_MAP[baseTicker]) {
    return KNOWN_SECTOR_MAP[baseTicker]
  }

  if (baseTicker.includes('BANK') || baseTicker.includes('FIN')) {
    return 'Financials'
  }

  if (
    baseTicker.includes('TECH') ||
    baseTicker.includes('SOFT') ||
    baseTicker.includes('INFO')
  ) {
    return 'Technology'
  }

  if (
    baseTicker.includes('OIL') ||
    baseTicker.includes('ENER') ||
    baseTicker.includes('GAS')
  ) {
    return 'Energy'
  }

  if (
    baseTicker.includes('PHARMA') ||
    baseTicker.includes('HEALTH') ||
    baseTicker.includes('MED')
  ) {
    return 'Healthcare'
  }

  if (
    baseTicker.includes('AUTO') ||
    baseTicker.includes('MOTOR') ||
    baseTicker.includes('CAR')
  ) {
    return 'Automotive'
  }

  if (
    baseTicker.includes('CONSUM') ||
    baseTicker.includes('RETAIL') ||
    baseTicker.includes('SHOP')
  ) {
    return 'Consumer'
  }

  return 'Other'
}

function calculateStandardDeviation(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return 0
  }

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
  return Math.sqrt(variance)
}

export function computePortfolioAnalytics(holdings, history, insights) {
  const normalizedHoldings = Array.isArray(holdings)
    ? holdings
        .map((holding) => {
          const currentValue = toFiniteNumber(holding?.current_value ?? holding?.market_value_base)
          const profitLoss = toFiniteNumber(holding?.profit_loss)
          if (currentValue === null || currentValue <= 0) {
            return null
          }

          return {
            ticker: typeof holding?.ticker === 'string' ? holding.ticker.trim().toUpperCase() : 'UNKNOWN',
            currentValue,
            profitLoss: profitLoss ?? 0,
            sector: inferSectorFromTicker(holding?.ticker),
          }
        })
        .filter(Boolean)
    : []

  const totalValue = normalizedHoldings.reduce((sum, holding) => sum + holding.currentValue, 0)
  const sortedHoldings = [...normalizedHoldings].sort(
    (left, right) => right.currentValue - left.currentValue
  )
  const topHolding = sortedHoldings[0] ?? null
  const largestHoldingPercent =
    topHolding && totalValue > 0 ? (topHolding.currentValue / totalValue) * 100 : 0
  const topThreeWeight =
    totalValue > 0
      ? sortedHoldings
          .slice(0, 3)
          .reduce((sum, holding) => sum + holding.currentValue, 0) / totalValue * 100
      : 0

  const sectorExposureMap = normalizedHoldings.reduce((map, holding) => {
    const previousValue = map.get(holding.sector) || 0
    map.set(holding.sector, previousValue + holding.currentValue)
    return map
  }, new Map())

  const sectorExposure = [...sectorExposureMap.entries()]
    .map(([sector, value]) => ({
      sector,
      value,
      weight: totalValue > 0 ? (value / totalValue) * 100 : 0,
    }))
    .sort((left, right) => right.value - left.value)

  const sectorDiversity = sectorExposure.length
  const normalizedCurve = normalizeEquityCurve(history)
  const dailyReturns = normalizedCurve
    .map((point) =>
      typeof point.dailyChangePct === 'number' && Number.isFinite(point.dailyChangePct)
        ? point.dailyChangePct / 100
        : null
    )
    .filter((value) => value !== null)
  const volatilityPercent = calculateStandardDeviation(dailyReturns) * 100
  const maxDrawdownPct = normalizedCurve.reduce(
    (minimumValue, point) =>
      typeof point.drawdownPct === 'number' && point.drawdownPct < minimumValue
        ? point.drawdownPct
        : minimumValue,
    0
  )

  const diversificationScore =
    typeof insights?.diversification_score === 'number' && Number.isFinite(insights.diversification_score)
      ? insights.diversification_score
      : totalValue > 0
        ? 1 /
          normalizedHoldings.reduce((sum, holding) => {
            const weight = holding.currentValue / totalValue
            return sum + weight ** 2
          }, 0)
        : 0

  const gainers = normalizedHoldings.filter((holding) => holding.profitLoss > 0)
  const losers = normalizedHoldings.filter((holding) => holding.profitLoss < 0)
  const flat = normalizedHoldings.length - gainers.length - losers.length

  let riskPoints = 0
  if (largestHoldingPercent >= 50) {
    riskPoints += 2
  } else if (largestHoldingPercent >= 35) {
    riskPoints += 1
  }

  if (volatilityPercent >= 2) {
    riskPoints += 2
  } else if (volatilityPercent >= 1) {
    riskPoints += 1
  }

  if (sectorDiversity <= 1) {
    riskPoints += 2
  } else if (sectorDiversity <= 2) {
    riskPoints += 1
  }

  if (diversificationScore < 2) {
    riskPoints += 2
  } else if (diversificationScore < 4) {
    riskPoints += 1
  }

  if (Math.abs(maxDrawdownPct) >= 10) {
    riskPoints += 2
  } else if (Math.abs(maxDrawdownPct) >= 5) {
    riskPoints += 1
  }

  const riskScore = riskPoints >= 6 ? 'High' : riskPoints >= 3 ? 'Moderate' : 'Low'

  return {
    totalValue,
    topHolding,
    largestHoldingPercent,
    topThreeWeight,
    sectorExposure,
    sectorDiversity,
    volatilityPercent,
    diversificationScore,
    maxDrawdownPct,
    riskScore,
    gainers: gainers.length,
    losers: losers.length,
    flat,
  }
}

export function buildPortfolioAdvisorInsights(analytics, backendRecommendations = []) {
  const recommendations = Array.isArray(backendRecommendations)
    ? backendRecommendations.filter((item) => typeof item === 'string' && item.trim())
    : []

  const messages = []

  if (analytics.topHolding && analytics.largestHoldingPercent >= 50) {
    messages.push(
      `Your portfolio is heavily concentrated in ${analytics.topHolding.ticker} (${formatPercent(
        analytics.largestHoldingPercent,
        1
      )}).`
    )
  } else if (analytics.topHolding && analytics.largestHoldingPercent >= 35) {
    messages.push(
      `${analytics.topHolding.ticker} is your largest holding at ${formatPercent(
        analytics.largestHoldingPercent,
        1
      )}, so position sizing should be monitored closely.`
    )
  }

  if (analytics.sectorExposure.length > 0) {
    const dominantSector = analytics.sectorExposure[0]
    if (dominantSector.weight >= 45) {
      const missingSectors = SUGGESTED_SECTORS.filter(
        (sector) => !analytics.sectorExposure.some((entry) => entry.sector === sector)
      ).slice(0, 2)
      const diversificationTarget =
        missingSectors.length > 0 ? missingSectors.join(' or ') : 'additional sectors'
      messages.push(
        `${dominantSector.sector} makes up ${formatPercent(
          dominantSector.weight,
          1
        )} of the portfolio. Diversifying into ${diversificationTarget} could reduce risk.`
      )
    }
  }

  if (analytics.topThreeWeight >= 75) {
    messages.push(
      `Your top three holdings account for ${formatPercent(
        analytics.topThreeWeight,
        1
      )} of portfolio value, so a review of concentration limits may help.`
    )
  }

  if (analytics.losers > analytics.gainers) {
    messages.push(
      'More holdings are in loss than in profit right now. Reviewing weaker positions may improve portfolio quality.'
    )
  } else if (analytics.gainers > 0 && analytics.losers === 0) {
    messages.push('Most positions are currently profitable, which gives you flexibility to rebalance from strength.')
  }

  const deduped = [...messages, ...recommendations].filter(
    (message, index, collection) => collection.indexOf(message) === index
  )

  if (deduped.length > 0) {
    return deduped
  }

  return ['Portfolio allocation looks balanced. Continue periodic rebalancing.']
}
