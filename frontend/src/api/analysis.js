import axios from 'axios'

const apiClient = axios.create({
  timeout: 30000,
})

const HISTORY_DAYS = 30

export async function analyzeMarket(selectedSymbol) {
  const symbol = typeof selectedSymbol === 'string' ? selectedSymbol.trim().toUpperCase() : ''

  if (!symbol) {
    throw new Error('Symbol is required')
  }

  const response = await apiClient.post('/api/analyze', { symbol })
  return response.data
}

export async function fetchRecentAnalyses(limit = 8) {
  const params = Number.isFinite(limit) ? { limit: Math.trunc(limit) } : undefined
  const response = await apiClient.get('/api/analyze/recent', { params })
  return response.data
}

export async function fetchStockHistory(selectedSymbol) {
  const symbol = typeof selectedSymbol === 'string' ? selectedSymbol.trim().toUpperCase() : ''

  if (!symbol) {
    throw new Error('Symbol is required')
  }

  const response = await apiClient.get(`/api/market/history/${encodeURIComponent(symbol)}`, {
    params: { days: HISTORY_DAYS },
  })

  return response.data
}

export async function fetchQuoteSnapshot(selectedSymbol) {
  const symbol = typeof selectedSymbol === 'string' ? selectedSymbol.trim().toUpperCase() : ''

  if (!symbol) {
    throw new Error('Symbol is required')
  }

  const response = await apiClient.get(`/api/market/quote/${encodeURIComponent(symbol)}`)
  return response.data
}

export async function fetchValidationReport(selectedSymbol, options = {}) {
  const symbol = typeof selectedSymbol === 'string' ? selectedSymbol.trim().toUpperCase() : ''

  if (!symbol) {
    throw new Error('Symbol is required')
  }

  const response = await apiClient.post('/api/analyze/validate', {
    symbol,
    start_date: options.start_date,
    end_date: options.end_date,
    interval: options.interval,
    horizon: options.horizon,
  })

  return response.data
}
