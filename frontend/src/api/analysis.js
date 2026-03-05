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

export async function fetchStockHistory(selectedSymbol) {
  const symbol = typeof selectedSymbol === 'string' ? selectedSymbol.trim().toUpperCase() : ''

  if (!symbol) {
    throw new Error('Symbol is required')
  }

  const response = await apiClient.get(`/market/history/${encodeURIComponent(symbol)}`, {
    params: { days: HISTORY_DAYS },
  })

  return response.data
}

