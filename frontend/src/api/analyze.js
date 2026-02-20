import axios from 'axios'

const apiClient = axios.create({
  timeout: 30000,
})

export async function analyzeMarket(selectedSymbol) {
  const symbol = typeof selectedSymbol === 'string' ? selectedSymbol.trim().toUpperCase() : ''

  if (!symbol) {
    throw new Error('Symbol is required')
  }

  const response = await apiClient.post('/api/analyze', { symbol })
  return response.data
}
