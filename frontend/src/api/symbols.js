import axios from 'axios'

const apiClient = axios.create({
  timeout: 30000,
})

export async function searchSymbols(query) {
  const response = await apiClient.get('/api/symbols/search', {
    params: { q: query },
  })
  return response.data
}

export async function normalizeSymbol(symbol) {
  const normalizedInput = typeof symbol === 'string' ? symbol.trim() : ''
  if (!normalizedInput) {
    throw new Error('symbol is required')
  }

  const response = await apiClient.get(
    `/api/symbols/normalize/${encodeURIComponent(normalizedInput)}`,
  )
  return response.data
}
