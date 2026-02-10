import axios from 'axios'

const apiClient = axios.create({
  timeout: 30000,
})

export async function analyzeMarket(payload) {
  const response = await apiClient.post('/api/analyze', payload)
  return response.data
}
