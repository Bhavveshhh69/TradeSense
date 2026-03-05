import axios from 'axios'

const apiClient = axios.create({
  timeout: 30000,
})

export async function fetchPortfolio() {
  const response = await apiClient.get('/api/portfolio')
  return response.data
}

export async function addPortfolioHolding(payload) {
  const response = await apiClient.post('/api/portfolio/add', payload)
  return response.data
}

export async function deletePortfolioHolding(id) {
  const response = await apiClient.delete(`/api/portfolio/${id}`)
  return response.data
}

export async function fetchPortfolioHistory(days = 30) {
  const params = Number.isFinite(days) ? { days: Math.trunc(days) } : undefined
  const response = await apiClient.get('/api/portfolio/history', { params })
  return response.data
}

export async function fetchPortfolioInsights(days = 30) {
  const params = Number.isFinite(days) ? { days: Math.trunc(days) } : undefined
  const response = await apiClient.get('/api/portfolio/insights', { params })
  return response.data
}

export async function fetchPortfolioAdvisor(days = 30) {
  const params = Number.isFinite(days) ? { days: Math.trunc(days) } : undefined
  const response = await apiClient.get('/api/portfolio/advisor', { params })
  return response.data
}
