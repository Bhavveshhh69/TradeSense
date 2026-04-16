import axios from 'axios'

const apiClient = axios.create({
  timeout: 30000,
})

export async function fetchCommandCenter() {
  const response = await apiClient.get('/api/command-center')
  return response.data
}
