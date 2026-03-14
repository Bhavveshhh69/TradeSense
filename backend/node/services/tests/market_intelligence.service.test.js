jest.mock('axios')

const axios = require('axios')
const marketIntelligence = require('../market_intelligence.service')

beforeEach(() => {
  jest.clearAllMocks()
  jest.useRealTimers()
  marketIntelligence.__clearState()
  process.env.FINNHUB_API_KEY = 'test-finnhub-key'
  process.env.ALPHA_VANTAGE_API_KEY = 'test-alpha-key'
  delete process.env.FINNHUB_COMPANY_NEWS_URL
  delete process.env.FINNHUB_TIMEOUT_MS
  delete process.env.ALPHA_VANTAGE_NEWS_URL
  delete process.env.ALPHA_VANTAGE_TIMEOUT_MS
  delete process.env.ALPHA_VANTAGE_MIN_INTERVAL_MS
})

afterEach(() => {
  jest.useRealTimers()
})

test('deriveSectorFromSymbol maps known tickers to expected sectors', () => {
  expect(marketIntelligence.deriveSectorFromSymbol('AAPL')).toBe('technology')
  expect(marketIntelligence.deriveSectorFromSymbol('NVDA')).toBe('technology')
  expect(marketIntelligence.deriveSectorFromSymbol('RELIANCE.NS')).toBe('energy')
  expect(marketIntelligence.deriveSectorFromSymbol('HDFCBANK.NS')).toBe('banks')
  expect(marketIntelligence.deriveSectorFromSymbol('UNKNOWNXYZ')).toBe('general')
})

test('fetchRecentNewsContext filters macro headlines by sector relevance and caches by symbol', async () => {
  jest.useFakeTimers().setSystemTime(new Date('2026-03-14T10:00:00.000Z'))
  axios.get.mockImplementation(async (url) => {
    if (url === 'https://finnhub.io/api/v1/company-news') {
      return {
        data: [
          { headline: 'Reliance expands upstream output plans', datetime: 10 },
          { headline: 'Reliance expands upstream output plans', datetime: 9 },
          { headline: 'Reliance retail unit reports steady growth', datetime: 8 },
          { headline: 'Reliance adds gas transmission assets', datetime: 7 },
        ],
      }
    }

    if (url === 'https://www.alphavantage.co/query') {
      return {
        data: {
          feed: [
            {
              title: 'Oil prices rise after OPEC supply commentary',
              summary: 'Crude markets react to OPEC guidance.',
            },
            {
              title: 'Global chip exports recover in key Asian markets',
              summary: 'Semiconductor demand improves for AI supply chains.',
            },
            {
              title: 'Geopolitical tension keeps crude benchmarks elevated',
              summary: 'Energy traders price geopolitical risk.',
            },
          ],
        },
      }
    }

    throw new Error(`Unexpected URL ${url}`)
  })

  const first = await marketIntelligence.fetchRecentNewsContext('RELIANCE.NS')
  const second = await marketIntelligence.fetchRecentNewsContext('reliance.ns')

  expect(first).toEqual({
    companyHeadlines: [
      'Reliance expands upstream output plans',
      'Reliance retail unit reports steady growth',
      'Reliance adds gas transmission assets',
    ],
    macroHeadlines: [
      'Oil prices rise after OPEC supply commentary',
      'Geopolitical tension keeps crude benchmarks elevated',
    ],
  })
  expect(second).toEqual(first)
  expect(axios.get).toHaveBeenCalledTimes(2)
  expect(axios.get).toHaveBeenCalledWith(
    'https://www.alphavantage.co/query',
    expect.objectContaining({
      params: expect.objectContaining({
        function: 'NEWS_SENTIMENT',
        sort: 'LATEST',
        limit: 50,
        apikey: 'test-alpha-key',
        keywords: expect.stringContaining('oil prices'),
      }),
      timeout: expect.any(Number),
    })
  )
})

test('fetchRecentNewsContext returns empty lists when both API keys are missing', async () => {
  delete process.env.FINNHUB_API_KEY
  delete process.env.ALPHA_VANTAGE_API_KEY

  const result = await marketIntelligence.fetchRecentNewsContext('RELIANCE.NS')

  expect(result).toEqual({ companyHeadlines: [], macroHeadlines: [] })
  expect(axios.get).not.toHaveBeenCalled()
})

test('fetchRecentNewsContext returns only macro headlines when Finnhub fails', async () => {
  axios.get.mockImplementation(async (url) => {
    if (url === 'https://finnhub.io/api/v1/company-news') {
      throw new Error('finnhub down')
    }

    if (url === 'https://www.alphavantage.co/query') {
      return {
        data: {
          feed: [
            {
              title: 'AI infrastructure spending remains elevated',
              summary: 'Technology firms raise AI capex this quarter.',
            },
            {
              title: 'Retail inflation prints surprise to the upside',
              summary: 'This headline is not technology-specific.',
            },
          ],
        },
      }
    }

    throw new Error(`Unexpected URL ${url}`)
  })

  const result = await marketIntelligence.fetchRecentNewsContext('AAPL')

  expect(result).toEqual({
    companyHeadlines: [],
    macroHeadlines: ['AI infrastructure spending remains elevated'],
  })
})

test('fetchRecentNewsContext reuses macro cache for symbols in the same sector', async () => {
  axios.get.mockImplementation(async (url, config) => {
    if (url === 'https://finnhub.io/api/v1/company-news') {
      return {
        data: [{ headline: `${config.params.symbol} company headline`, datetime: 10 }],
      }
    }

    if (url === 'https://www.alphavantage.co/query') {
      return {
        data: {
          feed: [
            {
              title: 'Semiconductor tariffs remain a policy risk',
              summary: 'Technology supply chains remain sensitive to tariffs.',
            },
          ],
        },
      }
    }

    throw new Error(`Unexpected URL ${url}`)
  })

  const aapl = await marketIntelligence.fetchRecentNewsContext('AAPL')
  const nvda = await marketIntelligence.fetchRecentNewsContext('NVDA')

  expect(aapl.macroHeadlines).toEqual(['Semiconductor tariffs remain a policy risk'])
  expect(nvda.macroHeadlines).toEqual(['Semiconductor tariffs remain a policy risk'])

  const finnhubCalls = axios.get.mock.calls.filter(
    ([url]) => url === 'https://finnhub.io/api/v1/company-news'
  )
  const alphaCalls = axios.get.mock.calls.filter(
    ([url]) => url === 'https://www.alphavantage.co/query'
  )
  expect(finnhubCalls).toHaveLength(2)
  expect(alphaCalls).toHaveLength(1)
})

test('macro filtering logs relevant and discarded headlines for diagnostics', async () => {
  const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {})
  axios.get.mockImplementation(async (url) => {
    if (url === 'https://finnhub.io/api/v1/company-news') {
      return {
        data: [{ headline: 'HDFCBANK expands branch network', datetime: 1 }],
      }
    }

    if (url === 'https://www.alphavantage.co/query') {
      return {
        data: {
          feed: [
            {
              title: 'RBI policy outlook keeps bank funding costs in focus',
              summary: 'Interest rates remain key for lenders.',
            },
            {
              title: 'Semiconductor cycle enters a stronger phase',
              summary: 'Chip demand rises in AI workloads.',
            },
          ],
        },
      }
    }

    throw new Error(`Unexpected URL ${url}`)
  })

  await marketIntelligence.fetchRecentNewsContext('HDFCBANK.NS')

  expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('macro relevant [banks]'))
  expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('macro discarded [banks]'))
  expect(logSpy).toHaveBeenCalledWith(expect.stringContaining('macro headlines filtered'))
  logSpy.mockRestore()
})
