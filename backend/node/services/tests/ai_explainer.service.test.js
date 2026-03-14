jest.mock('openai', () => jest.fn())
jest.mock('../market_intelligence.service', () => ({
  fetchRecentNewsContext: jest.fn(async () => ({ companyHeadlines: [], macroHeadlines: [] })),
}))

const OpenAI = require('openai')
const marketIntelligence = require('../market_intelligence.service')
const aiExplainer = require('../ai_explainer.service')

const mockCreate = jest.fn()

function mockGroqResponse(payload) {
  return {
    choices: [
      {
        message: {
          content: JSON.stringify(payload),
        },
      },
    ],
  }
}

function buildAnalysis(overrides = {}) {
  return {
    symbol: 'AAPL',
    decision: 'BUY',
    probability: 0.62,
    confidence_level: 'medium',
    sentiment: 'positive',
    market_condition: 'BULLISH',
    ...overrides,
  }
}

beforeEach(() => {
  jest.clearAllMocks()
  OpenAI.mockImplementation(() => ({
    chat: {
      completions: {
        create: mockCreate,
      },
    },
  }))
  aiExplainer.__clearState()
  process.env.GROQ_API_KEY = 'test-key'
  delete process.env.GROQ_API_BASE_URL
  delete process.env.GROQ_MODEL
  delete process.env.GROQ_TIMEOUT_MS
  delete process.env.AI_EXPLANATION_CACHE_TTL_MS
})

afterEach(() => {
  jest.useRealTimers()
})

test('generateExplanation and generateMarketInsight share a single Groq request', async () => {
  mockCreate.mockResolvedValue(
    mockGroqResponse({
      explanation: 'Momentum is improving and the model sees a supportive setup.',
      market_insight: 'Broader market tone remains constructive for now.',
    })
  )

  const analysis = buildAnalysis()
  const [explanation, marketInsight] = await Promise.all([
    aiExplainer.generateExplanation(analysis),
    aiExplainer.generateMarketInsight(analysis),
  ])

  expect(explanation).toBe('Momentum is improving and the model sees a supportive setup.')
  expect(marketInsight).toBe('Broader market tone remains constructive for now.')
  expect(mockCreate).toHaveBeenCalledTimes(1)
})

test('generateNarratives returns the recent symbol response while cooldown is active', async () => {
  jest.useFakeTimers().setSystemTime(new Date('2026-03-14T10:00:00.000Z'))
  mockCreate.mockResolvedValue(
    mockGroqResponse({
      explanation: 'The first explanation should be reused during cooldown.',
      market_insight: 'The first market insight should also be reused.',
    })
  )

  const first = await aiExplainer.generateNarratives(buildAnalysis())

  mockCreate.mockResolvedValue(
    mockGroqResponse({
      explanation: 'This second response should never be requested.',
      market_insight: 'This second market insight should never be requested.',
    })
  )

  jest.advanceTimersByTime(aiExplainer.MIN_REQUEST_INTERVAL_MS - 1000)

  const second = await aiExplainer.generateNarratives(
    buildAnalysis({
      decision: 'SELL',
      sentiment: 'negative',
      market_condition: 'BEARISH',
    })
  )

  expect(second).toEqual(first)
  expect(mockCreate).toHaveBeenCalledTimes(1)
})

test('generateNarratives falls back to a non-null explanation on Groq rate limits', async () => {
  mockCreate.mockRejectedValue(new Error('rate limited'))

  const result = await aiExplainer.generateNarratives(buildAnalysis())

  expect(result).toMatchObject({
    explanation: aiExplainer.FALLBACK_EXPLANATION,
    explanationIsFallback: true,
  })
  expect(result.marketInsight).toEqual(expect.any(String))
})

test('generateNarratives includes company and macro headlines in the Groq prompt', async () => {
  marketIntelligence.fetchRecentNewsContext.mockResolvedValue({
    companyHeadlines: [
      'Reliance expands clean energy investment plans.',
      'Telecom subscriber growth stays resilient this quarter.',
      'Retail segment reports steady demand recovery.',
    ],
    macroHeadlines: [
      'Global markets remain volatile as inflation expectations rise.',
      'Oil prices climb amid geopolitical tensions.',
    ],
  })
  mockCreate.mockResolvedValue(
    mockGroqResponse({
      explanation: 'News flow and model signals both support a constructive view.',
      market_insight: 'Recent headlines suggest a stable operating backdrop.',
    })
  )

  await aiExplainer.generateNarratives(
    buildAnalysis({ symbol: 'RELIANCE.NS', prediction_category: 'Strong Buy', confidence_tier: 'High' })
  )

  expect(mockCreate).toHaveBeenCalledTimes(1)
  const requestPayload = mockCreate.mock.calls[0][0]
  expect(requestPayload.messages[1].content).toContain('Recent Company News:')
  expect(requestPayload.messages[1].content).toContain('Recent Macro Market Events:')
  expect(requestPayload.messages[1].content).toContain('Prediction Category: Strong Buy')
  expect(requestPayload.messages[1].content).toContain('Confidence Tier: high')
  expect(requestPayload.messages[1].content).toContain(
    'Only reference headlines when they are clearly relevant to this company or sector.'
  )
  expect(requestPayload.messages[1].content).toContain(
    'If headlines are unrelated, ignore them completely.'
  )
  expect(requestPayload.messages[1].content).toContain(
    'Never force connections between unrelated companies or sectors.'
  )
  expect(requestPayload.messages[1].content).toContain(
    '- Reliance expands clean energy investment plans.'
  )
  expect(requestPayload.messages[1].content).toContain(
    '- Telecom subscriber growth stays resilient this quarter.'
  )
  expect(requestPayload.messages[1].content).toContain(
    '- Retail segment reports steady demand recovery.'
  )
  expect(requestPayload.messages[1].content).toContain(
    '- Global markets remain volatile as inflation expectations rise.'
  )
  expect(requestPayload.messages[1].content).toContain(
    '- Oil prices climb amid geopolitical tensions.'
  )
})

test('generateNarratives recovers explanation fields from loosely formatted JSON output', async () => {
  mockCreate.mockResolvedValue({
    choices: [
      {
        message: {
          content:
            '{\n"explanation": "Line one of the explanation.\n\nLine two keeps the same field.",\n"market_insight": "Recent context remains mixed but stable."\n}',
        },
      },
    ],
  })

  const result = await aiExplainer.generateNarratives(buildAnalysis())

  expect(result).toMatchObject({
    explanation: 'Line one of the explanation.\n\nLine two keeps the same field.',
    marketInsight: 'Recent context remains mixed but stable.',
    explanationIsFallback: false,
  })
})

test('generateNarratives falls back when model output is malformed JSON-like text', async () => {
  mockCreate.mockResolvedValue({
    choices: [
      {
        message: {
          content: '{"explanation": , "market_insight": "Missing explanation value"}',
        },
      },
    ],
  })

  const result = await aiExplainer.generateNarratives(buildAnalysis())

  expect(result.explanation).toBe(aiExplainer.FALLBACK_EXPLANATION)
  expect(result.explanationIsFallback).toBe(true)
  expect(result.marketInsight).toEqual(expect.any(String))
  expect(result.explanation).not.toContain('{"explanation"')
})
