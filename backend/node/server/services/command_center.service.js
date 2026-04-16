const portfolioService = require('../../portfolio/portfolio.service');
const recentAnalysisService = require('./recent_analysis.service');
const marketIntelligenceService = require('../../services/market_intelligence.service');

function getTimeParts(timeZone) {
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour12: false,
    weekday: 'short',
    hour: '2-digit',
    minute: '2-digit',
  });
  const parts = formatter.formatToParts(new Date());
  const values = Object.create(null);

  for (const part of parts) {
    if (part.type !== 'literal') {
      values[part.type] = part.value;
    }
  }

  return values;
}

function isWeekday(weekday) {
  return ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'].includes(weekday);
}

function toMinutes(hour, minute) {
  return Number(hour) * 60 + Number(minute);
}

function buildIndiaSession() {
  const parts = getTimeParts('Asia/Kolkata');
  const minuteOfDay = toMinutes(parts.hour, parts.minute);
  const tradingOpen = 9 * 60 + 15;
  const tradingClose = 15 * 60 + 30;
  const preOpen = 9 * 60;

  let sessionStatus = 'closed';
  if (isWeekday(parts.weekday)) {
    if (minuteOfDay >= tradingOpen && minuteOfDay < tradingClose) {
      sessionStatus = 'open';
    } else if (minuteOfDay >= preOpen && minuteOfDay < tradingOpen) {
      sessionStatus = 'pre-open';
    }
  }

  return {
    market: 'IN',
    label: 'India',
    time_zone: 'Asia/Kolkata',
    local_time: `${parts.hour}:${parts.minute}`,
    session_status: sessionStatus,
    is_open: sessionStatus === 'open',
    opens_at: '09:15',
    closes_at: '15:30',
  };
}

function buildUsSession() {
  const parts = getTimeParts('America/New_York');
  const minuteOfDay = toMinutes(parts.hour, parts.minute);
  const preMarketOpen = 4 * 60;
  const tradingOpen = 9 * 60 + 30;
  const tradingClose = 16 * 60;
  const afterHoursClose = 20 * 60;

  let sessionStatus = 'closed';
  if (isWeekday(parts.weekday)) {
    if (minuteOfDay >= tradingOpen && minuteOfDay < tradingClose) {
      sessionStatus = 'open';
    } else if (minuteOfDay >= preMarketOpen && minuteOfDay < tradingOpen) {
      sessionStatus = 'pre-market';
    } else if (minuteOfDay >= tradingClose && minuteOfDay < afterHoursClose) {
      sessionStatus = 'after-hours';
    }
  }

  return {
    market: 'US',
    label: 'United States',
    time_zone: 'America/New_York',
    local_time: `${parts.hour}:${parts.minute}`,
    session_status: sessionStatus,
    is_open: sessionStatus === 'open',
    opens_at: '09:30',
    closes_at: '16:00',
  };
}

function buildRiskHeadline(portfolioInsights) {
  const largestPosition = portfolioInsights?.largest_position;
  const concentrationRisk = portfolioInsights?.concentration_risk;
  const ticker =
    typeof largestPosition?.ticker === 'string' && largestPosition.ticker.trim()
      ? largestPosition.ticker.trim()
      : null;
  const weight = Number(largestPosition?.weight);

  if (concentrationRisk === 'HIGH' && ticker && Number.isFinite(weight)) {
    return `${ticker} is carrying ${weight.toFixed(1)}% of gross exposure.`;
  }

  if (concentrationRisk === 'MODERATE' && ticker && Number.isFinite(weight)) {
    return `${ticker} remains the largest allocation at ${weight.toFixed(1)}% of gross exposure.`;
  }

  if (typeof portfolioInsights?.volatility_level === 'string') {
    return `Portfolio volatility is ${portfolioInsights.volatility_level.toLowerCase()} based on recent equity swings.`;
  }

  return 'Portfolio risk is currently balanced with no immediate concentration alert.';
}

function buildDailyBrief({
  portfolio,
  portfolioInsights,
  advisor,
  recentSignals,
  newsContext,
  marketSessions,
}) {
  const summary = portfolio?.summary || {};
  const activePositions = Number(summary.active_positions || 0);
  const longPositions = Number(summary.long_positions || 0);
  const shortPositions = Number(summary.short_positions || 0);
  const topAction =
    Array.isArray(advisor?.recommendations) && advisor.recommendations.length > 0
      ? advisor.recommendations[0]
      : 'No immediate rebalance action is required.';
  const latestSignal = recentSignals[0];
  const marketOpenCount = marketSessions.filter((session) => session.is_open).length;
  const signalLine = latestSignal
    ? `${latestSignal.display_name || latestSignal.normalized} last scored ${latestSignal.signal}.`
    : 'No recent analysis is stored yet.';
  const companyHeadline = newsContext?.companyHeadlines?.[0] || null;
  const macroHeadline = newsContext?.macroHeadlines?.[0] || null;

  const bullets = [
    `${activePositions} active positions are live across ${longPositions} longs and ${shortPositions} shorts.`,
    topAction,
    signalLine,
    marketOpenCount > 0 ? `${marketOpenCount} tracked market sessions are live right now.` : 'Both tracked market sessions are closed right now.',
  ];

  if (companyHeadline) {
    bullets.push(`Company news: ${companyHeadline}`);
  }

  if (macroHeadline) {
    bullets.push(`Macro watch: ${macroHeadline}`);
  }

  return {
    headline: buildRiskHeadline(portfolioInsights),
    bullets,
  };
}

async function getCommandCenter() {
  const [portfolio, portfolioInsights, advisor, recentSignals] = await Promise.all([
    portfolioService.getHoldings(),
    portfolioService.getPortfolioInsights(30),
    portfolioService.getPortfolioAdvisor(30),
    recentAnalysisService.listRecentAnalyses(8),
  ]);

  const marketSessions = [buildIndiaSession(), buildUsSession()];
  const largestPositionTicker = portfolioInsights?.largest_position?.ticker;
  let newsContext = { companyHeadlines: [], macroHeadlines: [] };

  if (typeof largestPositionTicker === 'string' && largestPositionTicker.trim()) {
    newsContext = await marketIntelligenceService.fetchRecentNewsContext(largestPositionTicker);
  }

  return {
    generated_at: new Date().toISOString(),
    market_sessions: marketSessions,
    portfolio_summary: portfolio?.summary || {},
    portfolio,
    portfolio_intelligence: portfolioInsights,
    advisor,
    risk_headline: buildRiskHeadline(portfolioInsights),
    top_portfolio_action:
      Array.isArray(advisor?.recommendations) && advisor.recommendations.length > 0
        ? advisor.recommendations[0]
        : 'No action required.',
    recent_signals: recentSignals,
    daily_brief: buildDailyBrief({
      portfolio,
      portfolioInsights,
      advisor,
      recentSignals,
      newsContext,
      marketSessions,
    }),
    market_intelligence: newsContext,
  };
}

module.exports = {
  getCommandCenter,
};
