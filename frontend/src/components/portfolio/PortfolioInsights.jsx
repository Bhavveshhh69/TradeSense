import { formatMoney, formatPercent } from '../../utils/dashboard'

const EMPTY_INSIGHTS = {
  concentration_risk: 'LOW',
  largest_position: null,
  best_performer: null,
  worst_performer: null,
  diversification_score: 0,
  volatility_level: 'LOW',
  insights: [],
}

function formatWeight(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }

  const normalized = value <= 1 ? value * 100 : value
  return `${normalized.toFixed(2)}%`
}

function getRiskClass(value) {
  if (value === 'HIGH' || value === 'High') {
    return 'risk-high'
  }

  if (value === 'MODERATE' || value === 'Moderate') {
    return 'risk-moderate'
  }

  return 'risk-low'
}

function getDiversificationLabel(score) {
  if (score < 2) {
    return 'Poor'
  }

  if (score < 4) {
    return 'Moderate'
  }

  return 'Good'
}

export default function PortfolioInsights({ data, metrics, baseCurrency = 'INR' }) {
  const normalizedData = data && typeof data === 'object' ? data : EMPTY_INSIGHTS
  const diversificationScore =
    typeof normalizedData.diversification_score === 'number' &&
    Number.isFinite(normalizedData.diversification_score)
      ? normalizedData.diversification_score
      : 0
  const insights = Array.isArray(normalizedData.insights) ? normalizedData.insights : []
  const sectorExposure = Array.isArray(metrics?.sectorExposure) ? metrics.sectorExposure : []

  return (
    <section className="portfolio-insights-card insights-card">
      <div className="portfolio-insights-header">
        <h3>Portfolio Risk Metrics</h3>
      </div>

      <div className="portfolio-insights-groups">
        <div className="portfolio-insights-group">
          <h4 className="portfolio-insights-group-title">Core Metrics</h4>
          <div className="portfolio-insights-grid">
            <article className="portfolio-insights-item">
              <span>Concentration Risk</span>
              <strong className={getRiskClass(normalizedData.concentration_risk)}>
                {normalizedData.concentration_risk}
              </strong>
            </article>
            <article className="portfolio-insights-item">
              <span>Volatility</span>
              <strong className={getRiskClass(normalizedData.volatility_level)}>
                {formatPercent(metrics?.volatilityPercent, 2)}
              </strong>
            </article>
            <article className="portfolio-insights-item">
              <span>Diversification Score</span>
              <strong>
                {diversificationScore.toFixed(2)} ({getDiversificationLabel(diversificationScore)})
              </strong>
            </article>
            <article className="portfolio-insights-item">
              <span>Largest Holding</span>
              <strong>
                {normalizedData.largest_position
                  ? `${normalizedData.largest_position.ticker} (${formatWeight(normalizedData.largest_position.weight)})`
                  : 'N/A'}
              </strong>
            </article>
            <article className="portfolio-insights-item">
              <span>Risk Score</span>
              <strong className={getRiskClass(metrics?.riskScore)}>{metrics?.riskScore ?? 'Low'}</strong>
            </article>
            <article className="portfolio-insights-item">
              <span>Max Drawdown</span>
              <strong>{formatPercent(metrics?.maxDrawdownPct, 2)}</strong>
            </article>
          </div>
        </div>

        <div className="portfolio-insights-group">
          <h4 className="portfolio-insights-group-title">Performance</h4>
          <div className="portfolio-insights-grid">
            <article className="portfolio-insights-item">
              <span>Best Performer</span>
              <strong>
                {normalizedData.best_performer
                  ? `${normalizedData.best_performer.ticker} (${formatPercent(normalizedData.best_performer.profit_loss_percent, 2, { signed: true })})`
                  : 'N/A'}
              </strong>
            </article>
            <article className="portfolio-insights-item">
              <span>Worst Performer</span>
              <strong>
                {normalizedData.worst_performer
                  ? `${normalizedData.worst_performer.ticker} (${formatPercent(normalizedData.worst_performer.profit_loss_percent, 2, { signed: true })})`
                  : 'N/A'}
              </strong>
            </article>
            <article className="portfolio-insights-item">
              <span>Profit/Loss Spread</span>
              <strong>
                {metrics?.gainers ?? 0} gainers / {metrics?.losers ?? 0} losers / {metrics?.flat ?? 0} flat
              </strong>
            </article>
          </div>
        </div>
      </div>

      <div className="portfolio-insights-list-wrap">
        <h4>Sector Exposure</h4>
        {sectorExposure.length > 0 ? (
          <ul className="sector-exposure-list">
            {sectorExposure.map((sector) => (
              <li key={sector.sector}>
                <span>{sector.sector}</span>
                <strong>
                  {formatPercent(sector.weight, 1)} • {formatMoney(sector.value, baseCurrency)}
                </strong>
              </li>
            ))}
          </ul>
        ) : (
          <p className="portfolio-insights-empty">No sector exposure available.</p>
        )}
      </div>

      <div className="portfolio-insights-list-wrap">
        <h4>Insights</h4>
        {insights.length > 0 ? (
          <ul className="portfolio-insights-list">
            {insights.map((insight, index) => (
              <li key={`${insight}-${index}`}>{insight}</li>
            ))}
          </ul>
        ) : (
          <p className="portfolio-insights-empty">No insights available.</p>
        )}
      </div>
    </section>
  )
}
