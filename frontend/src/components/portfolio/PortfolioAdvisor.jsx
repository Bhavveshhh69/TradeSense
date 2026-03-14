function getRecommendationIcon(recommendation) {
  const text = typeof recommendation === 'string' ? recommendation.toLowerCase() : ''

  if (text.includes('reduce')) {
    return '!'
  }

  if (text.includes('diversify') || text.includes('diversification')) {
    return '+'
  }

  if (text.includes('rebalance') || text.includes('rebalancing')) {
    return '~'
  }

  return '>'
}

export default function PortfolioAdvisor({ data, title = 'Portfolio Advisor' }) {
  const recommendations = Array.isArray(data?.recommendations)
    ? data.recommendations.filter((item) => typeof item === 'string' && item.trim())
    : []

  return (
    <section className="portfolio-advisor-card advisor-card">
      <div className="portfolio-advisor-header">
        <h3>{title}</h3>
      </div>
      {recommendations.length > 0 ? (
        <ul className="portfolio-advisor-list">
          {recommendations.map((recommendation, index) => (
            <li key={`${recommendation}-${index}`}>
              <span className="portfolio-advisor-icon" aria-hidden="true">
                {getRecommendationIcon(recommendation)}
              </span>
              <span>{recommendation}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="portfolio-advisor-empty">No recommendations available.</p>
      )}
    </section>
  )
}
