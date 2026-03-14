const VALID_SEVERITIES = new Set(['low', 'medium', 'high']);

const TICKER_TO_SECTOR = {
  AAPL: 'Technology',
  NVDA: 'Technology',
  MSFT: 'Technology',
  TCS: 'Technology',
  'TCS.NS': 'Technology',
  INFY: 'Technology',
  'INFY.NS': 'Technology',
  HDFCBANK: 'Banking',
  'HDFCBANK.NS': 'Banking',
  ICICIBANK: 'Banking',
  'ICICIBANK.NS': 'Banking',
  SBIN: 'Banking',
  'SBIN.NS': 'Banking',
  RELIANCE: 'Energy',
  'RELIANCE.NS': 'Energy',
  XOM: 'Energy',
  CVX: 'Energy',
  AMZN: 'Consumer',
  WMT: 'Consumer',
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function toPositiveNumber(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return null;
  }

  return numericValue;
}

function normalizeTicker(ticker) {
  if (typeof ticker !== 'string') {
    return 'UNKNOWN';
  }

  const normalized = ticker.trim().toUpperCase();
  return normalized || 'UNKNOWN';
}

function normalizeTickerRoot(ticker) {
  return ticker.replace(/\.(NS|BO)$/i, '');
}

function normalizeCurrency(value) {
  if (typeof value !== 'string') {
    return null;
  }

  const normalized = value.trim().toUpperCase();
  return normalized || null;
}

function formatPercent(value) {
  return Number(value.toFixed(1));
}

function addInsight(insights, type, severity, message) {
  if (!VALID_SEVERITIES.has(severity)) {
    return;
  }

  if (typeof message !== 'string' || !message.trim()) {
    return;
  }

  insights.push({
    type,
    severity,
    message: message.trim(),
  });
}

function getHoldingValue(holding) {
  return toPositiveNumber(holding?.market_value_base) ?? toPositiveNumber(holding?.current_value);
}

function getHoldingCurrentPrice(holding) {
  return (
    toPositiveNumber(holding?.price_native) ??
    toPositiveNumber(holding?.current_price) ??
    toPositiveNumber(holding?.price_base)
  );
}

function mapTickerToSector(ticker) {
  const normalizedTicker = normalizeTicker(ticker);
  const tickerRoot = normalizeTickerRoot(normalizedTicker);
  return TICKER_TO_SECTOR[normalizedTicker] || TICKER_TO_SECTOR[tickerRoot] || null;
}

function normalizeSignal(signal) {
  if (typeof signal !== 'string') {
    return null;
  }

  const normalized = signal.trim().toLowerCase();
  if (!normalized) {
    return null;
  }

  if (normalized.includes('sell') || normalized.includes('bearish')) {
    return 'bearish';
  }

  if (normalized.includes('buy')) {
    return 'buy';
  }

  if (normalized.includes('hold') || normalized.includes('neutral')) {
    return 'hold';
  }

  return null;
}

function getHoldingSignal(holding) {
  const candidates = [
    holding?.signal,
    holding?.prediction_signal,
    holding?.predicted_signal,
    holding?.model_signal,
    holding?.recommendation,
    holding?.prediction?.signal,
    holding?.prediction?.recommendation,
    holding?.analysis?.signal,
    holding?.analysis?.recommendation,
  ];

  for (const candidate of candidates) {
    const normalized = normalizeSignal(candidate);
    if (normalized) {
      return normalized;
    }
  }

  return null;
}

function resolveBaseCurrency(portfolioData, holdings) {
  const summaryBaseCurrency = normalizeCurrency(portfolioData?.summary?.base_currency);
  if (summaryBaseCurrency) {
    return summaryBaseCurrency;
  }

  const payloadBaseCurrency = normalizeCurrency(portfolioData?.base_currency);
  if (payloadBaseCurrency) {
    return payloadBaseCurrency;
  }

  for (const holding of holdings) {
    const holdingBaseCurrency = normalizeCurrency(holding?.base_currency);
    if (holdingBaseCurrency) {
      return holdingBaseCurrency;
    }
  }

  return null;
}

function generatePortfolioInsights(portfolioData) {
  const payload = portfolioData && typeof portfolioData === 'object' ? portfolioData : {};
  const holdings = Array.isArray(payload.holdings) ? payload.holdings : [];
  const insights = [];
  let diversificationPenalty = 0;

  const valuedHoldings = holdings
    .map((holding) => ({
      holding,
      ticker: normalizeTicker(holding?.ticker),
      value: getHoldingValue(holding),
    }))
    .filter((entry) => Number.isFinite(entry.value) && entry.value > 0);

  const totalPortfolioValue = valuedHoldings.reduce((sum, entry) => sum + Number(entry.value), 0);

  if (Number.isFinite(totalPortfolioValue) && totalPortfolioValue > 0) {
    const largestHolding = valuedHoldings.reduce((currentMax, entry) => {
      if (!currentMax || entry.value > currentMax.value) {
        return entry;
      }
      return currentMax;
    }, null);

    const largestWeight = largestHolding ? largestHolding.value / totalPortfolioValue : 0;
    if (largestHolding && largestWeight > 0.5) {
      const concentrationPercent = formatPercent(largestWeight * 100);
      addInsight(
        insights,
        'concentration_risk',
        'high',
        `Largest position ${largestHolding.ticker} is approximately ${concentrationPercent}% of the portfolio, indicating high concentration risk.`
      );
      diversificationPenalty += 40;
    } else if (largestHolding && largestWeight > 0.35) {
      const concentrationPercent = formatPercent(largestWeight * 100);
      addInsight(
        insights,
        'concentration_risk',
        'medium',
        `Largest position ${largestHolding.ticker} is approximately ${concentrationPercent}% of the portfolio, indicating moderate concentration risk.`
      );
      diversificationPenalty += 20;
    }

    const sectorExposure = new Map();
    for (const entry of valuedHoldings) {
      const sector = mapTickerToSector(entry.ticker);
      if (!sector) {
        continue;
      }

      const previous = Number(sectorExposure.get(sector) || 0);
      sectorExposure.set(sector, previous + entry.value);
    }

    let dominantSector = null;
    let dominantSectorWeight = 0;
    for (const [sector, sectorValue] of sectorExposure.entries()) {
      const sectorWeight = sectorValue / totalPortfolioValue;
      if (sectorWeight > dominantSectorWeight) {
        dominantSector = sector;
        dominantSectorWeight = sectorWeight;
      }
    }

    if (dominantSector && dominantSectorWeight > 0.6) {
      const sectorPercent = formatPercent(dominantSectorWeight * 100);
      addInsight(
        insights,
        'sector_exposure',
        'high',
        `${dominantSector} sector exposure is elevated at approximately ${sectorPercent}% of the portfolio.`
      );
      diversificationPenalty += Math.min(40, 20 + Math.round((dominantSectorWeight - 0.6) * 50));
    }

    const baseCurrency = resolveBaseCurrency(payload, holdings);
    if (baseCurrency) {
      let foreignExposureValue = 0;
      for (const entry of valuedHoldings) {
        const instrumentCurrency = normalizeCurrency(entry.holding?.instrument_currency);
        if (instrumentCurrency && instrumentCurrency !== baseCurrency) {
          foreignExposureValue += entry.value;
        }
      }

      const foreignExposureWeight = foreignExposureValue / totalPortfolioValue;
      if (foreignExposureWeight > 0.5) {
        const foreignExposurePercent = formatPercent(foreignExposureWeight * 100);
        addInsight(
          insights,
          'currency_exposure',
          'medium',
          `Foreign currency exposure is approximately ${foreignExposurePercent}% versus base currency ${baseCurrency}.`
        );
      }
    }
  }

  for (const holding of holdings) {
    const ticker = normalizeTicker(holding?.ticker);
    const buyPrice = toPositiveNumber(holding?.buy_price);
    const currentPrice = getHoldingCurrentPrice(holding);

    if (!buyPrice || !currentPrice) {
      continue;
    }

    const gainPercent = ((currentPrice - buyPrice) / buyPrice) * 100;
    if (gainPercent > 70) {
      addInsight(
        insights,
        'overextended_position',
        'medium',
        `${ticker} is up approximately ${formatPercent(gainPercent)}% from buy price and may warrant a review for profit protection.`
      );
      continue;
    }

    if (gainPercent < -20) {
      addInsight(
        insights,
        'drawdown_detection',
        'medium',
        `${ticker} is down approximately ${formatPercent(Math.abs(gainPercent))}% from buy price, indicating sustained drawdown.`
      );
    }
  }

  const analyzedSignals = holdings
    .map((holding) => getHoldingSignal(holding))
    .filter((signal) => typeof signal === 'string');

  if (analyzedSignals.length > 0) {
    const bearishCount = analyzedSignals.filter((signal) => signal === 'bearish').length;
    const bearishWeight = bearishCount / analyzedSignals.length;
    if (bearishWeight > 0.5) {
      addInsight(
        insights,
        'portfolio_signal_alignment',
        'medium',
        `${formatPercent(
          bearishWeight * 100
        )}% of analyzed holdings currently indicate bearish or sell signals.`
      );
    }
  }

  const diversificationScore = clamp(Math.round(100 - diversificationPenalty), 0, 100);

  return {
    diversification_score: diversificationScore,
    insights,
  };
}

module.exports = {
  generatePortfolioInsights,
};
