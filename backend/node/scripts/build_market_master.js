const fs = require('fs/promises');
const path = require('path');
const axios = require('axios');

const OUTPUT_DIR = path.resolve(__dirname, '..', 'data', 'symbols');
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'market_master.json');

const HTTP_TIMEOUT_MS = 30000;
const REQUEST_HEADERS = {
  'User-Agent': 'TradeSense Market Master Builder/1.0',
};

const OFFICIAL_SOURCES = {
  nasdaqListed: 'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt',
  otherListed: 'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt',
  nseEquities: 'https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv',
};

const CURATED_INDICES = [
  {
    symbol: '^GSPC',
    normalized: '^GSPC',
    display_name: 'S&P 500',
    market: 'US',
    exchange: 'INDEX',
    instrument_type: 'Index',
    country: 'US',
    source: 'Curated benchmark index',
  },
  {
    symbol: '^IXIC',
    normalized: '^IXIC',
    display_name: 'Nasdaq Composite',
    market: 'US',
    exchange: 'INDEX',
    instrument_type: 'Index',
    country: 'US',
    source: 'Curated benchmark index',
  },
  {
    symbol: '^DJI',
    normalized: '^DJI',
    display_name: 'Dow Jones Industrial Average',
    market: 'US',
    exchange: 'INDEX',
    instrument_type: 'Index',
    country: 'US',
    source: 'Curated benchmark index',
  },
  {
    symbol: '^RUT',
    normalized: '^RUT',
    display_name: 'Russell 2000',
    market: 'US',
    exchange: 'INDEX',
    instrument_type: 'Index',
    country: 'US',
    source: 'Curated benchmark index',
  },
  {
    symbol: '^NSEI',
    normalized: '^NSEI',
    display_name: 'Nifty 50',
    market: 'IN',
    exchange: 'NSE',
    instrument_type: 'Index',
    country: 'IN',
    source: 'Curated benchmark index',
  },
  {
    symbol: '^NSEBANK',
    normalized: '^NSEBANK',
    display_name: 'Nifty Bank',
    market: 'IN',
    exchange: 'NSE',
    instrument_type: 'Index',
    country: 'IN',
    source: 'Curated benchmark index',
  },
  {
    symbol: '^CNXIT',
    normalized: '^CNXIT',
    display_name: 'Nifty IT',
    market: 'IN',
    exchange: 'NSE',
    instrument_type: 'Index',
    country: 'IN',
    source: 'Curated benchmark index',
  },
  {
    symbol: '^BSESN',
    normalized: '^BSESN',
    display_name: 'S&P BSE Sensex',
    market: 'IN',
    exchange: 'BSE',
    instrument_type: 'Index',
    country: 'IN',
    source: 'Curated benchmark index',
  },
];

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function csvSplit(line) {
  const cells = [];
  let current = '';
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (inQuotes && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === ',' && !inQuotes) {
      cells.push(current);
      current = '';
      continue;
    }

    current += char;
  }

  cells.push(current);
  return cells.map((cell) => cell.trim());
}

function parseDelimitedText(raw, delimiter) {
  const lines = String(raw)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    return [];
  }

  const headers = lines[0].split(delimiter).map((header) => header.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(delimiter);
    return headers.reduce((record, header, index) => {
      record[header] = normalizeText(values[index]);
      return record;
    }, {});
  });
}

function looksLikeOperatingCompany(name) {
  const upper = normalizeText(name).toUpperCase();
  if (!upper) {
    return false;
  }

  const excludedTokens = [
    ' ETF',
    ' ETN',
    ' FUND',
    ' TRUST',
    ' WARRANT',
    ' WTS',
    ' RIGHT',
    ' RT',
    ' UNIT',
    ' PREFERRED',
    ' PREF',
    ' DEPOSITARY',
    ' INCOME SHARES',
    ' ACQUISITION',
    ' NEXTSHARES',
  ];

  return !excludedTokens.some((token) => upper.includes(token));
}

function buildInstrument({
  symbol,
  normalized,
  displayName,
  market,
  exchange,
  instrumentType,
  country,
  source,
  extraSearchTerms = [],
}) {
  const rawSymbol = normalizeText(symbol).toUpperCase();
  const rawNormalized = normalizeText(normalized || rawSymbol).toUpperCase();
  const rawDisplayName = normalizeText(displayName);
  if (!rawSymbol || !rawNormalized || !rawDisplayName) {
    return null;
  }

  const searchTerms = [
    rawSymbol,
    rawNormalized,
    rawDisplayName,
    normalizeText(exchange).toUpperCase(),
    normalizeText(market).toUpperCase(),
    normalizeText(instrumentType).toUpperCase(),
    ...extraSearchTerms.map((term) => normalizeText(term)).filter(Boolean),
  ]
    .map((term) => term.toUpperCase())
    .filter(Boolean);

  return {
    id: `${normalizeText(market).toUpperCase()}:${rawNormalized}`,
    symbol: rawSymbol,
    normalized: rawNormalized,
    display_name: rawDisplayName,
    market: normalizeText(market).toUpperCase(),
    exchange: normalizeText(exchange).toUpperCase(),
    instrument_type: normalizeText(instrumentType) || 'Equity',
    country: normalizeText(country).toUpperCase(),
    search_terms: [...new Set(searchTerms)],
    source,
  };
}

async function fetchText(url) {
  const response = await axios.get(url, {
    timeout: HTTP_TIMEOUT_MS,
    headers: REQUEST_HEADERS,
    responseType: 'text',
  });
  return String(response.data || '');
}

async function fetchNasdaqListedInstruments() {
  const [nasdaqListedRaw, otherListedRaw] = await Promise.all([
    fetchText(OFFICIAL_SOURCES.nasdaqListed),
    fetchText(OFFICIAL_SOURCES.otherListed),
  ]);

  const instruments = [];
  const pushInstrument = (record, exchange) => {
    const testIssue = normalizeText(record['Test Issue']).toUpperCase();
    const etfFlag = normalizeText(record.ETF).toUpperCase();
    const symbol = normalizeText(record.Symbol || record['ACT Symbol']).toUpperCase();
    const displayName = normalizeText(record['Security Name']);

    if (!symbol || !displayName || testIssue === 'Y' || etfFlag === 'Y') {
      return;
    }

    if (!looksLikeOperatingCompany(displayName)) {
      return;
    }

    const instrument = buildInstrument({
      symbol,
      normalized: symbol,
      displayName,
      market: 'US',
      exchange,
      instrumentType: 'Equity',
      country: 'US',
      source: exchange === 'NASDAQ' ? OFFICIAL_SOURCES.nasdaqListed : OFFICIAL_SOURCES.otherListed,
    });

    if (instrument) {
      instruments.push(instrument);
    }
  };

  for (const record of parseDelimitedText(nasdaqListedRaw, '|')) {
    if (normalizeText(record.Symbol).toUpperCase() === 'FILE CREATION TIME') {
      continue;
    }
    pushInstrument(record, 'NASDAQ');
  }

  for (const record of parseDelimitedText(otherListedRaw, '|')) {
    if (normalizeText(record['ACT Symbol']).toUpperCase() === 'FILE CREATION TIME') {
      continue;
    }

    const exchangeCode = normalizeText(record.Exchange).toUpperCase();
    const exchange =
      exchangeCode === 'N'
        ? 'NYSE'
        : exchangeCode === 'A'
          ? 'AMEX'
          : exchangeCode === 'P'
            ? 'NYSE ARCA'
            : exchangeCode === 'Z'
              ? 'BATS'
              : 'US';
    pushInstrument(record, exchange);
  }

  return instruments;
}

async function fetchNseInstruments() {
  const csvRaw = await fetchText(OFFICIAL_SOURCES.nseEquities);
  const lines = String(csvRaw)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length <= 1) {
    return [];
  }

  const headers = csvSplit(lines[0]);
  const headerIndex = new Map(headers.map((header, index) => [header.toUpperCase(), index]));
  const symbolIndex = headerIndex.get('SYMBOL');
  const companyIndex = headerIndex.get('NAME OF COMPANY');
  const seriesIndex = headerIndex.get('SERIES');

  const allowedSeries = new Set(['EQ', 'BE', 'BZ', 'SM', 'ST']);
  const instruments = [];

  for (const line of lines.slice(1)) {
    const cells = csvSplit(line);
    const symbol = normalizeText(cells[symbolIndex]).toUpperCase();
    const displayName = normalizeText(cells[companyIndex]);
    const series = normalizeText(cells[seriesIndex]).toUpperCase();

    if (!symbol || !displayName || (series && !allowedSeries.has(series))) {
      continue;
    }

    const instrument = buildInstrument({
      symbol,
      normalized: `${symbol}.NS`,
      displayName,
      market: 'IN',
      exchange: 'NSE',
      instrumentType: 'Equity',
      country: 'IN',
      source: OFFICIAL_SOURCES.nseEquities,
      extraSearchTerms: series ? [series] : [],
    });

    if (instrument) {
      instruments.push(instrument);
    }
  }

  return instruments;
}

function curatedIndices() {
  return CURATED_INDICES.map((entry) =>
    buildInstrument({
      symbol: entry.symbol,
      normalized: entry.normalized,
      displayName: entry.display_name,
      market: entry.market,
      exchange: entry.exchange,
      instrumentType: entry.instrument_type,
      country: entry.country,
      source: entry.source,
    })
  ).filter(Boolean);
}

function dedupeInstruments(instruments) {
  const byId = new Map();
  for (const instrument of instruments) {
    if (!instrument?.id) {
      continue;
    }
    byId.set(instrument.id, instrument);
  }

  return [...byId.values()].sort((left, right) => {
    if (left.market !== right.market) {
      return left.market.localeCompare(right.market);
    }
    if (left.instrument_type !== right.instrument_type) {
      return left.instrument_type.localeCompare(right.instrument_type);
    }
    return left.normalized.localeCompare(right.normalized);
  });
}

async function buildMarketMaster() {
  const [usEquities, indiaEquities] = await Promise.all([
    fetchNasdaqListedInstruments(),
    fetchNseInstruments(),
  ]);
  const instruments = dedupeInstruments([
    ...usEquities,
    ...indiaEquities,
    ...curatedIndices(),
  ]);

  const counts = instruments.reduce(
    (summary, instrument) => {
      summary.total += 1;
      const key = `${instrument.market}_${instrument.instrument_type}`.toLowerCase();
      summary[key] = (summary[key] || 0) + 1;
      return summary;
    },
    { total: 0 }
  );

  return {
    generated_at: new Date().toISOString(),
    sources: OFFICIAL_SOURCES,
    notes: [
      'US equities come from Nasdaq Trader symbol directories and exclude ETFs, test issues, and likely non-operating issues.',
      'India equities come from the official NSE equity directory.',
      'Benchmark indices are curated explicitly to keep the picker strict and honest while the market universe remains equities-first.',
    ],
    counts,
    instruments,
  };
}

async function main() {
  const marketMaster = await buildMarketMaster();
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.writeFile(OUTPUT_FILE, JSON.stringify(marketMaster, null, 2), 'utf8');
  console.log(`Wrote ${marketMaster.counts.total} instruments to ${OUTPUT_FILE}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
