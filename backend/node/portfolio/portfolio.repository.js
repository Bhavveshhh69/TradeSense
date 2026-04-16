const fs = require('fs/promises');
const path = require('path');

const {
  createHttpError,
  normalizeStoredHolding,
  normalizeStoredTrade,
} = require('./portfolio.model');

const LEGACY_DATA_FILE_PATH = process.env.PORTFOLIO_DATA_FILE
  ? path.resolve(process.env.PORTFOLIO_DATA_FILE)
  : path.resolve(__dirname, '..', 'data', 'portfolio.json');
const TRADES_FILE_PATH = process.env.PORTFOLIO_TRADES_FILE
  ? path.resolve(process.env.PORTFOLIO_TRADES_FILE)
  : path.resolve(__dirname, '..', 'data', 'portfolio_trades.json');

let writeChain = Promise.resolve();

async function ensureDataFile(filePath, defaultContent = '[]\n') {
  const dataDir = path.dirname(filePath);
  await fs.mkdir(dataDir, { recursive: true });

  try {
    await fs.access(filePath);
  } catch (error) {
    await fs.writeFile(filePath, defaultContent, 'utf8');
  }
}

async function readRawJsonFile(filePath, { optional = false } = {}) {
  try {
    await ensureDataFile(filePath);
  } catch (error) {
    if (optional && error?.code === 'ENOENT') {
      return [];
    }
    throw error;
  }

  const raw = await fs.readFile(filePath, 'utf8');

  let parsed;
  try {
    parsed = JSON.parse(raw || '[]');
  } catch (error) {
    throw createHttpError(500, `Portfolio storage is corrupted (${path.basename(filePath)})`);
  }

  if (!Array.isArray(parsed)) {
    throw createHttpError(500, `Portfolio storage is corrupted (${path.basename(filePath)})`);
  }

  return parsed;
}

function sanitizeHoldings(records) {
  return records.map((record) => normalizeStoredHolding(record)).filter(Boolean);
}

function sanitizeTrades(records) {
  return records.map((record) => normalizeStoredTrade(record)).filter(Boolean);
}

async function writeRawJsonFile(filePath, records) {
  const serialized = `${JSON.stringify(records, null, 2)}\n`;
  const tempFilePath = `${filePath}.tmp`;
  await fs.writeFile(tempFilePath, serialized, 'utf8');
  await fs.rename(tempFilePath, filePath);
}

function runSerialized(operation) {
  const run = writeChain.then(operation, operation);
  writeChain = run.catch(() => {});
  return run;
}

async function getAllHoldings() {
  const records = await readRawJsonFile(LEGACY_DATA_FILE_PATH);
  return sanitizeHoldings(records);
}

async function addHolding(item) {
  return runSerialized(async () => {
    const records = await readRawJsonFile(LEGACY_DATA_FILE_PATH);
    const holdings = sanitizeHoldings(records);
    holdings.push(item);
    await writeRawJsonFile(LEGACY_DATA_FILE_PATH, holdings);
    return item;
  });
}

async function deleteHoldingById(id) {
  return runSerialized(async () => {
    const records = await readRawJsonFile(LEGACY_DATA_FILE_PATH);
    const holdings = sanitizeHoldings(records);
    const nextHoldings = holdings.filter((holding) => holding.id !== id);
    const deleted = nextHoldings.length !== holdings.length;

    if (deleted) {
      await writeRawJsonFile(LEGACY_DATA_FILE_PATH, nextHoldings);
    }

    return deleted;
  });
}

async function getAllTrades() {
  const records = await readRawJsonFile(TRADES_FILE_PATH);
  return sanitizeTrades(records);
}

async function replaceAllTrades(trades) {
  return runSerialized(async () => {
    const normalizedTrades = sanitizeTrades(trades);
    await writeRawJsonFile(TRADES_FILE_PATH, normalizedTrades);
    return normalizedTrades;
  });
}

async function appendTrades(items) {
  return runSerialized(async () => {
    const records = await readRawJsonFile(TRADES_FILE_PATH);
    const trades = sanitizeTrades(records);
    const normalizedItems = sanitizeTrades(items);
    const nextTrades = trades.concat(normalizedItems);
    await writeRawJsonFile(TRADES_FILE_PATH, nextTrades);
    return normalizedItems;
  });
}

module.exports = {
  addHolding,
  appendTrades,
  deleteHoldingById,
  getAllHoldings,
  getAllTrades,
  replaceAllTrades,
};
