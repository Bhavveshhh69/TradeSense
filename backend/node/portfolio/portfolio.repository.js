const fs = require('fs/promises');
const path = require('path');

const { createHttpError, normalizeStoredHolding } = require('./portfolio.model');

const DATA_FILE_PATH = process.env.PORTFOLIO_DATA_FILE
  ? path.resolve(process.env.PORTFOLIO_DATA_FILE)
  : path.resolve(__dirname, '..', 'data', 'portfolio.json');

let writeChain = Promise.resolve();

async function ensureDataFile() {
  const dataDir = path.dirname(DATA_FILE_PATH);
  await fs.mkdir(dataDir, { recursive: true });

  try {
    await fs.access(DATA_FILE_PATH);
  } catch (error) {
    await fs.writeFile(DATA_FILE_PATH, '[]\n', 'utf8');
  }
}

async function readRawHoldings() {
  await ensureDataFile();
  const raw = await fs.readFile(DATA_FILE_PATH, 'utf8');

  let parsed;
  try {
    parsed = JSON.parse(raw || '[]');
  } catch (error) {
    throw createHttpError(500, 'Portfolio storage is corrupted');
  }

  if (!Array.isArray(parsed)) {
    throw createHttpError(500, 'Portfolio storage is corrupted');
  }

  return parsed;
}

function sanitizeHoldings(records) {
  return records.map((record) => normalizeStoredHolding(record)).filter(Boolean);
}

async function writeRawHoldings(records) {
  const serialized = `${JSON.stringify(records, null, 2)}\n`;
  const tempFilePath = `${DATA_FILE_PATH}.tmp`;
  await fs.writeFile(tempFilePath, serialized, 'utf8');
  await fs.rename(tempFilePath, DATA_FILE_PATH);
}

function runSerialized(operation) {
  const run = writeChain.then(operation, operation);
  writeChain = run.catch(() => {});
  return run;
}

async function getAllHoldings() {
  const records = await readRawHoldings();
  return sanitizeHoldings(records);
}

async function addHolding(item) {
  return runSerialized(async () => {
    const records = await readRawHoldings();
    const holdings = sanitizeHoldings(records);
    holdings.push(item);
    await writeRawHoldings(holdings);
    return item;
  });
}

async function deleteHoldingById(id) {
  return runSerialized(async () => {
    const records = await readRawHoldings();
    const holdings = sanitizeHoldings(records);
    const nextHoldings = holdings.filter((holding) => holding.id !== id);
    const deleted = nextHoldings.length !== holdings.length;

    if (deleted) {
      await writeRawHoldings(nextHoldings);
    }

    return deleted;
  });
}

module.exports = {
  addHolding,
  deleteHoldingById,
  getAllHoldings,
};
