const fs = require('fs/promises');
const path = require('path');

const DATA_DIR = path.resolve(__dirname, '..', 'data', 'symbols');
const NSE_FILE = path.join(DATA_DIR, 'nse_symbols.json');
const US_FILE = path.join(DATA_DIR, 'us_symbols.json');
const INDICES_FILE = path.join(DATA_DIR, 'indices.json');
const BSE_FILE = path.join(DATA_DIR, 'bse_symbols.json');

function sanitizeSymbolList(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  return [...new Set(
    value
      .filter((item) => typeof item === 'string')
      .map((item) => item.trim().toUpperCase())
      .filter(Boolean)
  )];
}

async function readSymbolFile(filePath, { optional = false } = {}) {
  try {
    const raw = await fs.readFile(filePath, 'utf8');
    const parsed = JSON.parse(raw || '[]');
    return sanitizeSymbolList(parsed);
  } catch (error) {
    if (optional && error && error.code === 'ENOENT') {
      return [];
    }

    if (error instanceof SyntaxError) {
      const dataError = new Error(`Invalid symbol registry file: ${path.basename(filePath)}`);
      dataError.status = 500;
      throw dataError;
    }

    throw error;
  }
}

async function getSymbolRegistries() {
  const [nseSymbols, bseSymbols, usSymbols, indices] = await Promise.all([
    readSymbolFile(NSE_FILE),
    readSymbolFile(BSE_FILE, { optional: true }),
    readSymbolFile(US_FILE),
    readSymbolFile(INDICES_FILE),
  ]);

  return {
    nseSymbols,
    bseSymbols,
    usSymbols,
    indices,
  };
}

module.exports = {
  getSymbolRegistries,
};
