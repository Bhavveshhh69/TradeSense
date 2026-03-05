const repository = require('./symbols.repository');

function createHttpError(status, message) {
  const error = new Error(message);
  error.status = status;
  return error;
}

function normalizeInputSymbol(inputSymbol) {
  if (typeof inputSymbol !== 'string') {
    throw createHttpError(400, 'symbol is required');
  }

  const normalized = inputSymbol.trim().toUpperCase();
  if (!normalized) {
    throw createHttpError(400, 'symbol is required');
  }

  return normalized;
}

function buildSearchCatalog(registries) {
  const catalog = new Set();

  for (const symbol of registries.nseSymbols) {
    catalog.add(symbol);
    catalog.add(`${symbol}.NS`);
  }

  for (const symbol of registries.bseSymbols) {
    catalog.add(symbol);
    catalog.add(`${symbol}.BO`);
  }

  for (const symbol of registries.usSymbols) {
    catalog.add(symbol);
  }

  for (const symbol of registries.indices) {
    catalog.add(symbol);
  }

  return [...catalog];
}

async function normalizeSymbol(inputSymbol) {
  const symbol = normalizeInputSymbol(inputSymbol);

  if (symbol.endsWith('.NS') || symbol.endsWith('.BO')) {
    return symbol;
  }

  const registries = await repository.getSymbolRegistries();

  if (registries.indices.includes(symbol)) {
    return symbol;
  }

  if (registries.nseSymbols.includes(symbol)) {
    return `${symbol}.NS`;
  }

  if (registries.bseSymbols.includes(symbol)) {
    return `${symbol}.BO`;
  }

  if (registries.usSymbols.includes(symbol)) {
    return symbol;
  }

  return symbol;
}

async function validateSymbol(inputSymbol) {
  if (typeof inputSymbol !== 'string') {
    return false;
  }

  const symbol = inputSymbol.trim().toUpperCase();
  if (!symbol) {
    return false;
  }

  const registries = await repository.getSymbolRegistries();
  const catalog = buildSearchCatalog(registries);
  return catalog.includes(symbol);
}

async function searchSymbols(query) {
  if (typeof query !== 'string') {
    return [];
  }

  const normalizedQuery = query.trim().toUpperCase();
  if (!normalizedQuery) {
    return [];
  }

  const registries = await repository.getSymbolRegistries();
  const catalog = buildSearchCatalog(registries);

  return catalog
    .filter((symbol) => symbol.includes(normalizedQuery))
    .sort((a, b) => a.localeCompare(b));
}

module.exports = {
  normalizeSymbol,
  searchSymbols,
  validateSymbol,
};
