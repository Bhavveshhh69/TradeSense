const fs = require('fs/promises');
const path = require('path');
const { randomUUID } = require('crypto');

const DATA_FILE_PATH = path.resolve(__dirname, '../../data/analysis_recent.json');
const MAX_RECENT_ANALYSES = 50;

async function ensureDataFile() {
  try {
    await fs.access(DATA_FILE_PATH);
  } catch (error) {
    await fs.mkdir(path.dirname(DATA_FILE_PATH), { recursive: true });
    await fs.writeFile(DATA_FILE_PATH, '[]\n', 'utf8');
  }
}

async function readRawEntries() {
  await ensureDataFile();
  const content = await fs.readFile(DATA_FILE_PATH, 'utf8');

  try {
    return JSON.parse(content);
  } catch (error) {
    return [];
  }
}

async function writeEntries(entries) {
  await ensureDataFile();
  await fs.writeFile(DATA_FILE_PATH, `${JSON.stringify(entries, null, 2)}\n`, 'utf8');
}

function normalizeSignal(value) {
  if (typeof value !== 'string' || !value.trim()) {
    return 'NO_TRADE';
  }

  return value.trim().toUpperCase();
}

function normalizeRecentEntry(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return null;
  }

  const id = typeof raw.id === 'string' && raw.id.trim() ? raw.id.trim() : randomUUID();
  const normalized =
    typeof raw.normalized === 'string' && raw.normalized.trim()
      ? raw.normalized.trim().toUpperCase()
      : typeof raw.symbol === 'string' && raw.symbol.trim()
        ? raw.symbol.trim().toUpperCase()
        : null;

  if (!normalized) {
    return null;
  }

  const recordedAt =
    typeof raw.recorded_at === 'string' && raw.recorded_at.trim()
      ? raw.recorded_at.trim()
      : new Date().toISOString();

  return {
    id,
    symbol:
      typeof raw.symbol === 'string' && raw.symbol.trim()
        ? raw.symbol.trim().toUpperCase()
        : normalized.replace(/\.(NS|BO)$/i, ''),
    normalized,
    display_name:
      typeof raw.display_name === 'string' && raw.display_name.trim()
        ? raw.display_name.trim()
        : normalized,
    market:
      typeof raw.market === 'string' && raw.market.trim() ? raw.market.trim().toUpperCase() : null,
    exchange:
      typeof raw.exchange === 'string' && raw.exchange.trim()
        ? raw.exchange.trim().toUpperCase()
        : null,
    instrument_type:
      typeof raw.instrument_type === 'string' && raw.instrument_type.trim()
        ? raw.instrument_type.trim()
        : null,
    signal: normalizeSignal(raw.signal),
    decision_label:
      typeof raw.decision_label === 'string' && raw.decision_label.trim()
        ? raw.decision_label.trim()
        : null,
    confidence_level:
      typeof raw.confidence_level === 'string' && raw.confidence_level.trim()
        ? raw.confidence_level.trim()
        : null,
    current_price:
      Number.isFinite(Number(raw.current_price)) && Number(raw.current_price) > 0
        ? Number(raw.current_price)
        : null,
    price_error: raw.price_error === true,
    price_error_message:
      typeof raw.price_error_message === 'string' && raw.price_error_message.trim()
        ? raw.price_error_message.trim()
        : null,
    trend_summary:
      typeof raw.trend_summary === 'string' && raw.trend_summary.trim()
        ? raw.trend_summary.trim()
        : null,
    risk_summary:
      typeof raw.risk_summary === 'string' && raw.risk_summary.trim()
        ? raw.risk_summary.trim()
        : null,
    signal_explanation:
      typeof raw.signal_explanation === 'string' && raw.signal_explanation.trim()
        ? raw.signal_explanation.trim()
        : null,
    trade_actionable: raw.trade_actionable === true,
    actionability_state:
      typeof raw.actionability_state === 'string' && raw.actionability_state.trim()
        ? raw.actionability_state.trim()
        : raw.trade_actionable === true
          ? 'actionable'
          : null,
    decision_reason_type:
      typeof raw.decision_reason_type === 'string' && raw.decision_reason_type.trim()
        ? raw.decision_reason_type.trim()
        : null,
    no_trade_reason:
      typeof raw.no_trade_reason === 'string' && raw.no_trade_reason.trim()
        ? raw.no_trade_reason.trim()
        : null,
    recorded_at: recordedAt,
  };
}

function sortEntriesDescending(entries) {
  return [...entries].sort((left, right) =>
    String(right.recorded_at || '').localeCompare(String(left.recorded_at || ''))
  );
}

async function listRecentAnalyses(limitInput = 10) {
  const limit = Math.max(1, Math.min(Number(limitInput) || 10, MAX_RECENT_ANALYSES));
  const entries = await readRawEntries();

  return sortEntriesDescending(entries.map(normalizeRecentEntry).filter(Boolean)).slice(0, limit);
}

async function recordAnalysis(entry) {
  const normalizedEntry = normalizeRecentEntry(entry);
  if (!normalizedEntry) {
    return null;
  }

  const entries = await readRawEntries();
  const dedupedEntries = entries
    .map(normalizeRecentEntry)
    .filter(Boolean)
    .filter((existing) => existing.id !== normalizedEntry.id);

  const updatedEntries = sortEntriesDescending([normalizedEntry, ...dedupedEntries]).slice(
    0,
    MAX_RECENT_ANALYSES
  );

  await writeEntries(updatedEntries);
  return normalizedEntry;
}

async function __reset() {
  await writeEntries([]);
}

module.exports = {
  __reset,
  listRecentAnalyses,
  recordAnalysis,
};
