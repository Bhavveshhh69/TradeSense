const crypto = require('crypto');

const store = new Map();
const DEFAULT_TTL_SECONDS = Number(process.env.CACHE_TTL_SECONDS);
const DEFAULT_TTL_MS =
  Number.isFinite(DEFAULT_TTL_SECONDS) && DEFAULT_TTL_SECONDS > 0
    ? DEFAULT_TTL_SECONDS * 1000
    : 300 * 1000;

function stableStringify(value) {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(',')}]`;
  }
  const keys = Object.keys(value).sort();
  const entries = keys.map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`);
  return `{${entries.join(',')}}`;
}

function hashBody(body) {
  return crypto.createHash('sha256').update(stableStringify(body)).digest('hex');
}

function get(key) {
  const entry = store.get(key);
  if (!entry) {
    return null;
  }
  if (Date.now() >= entry.expiresAt) {
    store.delete(key);
    return null;
  }
  return entry.value;
}

function set(key, value, ttlMs = DEFAULT_TTL_MS) {
  store.set(key, { value, expiresAt: Date.now() + ttlMs });
}

function clear() {
  store.clear();
}

module.exports = {
  DEFAULT_TTL_MS,
  hashBody,
  get,
  set,
  clear,
};
