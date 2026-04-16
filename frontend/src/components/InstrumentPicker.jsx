import {
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'

import { searchSymbols } from '../api/symbols'

const MARKET_TABS = [
  { value: 'ALL', label: 'All markets' },
  { value: 'US', label: 'US' },
  { value: 'IN', label: 'India' },
]

const KIND_TABS = [
  { value: 'ALL', label: 'All' },
  { value: 'equity', label: 'Equities' },
  { value: 'index', label: 'Indices' },
]

function normalizeInstrument(instrument) {
  if (!instrument || typeof instrument !== 'object') {
    return null
  }

  const normalized =
    typeof instrument.normalized === 'string' && instrument.normalized.trim()
      ? instrument.normalized.trim().toUpperCase()
      : typeof instrument.symbol === 'string' && instrument.symbol.trim()
        ? instrument.symbol.trim().toUpperCase()
        : ''

  if (!normalized) {
    return null
  }

  return {
    id: instrument.id || normalized,
    symbol:
      typeof instrument.symbol === 'string' && instrument.symbol.trim()
        ? instrument.symbol.trim().toUpperCase()
        : normalized,
    normalized,
    display_name:
      typeof instrument.display_name === 'string' && instrument.display_name.trim()
        ? instrument.display_name.trim()
        : normalized,
    market:
      typeof instrument.market === 'string' && instrument.market.trim()
        ? instrument.market.trim().toUpperCase()
        : '',
    exchange:
      typeof instrument.exchange === 'string' && instrument.exchange.trim()
        ? instrument.exchange.trim().toUpperCase()
        : '',
    instrument_type:
      typeof instrument.instrument_type === 'string' && instrument.instrument_type.trim()
        ? instrument.instrument_type.trim()
        : '',
    country:
      typeof instrument.country === 'string' && instrument.country.trim()
        ? instrument.country.trim().toUpperCase()
        : '',
  }
}

function filterByTabs(rows, market, kind) {
  return (Array.isArray(rows) ? rows : [])
    .map(normalizeInstrument)
    .filter(Boolean)
    .filter((instrument) => (market === 'ALL' ? true : instrument.market === market))
    .filter((instrument) =>
      kind === 'ALL'
        ? true
        : instrument.instrument_type.toLowerCase() === kind.toLowerCase()
    )
}

function pickerOptionId(id) {
  return `instrument-picker-option-${String(id).replace(/[^a-zA-Z0-9_-]/g, '-')}`
}

export default function InstrumentPicker({
  value,
  onChange,
  recentSelections = [],
  placeholder = 'Search US or India stocks and indices',
}) {
  const rootRef = useRef(null)
  const [query, setQuery] = useState(value?.display_name || value?.normalized || '')
  const [open, setOpen] = useState(false)
  const [market, setMarket] = useState('ALL')
  const [kind, setKind] = useState('ALL')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [highlightIndex, setHighlightIndex] = useState(0)
  const deferredQuery = useDeferredValue(query)

  useEffect(() => {
    if (!open) {
      setQuery(value?.display_name || value?.normalized || '')
    }
  }, [open, value])

  useEffect(() => {
    function handlePointerDown(event) {
      if (rootRef.current && !rootRef.current.contains(event.target)) {
        setOpen(false)
      }
    }

    document.addEventListener('mousedown', handlePointerDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function runSearch() {
      const normalizedQuery = deferredQuery.trim()
      if (!open || !normalizedQuery) {
        startTransition(() => {
          setResults([])
          setLoading(false)
          setError(null)
          setHighlightIndex(0)
        })
        return
      }

      setLoading(true)
      setError(null)
      try {
        const payload = await searchSymbols(normalizedQuery, {
          market: market === 'ALL' ? undefined : market,
          kind: kind === 'ALL' ? undefined : kind,
          limit: 40,
        })

        if (cancelled) {
          return
        }

        startTransition(() => {
          setResults(Array.isArray(payload?.results) ? payload.results : [])
          setHighlightIndex(0)
        })
      } catch (requestError) {
        if (cancelled) {
          return
        }

        startTransition(() => {
          setResults([])
          setError(requestError?.response?.data?.error || requestError?.message || 'Search failed')
          setHighlightIndex(0)
        })
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    runSearch()

    return () => {
      cancelled = true
    }
  }, [deferredQuery, kind, market, open])

  const visibleResults = useMemo(() => {
    if (deferredQuery.trim()) {
      return filterByTabs(results, market, kind)
    }

    return filterByTabs(recentSelections, market, kind)
  }, [deferredQuery, kind, market, recentSelections, results])

  useEffect(() => {
    if (highlightIndex >= visibleResults.length) {
      setHighlightIndex(visibleResults.length > 0 ? 0 : -1)
    } else if (visibleResults.length > 0 && highlightIndex < 0) {
      setHighlightIndex(0)
    }
  }, [highlightIndex, visibleResults])

  function commitSelection(instrument) {
    const normalizedInstrument = normalizeInstrument(instrument)
    if (!normalizedInstrument) {
      return
    }

    setOpen(false)
    setQuery(normalizedInstrument.display_name || normalizedInstrument.normalized)
    onChange(normalizedInstrument)
  }

  function handleKeyDown(event) {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      if (!open) {
        setOpen(true)
        return
      }
      if (visibleResults.length > 0) {
        setHighlightIndex((current) => (current + 1) % visibleResults.length)
      }
      return
    }

    if (event.key === 'ArrowUp') {
      event.preventDefault()
      if (!open) {
        setOpen(true)
        return
      }
      if (visibleResults.length > 0) {
        setHighlightIndex((current) =>
          current <= 0 ? visibleResults.length - 1 : current - 1
        )
      }
      return
    }

    if (event.key === 'Enter' && open) {
      event.preventDefault()
      const instrument = visibleResults[highlightIndex]
      if (instrument) {
        commitSelection(instrument)
      }
      return
    }

    if (event.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div className="instrument-picker-shell" ref={rootRef}>
      <div className={`instrument-picker-input-shell${open ? ' is-open' : ''}`}>
        <input
          type="text"
          className="instrument-picker-input"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value)
            setOpen(true)
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          autoComplete="off"
          aria-expanded={open ? 'true' : 'false'}
          aria-controls="instrument-picker-panel"
          aria-activedescendant={
            open && visibleResults[highlightIndex]
              ? pickerOptionId(visibleResults[highlightIndex].id || visibleResults[highlightIndex].normalized)
              : undefined
          }
        />
        {value ? (
          <button
            type="button"
            className="instrument-picker-clear"
            onClick={() => {
              setQuery('')
              setOpen(false)
              onChange(null)
            }}
            aria-label="Clear active instrument"
          >
            Clear
          </button>
        ) : null}
      </div>

      {open ? (
        <div className="instrument-picker-panel" id="instrument-picker-panel">
          <div className="instrument-picker-filters">
            <div className="instrument-picker-tab-row">
              {MARKET_TABS.map((tab) => (
                <button
                  key={tab.value}
                  type="button"
                  className={`instrument-picker-tab${market === tab.value ? ' is-active' : ''}`}
                  onClick={() => setMarket(tab.value)}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <div className="instrument-picker-tab-row">
              {KIND_TABS.map((tab) => (
                <button
                  key={tab.value}
                  type="button"
                  className={`instrument-picker-tab${kind === tab.value ? ' is-active' : ''}`}
                  onClick={() => setKind(tab.value)}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          <div className="instrument-picker-results">
            <div className="instrument-picker-status">
              <strong>{deferredQuery.trim() ? 'Search results' : 'Recent picks'}</strong>
              <span>
                {loading
                  ? 'Searching...'
                  : `${visibleResults.length} ${visibleResults.length === 1 ? 'instrument' : 'instruments'}`}
              </span>
            </div>

            {error ? <p className="instrument-picker-error">{error}</p> : null}

            {!error && visibleResults.length === 0 && !loading ? (
              <div className="instrument-picker-empty">
                <strong>{deferredQuery.trim() ? 'No supported instrument found' : 'No recent picks yet'}</strong>
                <p>
                  {deferredQuery.trim()
                    ? 'TradeSense only accepts symbols present in the US and India market master.'
                    : 'Search for a stock or index to lock your first active instrument.'}
                </p>
              </div>
            ) : null}

            {visibleResults.length > 0 ? (
              <div className="instrument-picker-list" role="listbox">
                {visibleResults.map((instrument, index) => (
                  <button
                    key={instrument.id || instrument.normalized}
                    id={pickerOptionId(instrument.id || instrument.normalized)}
                    type="button"
                    className={`instrument-picker-option${highlightIndex === index ? ' is-highlighted' : ''}`}
                    onMouseEnter={() => setHighlightIndex(index)}
                    onClick={() => commitSelection(instrument)}
                    role="option"
                    aria-selected={highlightIndex === index ? 'true' : 'false'}
                  >
                    <div className="instrument-picker-option-main">
                      <strong>{instrument.display_name}</strong>
                      <span>{instrument.normalized}</span>
                    </div>
                    <div className="instrument-picker-option-meta">
                      <span>{instrument.market === 'IN' ? 'India' : 'US'}</span>
                      <span>{instrument.exchange}</span>
                      <span>{instrument.instrument_type}</span>
                    </div>
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}
