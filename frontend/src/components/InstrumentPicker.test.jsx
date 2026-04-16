import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import InstrumentPicker from './InstrumentPicker'

const searchSymbols = vi.fn()

vi.mock('../api/symbols', () => ({
  searchSymbols: (...args) => searchSymbols(...args),
}))

describe('InstrumentPicker', () => {
  const onChange = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it('searches supported instruments and emits the selected result', async () => {
    searchSymbols.mockResolvedValue({
      results: [
        {
          id: 'IN:RELIANCE.NS',
          symbol: 'RELIANCE',
          normalized: 'RELIANCE.NS',
          display_name: 'Reliance Industries',
          market: 'IN',
          exchange: 'NSE',
          instrument_type: 'equity',
          country: 'IN',
        },
      ],
    })

    render(<InstrumentPicker value={null} onChange={onChange} recentSelections={[]} />)

    const input = screen.getByRole('textbox')
    fireEvent.focus(input)
    fireEvent.change(input, { target: { value: 'rel' } })

    expect(await screen.findByText('Search results')).toBeInTheDocument()
    expect(await screen.findByText('Reliance Industries')).toBeInTheDocument()

    fireEvent.click(
      screen.getByRole('option', {
        name: /Reliance Industries/i,
      }),
    )

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          normalized: 'RELIANCE.NS',
          display_name: 'Reliance Industries',
          market: 'IN',
        }),
      )
    })
  })

  it('supports keyboard-first selection from recent picks', async () => {
    render(
      <InstrumentPicker
        value={null}
        onChange={onChange}
        recentSelections={[
          {
            id: 'US:^GSPC',
            symbol: '^GSPC',
            normalized: '^GSPC',
            display_name: 'S&P 500',
            market: 'US',
            exchange: 'SP',
            instrument_type: 'index',
            country: 'US',
          },
        ]}
      />,
    )

    const input = screen.getByRole('textbox')
    fireEvent.focus(input)

    expect(await screen.findByText('Recent picks')).toBeInTheDocument()
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'Enter' })

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          normalized: '^GSPC',
          display_name: 'S&P 500',
        }),
      )
    })
  })
})
