import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { vi, describe, it, beforeEach, expect } from 'vitest'
import Models from '../../src/pages/Models'

vi.mock('../../src/services/api', () => ({ listModels: vi.fn() }))
import { listModels } from '../../src/services/api'

describe('Models page', () => {
  beforeEach(() => {
    listModels.mockResolvedValue([{ id:1, model_name:'m', version:'1' }])
  })

  it('shows model entries from API', async () => {
    render(<Models />)
    await waitFor(() => expect(listModels).toHaveBeenCalled())
    const matches = screen.getAllByText(/m/i)
    expect(matches.length).toBeGreaterThan(0)
  })
})
