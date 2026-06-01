import React from 'react'
import { render, screen, waitFor, cleanup } from '@testing-library/react'
import { vi, describe, it, beforeEach, afterEach, expect } from 'vitest'

vi.mock('../../src/services/api')

import BatchScoring from '../../src/pages/BatchScoring'
import * as api from '../../src/services/api'

afterEach(() => { cleanup() })

beforeEach(() => {
  vi.clearAllMocks()
  api.getBatchScoringRuns.mockResolvedValue({ count: 0, items: [] })
  api.runBatchScoring.mockResolvedValue({ status: 'COMPLETED', batch_scoring_run_id: 1, total_scored: 0, high_count: 0, medium_count: 0, low_count: 0 })
  api.getBatchScoringResults.mockResolvedValue({ rows: [], total: 0, total_pages: 1, page: 1, page_size: 20 })
  api.getBatchScoringReport.mockResolvedValue(null)
  api.getBatchScoringMetadata.mockResolvedValue(null)
})

describe('BatchScoring page (unit)', () => {
  it('renders title and triggers getBatchScoringRuns on mount', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByText('Scoring por Lotes')).toBeTruthy()
  })

  it('muestra el aviso metodológico', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByText(/No confirma fraude automáticamente/i)).toBeTruthy()
  })

  it('muestra mensaje vacío cuando no hay ejecuciones', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByText(/No existen ejecuciones de scoring/i)).toBeTruthy()
  })
})
