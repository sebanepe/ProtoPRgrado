import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { vi, describe, it, beforeEach, expect } from 'vitest'
import BatchScoring from '../../src/pages/BatchScoring'

vi.mock('../../src/services/api', () => ({ runBatchScoring: vi.fn(), getModelConfig: vi.fn(), listModels: vi.fn() }))
import { runBatchScoring, getModelConfig } from '../../src/services/api'

describe('BatchScoring page', () => {
  beforeEach(()=>{
    runBatchScoring.mockResolvedValue({ created: 0 })
    getModelConfig.mockResolvedValue({ active_model_id: null, alert_threshold: 0.7 })
  })

  it('renders and can trigger scoring', async ()=>{
    render(<BatchScoring />)
    await waitFor(()=> expect(getModelConfig).toHaveBeenCalled())
    expect(screen.getByText(/Scoring por Lotes/i)).toBeTruthy()
  })
})
