import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { vi, describe, it, beforeEach, expect } from 'vitest'
import Settings from '../../src/pages/Settings'

vi.mock('../../src/services/api', () => ({ getModelConfig: vi.fn(), listModels: vi.fn(), setModelConfig: vi.fn() }))
import { getModelConfig, setModelConfig, listModels } from '../../src/services/api'

describe('Detection config validation', () => {
  beforeEach(()=>{
    getModelConfig.mockResolvedValue({ active_model_id: 2, alert_threshold: 0.6, updated_by: 'tester' })
    setModelConfig.mockResolvedValue({})
    // ensure listModels mock resolves for Settings component
    if (listModels && typeof listModels.mockResolvedValue === 'function') listModels.mockResolvedValue([])
  })

  it('prevents saving invalid threshold', async ()=>{
    render(<Settings />)
    await waitFor(()=> expect(getModelConfig).toHaveBeenCalled())
    const input = screen.getByDisplayValue('0.6')
    fireEvent.change(input, { target: { value: '-0.5' } })
    const save = screen.getByText(/Guardar/i)
    fireEvent.click(save)
    // setModelConfig should NOT be called due to validation
    expect(setModelConfig).not.toHaveBeenCalled()
  })
})
