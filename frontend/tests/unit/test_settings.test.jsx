/*
  Unit test for the `Settings` React page.
  Verifies that on mount the component calls `getModelConfig` and `listModels`
  and populates the form fields accordingly. The API module is mocked.
*/

import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { vi, describe, it, beforeEach, expect } from 'vitest'
import Settings from '../../src/pages/Settings'

// Mock API functions used by Settings
vi.mock('../../src/services/api', () => ({
  getModelConfig: vi.fn(),
  listModels: vi.fn(),
  setModelConfig: vi.fn(),
}))

import { getModelConfig, listModels } from '../../src/services/api'

describe('Settings page', () => {
  beforeEach(() => {
    getModelConfig.mockResolvedValue({ active_model_id: 2, alert_threshold: 0.6, updated_by: 'tester' })
    listModels.mockResolvedValue([{ id: 1, model_name: 'm1', version: '1' }, { id: 2, model_name: 'm2', version: '2' }])
  })

  it('renders and populates form from API', async () => {
    render(<Settings />)
    await waitFor(() => expect(getModelConfig).toHaveBeenCalled())
    // verifica que la opción m2 esté presente en el select de modelos
    expect(screen.getByText(/m2/i)).toBeTruthy()
    const inputThreshold = screen.getByDisplayValue('0.6')
    // el umbral mostrado debe corresponder al valor retornado por la API
    expect(Number(inputThreshold.value)).toBeCloseTo(0.6)
  })
})
