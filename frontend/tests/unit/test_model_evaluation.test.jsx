import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { vi, describe, it, beforeEach, expect } from 'vitest'
import ModelEvaluation from '../../src/pages/ModelEvaluation'

vi.mock('../../src/services/api', () => ({ getModelResults: vi.fn(), activateModel: vi.fn(), exportModelResults: vi.fn() }))
import { getModelResults, activateModel } from '../../src/services/api'

describe('ModelEvaluation page', () => {
  beforeEach(()=>{
    getModelResults.mockResolvedValue([{ id:1, model_name:'rf-v1', roc_auc:0.92, recall:0.8, precision:0.75, f1_score:0.77, is_active:false }])
  })

  it('renders model table and actions', async ()=>{
    render(<ModelEvaluation />)
    await waitFor(()=> expect(getModelResults).toHaveBeenCalled())
    expect(screen.getByText(/rf-v1/i)).toBeTruthy()
    // buttons for activate/export should be present
    expect(screen.getAllByText(/Activar|Exportar resultados/i).length).toBeGreaterThan(0)
  })
})
