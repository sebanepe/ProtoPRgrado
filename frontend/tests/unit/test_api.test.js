/*
  Unit tests for frontend API helpers.
  These tests mock the axios instance used in `src/services/api.js` and verify
  that helpers call the correct endpoints and return expected data shapes.
*/

import { vi, describe, it, expect, beforeEach } from 'vitest'

// Mock axios instance used inside api.js
vi.mock('axios', () => {
  const get = vi.fn()
  const post = vi.fn()
  const instance = { get, post }
  const create = () => instance
  return { default: { create }, __esModule: true }
})

import * as api from '../../src/services/api'

describe('API service functions', () => {
  beforeEach(() => {
    // ensure the module's exported axios instance has mockable methods
    api.default.get = vi.fn()
    api.default.post = vi.fn()
  })

  it('health should call /health', async () => {
    api.default.get.mockResolvedValueOnce({ data: { status: 'ok' } })
    const res = await api.health()
    expect(api.default.get).toHaveBeenCalledWith('/health')
    // verifica que la función devuelve el objeto de estado esperado
    expect(res).toEqual({ status: 'ok' })
  })

  it('listModels should call /models/results and return array', async () => {
    api.default.get.mockResolvedValueOnce({ data: { results: [{ id: 1, model_name: 'm' }] } })
    const res = await api.listModels()
    expect(api.default.get).toHaveBeenCalledWith('/models/results')
    // la respuesta debe ser un array de modelos
    expect(Array.isArray(res)).toBe(true)
    // el primer elemento debe tener el nombre de modelo esperado
    expect(res[0].model_name).toBe('m')
  })

  it('setModelConfig should post to /settings/model-config', async () => {
    api.default.post.mockResolvedValueOnce({ data: { model_config: { id: 1 } } })
    const payload = { active_model_id: 1, alert_threshold: 0.5 }
    const res = await api.setModelConfig(payload)
    expect(api.default.post).toHaveBeenCalledWith('/settings/model-config', payload)
    // la respuesta debe contener el id del model_config devuelto por la API
    expect(res.id).toBe(1)
  })

  it('getPreprocessingRunStages should call /preprocessing/runs/:id/stages and return stages', async () => {
    const mockStages = { run_id: 2, stages: { limpieza: 'COMPLETED', smote: 'NOT_APPLIED' }, status: 'COMPLETED' }
    api.default.get.mockResolvedValueOnce({ data: mockStages })
    const res = await api.getPreprocessingRunStages(2)
    expect(api.default.get).toHaveBeenCalledWith('/preprocessing/runs/2/stages')
    expect(res.stages.limpieza).toBe('COMPLETED')
    expect(res.status).toBe('COMPLETED')
  })

  it('downloadPreprocessingRun should call /preprocessing/runs/:id/download with blob', async () => {
    const fakeBlob = new Blob(['a,b\n1,2'], { type: 'text/csv' })
    api.default.get.mockResolvedValueOnce({ data: fakeBlob })
    const res = await api.downloadPreprocessingRun(5)
    expect(api.default.get).toHaveBeenCalledWith('/preprocessing/runs/5/download', { responseType: 'blob' })
    expect(res).toBeInstanceOf(Blob)
  })
})
