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
    expect(res).toEqual({ status: 'ok' })
  })

  it('listModels should call /models/results and return array', async () => {
    api.default.get.mockResolvedValueOnce({ data: { results: [{ id: 1, model_name: 'm' }] } })
    const res = await api.listModels()
    expect(api.default.get).toHaveBeenCalledWith('/models/results')
    expect(Array.isArray(res)).toBe(true)
    expect(res[0].model_name).toBe('m')
  })

  it('setModelConfig should post to /settings/model-config', async () => {
    api.default.post.mockResolvedValueOnce({ data: { model_config: { id: 1 } } })
    const payload = { active_model_id: 1, alert_threshold: 0.5 }
    const res = await api.setModelConfig(payload)
    expect(api.default.post).toHaveBeenCalledWith('/settings/model-config', payload)
    expect(res.id).toBe(1)
  })
})
