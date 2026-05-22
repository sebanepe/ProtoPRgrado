import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
vi.mock('../services/api')
import Dashboard from '../pages/Dashboard'
import * as api from '../services/api'

describe('Dashboard page', ()=>{
  it('renders KPI cards from dashboard summary', async ()=>{
    const mock = {
      transactions: 999,
      alerts: 7,
      risk: 0.12,
      model: 'rf-test',
      alertTrend: [],
      fraudRatio: {fraud:10, normal:90},
      recentAlerts: []
    }
    api.getDashboardSummary.mockResolvedValue(mock)
    render(<Dashboard />)
    // wait for async summary to be rendered
    await waitFor(()=>{
      expect(screen.getByText('Transacciones analizadas')).toBeTruthy()
      expect(screen.getByText('999')).toBeTruthy()
      expect(screen.getByText('Alertas activas')).toBeTruthy()
      expect(screen.getByText('7')).toBeTruthy()
      expect(screen.getByText('Modelo activo')).toBeTruthy()
      expect(screen.getByText('rf-test')).toBeTruthy()
    })
  })
})
