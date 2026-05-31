import React from 'react'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
vi.mock('../services/api')
import Dashboard from '../pages/Dashboard'
import * as api from '../services/api'

const overview = {
  source_run: 'preprocessed_run_26',
  anomaly_run: 'run_26',
  total_transactions: 548108,
  active_alerts: 12,
  average_risk_score: 84.5,
  active_model: {
    model_name: 'Isolation Forest',
    run_id: 'run_26',
    anomaly_count: 5481,
    anomaly_rate: 0.009999854
  },
  review_distribution: {
    confirmed_fraud: 2,
    dismissed: 3,
    in_review: 0,
    new: 0,
    total_reviews: 5,
    usable_total_labels: 5
  },
  alerts_evolution: [{date: '2026-04-14', count: 2, high: 1, medium: 1, low: 0}],
  recent_alerts: [
    {
      alert_id: '26-S-001',
      rule_code: 'RULE_A',
      customer_hash: 'cust_a',
      risk_score: 90,
      risk_level: 'HIGH',
      status: 'NEW',
      created_at: '2026-05-30T20:19:34+00:00'
    }
  ],
  warnings: []
}

function renderDashboard(){
  return render(
    <MemoryRouter>
      <Dashboard />
    </MemoryRouter>
  )
}

describe('Dashboard page', ()=>{
  beforeEach(()=>{
    vi.clearAllMocks()
  })

  afterEach(()=>{
    cleanup()
  })

  it('renders real dashboard overview metrics', async ()=>{
    api.getDashboardOverview.mockResolvedValue(overview)
    renderDashboard()

    await waitFor(()=> expect(api.getDashboardOverview).toHaveBeenCalledWith({ source_run: 'preprocessed_run_26', anomaly_run: 'run_26' }))
    expect(screen.getByText('Transacciones analizadas')).toBeTruthy()
    expect(screen.getByText('548.108')).toBeTruthy()
    expect(screen.getByText('Alertas activas')).toBeTruthy()
    expect(screen.getByText('12')).toBeTruthy()
    expect(screen.getByText('Modelo activo')).toBeTruthy()
    expect(screen.getByText('Isolation Forest / run_26')).toBeTruthy()
    expect(screen.getByText('Proporcion de revision')).toBeTruthy()
    expect(screen.getByText('Alertas recientes')).toBeTruthy()
    expect(screen.getByText('26-S-001')).toBeTruthy()
  })

  it('does not render the old fraud ratio wording or target label fields', async ()=>{
    api.getDashboardOverview.mockResolvedValue({...overview, review_distribution: { total_reviews: 0 }})
    const { container } = renderDashboard()

    await waitFor(()=> expect(screen.getByText('Sin revisiones humanas suficientes.')).toBeTruthy())
    expect(screen.queryByText('Proporcion fraude')).toBeNull()
    expect(container.textContent).not.toContain('is_fraud')
    expect(container.textContent).not.toContain('confirmed_fraud')
  })

  it('shows controlled empty states', async ()=>{
    api.getDashboardOverview.mockResolvedValue({
      ...overview,
      alerts_evolution: [],
      recent_alerts: [],
      review_distribution: { total_reviews: 0 }
    })
    renderDashboard()

    await waitFor(()=> expect(screen.getByText('No existen alertas para graficar.')).toBeTruthy())
    expect(screen.getByText('No hay alertas recientes.')).toBeTruthy()
    expect(screen.getByText('Sin revisiones humanas suficientes.')).toBeTruthy()
  })
})
