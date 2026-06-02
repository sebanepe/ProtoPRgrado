import React from 'react'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../services/api')

import Dashboard from '../pages/Dashboard'
import * as api from '../services/api'

const overview = {
  source_run: 'preprocessed_run_26',
  total_transactions: 548108,
  active_alerts: 12,
  review_distribution: { new: 1, in_review: 2, dismissed: 3, confirmed_fraud: 2 },
}

const runs = [{ run_id: 'preprocessed_run_26', filename: 'preprocessed_run_26.csv', created_at: '2026-06-01T10:00:00Z' }]
const grouped = { total_items: 25, items: [{ summary_alert_id: 'A1' }] }
const detailed = { total_items: 80, items: [{ alert_id: 'D1' }] }
const reviews = { total_items: 6, items: [] }
const humanSummary = { total_reviews: 8, new: 1, in_review: 2, dismissed: 3, false_positive_excluded: 0, confirmed_fraud: 2 }
const anomalyRuns = { runs: [{ algorithm: 'isolation_forest', source_run: 'preprocessed_run_26', anomaly_count: 5481, created_at: '2026-06-01T10:00:00Z' }] }
const supervisedRuns = {
  items: [
    { algorithm: 'logistic_regression', source_run: 'preprocessed_run_26', status: 'AVAILABLE', created_at: '2026-06-01T10:00:00Z', metrics: { f1_score: 0.72 } },
    { algorithm: 'random_forest', source_run: 'preprocessed_run_26', status: 'AVAILABLE', created_at: '2026-06-01T10:00:00Z', metrics: { f1_score: 0.8 } },
    { algorithm: 'gradient_boosting', source_run: 'preprocessed_run_26', status: 'AVAILABLE', created_at: '2026-06-01T10:00:00Z', metrics: { f1_score: 0.68 } },
  ],
}
const evaluation = { metrics: { autoencoder: { anomaly_count: 10 }, supervised: {} } }
const scoringRuns = {
  count: 1,
  items: [{ id: 7, source_run: 'preprocessed_run_26', algorithm: 'random_forest', total_scored: 100, high_count: 10, medium_count: 30, low_count: 60, status: 'COMPLETED', created_at: '2026-06-01T10:00:00Z' }],
}
const casesSummary = {
  total: 4,
  by_status: { OPEN: 2, IN_ANALYSIS: 1, ESCALATED: 0, CLOSED: 1 },
  by_priority: { HIGH: 2, CRITICAL: 1 },
}

function mockDashboardData(overrides = {}) {
  api.getDashboardOverview.mockResolvedValue(overrides.overview ?? overview)
  api.getPreprocessedRuns.mockResolvedValue(overrides.runs ?? runs)
  api.getRulesSummary.mockResolvedValue(overrides.grouped ?? grouped)
  api.getRulesAlerts.mockResolvedValue(overrides.detailed ?? detailed)
  api.getAlertReviews.mockResolvedValue(overrides.reviews ?? reviews)
  api.getHumanLabelSummary.mockResolvedValue(overrides.humanSummary ?? humanSummary)
  api.getAnomalyRuns.mockResolvedValue(overrides.anomalyRuns ?? anomalyRuns)
  api.getSupervisedTrainingRuns.mockResolvedValue(overrides.supervisedRuns ?? supervisedRuns)
  api.getModelEvaluationSummary.mockResolvedValue(overrides.evaluation ?? evaluation)
  api.getBatchScoringRuns.mockResolvedValue(overrides.scoringRuns ?? scoringRuns)
  api.getCasesSummary.mockResolvedValue(overrides.casesSummary ?? casesSummary)
}

function renderDashboard() {
  return render(<MemoryRouter><Dashboard /></MemoryRouter>)
}

describe('Dashboard General', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockDashboardData()
  })

  afterEach(() => cleanup())

  it('renderiza titulo, mensaje metodologico y metricas principales', async () => {
    renderDashboard()

    await waitFor(() => expect(api.getDashboardOverview).toHaveBeenCalled())
    expect(screen.getByText('Dashboard General')).toBeTruthy()
    expect(screen.getByText(/señales de apoyo analítico/i)).toBeTruthy()
    expect(screen.getByText('Transacciones procesadas')).toBeTruthy()
    expect(screen.getByText('548.108')).toBeTruthy()
    expect(screen.getByText('Alertas agrupadas')).toBeTruthy()
    expect(screen.getByText('Alertas detalladas')).toBeTruthy()
    expect(screen.getByText('Revisiones humanas')).toBeTruthy()
    expect(screen.getByText('Modelos entrenados')).toBeTruthy()
  })

  it('muestra estado por fases, modelos, scoring y casos', async () => {
    renderDashboard()

    expect(await screen.findByText('Estado por fases')).toBeTruthy()
    expect(screen.getByText('Fase A: Data Pipeline')).toBeTruthy()
    expect(screen.getByText('Fase B: Reglas y Alertas')).toBeTruthy()
    expect(screen.getByText('Fase C: Modelos')).toBeTruthy()
    expect(screen.getByText('Fase D: Monitoreo')).toBeTruthy()
    expect(screen.getByText('Estado de modelos')).toBeTruthy()
    expect(screen.getByText('Isolation Forest')).toBeTruthy()
    expect(screen.getByText('Autoencoder PyTorch')).toBeTruthy()
    expect(screen.getByText('Últimos scorings')).toBeTruthy()
    expect(screen.getByText('random_forest')).toBeTruthy()
    expect(screen.getByText('Resumen de casos')).toBeTruthy()
    expect(screen.getByText('Casos por estado')).toBeTruthy()
  })

  it('muestra graficos y accesos rapidos', async () => {
    renderDashboard()

    expect(await screen.findByText('Distribución de revisión humana')).toBeTruthy()
    expect(screen.getByText('Scoring por nivel de prioridad')).toBeTruthy()
    expect(screen.getByText('Ver alertas')).toBeTruthy()
    expect(screen.getByText('Ejecutar scoring')).toBeTruthy()
    expect(screen.getByText('Ver casos')).toBeTruthy()
  })

  it('maneja datos vacios sin romper la pantalla', async () => {
    mockDashboardData({
      overview: {},
      runs: [],
      grouped: { total_items: 0, items: [] },
      detailed: { total_items: 0, items: [] },
      reviews: { total_items: 0, items: [] },
      humanSummary: {},
      anomalyRuns: { runs: [] },
      supervisedRuns: { items: [] },
      evaluation: {},
      scoringRuns: { count: 0, items: [] },
      casesSummary: { total: 0, by_status: {}, by_priority: {} },
    })
    renderDashboard()

    expect(await screen.findByText('Dashboard General')).toBeTruthy()
    expect(screen.getAllByText('Sin datos').length).toBeGreaterThan(0)
    expect(screen.getByText('Sin datos de scoring.')).toBeTruthy()
    expect(screen.getAllByText('No hay datos suficientes para este gráfico.').length).toBeGreaterThan(0)
  })

  it('muestra warning cuando hay error parcial', async () => {
    api.getRulesSummary.mockRejectedValue(new Error('Network Error'))
    renderDashboard()

    expect(await screen.findByText('Algunas métricas no pudieron cargarse.')).toBeTruthy()
    expect(screen.getByText('Dashboard General')).toBeTruthy()
  })

  it('no muestra campos prohibidos en minusculas', async () => {
    const { container } = renderDashboard()
    await screen.findByText('Dashboard General')

    expect(container.textContent).not.toContain('is_fraud')
    expect(container.textContent).not.toContain('confirmed_fraud')
  })
})
