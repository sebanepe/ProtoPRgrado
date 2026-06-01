import React from 'react'
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react'
import { vi, describe, it, beforeEach, afterEach, expect } from 'vitest'
import ModelEvaluation from '../../src/pages/ModelEvaluation'

vi.mock('../../src/services/api', () => ({
  buildModelEvaluationComparison: vi.fn(),
  getModelEvaluationSummary: vi.fn(),
  getModelEvaluationAlertLevel: vi.fn(),
  getModelEvaluationTransactionLevel: vi.fn(),
  getModelEvaluationReport: vi.fn(),
  getModelEvaluationMetadata: vi.fn(),
  getModelEvaluationTopCases: vi.fn()
}))

import {
  buildModelEvaluationComparison,
  getModelEvaluationSummary,
  getModelEvaluationAlertLevel,
  getModelEvaluationTransactionLevel,
  getModelEvaluationReport,
  getModelEvaluationMetadata,
  getModelEvaluationTopCases
} from '../../src/services/api'

describe('ModelEvaluation page', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    getModelEvaluationSummary.mockResolvedValue({
      metrics: {
        rules: { total_alerts_grouped: 10, total_alerts_detailed: 25, total_rules: 2 },
        isolation_forest: { anomaly_count: 6 },
        autoencoder: { anomaly_count: 4 },
        supervised: {
          logistic_regression: { accuracy: 0.8, precision: 0.7, recall: 0.6, f1_score: 0.65, roc_auc: 0.77, false_positive_count: 2, false_negative_count: 3 },
          random_forest: { accuracy: 0.82, precision: 0.74, recall: 0.68, f1_score: 0.7, roc_auc: 0.8, false_positive_count: 1, false_negative_count: 2 }
        }
      },
      intersections: {
        rules_and_isolation_count: 3,
        rules_and_autoencoder_count: 2,
        isolation_and_autoencoder_count: 1,
        rules_and_supervised_positive_count: 4,
        all_available_methods_count: 2
      },
      available_methods: ['rules', 'isolation_forest', 'autoencoder', 'random_forest'],
      missing_methods: ['gradient_boosting']
    })
    getModelEvaluationTopCases.mockResolvedValue({ items: [{ summary_alert_id: 'A1', customer_hash: 'C1', rule_code: 'RULE_X' }] })
    getModelEvaluationAlertLevel.mockResolvedValue({ items: [{ summary_alert_id: 'A1', human_review_status: 'IN_REVIEW' }], total: 1, page: 1, page_size: 20 })
    getModelEvaluationTransactionLevel.mockResolvedValue({ items: [{ transaction_id: 'T1', flagged_by_rules: true }], total: 1, page: 1, page_size: 20 })
    getModelEvaluationReport.mockResolvedValue({ markdown: '# reporte' })
    getModelEvaluationMetadata.mockResolvedValue({ generated_at: '2026-06-01T00:00:00Z' })
    buildModelEvaluationComparison.mockResolvedValue({ status: 'ok', result: { status: 'READY' } })
  })

  it('renders C5.2 sections and calls summary api', async () => {
    render(<ModelEvaluation />)
    await waitFor(() => expect(getModelEvaluationSummary).toHaveBeenCalled())

    expect(screen.getByText(/Evaluacion de Modelos/i)).toBeTruthy()
    expect(screen.getByTestId('methodology-message').textContent).toMatch(/Ninguna constituye fraude confirmado automatico/i)
    expect(screen.getByLabelText('source_run')).toBeTruthy()
    expect(screen.getByText('Reglas (agrupadas)')).toBeTruthy()
    expect(screen.getByText('Isolation anomalas')).toBeTruthy()
    expect(screen.getByText('Autoencoder anomalas')).toBeTruthy()
    expect(screen.getAllByTestId('signals-chart-container').length).toBeGreaterThan(0)
    expect(screen.getByTestId('supervised-comparison-section')).toBeTruthy()
    expect(screen.getByTestId('top-cases-section')).toBeTruthy()
    expect(screen.getByTestId('alert-level-table')).toBeTruthy()
    expect(screen.getByTestId('transaction-level-table')).toBeTruthy()
    expect(screen.getByTestId('report-section')).toBeTruthy()
    expect(screen.getByTestId('metadata-section')).toBeTruthy()

    expect(screen.queryByText(/is_fraud/i)).toBeNull()
    expect(screen.queryByText(/confirmed_fraud/i)).toBeNull()
    expect(screen.queryByText(/PAN_TARJETA/i)).toBeNull()
    expect(screen.queryByText(/TARJETA/i)).toBeNull()
  })

  it('runs build comparison when update button is clicked', async () => {
    render(<ModelEvaluation />)
    await waitFor(() => expect(getModelEvaluationSummary).toHaveBeenCalled())
    fireEvent.click(screen.getAllByText(/Actualizar comparacion/i)[0])
    await waitFor(() => expect(buildModelEvaluationComparison).toHaveBeenCalled())
  })

  it('handles partial ready without crashing', async () => {
    buildModelEvaluationComparison.mockResolvedValueOnce({ status: 'ok', result: { status: 'PARTIAL_READY', missing_methods: ['autoencoder'] } })
    render(<ModelEvaluation />)
    await waitFor(() => expect(getModelEvaluationSummary).toHaveBeenCalled())
    fireEvent.click(screen.getAllByText(/Actualizar comparacion/i)[0])
    await waitFor(() => expect(screen.getByText(/generada parcialmente/i)).toBeTruthy())
  })
})
