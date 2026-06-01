import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { vi, describe, it, beforeEach, afterEach, expect } from 'vitest'
import Models from '../../src/pages/Models'

vi.mock('../../src/services/api', () => ({
  getAnomalyRuns: vi.fn(),
  getAnomalyMetrics: vi.fn(),
  getAnomalyScores: vi.fn(),
  getTopAnomalies: vi.fn(),
  getAnomalyReport: vi.fn(),
  getAnomalyModelMetadata: vi.fn(),
  trainAnomalyModel: vi.fn(),
  getAutoencoderMetrics: vi.fn(),
  getAutoencoderScores: vi.fn(),
  getAutoencoderReport: vi.fn(),
  getAutoencoderModelMetadata: vi.fn(),
  trainAutoencoderAnomaly: vi.fn(),
}))
import {
  getAnomalyRuns,
  getAnomalyMetrics,
  getAnomalyScores,
  getTopAnomalies,
  getAnomalyReport,
  getAnomalyModelMetadata,
} from '../../src/services/api'

describe('Models page', () => {
  beforeEach(() => {
    getAnomalyRuns.mockResolvedValue({
      runs: [
        {
          anomaly_run_id: 'run_26',
          source_run: 'preprocessed_run_26',
          model_name: 'isolation_forest',
          algorithm: 'isolation_forest',
          anomaly_count: 5481,
          anomaly_rate: 0.009999854,
          created_at: '2026-05-30T00:33:18.268840+00:00',
        },
      ],
    })
    getAnomalyMetrics.mockResolvedValue({
      total_transactions: 548108,
      anomaly_count: 5481,
      anomaly_rate: 0.009999854,
      model_name: 'unsupervised_anomaly_detection',
      algorithm: 'isolation_forest',
      contamination: 0.01,
      anomalies_by_country: { BO: 1 },
      anomalies_by_pos_entry_mode: { '5': 1 },
      anomalies_by_mcc: { '6011': 1 },
      anomalies_by_hour: { '0': 1 },
      top_customers_by_anomaly_count: [{ customer_hash: 'cust_1', count: 1 }],
    })
    getAnomalyScores.mockResolvedValue({ page: 1, page_size: 50, total_items: 1, total_pages: 1, items: [] })
    getTopAnomalies.mockResolvedValue({ items: [] })
    getAnomalyReport.mockResolvedValue({ report: 'source_run: preprocessed_run_26' })
    getAnomalyModelMetadata.mockResolvedValue({ metadata: { model_name: 'isolation_forest', model_type: 'unsupervised_anomaly_detection', algorithm: 'isolation_forest', contamination: 0.01, total_rows: 548108, anomaly_count: 5481, anomaly_rate: 0.009999854, numeric_features: [], categorical_features: [], model_input_columns: [], excluded_columns: [], model_path: '', score_file: '', feature_file: '', report_file: '' } })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('loads the unsupervised anomaly dashboard', async () => {
    render(<Models />)
    await waitFor(() => expect(getAnomalyRuns).toHaveBeenCalled())
    expect(screen.getAllByText('Modelos No Supervisados').length).toBeGreaterThan(0)
    expect(screen.getByText('Detectan comportamientos atípicos sin usar etiquetas humanas.')).toBeTruthy()
    expect(screen.getByText('Las anomalías detectadas por los modelos no supervisados no representan fraude confirmado. Son señales de comportamiento atípico que requieren revisión.')).toBeTruthy()
  })
})
