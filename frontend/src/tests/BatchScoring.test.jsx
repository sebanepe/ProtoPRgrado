import React from 'react'
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

vi.mock('../services/api')

import BatchScoring from '../pages/BatchScoring'
import * as api from '../services/api'

const MOCK_RUNS = {
  count: 1,
  items: [
    {
      id: 1,
      source_run: 'preprocessed_run_26',
      algorithm: 'logistic_regression',
      model_family: 'SUPERVISED_HUMAN',
      total_scored: 45,
      high_count: 17,
      medium_count: 3,
      low_count: 25,
      status: 'COMPLETED',
      created_at: '2026-05-30T10:00:00Z',
      results_file: 'scoring_results_run_26_logistic_regression.csv',
      report_file: 'scoring_report_run_26_logistic_regression.md',
      metadata_file: 'scoring_metadata_run_26_logistic_regression.json',
    },
  ],
}

const MOCK_RESULTS = {
  source_run: 'preprocessed_run_26',
  algorithm: 'logistic_regression',
  batch_scoring_run_id: 1,
  page: 1,
  page_size: 20,
  total: 3,
  total_pages: 1,
  rows: [
    {
      summary_alert_id: 'alert_001',
      representative_transaction_id: 'tx_001',
      customer_hash: 'hash_abc',
      rule_code: 'R01',
      rule_name: 'Regla prueba',
      ml_risk_score: 0.82,
      ml_risk_level: 'HIGH',
      algorithm: 'logistic_regression',
      scored_at: '2026-05-30T10:01:00Z',
    },
    {
      summary_alert_id: 'alert_002',
      representative_transaction_id: 'tx_002',
      customer_hash: 'hash_def',
      rule_code: 'R02',
      rule_name: 'Regla prueba 2',
      ml_risk_score: 0.61,
      ml_risk_level: 'MEDIUM',
      algorithm: 'logistic_regression',
      scored_at: '2026-05-30T10:01:01Z',
    },
    {
      summary_alert_id: 'alert_003',
      representative_transaction_id: 'tx_003',
      customer_hash: 'hash_ghi',
      rule_code: 'R03',
      rule_name: 'Regla prueba 3',
      ml_risk_score: 0.22,
      ml_risk_level: 'LOW',
      algorithm: 'logistic_regression',
      scored_at: '2026-05-30T10:01:02Z',
    },
  ],
}

const MOCK_REPORT = {
  source_run: 'preprocessed_run_26',
  algorithm: 'logistic_regression',
  batch_scoring_run_id: 1,
  report_file: 'scoring_report_run_26_logistic_regression.md',
  markdown: '# Reporte Scoring por Lotes D1\n\nTotal evaluados: 45\nHIGH: 17\nMEDIUM: 3\nLOW: 25',
}

const MOCK_METADATA = {
  source_run: 'preprocessed_run_26',
  algorithm: 'logistic_regression',
  batch_scoring_run_id: 1,
  metadata_file: 'scoring_metadata_run_26_logistic_regression.json',
  data: {
    source_run: 'preprocessed_run_26',
    algorithm: 'logistic_regression',
    model_family: 'SUPERVISED_HUMAN',
    total_scored: 45,
    high_count: 17,
    medium_count: 3,
    low_count: 25,
    low_medium_threshold: 0.5,
    medium_high_threshold: 0.75,
    methodology_warning: 'Scoring no confirma fraude automáticamente.',
  },
}

const MOCK_RUN_RESPONSE = {
  status: 'COMPLETED',
  batch_scoring_run_id: 1,
  source_run: 'preprocessed_run_26',
  algorithm: 'logistic_regression',
  total_scored: 45,
  high_count: 17,
  medium_count: 3,
  low_count: 25,
  low_medium_threshold: 0.5,
  medium_high_threshold: 0.75,
}

beforeEach(() => {
  vi.clearAllMocks()
  api.getBatchScoringRuns.mockResolvedValue(MOCK_RUNS)
  api.getBatchScoringResults.mockResolvedValue(MOCK_RESULTS)
  api.getBatchScoringReport.mockResolvedValue(MOCK_REPORT)
  api.getBatchScoringMetadata.mockResolvedValue(MOCK_METADATA)
  api.runBatchScoring.mockResolvedValue(MOCK_RUN_RESPONSE)
})

afterEach(() => {
  cleanup()
})

describe('BatchScoring – Fase D2', () => {
  // 1. Renderiza pantalla
  it('renderiza la pantalla Scoring por Lotes', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByText('Scoring por Lotes')).toBeTruthy()
  })

  // 2. Muestra mensaje metodológico
  it('muestra el mensaje metodológico obligatorio', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByText(/No confirma fraude automáticamente/i)).toBeTruthy()
  })

  // 3. Muestra formulario con campos
  it('muestra el formulario de ejecución con source_run, modelo y umbral', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByLabelText(/Source Run/i)).toBeTruthy()
    expect(screen.getByLabelText(/Modelo/i)).toBeTruthy()
    expect(screen.getByLabelText(/Umbral/i)).toBeTruthy()
  })

  // 4. Permite seleccionar algoritmo
  it('muestra los tres modelos en el selector', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    const select = screen.getByLabelText(/Modelo/i)
    expect(select).toBeTruthy()
    // getAllByText porque "Regresión logística" también aparece en la tabla y resumen
    expect(screen.getAllByText('Regresión logística').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Random Forest').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Gradient Boosting').length).toBeGreaterThan(0)
  })

  // 5. Permite cambiar threshold (UI-only)
  it('permite cambiar el umbral', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    const thresholdInput = screen.getByLabelText(/Umbral/i)
    fireEvent.change(thresholdInput, { target: { value: '0.7' } })
    expect(thresholdInput.value).toBe('0.7')
  })

  // 6. Llama runBatchScoring con payload correcto (sin threshold/sample_size)
  it('llama runBatchScoring con source_run y algorithm al ejecutar', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    const btn = screen.getByText('Ejecutar scoring')
    fireEvent.click(btn)
    await waitFor(() => expect(api.runBatchScoring).toHaveBeenCalledWith({
      source_run: 'preprocessed_run_26',
      algorithm: 'logistic_regression',
    }))
  })

  // 7. No incluye threshold ni sample_size en el payload
  it('no incluye threshold ni sample_size en el payload enviado al backend', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ejecutar scoring'))
    await waitFor(() => expect(api.runBatchScoring).toHaveBeenCalled())
    const call = api.runBatchScoring.mock.calls[0][0]
    expect(call.threshold).toBeUndefined()
    expect(call.sample_size).toBeUndefined()
  })

  // 8. Muestra lista de ejecuciones
  it('muestra la lista de ejecuciones de scoring', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByText('Ejecuciones de scoring')).toBeTruthy()
    // preprocessed_run_26 puede aparecer en múltiples celdas (resumen + tabla)
    expect(screen.getAllByText('preprocessed_run_26').length).toBeGreaterThan(0)
  })

  // 9. Muestra resumen de última ejecución con métricas
  it('muestra el resumen de última ejecución con total, HIGH, MEDIUM y LOW', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByText('Resumen de última ejecución')).toBeTruthy()
    expect(screen.getByText('Total evaluados')).toBeTruthy()
    const highs = screen.getAllByText('HIGH')
    expect(highs.length).toBeGreaterThan(0)
    const mediums = screen.getAllByText('MEDIUM')
    expect(mediums.length).toBeGreaterThan(0)
    const lows = screen.getAllByText('LOW')
    expect(lows.length).toBeGreaterThan(0)
  })

  // 10. Muestra botón "Ver resultados" para ejecuciones COMPLETED
  it('muestra el botón Ver resultados para ejecuciones completadas', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByText('Ver resultados')).toBeTruthy()
  })

  // 11. Al hacer click en "Ver resultados" carga los resultados
  it('carga resultados al hacer clic en Ver resultados', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados'))
    await waitFor(() => expect(api.getBatchScoringResults).toHaveBeenCalled())
    expect(screen.getByText('Resultados del scoring')).toBeTruthy()
  })

  // 12. Badge HIGH visible en resultados
  it('muestra badge HIGH en los resultados', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados'))
    await waitFor(() => expect(api.getBatchScoringResults).toHaveBeenCalled())
    const highBadges = screen.getAllByText('HIGH')
    expect(highBadges.length).toBeGreaterThan(0)
  })

  // 13. Badge MEDIUM visible en resultados
  it('muestra badge MEDIUM en los resultados', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados'))
    await waitFor(() => expect(api.getBatchScoringResults).toHaveBeenCalled())
    const medBadges = screen.getAllByText('MEDIUM')
    expect(medBadges.length).toBeGreaterThan(0)
  })

  // 14. Badge LOW visible en resultados
  it('muestra badge LOW en los resultados', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados'))
    await waitFor(() => expect(api.getBatchScoringResults).toHaveBeenCalled())
    const lowBadges = screen.getAllByText('LOW')
    expect(lowBadges.length).toBeGreaterThan(0)
  })

  // 15. Muestra sección de visualizaciones
  it('muestra los contenedores de visualización', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados'))
    await waitFor(() => expect(api.getBatchScoringResults).toHaveBeenCalled())
    expect(screen.getByText('Visualizaciones')).toBeTruthy()
    expect(screen.getByText('Distribución por nivel de prioridad')).toBeTruthy()
    expect(screen.getByText('Cantidad de resultados por predicción')).toBeTruthy()
    expect(screen.getByText('Distribución de probabilidades estimadas')).toBeTruthy()
  })

  // 16. Muestra reporte
  it('muestra el reporte del scoring', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados'))
    await waitFor(() => expect(api.getBatchScoringReport).toHaveBeenCalled())
    expect(screen.getByText('Reporte del scoring')).toBeTruthy()
    expect(screen.getByText(/Reporte Scoring por Lotes D1/i)).toBeTruthy()
  })

  // 17. Muestra metadata técnica
  it('muestra la metadata técnica al expandir', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados'))
    await waitFor(() => expect(api.getBatchScoringMetadata).toHaveBeenCalled())
    expect(screen.getByText(/Metadata técnica/i)).toBeTruthy()
  })

  // 18. No renderiza is_fraud
  it('no muestra is_fraud en los resultados', async () => {
    const resultsWithForbidden = {
      ...MOCK_RESULTS,
      rows: [...MOCK_RESULTS.rows, { summary_alert_id: 'x', is_fraud: 1, ml_risk_level: 'HIGH', ml_risk_score: 0.9 }],
    }
    api.getBatchScoringResults.mockResolvedValue(resultsWithForbidden)
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados'))
    await waitFor(() => expect(api.getBatchScoringResults).toHaveBeenCalled())
    expect(screen.queryByText('is_fraud')).toBeNull()
  })

  // 19. No renderiza confirmed_fraud
  it('no muestra confirmed_fraud en los resultados', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados'))
    await waitFor(() => expect(api.getBatchScoringResults).toHaveBeenCalled())
    expect(screen.queryByText('confirmed_fraud')).toBeNull()
  })

  // 20. No renderiza PAN_TARJETA ni TARJETA
  it('no muestra PAN_TARJETA ni TARJETA en los resultados', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados'))
    await waitFor(() => expect(api.getBatchScoringResults).toHaveBeenCalled())
    expect(screen.queryByText('PAN_TARJETA')).toBeNull()
    expect(screen.queryByText('TARJETA')).toBeNull()
  })

  // 21. Maneja error de backend sin romper
  it('muestra mensaje de error sin romper cuando el backend falla', async () => {
    api.getBatchScoringRuns.mockRejectedValue(new Error('Network Error'))
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByText('Scoring por Lotes')).toBeTruthy()
    expect(screen.getByText(/Backend no disponible/i)).toBeTruthy()
  })

  // 22. Estado vacío si no hay ejecuciones
  it('muestra mensaje vacío si no existen ejecuciones', async () => {
    api.getBatchScoringRuns.mockResolvedValue({ count: 0, items: [] })
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByText(/No existen ejecuciones de scoring/i)).toBeTruthy()
  })

  // 23. Muestra sección de interpretación guiada
  it('muestra la sección de interpretación guiada', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    expect(screen.getByText('Cómo interpretar el scoring')).toBeTruthy()
    expect(screen.getByText(/Estos niveles no son fraude confirmado/i)).toBeTruthy()
  })

  // 24. El botón se deshabilita durante ejecución
  it('deshabilita el botón durante la ejecución de scoring', async () => {
    let resolveRun
    api.runBatchScoring.mockReturnValue(new Promise(r => { resolveRun = r }))
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    const btn = screen.getByText('Ejecutar scoring')
    fireEvent.click(btn)
    // Tras el click, el botón muestra "Ejecutando..." (puede haber múltiples si se renderizan dos instancias)
    const execBtns = screen.getAllByText(/Ejecutando.../i)
    expect(execBtns.length).toBeGreaterThan(0)
    resolveRun(MOCK_RUN_RESPONSE)
  })

  // 25. Muestra filtro de nivel de prioridad en resultados
  it('muestra el filtro de nivel de prioridad en la sección de resultados', async () => {
    render(<BatchScoring />)
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados'))
    await waitFor(() => expect(api.getBatchScoringResults).toHaveBeenCalled())
    expect(screen.getByLabelText(/Nivel de prioridad/i)).toBeTruthy()
  })
})
