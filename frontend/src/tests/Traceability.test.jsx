import React from 'react'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('../services/api')

import Traceability from '../pages/Traceability'
import * as api from '../services/api'

const rowFull = {
  dataset_id: 1,
  dataset_name: 'Dataset Prueba',
  dataset_filename: 'prueba.csv',
  dataset_status: 'IMPORTED',
  dataset_total_records: 500,
  dataset_created_at: '2026-06-01T10:00:00Z',
  dataset_uploaded_by: 'María López',
  preprocessing_run_id: 3,
  preprocessing_run_status: 'SUCCESS',
  preprocessing_processed_records: 490,
  preprocessing_started_at: '2026-06-01T10:05:00Z',
  preprocessing_finished_at: '2026-06-01T10:10:00Z',
  preprocessing_executed_by: 'Juan Pérez',
  rule_run_id: 7,
  rule_run_status: 'AVAILABLE',
  rule_run_created_at: '2026-06-01T11:00:00Z',
  detailed_alert_count: 15,
  grouped_alert_count: 5,
  detailed_new_count: 10,
  detailed_in_review_count: 2,
  detailed_dismissed_count: 1,
  detailed_false_positive_count: 1,
  detailed_confirmed_by_review_count: 1,
  grouped_new_count: 4,
  grouped_in_review_count: 1,
  grouped_dismissed_count: 0,
  grouped_false_positive_count: 0,
  grouped_confirmed_by_review_count: 0,
  artifact_preprocessed_csv: 'AVAILABLE',
  artifact_rule_alerts_csv: 'AVAILABLE',
  artifact_rule_summary_csv: 'AVAILABLE',
}

const rowNoPrepRun = {
  ...rowFull,
  dataset_id: 2,
  dataset_name: 'Dataset Sin Prep',
  dataset_uploaded_by: null,
  preprocessing_run_id: null,
  preprocessing_run_status: null,
  preprocessing_processed_records: null,
  preprocessing_started_at: null,
  preprocessing_finished_at: null,
  preprocessing_executed_by: null,
  rule_run_id: null,
  rule_run_status: null,
  rule_run_created_at: null,
  detailed_alert_count: 0,
  grouped_alert_count: 0,
  detailed_confirmed_by_review_count: 0,
  grouped_confirmed_by_review_count: 0,
  artifact_preprocessed_csv: 'MISSING',
  artifact_rule_alerts_csv: 'MISSING',
  artifact_rule_summary_csv: 'MISSING',
}

const rowNoRuleRun = {
  ...rowFull,
  dataset_id: 3,
  rule_run_id: null,
  rule_run_status: null,
  rule_run_created_at: null,
  detailed_alert_count: 0,
  grouped_alert_count: 0,
  artifact_rule_alerts_csv: 'MISSING',
  artifact_rule_summary_csv: 'MISSING',
}

const rowDerived = {
  ...rowFull,
  dataset_id: 4,
  dataset_name: 'MuestraMayoDefensa',
  rule_run_id: null,
  rule_run_status: 'DERIVADO',
  rule_run_created_at: null,
  detailed_alert_count: 250,
  grouped_alert_count: 80,
  artifact_rule_alerts_csv: 'AVAILABLE',
  artifact_rule_summary_csv: 'AVAILABLE',
}

function renderPage() {
  return render(<MemoryRouter><Traceability /></MemoryRouter>)
}

afterEach(() => { cleanup(); vi.clearAllMocks() })

describe('Traceability — Cuadro Resumen de Trazabilidad', () => {
  it('muestra estado de carga inicial', () => {
    api.getImportAlertTraceabilitySummary.mockReturnValue(new Promise(() => {}))
    renderPage()
    expect(screen.getByText('Cargando trazabilidad...')).toBeTruthy()
  })

  it('renderiza tarjetas KPI con datos', async () => {
    api.getImportAlertTraceabilitySummary.mockResolvedValue([rowFull])
    renderPage()
    await waitFor(() => expect(api.getImportAlertTraceabilitySummary).toHaveBeenCalled())
    expect(screen.getByText('Datasets importados')).toBeTruthy()
    expect(screen.getByText('Total alertas detalladas')).toBeTruthy()
    expect(screen.getByText('Alertas confirmadas por analista')).toBeTruthy()
  })

  it('muestra Sin preprocesamiento para datasets sin run', async () => {
    api.getImportAlertTraceabilitySummary.mockResolvedValue([rowNoPrepRun])
    renderPage()
    await waitFor(() => expect(api.getImportAlertTraceabilitySummary).toHaveBeenCalled())
    expect(screen.getByText('Sin preprocesamiento')).toBeTruthy()
  })

  it('muestra Sin reglas para dataset con prep pero sin rule run y sin artefactos', async () => {
    api.getImportAlertTraceabilitySummary.mockResolvedValue([rowNoRuleRun])
    renderPage()
    await waitFor(() => expect(api.getImportAlertTraceabilitySummary).toHaveBeenCalled())
    expect(screen.getByText('Sin reglas')).toBeTruthy()
  })

  it('muestra estado DERIVADO y conteos cuando existen artefactos sin RuleRun', async () => {
    api.getImportAlertTraceabilitySummary.mockResolvedValue([rowDerived])
    renderPage()
    await waitFor(() => expect(api.getImportAlertTraceabilitySummary).toHaveBeenCalled())
    expect(screen.getByText('DERIVADO')).toBeTruthy()
    expect(screen.getByText('Sin registro')).toBeTruthy()
  })

  it('muestra usuario importador y preprocesador', async () => {
    api.getImportAlertTraceabilitySummary.mockResolvedValue([rowFull])
    renderPage()
    await waitFor(() => expect(api.getImportAlertTraceabilitySummary).toHaveBeenCalled())
    expect(screen.getByText('María López')).toBeTruthy()
    expect(screen.getByText('Juan Pérez')).toBeTruthy()
  })

  it('muestra mensaje vacío si no hay datos', async () => {
    api.getImportAlertTraceabilitySummary.mockResolvedValue([])
    renderPage()
    await waitFor(() => expect(api.getImportAlertTraceabilitySummary).toHaveBeenCalled())
    expect(screen.getByText('No existen datasets importados.')).toBeTruthy()
  })

  it('maneja error del backend con mensaje amigable', async () => {
    api.getImportAlertTraceabilitySummary.mockRejectedValue(new Error('Network error'))
    renderPage()
    await waitFor(() => expect(api.getImportAlertTraceabilitySummary).toHaveBeenCalled())
    expect(screen.getByText('Network error')).toBeTruthy()
  })
})
