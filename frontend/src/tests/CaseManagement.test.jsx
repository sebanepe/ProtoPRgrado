import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../services/api')

import Cases from '../pages/Cases'
import * as api from '../services/api'

const SUMMARY = {
  total: 2,
  by_status: { OPEN: 1, IN_ANALYSIS: 1, ESCALATED: 0, CLOSED: 0 },
  by_priority: { LOW: 0, MEDIUM: 1, HIGH: 1, CRITICAL: 0 },
}

const CASES = [
  {
    id: 1,
    case_code: 'CASE-202606-00001',
    title: 'Caso scoring alto',
    source_run: 'preprocessed_run_26',
    origin_type: 'SCORING_RESULT',
    priority: 'HIGH',
    status: 'OPEN',
    customer_hash: 'hash_abc',
    assigned_to: 'ana',
    created_at: '2026-06-01T10:00:00Z',
    is_fraud: 1,
    confirmed_fraud: true,
    PAN_TARJETA: '4111111111111111',
    TARJETA: '411111',
  },
  {
    id: 2,
    case_code: 'CASE-202606-00002',
    title: 'Caso manual',
    source_run: 'manual_run',
    origin_type: 'MANUAL',
    priority: 'MEDIUM',
    status: 'IN_ANALYSIS',
    customer_hash: 'hash_def',
    assigned_to: 'luis',
    created_at: '2026-06-01T11:00:00Z',
  },
]

const CASE_DETAIL = {
  ...CASES[0],
  description: 'Revisión operativa por prioridad alta',
  origin_ref_id: 'score_001',
  summary_alert_id: 'alert_001',
  transaction_id: 'txn_001',
  scoring_run_id: 'run_001',
  created_by: 'system',
  closed_by: null,
  conclusion: null,
  updated_at: '2026-06-01T10:10:00Z',
  closed_at: null,
}

const COMMENTS = [
  { id: 1, user_id: 'ana', comment_text: 'Revisando patrón', created_at: '2026-06-01T10:20:00Z' },
]

const HISTORY = [
  { id: 1, action: 'CASE_CREATED', old_value: null, new_value: 'CASE-202606-00001', changed_by: 'system', changed_at: '2026-06-01T10:00:00Z' },
  { id: 2, action: 'STATUS_CHANGED', old_value: 'OPEN', new_value: 'IN_ANALYSIS', changed_by: 'ana', changed_at: '2026-06-01T10:30:00Z' },
]

const SCORING_RUNS = {
  items: [
    { id: 10, source_run: 'preprocessed_run_26', algorithm: 'logistic_regression', status: 'COMPLETED' },
  ],
}

const SCORING_RESULTS = {
  rows: [
    {
      summary_alert_id: 'alert_score_001',
      representative_transaction_id: 'txn_score_001',
      customer_hash: 'hash_score',
      ml_risk_score: 0.91,
      ml_risk_level: 'HIGH',
      algorithm: 'logistic_regression',
      scored_at: '2026-06-01T12:00:00Z',
      batch_scoring_run_id: 10,
    },
  ],
}

beforeEach(() => {
  vi.clearAllMocks()
  api.getCasesSummary.mockResolvedValue(SUMMARY)
  api.getCases.mockResolvedValue({ total: CASES.length, items: CASES })
  api.getCaseById.mockResolvedValue(CASE_DETAIL)
  api.getCaseComments.mockResolvedValue(COMMENTS)
  api.getCaseHistory.mockResolvedValue(HISTORY)
  api.createCase.mockResolvedValue({ ...CASE_DETAIL, id: 3, case_code: 'CASE-202606-00003' })
  api.createCaseFromScoringResult.mockResolvedValue({ ...CASE_DETAIL, id: 4, case_code: 'CASE-202606-00004' })
  api.updateCase.mockResolvedValue({ ...CASE_DETAIL, priority: 'CRITICAL', status: 'ESCALATED' })
  api.addCaseComment.mockResolvedValue({ id: 2, comment_text: 'Nuevo comentario' })
  api.closeCase.mockResolvedValue({ ...CASE_DETAIL, status: 'CLOSED', conclusion: 'Conclusión operativa' })
  api.reopenCase.mockResolvedValue({ ...CASE_DETAIL, status: 'OPEN' })
  api.getBatchScoringRuns.mockResolvedValue(SCORING_RUNS)
  api.getBatchScoringResults.mockResolvedValue(SCORING_RESULTS)
})

afterEach(() => cleanup())

async function renderCases() {
  render(<MemoryRouter><Cases /></MemoryRouter>)
  await waitFor(() => expect(api.getCases).toHaveBeenCalled())
}

async function openDetail() {
  await renderCases()
  fireEvent.click(screen.getAllByText('Ver detalle')[0])
  await waitFor(() => expect(api.getCaseById).toHaveBeenCalledWith(1))
}

describe('CaseManagement - Fase D3.2', () => {
  it('renderiza pantalla Manejo de Casos', async () => {
    await renderCases()
    expect(screen.getByText('Manejo de Casos')).toBeTruthy()
  })

  it('muestra mensaje metodológico', async () => {
    await renderCases()
    expect(screen.getByText(/La creación o cierre de un caso no confirma fraude automáticamente/i)).toBeTruthy()
  })

  it('carga resumen', async () => {
    await renderCases()
    expect(screen.getByText('Total de casos')).toBeTruthy()
    expect(screen.getByText('Alta prioridad')).toBeTruthy()
  })

  it('muestra tabla de casos', async () => {
    await renderCases()
    expect(screen.getByText('CASE-202606-00001')).toBeTruthy()
    expect(screen.getByText('Caso scoring alto')).toBeTruthy()
  })

  it('filtra por estado/prioridad', async () => {
    await renderCases()
    fireEvent.change(screen.getByLabelText('Estado'), { target: { value: 'OPEN' } })
    await waitFor(() => expect(api.getCases).toHaveBeenLastCalledWith({ page: 1, page_size: 200, status: 'OPEN' }))
    fireEvent.change(screen.getByLabelText('Prioridad'), { target: { value: 'HIGH' } })
    await waitFor(() => expect(api.getCases).toHaveBeenLastCalledWith({ page: 1, page_size: 200, status: 'OPEN', priority: 'HIGH' }))
  })

  it('abre modal de nuevo caso', async () => {
    await renderCases()
    fireEvent.click(screen.getByText('Nuevo caso'))
    expect(screen.getByText('Crear caso')).toBeTruthy()
  })

  it('valida título requerido', async () => {
    await renderCases()
    fireEvent.click(screen.getByText('Nuevo caso'))
    fireEvent.change(screen.getByLabelText('description'), { target: { value: 'Descripción' } })
    fireEvent.click(screen.getByText('Crear caso'))
    expect(screen.getByText('El título es requerido.')).toBeTruthy()
  })

  it('llama createCase con payload correcto', async () => {
    await renderCases()
    fireEvent.click(screen.getByText('Nuevo caso'))
    fireEvent.change(screen.getByLabelText('title'), { target: { value: 'Nuevo caso manual' } })
    fireEvent.change(screen.getByLabelText('description'), { target: { value: 'Descripción operativa' } })
    fireEvent.change(screen.getByLabelText('priority'), { target: { value: 'HIGH' } })
    fireEvent.click(screen.getByText('Crear caso'))
    await waitFor(() => expect(api.createCase).toHaveBeenCalledWith({
      origin_type: 'MANUAL',
      title: 'Nuevo caso manual',
      description: 'Descripción operativa',
      priority: 'HIGH',
    }))
  })

  it('crea caso desde resultado de scoring prellenado', async () => {
    await renderCases()
    fireEvent.click(screen.getByText('Cargar ejecuciones'))
    await waitFor(() => expect(api.getBatchScoringRuns).toHaveBeenCalled())
    fireEvent.click(screen.getByText('Ver resultados de scoring'))
    await waitFor(() => expect(api.getBatchScoringResults).toHaveBeenCalledWith({
      source_run: 'preprocessed_run_26',
      algorithm: 'logistic_regression',
      page: 1,
      page_size: 50,
    }))
    fireEvent.click(screen.getByText('Usar como caso'))
    expect(screen.getByLabelText('title').value).toContain('alert_score_001')
    fireEvent.click(screen.getByText('Crear caso'))
    await waitFor(() => expect(api.createCaseFromScoringResult).toHaveBeenCalled())
    const payload = api.createCaseFromScoringResult.mock.calls[0][0]
    expect(payload.origin_type).toBe('SCORING_RESULT')
    expect(payload.summary_alert_id).toBe('alert_score_001')
    expect(payload.transaction_id).toBe('txn_score_001')
    expect(payload.scoring_run_id).toBe('10')
    expect(payload.priority).toBe('HIGH')
  })

  it('abre detalle de caso', async () => {
    await openDetail()
    expect(screen.getAllByText('Revisión operativa por prioridad alta').length).toBeGreaterThan(0)
    expect(screen.getByText('alert_001')).toBeTruthy()
  })

  it('actualiza prioridad/estado', async () => {
    await openDetail()
    fireEvent.change(screen.getAllByLabelText('Prioridad')[1], { target: { value: 'CRITICAL' } })
    fireEvent.change(screen.getAllByLabelText('Estado')[1], { target: { value: 'ESCALATED' } })
    fireEvent.click(screen.getByText('Guardar cambios'))
    await waitFor(() => expect(api.updateCase).toHaveBeenCalled())
    const [, payload] = api.updateCase.mock.calls[0]
    expect(payload.priority).toBe('CRITICAL')
    expect(payload.status).toBe('ESCALATED')
  })

  it('agrega comentario', async () => {
    await openDetail()
    fireEvent.change(screen.getByLabelText('Agregar comentario'), { target: { value: 'Comentario nuevo' } })
    fireEvent.click(screen.getAllByText('Agregar comentario')[1])
    await waitFor(() => expect(api.addCaseComment).toHaveBeenCalledWith(1, { comment_text: 'Comentario nuevo' }))
  })

  it('muestra historial', async () => {
    await openDetail()
    expect(screen.getByText('CASE_CREATED')).toBeTruthy()
    expect(screen.getByText('STATUS_CHANGED')).toBeTruthy()
  })

  it('cierra caso con conclusión', async () => {
    await openDetail()
    fireEvent.change(screen.getByLabelText('Conclusión'), { target: { value: 'Conclusión operativa' } })
    fireEvent.click(screen.getByText('Cerrar caso'))
    await waitFor(() => expect(api.closeCase).toHaveBeenCalledWith(1, { conclusion: 'Conclusión operativa' }))
  })

  it('bloquea cierre sin conclusión', async () => {
    await openDetail()
    fireEvent.click(screen.getByText('Cerrar caso'))
    expect(screen.getByText('Debe registrar una conclusión para cerrar el caso.')).toBeTruthy()
    expect(api.closeCase).not.toHaveBeenCalled()
  })

  it('reabre caso', async () => {
    api.getCaseById.mockResolvedValue({ ...CASE_DETAIL, status: 'CLOSED', closed_at: '2026-06-01T12:00:00Z', conclusion: 'Cerrado' })
    await openDetail()
    fireEvent.click(screen.getByText('Reabrir caso'))
    await waitFor(() => expect(api.reopenCase).toHaveBeenCalledWith(1))
  })

  it('no muestra is_fraud', async () => {
    await renderCases()
    expect(screen.queryByText('is_fraud')).toBeNull()
  })

  it('no muestra confirmed_fraud', async () => {
    await renderCases()
    expect(screen.queryByText('confirmed_fraud')).toBeNull()
  })

  it('no muestra PAN_TARJETA', async () => {
    await renderCases()
    expect(screen.queryByText('PAN_TARJETA')).toBeNull()
  })

  it('no muestra TARJETA', async () => {
    await renderCases()
    expect(screen.queryByText('TARJETA')).toBeNull()
  })

  it('maneja error de backend', async () => {
    api.getCases.mockRejectedValue(new Error('Network Error'))
    render(<MemoryRouter><Cases /></MemoryRouter>)
    await waitFor(() => expect(api.getCases).toHaveBeenCalled())
    expect(screen.getByText(/Backend no disponible/i)).toBeTruthy()
  })
})
