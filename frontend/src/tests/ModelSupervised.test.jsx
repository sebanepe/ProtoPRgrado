import React from 'react'
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'

vi.mock('../services/api')

import App from '../App'
import ModelSupervised from '../pages/ModelSupervised'
import * as api from '../services/api'

const summary = {
  source_run: 'preprocessed_run_26',
  total_reviews: 15,
  confirmed_fraud: 4,
  dismissed: 6,
  new: 2,
  in_review: 1,
  false_positive_excluded: 2,
  usable_positive_labels: 4,
  usable_negative_labels: 6,
  usable_total_labels: 10,
  technical_min_positive_required: 20,
  technical_min_negative_required: 20,
  recommended_positive_required: 50,
  recommended_negative_required: 120,
  strong_positive_target: 70,
  strong_negative_target: 180,
  technical_ready: false,
  recommended_ready: false,
  strong_ready: false,
  missing_for_technical: { positive: 16, negative: 14 },
  missing_for_recommended: { positive: 46, negative: 114 },
  missing_for_strong: { positive: 66, negative: 174 },
  verdict: 'INSUFFICIENT_HUMAN_LABELS'
}

const readiness = {
  source_run: 'preprocessed_run_26',
  technical_ready: false,
  recommended_ready: false,
  strong_ready: false,
  verdict: 'INSUFFICIENT_HUMAN_LABELS',
  message: 'No existen suficientes etiquetas humanas para entrenar un modelo supervisado.',
  current: { positive: 4, negative: 6, total: 10 },
  requirements: {
    technical: { positive: 20, negative: 20 },
    recommended: { positive: 50, negative: 120 },
    strong: { positive: 70, negative: 180 }
  },
  missing: {
    technical: { positive: 16, negative: 14 },
    recommended: { positive: 46, negative: 114 },
    strong: { positive: 66, negative: 174 }
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.setItem('user', JSON.stringify({ id: 1, email: 'user@example.com', token: 'token' }))
  api.me.mockResolvedValue({ id: 1, email: 'user@example.com', full_name: 'User' })
  api.getHumanLabelSummary.mockResolvedValue(summary)
  api.getHumanReadiness.mockResolvedValue(readiness)
  if (api.trainModel) api.trainModel.mockResolvedValue({})
  if (api.trainAnomalyModel) api.trainAnomalyModel.mockResolvedValue({})
})

afterEach(() => {
  cleanup()
})

describe('ModelSupervised page', () => {
  it('renders supervised models screen and loads human label APIs', async () => {
    render(<ModelSupervised />)

    expect(screen.getByText('Modelos Supervisados')).toBeTruthy()
    expect(screen.getByText('Clasificacion de alertas revisadas a partir de etiquetas humanas.')).toBeTruthy()
    await waitFor(() => expect(api.getHumanLabelSummary).toHaveBeenCalledWith('preprocessed_run_26'))
    await waitFor(() => expect(api.getHumanReadiness).toHaveBeenCalledWith('preprocessed_run_26'))
  })

  it('shows methodology warning and human label counts', async () => {
    render(<ModelSupervised />)

    await waitFor(() => expect(screen.getByText('Total de revisiones')).toBeTruthy())
    expect(screen.getByText(/Los modelos supervisados requieren etiquetas humanas confiables/)).toBeTruthy()
    expect(screen.getByText('Confirmed Fraud')).toBeTruthy()
    expect(screen.getByText('Dismissed')).toBeTruthy()
    expect(screen.getByText('False Positive excluido')).toBeTruthy()
    expect(screen.getByText('New excluido')).toBeTruthy()
    expect(screen.getByText('In Review excluido')).toBeTruthy()
    expect(screen.getByText('Positivas usables')).toBeTruthy()
    expect(screen.getByText('Negativas usables')).toBeTruthy()
    expect(screen.getByText('Total usable')).toBeTruthy()
    expect(screen.getAllByText('4').length).toBeGreaterThan(0)
    expect(screen.getAllByText('6').length).toBeGreaterThan(0)
    expect(screen.getByText('10')).toBeTruthy()
  })

  it('shows technical, recommended and strong progress requirements', async () => {
    render(<ModelSupervised />)

    await waitFor(() => expect(screen.getByText('Minimo tecnico')).toBeTruthy())
    expect(screen.getByText('Recomendado')).toBeTruthy()
    expect(screen.getByText('Meta fuerte')).toBeTruthy()
    expect(await screen.findByLabelText('Positivos: 4 / 20')).toBeTruthy()
    expect(await screen.findByLabelText('Negativos: 6 / 20')).toBeTruthy()
    expect(await screen.findByLabelText('Positivos: 4 / 50')).toBeTruthy()
    expect(await screen.findByLabelText('Negativos: 6 / 120')).toBeTruthy()
    expect(await screen.findByLabelText('Positivos: 4 / 70')).toBeTruthy()
    expect(await screen.findByLabelText('Negativos: 6 / 180')).toBeTruthy()
  })

  it('shows insufficient readiness and keeps supervised training disabled', async () => {
    render(<ModelSupervised />)

    await waitFor(() => expect(screen.getByText('INSUFFICIENT_HUMAN_LABELS')).toBeTruthy())
    expect(screen.getByText('No existen suficientes etiquetas humanas para entrenar un modelo supervisado.')).toBeTruthy()
    const trainButton = screen.getByRole('button', { name: 'Entrenar modelo supervisado' })
    expect(trainButton.disabled).toBe(true)
  })

  it('keeps training as future-only even when technically ready and does not call training endpoints', async () => {
    api.getHumanLabelSummary.mockResolvedValue({
      ...summary,
      confirmed_fraud: 20,
      dismissed: 20,
      usable_positive_labels: 20,
      usable_negative_labels: 20,
      usable_total_labels: 40,
      technical_ready: true,
      verdict: 'HUMAN_LABELS_TECHNICALLY_READY'
    })
    api.getHumanReadiness.mockResolvedValue({
      ...readiness,
      technical_ready: true,
      verdict: 'HUMAN_LABELS_TECHNICALLY_READY',
      current: { positive: 20, negative: 20, total: 40 }
    })

    render(<ModelSupervised />)

    await waitFor(() => expect(screen.getByText('HUMAN_LABELS_TECHNICALLY_READY')).toBeTruthy())
    const trainButton = screen.getByRole('button', { name: /Disponible en C4.4/ })
    expect(trainButton.disabled).toBe(true)
    expect(api.trainModel).not.toHaveBeenCalled()
    expect(api.trainAnomalyModel).not.toHaveBeenCalled()
  })

  it('reloads summary and readiness when source_run changes', async () => {
    render(<ModelSupervised />)

    await waitFor(() => expect(api.getHumanLabelSummary).toHaveBeenCalledWith('preprocessed_run_26'))
    fireEvent.change(screen.getByLabelText('source_run'), { target: { value: 'preprocessed_run_99' } })

    await waitFor(() => expect(api.getHumanLabelSummary).toHaveBeenCalledWith('preprocessed_run_99'))
    await waitFor(() => expect(api.getHumanReadiness).toHaveBeenCalledWith('preprocessed_run_99'))
  })

  it('shows controlled error messages without raw Axios details', async () => {
    api.getHumanLabelSummary.mockRejectedValue({ response: { status: 500, data: { detail: 'Traceback raw error' } } })
    api.getHumanReadiness.mockRejectedValue({ response: { status: 404, data: { detail: 'Not Found raw error' } } })

    render(<ModelSupervised />)

    await waitFor(() => expect(screen.getByText('El servicio de etiquetas humanas no esta disponible temporalmente.')).toBeTruthy())
    expect(screen.getByText('No se encontro informacion de etiquetas humanas para el run seleccionado.')).toBeTruthy()
    expect(screen.queryByText(/Traceback raw error/)).toBeNull()
    expect(screen.queryByText(/Not Found raw error/)).toBeNull()
  })

  it('does not display forbidden automatic target fields', async () => {
    render(<ModelSupervised />)

    await waitFor(() => expect(screen.getByText('Criterio metodologico')).toBeTruthy())
    expect(screen.queryByText('is_fraud')).toBeNull()
    expect(screen.queryByText('confirmed_fraud automatico')).toBeNull()
  })
})

describe('ModelSupervised route', () => {
  it('renders /models/supervised through App and sidebar', async () => {
    render(
      <MemoryRouter initialEntries={['/models/supervised']}>
        <App />
      </MemoryRouter>
    )

    await waitFor(() => expect(screen.getAllByText('Supervisados').length).toBeGreaterThan(0))
    await waitFor(() => expect(screen.getByText('Modelos Supervisados')).toBeTruthy())
  })
})
