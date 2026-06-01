import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'

vi.mock('../services/api')

import App from '../App'
import ModelSupervised from '../pages/ModelSupervised'
import * as api from '../services/api'

const summary = {
  source_run: 'preprocessed_run_26',
  total_reviews: 48,
  confirmed_fraud: 20,
  dismissed: 25,
  new: 1,
  in_review: 1,
  false_positive_excluded: 1,
  usable_positive_labels: 20,
  usable_negative_labels: 25,
  usable_total_labels: 45,
  technical_ready: true,
  recommended_ready: false,
  strong_ready: false,
  verdict: 'HUMAN_LABELS_TECHNICALLY_READY'
}

const readiness = {
  source_run: 'preprocessed_run_26',
  technical_ready: true,
  recommended_ready: false,
  strong_ready: false,
  verdict: 'HUMAN_LABELS_TECHNICALLY_READY',
  current: { positive: 20, negative: 25, total: 45 },
  requirements: {
    technical: { positive: 20, negative: 20 },
    recommended: { positive: 50, negative: 120 },
    strong: { positive: 70, negative: 180 }
  }
}

const preflight = {
  source_run: 'preprocessed_run_26',
  human_labels: { confirmed_fraud: 20, dismissed: 25, usable_total: 45, technical_ready: true, recommended_ready: false, strong_ready: false },
  dataset: { exists: true, file: 'supervised_human_alert_dataset_run_26.csv', rows: 45, positive_count: 20, negative_count: 25, verdict: 'HUMAN_SUPERVISED_DATASET_READY' },
  artifact_registry: { supervised_dataset_registered: true, supervised_report_registered: true },
  supervised_dataset_runs: { registered: true, status: 'READY_TECHNICAL' },
  can_train: true,
  blocking_reason: null,
  warnings: []
}

const runs = {
  source_run: 'preprocessed_run_26',
  count: 3,
  items: [
    { algorithm: 'logistic_regression', source_run: 'preprocessed_run_26', status: 'AVAILABLE', is_active: false, created_at: '2026-06-01', metrics: { accuracy: 0.75, precision: 0.6667, recall: 0.8, f1_score: 0.7273, roc_auc: 0.8857, confusion_matrix: [[5, 2], [1, 4]] } },
    { algorithm: 'random_forest', source_run: 'preprocessed_run_26', status: 'AVAILABLE', is_active: false, created_at: '2026-06-01', metrics: { accuracy: 0.75, precision: 0.6667, recall: 0.8, f1_score: 0.7273, roc_auc: 0.8857, confusion_matrix: [[5, 2], [1, 4]] } },
    { algorithm: 'gradient_boosting', source_run: 'preprocessed_run_26', status: 'AVAILABLE', is_active: false, created_at: '2026-06-01', metrics: { accuracy: 0.6667, precision: 0.6, recall: 0.6, f1_score: 0.6, roc_auc: 0.7429, confusion_matrix: [[5, 2], [2, 3]] } }
  ]
}

const metadata = {
  metadata: {
    source_run: 'preprocessed_run_26',
    run_token: '26',
    model_family: 'SUPERVISED_HUMAN',
    algorithm: 'random_forest',
    target: 'target_human_label',
    label_policy: 'HUMAN_REVIEW_CONFIRMED_FRAUD_DISMISSED',
    positive_label: 'CONFIRMED_FRAUD',
    negative_label: 'DISMISSED',
    total_rows: 45,
    positive_count: 20,
    negative_count: 25,
    test_size: 0.2,
    random_state: 42,
    metrics: { accuracy: 0.75, precision: 0.6667, recall: 0.8, f1_score: 0.7273, roc_auc: 0.8857, confusion_matrix: [[5, 2], [1, 4]] },
    model_file: 'model.pkl',
    metadata_file: 'metadata.json',
    report_file: 'report.md',
    predictions_file: 'predictions.csv',
    warnings: ['preliminar']
  }
}

const predictions = {
  page: 1,
  total_pages: 1,
  total: 2,
  rows: [
    { summary_alert_id: 'A1', y_true: 1, y_pred: 1, y_proba: 0.91, prediction_label: 'CONFIRMED_FRAUD', evaluation_result: 'TRUE_POSITIVE' },
    { summary_alert_id: 'A2', y_true: 0, y_pred: 1, y_proba: 0.72, prediction_label: 'CONFIRMED_FRAUD', evaluation_result: 'FALSE_POSITIVE' }
  ]
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.setItem('user', JSON.stringify({ id: 1, email: 'user@example.com', token: 'token' }))
  api.me.mockResolvedValue({ id: 1, email: 'user@example.com', full_name: 'User' })
  api.getHumanLabelSummary.mockResolvedValue(summary)
  api.getHumanReadiness.mockResolvedValue(readiness)
  api.getSupervisedTrainingPreflight.mockResolvedValue(preflight)
  api.getHumanDatasetSummary.mockResolvedValue({ exists: true, dataset_file: 'supervised_human_alert_dataset_run_26.csv', rows: 45, positives: 20, negatives: 25, status: 'READY_TECHNICAL', created_at: '2026-06-01' })
  api.getSupervisedTrainingRuns.mockResolvedValue(runs)
  api.getSupervisedModelMetadata.mockResolvedValue(metadata)
  api.getSupervisedModelReport.mockResolvedValue({ markdown: '# Reporte random_forest\naccuracy: 0.75' })
  api.getSupervisedModelPredictions.mockResolvedValue(predictions)
  api.getHumanDatasetPreview.mockResolvedValue({ rows: [{ source_run: 'preprocessed_run_26', summary_alert_id: 'A1', target_human_label: 1 }] })
  api.validateHumanDataset.mockResolvedValue({ verdict: 'HUMAN_SUPERVISED_DATASET_READY' })
  api.buildHumanSupervisedDataset.mockResolvedValue({ verdict: 'HUMAN_SUPERVISED_DATASET_REUSED' })
  api.trainHumanSupervisedModel.mockResolvedValue({ verdict: 'HUMAN_SUPERVISED_TRAINING_COMPLETED', model_type: 'random_forest' })
})

afterEach(() => cleanup())

describe('ModelSupervised C4.5', () => {
  it('renders supervised screen, human summary, readiness and dataset section', async () => {
    render(<ModelSupervised />)
    expect(screen.getByText('Modelos Supervisados')).toBeTruthy()
    await waitFor(() => expect(api.getHumanLabelSummary).toHaveBeenCalledWith('preprocessed_run_26'))
    expect(await screen.findByText('Total de revisiones')).toBeTruthy()
    expect(screen.getByText('HUMAN_LABELS_TECHNICALLY_READY')).toBeTruthy()
    expect(screen.getByText('Dataset supervisado humano')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Construir dataset supervisado' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Validar dataset' })).toBeTruthy()
  })

  it('shows training controls and allows supported models when ready', async () => {
    render(<ModelSupervised />)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Entrenar modelo supervisado' })).toBeTruthy())
    const select = screen.getByLabelText('Modelo')
    fireEvent.change(select, { target: { value: 'logistic_regression' } })
    expect(screen.getByRole('button', { name: 'Entrenar modelo supervisado' }).disabled).toBe(false)
    fireEvent.change(select, { target: { value: 'random_forest' } })
    expect(screen.getByRole('button', { name: 'Entrenar modelo supervisado' }).disabled).toBe(false)
    fireEvent.change(select, { target: { value: 'gradient_boosting' } })
    expect(screen.getByRole('button', { name: 'Entrenar modelo supervisado' }).disabled).toBe(false)
  })

  it('blocks training when technical readiness is false', async () => {
    api.getHumanReadiness.mockResolvedValue({ ...readiness, technical_ready: false, verdict: 'INSUFFICIENT_HUMAN_LABELS' })
    api.getSupervisedTrainingPreflight.mockResolvedValue({ ...preflight, human_labels: { ...preflight.human_labels, technical_ready: false }, can_train: false, blocking_reason: 'INSUFFICIENT_HUMAN_LABELS' })
    render(<ModelSupervised />)
    await screen.findByText('INSUFFICIENT_HUMAN_LABELS')
    expect(screen.getByRole('button', { name: 'Entrenar modelo supervisado' }).disabled).toBe(true)
  })

  it('blocks MLP when recommended readiness is false', async () => {
    render(<ModelSupervised />)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Entrenar modelo supervisado' })).toBeTruthy())
    fireEvent.change(screen.getByLabelText('Modelo'), { target: { value: 'mlp_classifier' } })
    expect(screen.getByText('MLP requiere meta recomendada de etiquetas humanas para evitar sobreajuste.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Entrenar modelo supervisado' }).disabled).toBe(true)
  })

  it('calls supervised training with the expected payload', async () => {
    render(<ModelSupervised />)
    await screen.findByText('HUMAN_SUPERVISED_DATASET_READY')
    fireEvent.change(screen.getByLabelText('Modelo'), { target: { value: 'logistic_regression' } })
    fireEvent.click(screen.getByRole('button', { name: 'Entrenar modelo supervisado' }))
    await waitFor(() => expect(api.trainHumanSupervisedModel).toHaveBeenCalledWith({
      source_run: 'preprocessed_run_26',
      model_type: 'logistic_regression',
      test_size: 0.2,
      random_state: 42,
      use_smote: false
    }))
  })

  it('shows trained models, metrics comparison, confusion matrix, report, metadata and predictions', async () => {
    render(<ModelSupervised />)
    expect(await screen.findByText('Modelos supervisados entrenados')).toBeTruthy()
    expect(screen.getAllByText('random_forest').length).toBeGreaterThan(0)
    expect(screen.getByText('Comparacion de modelos supervisados')).toBeTruthy()
    expect(screen.getByText('Matriz de confusion')).toBeTruthy()
    expect(screen.getByText('Verdadero Positivo')).toBeTruthy()
    expect(screen.getByText('Reporte del modelo')).toBeTruthy()
    expect(await screen.findByText(/Reporte random_forest/)).toBeTruthy()
    expect(screen.getByText('Metadata del modelo')).toBeTruthy()
    expect(screen.getByText(/SUPERVISED_HUMAN/)).toBeTruthy()
    expect(screen.getByText('Predicciones de evaluacion')).toBeTruthy()
    expect(screen.getAllByText('TRUE_POSITIVE').length).toBeGreaterThan(0)
    expect(screen.getAllByText('FALSE_POSITIVE').length).toBeGreaterThan(0)
  })

  it('validates and previews the supervised dataset', async () => {
    render(<ModelSupervised />)
    await screen.findByText('Dataset supervisado humano')
    fireEvent.click(screen.getByRole('button', { name: 'Validar dataset' }))
    await waitFor(() => expect(api.validateHumanDataset).toHaveBeenCalledWith('preprocessed_run_26'))
    fireEvent.click(screen.getByRole('button', { name: 'Ver preview' }))
    await waitFor(() => expect(api.getHumanDatasetPreview).toHaveBeenCalled())
    expect(await screen.findByText('target_human_label')).toBeTruthy()
  })

  it('shows the required methodology warning and hides forbidden fields', async () => {
    render(<ModelSupervised />)
    await waitFor(() => expect(screen.getAllByText(/predicciones apoyan la priorizacion analitica/).length).toBeGreaterThan(0))
    expect(screen.queryByText('is_fraud')).toBeNull()
    expect(screen.queryByText('confirmed_fraud')).toBeNull()
    expect(screen.queryByText('PAN_TARJETA')).toBeNull()
    expect(screen.queryByText('TARJETA')).toBeNull()
  })

  it('handles endpoint errors without rendering raw details', async () => {
    api.getHumanLabelSummary.mockRejectedValue({ response: { status: 500, data: { detail: 'Traceback raw error' } } })
    render(<ModelSupervised />)
    expect(await screen.findByText('El servicio no esta disponible temporalmente.')).toBeTruthy()
    expect(screen.queryByText(/Traceback raw error/)).toBeNull()
  })
})

describe('ModelSupervised route', () => {
  it('keeps /models/supervised route and sidebar entry', async () => {
    render(
      <MemoryRouter initialEntries={['/models/supervised']}>
        <App />
      </MemoryRouter>
    )
    await waitFor(() => expect(screen.getAllByText('Supervisados').length).toBeGreaterThan(0))
    expect(await screen.findByText('Modelos Supervisados')).toBeTruthy()
  })
})
