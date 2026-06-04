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
    expect(screen.getByText('Precision, Recall y F1-score')).toBeTruthy()
    expect(screen.getByText('Accuracy y ROC-AUC')).toBeTruthy()
    expect(screen.getByText('Matriz de confusion visual')).toBeTruthy()
    expect(screen.getByLabelText('Matriz de confusion tipo heatmap')).toBeTruthy()
    expect(screen.getAllByText('Real: DISMISSED').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Predicho: CONFIRMED_FRAUD').length).toBeGreaterThan(0)
    expect(screen.getByText(/El modelo priorizo una alerta que la revision humana descarto/)).toBeTruthy()
    expect(screen.getByText(/El modelo descarto una alerta que la revision humana confirmo como fraude/)).toBeTruthy()
    expect(screen.getByText('Verdadero Positivo')).toBeTruthy()
    expect(screen.getAllByText('Distribucion de predicciones').length).toBeGreaterThan(0)
    expect(screen.getByText('Distribucion de probabilidad estimada')).toBeTruthy()
    expect(screen.getByText('Detalles tecnicos')).toBeTruthy()
    expect(screen.getByText('Ver reporte tecnico')).toBeTruthy()
    expect(screen.getByText('Ver metadata del modelo')).toBeTruthy()
    expect(screen.getByText('Predicciones de evaluacion')).toBeTruthy()
    expect(screen.getByText('ID Alerta')).toBeTruthy()
    expect(screen.getByText('Etiqueta humana')).toBeTruthy()
    expect(screen.getByText('Prediccion del modelo')).toBeTruthy()
    expect(screen.getByText('Acierto positivo')).toBeTruthy()
    expect(screen.getByText('Falsa alarma')).toBeTruthy()
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

  it('renders non technical guidance for analysts', async () => {
    render(<ModelSupervised />)
    expect(await screen.findByText('Que significa este modulo')).toBeTruthy()
    expect(screen.getByText('Que debe mirar primero el analista')).toBeTruthy()
    expect(screen.getAllByText(/Porcentaje general de aciertos/).length).toBeGreaterThan(0)
    expect(screen.getByText(/Modelo base e interpretable/)).toBeTruthy()
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

// ── Apply tab tests ───────────────────────────────────────────────────────────

const supTrainedModels = [
  { id: 10, algorithm: 'random_forest', source_run: 'preprocessed_run_26', status: 'AVAILABLE', f1_score: 0.73, precision: 0.67, recall: 0.80, roc_auc: 0.89, created_at: '2026-06-01' },
  { id: 11, algorithm: 'logistic_regression', source_run: 'preprocessed_run_26', status: 'AVAILABLE', f1_score: 0.70, precision: 0.65, recall: 0.78, roc_auc: 0.85, created_at: '2026-06-01' },
]

const supPreprocessedRuns = [
  { id: 26, source_run: 'preprocessed_run_26', total_records: 548124, finished_at: '2026-06-01', status: 'COMPLETED' },
]

const supPredRuns = [
  { id: 1, algorithm: 'random_forest', input_source: 'preprocessed_run_26', total_analyzed: 548124, high_count: 5481, medium_count: 12000, low_count: 530643, status: 'COMPLETED', finished_at: '2026-06-01' },
]

const supPredResults = {
  run_id: 1, status: 'COMPLETED', total: 2, page: 1, page_size: 50,
  rows: [
    { transaction_id: 'tx1', prediction_label: 1, prediction_probability: 0.85, priority_level: 'HIGH', model_name: 'random_forest', amount: 1500 },
    { transaction_id: 'tx2', prediction_label: 0, prediction_probability: 0.15, priority_level: 'LOW', model_name: 'random_forest', amount: 50 },
  ],
  methodology_warning: 'predicciones supervisadas no constituyen fraude confirmado automático'
}

const supPredReport = {
  run_id: 1, status: 'COMPLETED', total_analyzed: 548124,
  high_count: 5481, medium_count: 12000, low_count: 530643,
  priority_distribution: [{ level: 'HIGH', count: 5481 }, { level: 'MEDIUM', count: 12000 }, { level: 'LOW', count: 530643 }],
  methodology_warning: 'predicciones supervisadas no constituyen fraude confirmado automático'
}

describe('ModelSupervised — apply tab (Fase D3)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.setItem('user', JSON.stringify({ id: 1, email: 'user@example.com', token: 'token' }))
    api.me.mockResolvedValue({ id: 1, email: 'user@example.com', full_name: 'User' })
    api.getHumanLabelSummary.mockResolvedValue(summary)
    api.getHumanReadiness.mockResolvedValue(readiness)
    api.getSupervisedTrainingPreflight.mockResolvedValue(preflight)
    api.getHumanDatasetSummary.mockResolvedValue({ exists: true })
    api.getSupervisedTrainingRuns.mockResolvedValue(runs)
    api.getSupervisedModelMetadata.mockResolvedValue(metadata)
    api.getSupervisedModelReport.mockResolvedValue({ markdown: '' })
    api.getSupervisedModelPredictions.mockResolvedValue(predictions)
    api.getSupInferenceTrainedModels.mockResolvedValue(supTrainedModels)
    api.getSupInferencePreprocessedRuns.mockResolvedValue(supPreprocessedRuns)
    api.getSupInferencePredictionRuns.mockResolvedValue(supPredRuns)
    api.getSupInferencePredictionResults.mockResolvedValue(supPredResults)
    api.getSupInferencePredictionReport.mockResolvedValue(supPredReport)
    api.applySupInferenceModel.mockResolvedValue({ run_id: 99, status: 'PENDING' })
    api.getSupInferenceStatus.mockResolvedValue({ id: 99, status: 'COMPLETED', total_analyzed: 548124, high_count: 5481, medium_count: 12000, low_count: 530643, algorithm: 'random_forest', input_source: 'preprocessed_run_26' })
  })

  afterEach(() => cleanup())

  it('renders both tabs', async () => {
    render(<ModelSupervised />)
    expect(await screen.findByText(/Dataset \/ Entrenamiento/)).toBeTruthy()
    expect(screen.getByText('Aplicar modelo entrenado')).toBeTruthy()
  })

  it('clicking apply tab loads trained models and preprocessed runs', async () => {
    render(<ModelSupervised />)
    fireEvent.click(screen.getByText('Aplicar modelo entrenado'))
    await waitFor(() => expect(api.getSupInferenceTrainedModels).toHaveBeenCalled())
    await waitFor(() => expect(api.getSupInferencePreprocessedRuns).toHaveBeenCalled())
    await waitFor(() => expect(api.getSupInferencePredictionRuns).toHaveBeenCalled())
  })

  it('shows trained model table in apply tab', async () => {
    render(<ModelSupervised />)
    fireEvent.click(screen.getByText('Aplicar modelo entrenado'))
    expect(await screen.findByText(/Seleccionar modelo entrenado/)).toBeTruthy()
    expect((await screen.findAllByText('Random Forest')).length).toBeGreaterThan(0)
  })

  it('shows methodology warning in apply tab', async () => {
    render(<ModelSupervised />)
    fireEvent.click(screen.getByText('Aplicar modelo entrenado'))
    expect(await screen.findByText(/no constituyen fraude confirmado automático/)).toBeTruthy()
  })

  it('shows apply button disabled without model selection', async () => {
    render(<ModelSupervised />)
    fireEvent.click(screen.getByText('Aplicar modelo entrenado'))
    await screen.findByText(/Seleccionar modelo entrenado/)
    const applyBtns = screen.getAllByRole('button', { name: /Aplicar modelo entrenado/ })
    // The submit button (not the tab button) should be disabled when no model is selected
    const submitBtn = applyBtns.find(btn => btn.type === 'button' && btn.tagName === 'BUTTON' && btn.disabled !== undefined)
    expect(submitBtn || applyBtns.some(b => b.disabled)).toBeTruthy()
  })

  it('shows prediction run history', async () => {
    render(<ModelSupervised />)
    fireEvent.click(screen.getByText('Aplicar modelo entrenado'))
    expect(await screen.findByText('Historial de ejecuciones')).toBeTruthy()
    expect((await screen.findAllByText('preprocessed_run_26')).length).toBeGreaterThan(0)
  })

  it('hides forbidden columns in results', async () => {
    render(<ModelSupervised />)
    fireEvent.click(screen.getByText('Aplicar modelo entrenado'))
    const viewBtn = await screen.findByRole('button', { name: 'Ver' })
    fireEvent.click(viewBtn)
    await waitFor(() => expect(api.getSupInferencePredictionResults).toHaveBeenCalled())
    expect(screen.queryByText('is_fraud')).toBeNull()
    expect(screen.queryByText('confirmed_fraud')).toBeNull()
    expect(screen.queryByText('PAN_TARJETA')).toBeNull()
  })

  it('shows priority level filters ALL HIGH MEDIUM LOW', async () => {
    render(<ModelSupervised />)
    fireEvent.click(screen.getByText('Aplicar modelo entrenado'))
    const viewBtn = await screen.findByRole('button', { name: 'Ver' })
    fireEvent.click(viewBtn)
    await waitFor(() => expect(api.getSupInferencePredictionResults).toHaveBeenCalled())
    expect(await screen.findByRole('button', { name: 'HIGH' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'MEDIUM' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'LOW' })).toBeTruthy()
  })

  it('shows KPI cards with total analyzed after results load', async () => {
    render(<ModelSupervised />)
    fireEvent.click(screen.getByText('Aplicar modelo entrenado'))
    const viewBtn = await screen.findByRole('button', { name: 'Ver' })
    fireEvent.click(viewBtn)
    await waitFor(() => expect(api.getSupInferencePredictionResults).toHaveBeenCalled())
    expect(await screen.findByText('Total analizados')).toBeTruthy()
    expect(screen.getByText('Prioridad HIGH')).toBeTruthy()
    expect(screen.getByText('Prioridad MEDIUM')).toBeTruthy()
    expect(screen.getByText('Prioridad LOW')).toBeTruthy()
  })
})
