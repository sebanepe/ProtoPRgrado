import React from 'react'
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'

vi.mock('../services/api')

import App from '../App'
import Models from '../pages/Models'
import * as api from '../services/api'

const runs = {
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
}

const metrics = {
  run_id: 'run_26',
  total_transactions: 548108,
  anomaly_count: 5481,
  anomaly_rate: 0.009999854,
  model_name: 'unsupervised_anomaly_detection',
  algorithm: 'isolation_forest',
  contamination: 0.01,
  anomalies_by_country: { BO: 3211, CL: 1052, AR: 832 },
  anomalies_by_pos_entry_mode: { '5': 1821, '10': 1600, '81': 2060 },
  anomalies_by_mcc: { '6011': 994, '7995': 950, UNKNOWN: 148 },
  anomalies_by_hour: { '0': 10, '1': 12, '2': 15 },
  top_customers_by_anomaly_count: [
    { customer_hash: 'cust_1', count: 9 },
    { customer_hash: 'cust_2', count: 7 },
  ],
}

const scores = {
  run_id: 'run_26',
  page: 1,
  page_size: 50,
  total_items: 2,
  total_pages: 1,
  items: [
    {
      anomaly_run_id: 'anomaly_run_26',
      source_run: 'preprocessed_run_26',
      transaction_id: 'tx_1',
      customer_hash: 'cust_1',
      transaction_datetime: '2026-05-29T10:00:00Z',
      amount: 250.42,
      country_code: 'BO',
      pos_entry_mode: '5',
      has_pinblock: 1,
      merchant_rubro_proxy: '6011',
      anomaly_model_name: 'isolation_forest',
      anomaly_score: 0.083118,
      anomaly_rank: 1,
      anomaly_flag: 1,
      anomaly_percentile: 99.99,
      created_at: '2026-05-30T00:33:18.268840+00:00',
    },
    {
      anomaly_run_id: 'anomaly_run_26',
      source_run: 'preprocessed_run_26',
      transaction_id: 'tx_2',
      customer_hash: 'cust_2',
      transaction_datetime: '2026-05-29T11:00:00Z',
      amount: 99.99,
      country_code: 'CL',
      pos_entry_mode: '10',
      has_pinblock: 0,
      merchant_rubro_proxy: '7995',
      anomaly_model_name: 'isolation_forest',
      anomaly_score: 0.062741,
      anomaly_rank: 2,
      anomaly_flag: 1,
      anomaly_percentile: 99.97,
      created_at: '2026-05-30T00:33:18.268840+00:00',
    },
  ],
}

const top = {
  run_id: 'run_26',
  limit: 20,
  count: 2,
  items: [
    {
      anomaly_run_id: 'anomaly_run_26',
      source_run: 'preprocessed_run_26',
      transaction_id: 'tx_1',
      customer_hash: 'cust_1',
      transaction_datetime: '2026-05-29T10:00:00Z',
      amount: 250.42,
      country_code: 'BO',
      pos_entry_mode: '5',
      has_pinblock: 1,
      merchant_rubro_proxy: '6011',
      anomaly_model_name: 'isolation_forest',
      anomaly_score: 0.083118,
      anomaly_rank: 1,
      anomaly_flag: 1,
      anomaly_percentile: 99.99,
      created_at: '2026-05-30T00:33:18.268840+00:00',
    },
    {
      anomaly_run_id: 'anomaly_run_26',
      source_run: 'preprocessed_run_26',
      transaction_id: 'tx_2',
      customer_hash: 'cust_2',
      transaction_datetime: '2026-05-29T11:00:00Z',
      amount: 99.99,
      country_code: 'CL',
      pos_entry_mode: '10',
      has_pinblock: 0,
      merchant_rubro_proxy: '7995',
      anomaly_model_name: 'isolation_forest',
      anomaly_score: 0.062741,
      anomaly_rank: 2,
      anomaly_flag: 1,
      anomaly_percentile: 99.97,
      created_at: '2026-05-30T00:33:18.268840+00:00',
    },
  ],
}

const report = {
  run_id: 'run_26',
  report: `# Anomaly Detection Report

- source_run: preprocessed_run_26
- source_run_token: 26
- total_transactions: 548108
- model: isolation_forest
- contamination: 0.01
- anomaly_count: 5481
- anomaly_rate: 0.010000

## Warnings
Las anomalías detectadas no representan fraude confirmado.
No se generó is_fraud. No se generó confirmed_fraud. No se usaron reglas como etiquetas.
`,
}

const metadata = {
  run_id: 'run_26',
  metadata: {
    source_run: 'preprocessed_run_26',
    source_run_token: 26,
    model_name: 'isolation_forest',
    model_type: 'unsupervised_anomaly_detection',
    algorithm: 'isolation_forest',
    contamination: 0.01,
    total_rows: 548108,
    anomaly_count: 5481,
    anomaly_rate: 0.009999854,
    numeric_features: ['amount', 'amount_log', 'hour_of_day'],
    categorical_features: ['country_code', 'pos_entry_mode', 'merchant_rubro_proxy'],
    model_input_columns: ['amount', 'amount_log', 'hour_of_day', 'country_code', 'pos_entry_mode', 'merchant_rubro_proxy'],
    excluded_columns: ['is_fraud', 'confirmed_fraud', 'target_is_fraud'],
    model_path: 'data/models/isolation_forest_run_26.pkl',
    score_file: 'data/processed/anomaly_scores_run_26.csv',
    feature_file: 'data/processed/unsupervised_feature_set_run_26.csv',
    report_file: 'data/processed/anomaly_report_run_26.md',
  },
}

const autoencoderMetrics = {
  source_run: 'preprocessed_run_26',
  algorithm: 'autoencoder_pytorch',
  total_records: 100,
  anomaly_count: 1,
  anomaly_rate: 0.01,
  threshold: 0.123456,
  contamination: 0.01,
  created_at: '2026-05-31T00:00:00Z',
}

const autoencoderScores = {
  source_run: 'preprocessed_run_26',
  page: 1,
  page_size: 50,
  total_items: 1,
  total_pages: 1,
  items: [
    {
      source_run: 'preprocessed_run_26',
      transaction_id: 'tx_ae_1',
      customer_hash: 'cust_ae_1',
      transaction_datetime: '2026-05-31T10:00:00Z',
      amount: 500,
      country_code: 'BO',
      merchant_rubro_proxy: '6011',
      reconstruction_error: 0.9,
      autoencoder_anomaly_score: 1,
      autoencoder_anomaly_flag: 1,
      anomaly_rank: 1,
    },
  ],
}

const autoencoderReport = {
  source_run: 'preprocessed_run_26',
  report: '# Autoencoder PyTorch Report\nLos resultados no representan fraude confirmado.',
}

const autoencoderMetadata = {
  source_run: 'preprocessed_run_26',
  metadata: {
    source_run: 'preprocessed_run_26',
    model_family: 'UNSUPERVISED',
    algorithm: 'autoencoder_pytorch',
    framework: 'pytorch',
    epochs: 30,
    batch_size: 512,
    latent_dim: 16,
    learning_rate: 0.001,
    contamination: 0.01,
    threshold: 0.123456,
    total_records: 100,
    anomaly_count: 1,
    anomaly_rate: 0.01,
    feature_columns: ['amount_log', 'hour_of_day'],
    scores_file: 'autoencoder_scores_run_26.csv',
    report_file: 'autoencoder_report_run_26.md',
    model_file: 'autoencoder_model_run_26.pt',
  },
}

const trainedModels = [
  {
    id: 1,
    algorithm: 'isolation_forest',
    source_run: 'preprocessed_run_26',
    status: 'AVAILABLE',
    is_active: false,
    created_at: '2026-05-30T00:00:00Z',
    anomaly_rate: 0.01,
    contamination: 0.01,
    total_records: 548108,
  },
  {
    id: 2,
    algorithm: 'autoencoder_pytorch',
    source_run: 'preprocessed_run_26',
    status: 'AVAILABLE',
    is_active: true,
    created_at: '2026-05-31T00:00:00Z',
    anomaly_rate: 0.01,
    contamination: 0.01,
    total_records: 548108,
  },
]

const preprocessedRuns = [
  { id: 26, output_file_path: '/data/processed/preprocessed_run_26.csv', total_records: 548108, finished_at: '2026-05-29T00:00:00Z', status: 'COMPLETED' },
]

const predictionRunsList = [
  { id: 1, algorithm: 'isolation_forest', model_source_run: 'preprocessed_run_26', input_type: 'preprocessed_run', input_source: 'preprocessed_run_26', total_analyzed: 100, anomaly_count: 2, anomaly_rate: 0.02, status: 'COMPLETED', created_at: '2026-06-01T00:00:00Z' },
]

const predResultsPayload = {
  run_id: 1,
  status: 'COMPLETED',
  total: 2,
  page: 1,
  page_size: 50,
  rows: [
    { transaction_id: 'tx1', customer_hash: 'cust1', anomaly_score: 0.9, anomaly_flag: 1, anomaly_rank: 1 },
    { transaction_id: 'tx2', customer_hash: 'cust2', anomaly_score: 0.1, anomaly_flag: 0, anomaly_rank: 2 },
  ],
  methodology_warning: 'Las anomalías detectadas por modelos no supervisados representan comportamientos atípicos y no constituyen fraude confirmado.',
}

const predReportPayload = {
  run_id: 1,
  algorithm: 'isolation_forest',
  model_source_run: 'preprocessed_run_26',
  input_type: 'preprocessed_run',
  input_source: 'preprocessed_run_26',
  total_analyzed: 100,
  anomaly_count: 2,
  anomaly_rate: 0.02,
  status: 'COMPLETED',
  score_distribution: [{ bucket: '0.0–0.5', count: 90 }, { bucket: '0.5–1.0', count: 10 }],
  methodology_warning: 'Las anomalías detectadas por modelos no supervisados representan comportamientos atípicos y no constituyen fraude confirmado.',
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.setItem('user', JSON.stringify({ id: 1, email: 'user@example.com', token: 'token' }))
  Object.defineProperty(navigator, 'clipboard', {
    value: {
      writeText: vi.fn().mockResolvedValue(undefined),
    },
    configurable: true,
  })
  api.me.mockResolvedValue({ id: 1, email: 'user@example.com', full_name: 'User' })
  api.getAnomalyRuns.mockResolvedValue(runs)
  api.getAnomalyMetrics.mockResolvedValue(metrics)
  api.getAnomalyScores.mockResolvedValue(scores)
  api.getTopAnomalies.mockResolvedValue(top)
  api.getAnomalyReport.mockResolvedValue(report)
  api.getAnomalyModelMetadata.mockResolvedValue(metadata)
  api.trainAnomalyModel.mockResolvedValue({ status: 'COMPLETED', anomaly_run_id: 'run_26', source_run: 'preprocessed_run_26' })
  api.getAutoencoderMetrics.mockResolvedValue(autoencoderMetrics)
  api.getAutoencoderScores.mockResolvedValue(autoencoderScores)
  api.getAutoencoderReport.mockResolvedValue(autoencoderReport)
  api.getAutoencoderModelMetadata.mockResolvedValue(autoencoderMetadata)
  api.trainAutoencoderAnomaly.mockResolvedValue({ status: 'COMPLETED', algorithm: 'autoencoder_pytorch', source_run: 'preprocessed_run_26' })
  // Apply tab mocks
  api.getUnsupervisedTrainedModels.mockResolvedValue(trainedModels)
  api.getUnsupervisedPreprocessedRuns.mockResolvedValue(preprocessedRuns)
  api.applyUnsupervisedModel.mockResolvedValue({ id: 1, status: 'COMPLETED', total_analyzed: 100, anomaly_count: 2, anomaly_rate: 0.02, algorithm: 'isolation_forest', model_source_run: 'preprocessed_run_26', methodology_warning: 'Las anomalías...' })
  api.getUnsupervisedPredictionRuns.mockResolvedValue(predictionRunsList)
  api.getUnsupervisedPredictionResults.mockResolvedValue(predResultsPayload)
  api.getUnsupervisedPredictionReport.mockResolvedValue(predReportPayload)
})

afterEach(() => {
  cleanup()
})

describe('Models page', () => {
  it('renders unsupervised screen with Isolation Forest selected by default', async () => {
    render(<Models />)

    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())
    expect(screen.getByText('Modelos No Supervisados')).toBeTruthy()
    expect(screen.getByText('Detectan comportamientos atípicos sin usar etiquetas humanas.')).toBeTruthy()
    expect(screen.getByText('Estos modelos analizan patrones transaccionales y marcan operaciones que se alejan del comportamiento general. Una anomalía no significa fraude confirmado.')).toBeTruthy()
    expect(screen.getByLabelText('Modelo no supervisado').value).toBe('isolation_forest')
    expect(screen.getByText('Busca transacciones que se aíslan fácilmente del resto.')).toBeTruthy()
    expect(screen.getByText('Aprende a reconstruir transacciones normales.')).toBeTruthy()
    expect(screen.getByText('Porcentaje aproximado de transacciones que el modelo marcará como anómalas.')).toBeTruthy()
    expect(screen.getByLabelText('n_estimators')).toBeTruthy()
    expect(screen.getByLabelText('max_categories')).toBeTruthy()
    expect(screen.queryByLabelText('epochs')).toBeNull()
    expect(screen.queryByLabelText('batch_size')).toBeNull()
    expect(screen.queryByLabelText('latent_dim')).toBeNull()
    expect(screen.queryByLabelText('learning_rate')).toBeNull()
  })

  it('shows Autoencoder parameters and calls the Autoencoder endpoint', async () => {
    render(<Models />)

    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())
    fireEvent.change(screen.getByLabelText('Modelo no supervisado'), { target: { value: 'autoencoder_pytorch' } })

    expect(screen.getByLabelText('epochs')).toBeTruthy()
    expect(screen.getByLabelText('batch_size')).toBeTruthy()
    expect(screen.getByLabelText('latent_dim')).toBeTruthy()
    expect(screen.getByLabelText('learning_rate')).toBeTruthy()
    expect(screen.queryByLabelText('n_estimators')).toBeNull()
    expect(screen.queryByLabelText('max_categories')).toBeNull()

    await waitFor(() => expect(api.getAutoencoderMetrics).toHaveBeenCalledWith('preprocessed_run_26'))
    await waitFor(() => expect(api.getAutoencoderScores).toHaveBeenCalledWith('preprocessed_run_26', expect.objectContaining({ page: 1, page_size: 50, anomaly_flag: 1 })))

    fireEvent.click(screen.getByRole('button', { name: 'Ejecutar entrenamiento' }))
    await waitFor(() => expect(api.trainAutoencoderAnomaly).toHaveBeenCalledWith(expect.objectContaining({
      source_run: 'preprocessed_run_26',
      contamination: 0.01,
      epochs: 30,
      batch_size: 512,
      latent_dim: 16,
      learning_rate: 0.001,
      sample_size: null,
    })))
  })

  it('shows controlled PyTorch dependency message for Autoencoder', async () => {
    api.trainAutoencoderAnomaly.mockResolvedValueOnce({ status: 'AUTOENCODER_DEPENDENCY_NOT_AVAILABLE', algorithm: 'autoencoder_pytorch' })
    render(<Models />)

    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())
    fireEvent.change(screen.getByLabelText('Modelo no supervisado'), { target: { value: 'autoencoder_pytorch' } })
    fireEvent.click(screen.getByRole('button', { name: 'Ejecutar entrenamiento' }))

    await waitFor(() => expect(screen.getByText('PyTorch no está disponible. Puede seguir usando Isolation Forest o instalar PyTorch para habilitar Autoencoder.')).toBeTruthy())
  })

  it('shows Autoencoder metrics and scores without fraud labels', async () => {
    render(<Models />)

    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())
    fireEvent.change(screen.getByLabelText('Modelo no supervisado'), { target: { value: 'autoencoder_pytorch' } })

    await waitFor(() => expect(screen.getAllByText('autoencoder_pytorch').length).toBeGreaterThan(0))
    expect(screen.getByText('Error de reconstrucción')).toBeTruthy()
    expect(screen.getByText('Score Autoencoder')).toBeTruthy()
    expect(screen.getByText('Marcada como anomalía')).toBeTruthy()
    expect(screen.queryByText('is_fraud')).toBeNull()
    expect(screen.queryByText('confirmed_fraud')).toBeNull()
  })

  it('loads anomaly runs, metrics, paginated scores, report and metadata', async () => {
    render(<Models />)

    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())
    await waitFor(() => expect(api.getAnomalyMetrics).toHaveBeenCalledWith('run_26'))
    await waitFor(() => expect(api.getAnomalyScores).toHaveBeenCalledWith('run_26', expect.objectContaining({ page: 1, page_size: 50, anomaly_flag: 1 })))
    await waitFor(() => expect(api.getTopAnomalies).toHaveBeenCalledWith('run_26', 20))
    await waitFor(() => expect(api.getAnomalyReport).toHaveBeenCalledWith('run_26'))
    await waitFor(() => expect(api.getAnomalyModelMetadata).toHaveBeenCalledWith('run_26'))

    expect(screen.getAllByText('Modelos No Supervisados').length).toBeGreaterThan(0)
    expect(screen.getByText('Las anomalías detectadas por los modelos no supervisados no representan fraude confirmado. Son señales de comportamiento atípico que requieren revisión.')).toBeTruthy()
    expect(screen.getByText('Resumen visual del entrenamiento')).toBeTruthy()
    expect(screen.getByText('Anomalías detectadas')).toBeTruthy()
    expect(screen.getByText('Distribución de resultados')).toBeTruthy()
    expect(screen.getByText('Top anomalías por score')).toBeTruthy()
    expect(screen.getByText('Distribución de scores')).toBeTruthy()
    expect(screen.getAllByText('ID Transacción').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Cliente anonimizado').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Score de anomalía').length).toBeGreaterThan(0)
    expect(screen.getByText('Cómo interpretar los resultados')).toBeTruthy()
    expect(screen.getByText('Metadata del modelo')).toBeTruthy()
    expect(screen.getAllByText((_, element) => (element?.textContent || '').replace(/\D/g, '') === '5481').length).toBeGreaterThan(0)
    expect(screen.getAllByText((_, element) => (element?.textContent || '').includes('Las anomalías detectadas no representan fraude confirmado.')).length).toBeGreaterThan(0)
    expect(screen.getAllByText((_, element) => (element?.textContent || '').includes('source_run: preprocessed_run_26')).length).toBeGreaterThan(0)
    expect(screen.getAllByText((_, element) => (element?.textContent || '').includes('source_run_token: 26')).length).toBeGreaterThan(0)
    expect(screen.getByText('model_name')).toBeTruthy()
    expect(screen.getAllByText('isolation_forest').length).toBeGreaterThan(0)
  })

  it('applies filters, opens detail, copies report and trains a model without replacing previous results', async () => {
    let resolveTrain
    api.trainAnomalyModel.mockImplementation(() => new Promise((resolve) => { resolveTrain = resolve }))

    render(<Models />)

    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())

    fireEvent.change(screen.getByLabelText('País'), { target: { value: 'BO' } })
    fireEvent.change(screen.getByLabelText('MCC / Rubro'), { target: { value: '6011' } })
    fireEvent.click(screen.getAllByRole('button', { name: 'Aplicar filtros' })[0])

    await waitFor(() => {
      const lastCall = api.getAnomalyScores.mock.calls.at(-1)
      expect(lastCall[0]).toBe('run_26')
      expect(lastCall[1]).toEqual(expect.objectContaining({ page: 1, page_size: 50, anomaly_flag: 1, country_code: 'BO', merchant_rubro_proxy: '6011' }))
    })

    fireEvent.click(screen.getAllByRole('button', { name: 'Ver detalle' })[0])
    expect(screen.getByText('Detalle de anomalía')).toBeTruthy()
    expect(screen.getByText('Esta transacción fue marcada como anomalía estadística. No constituye fraude confirmado.')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Copiar reporte' }))
    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalled())

    fireEvent.change(screen.getByLabelText('source_run'), { target: { value: 'preprocessed_run_26' } })
    fireEvent.click(screen.getByRole('button', { name: 'Ejecutar entrenamiento' }))

    expect(screen.getByRole('button', { name: 'Ejecutando entrenamiento...' }).disabled).toBe(true)
    resolveTrain({ status: 'COMPLETED', anomaly_run_id: 'run_26', source_run: 'preprocessed_run_26' })

    await waitFor(() => expect(api.getAnomalyRuns.mock.calls.length).toBeGreaterThan(1))
    expect(api.trainAnomalyModel).toHaveBeenCalledWith(expect.objectContaining({
      source_run: 'preprocessed_run_26',
      contamination: '0.01',
      n_estimators: '200',
      max_categories: '50',
      sample_size: '',
    }))
  })

  it('keeps forbidden sensitive fields out of the unsupervised UI', async () => {
    render(<Models />)

    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())

    expect(screen.queryByText('is_fraud')).toBeNull()
    expect(screen.queryByText('confirmed_fraud')).toBeNull()
    expect(screen.queryByText('PAN_TARJETA')).toBeNull()
    expect(screen.queryByText('TARJETA')).toBeNull()
  })
})

describe('Models route and sidebar', () => {
  it('renders /models/unsupervised with the sidebar entry', async () => {
    render(
      <MemoryRouter initialEntries={['/models/unsupervised']}>
        <App />
      </MemoryRouter>
    )

    await waitFor(() => expect(screen.getAllByText('No Supervisados').length).toBeGreaterThan(0))
    await waitFor(() => expect(screen.getAllByText('No Supervisados').length).toBeGreaterThan(0))
  })
})

describe('Models page — Aplicar modelo entrenado tab', () => {
  it('renders both tabs in the UI', async () => {
    render(<Models />)
    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())

    expect(screen.getByRole('button', { name: 'Entrenamiento / Resultados actuales' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Aplicar modelo entrenado' })).toBeTruthy()
  })

  it('clicking "Aplicar modelo entrenado" tab shows apply section', async () => {
    render(<Models />)
    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: 'Aplicar modelo entrenado' }))

    await waitFor(() => expect(api.getUnsupervisedTrainedModels).toHaveBeenCalled())
    await waitFor(() => expect(api.getUnsupervisedPreprocessedRuns).toHaveBeenCalled())
    await waitFor(() => expect(api.getUnsupervisedPredictionRuns).toHaveBeenCalled())
  })

  it('apply tab shows methodology warning about no confirmed fraud', async () => {
    render(<Models />)
    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: 'Aplicar modelo entrenado' }))

    await waitFor(() => expect(screen.getByText(/Advertencia metodológica/i)).toBeTruthy())
    expect(screen.getByText(/no constituyen fraude confirmado/i)).toBeTruthy()
  })

  it('apply tab lists trained models from API', async () => {
    render(<Models />)
    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: 'Aplicar modelo entrenado' }))

    await waitFor(() => expect(api.getUnsupervisedTrainedModels).toHaveBeenCalled())
    await waitFor(() => expect(screen.getAllByText('isolation_forest').length).toBeGreaterThan(0))
    await waitFor(() => expect(screen.getAllByText('autoencoder_pytorch').length).toBeGreaterThan(0))
  })

  it('apply tab "Aplicar modelo" button is disabled when no model selected', async () => {
    render(<Models />)
    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: 'Aplicar modelo entrenado' }))

    await waitFor(() => expect(screen.getByRole('button', { name: 'Aplicar modelo' })).toBeTruthy())
    expect(screen.getByRole('button', { name: 'Aplicar modelo' }).disabled).toBe(true)
  })

  it('apply tab shows prediction runs and results after loading', async () => {
    render(<Models />)
    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: 'Aplicar modelo entrenado' }))

    await waitFor(() => expect(api.getUnsupervisedPredictionRuns).toHaveBeenCalled())
    await waitFor(() => expect(api.getUnsupervisedPredictionResults).toHaveBeenCalled())
    await waitFor(() => expect(api.getUnsupervisedPredictionReport).toHaveBeenCalled())
  })

  it('apply tab results do not show is_fraud or confirmed_fraud', async () => {
    render(<Models />)
    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: 'Aplicar modelo entrenado' }))

    await waitFor(() => expect(api.getUnsupervisedTrainedModels).toHaveBeenCalled())

    expect(screen.queryByText('is_fraud')).toBeNull()
    expect(screen.queryByText('confirmed_fraud')).toBeNull()
    expect(screen.queryByText('PAN_TARJETA')).toBeNull()
  })

  it('switching back to training tab hides apply section', async () => {
    render(<Models />)
    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: 'Aplicar modelo entrenado' }))
    await waitFor(() => expect(screen.getByText(/Advertencia metodológica/i)).toBeTruthy())

    fireEvent.click(screen.getByRole('button', { name: 'Entrenamiento / Resultados actuales' }))

    await waitFor(() => {
      expect(screen.queryByText(/Advertencia metodológica/i)).toBeNull()
    })
    expect(screen.getByText('Entrenar modelo no supervisado')).toBeTruthy()
  })

  it('apply tab shows KPI cards with prediction report data', async () => {
    render(<Models />)
    await waitFor(() => expect(api.getAnomalyRuns).toHaveBeenCalled())

    fireEvent.click(screen.getByRole('button', { name: 'Aplicar modelo entrenado' }))

    await waitFor(() => {
      expect(api.getUnsupervisedPredictionReport).toHaveBeenCalled()
      expect(screen.getByText('Registros analizados')).toBeTruthy()
    })
    expect(screen.getByText('Anomalías detectadas')).toBeTruthy()
    expect(screen.getByText('Porcentaje de anomalías')).toBeTruthy()
    expect(screen.getByText('Modelo aplicado')).toBeTruthy()
  })
})
