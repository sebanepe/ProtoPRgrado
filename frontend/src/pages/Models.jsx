import React, { useEffect, useMemo, useState } from 'react'
import KPICard from '../components/KPICard'
import Table from '../components/Table'
import {
  getAutoencoderMetrics,
  getAutoencoderModelMetadata,
  getAutoencoderReport,
  getAutoencoderScores,
  getAnomalyMetrics,
  getAnomalyModelMetadata,
  getAnomalyReport,
  getAnomalyRuns,
  getAnomalyScores,
  getTopAnomalies,
  trainAutoencoderAnomaly,
  trainAnomalyModel,
} from '../services/api'

const DEFAULT_TRAINING_FORM = {
  source_run: 'preprocessed_run_26',
  contamination: '0.01',
  n_estimators: '200',
  max_categories: '50',
  epochs: '30',
  batch_size: '512',
  latent_dim: '16',
  learning_rate: '0.001',
  sample_size: '',
}

const DEFAULT_FILTERS = {
  anomaly_flag: true,
  country_code: '',
  pos_entry_mode: '',
  merchant_rubro_proxy: '',
  customer_hash: '',
  min_score: '',
  max_score: '',
}

const FORBIDDEN_DISPLAY_TOKENS = ['is_fraud', 'confirmed_fraud', 'PAN_TARJETA', 'TARJETA', 'pan_card', 'raw_card']

function extractErrorMessage(error, fallback) {
  return error?.response?.data?.detail || error?.message || fallback
}

function compactObject(values) {
  return Object.fromEntries(Object.entries(values).filter(([, value]) => value !== '' && value !== null && value !== undefined))
}

function sanitizeDisplayValue(value) {
  if (Array.isArray(value)) {
    return value.filter((item) => !FORBIDDEN_DISPLAY_TOKENS.includes(String(item)))
  }
  if (typeof value !== 'string') return value
  return FORBIDDEN_DISPLAY_TOKENS.reduce((text, token) => text.replaceAll(token, '[columna protegida]'), value)
}

function formatNumber(value, decimals = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A'
  return Number(value).toLocaleString('es-ES', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A'
  return `${(Number(value) * 100).toFixed(2)}%`
}

function formatMaybeDate(value) {
  if (!value) return 'N/A'
  if (typeof value === 'number') {
    const millis = value > 1e12 ? value : value * 1000
    return new Date(millis).toLocaleString('es-ES')
  }
  const parsed = new Date(value)
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toLocaleString('es-ES')
  }
  return String(value)
}

function dependencyMessage(status) {
  if (status === 'AUTOENCODER_DEPENDENCY_NOT_AVAILABLE') {
    return 'PyTorch no está disponible en el entorno actual. Instale o habilite PyTorch para entrenar el Autoencoder.'
  }
  return ''
}

function topEntry(source) {
  if (!source) return null
  const entries = Object.entries(source)
  if (!entries.length) return null
  return entries.sort((a, b) => Number(b[1]) - Number(a[1]))[0]
}

function renderDistributionItems(source, limit = 6) {
  return Object.entries(source || {})
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, limit)
}

function InfoList({ title, items, emptyText = 'Sin datos disponibles.' }) {
  return (
    <div className="card detail-section">
      <h4>{title}</h4>
      {items && items.length > 0 ? (
        <div className="table-scroll">
          <table className="table small">
            <tbody>
              {items.map(([label, value]) => (
                <tr key={label}>
                  <td style={{ width: '55%', fontWeight: 700 }}>{label}</td>
                  <td>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty-state">{emptyText}</div>
      )}
    </div>
  )
}

function KeyValueGrid({ title, items }) {
  return (
    <div className="card detail-section">
      <h4>{title}</h4>
      <div className="metadata-grid">
        {items.map(({ label, value, highlight }) => (
          <div className={`metadata-item${highlight ? ' metadata-item-highlight' : ''}`} key={label}>
            <div className="metric-label">{label}</div>
            <div className="metadata-value">{value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Models() {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('isolation_forest')
  const [runs, setRuns] = useState([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [runsLoading, setRunsLoading] = useState(true)
  const [runsError, setRunsError] = useState('')

  const [metrics, setMetrics] = useState(null)
  const [metricsLoading, setMetricsLoading] = useState(false)
  const [metricsError, setMetricsError] = useState('')

  const [scoresPayload, setScoresPayload] = useState({ run_id: '', page: 1, page_size: 50, total_items: 0, total_pages: 0, items: [] })
  const [scoresLoading, setScoresLoading] = useState(false)
  const [scoresError, setScoresError] = useState('')

  const [topAnomalies, setTopAnomalies] = useState([])
  const [topLoading, setTopLoading] = useState(false)
  const [topError, setTopError] = useState('')

  const [report, setReport] = useState('')
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState('')

  const [metadata, setMetadata] = useState(null)
  const [metadataLoading, setMetadataLoading] = useState(false)
  const [metadataError, setMetadataError] = useState('')

  const [trainingForm, setTrainingForm] = useState(DEFAULT_TRAINING_FORM)
  const [trainingRunning, setTrainingRunning] = useState(false)
  const [trainingMessage, setTrainingMessage] = useState('')
  const [trainingError, setTrainingError] = useState('')

  const [draftFilters, setDraftFilters] = useState(DEFAULT_FILTERS)
  const [appliedFilters, setAppliedFilters] = useState(DEFAULT_FILTERS)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)
  const [selectedAnomaly, setSelectedAnomaly] = useState(null)
  const [copyState, setCopyState] = useState('')

  const selectedRun = useMemo(() => runs.find((run) => run.anomaly_run_id === selectedRunId) || null, [runs, selectedRunId])
  const selectedSourceRun = trainingForm.source_run.trim() || selectedRun?.source_run || 'preprocessed_run_26'
  const isAutoencoder = selectedAlgorithm === 'autoencoder_pytorch'

  const normalizedScoreFilters = useMemo(() => {
    const params = {}
    if (appliedFilters.anomaly_flag) params.anomaly_flag = 1
    if (isAutoencoder) return params
    if (appliedFilters.country_code.trim()) params.country_code = appliedFilters.country_code.trim()
    if (appliedFilters.pos_entry_mode.trim()) params.pos_entry_mode = appliedFilters.pos_entry_mode.trim()
    if (appliedFilters.merchant_rubro_proxy.trim()) params.merchant_rubro_proxy = appliedFilters.merchant_rubro_proxy.trim()
    if (appliedFilters.customer_hash.trim()) params.customer_hash = appliedFilters.customer_hash.trim()
    if (appliedFilters.min_score.trim() !== '') params.min_score = Number(appliedFilters.min_score)
    if (appliedFilters.max_score.trim() !== '') params.max_score = Number(appliedFilters.max_score)
    return params
  }, [appliedFilters, isAutoencoder])

  async function loadRuns(preferredRunId = '') {
    setRunsLoading(true)
    setRunsError('')
    try {
      const response = await getAnomalyRuns()
      const items = Array.isArray(response) ? response : response?.runs || []
      setRuns(items)
      const nextRunId = preferredRunId && items.some((run) => run.anomaly_run_id === preferredRunId)
        ? preferredRunId
        : items[0]?.anomaly_run_id || ''
      setSelectedRunId(nextRunId)
      return nextRunId
    } catch (error) {
      setRuns([])
      setSelectedRunId('')
      setRunsError(extractErrorMessage(error, 'No fue posible cargar las ejecuciones de anomalías.'))
      return ''
    } finally {
      setRunsLoading(false)
    }
  }

  async function loadMetrics(runId, algorithm = selectedAlgorithm) {
    if (!runId) return
    setMetricsLoading(true)
    setMetricsError('')
    try {
      const response = algorithm === 'autoencoder_pytorch'
        ? await getAutoencoderMetrics(runId)
        : await getAnomalyMetrics(runId)
      setMetrics(response)
    } catch (error) {
      setMetrics(null)
      setMetricsError(extractErrorMessage(error, 'No fue posible cargar las métricas del modelo.'))
    } finally {
      setMetricsLoading(false)
    }
  }

  async function loadScores(runId, currentPage = page, currentPageSize = pageSize, filters = normalizedScoreFilters, algorithm = selectedAlgorithm) {
    if (!runId) return
    setScoresLoading(true)
    setScoresError('')
    try {
      const response = algorithm === 'autoencoder_pytorch'
        ? await getAutoencoderScores(runId, { page: currentPage, page_size: currentPageSize, ...filters })
        : await getAnomalyScores(runId, { page: currentPage, page_size: currentPageSize, ...filters })
      setScoresPayload(response)
    } catch (error) {
      setScoresPayload({ run_id: runId, page: currentPage, page_size: currentPageSize, total_items: 0, total_pages: 0, items: [] })
      setScoresError(extractErrorMessage(error, 'No fue posible cargar las anomalías.'))
    } finally {
      setScoresLoading(false)
    }
  }

  async function loadTop(runId, algorithm = selectedAlgorithm) {
    if (!runId) return
    if (algorithm === 'autoencoder_pytorch') {
      setTopAnomalies([])
      setTopError('')
      return
    }
    setTopLoading(true)
    setTopError('')
    try {
      const response = await getTopAnomalies(runId, 20)
      setTopAnomalies(response?.items || [])
    } catch (error) {
      setTopAnomalies([])
      setTopError(extractErrorMessage(error, 'No fue posible cargar el top de anomalías.'))
    } finally {
      setTopLoading(false)
    }
  }

  async function loadReport(runId, algorithm = selectedAlgorithm) {
    if (!runId) return
    setReportLoading(true)
    setReportError('')
    try {
      const response = algorithm === 'autoencoder_pytorch'
        ? await getAutoencoderReport(runId)
        : await getAnomalyReport(runId)
      setReport(response?.report || '')
    } catch (error) {
      setReport('')
      setReportError(extractErrorMessage(error, 'No fue posible cargar el reporte del modelo.'))
    } finally {
      setReportLoading(false)
    }
  }

  async function loadMetadata(runId, algorithm = selectedAlgorithm) {
    if (!runId) return
    setMetadataLoading(true)
    setMetadataError('')
    try {
      const response = algorithm === 'autoencoder_pytorch'
        ? await getAutoencoderModelMetadata(runId)
        : await getAnomalyModelMetadata(runId)
      setMetadata(response?.metadata || null)
    } catch (error) {
      setMetadata(null)
      setMetadataError(extractErrorMessage(error, 'No fue posible cargar la metadata del modelo.'))
    } finally {
      setMetadataLoading(false)
    }
  }

  useEffect(() => {
    loadRuns()
  }, [])

  useEffect(() => {
    const lookupRun = isAutoencoder ? selectedSourceRun : selectedRunId
    if (!lookupRun) return
    loadMetrics(lookupRun, selectedAlgorithm)
    loadTop(lookupRun, selectedAlgorithm)
    loadReport(lookupRun, selectedAlgorithm)
    loadMetadata(lookupRun, selectedAlgorithm)
  }, [selectedRunId, selectedAlgorithm, selectedSourceRun])

  useEffect(() => {
    const lookupRun = isAutoencoder ? selectedSourceRun : selectedRunId
    if (!lookupRun) return
    loadScores(lookupRun, page, pageSize, normalizedScoreFilters, selectedAlgorithm)
  }, [selectedRunId, selectedAlgorithm, selectedSourceRun, page, pageSize, normalizedScoreFilters])

  const anomalyColumns = [
    { key: 'anomaly_rank', title: 'anomaly_rank' },
    { key: 'transaction_id', title: 'transaction_id' },
    { key: 'customer_hash', title: 'customer_hash' },
    { key: 'transaction_datetime', title: 'transaction_datetime' },
    { key: 'amount', title: 'amount' },
    { key: 'country_code', title: 'country_code' },
    { key: 'pos_entry_mode', title: 'pos_entry_mode' },
    { key: 'has_pinblock', title: 'has_pinblock' },
    { key: 'merchant_rubro_proxy', title: 'merchant_rubro_proxy' },
    { key: 'anomaly_score', title: 'anomaly_score' },
    { key: 'anomaly_percentile', title: 'anomaly_percentile' },
    { key: 'anomaly_flag', title: 'anomaly_flag' },
    { key: 'anomaly_model_name', title: 'anomaly_model_name' },
    { key: '__actions', title: 'Acción' },
  ]

  const autoencoderColumns = [
    { key: 'anomaly_rank', title: 'anomaly_rank' },
    { key: 'transaction_id', title: 'transaction_id' },
    { key: 'customer_hash', title: 'customer_hash' },
    { key: 'transaction_datetime', title: 'transaction_datetime' },
    { key: 'amount', title: 'amount' },
    { key: 'country_code', title: 'country_code' },
    { key: 'pos_entry_mode', title: 'pos_entry_mode' },
    { key: 'has_pinblock', title: 'has_pinblock' },
    { key: 'merchant_rubro_proxy', title: 'merchant_rubro_proxy' },
    { key: 'anomaly_score', title: 'reconstruction_error' },
    { key: 'anomaly_percentile', title: 'autoencoder_anomaly_score' },
    { key: 'anomaly_flag', title: 'autoencoder_anomaly_flag' },
    { key: 'anomaly_model_name', title: 'algorithm' },
    { key: '__actions', title: 'Accion' },
  ]

  const activeScoreColumns = isAutoencoder ? autoencoderColumns : anomalyColumns

  const topColumns = [
    { key: 'anomaly_rank', title: 'Rank' },
    { key: 'transaction_id', title: 'transaction_id' },
    { key: 'customer_hash', title: 'customer_hash' },
    { key: 'amount', title: 'amount' },
    { key: 'country_code', title: 'country_code' },
    { key: 'pos_entry_mode', title: 'pos_entry_mode' },
    { key: 'merchant_rubro_proxy', title: 'merchant_rubro_proxy' },
    { key: 'anomaly_score', title: 'anomaly_score' },
    { key: 'anomaly_percentile', title: 'anomaly_percentile' },
  ]

  const tableRows = useMemo(() => scoresPayload.items.map((item) => ({
    ...item,
    amount: formatNumber(item.amount, 2),
    anomaly_score: item.anomaly_score !== null && item.anomaly_score !== undefined ? Number(item.anomaly_score).toFixed(6) : 'N/A',
    anomaly_percentile: item.anomaly_percentile !== null && item.anomaly_percentile !== undefined ? Number(item.anomaly_percentile).toFixed(2) : 'N/A',
    anomaly_flag: Number(item.anomaly_flag) === 1 ? '1' : '0',
    reconstruction_error: item.reconstruction_error !== null && item.reconstruction_error !== undefined ? Number(item.reconstruction_error).toFixed(6) : 'N/A',
    autoencoder_anomaly_score: item.autoencoder_anomaly_score !== null && item.autoencoder_anomaly_score !== undefined ? Number(item.autoencoder_anomaly_score).toFixed(6) : 'N/A',
    autoencoder_anomaly_flag: Number(item.autoencoder_anomaly_flag) === 1 ? '1' : '0',
    ...(isAutoencoder ? {
      pos_entry_mode: item.pos_entry_mode || 'N/A',
      has_pinblock: item.has_pinblock ?? 'N/A',
      anomaly_score: item.reconstruction_error !== null && item.reconstruction_error !== undefined ? Number(item.reconstruction_error).toFixed(6) : 'N/A',
      anomaly_percentile: item.autoencoder_anomaly_score !== null && item.autoencoder_anomaly_score !== undefined ? Number(item.autoencoder_anomaly_score).toFixed(6) : 'N/A',
      anomaly_flag: Number(item.autoencoder_anomaly_flag) === 1 ? '1' : '0',
      anomaly_model_name: 'autoencoder_pytorch',
    } : {}),
  })), [scoresPayload.items, isAutoencoder])

  const safeReport = useMemo(() => sanitizeDisplayValue(report || ''), [report])

  const topRows = useMemo(() => topAnomalies.map((item) => ({
    ...item,
    amount: formatNumber(item.amount, 2),
    anomaly_score: item.anomaly_score !== null && item.anomaly_score !== undefined ? Number(item.anomaly_score).toFixed(6) : 'N/A',
    anomaly_percentile: item.anomaly_percentile !== null && item.anomaly_percentile !== undefined ? Number(item.anomaly_percentile).toFixed(2) : 'N/A',
  })), [topAnomalies])

  const metricHighlights = useMemo(() => {
    if (isAutoencoder) {
      return [
        { label: 'Total de registros', value: formatNumber(metrics?.total_records ?? metrics?.total_transactions) },
        { label: 'Total de anomalÃ­as', value: formatNumber(metrics?.anomaly_count) },
        { label: 'Porcentaje de anomalÃ­as', value: formatPercent(metrics?.anomaly_rate), highlight: true },
        { label: 'Contamination', value: metrics?.contamination ?? 'N/A' },
        { label: 'Threshold de reconstrucciÃ³n', value: metrics?.threshold !== undefined && metrics?.threshold !== null ? Number(metrics.threshold).toFixed(6) : 'N/A' },
        { label: 'Algoritmo', value: 'Autoencoder PyTorch' },
        { label: 'Run', value: metrics?.source_run || selectedSourceRun || 'N/A' },
        { label: 'epochs', value: metadata?.epochs ?? 'N/A' },
        { label: 'batch_size', value: metadata?.batch_size ?? 'N/A' },
        { label: 'latent_dim', value: metadata?.latent_dim ?? 'N/A' },
        { label: 'learning_rate', value: metadata?.learning_rate ?? 'N/A' },
      ]
    }
    const countryTop = topEntry(metrics?.anomalies_by_country)
    const mccTop = topEntry(metrics?.anomalies_by_mcc)
    const posTop = topEntry(metrics?.anomalies_by_pos_entry_mode)
    return [
      { label: 'Total de transacciones', value: formatNumber(metrics?.total_transactions) },
      { label: 'Total de anomalías', value: formatNumber(metrics?.anomaly_count) },
      { label: 'Porcentaje de anomalías', value: formatPercent(metrics?.anomaly_rate), highlight: true },
      { label: 'Modelo', value: metrics?.model_name || 'N/A' },
      { label: 'Algoritmo', value: metrics?.algorithm || 'N/A' },
      { label: 'Contamination', value: metrics?.contamination ?? 'N/A' },
      { label: 'País con más anomalías', value: countryTop ? `${countryTop[0]} (${countryTop[1]})` : 'N/A' },
      { label: 'MCC con más anomalías', value: mccTop ? `${mccTop[0]} (${mccTop[1]})` : 'N/A' },
      { label: 'POS Entry Mode más frecuente', value: posTop ? `${posTop[0]} (${posTop[1]})` : 'N/A' },
    ]
  }, [metrics, metadata, isAutoencoder, selectedSourceRun])

  const metadataItems = useMemo(() => {
    if (!metadata) return []
    if (isAutoencoder) {
      return [
        { label: 'model_family', value: metadata.model_family || 'UNSUPERVISED' },
        { label: 'algorithm', value: metadata.algorithm || 'autoencoder_pytorch' },
        { label: 'framework', value: metadata.framework || 'pytorch' },
        { label: 'contamination', value: metadata.contamination ?? 'N/A' },
        { label: 'threshold', value: metadata.threshold !== undefined ? Number(metadata.threshold).toFixed(6) : 'N/A' },
        { label: 'total_records', value: formatNumber(metadata.total_records) },
        { label: 'anomaly_count', value: formatNumber(metadata.anomaly_count) },
        { label: 'anomaly_rate', value: metadata.anomaly_rate !== undefined ? `${(Number(metadata.anomaly_rate) * 100).toFixed(2)}%` : 'N/A' },
        { label: 'epochs', value: metadata.epochs ?? 'N/A' },
        { label: 'batch_size', value: metadata.batch_size ?? 'N/A' },
        { label: 'latent_dim', value: metadata.latent_dim ?? 'N/A' },
        { label: 'learning_rate', value: metadata.learning_rate ?? 'N/A' },
        { label: 'model_file', value: metadata.model_file || 'N/A' },
        { label: 'scores_file', value: metadata.scores_file || 'N/A' },
        { label: 'feature_file', value: metadata.feature_file || 'N/A' },
        { label: 'report_file', value: metadata.report_file || 'N/A' },
      ]
    }
    return [
      { label: 'model_name', value: metadata.model_name || 'N/A' },
      { label: 'model_type', value: metadata.model_type || 'N/A' },
      { label: 'algorithm', value: metadata.algorithm || 'N/A' },
      { label: 'contamination', value: metadata.contamination ?? 'N/A' },
      { label: 'total_rows', value: formatNumber(metadata.total_rows) },
      { label: 'anomaly_count', value: formatNumber(metadata.anomaly_count) },
      { label: 'anomaly_rate', value: metadata.anomaly_rate !== undefined ? `${(Number(metadata.anomaly_rate) * 100).toFixed(2)}%` : 'N/A' },
      { label: 'model_path', value: metadata.model_path || 'N/A' },
      { label: 'score_file', value: metadata.score_file || 'N/A' },
      { label: 'feature_file', value: metadata.feature_file || 'N/A' },
      { label: 'report_file', value: metadata.report_file || 'N/A' },
    ]
  }, [metadata, isAutoencoder])

  const detailItems = selectedAnomaly && isAutoencoder ? [
    ['transaction_id', selectedAnomaly.transaction_id || 'N/A'],
    ['customer_hash', selectedAnomaly.customer_hash || 'N/A'],
    ['transaction_datetime', formatMaybeDate(selectedAnomaly.transaction_datetime)],
    ['amount', formatNumber(selectedAnomaly.amount, 2)],
    ['country_code', selectedAnomaly.country_code || 'N/A'],
    ['merchant_rubro_proxy', selectedAnomaly.merchant_rubro_proxy || 'N/A'],
    ['reconstruction_error', selectedAnomaly.reconstruction_error !== undefined ? Number(selectedAnomaly.reconstruction_error).toFixed(6) : 'N/A'],
    ['autoencoder_anomaly_score', selectedAnomaly.autoencoder_anomaly_score !== undefined ? Number(selectedAnomaly.autoencoder_anomaly_score).toFixed(6) : 'N/A'],
    ['autoencoder_anomaly_flag', Number(selectedAnomaly.autoencoder_anomaly_flag) === 1 ? '1' : '0'],
    ['anomaly_rank', selectedAnomaly.anomaly_rank ?? 'N/A'],
  ] : selectedAnomaly ? [
    ['transaction_id', selectedAnomaly.transaction_id || 'N/A'],
    ['customer_hash', selectedAnomaly.customer_hash || 'N/A'],
    ['transaction_datetime', formatMaybeDate(selectedAnomaly.transaction_datetime)],
    ['amount', formatNumber(selectedAnomaly.amount, 2)],
    ['country_code', selectedAnomaly.country_code || 'N/A'],
    ['pos_entry_mode', selectedAnomaly.pos_entry_mode || 'N/A'],
    ['has_pinblock', selectedAnomaly.has_pinblock === true || selectedAnomaly.has_pinblock === 1 ? 'Sí' : 'No'],
    ['merchant_rubro_proxy', selectedAnomaly.merchant_rubro_proxy || 'N/A'],
    ['anomaly_model_name', selectedAnomaly.anomaly_model_name || 'N/A'],
    ['anomaly_score', selectedAnomaly.anomaly_score !== undefined ? Number(selectedAnomaly.anomaly_score).toFixed(6) : 'N/A'],
    ['anomaly_rank', selectedAnomaly.anomaly_rank ?? 'N/A'],
    ['anomaly_percentile', selectedAnomaly.anomaly_percentile !== undefined ? Number(selectedAnomaly.anomaly_percentile).toFixed(2) : 'N/A'],
    ['anomaly_flag', Number(selectedAnomaly.anomaly_flag) === 1 ? '1' : '0'],
    ['created_at', formatMaybeDate(selectedAnomaly.created_at)],
  ] : []

  async function handleCopyReport() {
    if (!safeReport) return
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(safeReport)
      }
      setCopyState('Reporte copiado al portapapeles.')
    } catch (error) {
      setCopyState('No se pudo copiar el reporte.')
    }
  }

  function updateTrainingField(field, value) {
    setTrainingForm((current) => ({ ...current, [field]: value }))
  }

  function updateFilterField(field, value) {
    setDraftFilters((current) => ({ ...current, [field]: value }))
  }

  async function handleTrainModel(event) {
    event.preventDefault()
    setTrainingRunning(true)
    setTrainingMessage('Entrenando modelo no supervisado. Este proceso puede tardar varios minutos.')
    setTrainingError('')
    try {
      const payload = isAutoencoder
        ? {
            source_run: trainingForm.source_run.trim(),
            contamination: Number(trainingForm.contamination),
            epochs: Number(trainingForm.epochs),
            batch_size: Number(trainingForm.batch_size),
            latent_dim: Number(trainingForm.latent_dim),
            learning_rate: Number(trainingForm.learning_rate),
            sample_size: trainingForm.sample_size.trim() ? Number(trainingForm.sample_size) : null,
          }
        : {
            source_run: trainingForm.source_run.trim(),
            contamination: trainingForm.contamination,
            n_estimators: trainingForm.n_estimators,
            max_categories: trainingForm.max_categories,
            sample_size: trainingForm.sample_size.trim(),
          }
      const response = isAutoencoder ? await trainAutoencoderAnomaly(payload) : await trainAnomalyModel(payload)
      const dependencyError = dependencyMessage(response?.status)
      if (dependencyError) {
        setTrainingError(dependencyError)
        return
      }
      setTrainingMessage(`Entrenamiento completado: ${response?.algorithm || selectedAlgorithm} - ${response?.anomaly_run_id || response?.source_run || 'modelo actualizado'}.`)
      const nextRunId = isAutoencoder ? (response?.source_run || payload.source_run) : (response?.anomaly_run_id || '')
      const resolvedRunId = isAutoencoder ? nextRunId : await loadRuns(nextRunId)
      const finalRunId = resolvedRunId || nextRunId || (isAutoencoder ? selectedSourceRun : selectedRunId)
      if (finalRunId) {
        if (!isAutoencoder) setSelectedRunId(finalRunId)
        setPage(1)
        await Promise.all([
          loadMetrics(finalRunId, selectedAlgorithm),
          loadTop(finalRunId, selectedAlgorithm),
          loadReport(finalRunId, selectedAlgorithm),
          loadMetadata(finalRunId, selectedAlgorithm),
          loadScores(finalRunId, 1, pageSize, normalizedScoreFilters, selectedAlgorithm),
        ])
      }
    } catch (error) {
      setTrainingError(extractErrorMessage(error, 'No fue posible ejecutar el entrenamiento no supervisado.'))
    } finally {
      setTrainingRunning(false)
    }
  }

  function handleApplyFilters(event) {
    event.preventDefault()
    setPage(1)
    setAppliedFilters(draftFilters)
  }

  function handleClearFilters() {
    setDraftFilters(DEFAULT_FILTERS)
    setAppliedFilters(DEFAULT_FILTERS)
    setPage(1)
  }

  function handleSelectRun(runId) {
    setSelectedRunId(runId)
    setPage(1)
  }

  function renderEmptyMessage() {
    if (scoresLoading) return 'Cargando anomalías...'
    if (scoresError) return scoresError
    if ((scoresPayload.items || []).length === 0 && scoresPayload.total_items === 0) {
      return 'No hay anomalías que coincidan con los filtros seleccionados.'
    }
    return 'No existen ejecuciones de anomalías todavía. Ejecute primero el entrenamiento no supervisado.'
  }

  return (
    <div className="models-page">
      <div className="header">
        <div>
          <h2>No Supervisados</h2>
          <div className="page-subtitle">Entrenamiento y consulta de modelos no supervisados para detección de anomalías.</div>
        </div>
        <div className="header-actions">
          <span className="warning-badge">Las anomalías no representan fraude confirmado</span>
        </div>
      </div>

      <div className="card warning-banner">
        Las anomalías detectadas por los modelos no supervisados no representan fraude confirmado. Son señales de comportamiento atípico que requieren revisión.
      </div>

      <div className="card detail-section">
        <h3>Entrenar modelo no supervisado</h3>
        <p className="section-help">El entrenamiento puede tardar varios minutos con datasets grandes.</p>
        <form onSubmit={handleTrainModel}>
          <div className="filters-grid training-grid">
            <div className="form-row">
              <label htmlFor="selected_algorithm">Modelo no supervisado</label>
              <select
                id="selected_algorithm"
                className="input"
                value={selectedAlgorithm}
                onChange={(e) => {
                  setSelectedAlgorithm(e.target.value)
                  setPage(1)
                  setSelectedAnomaly(null)
                }}
              >
                <option value="isolation_forest">Isolation Forest</option>
                <option value="autoencoder_pytorch">Autoencoder PyTorch</option>
              </select>
            </div>
            <div className="form-row">
              <label htmlFor="source_run">source_run</label>
              <input id="source_run" className="input" value={trainingForm.source_run} onChange={(e) => updateTrainingField('source_run', e.target.value)} />
            </div>
            <div className="form-row">
              <label htmlFor="contamination">contamination</label>
              <input id="contamination" className="input" type="number" step="0.001" min="0.001" max="0.5" value={trainingForm.contamination} onChange={(e) => updateTrainingField('contamination', e.target.value)} />
            </div>
            {!isAutoencoder && (
              <>
                <div className="form-row">
                  <label htmlFor="n_estimators">n_estimators</label>
                  <input id="n_estimators" className="input" type="number" min="1" value={trainingForm.n_estimators} onChange={(e) => updateTrainingField('n_estimators', e.target.value)} />
                </div>
                <div className="form-row">
                  <label htmlFor="max_categories">max_categories</label>
                  <input id="max_categories" className="input" type="number" min="1" value={trainingForm.max_categories} onChange={(e) => updateTrainingField('max_categories', e.target.value)} />
                </div>
              </>
            )}
            {isAutoencoder && (
              <>
                <div className="form-row">
                  <label htmlFor="epochs">epochs</label>
                  <input id="epochs" className="input" type="number" min="1" value={trainingForm.epochs} onChange={(e) => updateTrainingField('epochs', e.target.value)} />
                </div>
                <div className="form-row">
                  <label htmlFor="batch_size">batch_size</label>
                  <input id="batch_size" className="input" type="number" min="1" value={trainingForm.batch_size} onChange={(e) => updateTrainingField('batch_size', e.target.value)} />
                </div>
                <div className="form-row">
                  <label htmlFor="latent_dim">latent_dim</label>
                  <input id="latent_dim" className="input" type="number" min="1" value={trainingForm.latent_dim} onChange={(e) => updateTrainingField('latent_dim', e.target.value)} />
                </div>
                <div className="form-row">
                  <label htmlFor="learning_rate">learning_rate</label>
                  <input id="learning_rate" className="input" type="number" min="0.000001" step="0.000001" value={trainingForm.learning_rate} onChange={(e) => updateTrainingField('learning_rate', e.target.value)} />
                </div>
              </>
            )}
            <div className="form-row">
              <label htmlFor="sample_size">sample_size (opcional)</label>
              <input id="sample_size" className="input" type="number" min="1" placeholder="Vacío para no limitar" value={trainingForm.sample_size} onChange={(e) => updateTrainingField('sample_size', e.target.value)} />
            </div>
          </div>
          <div className="action-row">
            <button className="button" type="submit" disabled={trainingRunning}>
              {trainingRunning ? 'Ejecutando entrenamiento...' : 'Ejecutar entrenamiento'}
            </button>
            <button className="button button-secondary" type="button" onClick={() => setTrainingForm(DEFAULT_TRAINING_FORM)} disabled={trainingRunning}>
              Restablecer
            </button>
          </div>
        </form>
        {trainingMessage && <div className="status-banner status-success">{trainingMessage}</div>}
        {trainingError && <div className="status-banner status-error">{trainingError}</div>}
      </div>

      <div className="card detail-section">
        <h3>Selector de ejecuciones no supervisadas</h3>
        <p className="section-help">Seleccione una ejecución para ver métricas, anomalías, reporte y metadata.</p>
        {isAutoencoder ? (
          <div className="status-banner status-info">
            Autoencoder PyTorch consulta resultados separados usando source_run: {selectedSourceRun}. Los resultados no representan fraude confirmado.
          </div>
        ) : runsLoading ? (
          <div className="empty-state">Cargando ejecuciones...</div>
        ) : runsError ? (
          <div className="status-banner status-error">{runsError}</div>
        ) : runs.length === 0 ? (
          <div className="empty-state">No existen ejecuciones de anomalías todavía. Ejecute primero el entrenamiento no supervisado.</div>
        ) : (
          <>
            <div className="form-row" style={{ maxWidth: 420 }}>
              <label htmlFor="run_selector">Ejecución activa</label>
              <select id="run_selector" className="input" value={selectedRunId} onChange={(e) => handleSelectRun(e.target.value)}>
                {runs.map((run) => (
                  <option key={run.anomaly_run_id} value={run.anomaly_run_id}>
                    {run.anomaly_run_id} - {run.source_run || 'N/A'}
                  </option>
                ))}
              </select>
            </div>
            <div className="run-card-grid">
              {runs.map((run) => (
                <button
                  key={run.anomaly_run_id}
                  type="button"
                  className={`card run-selector-card run-item-button ${selectedRunId === run.anomaly_run_id ? 'selected-run-card' : ''}`}
                  onClick={() => handleSelectRun(run.anomaly_run_id)}
                >
                  <div className="run-item-title">{run.anomaly_run_id}</div>
                  <div>source_run: {run.source_run || 'N/A'}</div>
                  <div>model_name: {run.model_name || 'N/A'}</div>
                  <div>algorithm: {run.algorithm || 'N/A'}</div>
                  <div>anomaly_count: {formatNumber(run.anomaly_count)}</div>
                  <div>anomaly_rate: {formatPercent(run.anomaly_rate)}</div>
                  <div>created_at: {formatMaybeDate(run.created_at)}</div>
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="card detail-section">
        <h3>Métricas generales</h3>
        {metricsLoading ? (
          <div className="empty-state">Cargando métricas...</div>
        ) : metricsError ? (
          <div className="status-banner status-error">{metricsError}</div>
        ) : (
          <div className="kpi-grid models-kpi-grid">
            {metricHighlights.map((metric) => (
              <KPICard key={metric.label} title={metric.label} value={metric.value} />
            ))}
          </div>
        )}
      </div>

      <div className="card detail-section">
        <h3>Distribuciones</h3>
        {isAutoencoder && <div className="empty-state">Autoencoder muestra métricas, scores, reporte y metadata propios separados de Isolation Forest.</div>}
        {metricsLoading && <div className="empty-state">Cargando distribuciones...</div>}
        {!isAutoencoder && !metricsLoading && metrics && (
          <div className="distribution-grid">
            <InfoList title="Anomalías por país" items={renderDistributionItems(metrics.anomalies_by_country)} />
            <InfoList title="Anomalías por POS Entry Mode" items={renderDistributionItems(metrics.anomalies_by_pos_entry_mode)} />
            <InfoList title="Anomalías por MCC / rubro" items={renderDistributionItems(metrics.anomalies_by_mcc)} />
            <InfoList title="Anomalías por hora" items={renderDistributionItems(metrics.anomalies_by_hour)} />
            <InfoList title="Top clientes por cantidad de anomalías" items={(metrics.top_customers_by_anomaly_count || []).map((item) => [item.customer_hash, item.count])} />
          </div>
        )}
      </div>

      <div className="card detail-section">
        <h3>Tabla de anomalías</h3>
        <p className="section-help">Los resultados se cargan paginados. No se descarga el CSV completo en la UI.</p>
        <form className="filters-panel" onSubmit={handleApplyFilters}>
          <div className="filters-grid">
            <div className="form-row checkbox-row">
              <label htmlFor="filter_anomaly_flag">Solo anomalías</label>
              <input
                id="filter_anomaly_flag"
                type="checkbox"
                checked={draftFilters.anomaly_flag}
                onChange={(e) => updateFilterField('anomaly_flag', e.target.checked)}
              />
            </div>
            <div className="form-row">
              <label htmlFor="filter_country_code">País</label>
              <input id="filter_country_code" className="input" value={draftFilters.country_code} onChange={(e) => updateFilterField('country_code', e.target.value)} placeholder="BO, CL, AR..." />
            </div>
            <div className="form-row">
              <label htmlFor="filter_pos_entry_mode">POS Entry Mode</label>
              <input id="filter_pos_entry_mode" className="input" value={draftFilters.pos_entry_mode} onChange={(e) => updateFilterField('pos_entry_mode', e.target.value)} placeholder="5, 10, 81..." />
            </div>
            <div className="form-row">
              <label htmlFor="filter_mcc">MCC / Rubro</label>
              <input id="filter_mcc" className="input" value={draftFilters.merchant_rubro_proxy} onChange={(e) => updateFilterField('merchant_rubro_proxy', e.target.value)} placeholder="6011, 7995..." />
            </div>
            <div className="form-row">
              <label htmlFor="filter_customer_hash">Cliente</label>
              <input id="filter_customer_hash" className="input" value={draftFilters.customer_hash} onChange={(e) => updateFilterField('customer_hash', e.target.value)} placeholder="Hash del cliente" />
            </div>
            <div className="form-row">
              <label htmlFor="filter_min_score">Score mínimo</label>
              <input id="filter_min_score" className="input" type="number" step="0.000001" value={draftFilters.min_score} onChange={(e) => updateFilterField('min_score', e.target.value)} />
            </div>
            <div className="form-row">
              <label htmlFor="filter_max_score">Score máximo</label>
              <input id="filter_max_score" className="input" type="number" step="0.000001" value={draftFilters.max_score} onChange={(e) => updateFilterField('max_score', e.target.value)} />
            </div>
          </div>
          <div className="action-row">
            <button className="button" type="submit">Aplicar filtros</button>
            <button className="button button-secondary" type="button" onClick={handleClearFilters}>Limpiar filtros</button>
            <div className="page-size-control">
              <label htmlFor="page_size">Filas por página</label>
              <select id="page_size" className="input" value={pageSize} onChange={(e) => { setPage(1); setPageSize(Number(e.target.value)) }}>
                {[20, 50, 100, 200].map((size) => <option key={size} value={size}>{size}</option>)}
              </select>
            </div>
          </div>
        </form>

        {scoresLoading ? (
          <div className="empty-state">Cargando anomalías...</div>
        ) : scoresError ? (
          <div className="status-banner status-error">{scoresError}</div>
        ) : (scoresPayload.items || []).length === 0 ? (
          <div className="empty-state">{renderEmptyMessage()}</div>
        ) : (
          <>
            <div className="table-scroll">
              <table className="table anomaly-table">
                <thead>
                  <tr>
                    {activeScoreColumns.map((column) => (
                      <th key={column.key}>{column.title}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableRows.map((row, index) => (
                    <tr key={`${row.transaction_id}-${row.anomaly_rank}-${index}`}>
                      <td>{row.anomaly_rank}</td>
                      <td>{row.transaction_id}</td>
                      <td>{row.customer_hash}</td>
                      <td>{formatMaybeDate(row.transaction_datetime)}</td>
                      <td>{row.amount}</td>
                      <td>{row.country_code}</td>
                      <td>{row.pos_entry_mode}</td>
                      <td>{row.has_pinblock === true || row.has_pinblock === 1 ? 'Sí' : 'No'}</td>
                      <td>{row.merchant_rubro_proxy}</td>
                      <td>{row.anomaly_score}</td>
                      <td>{row.anomaly_percentile}</td>
                      <td>{row.anomaly_flag}</td>
                      <td>{row.anomaly_model_name}</td>
                      <td>
                        <button type="button" className="button button-secondary table-row-action" onClick={() => setSelectedAnomaly(scoresPayload.items[index])}>
                          Ver detalle
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination-bar">
              <div>
                Página {scoresPayload.page || 1} de {scoresPayload.total_pages || 0} · {formatNumber(scoresPayload.total_items)} resultados
              </div>
              <div className="action-row">
                <button className="button button-secondary" type="button" disabled={page <= 1 || scoresLoading} onClick={() => setPage((current) => Math.max(1, current - 1))}>Anterior</button>
                <button className="button button-secondary" type="button" disabled={page >= (scoresPayload.total_pages || 0) || scoresLoading} onClick={() => setPage((current) => current + 1)}>Siguiente</button>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="card detail-section">
        <h3>Top 20 anomalías detectadas</h3>
        <p className="section-help">Ranking de anomalías estadísticas, no de fraude confirmado.</p>
        {isAutoencoder ? (
          <div className="empty-state">Use la tabla paginada de Autoencoder ordenada por anomaly_rank para revisar los mayores errores de reconstrucción.</div>
        ) : topLoading ? (
          <div className="empty-state">Cargando top anomalías...</div>
        ) : topError ? (
          <div className="status-banner status-error">{topError}</div>
        ) : topRows.length > 0 ? (
          <Table columns={topColumns} data={topRows} />
        ) : (
          <div className="empty-state">No hay top anomalías para la ejecución seleccionada.</div>
        )}
      </div>

      <div className="card detail-section">
        <h3>Reporte de anomalías</h3>
        <div className="action-row">
          <button className="button button-secondary" type="button" onClick={handleCopyReport} disabled={!report}>Copiar reporte</button>
          {copyState && <div className="inline-message">{copyState}</div>}
        </div>
        {reportLoading ? (
          <div className="empty-state">Cargando reporte...</div>
        ) : reportError ? (
          <div className="status-banner status-error">{reportError}</div>
        ) : safeReport ? (
          <pre className="report-pre report-box">{safeReport}</pre>
        ) : (
          <div className="empty-state">No hay reporte disponible para la ejecución seleccionada.</div>
        )}
      </div>

      <div className="card detail-section">
        <h3>Metadata del modelo</h3>
        {metadataLoading ? (
          <div className="empty-state">Cargando metadata...</div>
        ) : metadataError ? (
          <div className="status-banner status-error">{metadataError}</div>
        ) : metadata ? (
          <>
            <KeyValueGrid title="Resumen de metadata" items={metadataItems} />
            <InfoList title="Features usadas" items={[
              ['numeric_features', (metadata.numeric_features || []).join(', ') || 'N/A'],
              ['categorical_features', (metadata.categorical_features || []).join(', ') || 'N/A'],
              ['model_input_columns', (metadata.model_input_columns || []).join(', ') || 'N/A'],
              ['excluded_columns', sanitizeDisplayValue(metadata.excluded_columns || []).join(', ') || 'N/A'],
            ]} />
            <div className="status-banner status-info">
              customer_hash se conserva como contexto, no como predictor directo. No se usaron etiquetas de fraude confirmado ni reglas como etiquetas.
            </div>
          </>
        ) : (
          <div className="empty-state">No hay metadata disponible para la ejecución seleccionada.</div>
        )}
      </div>

      {selectedAnomaly && (
        <div className="modal-backdrop" onClick={() => setSelectedAnomaly(null)}>
          <div className="modal-panel anomaly-detail-panel" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3>Detalle de anomalía</h3>
                <p className="section-help">Esta transacción fue marcada como anomalía estadística. No constituye fraude confirmado.</p>
              </div>
              <button className="icon-button" type="button" onClick={() => setSelectedAnomaly(null)}>×</button>
            </div>
            <div className="table-scroll">
              <table className="table detail-table">
                <tbody>
                  {detailItems.map(([label, value]) => (
                    <tr key={label}>
                      <td style={{ fontWeight: 700, width: '40%' }}>{label}</td>
                      <td>{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="action-row modal-actions-row">
              <button className="button" type="button" onClick={() => setSelectedAnomaly(null)}>Cerrar</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
