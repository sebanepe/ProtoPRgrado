import React, { useEffect, useMemo, useState } from 'react'
import KPICard from '../components/KPICard'
import {
  buildHumanSupervisedDataset,
  getHumanDatasetPreview,
  getHumanDatasetSummary,
  getHumanLabelSummary,
  getHumanReadiness,
  getSupervisedModelMetadata,
  getSupervisedModelPredictions,
  getSupervisedModelReport,
  getSupervisedTrainingPreflight,
  getSupervisedTrainingRuns,
  trainHumanSupervisedModel,
  validateHumanDataset,
  getSupInferenceTrainedModels,
  getSupInferencePreprocessedRuns,
  applySupInferenceModel,
  getSupInferenceStatus,
  getSupInferencePredictionRuns,
  getSupInferencePredictionResults,
  getSupInferencePredictionReport
} from '../services/api'

const DEFAULT_SOURCE_RUN = 'preprocessed_run_26'
const MODEL_OPTIONS = [
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'logistic_regression', label: 'Logistic Regression' },
  { value: 'gradient_boosting', label: 'Gradient Boosting' },
  { value: 'mlp_classifier', label: 'MLP Classifier' }
]
const MODEL_LABELS = {
  logistic_regression: 'Regresion logistica',
  random_forest: 'Random Forest',
  gradient_boosting: 'Gradient Boosting',
  mlp_classifier: 'Red neuronal MLP',
  mlp: 'Red neuronal MLP'
}
const MODEL_DESCRIPTIONS = {
  logistic_regression: 'Modelo base e interpretable.',
  random_forest: 'Modelo robusto basado en multiples arboles.',
  gradient_boosting: 'Modelo secuencial que corrige errores progresivamente.',
  mlp_classifier: 'Red neuronal. Requiere mas etiquetas para evitar sobreajuste.'
}
const METRIC_HELP = [
  ['Accuracy', 'Porcentaje general de aciertos. No debe evaluarse sola.'],
  ['Precision', 'De los casos que el modelo marco como positivos, cuantos eran CONFIRMED_FRAUD.'],
  ['Recall', 'De los fraudes confirmados, cuantos logro detectar el modelo.'],
  ['F1-score', 'Balance entre precision y recall.'],
  ['ROC-AUC', 'Capacidad del modelo para separar alertas confirmadas y descartadas.']
]
const PREDICTION_LABELS = {
  summary_alert_id: 'ID Alerta',
  y_true: 'Etiqueta humana',
  y_pred: 'Prediccion del modelo',
  y_proba: 'Probabilidad estimada',
  prediction_label: 'Resultado predicho',
  evaluation_result: 'Evaluacion'
}
const EVALUATION_LABELS = {
  TRUE_POSITIVE: 'Acierto positivo',
  TRUE_NEGATIVE: 'Acierto negativo',
  FALSE_POSITIVE: 'Falsa alarma',
  FALSE_NEGATIVE: 'Fraude no detectado'
}
const FORBIDDEN_COLUMNS = new Set(['is_fraud', 'confirmed_fraud', 'PAN_TARJETA', 'TARJETA', 'pan_card', 'raw_card'])
const METHODOLOGY_MESSAGE = 'El modelo supervisado fue entrenado con etiquetas humanas de revision. Sus predicciones apoyan la priorizacion analitica y no constituyen fraude confirmado automatico.'

const safeNumber = (value) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

const formatCount = (value) => safeNumber(value).toLocaleString()
const formatMetric = (value) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed.toFixed(4) : 'No disponible'
}

const friendlyError = (error, fallback) => {
  const status = error?.response?.status
  const detail = error?.response?.data?.detail
  const reason = typeof detail === 'object' ? detail?.blocking_reason : null
  if (status === 404) return 'No existe informacion disponible para esta seleccion.'
  if (status === 409 && reason) return `Entrenamiento bloqueado: ${reason}.`
  if (status >= 500) return 'El servicio no esta disponible temporalmente.'
  return fallback
}

const cleanRows = (rows = []) => rows.map((row) => Object.fromEntries(
  Object.entries(row || {}).filter(([key]) => !FORBIDDEN_COLUMNS.has(key))
))

const verdictMessage = (verdict) => {
  if (verdict === 'HUMAN_LABELS_STRONG_READY') return 'Existen etiquetas suficientes para un entrenamiento mas solido.'
  if (verdict === 'HUMAN_LABELS_RECOMMENDED_READY') return 'Existen etiquetas suficientes para un entrenamiento recomendado.'
  if (verdict === 'HUMAN_LABELS_TECHNICALLY_READY') return 'Existen etiquetas suficientes para una prueba tecnica inicial.'
  return 'No existen suficientes etiquetas humanas para entrenar un modelo supervisado.'
}

function StatusBadge({ children, tone = 'neutral' }) {
  const colors = {
    success: ['rgba(39, 209, 127, 0.12)', '#a7f3d0', 'rgba(39, 209, 127, 0.24)'],
    warning: ['rgba(247, 185, 85, 0.12)', '#ffe4b8', 'rgba(247, 185, 85, 0.24)'],
    error: ['rgba(255, 107, 107, 0.12)', '#fecaca', 'rgba(255, 107, 107, 0.24)'],
    neutral: ['rgba(56, 214, 214, 0.10)', 'var(--text)', 'rgba(56, 214, 214, 0.18)']
  }
  const [background, color, borderColor] = colors[tone] || colors.neutral
  return (
    <span className="warning-badge" style={{ borderRadius: 8, background, color, borderColor }}>
      {children}
    </span>
  )
}

function ProgressLine({ label, current, required }) {
  const safeCurrent = safeNumber(current)
  const safeRequired = Math.max(safeNumber(required), 1)
  const percent = Math.min((safeCurrent / safeRequired) * 100, 100)
  return (
    <div style={{ display: 'grid', gap: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, fontSize: 13 }}>
        <span>{label}</span>
        <strong>{formatCount(safeCurrent)} / {formatCount(safeRequired)}</strong>
      </div>
      <div aria-label={`${label}: ${safeCurrent} / ${safeRequired}`} style={{ height: 8, borderRadius: 999, overflow: 'hidden', background: 'rgba(36, 52, 71, 0.95)', border: '1px solid rgba(56, 214, 214, 0.16)' }}>
        <div style={{ width: `${percent}%`, height: '100%', background: percent >= 100 ? 'var(--success)' : 'var(--accent-blue)' }} />
      </div>
    </div>
  )
}

function GoalCard({ title, positiveCurrent, negativeCurrent, positiveRequired, negativeRequired, ready }) {
  return (
    <div className="card" style={{ margin: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 14 }}>
        <h4 style={{ margin: 0 }}>{title}</h4>
        <StatusBadge tone={ready ? 'success' : 'warning'}>{ready ? 'Listo' : 'Pendiente'}</StatusBadge>
      </div>
      <div style={{ display: 'grid', gap: 12 }}>
        <ProgressLine label="Positivos" current={positiveCurrent} required={positiveRequired} />
        <ProgressLine label="Negativos" current={negativeCurrent} required={negativeRequired} />
      </div>
    </div>
  )
}

function DataTable({ rows, emptyText = 'Sin datos disponibles.' }) {
  const columns = useMemo(() => {
    const first = cleanRows(rows)[0] || {}
    return Object.keys(first).filter((column) => !FORBIDDEN_COLUMNS.has(column))
  }, [rows])
  const safeRows = cleanRows(rows)
  if (!safeRows.length || !columns.length) return <div className="detail-status" style={{ padding: 12, borderRadius: 8 }}>{emptyText}</div>
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="table">
        <thead>
          <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {safeRows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => <td key={column}>{String(row[column] ?? '')}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MetricInfoCards() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 12 }}>
      {METRIC_HELP.map(([title, text]) => (
        <div key={title} className="card" style={{ margin: 0, padding: 14 }}>
          <strong>{title}</strong>
          <p style={{ color: 'var(--text-muted)', margin: '8px 0 0', fontSize: 13 }}>{text}</p>
        </div>
      ))}
    </div>
  )
}

function MiniBarChart({ title, data, metrics }) {
  const colors = ['#38d6d6', '#27d17f', '#f7b955', '#8fb3ff', '#c084fc']
  if (!data.length) return <div className="detail-status" style={{ padding: 12, borderRadius: 8 }}>No hay metricas para graficar.</div>
  return (
    <div className="card" style={{ margin: 0 }}>
      <h4 style={{ marginTop: 0 }}>{title}</h4>
      <div style={{ display: 'grid', gap: 14 }}>
        {data.map((item) => (
          <div key={item.model}>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>{MODEL_LABELS[item.model] || item.model}</div>
            <div style={{ display: 'grid', gap: 6 }}>
              {metrics.map((metric, index) => {
                const value = Math.max(0, Math.min(1, safeNumber(item[metric])))
                return (
                  <div key={metric} style={{ display: 'grid', gridTemplateColumns: '88px 1fr 58px', gap: 8, alignItems: 'center', fontSize: 12 }}>
                    <span>{metric}</span>
                    <div style={{ height: 10, background: 'rgba(36, 52, 71, 0.95)', borderRadius: 999, overflow: 'hidden' }}>
                      <div aria-label={`${item.model} ${metric} ${formatMetric(value)}`} style={{ width: `${value * 100}%`, height: '100%', background: colors[index % colors.length] }} />
                    </div>
                    <strong>{formatMetric(value)}</strong>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function ConfusionMatrix({ matrix }) {
  const safe = Array.isArray(matrix) ? matrix : [[0, 0], [0, 0]]
  const maxValue = Math.max(1, ...safe.flat().map((value) => safeNumber(value)))
  const cells = [
    ['Verdadero Negativo', safe?.[0]?.[0] ?? 0, 'Real: DISMISSED', 'Predicho: DISMISSED', 'El modelo descarto correctamente una alerta DISMISSED.', 'rgba(56, 214, 214, VALUE)'],
    ['Falso Positivo', safe?.[0]?.[1] ?? 0, 'Real: DISMISSED', 'Predicho: CONFIRMED_FRAUD', 'El modelo priorizo una alerta que la revision humana descarto.', 'rgba(247, 185, 85, VALUE)'],
    ['Falso Negativo', safe?.[1]?.[0] ?? 0, 'Real: CONFIRMED_FRAUD', 'Predicho: DISMISSED', 'El modelo descarto una alerta que la revision humana confirmo como fraude.', 'rgba(255, 107, 107, VALUE)'],
    ['Verdadero Positivo', safe?.[1]?.[1] ?? 0, 'Real: CONFIRMED_FRAUD', 'Predicho: CONFIRMED_FRAUD', 'El modelo detecto una alerta que la revision humana confirmo como fraude.', 'rgba(39, 209, 127, VALUE)']
  ]
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '150px repeat(2, minmax(180px, 1fr))', gap: 8, alignItems: 'stretch', maxWidth: 760 }} aria-label="Matriz de confusion tipo heatmap">
        <div />
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontWeight: 700 }}>Predicho: DISMISSED</div>
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontWeight: 700 }}>Predicho: CONFIRMED_FRAUD</div>
        <div style={{ display: 'flex', alignItems: 'center', color: 'var(--text-muted)', fontWeight: 700 }}>Real: DISMISSED</div>
        {cells.slice(0, 2).map(([label, value, real, predicted, description, color]) => (
          <div key={label} style={{ padding: 16, borderRadius: 8, border: '1px solid rgba(255,255,255,0.10)', background: color.replace('VALUE', String(0.18 + (safeNumber(value) / maxValue) * 0.32)) }}>
            <strong>{label}</strong>
            <div style={{ fontSize: 34, fontWeight: 800, margin: '8px 0' }}>{value}</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{real} / {predicted}</div>
            <p style={{ margin: '8px 0 0', fontSize: 13 }}>{description}</p>
          </div>
        ))}
        <div style={{ display: 'flex', alignItems: 'center', color: 'var(--text-muted)', fontWeight: 700 }}>Real: CONFIRMED_FRAUD</div>
        {cells.slice(2).map(([label, value, real, predicted, description, color]) => (
          <div key={label} style={{ padding: 16, borderRadius: 8, border: '1px solid rgba(255,255,255,0.10)', background: color.replace('VALUE', String(0.18 + (safeNumber(value) / maxValue) * 0.32)) }}>
            <strong>{label}</strong>
            <div style={{ fontSize: 34, fontWeight: 800, margin: '8px 0' }}>{value}</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{real} / {predicted}</div>
            <p style={{ margin: '8px 0 0', fontSize: 13 }}>{description}</p>
          </div>
        ))}
      </div>
      <p style={{ color: 'var(--text-muted)' }}>
        Formato tecnico: [[TN, FP], [FN, TP]]. Este falso positivo es una metrica del modelo, no el estado manual FALSE_POSITIVE.
      </p>
    </div>
  )
}

function PredictionDistribution({ rows }) {
  const safeRows = cleanRows(rows)
  const positives = safeRows.filter((row) => Number(row.y_pred) === 1 || row.prediction_label === 'CONFIRMED_FRAUD').length
  const negatives = safeRows.filter((row) => Number(row.y_pred) === 0 || row.prediction_label === 'DISMISSED').length
  const total = Math.max(1, positives + negatives)
  const ranges = [
    ['0.0 - 0.2', 0, 0.2],
    ['0.2 - 0.4', 0.2, 0.4],
    ['0.4 - 0.6', 0.4, 0.6],
    ['0.6 - 0.8', 0.6, 0.8],
    ['0.8 - 1.0', 0.8, 1.01]
  ].map(([label, min, max]) => ({
    label,
    count: safeRows.filter((row) => row.y_proba !== undefined && Number(row.y_proba) >= min && Number(row.y_proba) < max).length
  }))
  const hasProba = safeRows.some((row) => row.y_proba !== undefined && row.y_proba !== null && row.y_proba !== '')
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
      <div className="card" style={{ margin: 0 }}>
        <h4 style={{ marginTop: 0 }}>Distribucion de predicciones</h4>
        <div style={{ display: 'grid', gap: 10 }}>
          <ProgressLine label="Predicho DISMISSED" current={negatives} required={total} />
          <ProgressLine label="Predicho CONFIRMED_FRAUD" current={positives} required={total} />
        </div>
      </div>
      <div className="card" style={{ margin: 0 }}>
        <h4 style={{ marginTop: 0 }}>Distribucion de probabilidad estimada</h4>
        <p style={{ color: 'var(--text-muted)', marginTop: 0 }}>Permite observar que tan seguros son los modelos en sus predicciones.</p>
        {!hasProba && <div className="detail-status" style={{ padding: 10, borderRadius: 8 }}>Este modelo no expone probabilidad estimada.</div>}
        {hasProba && ranges.map((item) => <ProgressLine key={item.label} label={item.label} current={item.count} required={safeRows.length} />)}
      </div>
    </div>
  )
}

function FriendlyPredictionsTable({ rows, emptyText = 'No existen predicciones para este modelo.' }) {
  const displayColumns = ['summary_alert_id', 'y_true', 'y_pred', 'y_proba', 'prediction_label', 'evaluation_result']
  const safeRows = cleanRows(rows)
  if (!safeRows.length) return <div className="detail-status" style={{ padding: 12, borderRadius: 8 }}>{emptyText}</div>
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="table">
        <thead>
          <tr>{displayColumns.map((column) => <th key={column}>{PREDICTION_LABELS[column]}</th>)}</tr>
        </thead>
        <tbody>
          {safeRows.map((row, index) => (
            <tr key={index}>
              {displayColumns.map((column) => {
                if (column === 'evaluation_result') {
                  return (
                    <td key={column}>
                      <StatusBadge tone={String(row[column]).includes('FALSE') ? 'warning' : 'success'}>{EVALUATION_LABELS[row[column]] || row[column]}</StatusBadge>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>{row[column]}</div>
                    </td>
                  )
                }
                if (column === 'y_true' || column === 'y_pred') {
                  return <td key={column}>{Number(row[column]) === 1 ? 'CONFIRMED_FRAUD' : 'DISMISSED'} <span style={{ color: 'var(--text-muted)' }}>({row[column]})</span></td>
                }
                if (column === 'y_proba') return <td key={column}>{row[column] == null ? 'No disponible' : formatMetric(row[column])}</td>
                return <td key={column}>{String(row[column] ?? '')}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function ModelSupervised() {
  const [activeTab, setActiveTab] = useState('training')

  // ── Training tab state ──────────────────────────────────────────────────────
  const [sourceRun, setSourceRun] = useState(DEFAULT_SOURCE_RUN)
  const [summary, setSummary] = useState(null)
  const [readiness, setReadiness] = useState(null)
  const [preflight, setPreflight] = useState(null)
  const [datasetSummary, setDatasetSummary] = useState(null)
  const [datasetValidation, setDatasetValidation] = useState(null)
  const [datasetPreview, setDatasetPreview] = useState(null)
  const [trainingRuns, setTrainingRuns] = useState([])
  const [selectedModel, setSelectedModel] = useState('random_forest')
  const [metadata, setMetadata] = useState(null)
  const [report, setReport] = useState('')
  const [predictions, setPredictions] = useState({ rows: [], page: 1, total_pages: 1, total: 0 })
  const [predictionFilters, setPredictionFilters] = useState({ quick_filter: '', evaluation_result: '', y_true: '', y_pred: '', prediction_label: '' })
  const [loading, setLoading] = useState({})
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [modelConfig, setModelConfig] = useState({ model_type: 'random_forest', test_size: '0.2', random_state: '42', use_smote: false })

  // ── Apply tab state ─────────────────────────────────────────────────────────
  const [supTrainedModels, setSupTrainedModels] = useState([])
  const [supPreprocessedRuns, setSupPreprocessedRuns] = useState([])
  const [supSelectedModelId, setSupSelectedModelId] = useState('')
  const [supInputType, setSupInputType] = useState('preprocessed_run')
  const [supCsvFile, setSupCsvFile] = useState(null)
  const [supPreprocessedRunId, setSupPreprocessedRunId] = useState('')
  const [supApplyLoading, setSupApplyLoading] = useState(false)
  const [supApplyError, setSupApplyError] = useState('')
  const [supApplySuccess, setSupApplySuccess] = useState('')
  const [supPredRuns, setSupPredRuns] = useState([])
  const [supSelectedPredRunId, setSupSelectedPredRunId] = useState('')
  const [supPredResults, setSupPredResults] = useState({ rows: [], total: 0 })
  const [supPredReport, setSupPredReport] = useState(null)
  const [supPollRunId, setSupPollRunId] = useState(null)
  const [supPollStatus, setSupPollStatus] = useState('')
  const [supPriorityFilter, setSupPriorityFilter] = useState('ALL')
  const [supPredPage, setSupPredPage] = useState(1)
  const [supSameRunWarning, setSupSameRunWarning] = useState(null)
  const SUP_PAGE_SIZE = 50

  const normalizedRun = sourceRun.trim() || DEFAULT_SOURCE_RUN

  const loadBase = async () => {
    setLoading((prev) => ({ ...prev, base: true }))
    setError('')
    try {
      const [summaryPayload, readinessPayload, preflightPayload, datasetPayload, runsPayload] = await Promise.all([
        getHumanLabelSummary(normalizedRun),
        getHumanReadiness(normalizedRun),
        getSupervisedTrainingPreflight(normalizedRun).catch((err) => ({ error: friendlyError(err, 'No se pudo cargar preflight.') })),
        getHumanDatasetSummary(normalizedRun).catch((err) => ({ error: friendlyError(err, 'No se pudo cargar dataset supervisado.') })),
        getSupervisedTrainingRuns(normalizedRun).catch(() => ({ items: [] }))
      ])
      setSummary(summaryPayload || null)
      setReadiness(readinessPayload || null)
      setPreflight(preflightPayload?.error ? null : preflightPayload)
      setDatasetSummary(datasetPayload?.error ? null : datasetPayload)
      setTrainingRuns(runsPayload?.items || [])
      if (preflightPayload?.error || datasetPayload?.error) setError(preflightPayload?.error || datasetPayload?.error)
    } catch (err) {
      setSummary(null)
      setReadiness(null)
      setError(friendlyError(err, 'No se pudo cargar la pantalla Supervisados.'))
    } finally {
      setLoading((prev) => ({ ...prev, base: false }))
    }
  }

  const loadModelDetails = async (modelType = selectedModel, page = 1) => {
    setLoading((prev) => ({ ...prev, details: true }))
    const params = { page, page_size: 20 }
    Object.entries(predictionFilters).forEach(([key, value]) => {
      if (value !== '' && key !== 'quick_filter') params[key] = value
    })
    try {
      const [metadataPayload, reportPayload, predictionsPayload] = await Promise.all([
        getSupervisedModelMetadata(normalizedRun, modelType).catch((err) => ({ error: friendlyError(err, 'No existe metadata para este modelo.') })),
        getSupervisedModelReport(normalizedRun, modelType).catch((err) => ({ error: friendlyError(err, 'No existe reporte para este modelo.') })),
        getSupervisedModelPredictions(normalizedRun, modelType, params).catch((err) => ({ error: friendlyError(err, 'No existen predicciones para este modelo.') }))
      ])
      setMetadata(metadataPayload?.metadata || null)
      setReport(reportPayload?.markdown || '')
      setPredictions(predictionsPayload?.rows ? predictionsPayload : { rows: [], page: 1, total_pages: 1, total: 0 })
    } finally {
      setLoading((prev) => ({ ...prev, details: false }))
    }
  }

  useEffect(() => {
    loadBase()
  }, [normalizedRun])

  useEffect(() => {
    if (trainingRuns.length) loadModelDetails(selectedModel, 1)
  }, [selectedModel, trainingRuns.length])

  // ── Apply tab loaders ──────────────────────────────────────────────────────
  const loadSupTrainedModels = async () => {
    try { setSupTrainedModels(await getSupInferenceTrainedModels()) } catch (_) {}
  }
  const loadSupPreprocessedRuns = async () => {
    try { setSupPreprocessedRuns(await getSupInferencePreprocessedRuns() ?? []) } catch (_) {}
  }
  const loadSupPredRuns = async () => {
    try { setSupPredRuns(await getSupInferencePredictionRuns()) } catch (_) {}
  }
  const loadSupPredResults = async (runId, page, priorityFilter) => {
    try {
      const data = await getSupInferencePredictionResults(runId, { page, page_size: SUP_PAGE_SIZE, priority_filter: priorityFilter })
      setSupPredResults(data)
    } catch (_) {}
  }
  const loadSupPredReport = async (runId) => {
    try { setSupPredReport(await getSupInferencePredictionReport(runId)) } catch (_) {}
  }

  useEffect(() => {
    if (activeTab === 'apply') {
      loadSupTrainedModels()
      loadSupPreprocessedRuns()
      loadSupPredRuns()
    }
  }, [activeTab])

  // Polling useEffect
  useEffect(() => {
    if (!supPollRunId) return
    const interval = setInterval(async () => {
      try {
        const statusData = await getSupInferenceStatus(supPollRunId)
        setSupPollStatus(statusData.status)
        if (statusData.status === 'COMPLETED') {
          clearInterval(interval)
          setSupPollRunId(null)
          setSupApplyLoading(false)
          setSupApplySuccess(`Modelo aplicado. Run ID: ${statusData.id} — ${statusData.total_analyzed?.toLocaleString()} alertas analizadas.`)
          setSupSameRunWarning(statusData.same_run_warning || null)
          setSupSelectedPredRunId(String(supPollRunId))
          await loadSupPredRuns()
          await loadSupPredResults(supPollRunId, 1, 'ALL')
          await loadSupPredReport(supPollRunId)
          setSupPriorityFilter('ALL')
          setSupPredPage(1)
        } else if (statusData.status === 'FAILED') {
          clearInterval(interval)
          setSupPollRunId(null)
          setSupApplyLoading(false)
          setSupApplyError(statusData.error_message || 'La inferencia supervisada falló.')
        }
      } catch (_) {
        clearInterval(interval)
        setSupPollRunId(null)
        setSupApplyLoading(false)
        setSupApplyError('No se pudo obtener el estado de la inferencia.')
      }
    }, 3000)
    return () => clearInterval(interval)
  }, [supPollRunId])

  const handleSupApplyModel = async (event) => {
    event.preventDefault()
    if (!supSelectedModelId) { setSupApplyError('Seleccione un modelo entrenado.'); return }
    if (supInputType === 'csv_upload' && !supCsvFile) { setSupApplyError('Seleccione un archivo CSV.'); return }
    if (supInputType === 'preprocessed_run' && !supPreprocessedRunId) { setSupApplyError('Seleccione un preprocessed_run.'); return }
    setSupApplyLoading(true)
    setSupApplyError('')
    setSupApplySuccess('')
    setSupSameRunWarning(null)
    try {
      const formData = new FormData()
      formData.append('model_registry_id', supSelectedModelId)
      formData.append('input_type', supInputType)
      if (supInputType === 'csv_upload') { formData.append('file', supCsvFile) }
      else { formData.append('preprocessed_run_id', supPreprocessedRunId) }
      const result = await applySupInferenceModel(formData)
      setSupPollStatus('PENDING')
      setSupPollRunId(result.run_id)
    } catch (err) {
      const detail = err?.response?.data?.detail
      setSupApplyError(typeof detail === 'string' ? detail : 'No se pudo iniciar la inferencia.')
      setSupApplyLoading(false)
    }
  }

  const handleSupFilterChange = async (newFilter) => {
    setSupPriorityFilter(newFilter)
    setSupPredPage(1)
    if (supSelectedPredRunId) await loadSupPredResults(supSelectedPredRunId, 1, newFilter)
  }

  const handleSupSelectRun = async (runId) => {
    setSupSelectedPredRunId(runId)
    setSupPriorityFilter('ALL')
    setSupPredPage(1)
    if (runId) {
      await loadSupPredResults(runId, 1, 'ALL')
      await loadSupPredReport(runId)
    }
  }

  const currentPositive = safeNumber(summary?.usable_positive_labels ?? readiness?.current?.positive)
  const currentNegative = safeNumber(summary?.usable_negative_labels ?? readiness?.current?.negative)
  const technicalReady = Boolean(preflight?.human_labels?.technical_ready ?? readiness?.technical_ready ?? summary?.technical_ready)
  const recommendedReady = Boolean(preflight?.human_labels?.recommended_ready ?? readiness?.recommended_ready ?? summary?.recommended_ready)
  const datasetReady = preflight?.dataset?.verdict === 'HUMAN_SUPERVISED_DATASET_READY' || datasetValidation?.verdict === 'HUMAN_SUPERVISED_DATASET_READY'
  const mlpBlocked = modelConfig.model_type === 'mlp_classifier' && !recommendedReady
  const canTrain = technicalReady && datasetReady && !mlpBlocked && !loading.training

  const metrics = useMemo(() => [
    { title: 'Total de revisiones', value: formatCount(summary?.total_reviews) },
    { title: 'CONFIRMED_FRAUD', value: formatCount(summary?.confirmed_fraud) },
    { title: 'DISMISSED', value: formatCount(summary?.dismissed) },
    { title: 'FALSE_POSITIVE excluido', value: formatCount(summary?.false_positive_excluded) },
    { title: 'NEW excluido', value: formatCount(summary?.new) },
    { title: 'IN_REVIEW excluido', value: formatCount(summary?.in_review) },
    { title: 'Positivas usables', value: formatCount(summary?.usable_positive_labels) },
    { title: 'Negativas usables', value: formatCount(summary?.usable_negative_labels) },
    { title: 'Total usable', value: formatCount(summary?.usable_total_labels) }
  ], [summary])

  const bestModel = useMemo(() => {
    return [...trainingRuns].sort((a, b) => safeNumber(b.metrics?.f1_score ?? b.metrics_json?.f1_score) - safeNumber(a.metrics?.f1_score ?? a.metrics_json?.f1_score))[0]
  }, [trainingRuns])

  const selectedMetrics = metadata?.metrics || trainingRuns.find((item) => item.algorithm === selectedModel)?.metrics || {}
  const selectedRun = trainingRuns.find((item) => item.algorithm === selectedModel)
  const chartData = trainingRuns.map((run) => {
    const runMetrics = run.metrics || run.metrics_json || {}
    return {
      model: run.algorithm,
      Accuracy: safeNumber(runMetrics.accuracy),
      Precision: safeNumber(runMetrics.precision),
      Recall: safeNumber(runMetrics.recall),
      'F1-score': safeNumber(runMetrics.f1_score),
      'ROC-AUC': safeNumber(runMetrics.roc_auc)
    }
  })
  const bestRecall = [...trainingRuns].sort((a, b) => safeNumber((b.metrics || b.metrics_json || {}).recall) - safeNumber((a.metrics || a.metrics_json || {}).recall))[0]
  const bestPrecision = [...trainingRuns].sort((a, b) => safeNumber((b.metrics || b.metrics_json || {}).precision) - safeNumber((a.metrics || a.metrics_json || {}).precision))[0]
  const visiblePredictionRows = useMemo(() => {
    const rows = cleanRows(predictions.rows || [])
    const filter = predictionFilters.quick_filter
    if (filter === 'hits') return rows.filter((row) => String(row.evaluation_result).startsWith('TRUE'))
    if (filter === 'errors') return rows.filter((row) => String(row.evaluation_result).startsWith('FALSE'))
    if (filter === 'false_positive') return rows.filter((row) => row.evaluation_result === 'FALSE_POSITIVE')
    if (filter === 'false_negative') return rows.filter((row) => row.evaluation_result === 'FALSE_NEGATIVE')
    if (filter === 'predicted_positive') return rows.filter((row) => Number(row.y_pred) === 1 || row.prediction_label === 'CONFIRMED_FRAUD')
    if (filter === 'predicted_negative') return rows.filter((row) => Number(row.y_pred) === 0 || row.prediction_label === 'DISMISSED')
    return rows
  }, [predictions.rows, predictionFilters.quick_filter])

  const handleBuildDataset = async () => {
    setLoading((prev) => ({ ...prev, datasetBuild: true }))
    setNotice('')
    setError('')
    try {
      const result = await buildHumanSupervisedDataset(normalizedRun, { force: false })
      setNotice(`Dataset supervisado: ${result.verdict || 'operacion completada'}.`)
      await loadBase()
    } catch (err) {
      setError(friendlyError(err, 'No se pudo construir el dataset supervisado.'))
    } finally {
      setLoading((prev) => ({ ...prev, datasetBuild: false }))
    }
  }

  const handleValidateDataset = async () => {
    setLoading((prev) => ({ ...prev, datasetValidate: true }))
    try {
      const result = await validateHumanDataset(normalizedRun)
      setDatasetValidation(result)
      setNotice(`Validacion dataset: ${result.verdict}.`)
    } catch (err) {
      setError(friendlyError(err, 'No se pudo validar el dataset supervisado.'))
    } finally {
      setLoading((prev) => ({ ...prev, datasetValidate: false }))
    }
  }

  const handlePreviewDataset = async () => {
    setLoading((prev) => ({ ...prev, datasetPreview: true }))
    try {
      setDatasetPreview(await getHumanDatasetPreview(normalizedRun, { limit: 20 }))
    } catch (err) {
      setError(friendlyError(err, 'No se pudo cargar el preview del dataset.'))
    } finally {
      setLoading((prev) => ({ ...prev, datasetPreview: false }))
    }
  }

  const handleTrain = async () => {
    setLoading((prev) => ({ ...prev, training: true }))
    setNotice('Entrenando modelo supervisado. Este proceso puede tardar unos minutos.')
    setError('')
    try {
      const payload = {
        source_run: normalizedRun,
        model_type: modelConfig.model_type,
        test_size: Number(modelConfig.test_size),
        random_state: Number(modelConfig.random_state),
        use_smote: Boolean(modelConfig.use_smote)
      }
      const result = await trainHumanSupervisedModel(payload)
      setSelectedModel(result.model_type || modelConfig.model_type)
      setNotice(`Entrenamiento completado: ${result.verdict}.`)
      await loadBase()
      await loadModelDetails(result.model_type || modelConfig.model_type, 1)
    } catch (err) {
      setError(friendlyError(err, 'No se pudo entrenar el modelo supervisado.'))
    } finally {
      setLoading((prev) => ({ ...prev, training: false }))
    }
  }

  const copyReport = async () => {
    try {
      await navigator.clipboard.writeText(report || '')
      setNotice('Reporte copiado.')
    } catch (_) {
      setNotice('No se pudo copiar el reporte desde este navegador.')
    }
  }

  return (
    <div>
      <div className="header">
        <div>
          <h2>Modelos Supervisados</h2>
          <div className="page-subtitle">Clasificacion de alertas revisadas a partir de etiquetas humanas.</div>
        </div>
      </div>

      <div className="card warning-banner">{METHODOLOGY_MESSAGE}</div>

      <div className="tab-bar" style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button className={activeTab === 'training' ? 'tab active' : 'tab'} onClick={() => setActiveTab('training')}>
          Dataset / Entrenamiento / Métricas
        </button>
        <button className={activeTab === 'apply' ? 'tab active' : 'tab'} onClick={() => setActiveTab('apply')}>
          Aplicar modelo entrenado
        </button>
      </div>

      {activeTab === 'training' && <>
      <div className="card warning-banner">
        Este modulo usa unicamente revisiones humanas. Las reglas, anomaly_flag, autoencoder_anomaly_flag y risk_score pueden ser senales analiticas, pero no etiquetas de fraude. CONFIRMED_FRAUD es clase positiva; DISMISSED es clase negativa; NEW, IN_REVIEW y FALSE_POSITIVE se excluyen del entrenamiento.
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14 }}>
        <div className="card">
          <h3>Que significa este modulo</h3>
          <p>
            El modelo supervisado aprende de revisiones humanas anteriores. Usa CONFIRMED_FRAUD como clase positiva y DISMISSED como clase negativa. Su salida es una prediccion de apoyo para priorizacion, no una confirmacion automatica de fraude.
          </p>
        </div>
        <div className="card">
          <h3>Que debe mirar primero el analista</h3>
          <ul style={{ marginTop: 0, color: 'var(--text-muted)' }}>
            <li>Recall: cuantos fraudes confirmados detecta.</li>
            <li>Precision: cuantas falsas alarmas genera.</li>
            <li>Falsos negativos: casos criticos que el modelo no detecto.</li>
            <li>Falsos positivos: casos priorizados sin confirmacion humana.</li>
          </ul>
        </div>
      </div>

      {error && <div className="detail-status detail-status-error" style={{ padding: 12, borderRadius: 8, marginBottom: 16 }}>{error}</div>}
      {notice && <div className="detail-status detail-status-success" style={{ padding: 12, borderRadius: 8, marginBottom: 16 }}>{notice}</div>}

      <div className="card">
        <div className="form-row" style={{ maxWidth: 420 }}>
          <label htmlFor="supervised_source_run">source_run</label>
          <input id="supervised_source_run" className="input" value={sourceRun} onChange={(event) => setSourceRun(event.target.value)} placeholder={DEFAULT_SOURCE_RUN} />
        </div>
      </div>

      <div className="card">
        <h3>Resumen de etiquetas humanas</h3>
        {loading.base && <div>Cargando resumen de etiquetas humanas...</div>}
        {!loading.base && summary && <div className="kpi-grid">{metrics.map((metric) => <KPICard key={metric.title} title={metric.title} value={metric.value} />)}</div>}
      </div>

      <div className="card">
        <h3>Readiness supervisado</h3>
        <div className={technicalReady ? 'detail-status detail-status-success' : 'detail-status'} style={{ padding: 12, borderRadius: 8, marginBottom: 16 }}>
          <strong>{readiness?.verdict || summary?.verdict || 'INSUFFICIENT_HUMAN_LABELS'}</strong>
          <div style={{ marginTop: 6 }}>{verdictMessage(readiness?.verdict || summary?.verdict)}</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
          <GoalCard title="Minimo tecnico" positiveCurrent={currentPositive} negativeCurrent={currentNegative} positiveRequired={readiness?.requirements?.technical?.positive ?? 20} negativeRequired={readiness?.requirements?.technical?.negative ?? 20} ready={Boolean(readiness?.technical_ready)} />
          <GoalCard title="Recomendado" positiveCurrent={currentPositive} negativeCurrent={currentNegative} positiveRequired={readiness?.requirements?.recommended?.positive ?? 50} negativeRequired={readiness?.requirements?.recommended?.negative ?? 120} ready={Boolean(readiness?.recommended_ready)} />
          <GoalCard title="Meta fuerte" positiveCurrent={currentPositive} negativeCurrent={currentNegative} positiveRequired={readiness?.requirements?.strong?.positive ?? 70} negativeRequired={readiness?.requirements?.strong?.negative ?? 180} ready={Boolean(readiness?.strong_ready)} />
        </div>
      </div>

      <div className="card">
        <h3>Dataset supervisado humano</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 12 }}>
          <KPICard title="Dataset existe" value={preflight?.dataset?.exists || datasetSummary?.exists ? 'Si' : 'No'} />
          <KPICard title="Filas" value={formatCount(preflight?.dataset?.rows ?? datasetSummary?.rows)} />
          <KPICard title="Positivos" value={formatCount(preflight?.dataset?.positive_count ?? datasetSummary?.positives)} />
          <KPICard title="Negativos" value={formatCount(preflight?.dataset?.negative_count ?? datasetSummary?.negatives)} />
        </div>
        <div style={{ display: 'grid', gap: 8, marginTop: 16 }}>
          <div>Archivo: <strong>{preflight?.dataset?.file || datasetSummary?.dataset_file || 'No disponible'}</strong></div>
          <div>Verdict: <StatusBadge tone={datasetReady ? 'success' : 'warning'}>{datasetValidation?.verdict || preflight?.dataset?.verdict || datasetSummary?.verdict || 'No validado'}</StatusBadge></div>
          <div>supervised_dataset_runs: <StatusBadge tone={preflight?.supervised_dataset_runs?.registered ? 'success' : 'warning'}>{preflight?.supervised_dataset_runs?.status || datasetSummary?.status || 'No registrado'}</StatusBadge></div>
          <div>artifact_registry: <StatusBadge tone={preflight?.artifact_registry?.supervised_dataset_registered ? 'success' : 'warning'}>{preflight?.artifact_registry?.supervised_dataset_registered ? 'Registrado' : 'No registrado'}</StatusBadge></div>
          <div>Creado: {datasetSummary?.created_at || 'No disponible'}</div>
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 16 }}>
          <button className="button" onClick={handleBuildDataset} disabled={loading.datasetBuild}>{loading.datasetBuild ? 'Construyendo...' : 'Construir dataset supervisado'}</button>
          <button className="button button-secondary" onClick={handleValidateDataset} disabled={loading.datasetValidate}>Validar dataset</button>
          <button className="button button-secondary" onClick={handlePreviewDataset} disabled={loading.datasetPreview}>Ver preview</button>
        </div>
        {datasetPreview?.rows && <div style={{ marginTop: 16 }}><DataTable rows={datasetPreview.rows} emptyText="No hay filas para preview." /></div>}
      </div>

      <div className="card">
        <h3>Entrenar modelo supervisado</h3>
        <p style={{ color: 'var(--text-muted)' }}>Seleccione un modelo y entrene con el dataset supervisado validado.</p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
          <div className="form-row">
            <label htmlFor="supervised_model">Modelo</label>
            <select id="supervised_model" className="input" value={modelConfig.model_type} onChange={(event) => setModelConfig((prev) => ({ ...prev, model_type: event.target.value }))}>
              {MODEL_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
            </select>
          </div>
          <div className="form-row">
            <label htmlFor="supervised_test_size">test_size</label>
            <input id="supervised_test_size" className="input" value={modelConfig.test_size} onChange={(event) => setModelConfig((prev) => ({ ...prev, test_size: event.target.value }))} />
          </div>
          <div className="form-row">
            <label htmlFor="supervised_random_state">random_state</label>
            <input id="supervised_random_state" className="input" value={modelConfig.random_state} onChange={(event) => setModelConfig((prev) => ({ ...prev, random_state: event.target.value }))} />
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 24 }}>
            <input type="checkbox" checked={modelConfig.use_smote} onChange={(event) => setModelConfig((prev) => ({ ...prev, use_smote: event.target.checked }))} />
            use_smote
          </label>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10, marginTop: 14 }}>
          {MODEL_OPTIONS.map((option) => (
            <div key={option.value} className="card" style={{ margin: 0, padding: 12 }}>
              <strong>{MODEL_LABELS[option.value]}</strong>
              {option.value === 'mlp_classifier' && !recommendedReady && <div style={{ marginTop: 6 }}><StatusBadge tone="warning">Bloqueado hasta meta recomendada</StatusBadge></div>}
              <p style={{ color: 'var(--text-muted)', margin: '8px 0 0', fontSize: 13 }}>{MODEL_DESCRIPTIONS[option.value]}</p>
            </div>
          ))}
        </div>
        {mlpBlocked && <div className="detail-status" style={{ padding: 12, borderRadius: 8, marginTop: 12 }}>MLP requiere meta recomendada de etiquetas humanas para evitar sobreajuste.</div>}
        {!technicalReady && <div className="detail-status" style={{ padding: 12, borderRadius: 8, marginTop: 12 }}>Requiere al menos 20 positivos y 20 negativos humanos.</div>}
        {technicalReady && !datasetReady && <div className="detail-status" style={{ padding: 12, borderRadius: 8, marginTop: 12 }}>Dataset inexistente o invalido. Valida el dataset antes de entrenar.</div>}
        <button className="button" style={{ marginTop: 16 }} disabled={!canTrain} onClick={handleTrain}>
          {loading.training ? 'Entrenando modelo supervisado...' : 'Entrenar modelo supervisado'}
        </button>
      </div>

      <div className="card">
        <h3>Modelos supervisados entrenados</h3>
        <DataTable rows={trainingRuns.map((run) => ({
          algoritmo: run.algorithm,
          source_run: run.source_run,
          status: run.status,
          accuracy: formatMetric(run.metrics?.accuracy ?? run.metrics_json?.accuracy),
          precision: formatMetric(run.metrics?.precision ?? run.metrics_json?.precision),
          recall: formatMetric(run.metrics?.recall ?? run.metrics_json?.recall),
          f1_score: formatMetric(run.metrics?.f1_score ?? run.metrics_json?.f1_score),
          roc_auc: formatMetric(run.metrics?.roc_auc ?? run.metrics_json?.roc_auc),
          created_at: run.created_at,
          is_active: String(Boolean(run.is_active))
        }))} emptyText="No hay modelos supervisados entrenados." />
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 14 }}>
          {MODEL_OPTIONS.filter((option) => option.value !== 'mlp_classifier').map((option) => (
            <button key={option.value} className={selectedModel === option.value ? 'button' : 'button button-secondary'} onClick={() => setSelectedModel(option.value)}>
              Ver resultados {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>Comparacion de modelos supervisados</h3>
        <p style={{ color: 'var(--text-muted)' }}>Estas metricas comparan el comportamiento de cada modelo. En fraude financiero, precision y recall son mas importantes que accuracy por si sola.</p>
        <MetricInfoCards />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14, marginTop: 14 }}>
          <MiniBarChart title="Precision, Recall y F1-score" data={chartData} metrics={['Precision', 'Recall', 'F1-score']} />
          <MiniBarChart title="Accuracy y ROC-AUC" data={chartData} metrics={['Accuracy', 'ROC-AUC']} />
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', margin: '14px 0' }}>
          {bestModel && <StatusBadge tone="success">Mayor F1-score: {MODEL_LABELS[bestModel.algorithm] || bestModel.algorithm}</StatusBadge>}
          {bestRecall && <StatusBadge tone="neutral">Mayor recall: {MODEL_LABELS[bestRecall.algorithm] || bestRecall.algorithm}</StatusBadge>}
          {bestPrecision && <StatusBadge tone="neutral">Mayor precision: {MODEL_LABELS[bestPrecision.algorithm] || bestPrecision.algorithm}</StatusBadge>}
        </div>
        <div className="detail-status" style={{ padding: 12, borderRadius: 8, marginBottom: 12 }}>
          Las metricas son preliminares porque el dataset etiquetado actual cumple el minimo tecnico, pero aun no alcanza la meta recomendada.
        </div>
        <DataTable rows={trainingRuns.map((run) => {
          const matrix = run.metrics?.confusion_matrix || run.metrics_json?.confusion_matrix || [[0, 0], [0, 0]]
          return {
            Modelo: run.algorithm,
            Accuracy: formatMetric(run.metrics?.accuracy ?? run.metrics_json?.accuracy),
            Precision: formatMetric(run.metrics?.precision ?? run.metrics_json?.precision),
            Recall: formatMetric(run.metrics?.recall ?? run.metrics_json?.recall),
            'F1-score': formatMetric(run.metrics?.f1_score ?? run.metrics_json?.f1_score),
            'ROC-AUC': formatMetric(run.metrics?.roc_auc ?? run.metrics_json?.roc_auc),
            'Falsos positivos': matrix?.[0]?.[1] ?? 0,
            'Falsos negativos': matrix?.[1]?.[0] ?? 0
          }
        })} />
      </div>

      <div className="card">
        <h3>Resultado del modelo seleccionado</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: 16 }}>
          <KPICard title="Modelo" value={MODEL_LABELS[selectedModel] || selectedModel} />
          <KPICard title="Estado" value={selectedRun?.status || metadata?.status || 'No disponible'} />
          <KPICard title="Filas usadas" value={formatCount(metadata?.total_rows)} />
          <KPICard title="Positivos / negativos" value={`${formatCount(metadata?.positive_count)} / ${formatCount(metadata?.negative_count)}`} />
          <KPICard title="Test size" value={formatMetric(metadata?.test_size)} />
          <KPICard title="Fecha entrenamiento" value={metadata?.created_at || selectedRun?.created_at || 'No disponible'} />
        </div>
        <MetricInfoCards />
      </div>

      <div className="card">
        <h3>Matriz de confusion visual</h3>
        <ConfusionMatrix matrix={selectedMetrics?.confusion_matrix} />
      </div>

      <div className="card">
        <h3>Distribucion de predicciones</h3>
        <PredictionDistribution rows={visiblePredictionRows} />
      </div>

      <div className="card">
        <h3>Predicciones de evaluacion</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 10, marginBottom: 12 }}>
          <select className="input" aria-label="filtro rapido" value={predictionFilters.quick_filter} onChange={(event) => setPredictionFilters((prev) => ({ ...prev, quick_filter: event.target.value }))}>
            <option value="">Todos</option>
            <option value="hits">Aciertos</option>
            <option value="errors">Errores</option>
            <option value="false_positive">Falsos positivos</option>
            <option value="false_negative">Falsos negativos</option>
            <option value="predicted_positive">Predichos positivos</option>
            <option value="predicted_negative">Predichos negativos</option>
          </select>
          <select className="input" aria-label="evaluation_result" value={predictionFilters.evaluation_result} onChange={(event) => setPredictionFilters((prev) => ({ ...prev, evaluation_result: event.target.value }))}>
            <option value="">Todos los resultados</option>
            <option value="TRUE_POSITIVE">TRUE_POSITIVE</option>
            <option value="TRUE_NEGATIVE">TRUE_NEGATIVE</option>
            <option value="FALSE_POSITIVE">FALSE_POSITIVE</option>
            <option value="FALSE_NEGATIVE">FALSE_NEGATIVE</option>
          </select>
          <select className="input" aria-label="y_true" value={predictionFilters.y_true} onChange={(event) => setPredictionFilters((prev) => ({ ...prev, y_true: event.target.value }))}>
            <option value="">y_true</option>
            <option value="1">1</option>
            <option value="0">0</option>
          </select>
          <select className="input" aria-label="y_pred" value={predictionFilters.y_pred} onChange={(event) => setPredictionFilters((prev) => ({ ...prev, y_pred: event.target.value }))}>
            <option value="">y_pred</option>
            <option value="1">1</option>
            <option value="0">0</option>
          </select>
          <select className="input" aria-label="prediction_label" value={predictionFilters.prediction_label} onChange={(event) => setPredictionFilters((prev) => ({ ...prev, prediction_label: event.target.value }))}>
            <option value="">prediction_label</option>
            <option value="CONFIRMED_FRAUD">CONFIRMED_FRAUD</option>
            <option value="DISMISSED">DISMISSED</option>
          </select>
          <button className="button button-secondary" onClick={() => loadModelDetails(selectedModel, 1)}>Aplicar filtros</button>
        </div>
        <FriendlyPredictionsTable rows={visiblePredictionRows} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 12 }}>
          <button className="button button-secondary" disabled={(predictions.page || 1) <= 1} onClick={() => loadModelDetails(selectedModel, (predictions.page || 1) - 1)}>Anterior</button>
          <span>Pagina {predictions.page || 1} de {predictions.total_pages || 1}</span>
          <button className="button button-secondary" disabled={(predictions.page || 1) >= (predictions.total_pages || 1)} onClick={() => loadModelDetails(selectedModel, (predictions.page || 1) + 1)}>Siguiente</button>
        </div>
      </div>

      <div className="card">
        <h3>Detalles tecnicos</h3>
        <details>
          <summary>Ver reporte tecnico</summary>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', margin: '12px 0' }}>
            <button className="button button-secondary" onClick={() => loadModelDetails(selectedModel, predictions.page || 1)}>Refrescar</button>
            <button className="button button-secondary" onClick={copyReport} disabled={!report}>Copiar contenido</button>
          </div>
          <pre style={{ whiteSpace: 'pre-wrap', background: 'rgba(4, 12, 22, 0.45)', padding: 14, borderRadius: 8, maxHeight: 360, overflow: 'auto' }}>
            {report || 'No existe reporte para este modelo.'}
          </pre>
        </details>
        <details style={{ marginTop: 12 }}>
          <summary>Ver metadata del modelo</summary>
          <pre style={{ whiteSpace: 'pre-wrap', background: 'rgba(4, 12, 22, 0.45)', padding: 14, borderRadius: 8, maxHeight: 420, overflow: 'auto' }}>
            {metadata ? JSON.stringify({ ...metadata, use_smote: metadata.use_smote ?? modelConfig.use_smote }, null, 2) : 'No existe metadata para este modelo.'}
          </pre>
        </details>
        <details style={{ marginTop: 12 }}>
          <summary>Ver archivos generados</summary>
          <DataTable rows={metadata ? [{
            model_file: metadata.model_file,
            metadata_file: metadata.metadata_file,
            report_file: metadata.report_file,
            predictions_file: metadata.predictions_file
          }] : []} emptyText="No hay archivos registrados para este modelo." />
        </details>
      </div>

      <div className="card">
        <h3>Estado metodologico</h3>
        <p>{METHODOLOGY_MESSAGE}</p>
        <p>MLP queda bloqueado mientras no exista la meta recomendada de 50 CONFIRMED_FRAUD y 120 DISMISSED.</p>
      </div>
      </>}

      {activeTab === 'apply' && (
        <div>
          <div className="card warning-banner" style={{ background: '#3d2e00', color: '#ffe082', marginBottom: 16 }}>
            Las predicciones generadas por modelos supervisados son apoyo analítico y no constituyen fraude confirmado automático. No se reentrenan modelos.
          </div>

          {/* Selector de modelo entrenado */}
          <div className="card">
            <h3>1. Seleccionar modelo entrenado</h3>
            {supTrainedModels.length === 0
              ? <div className="detail-status" style={{ padding: 12, borderRadius: 8 }}>No hay modelos supervisados AVAILABLE. Entrene un modelo primero en la pestaña anterior.</div>
              : (
                <div style={{ overflowX: 'auto' }}>
                  <table className="table">
                    <thead>
                      <tr>
                        <th></th>
                        <th>Algoritmo</th>
                        <th>source_run</th>
                        <th>F1-score</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>ROC-AUC</th>
                        <th>Estado</th>
                        <th>Fecha</th>
                      </tr>
                    </thead>
                    <tbody>
                      {supTrainedModels.map((m) => (
                        <tr key={m.id} style={{ cursor: 'pointer', background: supSelectedModelId === String(m.id) ? 'rgba(56,214,214,0.08)' : undefined }} onClick={() => setSupSelectedModelId(String(m.id))}>
                          <td><input type="radio" readOnly checked={supSelectedModelId === String(m.id)} aria-label={`Seleccionar ${m.algorithm}`} /></td>
                          <td><strong>{MODEL_LABELS[m.algorithm] || m.algorithm}</strong></td>
                          <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{m.source_run}</td>
                          <td>{m.f1_score != null ? Number(m.f1_score).toFixed(4) : 'N/A'}</td>
                          <td>{m.precision != null ? Number(m.precision).toFixed(4) : 'N/A'}</td>
                          <td>{m.recall != null ? Number(m.recall).toFixed(4) : 'N/A'}</td>
                          <td>{m.roc_auc != null ? Number(m.roc_auc).toFixed(4) : 'N/A'}</td>
                          <td><StatusBadge tone="success">{m.status}</StatusBadge></td>
                          <td style={{ fontSize: 12 }}>{m.created_at ? m.created_at.slice(0, 10) : ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )
            }
          </div>

          {/* Selector de dataset */}
          <div className="card">
            <h3>2. Origen del dataset de alertas</h3>
            <div style={{ display: 'flex', gap: 16, marginBottom: 14 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                <input type="radio" value="preprocessed_run" checked={supInputType === 'preprocessed_run'} onChange={() => setSupInputType('preprocessed_run')} />
                Usar alertas generadas para un preprocessed_run
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                <input type="radio" value="csv_upload" checked={supInputType === 'csv_upload'} onChange={() => setSupInputType('csv_upload')} />
                Subir CSV de resumen de alertas
              </label>
            </div>
            {supInputType === 'preprocessed_run' && (
              <div>
                <select className="input" value={supPreprocessedRunId} onChange={(e) => setSupPreprocessedRunId(e.target.value)} style={{ maxWidth: 540 }}>
                  <option value="">— Seleccione un preprocessed_run —</option>
                  {supPreprocessedRuns.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.source_run || `Run #${r.id}`} — {r.total_records != null ? `${r.total_records.toLocaleString()} registros` : ''} — {r.finished_at?.slice(0, 10)}
                    </option>
                  ))}
                </select>
                {supPreprocessedRuns.length === 0 && (
                  <div style={{ marginTop: 8, color: 'var(--text-muted)', fontSize: 13 }}>
                    No hay preprocessing runs completados disponibles.
                  </div>
                )}
                <div style={{ marginTop: 8, color: 'var(--text-muted)', fontSize: 12 }}>
                  El backend buscará automáticamente las alertas generadas por el motor de reglas para este run. Si no existen, se mostrará un error con instrucciones.
                </div>
              </div>
            )}
            {supInputType === 'csv_upload' && (
              <div>
                <input type="file" accept=".csv" className="input" onChange={(e) => setSupCsvFile(e.target.files?.[0] || null)} style={{ maxWidth: 480 }} />
                {supCsvFile && <div style={{ marginTop: 8, color: 'var(--text-muted)', fontSize: 13 }}>Archivo seleccionado: {supCsvFile.name}</div>}
                <div style={{ marginTop: 6, color: 'var(--text-muted)', fontSize: 12 }}>
                  El CSV debe tener esquema de resumen de alertas: rule_code, rule_name, summary_alert_id, countries_detected, merchant_rubro_values, etc. No se aceptan CSVs de transacciones crudas.
                </div>
              </div>
            )}
          </div>

          {/* Botón Aplicar */}
          <div className="card">
            <h3>3. Aplicar modelo</h3>
            {supApplyError && <div className="detail-status detail-status-error" style={{ padding: 12, borderRadius: 8, marginBottom: 12 }}>{supApplyError}</div>}
            {supApplySuccess && <div className="detail-status detail-status-success" style={{ padding: 12, borderRadius: 8, marginBottom: 12 }}>{supApplySuccess}</div>}
            {supApplyLoading && (
              <div style={{ marginBottom: 12, color: 'var(--text-muted)' }}>
                Procesando... estado: <strong>{supPollStatus}</strong>. El análisis puede tardar varios minutos según el tamaño del dataset.
              </div>
            )}
            <button className="button" onClick={handleSupApplyModel} disabled={supApplyLoading || !supSelectedModelId}>
              {supApplyLoading ? 'Aplicando modelo...' : 'Aplicar modelo entrenado'}
            </button>
          </div>

          {/* Historial de runs */}
          {supPredRuns.length > 0 && (
            <div className="card">
              <h3>Historial de ejecuciones</h3>
              <div style={{ overflowX: 'auto' }}>
                <table className="table">
                  <thead>
                    <tr><th>ID</th><th>Algoritmo</th><th>Dataset</th><th>Total</th><th>HIGH</th><th>MEDIUM</th><th>LOW</th><th>Estado</th><th>Fecha</th><th></th></tr>
                  </thead>
                  <tbody>
                    {supPredRuns.map((r) => (
                      <tr key={r.id} style={{ background: supSelectedPredRunId === String(r.id) ? 'rgba(56,214,214,0.08)' : undefined }}>
                        <td>{r.id}</td>
                        <td>{MODEL_LABELS[r.algorithm] || r.algorithm}</td>
                        <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{r.input_source}</td>
                        <td>{r.total_analyzed?.toLocaleString()}</td>
                        <td style={{ color: '#ff4757' }}>{r.high_count?.toLocaleString()}</td>
                        <td style={{ color: '#f7b955' }}>{r.medium_count?.toLocaleString()}</td>
                        <td style={{ color: '#2ed573' }}>{r.low_count?.toLocaleString()}</td>
                        <td><StatusBadge tone={r.status === 'COMPLETED' ? 'success' : r.status === 'FAILED' ? 'error' : 'neutral'}>{r.status}</StatusBadge></td>
                        <td style={{ fontSize: 12 }}>{r.finished_at?.slice(0, 10)}</td>
                        <td>{r.status === 'COMPLETED' && <button className="button button-secondary" style={{ padding: '4px 10px', fontSize: 12 }} onClick={() => handleSupSelectRun(String(r.id))}>Ver</button>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Resultados */}
          {supSelectedPredRunId && supPredResults.total >= 0 && (() => {
            const run = supPredRuns.find((r) => String(r.id) === supSelectedPredRunId)
            return (
              <div>
                {supSameRunWarning && (
                  <div className="card warning-banner" style={{ background: '#1a2a1a', color: '#81c784', marginBottom: 16 }}>
                    {supSameRunWarning}
                  </div>
                )}
                {/* KPI cards */}
                <div className="kpi-grid" style={{ marginBottom: 16 }}>
                  <KPICard title="Total analizados" value={(run?.total_analyzed ?? supPredResults.total)?.toLocaleString()} />
                  <KPICard title="Prioridad HIGH" value={(run?.high_count ?? 0)?.toLocaleString()} />
                  <KPICard title="Prioridad MEDIUM" value={(run?.medium_count ?? 0)?.toLocaleString()} />
                  <KPICard title="Prioridad LOW" value={(run?.low_count ?? 0)?.toLocaleString()} />
                  <KPICard title="Modelo aplicado" value={MODEL_LABELS[run?.algorithm] || run?.algorithm || '—'} />
                  <KPICard title="Dataset usado" value={run?.input_source || '—'} />
                </div>

                {/* Distribución por prioridad */}
                {supPredReport?.priority_distribution?.length > 0 && (
                  <div className="card" style={{ marginBottom: 16 }}>
                    <h4 style={{ marginTop: 0 }}>Distribución por nivel de prioridad</h4>
                    <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap' }}>
                      {supPredReport.priority_distribution.map((item) => {
                        const total = supPredReport.priority_distribution.reduce((s, x) => s + x.count, 0)
                        const pct = total > 0 ? Math.round((item.count / total) * 100) : 0
                        const cfg = { HIGH: { color: '#ff4757', label: 'HIGH — Revisión prioritaria' }, MEDIUM: { color: '#f7b955', label: 'MEDIUM — Revisión normal' }, LOW: { color: '#2ed573', label: 'LOW — Baja prioridad' } }
                        const c = cfg[item.level] || { color: '#747d8c', label: item.level }
                        return (
                          <div key={item.level} style={{ flex: '1 1 160px', textAlign: 'center' }}>
                            <div style={{ height: 80, background: 'rgba(4,12,22,0.45)', borderRadius: 8, overflow: 'hidden', display: 'flex', alignItems: 'flex-end' }}>
                              <div style={{ width: '100%', height: `${pct}%`, background: c.color, transition: 'height 0.4s' }} />
                            </div>
                            <div style={{ marginTop: 6, fontSize: 13, fontWeight: 700, color: c.color }}>{item.level}</div>
                            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{item.count?.toLocaleString()} ({pct}%)</div>
                            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{c.label}</div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Tabla de resultados con filtros */}
                <div className="card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 14 }}>
                    <h4 style={{ margin: 0 }}>Resultados — {supPredResults.total?.toLocaleString()} registros {supPriorityFilter !== 'ALL' ? `(filtro: ${supPriorityFilter})` : ''}</h4>
                    <div style={{ display: 'flex', gap: 8 }}>
                      {['ALL', 'HIGH', 'MEDIUM', 'LOW'].map((f) => {
                        const colors = { HIGH: '#ff4757', MEDIUM: '#f7b955', LOW: '#2ed573', ALL: 'var(--accent-blue)' }
                        return (
                          <button key={f} onClick={() => handleSupFilterChange(f)} style={{ padding: '4px 12px', borderRadius: 20, border: `1px solid ${supPriorityFilter === f ? colors[f] : 'rgba(255,255,255,0.12)'}`, background: supPriorityFilter === f ? `${colors[f]}22` : 'transparent', color: supPriorityFilter === f ? colors[f] : 'var(--text-muted)', cursor: 'pointer', fontSize: 13 }}>
                            {f}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                  {supPredResults.rows?.length === 0
                    ? <div className="detail-status" style={{ padding: 12, borderRadius: 8 }}>No hay resultados para este filtro.</div>
                    : (
                      <div style={{ overflowX: 'auto' }}>
                        <table className="table">
                          <thead>
                            <tr>
                              <th>ID Alerta</th>
                              <th>Cliente</th>
                              <th>Prob. fraude</th>
                              <th>Prioridad revisión</th>
                              <th>Predicción</th>
                              <th>Modelo</th>
                              <th>source_run</th>
                            </tr>
                          </thead>
                          <tbody>
                            {cleanRows(supPredResults.rows).map((row, idx) => {
                              const priorityColors = { HIGH: '#ff4757', MEDIUM: '#f7b955', LOW: '#2ed573' }
                              const pc = priorityColors[row.priority_level] || '#747d8c'
                              return (
                                <tr key={idx}>
                                  <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{String(row.summary_alert_id ?? row.transaction_id ?? '')}</td>
                                  <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{String(row.customer_hash ?? '')}</td>
                                  <td style={{ textAlign: 'right' }}>{row.prediction_probability != null ? Number(row.prediction_probability).toFixed(4) : 'N/A'}</td>
                                  <td>
                                    <span style={{ background: `${pc}22`, color: pc, border: `1px solid ${pc}55`, borderRadius: 12, padding: '2px 10px', fontWeight: 700, fontSize: 12 }}>
                                      {row.priority_level}
                                    </span>
                                  </td>
                                  <td>{Number(row.prediction_label) === 1 ? 'Sospechoso' : 'Normal'}</td>
                                  <td style={{ fontSize: 12 }}>{MODEL_LABELS[row.model_name] || row.model_name}</td>
                                  <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{row.source_run ?? ''}</td>
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                        {supPredResults.total > SUP_PAGE_SIZE && (
                          <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 12 }}>
                            <button className="button button-secondary" disabled={supPredPage <= 1} onClick={async () => { const p = supPredPage - 1; setSupPredPage(p); await loadSupPredResults(supSelectedPredRunId, p, supPriorityFilter) }}>← Anterior</button>
                            <span style={{ alignSelf: 'center', color: 'var(--text-muted)', fontSize: 13 }}>Pág. {supPredPage} · {supPredResults.total?.toLocaleString()} total</span>
                            <button className="button button-secondary" disabled={supPredPage * SUP_PAGE_SIZE >= supPredResults.total} onClick={async () => { const p = supPredPage + 1; setSupPredPage(p); await loadSupPredResults(supSelectedPredRunId, p, supPriorityFilter) }}>Siguiente →</button>
                          </div>
                        )}
                      </div>
                    )
                  }
                </div>

                {/* Guía de interpretación */}
                <div className="card" style={{ marginTop: 0 }}>
                  <h4 style={{ marginTop: 0 }}>Guía de revisión de alertas para el analista</h4>
                  <p style={{ color: 'var(--text-muted)', marginTop: 0, fontSize: 13 }}>
                    HIGH, MEDIUM y LOW indican prioridad de revisión de alerta, no fraude confirmado automático.
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
                    {[
                      { level: 'HIGH', color: '#ff4757', title: 'Prioridad alta', desc: 'Prob. ≥ 70%. Revisar primero. El modelo tiene alta confianza en comportamiento sospechoso. Requiere revisión analítica.' },
                      { level: 'MEDIUM', color: '#f7b955', title: 'Prioridad media', desc: 'Prob. 40–70%. Revisar en segundo turno. Señales mixtas; requiere criterio analítico antes de actuar.' },
                      { level: 'LOW', color: '#2ed573', title: 'Baja prioridad', desc: 'Prob. < 40%. Comportamiento típico según el modelo. Revisar si el volumen lo permite.' },
                    ].map((item) => (
                      <div key={item.level} className="card" style={{ margin: 0, borderLeft: `3px solid ${item.color}` }}>
                        <span style={{ color: item.color, fontWeight: 700 }}>{item.title}</span>
                        <p style={{ margin: '6px 0 0', fontSize: 13, color: 'var(--text-muted)' }}>{item.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )
          })()}
        </div>
      )}
    </div>
  )
}
