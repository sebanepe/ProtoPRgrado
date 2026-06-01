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
  validateHumanDataset
} from '../services/api'

const DEFAULT_SOURCE_RUN = 'preprocessed_run_26'
const MODEL_OPTIONS = [
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'logistic_regression', label: 'Logistic Regression' },
  { value: 'gradient_boosting', label: 'Gradient Boosting' },
  { value: 'mlp_classifier', label: 'MLP Classifier' }
]
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

function ConfusionMatrix({ matrix }) {
  const safe = Array.isArray(matrix) ? matrix : [[0, 0], [0, 0]]
  const cells = [
    ['Verdadero Negativo', safe?.[0]?.[0] ?? 0, 'success'],
    ['Falso Positivo', safe?.[0]?.[1] ?? 0, 'warning'],
    ['Falso Negativo', safe?.[1]?.[0] ?? 0, 'error'],
    ['Verdadero Positivo', safe?.[1]?.[1] ?? 0, 'success']
  ]
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(130px, 1fr))', gap: 10, maxWidth: 520 }}>
        {cells.map(([label, value, tone]) => (
          <div key={label} className="card" style={{ margin: 0, padding: 14 }}>
            <StatusBadge tone={tone}>{label}</StatusBadge>
            <div style={{ fontSize: 28, fontWeight: 800, marginTop: 10 }}>{value}</div>
          </div>
        ))}
      </div>
      <p style={{ color: 'var(--text-muted)' }}>
        Formato [[TN, FP], [FN, TP]]. Falso positivo: el modelo priorizo como positivo una alerta DISMISSED. Falso negativo: el modelo descarto una alerta CONFIRMED_FRAUD.
      </p>
    </div>
  )
}

export default function ModelSupervised() {
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
  const [predictionFilters, setPredictionFilters] = useState({ evaluation_result: '', y_true: '', y_pred: '', prediction_label: '' })
  const [loading, setLoading] = useState({})
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [modelConfig, setModelConfig] = useState({ model_type: 'random_forest', test_size: '0.2', random_state: '42', use_smote: false })

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
      if (value !== '') params[key] = value
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
      <div className="card warning-banner">
        Este modulo usa unicamente revisiones humanas. Las reglas, anomaly_flag, autoencoder_anomaly_flag y risk_score pueden ser senales analiticas, pero no etiquetas de fraude. CONFIRMED_FRAUD es clase positiva; DISMISSED es clase negativa; NEW, IN_REVIEW y FALSE_POSITIVE se excluyen del entrenamiento.
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
        {bestModel && <div style={{ marginBottom: 12 }}>Mejor modelo por F1-score: <strong>{bestModel.algorithm}</strong></div>}
        <div className="detail-status" style={{ padding: 12, borderRadius: 8, marginBottom: 12 }}>
          Las metricas son preliminares porque el conjunto etiquetado actual corresponde al minimo tecnico. Recall alto reduce perdida de positivos confirmados; precision alta reduce alertas falsas para el analista. Accuracy no debe ser la unica metrica principal.
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
        <h3>Matriz de confusion</h3>
        <ConfusionMatrix matrix={selectedMetrics?.confusion_matrix} />
      </div>

      <div className="card">
        <h3>Reporte del modelo</h3>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 12 }}>
          <button className="button button-secondary" onClick={() => loadModelDetails(selectedModel, predictions.page || 1)}>Refrescar</button>
          <button className="button button-secondary" onClick={copyReport} disabled={!report}>Copiar contenido</button>
        </div>
        <pre style={{ whiteSpace: 'pre-wrap', background: 'rgba(4, 12, 22, 0.45)', padding: 14, borderRadius: 8, maxHeight: 360, overflow: 'auto' }}>
          {report || 'No existe reporte para este modelo.'}
        </pre>
      </div>

      <div className="card">
        <h3>Metadata del modelo</h3>
        <pre style={{ whiteSpace: 'pre-wrap', background: 'rgba(4, 12, 22, 0.45)', padding: 14, borderRadius: 8, maxHeight: 420, overflow: 'auto' }}>
          {metadata ? JSON.stringify({ ...metadata, use_smote: metadata.use_smote ?? modelConfig.use_smote }, null, 2) : 'No existe metadata para este modelo.'}
        </pre>
      </div>

      <div className="card">
        <h3>Predicciones de evaluacion</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 10, marginBottom: 12 }}>
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
        <DataTable rows={predictions.rows || []} emptyText="No existen predicciones para este modelo." />
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 12 }}>
          <button className="button button-secondary" disabled={(predictions.page || 1) <= 1} onClick={() => loadModelDetails(selectedModel, (predictions.page || 1) - 1)}>Anterior</button>
          <span>Pagina {predictions.page || 1} de {predictions.total_pages || 1}</span>
          <button className="button button-secondary" disabled={(predictions.page || 1) >= (predictions.total_pages || 1)} onClick={() => loadModelDetails(selectedModel, (predictions.page || 1) + 1)}>Siguiente</button>
        </div>
      </div>

      <div className="card">
        <h3>Estado metodologico</h3>
        <p>{METHODOLOGY_MESSAGE}</p>
        <p>MLP queda bloqueado mientras no exista la meta recomendada de 50 CONFIRMED_FRAUD y 120 DISMISSED.</p>
      </div>
    </div>
  )
}
