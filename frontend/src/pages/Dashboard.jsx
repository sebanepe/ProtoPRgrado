import React, { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import KPICard from '../components/KPICard'
import {
  getAlertReviews,
  getAnomalyRuns,
  getBatchScoringRuns,
  getCasesSummary,
  getDashboardOverview,
  getHumanLabelSummary,
  getModelEvaluationSummary,
  getPreprocessedRuns,
  getRulesAlerts,
  getRulesSummary,
  getSupervisedTrainingRuns,
} from '../services/api'

const SOURCE_RUN = 'preprocessed_run_26'
const ANOMALY_RUN = 'run_26'

const REVIEW_STATES = ['NEW', 'IN_REVIEW', 'DISMISSED', 'FALSE_POSITIVE', 'CONFIRMED_FRAUD']
const CASE_STATES = ['OPEN', 'IN_ANALYSIS', 'ESCALATED', 'CLOSED']
const CASE_PRIORITIES = ['HIGH', 'CRITICAL']
const SCORE_LEVELS = ['HIGH', 'MEDIUM', 'LOW']

const MODEL_ROWS = [
  { key: 'isolation_forest', label: 'Isolation Forest' },
  { key: 'autoencoder_pytorch', label: 'Autoencoder PyTorch' },
  { key: 'logistic_regression', label: 'Logistic Regression' },
  { key: 'random_forest', label: 'Random Forest' },
  { key: 'gradient_boosting', label: 'Gradient Boosting' },
]

const numberFormatter = new Intl.NumberFormat('es-BO')

function formatNumber(value) {
  if (value === null || value === undefined || value === '') return 'Sin datos'
  const n = Number(value)
  return Number.isNaN(n) ? String(value) : numberFormatter.format(n)
}

function formatShort(value) {
  if (value === null || value === undefined || value === '') return '--'
  const n = Number(value)
  return Number.isNaN(n) ? String(value) : numberFormatter.format(n)
}

function formatDate(value) {
  if (!value) return '--'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return String(value)
  return parsed.toLocaleString('es-BO', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function normalizeItems(payload) {
  if (!payload) return []
  if (Array.isArray(payload)) return payload
  return payload.items || payload.runs || payload.results || []
}

function pickTotal(payload) {
  if (!payload) return null
  return payload.total_items ?? payload.total ?? payload.count ?? normalizeItems(payload).length
}

function resultValue(result, fallback = null) {
  return result && result.status === 'fulfilled' ? result.value : fallback
}

function buildReviewDistribution(overview, humanSummary) {
  const raw = {
    NEW: humanSummary?.new ?? overview?.review_distribution?.new,
    IN_REVIEW: humanSummary?.in_review ?? overview?.review_distribution?.in_review,
    DISMISSED: humanSummary?.dismissed ?? overview?.review_distribution?.dismissed,
    FALSE_POSITIVE: humanSummary?.false_positive_excluded ?? humanSummary?.false_positive ?? overview?.review_distribution?.false_positive,
    CONFIRMED_FRAUD: humanSummary?.confirmed_fraud ?? overview?.review_distribution?.confirmed_fraud,
  }
  return REVIEW_STATES.reduce((acc, key) => ({ ...acc, [key]: Math.max(Number(raw[key]) || 0, 0) }), {})
}

function buildModelRows({ anomalyRuns, supervisedRuns, evaluation }) {
  const anomalyItems = normalizeItems(anomalyRuns)
  const supervisedItems = normalizeItems(supervisedRuns)
  const available = new Map()

  anomalyItems.forEach((run) => {
    const key = run.algorithm || run.model_name
    available.set(key, {
      available: true,
      source_run: run.source_run,
      created_at: run.created_at,
      status: run.status || 'AVAILABLE',
      metric: run.anomaly_count != null ? `Anomalías: ${formatShort(run.anomaly_count)}` : '--',
    })
  })

  supervisedItems.forEach((run) => {
    available.set(run.algorithm, {
      available: true,
      source_run: run.source_run,
      created_at: run.created_at,
      status: run.status || 'AVAILABLE',
      metric: run.metrics?.f1_score != null ? `F1: ${Number(run.metrics.f1_score).toFixed(3)}` : '--',
    })
  })

  const supervisedMetrics = evaluation?.metrics?.supervised || {}
  Object.entries(supervisedMetrics).forEach(([key, metrics]) => {
    if (!available.has(key)) {
      available.set(key, {
        available: true,
        source_run: SOURCE_RUN,
        created_at: evaluation?.generated_at,
        status: 'DISPONIBLE',
        metric: metrics?.f1_score != null ? `F1: ${Number(metrics.f1_score).toFixed(3)}` : '--',
      })
    }
  })

  if (evaluation?.metrics?.autoencoder && !available.has('autoencoder_pytorch')) {
    available.set('autoencoder_pytorch', {
      available: true,
      source_run: SOURCE_RUN,
      created_at: evaluation?.generated_at,
      status: 'DISPONIBLE',
      metric: `Anomalías: ${formatShort(evaluation.metrics.autoencoder.anomaly_count)}`,
    })
  }

  return MODEL_ROWS.map((model) => ({ ...model, ...(available.get(model.key) || { available: false }) }))
}

function SimpleBars({ data, color = 'var(--accent-blue)' }) {
  const entries = Object.entries(data || {})
  const total = entries.reduce((sum, [, value]) => sum + (Number(value) || 0), 0)
  if (!entries.length || total <= 0) return <div className="empty-state">No hay datos suficientes para este gráfico.</div>
  const max = Math.max(...entries.map(([, value]) => Number(value) || 0), 1)
  return (
    <div className="simple-bar-chart">
      {entries.map(([label, value]) => {
        const width = Math.max(4, Math.round(((Number(value) || 0) / max) * 100))
        return (
          <div className="simple-bar-row" key={label}>
            <span className="simple-bar-label">{label}</span>
            <div className="simple-bar-track"><span className="simple-bar-fill" style={{ width: `${width}%`, background: color }} /></div>
            <span className="simple-bar-value">{formatShort(value)}</span>
          </div>
        )
      })}
    </div>
  )
}

function ReviewDonut({ data }) {
  const entries = Object.entries(data || {}).filter(([, value]) => Number(value) > 0)
  const total = entries.reduce((sum, [, value]) => sum + Number(value), 0)
  if (!entries.length || total <= 0) return <div className="empty-state">No hay datos suficientes para este gráfico.</div>
  const colors = ['#2d8cff', '#38d6d6', '#27d17f', '#f7b955', '#ff5c6c']
  let cursor = 0
  const stops = entries.map(([label, value], index) => {
    const start = cursor
    cursor += (Number(value) / total) * 100
    return `${colors[index % colors.length]} ${start}% ${cursor}%`
  }).join(', ')
  return (
    <div className="donut-layout">
      <div className="donut-chart" style={{ background: `conic-gradient(${stops})` }}><span>{formatShort(total)}</span></div>
      <div className="chart-legend">
        {entries.map(([label, value], index) => (
          <div key={label}><span className="legend-dot" style={{ background: colors[index % colors.length] }} /> {label}: {formatShort(value)}</div>
        ))}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [state, setState] = useState({
    loading: true,
    partialError: false,
    overview: null,
    preprocessedRuns: [],
    groupedAlerts: null,
    detailedAlerts: null,
    reviews: null,
    humanSummary: null,
    anomalyRuns: null,
    supervisedRuns: null,
    evaluation: null,
    scoringRuns: null,
    casesSummary: null,
  })

  useEffect(() => {
    let mounted = true
    async function loadDashboard() {
      const first = await Promise.allSettled([
        getDashboardOverview({ source_run: SOURCE_RUN, anomaly_run: ANOMALY_RUN }),
        getPreprocessedRuns(),
        getRulesSummary(SOURCE_RUN, { page: 1, page_size: 1 }),
        getRulesAlerts(SOURCE_RUN, { page: 1, page_size: 1 }),
        getAlertReviews(SOURCE_RUN, { page: 1, page_size: 1 }),
        getHumanLabelSummary(SOURCE_RUN),
        getAnomalyRuns(),
        getSupervisedTrainingRuns(SOURCE_RUN),
        getModelEvaluationSummary(SOURCE_RUN),
        getBatchScoringRuns(),
        getCasesSummary(),
      ])
      if (!mounted) return
      setState({
        loading: false,
        partialError: first.some((item) => item.status === 'rejected'),
        overview: resultValue(first[0]),
        preprocessedRuns: normalizeItems(resultValue(first[1])),
        groupedAlerts: resultValue(first[2]),
        detailedAlerts: resultValue(first[3]),
        reviews: resultValue(first[4]),
        humanSummary: resultValue(first[5]),
        anomalyRuns: resultValue(first[6]),
        supervisedRuns: resultValue(first[7]),
        evaluation: resultValue(first[8]),
        scoringRuns: resultValue(first[9]),
        casesSummary: resultValue(first[10]),
      })
    }
    loadDashboard()
    return () => { mounted = false }
  }, [])

  const reviewDistribution = useMemo(
    () => buildReviewDistribution(state.overview, state.humanSummary),
    [state.overview, state.humanSummary]
  )
  const scoringItems = normalizeItems(state.scoringRuns)
  const latestScoring = scoringItems[0] || null
  const casesByStatus = CASE_STATES.reduce((acc, key) => ({ ...acc, [key]: Number(state.casesSummary?.by_status?.[key]) || 0 }), {})
  const casesByPriority = CASE_PRIORITIES.reduce((acc, key) => ({ ...acc, [key]: Number(state.casesSummary?.by_priority?.[key]) || 0 }), {})
  const modelRows = buildModelRows(state)
  const trainedModels = modelRows.filter((row) => row.available).length
  const latestRun = state.preprocessedRuns[0]

  const phaseCards = [
    {
      title: 'Fase A: Data Pipeline',
      status: latestRun ? 'Con datos' : 'Sin datos',
      run: latestRun?.run_id || state.overview?.source_run,
      artifact: latestRun?.filename || '--',
      description: 'Importación y preprocesamiento de transacciones.',
    },
    {
      title: 'Fase B: Reglas y Alertas',
      status: pickTotal(state.groupedAlerts) ? 'Alertas disponibles' : 'Sin alertas',
      run: SOURCE_RUN,
      artifact: `${formatShort(pickTotal(state.groupedAlerts))} agrupadas / ${formatShort(pickTotal(state.detailedAlerts))} detalladas`,
      description: 'Reglas operativas y revision humana.',
    },
    {
      title: 'Fase C: Modelos',
      status: trainedModels ? 'Modelos disponibles' : 'Sin modelos',
      run: SOURCE_RUN,
      artifact: `${formatShort(trainedModels)} modelos`,
      description: 'Modelos no supervisados, supervisados y evaluación.',
    },
    {
      title: 'Fase D: Monitoreo',
      status: scoringItems.length || state.casesSummary?.total ? 'Monitoreo activo' : 'Sin actividad',
      run: latestScoring?.source_run || SOURCE_RUN,
      artifact: `${formatShort(scoringItems.length)} scorings / ${formatShort(state.casesSummary?.total)} casos`,
      description: 'Scoring por lotes y manejo operativo de casos.',
    },
  ]

  return (
    <div className="dashboard-page">
      <div className="header dashboard-header">
        <div>
          <h2>Dashboard General</h2>
          <div className="page-subtitle">Resumen operativo del sistema de detección y monitoreo de fraude bancario.</div>
        </div>
      </div>

      <div className="card warning-banner" role="alert">
        Las alertas, anomalías, predicciones y casos son señales de apoyo analítico. No constituyen fraude confirmado automático.
      </div>

      {state.loading && <div className="card status-banner status-info">Cargando resumen operativo...</div>}
      {!state.loading && state.partialError && <div className="card status-banner status-error">Algunas métricas no pudieron cargarse.</div>}

      <div className="kpi-grid dashboard-kpis">
        <KPICard title="Transacciones procesadas" value={formatNumber(state.overview?.total_transactions)} />
        <KPICard title="Alertas agrupadas" value={formatNumber(pickTotal(state.groupedAlerts) ?? state.overview?.active_alerts)} />
        <KPICard title="Alertas detalladas" value={formatNumber(pickTotal(state.detailedAlerts))} />
        <KPICard title="Revisiones humanas" value={formatNumber(state.humanSummary?.total_reviews ?? pickTotal(state.reviews) ?? state.overview?.review_distribution?.total_reviews)} />
        <KPICard title="Modelos entrenados" value={formatNumber(trainedModels)} />
        <KPICard title="Ejecuciones de scoring" value={formatNumber(scoringItems.length || state.scoringRuns?.count)} />
        <KPICard title="Casos abiertos" value={formatNumber(casesByStatus.OPEN)} />
        <KPICard title="Casos cerrados" value={formatNumber(casesByStatus.CLOSED)} />
      </div>

      <div className="card">
        <h3>Estado por fases</h3>
        <div className="phase-grid">
          {phaseCards.map((phase) => (
            <div className="metadata-item" key={phase.title}>
              <div className="metric-label">{phase.title}</div>
              <div className="metadata-value">{phase.status}</div>
              <div>Último run: {phase.run || '--'}</div>
              <div>Artefacto: {phase.artifact || '--'}</div>
              <div className="section-help">{phase.description}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="dashboard-chart-grid">
        <div className="card">
          <h3>Distribución de revisión humana</h3>
          <ReviewDonut data={reviewDistribution} />
          <div className="section-help" style={{ marginTop: 12 }}>
            Solo CONFIRMED_FRAUD y DISMISSED se usan para entrenamiento supervisado.
          </div>
        </div>
        <div className="card">
          <h3>Scoring por nivel de prioridad</h3>
          <SimpleBars data={SCORE_LEVELS.reduce((acc, key) => ({ ...acc, [key]: Number(latestScoring?.[`${key.toLowerCase()}_count`]) || 0 }), {})} color="linear-gradient(90deg, #ff5c6c, #f7b955)" />
        </div>
      </div>

      <div className="card">
        <h3>Estado de modelos</h3>
        <div className="table-scroll">
          <table className="table dashboard-models-table">
            <thead><tr><th>Modelo</th><th>Disponible</th><th>Source Run</th><th>Fecha</th><th>Estado</th><th>Métrica principal</th></tr></thead>
            <tbody>
              {modelRows.map((model) => (
                <tr key={model.key}>
                  <td>{model.label}</td>
                  <td>{model.available ? 'Disponible' : 'No disponible'}</td>
                  <td>{model.source_run || '--'}</td>
                  <td>{formatDate(model.created_at)}</td>
                  <td>{model.status || '--'}</td>
                  <td>{model.metric || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3>Últimos scorings</h3>
        <div className="table-scroll">
          <table className="table dashboard-scoring-table">
            <thead><tr><th>ID</th><th>Source Run</th><th>Algoritmo</th><th>Total Records</th><th>HIGH</th><th>MEDIUM</th><th>LOW</th><th>Status</th><th>Created At</th></tr></thead>
            <tbody>
              {scoringItems.slice(0, 6).map((run) => (
                <tr key={run.id}>
                  <td>{run.id}</td>
                  <td>{run.source_run || '--'}</td>
                  <td>{run.algorithm || '--'}</td>
                  <td>{formatShort(run.total_records ?? run.total_scored)}</td>
                  <td>{formatShort(run.high_count)}</td>
                  <td>{formatShort(run.medium_count)}</td>
                  <td>{formatShort(run.low_count)}</td>
                  <td>{run.status || '--'}</td>
                  <td>{formatDate(run.created_at)}</td>
                </tr>
              ))}
              {scoringItems.length === 0 && <tr><td colSpan={9}>Sin datos de scoring.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>

      <div className="dashboard-chart-grid">
        <div className="card">
          <h3>Resumen de casos</h3>
          <div className="metadata-grid">
            {CASE_STATES.map((key) => <div className="metadata-item" key={key}><div className="metric-label">{key}</div><div className="metadata-value">{formatShort(casesByStatus[key])}</div></div>)}
            {CASE_PRIORITIES.map((key) => <div className="metadata-item" key={key}><div className="metric-label">{key}</div><div className="metadata-value">{formatShort(casesByPriority[key])}</div></div>)}
          </div>
        </div>
        <div className="card">
          <h3>Casos por estado</h3>
          <SimpleBars data={casesByStatus} color="linear-gradient(90deg, var(--accent-cyan), var(--success))" />
        </div>
      </div>

      <div className="card">
        <h3>Accesos rápidos</h3>
        <div className="quick-actions">
          <Link className="button" to="/rules-alerts">Ver alertas</Link>
          <Link className="button button-secondary" to="/models/unsupervised">Ver no supervisados</Link>
          <Link className="button button-secondary" to="/models/supervised">Ver supervisados</Link>
          <Link className="button button-secondary" to="/models/evaluation">Ver evaluación de modelos</Link>
          <Link className="button" to="/monitoring/scoring">Ejecutar scoring</Link>
          <Link className="button" to="/monitoring/cases">Ver casos</Link>
        </div>
      </div>
    </div>
  )
}
