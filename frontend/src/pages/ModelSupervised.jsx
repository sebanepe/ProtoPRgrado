import React, { useEffect, useMemo, useState } from 'react'
import KPICard from '../components/KPICard'
import { getHumanLabelSummary, getHumanReadiness } from '../services/api'

const DEFAULT_SOURCE_RUN = 'preprocessed_run_26'

const safeNumber = (value) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

const formatCount = (value) => safeNumber(value).toLocaleString()

const friendlyError = (error, fallback) => {
  const status = error?.response?.status
  if (status === 404) return 'No se encontro informacion de etiquetas humanas para el run seleccionado.'
  if (status >= 500) return 'El servicio de etiquetas humanas no esta disponible temporalmente.'
  return fallback
}

const verdictMessage = (verdict) => {
  if (verdict === 'HUMAN_LABELS_STRONG_READY') {
    return 'Existen etiquetas suficientes para un entrenamiento mas solido.'
  }
  if (verdict === 'HUMAN_LABELS_RECOMMENDED_READY') {
    return 'Existen etiquetas suficientes para un entrenamiento recomendado.'
  }
  if (verdict === 'HUMAN_LABELS_TECHNICALLY_READY') {
    return 'Existen etiquetas suficientes para una prueba tecnica inicial.'
  }
  return 'No existen suficientes etiquetas humanas para entrenar un modelo supervisado.'
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
      <div
        aria-label={`${label}: ${safeCurrent} / ${safeRequired}`}
        style={{
          height: 8,
          borderRadius: 999,
          overflow: 'hidden',
          background: 'rgba(36, 52, 71, 0.95)',
          border: '1px solid rgba(56, 214, 214, 0.16)'
        }}
      >
        <div
          style={{
            width: `${percent}%`,
            height: '100%',
            background: percent >= 100 ? 'var(--success)' : 'var(--accent-blue)'
          }}
        />
      </div>
    </div>
  )
}

function GoalCard({ title, positiveCurrent, negativeCurrent, positiveRequired, negativeRequired, ready }) {
  return (
    <div className="card" style={{ margin: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', marginBottom: 14 }}>
        <h4 style={{ margin: 0 }}>{title}</h4>
        <span className="warning-badge" style={{
          borderRadius: 8,
          color: ready ? '#a7f3d0' : '#ffe4b8',
          background: ready ? 'rgba(39, 209, 127, 0.12)' : 'rgba(247, 185, 85, 0.12)',
          borderColor: ready ? 'rgba(39, 209, 127, 0.24)' : 'rgba(247, 185, 85, 0.24)'
        }}>
          {ready ? 'Listo' : 'Pendiente'}
        </span>
      </div>
      <div style={{ display: 'grid', gap: 12 }}>
        <ProgressLine label="Positivos" current={positiveCurrent} required={positiveRequired} />
        <ProgressLine label="Negativos" current={negativeCurrent} required={negativeRequired} />
      </div>
    </div>
  )
}

export default function ModelSupervised() {
  const [sourceRun, setSourceRun] = useState(DEFAULT_SOURCE_RUN)
  const [summary, setSummary] = useState(null)
  const [readiness, setReadiness] = useState(null)
  const [loadingSummary, setLoadingSummary] = useState(false)
  const [loadingReadiness, setLoadingReadiness] = useState(false)
  const [summaryError, setSummaryError] = useState('')
  const [readinessError, setReadinessError] = useState('')
  const [modelConfig, setModelConfig] = useState({
    model: 'random_forest',
    test_size: '0.2',
    random_state: '42',
    use_smote: false
  })

  useEffect(() => {
    const normalizedRun = sourceRun.trim()
    let cancelled = false

    setLoadingSummary(true)
    setSummaryError('')
    getHumanLabelSummary(normalizedRun || null)
      .then((payload) => {
        if (!cancelled) setSummary(payload || null)
      })
      .catch((error) => {
        if (!cancelled) {
          setSummary(null)
          setSummaryError(friendlyError(error, 'No se pudo cargar el resumen de etiquetas humanas.'))
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingSummary(false)
      })

    setLoadingReadiness(true)
    setReadinessError('')
    getHumanReadiness(normalizedRun || null)
      .then((payload) => {
        if (!cancelled) setReadiness(payload || null)
      })
      .catch((error) => {
        if (!cancelled) {
          setReadiness(null)
          setReadinessError(friendlyError(error, 'No se pudo cargar el estado de preparacion.'))
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingReadiness(false)
      })

    return () => { cancelled = true }
  }, [sourceRun])

  const currentPositive = safeNumber(summary?.usable_positive_labels ?? readiness?.current?.positive)
  const currentNegative = safeNumber(summary?.usable_negative_labels ?? readiness?.current?.negative)
  const technicalReady = Boolean(readiness?.technical_ready || summary?.technical_ready)
  const noReviews = !loadingSummary && !summaryError && safeNumber(summary?.total_reviews) === 0

  const metrics = useMemo(() => [
    { title: 'Total de revisiones', value: formatCount(summary?.total_reviews) },
    { title: 'Confirmed Fraud', value: formatCount(summary?.confirmed_fraud) },
    { title: 'Dismissed', value: formatCount(summary?.dismissed) },
    { title: 'False Positive excluido', value: formatCount(summary?.false_positive_excluded) },
    { title: 'New excluido', value: formatCount(summary?.new) },
    { title: 'In Review excluido', value: formatCount(summary?.in_review) },
    { title: 'Positivas usables', value: formatCount(summary?.usable_positive_labels) },
    { title: 'Negativas usables', value: formatCount(summary?.usable_negative_labels) },
    { title: 'Total usable', value: formatCount(summary?.usable_total_labels) },
  ], [summary])

  return (
    <div>
      <div className="header">
        <div>
          <h2>Modelos Supervisados</h2>
          <div className="page-subtitle">
            Clasificacion de alertas revisadas a partir de etiquetas humanas.
          </div>
        </div>
      </div>

      <div className="card warning-banner">
        Los modelos supervisados requieren etiquetas humanas confiables. Las reglas automaticas y las anomalias no supervisadas no se consideran fraude confirmado por si mismas.
      </div>

      <div className="card">
        <div className="form-row" style={{ maxWidth: 420 }}>
          <label htmlFor="supervised_source_run">source_run</label>
          <input
            id="supervised_source_run"
            className="input"
            value={sourceRun}
            onChange={(event) => setSourceRun(event.target.value)}
            placeholder="preprocessed_run_26"
          />
        </div>
      </div>

      <div className="card">
        <h3>Resumen de etiquetas humanas</h3>
        {loadingSummary && <div>Cargando resumen de etiquetas humanas...</div>}
        {!loadingSummary && summaryError && <div className="detail-status detail-status-error" style={{ padding: 12, borderRadius: 8 }}>{summaryError}</div>}
        {!loadingSummary && !summaryError && noReviews && (
          <div className="detail-status" style={{ padding: 12, borderRadius: 8 }}>
            No hay revisiones humanas registradas para este run.
          </div>
        )}
        {!loadingSummary && !summaryError && summary && (
          <div className="kpi-grid">
            {metrics.map((metric) => (
              <KPICard key={metric.title} title={metric.title} value={metric.value} />
            ))}
          </div>
        )}
      </div>

      <div className="card">
        <h3>Readiness supervisado</h3>
        {loadingReadiness && <div>Cargando readiness...</div>}
        {!loadingReadiness && readinessError && <div className="detail-status detail-status-error" style={{ padding: 12, borderRadius: 8 }}>{readinessError}</div>}
        {!loadingReadiness && !readinessError && (
          <div className={technicalReady ? 'detail-status detail-status-success' : 'detail-status'} style={{ padding: 12, borderRadius: 8, marginBottom: 16 }}>
            <strong>{readiness?.verdict || 'INSUFFICIENT_HUMAN_LABELS'}</strong>
            <div style={{ marginTop: 6 }}>
              {verdictMessage(readiness?.verdict || summary?.verdict)}
            </div>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
          <GoalCard
            title="Minimo tecnico"
            positiveCurrent={currentPositive}
            negativeCurrent={currentNegative}
            positiveRequired={readiness?.requirements?.technical?.positive ?? 20}
            negativeRequired={readiness?.requirements?.technical?.negative ?? 20}
            ready={Boolean(readiness?.technical_ready)}
          />
          <GoalCard
            title="Recomendado"
            positiveCurrent={currentPositive}
            negativeCurrent={currentNegative}
            positiveRequired={readiness?.requirements?.recommended?.positive ?? 50}
            negativeRequired={readiness?.requirements?.recommended?.negative ?? 120}
            ready={Boolean(readiness?.recommended_ready)}
          />
          <GoalCard
            title="Meta fuerte"
            positiveCurrent={currentPositive}
            negativeCurrent={currentNegative}
            positiveRequired={readiness?.requirements?.strong?.positive ?? 70}
            negativeRequired={readiness?.requirements?.strong?.negative ?? 180}
            ready={Boolean(readiness?.strong_ready)}
          />
        </div>
      </div>

      <div className="card">
        <h3>Entrenamiento supervisado</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
          <div className="form-row">
            <label htmlFor="supervised_model">Modelo</label>
            <select
              id="supervised_model"
              className="input"
              value={modelConfig.model}
              onChange={(event) => setModelConfig((prev) => ({ ...prev, model: event.target.value }))}
            >
              <option value="random_forest">Random Forest</option>
              <option value="logistic_regression">Logistic Regression</option>
              <option value="gradient_boosting">Gradient Boosting</option>
            </select>
          </div>
          <div className="form-row">
            <label htmlFor="supervised_test_size">test_size</label>
            <input
              id="supervised_test_size"
              className="input"
              value={modelConfig.test_size}
              onChange={(event) => setModelConfig((prev) => ({ ...prev, test_size: event.target.value }))}
            />
          </div>
          <div className="form-row">
            <label htmlFor="supervised_random_state">random_state</label>
            <input
              id="supervised_random_state"
              className="input"
              value={modelConfig.random_state}
              onChange={(event) => setModelConfig((prev) => ({ ...prev, random_state: event.target.value }))}
            />
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 24 }}>
            <input
              type="checkbox"
              checked={modelConfig.use_smote}
              onChange={(event) => setModelConfig((prev) => ({ ...prev, use_smote: event.target.checked }))}
            />
            use_smote
          </label>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginTop: 16 }}>
          <button className="button" disabled>
            {technicalReady ? 'Entrenar modelo supervisado - Disponible en C4.4' : 'Entrenar modelo supervisado'}
          </button>
          <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>
            {technicalReady ? 'La ejecucion real de entrenamiento queda pendiente para C4.4.' : 'Requiere al menos 20 positivos y 20 negativos humanos.'}
          </span>
        </div>
      </div>

      <div className="card">
        <h3>Criterio metodologico</h3>
        <p>
          Para esta fase, solo se consideran etiquetas humanas: CONFIRMED_FRAUD se utilizara como clase positiva. DISMISSED se utilizara como clase negativa. FALSE_POSITIVE, NEW e IN_REVIEW no se usaran para entrenar en esta etapa.
        </p>
      </div>
    </div>
  )
}
