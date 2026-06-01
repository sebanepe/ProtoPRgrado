import React, { useEffect, useMemo, useState } from 'react'
import {
  runBatchScoring,
  getBatchScoringRuns,
  getBatchScoringResults,
  getBatchScoringReport,
  getBatchScoringMetadata,
} from '../services/api'

const FORBIDDEN_COLS = new Set([
  'is_fraud', 'confirmed_fraud', 'PAN_TARJETA', 'TARJETA',
  'pan_card', 'raw_card', 'target_human_label', 'target_label_source',
  'target_label_meaning', 'human_review_comment', 'reviewed_by',
])

const ALGORITHM_LABELS = {
  logistic_regression: 'Regresión logística',
  random_forest: 'Random Forest',
  gradient_boosting: 'Gradient Boosting',
}

const DEFAULT_FORM = {
  source_run: 'preprocessed_run_26',
  algorithm: 'logistic_regression',
  threshold: '0.5',
  sample_size: '',
}

function extractScoringError(e) {
  const detail = e?.response?.data?.detail
  if (!detail) return 'Backend no disponible. Verifique que el servicio esté activo.'
  if (typeof detail === 'string') return detail
  if (detail && detail.error === 'algorithm_invalid') {
    return `Algoritmo no válido. Opciones válidas: ${detail.valid_algorithms?.join(', ') || ''}`
  }
  if (detail && detail.error === 'dataset_not_found') return 'Dataset de preprocesamiento no encontrado.'
  if (detail && detail.error === 'scoring_error') return `Error de scoring: ${detail.message || ''}`
  return typeof detail === 'object' ? JSON.stringify(detail) : String(detail)
}

function formatNumber(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
  return Number(v).toLocaleString('es-ES')
}

function formatDate(v) {
  if (!v) return '—'
  const d = new Date(v)
  return Number.isNaN(d.getTime()) ? String(v) : d.toLocaleString('es-ES')
}

function formatProba(v) {
  if (v === null || v === undefined || v === '') return '—'
  const n = Number(v)
  return Number.isNaN(n) ? String(v) : n.toFixed(4)
}

function RiskBadge({ level }) {
  const styles = {
    HIGH: { background: 'rgba(255,92,108,0.18)', color: '#fecdd3', border: '1px solid rgba(255,92,108,0.35)' },
    MEDIUM: { background: 'rgba(247,185,85,0.18)', color: '#ffe9bf', border: '1px solid rgba(247,185,85,0.35)' },
    LOW: { background: 'rgba(39,209,127,0.18)', color: '#a7f3d0', border: '1px solid rgba(39,209,127,0.35)' },
  }
  const s = styles[level] || {
    background: 'rgba(255,255,255,0.08)',
    color: 'var(--text-muted)',
    border: '1px solid var(--border-soft)',
  }
  return (
    <span style={{ ...s, padding: '2px 10px', borderRadius: 999, fontSize: 12, fontWeight: 700, display: 'inline-block' }}>
      {level || '—'}
    </span>
  )
}

function ScoringDonutChart({ high, medium, low }) {
  const h = Math.max(Number(high) || 0, 0)
  const m = Math.max(Number(medium) || 0, 0)
  const l = Math.max(Number(low) || 0, 0)
  const total = h + m + l
  if (!total) return <div className="empty-state">No hay datos para el gráfico.</div>
  const hPct = (h / total) * 100
  const mPct = (m / total) * 100
  const gradient = `conic-gradient(#ff5c6c 0 ${hPct}%, #f7b955 ${hPct}% ${hPct + mPct}%, #27d17f ${hPct + mPct}% 100%)`
  return (
    <div className="donut-layout">
      <div className="donut-chart" style={{ background: gradient }}>
        <span>{total}</span>
      </div>
      <div className="chart-legend">
        <div><span className="legend-dot" style={{ background: '#ff5c6c' }} /> HIGH: {formatNumber(h)} ({hPct.toFixed(1)}%)</div>
        <div><span className="legend-dot" style={{ background: '#f7b955' }} /> MEDIUM: {formatNumber(m)} ({mPct.toFixed(1)}%)</div>
        <div><span className="legend-dot" style={{ background: '#27d17f' }} /> LOW: {formatNumber(l)} ({(100 - hPct - mPct).toFixed(1)}%)</div>
      </div>
    </div>
  )
}

function PredictionBars({ high, medium, low }) {
  const positive = Math.max(Number(high) || 0, 0) + Math.max(Number(medium) || 0, 0)
  const negative = Math.max(Number(low) || 0, 0)
  const maxV = Math.max(positive, negative, 1)
  const items = [
    { label: 'Predicción positiva (HIGH + MEDIUM)', value: positive },
    { label: 'Predicción negativa (LOW)', value: negative },
  ]
  if (!positive && !negative) return <div className="empty-state">Sin datos.</div>
  return (
    <div className="simple-bar-chart">
      {items.map(item => {
        const width = Math.max(4, Math.round((item.value / maxV) * 100))
        return (
          <div className="simple-bar-row" key={item.label}>
            <span className="simple-bar-label">{item.label}</span>
            <div className="simple-bar-track">
              <span className="simple-bar-fill" style={{ width: `${width}%` }} />
            </div>
            <span className="simple-bar-value">{String(item.value)}</span>
          </div>
        )
      })}
    </div>
  )
}

function ProbaHistogram({ rows }) {
  if (!rows || !rows.length) return <div className="empty-state">Sin datos.</div>
  const hasProba = rows.some(r => r.ml_risk_score !== undefined && r.ml_risk_score !== null)
  if (!hasProba) return <div className="empty-state">No hay probabilidades estimadas disponibles.</div>
  const buckets = [
    { label: '0.00 – 0.25', min: 0, max: 0.25, count: 0 },
    { label: '0.25 – 0.50', min: 0.25, max: 0.50, count: 0 },
    { label: '0.50 – 0.75', min: 0.50, max: 0.75, count: 0 },
    { label: '0.75 – 1.00', min: 0.75, max: 1.01, count: 0 },
  ]
  rows.forEach(r => {
    const v = Number(r.ml_risk_score)
    if (!Number.isNaN(v)) {
      const b = buckets.find(bk => v >= bk.min && v < bk.max)
      if (b) b.count++
    }
  })
  const maxC = Math.max(...buckets.map(b => b.count), 1)
  return (
    <div className="simple-bar-chart">
      {buckets.map(b => {
        const width = Math.max(4, Math.round((b.count / maxC) * 100))
        return (
          <div className="simple-bar-row" key={b.label}>
            <span className="simple-bar-label">{b.label}</span>
            <div className="simple-bar-track">
              <span className="simple-bar-fill" style={{ width: `${width}%` }} />
            </div>
            <span className="simple-bar-value">{b.count}</span>
          </div>
        )
      })}
    </div>
  )
}

export default function BatchScoring() {
  const [form, setForm] = useState(DEFAULT_FORM)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState('')
  const [runSuccess, setRunSuccess] = useState('')

  const [runs, setRuns] = useState([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [runsError, setRunsError] = useState('')

  // Active context (source_run + algorithm) for results/report/metadata
  const [activeCtx, setActiveCtx] = useState(null)

  const [results, setResults] = useState(null)
  const [resultsLoading, setResultsLoading] = useState(false)
  const [resultsError, setResultsError] = useState('')
  const [resultsPage, setResultsPage] = useState(1)
  const [filterLevel, setFilterLevel] = useState('')

  const [report, setReport] = useState(null)
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState('')

  const [metadata, setMetadata] = useState(null)
  const [metadataLoading, setMetadataLoading] = useState(false)
  const [metadataError, setMetadataError] = useState('')
  const [metadataOpen, setMetadataOpen] = useState(false)

  const lastRun = useMemo(() => (runs.length > 0 ? runs[0] : null), [runs])
  const resultRows = useMemo(() => (results?.rows || []).filter(r => {
    // Safety: omit any forbidden column values from display (rows themselves are filtered on backend)
    return r
  }), [results])

  async function loadRuns() {
    setRunsLoading(true)
    setRunsError('')
    try {
      const data = await getBatchScoringRuns()
      setRuns(data.items || [])
    } catch (e) {
      setRunsError(extractScoringError(e))
    } finally {
      setRunsLoading(false)
    }
  }

  async function loadResults(ctx, page, level) {
    if (!ctx) return
    setResultsLoading(true)
    setResultsError('')
    try {
      const params = { source_run: ctx.source_run, algorithm: ctx.algorithm, page, page_size: 20 }
      if (level) params.ml_risk_level = level
      const data = await getBatchScoringResults(params)
      setResults(data)
    } catch (e) {
      setResultsError(extractScoringError(e))
      setResults(null)
    } finally {
      setResultsLoading(false)
    }
  }

  async function loadReport(ctx) {
    if (!ctx) return
    setReportLoading(true)
    setReportError('')
    try {
      const data = await getBatchScoringReport({ source_run: ctx.source_run, algorithm: ctx.algorithm })
      setReport(data)
    } catch (e) {
      setReportError(extractScoringError(e))
      setReport(null)
    } finally {
      setReportLoading(false)
    }
  }

  async function loadMetadata(ctx) {
    if (!ctx) return
    setMetadataLoading(true)
    setMetadataError('')
    try {
      const data = await getBatchScoringMetadata({ source_run: ctx.source_run, algorithm: ctx.algorithm })
      setMetadata(data)
    } catch (e) {
      setMetadataError(extractScoringError(e))
      setMetadata(null)
    } finally {
      setMetadataLoading(false)
    }
  }

  useEffect(() => { loadRuns() }, [])

  useEffect(() => {
    if (activeCtx) {
      loadResults(activeCtx, 1, '')
      loadReport(activeCtx)
      loadMetadata(activeCtx)
      setResultsPage(1)
      setFilterLevel('')
    }
  }, [activeCtx])

  async function handleRun() {
    if (!form.source_run.trim()) return
    setRunning(true)
    setRunError('')
    setRunSuccess('')
    try {
      const payload = { source_run: form.source_run.trim(), algorithm: form.algorithm }
      const res = await runBatchScoring(payload)
      setRunSuccess(
        `Scoring completado. Total evaluados: ${res.total_scored}. ` +
        `HIGH: ${res.high_count}, MEDIUM: ${res.medium_count}, LOW: ${res.low_count}.`
      )
      const ctx = { source_run: form.source_run.trim(), algorithm: form.algorithm }
      setActiveCtx(ctx)
      await loadRuns()
    } catch (e) {
      setRunError(extractScoringError(e))
    } finally {
      setRunning(false)
    }
  }

  function selectRun(run) {
    setActiveCtx({ source_run: run.source_run, algorithm: run.algorithm })
  }

  function handlePageChange(newPage) {
    setResultsPage(newPage)
    loadResults(activeCtx, newPage, filterLevel)
  }

  function handleFilterLevel(value) {
    setFilterLevel(value)
    setResultsPage(1)
    loadResults(activeCtx, 1, value)
  }

  return (
    <div className="models-page">
      {/* 1. Header */}
      <div className="header">
        <h2>Scoring por Lotes</h2>
        <div>Aplicación de modelos entrenados sobre lotes de alertas o transacciones para priorización analítica.</div>
      </div>

      {/* 2. Banner metodológico obligatorio */}
      <div className="card warning-banner" role="alert">
        <strong>Aviso metodológico:</strong> El scoring por lotes genera predicciones de apoyo analítico. No confirma fraude automáticamente y no reemplaza la revisión humana.
      </div>

      {/* 3. Ejecutar scoring */}
      <div className="card">
        <h3>Ejecutar scoring por lotes</h3>
        <div className="filters-grid">
          <div className="form-row">
            <label htmlFor="sc-source-run">Source Run</label>
            <input
              id="sc-source-run"
              className="input"
              value={form.source_run}
              onChange={e => setForm(f => ({ ...f, source_run: e.target.value }))}
              placeholder="preprocessed_run_26"
            />
          </div>
          <div className="form-row">
            <label htmlFor="sc-algorithm">Modelo</label>
            <select
              id="sc-algorithm"
              className="input"
              value={form.algorithm}
              onChange={e => setForm(f => ({ ...f, algorithm: e.target.value }))}
            >
              <option value="logistic_regression">Regresión logística</option>
              <option value="random_forest">Random Forest</option>
              <option value="gradient_boosting">Gradient Boosting</option>
            </select>
          </div>
          <div className="form-row">
            <label htmlFor="sc-threshold">Umbral (informativo)</label>
            <input
              id="sc-threshold"
              className="input"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={form.threshold}
              onChange={e => setForm(f => ({ ...f, threshold: e.target.value }))}
            />
            <div className="field-help">
              Umbral desde el cual una predicción se considera positiva. Fijo por modelo: &lt;0.5 LOW · 0.5–0.75 MEDIUM · ≥0.75 HIGH.
            </div>
          </div>
          <div className="form-row">
            <label htmlFor="sc-sample">Muestra (opcional)</label>
            <input
              id="sc-sample"
              className="input"
              type="number"
              min="1"
              value={form.sample_size}
              onChange={e => setForm(f => ({ ...f, sample_size: e.target.value }))}
              placeholder="Sin límite"
            />
            <div className="field-help">Opcional. Limita la cantidad de registros para pruebas rápidas.</div>
          </div>
        </div>
        <button
          className="button"
          onClick={handleRun}
          disabled={running || !form.source_run.trim()}
        >
          {running ? 'Ejecutando...' : 'Ejecutar scoring'}
        </button>
        {running && (
          <div className="status-banner status-info" style={{ marginTop: 12 }}>
            Ejecutando scoring por lotes. Este proceso puede tardar unos segundos.
          </div>
        )}
        {runSuccess && <div className="status-banner status-success" style={{ marginTop: 12 }}>{runSuccess}</div>}
        {runError && <div className="status-banner status-error" style={{ marginTop: 12 }}>{runError}</div>}
      </div>

      {/* 4. Resumen de última ejecución */}
      {lastRun && (
        <div className="card">
          <h3>Resumen de última ejecución</h3>
          <div className="kpi-grid">
            <div className="card metric-card">
              <div className="metric-label">Total evaluados</div>
              <div className="metric-value">{formatNumber(lastRun.total_scored)}</div>
            </div>
            <div className="card metric-card">
              <div className="metric-label">HIGH</div>
              <div className="metric-value" style={{ color: '#fca5a5' }}>{formatNumber(lastRun.high_count)}</div>
            </div>
            <div className="card metric-card">
              <div className="metric-label">MEDIUM</div>
              <div className="metric-value" style={{ color: '#fde68a' }}>{formatNumber(lastRun.medium_count)}</div>
            </div>
            <div className="card metric-card">
              <div className="metric-label">LOW</div>
              <div className="metric-value" style={{ color: '#a7f3d0' }}>{formatNumber(lastRun.low_count)}</div>
            </div>
          </div>
          <div className="metadata-grid" style={{ marginTop: 12 }}>
            <div className="metadata-item">
              <div className="metric-label">Modelo usado</div>
              <div>{ALGORITHM_LABELS[lastRun.algorithm] || lastRun.algorithm || '—'}</div>
            </div>
            <div className="metadata-item">
              <div className="metric-label">Estado</div>
              <div>{lastRun.status || '—'}</div>
            </div>
            <div className="metadata-item">
              <div className="metric-label">Fecha de ejecución</div>
              <div>{formatDate(lastRun.created_at)}</div>
            </div>
            <div className="metadata-item">
              <div className="metric-label">Source Run</div>
              <div>{lastRun.source_run || '—'}</div>
            </div>
          </div>
          <div className="section-help" style={{ marginTop: 12, fontSize: 13 }}>
            HIGH, MEDIUM y LOW son niveles de priorización generados por el modelo. No representan confirmación automática de fraude.
          </div>
        </div>
      )}

      {/* 5. Lista de ejecuciones */}
      <div className="card">
        <h3>Ejecuciones de scoring</h3>
        {runsLoading && <div className="empty-state">Cargando ejecuciones...</div>}
        {runsError && <div className="status-banner status-error">{runsError}</div>}
        {!runsLoading && !runsError && runs.length === 0 && (
          <div className="empty-state">
            No existen ejecuciones de scoring para este run. Ejecute scoring para generar resultados.
          </div>
        )}
        {!runsLoading && runs.length > 0 && (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Source Run</th>
                  <th>Modelo</th>
                  <th>Total</th>
                  <th>HIGH</th>
                  <th>MEDIUM</th>
                  <th>LOW</th>
                  <th>Estado</th>
                  <th>Fecha</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {runs.map(r => (
                  <tr key={r.id} style={{ opacity: r.status !== 'COMPLETED' ? 0.65 : 1 }}>
                    <td>{r.id}</td>
                    <td>{r.source_run}</td>
                    <td>{ALGORITHM_LABELS[r.algorithm] || r.algorithm}</td>
                    <td>{formatNumber(r.total_scored)}</td>
                    <td>{formatNumber(r.high_count)}</td>
                    <td>{formatNumber(r.medium_count)}</td>
                    <td>{formatNumber(r.low_count)}</td>
                    <td>{r.status}</td>
                    <td>{formatDate(r.created_at)}</td>
                    <td>
                      {r.status === 'COMPLETED' && (
                        <button className="button table-row-action" onClick={() => selectRun(r)}>
                          Ver resultados
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* 6. Resultados del scoring */}
      {activeCtx && (
        <div className="card">
          <h3>Resultados del scoring</h3>
          <div className="section-help" style={{ marginBottom: 10 }}>
            Scoring sobre <strong>{activeCtx.source_run}</strong> · Modelo:{' '}
            <strong>{ALGORITHM_LABELS[activeCtx.algorithm] || activeCtx.algorithm}</strong>
          </div>
          <div className="section-help" style={{ marginBottom: 14, fontSize: 13 }}>
            Estos resultados sirven para priorizar revisión. La decisión final corresponde al analista.
          </div>
          <div className="filters-grid" style={{ marginBottom: 14 }}>
            <div className="form-row">
              <label htmlFor="filter-level">Nivel de prioridad</label>
              <select
                id="filter-level"
                className="input"
                value={filterLevel}
                onChange={e => handleFilterLevel(e.target.value)}
              >
                <option value="">Todos</option>
                <option value="HIGH">HIGH</option>
                <option value="MEDIUM">MEDIUM</option>
                <option value="LOW">LOW</option>
              </select>
            </div>
          </div>
          {resultsLoading && <div className="empty-state">Cargando resultados...</div>}
          {resultsError && <div className="status-banner status-error">{resultsError}</div>}
          {!resultsLoading && !resultsError && resultRows.length === 0 && (
            <div className="empty-state">Sin resultados para los filtros aplicados.</div>
          )}
          {!resultsLoading && resultRows.length > 0 && (
            <>
              <div className="table-scroll">
                <table className="table">
                  <thead>
                    <tr>
                      <th>ID Alerta</th>
                      <th>ID Transacción</th>
                      <th>Cliente anonimizado</th>
                      <th>Código regla</th>
                      <th>Nombre regla</th>
                      <th>Probabilidad estimada</th>
                      <th>Nivel de prioridad</th>
                      <th>Modelo</th>
                      <th>Fecha scoring</th>
                    </tr>
                  </thead>
                  <tbody>
                    {resultRows.map((row, i) => (
                      <tr key={i}>
                        <td>{row.summary_alert_id || '—'}</td>
                        <td>{row.representative_transaction_id || '—'}</td>
                        <td>{row.customer_hash || '—'}</td>
                        <td>{row.rule_code || '—'}</td>
                        <td>{row.rule_name || '—'}</td>
                        <td>{formatProba(row.ml_risk_score)}</td>
                        <td><RiskBadge level={row.ml_risk_level} /></td>
                        <td>{ALGORITHM_LABELS[row.algorithm] || row.algorithm || '—'}</td>
                        <td>{formatDate(row.scored_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="pagination-bar" style={{ marginTop: 12 }}>
                <span className="section-help">
                  Página {results.page} de {results.total_pages} · {formatNumber(results.total)} registros
                </span>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button
                    className="button"
                    disabled={resultsPage <= 1}
                    onClick={() => handlePageChange(resultsPage - 1)}
                  >
                    Anterior
                  </button>
                  <button
                    className="button"
                    disabled={resultsPage >= (results.total_pages || 1)}
                    onClick={() => handlePageChange(resultsPage + 1)}
                  >
                    Siguiente
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* 7. Visualizaciones */}
      {activeCtx && (
        <div className="card">
          <h3>Visualizaciones</h3>
          <div className="distribution-grid">
            <div className="card detail-section chart-panel">
              <h4>Distribución por nivel de prioridad</h4>
              {lastRun ? (
                <ScoringDonutChart
                  high={lastRun.high_count}
                  medium={lastRun.medium_count}
                  low={lastRun.low_count}
                />
              ) : (
                <div className="empty-state">Sin datos de ejecución cargados.</div>
              )}
            </div>
            <div className="card detail-section chart-panel">
              <h4>Cantidad de resultados por predicción</h4>
              {lastRun ? (
                <PredictionBars
                  high={lastRun.high_count}
                  medium={lastRun.medium_count}
                  low={lastRun.low_count}
                />
              ) : (
                <div className="empty-state">Sin datos.</div>
              )}
            </div>
            <div className="card detail-section chart-panel">
              <h4>Distribución de probabilidades estimadas</h4>
              {resultRows.length > 0 ? (
                <ProbaHistogram rows={resultRows} />
              ) : (
                <div className="empty-state">Cargue resultados para ver este gráfico.</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 8. Reporte del scoring */}
      {activeCtx && (
        <div className="card">
          <h3>Reporte del scoring</h3>
          <div className="section-help" style={{ marginBottom: 10, fontSize: 13 }}>
            Este reporte documenta la ejecución, modelo utilizado, distribución de resultados y advertencias metodológicas.
          </div>
          {reportLoading && <div className="empty-state">Cargando reporte...</div>}
          {reportError && <div className="status-banner status-error">{reportError}</div>}
          {!reportLoading && !reportError && !report && (
            <div className="empty-state">Reporte no disponible para esta ejecución.</div>
          )}
          {!reportLoading && report?.markdown && (
            <div className="report-box">
              <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: 'var(--text-soft)', fontSize: 13, lineHeight: 1.7, margin: 0 }}>
                {report.markdown}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* 9. Metadata técnica */}
      {activeCtx && (
        <div className="card">
          <details onToggle={e => setMetadataOpen(e.currentTarget.open)}>
            <summary style={{ cursor: 'pointer', fontWeight: 700, fontSize: '1rem', color: '#eef3fb', userSelect: 'none' }}>
              Metadata técnica {metadataOpen ? '▲' : '▼'}
            </summary>
            <div style={{ marginTop: 14 }}>
              {metadataLoading && <div className="empty-state">Cargando metadata...</div>}
              {metadataError && <div className="status-banner status-error">{metadataError}</div>}
              {!metadataLoading && !metadataError && !metadata && (
                <div className="empty-state">Metadata no disponible para esta ejecución.</div>
              )}
              {!metadataLoading && metadata?.data && (
                <div className="table-scroll">
                  <table className="table">
                    <thead>
                      <tr>
                        <th style={{ width: '40%' }}>Campo</th>
                        <th>Valor</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(metadata.data).map(([k, v]) => (
                        <tr key={k}>
                          <td style={{ fontWeight: 600 }}>{k}</td>
                          <td style={{ wordBreak: 'break-all' }}>
                            {v === null || v === undefined ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </details>
        </div>
      )}

      {/* 10. Interpretación guiada */}
      <div className="card">
        <h3>Cómo interpretar el scoring</h3>
        <p className="section-help">
          El scoring aplica un modelo entrenado sobre un lote de registros y asigna niveles de prioridad. HIGH indica mayor prioridad de revisión, MEDIUM prioridad intermedia y LOW menor prioridad. Estos niveles no son fraude confirmado.
        </p>
        <h4>Qué debe mirar primero el analista</h4>
        <ul style={{ color: 'var(--text-soft)', lineHeight: 1.8 }}>
          <li>Casos HIGH: mayor prioridad de revisión inmediata.</li>
          <li>Casos con mayor probabilidad estimada (ml_risk_score elevado).</li>
          <li>Casos con prioridad alta que además tengan reglas previas activas.</li>
          <li>Casos que puedan convertirse en caso de investigación en la siguiente fase.</li>
        </ul>
      </div>
    </div>
  )
}
