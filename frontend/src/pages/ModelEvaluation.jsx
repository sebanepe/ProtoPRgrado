import React, { useEffect, useMemo, useState } from 'react'
import KPICard from '../components/KPICard'
import {
  buildModelEvaluationComparison,
  getModelEvaluationSummary,
  getModelEvaluationAlertLevel,
  getModelEvaluationTransactionLevel,
  getModelEvaluationReport,
  getModelEvaluationMetadata,
  getModelEvaluationTopCases
} from '../services/api'

const DEFAULT_SOURCE_RUN = 'preprocessed_run_26'
const FORBIDDEN_COLUMNS = new Set(['is_fraud', 'confirmed_fraud', 'PAN_TARJETA', 'TARJETA', 'pan_card', 'raw_card'])
const REQUIRED_MESSAGE = 'Las reglas, anomalias y predicciones son senales de apoyo analitico. Ninguna constituye fraude confirmado automatico. La confirmacion depende de revision humana.'

const MODEL_LABELS = {
  logistic_regression: 'Regresion logistica',
  random_forest: 'Random Forest',
  gradient_boosting: 'Gradient Boosting'
}

const STATUS_BADGE = {
  CONFIRMED_FRAUD: 'detail-status detail-status-error',
  DISMISSED: 'detail-status detail-status-success',
  IN_REVIEW: 'detail-status',
  NEW: 'detail-status'
}

const friendlyError = (error, fallback) => {
  const status = error?.response?.status
  const detail = error?.response?.data?.detail
  if (status === 404) return 'No existe una comparacion generada para este run. Presione \'Actualizar comparacion\' para construirla.'
  if (status >= 500) return 'Backend no disponible temporalmente. Intente nuevamente en unos minutos.'
  if (typeof detail === 'string' && detail.trim()) return detail
  return fallback
}

const formatNumber = (value, digits = 0) => {
  const num = Number(value)
  if (!Number.isFinite(num)) return 'N/A'
  return digits > 0 ? num.toFixed(digits) : num.toLocaleString()
}

const cleanRow = (row = {}) => Object.fromEntries(Object.entries(row).filter(([key]) => !FORBIDDEN_COLUMNS.has(key)))

function MiniBars({ title, rows }) {
  const max = Math.max(1, ...rows.map((item) => Number(item.value) || 0))
  return (
    <div className="card chart-panel">
      <h4>{title}</h4>
      {rows.length === 0 && <div className="empty-state">No existen datos suficientes para construir este grafico.</div>}
      {rows.length > 0 && (
        <div className="simple-bar-chart" data-testid="signals-chart-container">
          {rows.map((item) => {
            const value = Number(item.value) || 0
            const width = `${Math.max(4, (value / max) * 100)}%`
            return (
              <div key={item.label} className="simple-bar-row">
                <span className="simple-bar-label">{item.label}</span>
                <div className="simple-bar-track"><span className="simple-bar-fill" style={{ width }} /></div>
                <span className="simple-bar-value">{formatNumber(value)}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function DataTable({ rows, labels = {}, testId, emptyText = 'Sin datos disponibles.' }) {
  const safeRows = rows.map(cleanRow)
  const labelKeys = Object.keys(labels)
  const cols = safeRows[0]
    ? (labelKeys.length > 0
        ? labelKeys.filter((c) => !FORBIDDEN_COLUMNS.has(c))
        : Object.keys(safeRows[0]).filter((c) => !FORBIDDEN_COLUMNS.has(c)))
    : []
  if (!safeRows.length || !cols.length) return <div className="detail-status" data-testid={testId}>{emptyText}</div>
  return (
    <div className="table-scroll" data-testid={testId}>
      <table className="table">
        <thead><tr>{cols.map((c) => <th key={c}>{labels[c] || c}</th>)}</tr></thead>
        <tbody>
          {safeRows.map((row, i) => (
            <tr key={i}>{cols.map((c) => <td key={c}>{String(row[c] ?? '')}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function ModelEvaluation() {
  const [sourceRun, setSourceRun] = useState(DEFAULT_SOURCE_RUN)
  const [summary, setSummary] = useState(null)
  const [topCases, setTopCases] = useState([])
  const [alertLevel, setAlertLevel] = useState({ items: [], page: 1, page_size: 20, total: 0 })
  const [transactionLevel, setTransactionLevel] = useState({ items: [], page: 1, page_size: 20, total: 0 })
  const [report, setReport] = useState('')
  const [metadata, setMetadata] = useState(null)
  const [alertFilters, setAlertFilters] = useState({ rule_code: '', human_review_status: '', supervised_positive_any: '', unsupervised_anomaly_any: '', risk_level: '' })
  const [txFilters, setTxFilters] = useState({ flagged_by_rules: '', flagged_by_isolation_forest: '', flagged_by_autoencoder: '', country_code: '', merchant_rubro_proxy: '' })
  const [loading, setLoading] = useState({ main: false, build: false })
  const [statusMsg, setStatusMsg] = useState('')

  const normalizedRun = sourceRun.trim() || DEFAULT_SOURCE_RUN

  const loadAll = async () => {
    setLoading((p) => ({ ...p, main: true }))
    try {
      const [sum, top, alert, tx, rep, meta] = await Promise.all([
        getModelEvaluationSummary(normalizedRun),
        getModelEvaluationTopCases(normalizedRun, 20),
        getModelEvaluationAlertLevel(normalizedRun, { page: 1, page_size: 20, ...alertFilters }),
        getModelEvaluationTransactionLevel(normalizedRun, { page: 1, page_size: 20, ...txFilters }),
        getModelEvaluationReport(normalizedRun),
        getModelEvaluationMetadata(normalizedRun)
      ])
      setSummary(sum || null)
      setTopCases(top?.items || [])
      setAlertLevel(alert || { items: [] })
      setTransactionLevel(tx || { items: [] })
      setReport(rep?.markdown || '')
      setMetadata(meta || null)
      if ((sum?.status === 'NOT_AVAILABLE')) setStatusMsg('No existe una comparacion generada para este run. Presione \'Actualizar comparacion\' para construirla.')
      else setStatusMsg('')
    } catch (error) {
      setStatusMsg(friendlyError(error, 'No se pudo cargar la evaluacion comparativa.'))
    } finally {
      setLoading((p) => ({ ...p, main: false }))
    }
  }

  useEffect(() => {
    loadAll()
  }, [normalizedRun])

  const refreshComparison = async () => {
    setLoading((p) => ({ ...p, build: true }))
    setStatusMsg('Construyendo comparacion de modelos. Este proceso puede tardar unos segundos.')
    try {
      const result = await buildModelEvaluationComparison({ source_run: normalizedRun })
      const partial = String(result?.result?.status || '').toUpperCase().includes('PARTIAL_READY') || (Array.isArray(result?.result?.missing_methods) && result.result.missing_methods.length > 0)
      await loadAll()
      if (partial) setStatusMsg('La comparacion fue generada parcialmente porque uno o mas metodos no tienen artefactos disponibles.')
      else setStatusMsg('Comparacion actualizada correctamente.')
    } catch (error) {
      setStatusMsg(friendlyError(error, 'No fue posible actualizar la comparacion.'))
    } finally {
      setLoading((p) => ({ ...p, build: false }))
    }
  }

  const metrics = summary?.metrics || metadata?.metrics || {}
  const intersections = summary?.intersections || metadata?.intersections || {}
  const supervised = metrics.supervised || {}
  const supervisedRows = Object.entries(supervised).map(([model, values]) => ({
    model,
    accuracy: values?.accuracy,
    precision: values?.precision,
    recall: values?.recall,
    f1_score: values?.f1_score,
    roc_auc: values?.roc_auc,
    false_positive_count: values?.false_positive_count,
    false_negative_count: values?.false_negative_count
  }))

  const methodsAvailable = summary?.available_methods || metadata?.available_methods || []
  const methodsMissing = summary?.missing_methods || metadata?.missing_methods || []

  const bestBy = (key) => supervisedRows.reduce((acc, row) => ((Number(row[key]) || -1) > (Number(acc?.[key]) || -1) ? row : acc), null)
  const bestF1 = bestBy('f1_score')
  const bestRecall = bestBy('recall')

  const signalsByMethod = [
    { label: 'Reglas agrupadas', value: metrics?.rules?.total_alerts_grouped || 0 },
    { label: 'Isolation Forest', value: metrics?.isolation_forest?.anomaly_count || 0 },
    { label: 'Autoencoder', value: metrics?.autoencoder?.anomaly_count || 0 },
    { label: 'Supervisado positivo', value: intersections?.rules_and_supervised_positive_count || 0 }
  ]

  const supervisedChart = supervisedRows.map((row) => ({ label: MODEL_LABELS[row.model] || row.model, value: Number(row.f1_score) || 0 }))
  const intersectionChart = [
    { label: 'Reglas ∩ Isolation', value: intersections.rules_and_isolation_count || 0 },
    { label: 'Reglas ∩ Autoencoder', value: intersections.rules_and_autoencoder_count || 0 },
    { label: 'Isolation ∩ Autoencoder', value: intersections.isolation_and_autoencoder_count || 0 },
    { label: 'Reglas ∩ Supervisado', value: intersections.rules_and_supervised_positive_count || 0 },
    { label: 'Todos', value: intersections.all_available_methods_count || 0 }
  ]

  const totalMethods = methodsAvailable.length + methodsMissing.length
  const donutPercent = totalMethods ? Math.round((methodsAvailable.length / totalMethods) * 100) : 0

  return (
    <div>
      <div className="header">
        <div>
          <h2>Evaluacion de Modelos</h2>
          <div className="page-subtitle">Comparacion entre reglas, anomalias no supervisadas y modelos supervisados.</div>
        </div>
      </div>

      <div className="card warning-banner">Esta pantalla permite contrastar que senales genera cada enfoque del sistema. Sirve para analisis comparativo, priorizacion de casos y validacion de resultados, no para confirmar fraude automaticamente.</div>
      <div className="card warning-banner" data-testid="methodology-message">{REQUIRED_MESSAGE}</div>

      {statusMsg && <div className="status-banner status-info">{statusMsg}</div>}

      <div className="card">
        <h3>Run de analisis</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(220px, 380px) auto auto', gap: 10, alignItems: 'end' }}>
          <div className="form-row" style={{ marginBottom: 0 }}>
            <label htmlFor="source_run">source_run</label>
            <input id="source_run" className="input" value={sourceRun} onChange={(e) => setSourceRun(e.target.value)} />
          </div>
          <button className="button button-secondary" onClick={loadAll} disabled={loading.main}>Cargar evaluacion</button>
          <button className="button" onClick={refreshComparison} disabled={loading.build}>Actualizar comparacion</button>
        </div>
      </div>

      <div className="card">
        <h3>Metodos disponibles y faltantes</h3>
        <p><strong>Disponibles:</strong> {methodsAvailable.length ? methodsAvailable.join(', ') : 'Ninguno'}</p>
        <p><strong>Faltantes:</strong> {methodsMissing.length ? methodsMissing.join(', ') : 'Ninguno'}</p>
      </div>

      <div className="card">
        <h3>Resumen comparativo</h3>
        <div className="kpi-grid">
          <KPICard title="Reglas (agrupadas)" value={formatNumber(metrics?.rules?.total_alerts_grouped)} />
          <KPICard title="Reglas (detalladas)" value={formatNumber(metrics?.rules?.total_alerts_detailed)} />
          <KPICard title="Isolation anomalas" value={formatNumber(metrics?.isolation_forest?.anomaly_count)} />
          <KPICard title="Autoencoder anomalas" value={formatNumber(metrics?.autoencoder?.anomaly_count)} />
          <KPICard title="Supervisados entrenados" value={formatNumber(supervisedRows.length)} />
          <KPICard title="Coincidencias totales" value={formatNumber(intersections?.all_available_methods_count)} />
        </div>
        <div className="compact-list">
          <div>Reglas: Detectan patrones definidos de riesgo.</div>
          <div>Isolation Forest: Detecta transacciones estadisticamente atipicas.</div>
          <div>Autoencoder: Detecta transacciones con alto error de reconstruccion.</div>
          <div>Supervisado: Predice a partir de revisiones humanas.</div>
          <div>Coincidencias: Muestran casos senalados por mas de un metodo.</div>
        </div>
      </div>

      <div className="chart-grid">
        <MiniBars title="Cantidad de senales por metodo" rows={signalsByMethod} />
        <MiniBars title="Comparacion de modelos supervisados (F1-score)" rows={supervisedChart} />
        <MiniBars title="Coincidencias entre metodos" rows={intersectionChart} />
        <div className="card chart-panel">
          <h4>Disponibilidad de metodos</h4>
          <div className="donut-layout">
            <div className="donut-chart" style={{ background: `conic-gradient(var(--success) ${donutPercent}%, rgba(255,255,255,0.08) 0)` }}><span>{donutPercent}%</span></div>
            <div className="chart-legend">
              <div><span className="legend-dot" style={{ background: 'var(--success)' }} />disponibles: {methodsAvailable.length}</div>
              <div><span className="legend-dot" style={{ background: 'rgba(255,255,255,0.35)' }} />faltantes: {methodsMissing.length}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="card" data-testid="supervised-comparison-section">
        <h3>Modelos supervisados</h3>
        <div className="detail-status">Estas metricas son preliminares si el dataset etiquetado solo cumple el minimo tecnico.</div>
        <div style={{ marginTop: 10 }}>
          <DataTable
            rows={supervisedRows.map((row) => ({
              modelo: MODEL_LABELS[row.model] || row.model,
              accuracy: formatNumber(row.accuracy, 4),
              precision: formatNumber(row.precision, 4),
              recall: formatNumber(row.recall, 4),
              f1_score: formatNumber(row.f1_score, 4),
              roc_auc: formatNumber(row.roc_auc, 4),
              falsos_positivos: formatNumber(row.false_positive_count),
              falsos_negativos: formatNumber(row.false_negative_count),
              comentario: row.model === bestF1?.model ? 'Mejor balance general.' : row.model === bestRecall?.model ? 'Detecta mas positivos.' : 'Genera menos falsas alarmas.'
            }))}
          />
        </div>
      </div>

      <div className="card" data-testid="top-cases-section">
        <h3>Casos prioritarios por coincidencia de metodos</h3>
        <p>Este ranking no confirma fraude. Solo prioriza casos donde varios metodos coinciden o donde existen senales relevantes.</p>
        <DataTable
          testId="top-cases-table"
          rows={topCases}
          labels={{ summary_alert_id: 'ID Alerta', customer_hash: 'Cliente anonimizado', rule_code: 'Regla', risk_level: 'Riesgo', transactions_detected: 'Transacciones', human_review_status: 'Revision humana', supervised_positive_any: 'Prediccion supervisada positiva', unsupervised_anomaly_any: 'Anomalia no supervisada', methods_agree_count: 'Metodos coincidentes', comparison_priority_score: 'Prioridad comparativa', priority_reason: 'Razon de prioridad' }}
          emptyText="Sin top cases para este run."
        />
      </div>

      <div className="card">
        <h3>Comparacion por alerta agrupada</h3>
        <div className="filters-grid">
          {Object.keys(alertFilters).map((k) => <input key={k} className="input" placeholder={k} value={alertFilters[k]} onChange={(e) => setAlertFilters((p) => ({ ...p, [k]: e.target.value }))} />)}
        </div>
        <button className="button button-secondary" onClick={loadAll}>Aplicar filtros</button>
        <DataTable testId="alert-level-table" rows={alertLevel.items || []} emptyText="Sin alert-level para este run." />
      </div>

      <div className="card">
        <h3>Comparacion por transaccion</h3>
        <div className="filters-grid">
          {Object.keys(txFilters).map((k) => <input key={k} className="input" placeholder={k} value={txFilters[k]} onChange={(e) => setTxFilters((p) => ({ ...p, [k]: e.target.value }))} />)}
        </div>
        <button className="button button-secondary" onClick={loadAll}>Aplicar filtros</button>
        <DataTable testId="transaction-level-table" rows={transactionLevel.items || []} emptyText="Sin transaction-level para este run." />
      </div>

      <div className="card">
        <h3>Como interpretar esta evaluacion</h3>
        <p>Esta pantalla compara senales provenientes de enfoques distintos. Las reglas detectan condiciones operativas conocidas. Isolation Forest identifica rarezas estadisticas. Autoencoder identifica transacciones dificiles de reconstruir. Los modelos supervisados aprenden de revisiones humanas previas. Cuando varios metodos coinciden, el caso puede ser mas relevante para revision, pero no se convierte automaticamente en fraude confirmado.</p>
        <h4>Que mirar primero</h4>
        <ul>
          <li>Casos con mas metodos coincidentes.</li>
          <li>Casos con riesgo alto por reglas.</li>
          <li>Casos con prediccion supervisada positiva.</li>
          <li>Casos con anomalias en ambos modelos no supervisados.</li>
          <li>Falsos negativos del supervisado, si existen.</li>
          <li>Diferencias entre precision y recall.</li>
        </ul>
      </div>

      <div className="card" data-testid="report-section">
        <h3>Reporte tecnico de evaluacion comparativa</h3>
        <p>Este reporte documenta los metodos disponibles, metricas, intersecciones, limitaciones y advertencias.</p>
        <pre className="report-pre">{report || 'No hay reporte disponible para este run.'}</pre>
      </div>

      <div className="card" data-testid="metadata-section">
        <details className="technical-details">
          <summary>Metadata de comparacion</summary>
          <pre className="report-pre">{metadata ? JSON.stringify(metadata, null, 2) : 'No hay metadata disponible para este run.'}</pre>
        </details>
      </div>

      <div className="card">
        <h3>Estado metodologico</h3>
        <p>{REQUIRED_MESSAGE}</p>
        <div className={STATUS_BADGE.NEW}>Fraude confirmado = decision humana.</div>
      </div>
    </div>
  )
}
