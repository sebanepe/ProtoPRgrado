import React, { useState, useEffect } from 'react'
import Table from '../components/Table'
import KPICard from '../components/KPICard'
import {
  getPreprocessedRuns,
  analyzeRules,
  getRulesSummary,
  getRulesReport,
  getRulesMetrics,
  getRulesAlerts,
  getRuleAlertDetail
} from '../services/api'

const getRunId = (run) => run?.run_id ?? run?.id ?? null

const normalizeRun = (run) => {
  if (!run) return run
  const runId = run.run_id ?? run.id ?? null
  return {
    ...run,
    id: runId,
    run_id: runId,
    file: run.filename ?? run.file ?? '',
    file_size: run.size_bytes ?? run.file_size ?? 0,
    created_at: run.created_at ?? run.createdAt ?? '',
    has_alerts: Boolean(run.has_alerts),
    has_summary: Boolean(run.has_summary),
    has_report: Boolean(run.has_report)
  }
}

const getResponseItems = (payload) => {
  if (Array.isArray(payload)) return payload
  if (Array.isArray(payload?.items)) return payload.items
  if (Array.isArray(payload?.data)) return payload.data
  return []
}

const selectTopEntry = (counts = {}) => {
  const entries = Object.entries(counts || {})
  if (entries.length === 0) return null
  return entries.sort((first, second) => second[1] - first[1])[0]
}

const normalizeMetrics = (metrics) => {
  if (!metrics) return metrics
  const alertsByRule = metrics.alerts_by_rule || metrics.rules_distribution || {}
  const alertsByRiskLevel = metrics.alerts_by_risk_level || {}
  const alertsByMcc = metrics.alerts_by_mcc || {}
  const topCustomers = metrics.top_customers || []
  const topRule = selectTopEntry(alertsByRule)
  const topRiskLevel = selectTopEntry(alertsByRiskLevel)
  const topMcc = selectTopEntry(alertsByMcc)

  return {
    ...metrics,
    total_detailed_alerts: metrics.total_alerts ?? metrics.total_detailed_alerts ?? 0,
    total_grouped_alerts: metrics.total_summary_alerts ?? metrics.total_grouped_alerts ?? 0,
    top_rule_code: metrics.top_rule_code ?? topRule?.[0] ?? 'N/A',
    top_risk_level: metrics.top_risk_level ?? topRiskLevel?.[0] ?? 'N/A',
    top_merchant_rubro: metrics.top_merchant_rubro ?? topMcc?.[0] ?? 'N/A',
    top_customer_count: metrics.top_customer_count ?? topCustomers[0]?.alert_count ?? 0,
    rules_distribution: alertsByRule
  }
}

export default function RulesAlerts() {
  // State for runs list
  const [runs, setRuns] = useState([])
  const [loadingRuns, setLoadingRuns] = useState(false)
  const [selectedRun, setSelectedRun] = useState(null)

  // State for analysis
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisMsg, setAnalysisMsg] = useState('')

  // State for metrics
  const [metrics, setMetrics] = useState(null)
  const [loadingMetrics, setLoadingMetrics] = useState(false)

  // State for summary table
  const [summary, setSummary] = useState([])
  const [loadingSummary, setLoadingSummary] = useState(false)
  const [pagination, setPagination] = useState({
    page: 1,
    page_size: 20,
    total: 0,
    has_next: false,
    has_prev: false
  })

  // State for filters
  const [filters, setFilters] = useState({
    rule_code: '',
    risk_level: '',
    status: '',
    country_code: '',
    merchant_rubro_proxy: '',
    customer_hash: ''
  })

  // State for detail modal
  const [showDetailModal, setShowDetailModal] = useState(false)
  const [selectedAlert, setSelectedAlert] = useState(null)
  const [detailAlerts, setDetailAlerts] = useState([])
  const [loadingDetail, setLoadingDetail] = useState(false)

  // State for report
  const [showReportModal, setShowReportModal] = useState(false)
  const [report, setReport] = useState('')
  const [loadingReport, setLoadingReport] = useState(false)

  // Load runs on component mount
  useEffect(() => {
    loadRuns()
  }, [])

  const loadRuns = async () => {
    setLoadingRuns(true)
    try {
      const runsData = getResponseItems(await getPreprocessedRuns()).map(normalizeRun)
      setRuns(runsData || [])
      if (runsData && runsData.length > 0) {
        setSelectedRun(runsData[0])
      }
    } catch (e) {
      setAnalysisMsg('Error al cargar runs: ' + (e?.response?.data?.detail || e?.message || String(e)))
    } finally {
      setLoadingRuns(false)
    }
  }

  const handleAnalyzeAndGenerate = async () => {
    if (!selectedRun) {
      setAnalysisMsg('Por favor selecciona un run')
      return
    }

    setAnalyzing(true)
    setAnalysisMsg('Analizando reglas y generando alertas...')
    try {
      const result = await analyzeRules(getRunId(selectedRun), false, {})
      if (result.status === 'ALREADY_EXISTS') {
        setAnalysisMsg('El análisis ya existe. Se cargará el resumen disponible.')
      } else if (result.status === 'COMPLETED') {
        setAnalysisMsg('Análisis de reglas completado correctamente.')
      } else {
        setAnalysisMsg('Análisis completado con estado: ' + result.status)
      }
      // Load metrics and summary after analysis
      await Promise.all([loadMetrics(), loadSummary()])
    } catch (e) {
      setAnalysisMsg('Error al analizar reglas: ' + (e?.response?.data?.detail || e?.message || String(e)))
    } finally {
      setAnalyzing(false)
    }
  }

  const loadMetrics = async () => {
    if (!selectedRun) return
    setLoadingMetrics(true)
    try {
      const metricsData = normalizeMetrics(await getRulesMetrics(getRunId(selectedRun)))
      setMetrics(metricsData)
    } catch (e) {
      console.error('Error loading metrics:', e)
    } finally {
      setLoadingMetrics(false)
    }
  }

  const loadSummary = async (page = 1) => {
    if (!selectedRun) return
    setLoadingSummary(true)
    try {
      const runId = getRunId(selectedRun)
      const params = {
        page: page,
        page_size: pagination.page_size,
        ...Object.fromEntries(
          Object.entries(filters).filter(([, v]) => v !== '')
        )
      }
      const summaryData = await getRulesSummary(runId, params)
      const summaryItems = getResponseItems(summaryData)
      setSummary(summaryItems)
      setPagination({
        page: summaryData.page || 1,
        page_size: summaryData.page_size || params.page_size || 20,
        total: summaryData.total_items ?? summaryData.total ?? summaryItems.length,
        has_next: summaryData.total_pages ? (summaryData.page || 1) < summaryData.total_pages : (summaryData.has_next || false),
        has_prev: summaryData.total_pages ? (summaryData.page || 1) > 1 : (summaryData.has_prev || false)
      })
    } catch (e) {
      console.error('Error loading summary:', e)
    } finally {
      setLoadingSummary(false)
    }
  }

  const handleApplyFilters = () => {
    setPagination({ ...pagination, page: 1 })
    loadSummary(1)
  }

  const handleClearFilters = () => {
    setFilters({
      rule_code: '',
      risk_level: '',
      status: '',
      country_code: '',
      merchant_rubro_proxy: '',
      customer_hash: ''
    })
    setPagination({ ...pagination, page: 1 })
    loadSummary(1)
  }

  const handleViewDetail = async (summaryAlert) => {
    setSelectedAlert(summaryAlert)
    setShowDetailModal(true)
    setLoadingDetail(true)
    try {
      const runId = getRunId(selectedRun)
      const alertsData = await getRulesAlerts(runId, {
        page: 1,
        page_size: 50,
        customer_hash: summaryAlert.customer_hash,
        rule_code: summaryAlert.rule_code
      })
      const alertCandidates = getResponseItems(alertsData)
      const alertId = summaryAlert.alert_id ?? summaryAlert.representative_alert_id ?? alertCandidates[0]?.alert_id ?? alertCandidates[0]?.id
      let detailAlert = alertCandidates[0] || summaryAlert
      if (alertId) {
        try {
          detailAlert = await getRuleAlertDetail(alertId, runId)
        } catch (detailError) {
          console.error('Error loading alert detail, falling back to alert list:', detailError)
        }
      }
      setDetailAlerts(detailAlert ? [detailAlert] : [])
    } catch (e) {
      console.error('Error loading detail alerts:', e)
    } finally {
      setLoadingDetail(false)
    }
  }

  const handleViewReport = async () => {
    if (!selectedRun) return
    setShowReportModal(true)
    setLoadingReport(true)
    try {
      const reportData = await getRulesReport(getRunId(selectedRun))
      setReport(reportData.content || reportData.report || '')
    } catch (e) {
      setReport('Error al cargar el reporte: ' + (e?.response?.data?.detail || e?.message || String(e)))
    } finally {
      setLoadingReport(false)
    }
  }

  const handleCopyReport = () => {
    navigator.clipboard.writeText(report)
    alert('Reporte copiado al portapapeles')
  }

  const handleReprocesar = async () => {
    if (!selectedRun) return
    if (!window.confirm('¿Estás seguro de que deseas reprocesar este run? Esto puede tomar tiempo.')) {
      return
    }
    setAnalyzing(true)
    setAnalysisMsg('Reprocesando reglas (force=true)...')
    try {
      const result = await analyzeRules(getRunId(selectedRun), true, {})
      setAnalysisMsg('Reprocesamiento completado: ' + result.status)
      await Promise.all([loadMetrics(), loadSummary(1)])
    } catch (e) {
      setAnalysisMsg('Error al reprocesar: ' + (e?.response?.data?.detail || e?.message || String(e)))
    } finally {
      setAnalyzing(false)
    }
  }

  const summaryColumns = [
    { key: 'summary_alert_id', title: 'ID Alerta' },
    { key: 'customer_hash', title: 'Cliente' },
    { key: 'rule_code', title: 'Regla' },
    { key: 'rule_name', title: 'Nombre Regla' },
    { key: 'risk_level', title: 'Nivel Riesgo' },
    { key: 'max_risk_score', title: 'Score' },
    { key: 'count_transactions', title: 'Tx Detectadas' },
    { key: 'countries_detected', title: 'Países' },
    { key: 'status', title: 'Estado' }
  ]

  return (
    <div>
      {/* Header */}
      <div className="header">
        <h2>Reglas y Alertas - Fase B</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="button" onClick={loadRuns} disabled={loadingRuns}>
            {loadingRuns ? 'Cargando...' : 'Refrescar Runs'}
          </button>
        </div>
      </div>

      {/* Warning */}
      <div className="card" style={{ backgroundColor: '#fef3c7', borderLeft: '4px solid #f59e0b', padding: 12 }}>
        <strong>⚠️ Advertencia:</strong> Las alertas generadas no representan fraude confirmado. Requieren revisión humana.
      </div>

      {/* Runs selector */}
      <div className="card">
        <h3>1. Seleccionar Run de Preprocesamiento</h3>
        {loadingRuns && <div>Cargando runs...</div>}
        {!loadingRuns && runs.length === 0 && <div>No hay runs disponibles.</div>}
        {!loadingRuns && runs.length > 0 && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 12 }}>
              {runs.map(run => (
                <div
                  key={run.id}
                  onClick={() => setSelectedRun(run)}
                  style={{
                    padding: 12,
                    border: selectedRun?.id === run.id ? '2px solid #2563eb' : '2px solid #ddd',
                    borderRadius: 8,
                    cursor: 'pointer',
                    backgroundColor: selectedRun?.id === run.id ? '#eff6ff' : '#fff'
                  }}
                >
                  <div style={{ fontWeight: 'bold' }}>Run {run.id}</div>
                  <div style={{ fontSize: 12, color: '#666' }}>{run.file || 'preprocessed_run_' + run.id + '.csv'}</div>
                  <div style={{ fontSize: 11, color: '#999', marginTop: 4 }}>
                    Tamaño: {run.file_size ? (run.file_size / 1024 / 1024).toFixed(2) + ' MB' : 'N/A'}
                  </div>
                  <div style={{ fontSize: 11, color: '#999' }}>
                    Alertas: {run.has_alerts ? 'Sí' : 'No'} | Resumen: {run.has_summary ? 'Sí' : 'No'}
                  </div>
                </div>
              ))}
            </div>
            <table className="table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Archivo</th>
                  <th>Tamaño</th>
                  <th>Fecha</th>
                  <th>Alertas</th>
                  <th>Resumen</th>
                  <th>Reporte</th>
                  <th>Acciones</th>
                </tr>
              </thead>
              <tbody>
                {runs.map(run => (
                  <tr key={run.id} style={{ backgroundColor: selectedRun?.id === run.id ? '#eff6ff' : '' }}>
                    <td>{run.id}</td>
                    <td>{run.file || 'preprocessed_run_' + run.id}</td>
                    <td>{run.file_size ? (run.file_size / 1024 / 1024).toFixed(2) + ' MB' : 'N/A'}</td>
                    <td>{run.created_at || run.date || 'N/A'}</td>
                    <td>{run.has_alerts ? '✓' : '✗'}</td>
                    <td>{run.has_summary ? '✓' : '✗'}</td>
                    <td>{run.has_report ? '✓' : '✗'}</td>
                    <td style={{ display: 'flex', gap: 8 }}>
                      <button className="button" onClick={() => setSelectedRun(run)}>Seleccionar</button>
                      <button className="button" onClick={() => handleViewReport()}>Ver Reporte</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Analysis section */}
      {selectedRun && (
        <div className="card">
          <h3>2. Analizar y Generar Alertas para Run {selectedRun.id}</h3>
          <p style={{ fontSize: 12, color: '#666' }}>
            Ejecutar el motor de reglas en el preprocessed_run_{selectedRun.id}.csv para generar alertas detalladas y resumen agrupado.
          </p>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              className="button"
              onClick={handleAnalyzeAndGenerate}
              disabled={analyzing}
              style={{ backgroundColor: analyzing ? '#ccc' : '#2563eb' }}
            >
              {analyzing ? 'Analizando...' : 'Analizar y Generar Alertas'}
            </button>
            <button
              className="button"
              onClick={handleReprocesar}
              disabled={analyzing}
              style={{ backgroundColor: analyzing ? '#ccc' : '#f59e0b' }}
            >
              Reprocesar (force=true)
            </button>
          </div>
          {analysisMsg && (
            <div style={{ marginTop: 12, padding: 12, backgroundColor: '#f0f9ff', borderRadius: 6, fontSize: 14 }}>
              {analysisMsg}
            </div>
          )}
        </div>
      )}

      {/* Metrics section */}
      {selectedRun && metrics && (
        <div>
          <h3>3. Métricas del Run {selectedRun.id}</h3>
          <div className="kpi-grid">
            <KPICard
              title="Total Alertas Detalladas"
              value={metrics.total_detailed_alerts?.toLocaleString() || '0'}
            />
            <KPICard
              title="Total Alertas Agrupadas"
              value={metrics.total_grouped_alerts?.toLocaleString() || '0'}
            />
            <KPICard
              title="Regla con Más Alertas"
              value={metrics.top_rule_code || 'N/A'}
            />
            <KPICard
              title="Riesgo Predominante"
              value={metrics.top_risk_level || 'N/A'}
            />
            <KPICard
              title="MCC Más Frecuente"
              value={metrics.top_merchant_rubro || 'N/A'}
            />
            <KPICard
              title="Clientes con Más Alertas"
              value={metrics.top_customer_count || '0'}
            />
          </div>
          {metrics.rules_distribution && (
            <div className="card">
              <h4>Distribución por Regla</h4>
              <table className="table">
                <thead>
                  <tr>
                    <th>Regla</th>
                    <th>Cantidad</th>
                    <th>Porcentaje</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(metrics.rules_distribution).map(([rule, count]) => (
                    <tr key={rule}>
                      <td>{rule}</td>
                      <td>{count?.toLocaleString() || 0}</td>
                      <td>
                        {(
                          ((count || 0) / (metrics.total_detailed_alerts || 1)) *
                          100
                        ).toFixed(2)}
                        %
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Filters section */}
      {selectedRun && (
        <div className="card">
          <h3>4. Filtros de Resumen</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 12 }}>
            <div className="form-row">
              <label>Regla:</label>
              <input
                type="text"
                className="input"
                placeholder="Ej: RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY"
                value={filters.rule_code}
                onChange={(e) => setFilters({ ...filters, rule_code: e.target.value })}
              />
            </div>
            <div className="form-row">
              <label>Nivel de Riesgo:</label>
              <select
                className="input"
                value={filters.risk_level}
                onChange={(e) => setFilters({ ...filters, risk_level: e.target.value })}
              >
                <option value="">-- Todos --</option>
                <option value="LOW">Bajo</option>
                <option value="MEDIUM">Medio</option>
                <option value="HIGH">Alto</option>
                <option value="CRITICAL">Crítico</option>
              </select>
            </div>
            <div className="form-row">
              <label>Estado:</label>
              <select
                className="input"
                value={filters.status}
                onChange={(e) => setFilters({ ...filters, status: e.target.value })}
              >
                <option value="">-- Todos --</option>
                <option value="NEW">Nuevo</option>
                <option value="UNDER_REVIEW">En Revisión</option>
                <option value="RESOLVED">Resuelto</option>
              </select>
            </div>
            <div className="form-row">
              <label>País:</label>
              <input
                type="text"
                className="input"
                placeholder="Ej: BO"
                value={filters.country_code}
                onChange={(e) => setFilters({ ...filters, country_code: e.target.value.toUpperCase() })}
              />
            </div>
            <div className="form-row">
              <label>MCC/Rubro:</label>
              <input
                type="text"
                className="input"
                placeholder="Ej: JEWELRY"
                value={filters.merchant_rubro_proxy}
                onChange={(e) => setFilters({ ...filters, merchant_rubro_proxy: e.target.value })}
              />
            </div>
            <div className="form-row">
              <label>Cliente (hash):</label>
              <input
                type="text"
                className="input"
                placeholder="Hash del cliente"
                value={filters.customer_hash}
                onChange={(e) => setFilters({ ...filters, customer_hash: e.target.value })}
              />
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="button" onClick={handleApplyFilters} disabled={loadingSummary}>
              {loadingSummary ? 'Aplicando...' : 'Aplicar Filtros'}
            </button>
            <button className="button" onClick={handleClearFilters} style={{ backgroundColor: '#6b7280' }}>
              Limpiar Filtros
            </button>
          </div>
        </div>
      )}

      {/* Summary table */}
      {selectedRun && (
        <div className="card">
          <h3>5. Tabla de Alertas Agrupadas</h3>
          {loadingSummary && <div>Cargando resumen...</div>}
          {!loadingSummary && summary.length === 0 && (
            <div>No hay alertas para este run. Ejecuta "Analizar y Generar Alertas" primero.</div>
          )}
          {!loadingSummary && summary.length > 0 && (
            <div>
              <p style={{ fontSize: 12, color: '#666', marginBottom: 12 }}>
                Mostrando {summary.length} de {pagination.total} alertas agrupadas (página {pagination.page})
              </p>
              <table className="table">
                <thead>
                  <tr>
                    {summaryColumns.map(col => (
                      <th key={col.key}>{col.title}</th>
                    ))}
                    <th>Acciones</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.map((row, idx) => (
                    <tr key={idx}>
                      {summaryColumns.map(col => (
                        <td key={col.key}>{row[col.key] || 'N/A'}</td>
                      ))}
                      <td>
                        <button
                          className="button"
                          onClick={() => handleViewDetail(row)}
                          style={{ fontSize: 12, padding: '6px 10px' }}
                        >
                          Ver Detalle
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Pagination */}
              <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 16, alignItems: 'center' }}>
                <button
                  className="button"
                  onClick={() => loadSummary(pagination.page - 1)}
                  disabled={!pagination.has_prev}
                >
                  ← Anterior
                </button>
                <span style={{ marginX: 12 }}>
                  Página {pagination.page} de {Math.ceil(pagination.total / pagination.page_size)}
                </span>
                <button
                  className="button"
                  onClick={() => loadSummary(pagination.page + 1)}
                  disabled={!pagination.has_next}
                >
                  Siguiente →
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Detail Modal */}
      {showDetailModal && selectedAlert && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: '#fff',
            borderRadius: 8,
            padding: 24,
            maxWidth: '90%',
            maxHeight: '90vh',
            overflow: 'auto',
            width: '800px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h3>Detalle de Alerta Agrupada</h3>
              <button onClick={() => setShowDetailModal(false)} style={{ fontSize: 20, background: 'none', border: 'none', cursor: 'pointer' }}>×</button>
            </div>

            <div style={{ marginBottom: 16, padding: 12, backgroundColor: '#fef3c7', borderRadius: 6 }}>
              <strong>ℹ️ Nota:</strong> Esta alerta representa una señal de riesgo. No constituye fraude confirmado.
            </div>

            {/* Summary Alert Info */}
            <div style={{ marginBottom: 16 }}>
              <h4>Información de Alerta Agrupada</h4>
              <table className="table">
                <tbody>
                  <tr>
                    <td style={{ fontWeight: 'bold', width: '30%' }}>ID Alerta:</td>
                    <td>{selectedAlert.summary_alert_id || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 'bold' }}>Cliente:</td>
                    <td>{selectedAlert.customer_hash || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 'bold' }}>Regla:</td>
                    <td>{selectedAlert.rule_code || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 'bold' }}>Nombre Regla:</td>
                    <td>{selectedAlert.rule_name || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 'bold' }}>Nivel Riesgo:</td>
                    <td>{selectedAlert.risk_level || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 'bold' }}>Score Máximo:</td>
                    <td>{selectedAlert.max_risk_score || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 'bold' }}>Transacciones Detectadas:</td>
                    <td>{selectedAlert.count_transactions || '0'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 'bold' }}>Países Detectados:</td>
                    <td>{selectedAlert.countries_detected || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 'bold' }}>Ventana Inicio:</td>
                    <td>{selectedAlert.window_start || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 'bold' }}>Ventana Fin:</td>
                    <td>{selectedAlert.window_end || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 'bold' }}>MCC/Rubro:</td>
                    <td>{selectedAlert.merchant_rubro_proxy || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontWeight: 'bold' }}>Estado:</td>
                    <td>{selectedAlert.status || 'NEW'}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Detail Transactions */}
            <div>
              <h4>Transacciones Detalladas ({detailAlerts.length})</h4>
              {loadingDetail && <div>Cargando transacciones...</div>}
              {!loadingDetail && detailAlerts.length === 0 && <div>No hay transacciones detalladas disponibles.</div>}
              {!loadingDetail && detailAlerts.length > 0 && (
                <table className="table" style={{ fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Tx ID</th>
                      <th>Fecha/Hora</th>
                      <th>Monto</th>
                      <th>País</th>
                      <th>POS Mode</th>
                      <th>MCC</th>
                      <th>Score</th>
                      <th>Razón</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detailAlerts.map((alert, idx) => (
                      <tr key={idx}>
                        <td>{alert.alert_id || idx}</td>
                        <td>{alert.transaction_id || 'N/A'}</td>
                        <td>{alert.transaction_datetime || 'N/A'}</td>
                        <td>{alert.amount || '0'}</td>
                        <td>{alert.country_code || 'N/A'}</td>
                        <td>{alert.pos_entry_mode || 'N/A'}</td>
                        <td>{alert.merchant_rubro_proxy || 'N/A'}</td>
                        <td>{alert.risk_score || 'N/A'}</td>
                        <td style={{ maxWidth: 200, wordBreak: 'break-word' }}>{alert.alert_reason || 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

            <div style={{ marginTop: 16, textAlign: 'right' }}>
              <button className="button" onClick={() => setShowDetailModal(false)}>
                Cerrar
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Report Modal */}
      {showReportModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: '#fff',
            borderRadius: 8,
            padding: 24,
            maxWidth: '90%',
            maxHeight: '90vh',
            overflow: 'auto',
            width: '900px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h3>Reporte de Reglas - Run {selectedRun?.id}</h3>
              <button onClick={() => setShowReportModal(false)} style={{ fontSize: 20, background: 'none', border: 'none', cursor: 'pointer' }}>×</button>
            </div>

            {loadingReport && <div>Cargando reporte...</div>}
            {!loadingReport && report && (
              <div>
                <div style={{ marginBottom: 12 }}>
                  <button className="button" onClick={handleCopyReport}>
                    📋 Copiar Contenido
                  </button>
                </div>
                <pre style={{
                  backgroundColor: '#f5f5f5',
                  padding: 16,
                  borderRadius: 6,
                  overflow: 'auto',
                  maxHeight: '400px',
                  fontSize: 12,
                  fontFamily: 'monospace'
                }}>
                  {report}
                </pre>
              </div>
            )}

            <div style={{ marginTop: 16, textAlign: 'right' }}>
              <button className="button" onClick={() => setShowReportModal(false)}>
                Cerrar
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
