import React, { useState, useEffect } from 'react'
import Table from '../components/Table'
import KPICard from '../components/KPICard'
import {
  getPreprocessedRuns,
  analyzeRules,
  getRulesSummary,
  getSummaryFilterOptions,
  getRulesReport,
  getRulesMetrics,
  getRulesAlerts,
  getRuleAlertDetail,
  updateAlertReviewStatus,
  updateSummaryAlertReviewStatus,
  getAlertReviewHistory,
  getSummaryAlertReviewHistory
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

const extractRunNumber = (runId) => {
  if (!runId) return -1
  const value = String(runId)
  const matches = value.match(/(\d+)(?!.*\d)/)
  if (!matches) return -1
  const parsed = Number.parseInt(matches[1], 10)
  return Number.isNaN(parsed) ? -1 : parsed
}

const getBestDefaultRun = (runs) => {
  if (!Array.isArray(runs) || runs.length === 0) return null

  const sortedRuns = [...runs].sort((first, second) => {
    const firstNum = extractRunNumber(getRunId(first))
    const secondNum = extractRunNumber(getRunId(second))
    return secondNum - firstNum
  })

  const withSummaryAndAlerts = sortedRuns.find((run) => run?.has_summary === true && run?.has_alerts === true)
  if (withSummaryAndAlerts) return withSummaryAndAlerts

  const withSummary = sortedRuns.find((run) => run?.has_summary === true)
  if (withSummary) return withSummary

  const withAlerts = sortedRuns.find((run) => run?.has_alerts === true)
  if (withAlerts) return withAlerts

  return sortedRuns[0]
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
  const [summaryError, setSummaryError] = useState('')
  const [pagination, setPagination] = useState({
    page: 1,
    page_size: 20,
    total: 0,
    total_pages: 0,
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
  const [filterOptions, setFilterOptions] = useState({
    rule_code: [],
    risk_level: [],
    status: [],
    country_code: [],
    merchant_rubro_proxy: [],
    customer_hash: []
  })
  const [loadingFilterOptions, setLoadingFilterOptions] = useState(false)
  const [filterOptionsError, setFilterOptionsError] = useState('')
  const lastLoadedFilterRun = React.useRef(null)

  // PHASE B.3: State for alert review
  const [reviewHistory, setReviewHistory] = useState([])
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [showHistoryTab, setShowHistoryTab] = useState(false)
  const [selectedStatusForUpdate, setSelectedStatusForUpdate] = useState('IN_REVIEW')
  const [analystNotes, setAnalystNotes] = useState('')
  const [updatingStatus, setUpdatingStatus] = useState(false)
  const [reviewMsg, setReviewMsg] = useState('')

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

  useEffect(() => {
    if (selectedRun) {
      loadSummaryFilterOptions()
    }
  }, [selectedRun])

  const loadRuns = async () => {
    setLoadingRuns(true)
    try {
      const runsData = getResponseItems(await getPreprocessedRuns()).map(normalizeRun)
      setRuns(runsData || [])
      if (runsData && runsData.length > 0) {
        setSelectedRun(getBestDefaultRun(runsData))
      } else {
        setSelectedRun(null)
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
      await Promise.all([loadMetrics(), loadSummary(1, filters)])
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

  const loadSummaryFilterOptions = async () => {
    if (!selectedRun || typeof getSummaryFilterOptions !== 'function') return
    const runId = getRunId(selectedRun)

    if (selectedRun?.has_summary !== true) {
      lastLoadedFilterRun.current = null
      setFilterOptionsError('')
      setFilterOptions({
        rule_code: [],
        risk_level: [],
        status: [],
        country_code: [],
        merchant_rubro_proxy: [],
        customer_hash: []
      })
      return
    }

    // Avoid reloading options repeatedly for the same run
    if (lastLoadedFilterRun.current === runId) return
    lastLoadedFilterRun.current = runId
    setLoadingFilterOptions(true)
    setFilterOptionsError('')
    try {
      const options = await getSummaryFilterOptions(runId)
      setFilterOptions({
        rule_code: options.rule_code || [],
        risk_level: options.risk_level || [],
        status: options.status || [],
        country_code: options.country_code || [],
        merchant_rubro_proxy: options.merchant_rubro_proxy || [],
        customer_hash: options.customer_hash || []
      })
    } catch (e) {
      console.error('Error loading summary filter options:', e)
      const statusCode = e?.response?.status || null
      if (statusCode === 404) {
        setFilterOptionsError('No se pudieron cargar opciones de filtros para este run.')
      } else {
        setFilterOptionsError('No se pudieron cargar opciones de filtros. Se puede seguir usando la tabla sin filtros dinámicos.')
      }
      setFilterOptions({
        rule_code: [],
        risk_level: [],
        status: [],
        country_code: [],
        merchant_rubro_proxy: [],
        customer_hash: []
      })
    } finally {
      setLoadingFilterOptions(false)
    }
  }

  const loadSummary = async (page = 1, activeFilters = filters) => {
    if (!selectedRun) return
    setLoadingSummary(true)
    setSummaryError('')
    try {
      const runId = getRunId(selectedRun)
      const normalizedFilters = Object.fromEntries(
        Object.entries(activeFilters || {})
          .map(([key, value]) => [key, typeof value === 'string' ? value.trim() : value])
          .filter(([, value]) => value !== '' && value !== null && value !== undefined)
      )
      const params = {
        page: page,
        page_size: pagination.page_size,
        ...normalizedFilters
      }
      const summaryData = await getRulesSummary(runId, params)
      const summaryItems = getResponseItems(summaryData)
      setSummary(summaryItems)
      setPagination({
        page: summaryData.page ?? page ?? 1,
        page_size: summaryData.page_size ?? params.page_size ?? 20,
        total: summaryData.total_items ?? summaryData.total ?? summaryItems.length,
        total_pages: summaryData.total_pages ?? 0,
        has_next: summaryData.total_pages ? (summaryData.page ?? page ?? 1) < summaryData.total_pages : (summaryData.has_next || false),
        has_prev: summaryData.total_pages ? (summaryData.page ?? page ?? 1) > 1 : (summaryData.has_prev || false)
      })
    } catch (e) {
      setSummary([])
      setPagination(prev => ({ ...prev, total: 0, total_pages: 0, has_next: false, has_prev: false }))
      setSummaryError(e?.response?.data?.detail || e?.message || String(e))
      console.error('Error loading summary:', e)
    } finally {
      setLoadingSummary(false)
    }
  }

  const handleApplyFilters = () => {
    const activeFilters = { ...filters }
    if (import.meta.env.DEV) {
      console.debug('Applying summary filters', activeFilters)
    }
    setPagination(prev => ({ ...prev, page: 1 }))
    loadSummary(1, activeFilters)
  }

  const handleClearFilters = () => {
    const clearedFilters = {
      rule_code: '',
      risk_level: '',
      status: '',
      country_code: '',
      merchant_rubro_proxy: '',
      customer_hash: ''
    }
    setFilters(clearedFilters)
    setPagination(prev => ({ ...prev, page: 1 }))
    loadSummary(1, clearedFilters)
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
      await Promise.all([loadMetrics(), loadSummary(1, filters)])
    } catch (e) {
      setAnalysisMsg('Error al reprocesar: ' + (e?.response?.data?.detail || e?.message || String(e)))
    } finally {
      setAnalyzing(false)
    }
  }

  // PHASE B.3: Load review history for alert
  const loadReviewHistory = async (alert) => {
    setLoadingHistory(true)
    setReviewMsg('')
    try {
      const runId = getRunId(selectedRun)
      if (alert.summary_alert_id) {
        const history = await getSummaryAlertReviewHistory(alert.summary_alert_id, runId)
        setReviewHistory(history.history || [])
      } else if (alert.alert_id) {
        const history = await getAlertReviewHistory(alert.alert_id, runId)
        setReviewHistory(history.history || [])
      }
    } catch (e) {
      console.error('Error loading review history:', e)
      setReviewHistory([])
    } finally {
      setLoadingHistory(false)
    }
  }

  // PHASE B.3: Update alert status
  const handleUpdateAlertStatus = async () => {
    if (!selectedAlert || !selectedRun) return
    if (!selectedStatusForUpdate) {
      setReviewMsg('Por favor selecciona un estado')
      return
    }

    setUpdatingStatus(true)
    setReviewMsg('Actualizando estado...')
    try {
      const runId = getRunId(selectedRun)
      if (selectedAlert.summary_alert_id) {
        await updateSummaryAlertReviewStatus(
          selectedAlert.summary_alert_id,
          runId,
          selectedStatusForUpdate,
          analystNotes
        )
      } else if (selectedAlert.alert_id) {
        await updateAlertReviewStatus(
          selectedAlert.alert_id,
          runId,
          selectedStatusForUpdate,
          analystNotes
        )
      }
      setReviewMsg(`✓ Estado actualizado a ${selectedStatusForUpdate}`)
      // Reload history
      await loadReviewHistory(selectedAlert)
      // Reload summary to show updated status
      await loadSummary(pagination.page, filters)
    } catch (e) {
      setReviewMsg('Error al actualizar estado: ' + (e?.response?.data?.detail || e?.message || String(e)))
    } finally {
      setUpdatingStatus(false)
    }
  }

  const hasActiveSummaryFilters = Object.values(filters).some((value) => String(value || '').trim() !== '')

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
                  className={selectedRun?.id === run.id ? 'run-selector-card selected-run-card' : 'run-selector-card'}
                  style={{
                    padding: 12,
                    cursor: 'pointer'
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
                  <tr key={run.id} className={selectedRun?.id === run.id ? 'selected-run-row' : ''}>
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
          {loadingFilterOptions && (
            <div style={{ marginBottom: 12, fontSize: 12, color: '#6b7280' }}>
              Cargando opciones de filtro...
            </div>
          )}
          {filterOptionsError && (
            <div style={{ marginBottom: 12, padding: 10, backgroundColor: '#fee2e2', borderRadius: 6, color: '#991b1b' }}>
              {filterOptionsError}
            </div>
          )}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 12 }}>
            <div className="form-row">
              <label>Regla:</label>
              <select
                className="input"
                aria-label="Regla"
                value={filters.rule_code}
                  onChange={(e) => setFilters((prev) => ({ ...prev, rule_code: e.target.value }))}
              >
                <option value="">-- Todas --</option>
                {filterOptions.rule_code.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <label>Nivel de Riesgo:</label>
              <select
                className="input"
                aria-label="Nivel de Riesgo"
                value={filters.risk_level}
                  onChange={(e) => setFilters((prev) => ({ ...prev, risk_level: e.target.value }))}
              >
                <option value="">-- Todos --</option>
                {(filterOptions.risk_level.length > 0 ? filterOptions.risk_level : ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']).map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <label>Estado:</label>
              <select
                className="input"
                aria-label="Estado"
                value={filters.status}
                  onChange={(e) => setFilters((prev) => ({ ...prev, status: e.target.value }))}
              >
                <option value="">-- Todos --</option>
                {(filterOptions.status.length > 0 ? filterOptions.status : ['NEW', 'IN_REVIEW', 'DISMISSED', 'FALSE_POSITIVE', 'CONFIRMED_FRAUD']).map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <label>País:</label>
              <select
                className="input"
                aria-label="País"
                value={filters.country_code}
                  onChange={(e) => setFilters((prev) => ({ ...prev, country_code: e.target.value }))}
              >
                <option value="">-- Todos --</option>
                {filterOptions.country_code.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <label>MCC/Rubro:</label>
              <select
                className="input"
                aria-label="MCC/Rubro"
                value={filters.merchant_rubro_proxy}
                  onChange={(e) => setFilters((prev) => ({ ...prev, merchant_rubro_proxy: e.target.value }))}
              >
                <option value="">-- Todos --</option>
                {filterOptions.merchant_rubro_proxy.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <label>Cliente (hash):</label>
              <input
                type="text"
                className="input"
                placeholder="Hash del cliente"
                list="customer-hash-options"
                value={filters.customer_hash}
                  onChange={(e) => setFilters((prev) => ({ ...prev, customer_hash: e.target.value }))}
              />
              {filterOptions.customer_hash.length > 0 && (
                <datalist id="customer-hash-options">
                  {filterOptions.customer_hash.slice(0, 100).map((option) => (
                    <option key={option} value={option} />
                  ))}
                </datalist>
              )}
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
          {summaryError && !loadingSummary && (
            <div style={{ marginBottom: 12, padding: 12, backgroundColor: '#fee2e2', borderRadius: 6, color: '#991b1b' }}>
              Error al cargar el resumen: {summaryError}
            </div>
          )}
          {loadingSummary && <div>Cargando resumen...</div>}
          {!loadingSummary && !summaryError && summary.length === 0 && (
            <div>
              {pagination.total === 0
                ? (hasActiveSummaryFilters
                  ? 'No hay coincidencias para los filtros seleccionados.'
                  : (selectedRun?.has_summary ? 'No hay alertas agrupadas para este run.' : 'Ejecuta "Analizar y Generar Alertas" primero.'))
                : (pagination.total_pages && pagination.page > pagination.total_pages
                  ? 'No hay alertas para esta página.'
                  : (hasActiveSummaryFilters ? 'No hay coincidencias para los filtros seleccionados.' : 'No hay alertas disponibles en este momento.'))}
            </div>
          )}
          {!loadingSummary && summary.length > 0 && (
            <div>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
                Mostrando {summary.length} de {pagination.total} alertas agrupadas (página {pagination.page})
              </p>
              <div className="table-scroll">
                <table className="table alert-summary-table">
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
                        <td style={{ display: 'flex', gap: 6, whiteSpace: 'nowrap' }}>
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
              </div>

              {/* Pagination */}
              <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 16, alignItems: 'center' }}>
                <button
                  className="button"
                  onClick={() => loadSummary(pagination.page - 1, filters)}
                  disabled={!pagination.has_prev}
                >
                  ← Anterior
                </button>
                <span style={{ marginX: 12 }}>
                  Página {pagination.page} de {pagination.total_pages || Math.ceil(pagination.total / pagination.page_size)}
                </span>
                <button
                  className="button"
                  onClick={() => loadSummary(pagination.page + 1, filters)}
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
            backgroundColor: '#101b2a',
            color: 'var(--text-main)',
            borderRadius: 8,
            padding: 24,
            maxWidth: '90%',
            maxHeight: '90vh',
            overflow: 'auto',
            width: '900px',
            border: '1px solid rgba(36, 52, 71, 0.95)',
            boxShadow: '0 18px 48px rgba(0, 0, 0, 0.5)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h3>Detalle de Alerta Agrupada</h3>
              <button className="icon-button" onClick={() => setShowDetailModal(false)} style={{ fontSize: 20 }}>×</button>
            </div>

            <div className="detail-note" style={{ marginBottom: 16, padding: 12, borderRadius: 6 }}>
              <strong>ℹ️ Nota:</strong> Esta alerta representa una señal de riesgo. No constituye fraude confirmado.
            </div>

            {/* Summary Alert Info */}
            <div className="detail-section" style={{ marginBottom: 16 }}>
              <h4>Información de Alerta Agrupada</h4>
              <div className="table-scroll">
                <table className="table detail-table">
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
                    <td style={{ fontWeight: 'bold' }}>Estado Actual:</td>
                    <td style={{ fontWeight: 'bold', color: selectedAlert.status === 'CONFIRMED_FRAUD' ? '#dc2626' : '#2563eb' }}>
                      {selectedAlert.status || 'NEW'}
                    </td>
                  </tr>
                </tbody>
                </table>
              </div>
            </div>

            {/* PHASE B.3: Status Update Section */}
            <div className="detail-section" style={{ marginBottom: 16, padding: 12, borderRadius: 6 }}>
              <h4>Revisión Humana - Cambiar Estado</h4>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
                ⚠️ Confirmar fraude es una decisión del analista. El sistema no confirma fraude automáticamente.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                <div className="form-row">
                  <label>Nuevo Estado:</label>
                  <select
                    className="input"
                    value={selectedStatusForUpdate}
                    onChange={(e) => setSelectedStatusForUpdate(e.target.value)}
                  >
                    <option value="NEW">Nuevo</option>
                    <option value="IN_REVIEW">En Revisión</option>
                    <option value="DISMISSED">Desestimado</option>
                    <option value="FALSE_POSITIVE">Falso Positivo</option>
                    <option value="CONFIRMED_FRAUD">Fraude Confirmado (Manual)</option>
                  </select>
                </div>
              </div>
              <div className="form-row" style={{ marginBottom: 12 }}>
                <label>Observaciones del Analista:</label>
                <textarea
                  className="input"
                  placeholder="Escriba observaciones sobre esta alerta..."
                  value={analystNotes}
                  onChange={(e) => setAnalystNotes(e.target.value)}
                  rows="3"
                  style={{ resize: 'vertical' }}
                />
              </div>
              <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                <button
                  className="button"
                  onClick={handleUpdateAlertStatus}
                  disabled={updatingStatus}
                  style={{ backgroundColor: updatingStatus ? '#ccc' : '#2563eb' }}
                >
                  {updatingStatus ? 'Guardando...' : 'Guardar Revisión'}
                </button>
                <button
                  className="button"
                  onClick={() => {
                    setShowHistoryTab(!showHistoryTab)
                    if (!showHistoryTab) {
                      loadReviewHistory(selectedAlert)
                    }
                  }}
                  style={{ backgroundColor: '#6b7280' }}
                >
                  {showHistoryTab ? 'Ocultar' : 'Ver'} Historial
                </button>
              </div>
              {reviewMsg && (
                <div className={reviewMsg.includes('Error') ? 'detail-status detail-status-error' : 'detail-status detail-status-success'} style={{ padding: 12, borderRadius: 6, fontSize: 12 }}>
                  {reviewMsg}
                </div>
              )}
            </div>

            {/* Review History */}
            {showHistoryTab && (
              <div className="detail-section" style={{ marginBottom: 16, padding: 12, borderRadius: 6 }}>
                <h4>Historial de Revisiones</h4>
                {loadingHistory && <div>Cargando historial...</div>}
                {!loadingHistory && reviewHistory.length === 0 && (
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No hay revisiones previas para esta alerta.</div>
                )}
                {!loadingHistory && reviewHistory.length > 0 && (
                  <div className="table-scroll">
                    <table className="table detail-table" style={{ fontSize: 11 }}>
                      <thead>
                        <tr>
                          <th>Fecha</th>
                          <th>Estado Anterior</th>
                          <th>Nuevo Estado</th>
                          <th>Observaciones</th>
                          <th>Revisor ID</th>
                        </tr>
                      </thead>
                      <tbody>
                        {reviewHistory.map((entry, idx) => (
                          <tr key={idx}>
                            <td>{new Date(entry.reviewed_at).toLocaleString()}</td>
                            <td>{entry.previous_status || 'N/A'}</td>
                            <td style={{ fontWeight: 'bold', color: entry.new_status === 'CONFIRMED_FRAUD' ? '#ff7b89' : '#6aa9ff' }}>
                              {entry.new_status}
                            </td>
                            <td style={{ maxWidth: 250, wordBreak: 'break-word', fontSize: 11 }}>
                              {entry.analyst_notes || '-'}
                            </td>
                            <td>{entry.reviewed_by_id || '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {/* Detail Transactions */}
            <div>
              <h4>Transacciones Detalladas ({detailAlerts.length})</h4>
              {loadingDetail && <div>Cargando transacciones...</div>}
              {!loadingDetail && detailAlerts.length === 0 && <div>No hay transacciones detalladas disponibles.</div>}
              {!loadingDetail && detailAlerts.length > 0 && (
                <div className="table-scroll">
                  <table className="table detail-table" style={{ fontSize: 12 }}>
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
                </div>
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
            backgroundColor: '#101b2a',
            color: 'var(--text-main)',
            borderRadius: 8,
            padding: 24,
            maxWidth: '90%',
            maxHeight: '90vh',
            overflow: 'auto',
            width: '900px',
            border: '1px solid rgba(36, 52, 71, 0.95)',
            boxShadow: '0 18px 48px rgba(0, 0, 0, 0.5)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h3>Reporte de Reglas - Run {selectedRun?.id}</h3>
              <button className="icon-button" onClick={() => setShowReportModal(false)} style={{ fontSize: 20 }}>×</button>
            </div>

            {loadingReport && <div>Cargando reporte...</div>}
            {!loadingReport && report && (
              <div>
                <div style={{ marginBottom: 12 }}>
                  <button className="button" onClick={handleCopyReport}>
                    📋 Copiar Contenido
                  </button>
                </div>
                <pre className="report-pre" style={{
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
