import React, { useEffect, useMemo, useState } from 'react'
import { useLocation } from 'react-router-dom'
import {
  addCaseComment,
  closeCase,
  createCase,
  createCaseFromScoringResult,
  getCaseById,
  getCaseComments,
  getCaseHistory,
  getBatchScoringResults,
  getBatchScoringRuns,
  getCases,
  getCasesSummary,
  reopenCase,
  updateCase,
} from '../services/api'

const STATUS_OPTIONS = ['OPEN', 'IN_ANALYSIS', 'ESCALATED', 'CLOSED']
const PRIORITY_OPTIONS = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
const ORIGIN_OPTIONS = ['RULE_ALERT', 'SCORING_RESULT', 'MODEL_EVALUATION', 'MANUAL']
const METHODOLOGY_MESSAGE = 'El manejo de casos permite dar seguimiento operativo a eventos sospechosos. La creación o cierre de un caso no confirma fraude automáticamente.'

const EMPTY_CASE_FORM = {
  source_run: '',
  origin_type: 'MANUAL',
  origin_ref_id: '',
  summary_alert_id: '',
  transaction_id: '',
  scoring_run_id: '',
  customer_hash: '',
  title: '',
  description: '',
  priority: 'MEDIUM',
  assigned_to: '',
}

const EMPTY_FILTERS = {
  source_run: '',
  status: '',
  priority: '',
  origin_type: '',
  customer_hash: '',
  assigned_to: '',
}

function formatDate(value) {
  if (!value) return '-'
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString('es-ES')
}

function extractError(e, fallback = 'Backend no disponible. Verifique que el servicio esté activo.') {
  const detail = e?.response?.data?.detail
  if (!detail) return fallback
  return typeof detail === 'string' ? detail : JSON.stringify(detail)
}

function cleanPayload(payload) {
  return Object.fromEntries(
    Object.entries(payload)
      .map(([k, v]) => [k, typeof v === 'string' ? v.trim() : v])
      .filter(([, v]) => v !== '' && v !== null && v !== undefined)
  )
}

function Badge({ value, type = 'neutral' }) {
  const styles = {
    OPEN: ['rgba(45,140,255,0.18)', '#cfe3ff', 'rgba(45,140,255,0.35)'],
    IN_ANALYSIS: ['rgba(247,185,85,0.18)', '#ffe9bf', 'rgba(247,185,85,0.35)'],
    ESCALATED: ['rgba(255,92,108,0.18)', '#fecdd3', 'rgba(255,92,108,0.35)'],
    CLOSED: ['rgba(39,209,127,0.16)', '#a7f3d0', 'rgba(39,209,127,0.3)'],
    LOW: ['rgba(39,209,127,0.16)', '#a7f3d0', 'rgba(39,209,127,0.3)'],
    MEDIUM: ['rgba(247,185,85,0.18)', '#ffe9bf', 'rgba(247,185,85,0.35)'],
    HIGH: ['rgba(255,129,92,0.18)', '#fed7aa', 'rgba(255,129,92,0.35)'],
    CRITICAL: ['rgba(255,92,108,0.2)', '#fecdd3', 'rgba(255,92,108,0.4)'],
    neutral: ['rgba(255,255,255,0.08)', 'var(--text-soft)', 'rgba(148,163,184,0.18)'],
  }
  const [background, color, border] = styles[value] || styles[type] || styles.neutral
  return (
    <span style={{ display: 'inline-block', padding: '3px 10px', borderRadius: 999, background, color, border: `1px solid ${border}`, fontSize: 12, fontWeight: 800 }}>
      {value || '-'}
    </span>
  )
}

function Metric({ title, value, badge }) {
  return (
    <div className="card metric-card">
      <div className="metric-label">{title}</div>
      <div className="metric-value">{value ?? 0}</div>
      {badge && <div><Badge value={badge} /></div>}
    </div>
  )
}

function Field({ label, value }) {
  return (
    <div className="metadata-item">
      <div className="metric-label">{label}</div>
      <div className="metadata-value">{value || '-'}</div>
    </div>
  )
}

export default function Cases() {
  const location = useLocation()
  const [summary, setSummary] = useState({ total: 0, by_status: {}, by_priority: {} })
  const [cases, setCases] = useState([])
  const [filters, setFilters] = useState(EMPTY_FILTERS)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const [showCreate, setShowCreate] = useState(false)
  const [caseForm, setCaseForm] = useState(EMPTY_CASE_FORM)
  const [createError, setCreateError] = useState('')

  const [selectedCase, setSelectedCase] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState('')
  const [editForm, setEditForm] = useState({ title: '', description: '', priority: 'MEDIUM', status: 'OPEN', assigned_to: '' })

  const [comments, setComments] = useState([])
  const [history, setHistory] = useState([])
  const [commentText, setCommentText] = useState('')
  const [commentError, setCommentError] = useState('')
  const [closeConclusion, setCloseConclusion] = useState('')
  const [closeError, setCloseError] = useState('')
  const [scoringRuns, setScoringRuns] = useState([])
  const [selectedScoringRun, setSelectedScoringRun] = useState('')
  const [scoringResults, setScoringResults] = useState([])
  const [scoringLoading, setScoringLoading] = useState(false)
  const [scoringError, setScoringError] = useState('')

  const filteredCases = useMemo(() => {
    return cases.filter(c => {
      if (filters.source_run && !String(c.source_run || '').toLowerCase().includes(filters.source_run.toLowerCase())) return false
      if (filters.customer_hash && !String(c.customer_hash || '').toLowerCase().includes(filters.customer_hash.toLowerCase())) return false
      if (filters.assigned_to && !String(c.assigned_to || '').toLowerCase().includes(filters.assigned_to.toLowerCase())) return false
      return true
    })
  }, [cases, filters.source_run, filters.customer_hash, filters.assigned_to])

  async function loadSummary() {
    const data = await getCasesSummary()
    setSummary(data || { total: 0, by_status: {}, by_priority: {} })
  }

  async function loadCases(nextFilters = filters) {
    setLoading(true)
    setError('')
    try {
      const params = { page: 1, page_size: 200 }
      if (nextFilters.status) params.status = nextFilters.status
      if (nextFilters.priority) params.priority = nextFilters.priority
      if (nextFilters.origin_type) params.origin_type = nextFilters.origin_type
      const [summaryData, casesData] = await Promise.all([getCasesSummary(), getCases(params)])
      setSummary(summaryData || { total: 0, by_status: {}, by_priority: {} })
      setCases(casesData?.items || [])
    } catch (e) {
      setError(extractError(e))
      setCases([])
    } finally {
      setLoading(false)
    }
  }

  async function loadCaseDetail(id) {
    setDetailLoading(true)
    setDetailError('')
    setCommentError('')
    setCloseError('')
    try {
      const [caseData, commentsData, historyData] = await Promise.all([
        getCaseById(id),
        getCaseComments(id),
        getCaseHistory(id),
      ])
      setSelectedCase(caseData)
      setEditForm({
        title: caseData.title || '',
        description: caseData.description || '',
        priority: caseData.priority || 'MEDIUM',
        status: caseData.status || 'OPEN',
        assigned_to: caseData.assigned_to || '',
      })
      setComments(commentsData || [])
      setHistory(historyData || [])
    } catch (e) {
      setDetailError(extractError(e, 'Caso no encontrado.'))
      setSelectedCase(null)
      setComments([])
      setHistory([])
    } finally {
      setDetailLoading(false)
    }
  }

  async function loadScoringRuns() {
    setScoringLoading(true)
    setScoringError('')
    try {
      const data = await getBatchScoringRuns()
      const completed = (data?.items || []).filter(r => r.status === 'COMPLETED')
      setScoringRuns(completed)
      if (completed.length > 0) setSelectedScoringRun(String(completed[0].id))
    } catch (e) {
      setScoringError(extractError(e, 'No se pudieron cargar las ejecuciones de scoring.'))
      setScoringRuns([])
    } finally {
      setScoringLoading(false)
    }
  }

  async function loadScoringResults() {
    const run = scoringRuns.find(r => String(r.id) === String(selectedScoringRun))
    if (!run) {
      setScoringError('Seleccione una ejecución de scoring completada.')
      return
    }
    setScoringLoading(true)
    setScoringError('')
    try {
      const data = await getBatchScoringResults({
        source_run: run.source_run,
        algorithm: run.algorithm,
        page: 1,
        page_size: 50,
      })
      setScoringResults(data?.rows || [])
    } catch (e) {
      setScoringError(extractError(e, 'No se pudieron cargar los resultados de scoring.'))
      setScoringResults([])
    } finally {
      setScoringLoading(false)
    }
  }

  function scoringPriority(level) {
    if (level === 'HIGH') return 'HIGH'
    if (level === 'LOW') return 'LOW'
    return 'MEDIUM'
  }

  function useScoringResultAsCase(row) {
    const run = scoringRuns.find(r => String(r.id) === String(selectedScoringRun)) || {}
    const summaryId = row.summary_alert_id || ''
    const txId = row.representative_transaction_id || row.transaction_id || ''
    const level = row.ml_risk_level || 'MEDIUM'
    const score = row.ml_risk_score === null || row.ml_risk_score === undefined ? '-' : row.ml_risk_score
    setCaseForm({
      ...EMPTY_CASE_FORM,
      source_run: run.source_run || row.source_run || '',
      origin_type: 'SCORING_RESULT',
      origin_ref_id: summaryId || txId || String(row.id || ''),
      summary_alert_id: summaryId,
      transaction_id: txId,
      scoring_run_id: String(row.batch_scoring_run_id || run.id || ''),
      customer_hash: row.customer_hash || '',
      title: `Resultado de scoring ${level}${summaryId ? ` - ${summaryId}` : ''}`,
      description: `Resultado de scoring con prioridad ${level} y score ${score}. Revisar operativamente antes de cualquier conclusión.`,
      priority: scoringPriority(level),
      assigned_to: '',
    })
    setCreateError('')
    setShowCreate(true)
  }

  useEffect(() => {
    loadCases()
  }, [])

  useEffect(() => {
    const incoming = location.state?.casePayload || location.state?.scoringCasePayload
    if (incoming) {
      setCaseForm(f => ({ ...f, ...incoming, origin_type: incoming.origin_type || 'SCORING_RESULT' }))
      setShowCreate(true)
    }
  }, [location.state])

  function setFilter(name, value) {
    const next = { ...filters, [name]: value }
    setFilters(next)
    if (['status', 'priority', 'origin_type'].includes(name)) loadCases(next)
  }

  async function handleCreateCase(e) {
    e.preventDefault()
    setCreateError('')
    const payload = cleanPayload(caseForm)
    if (!payload.title) {
      setCreateError('El título es requerido.')
      return
    }
    if (!payload.priority) {
      setCreateError('La prioridad es requerida.')
      return
    }
    if (!payload.origin_type) {
      setCreateError('El origen es requerido.')
      return
    }
    if (!payload.description && !payload.origin_ref_id) {
      setCreateError('Debe registrar una descripción u origin_ref_id.')
      return
    }
    try {
      const created = payload.origin_type === 'SCORING_RESULT'
        ? await createCaseFromScoringResult(payload)
        : await createCase(payload)
      setMessage('Caso creado correctamente.')
      setShowCreate(false)
      setCaseForm(EMPTY_CASE_FORM)
      await loadCases()
      await loadCaseDetail(created.id)
    } catch (e) {
      setCreateError(extractError(e, 'No se pudo crear el caso.'))
    }
  }

  async function handleUpdateCase() {
    if (!selectedCase) return
    try {
      const payload = cleanPayload(editForm)
      const updated = await updateCase(selectedCase.id, payload)
      setSelectedCase(updated)
      setMessage('Caso actualizado correctamente.')
      await Promise.all([loadCases(), loadCaseDetail(selectedCase.id)])
    } catch (e) {
      setDetailError(extractError(e, 'No se pudo actualizar el caso.'))
    }
  }

  async function handleAddComment() {
    setCommentError('')
    if (!commentText.trim()) {
      setCommentError('No se permite comentario vacío.')
      return
    }
    try {
      await addCaseComment(selectedCase.id, { comment_text: commentText.trim() })
      setCommentText('')
      setMessage('Comentario agregado correctamente.')
      await loadCaseDetail(selectedCase.id)
    } catch (e) {
      setCommentError(extractError(e, 'No se pudo agregar el comentario.'))
    }
  }

  async function handleCloseCase() {
    setCloseError('')
    if (!closeConclusion.trim()) {
      setCloseError('Debe registrar una conclusión para cerrar el caso.')
      return
    }
    try {
      await closeCase(selectedCase.id, { conclusion: closeConclusion.trim() })
      setCloseConclusion('')
      setMessage('Caso cerrado correctamente.')
      await Promise.all([loadCases(), loadCaseDetail(selectedCase.id), loadSummary()])
    } catch (e) {
      setCloseError(extractError(e, 'No se pudo cerrar el caso.'))
    }
  }

  async function handleReopenCase() {
    try {
      await reopenCase(selectedCase.id)
      setMessage('Caso reabierto correctamente.')
      await Promise.all([loadCases(), loadCaseDetail(selectedCase.id), loadSummary()])
    } catch (e) {
      setDetailError(extractError(e, 'No se pudo reabrir el caso.'))
    }
  }

  return (
    <div className="models-page">
      <div className="header">
        <div>
          <h2>Manejo de Casos</h2>
          <div className="page-subtitle">Seguimiento operativo de alertas, predicciones y eventos sospechosos.</div>
        </div>
        <button className="button" onClick={() => setShowCreate(true)}>Nuevo caso</button>
      </div>

      <div className="card warning-banner" role="alert">
        <strong>Aviso metodológico:</strong> {METHODOLOGY_MESSAGE}
      </div>
      {message && <div className="status-banner status-success">{message}</div>}

      <div className="card">
        <h3>Resumen de casos</h3>
        <div className="kpi-grid">
          <Metric title="Total de casos" value={summary.total || 0} />
          <Metric title="Abiertos" value={summary.by_status?.OPEN || 0} badge="OPEN" />
          <Metric title="En análisis" value={summary.by_status?.IN_ANALYSIS || 0} badge="IN_ANALYSIS" />
          <Metric title="Escalados" value={summary.by_status?.ESCALATED || 0} badge="ESCALATED" />
          <Metric title="Cerrados" value={summary.by_status?.CLOSED || 0} badge="CLOSED" />
          <Metric title="Alta prioridad" value={summary.by_priority?.HIGH || 0} badge="HIGH" />
          <Metric title="Críticos" value={summary.by_priority?.CRITICAL || 0} badge="CRITICAL" />
        </div>
        <div className="action-row">
          <div>{STATUS_OPTIONS.map(s => <Badge key={s} value={s} />)}</div>
          <div>{PRIORITY_OPTIONS.map(p => <Badge key={p} value={p} />)}</div>
        </div>
      </div>

      <div className="card">
        <h3>Filtros</h3>
        <div className="filters-grid">
          <div className="form-row">
            <label htmlFor="case-source-run">source_run</label>
            <input id="case-source-run" value={filters.source_run} onChange={e => setFilter('source_run', e.target.value)} />
          </div>
          <div className="form-row">
            <label htmlFor="case-status">Estado</label>
            <select id="case-status" value={filters.status} onChange={e => setFilter('status', e.target.value)}>
              <option value="">Todos</option>
              {STATUS_OPTIONS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="form-row">
            <label htmlFor="case-priority">Prioridad</label>
            <select id="case-priority" value={filters.priority} onChange={e => setFilter('priority', e.target.value)}>
              <option value="">Todas</option>
              {PRIORITY_OPTIONS.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="form-row">
            <label htmlFor="case-origin-type">Origen</label>
            <select id="case-origin-type" value={filters.origin_type} onChange={e => setFilter('origin_type', e.target.value)}>
              <option value="">Todos</option>
              {ORIGIN_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>
          <div className="form-row">
            <label htmlFor="case-customer">customer_hash</label>
            <input id="case-customer" value={filters.customer_hash} onChange={e => setFilter('customer_hash', e.target.value)} />
          </div>
          <div className="form-row">
            <label htmlFor="case-assigned">assigned_to</label>
            <input id="case-assigned" value={filters.assigned_to} onChange={e => setFilter('assigned_to', e.target.value)} />
          </div>
        </div>
      </div>

      <div className="card">
        <div className="action-row">
          <h3>Crear desde scoring</h3>
          <button className="button button-secondary" onClick={loadScoringRuns}>Cargar ejecuciones</button>
        </div>
        <p className="section-help">
          Seleccione una ejecución de scoring completada para crear un caso operativo con los datos del resultado. Esto no modifica scoring ni confirma fraude.
        </p>
        {scoringRuns.length > 0 && (
          <div className="filters-grid">
            <div className="form-row">
              <label htmlFor="scoring-run-select">Ejecución de scoring</label>
              <select id="scoring-run-select" value={selectedScoringRun} onChange={e => setSelectedScoringRun(e.target.value)}>
                {scoringRuns.map(r => (
                  <option key={r.id} value={r.id}>
                    {r.id} - {r.source_run} - {r.algorithm}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-row">
              <label htmlFor="scoring-load-results">Resultados</label>
              <button id="scoring-load-results" className="button" onClick={loadScoringResults} type="button">Ver resultados de scoring</button>
            </div>
          </div>
        )}
        {scoringLoading && <div className="empty-state">Cargando scoring...</div>}
        {scoringError && <div className="status-banner status-error">{scoringError}</div>}
        {!scoringLoading && scoringRuns.length === 0 && !scoringError && (
          <div className="empty-state">Cargue ejecuciones de scoring para seleccionar resultados y crear casos prellenados.</div>
        )}
        {scoringResults.length > 0 && (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>ID alerta</th>
                  <th>ID transacción</th>
                  <th>Cliente anonimizado</th>
                  <th>Score</th>
                  <th>Prioridad scoring</th>
                  <th>Modelo</th>
                  <th>Fecha scoring</th>
                  <th>Acción</th>
                </tr>
              </thead>
              <tbody>
                {scoringResults.map((row, idx) => (
                  <tr key={`${row.summary_alert_id || row.representative_transaction_id || idx}`}>
                    <td>{row.summary_alert_id || '-'}</td>
                    <td>{row.representative_transaction_id || row.transaction_id || '-'}</td>
                    <td>{row.customer_hash || '-'}</td>
                    <td>{row.ml_risk_score ?? '-'}</td>
                    <td><Badge value={row.ml_risk_level} /></td>
                    <td>{row.algorithm || '-'}</td>
                    <td>{formatDate(row.scored_at)}</td>
                    <td><button className="button table-row-action" onClick={() => useScoringResultAsCase(row)}>Usar como caso</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card">
        <h3>Tabla de casos</h3>
        {loading && <div className="empty-state">Cargando casos...</div>}
        {error && <div className="status-banner status-error">{error}</div>}
        {!loading && !error && filteredCases.length === 0 && (
          <div className="empty-state">No existen casos registrados. Cree un caso manualmente o desde un resultado de scoring.</div>
        )}
        {!loading && !error && filteredCases.length > 0 && (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Código</th>
                  <th>Título</th>
                  <th>source_run</th>
                  <th>Origen</th>
                  <th>Prioridad</th>
                  <th>Estado</th>
                  <th>Cliente anonimizado</th>
                  <th>assigned_to</th>
                  <th>created_at</th>
                  <th>Acción</th>
                </tr>
              </thead>
              <tbody>
                {filteredCases.map(c => (
                  <tr key={c.id}>
                    <td>{c.case_code || '-'}</td>
                    <td>{c.title || '-'}</td>
                    <td>{c.source_run || '-'}</td>
                    <td><Badge value={c.origin_type} /></td>
                    <td><Badge value={c.priority} /></td>
                    <td><Badge value={c.status} /></td>
                    <td>{c.customer_hash || '-'}</td>
                    <td>{c.assigned_to || '-'}</td>
                    <td>{formatDate(c.created_at)}</td>
                    <td><button className="button table-row-action" onClick={() => loadCaseDetail(c.id)}>Ver detalle</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card">
        <h3>Detalle del caso</h3>
        {detailLoading && <div className="empty-state">Cargando detalle...</div>}
        {detailError && <div className="status-banner status-error">{detailError}</div>}
        {!detailLoading && !detailError && !selectedCase && <div className="empty-state">Seleccione un caso para ver el detalle.</div>}
        {selectedCase && (
          <>
            <div className="metadata-grid">
              <Field label="case_code" value={selectedCase.case_code} />
              <Field label="title" value={selectedCase.title} />
              <Field label="description" value={selectedCase.description} />
              <Field label="source_run" value={selectedCase.source_run} />
              <Field label="origin_type" value={selectedCase.origin_type} />
              <Field label="origin_ref_id" value={selectedCase.origin_ref_id} />
              <Field label="summary_alert_id" value={selectedCase.summary_alert_id} />
              <Field label="transaction_id" value={selectedCase.transaction_id} />
              <Field label="scoring_run_id" value={selectedCase.scoring_run_id} />
              <Field label="customer_hash" value={selectedCase.customer_hash} />
              <Field label="priority" value={selectedCase.priority} />
              <Field label="status" value={selectedCase.status} />
              <Field label="assigned_to" value={selectedCase.assigned_to} />
              <Field label="created_by" value={selectedCase.created_by} />
              <Field label="closed_by" value={selectedCase.closed_by} />
              <Field label="conclusion" value={selectedCase.conclusion} />
              <Field label="created_at" value={formatDate(selectedCase.created_at)} />
              <Field label="updated_at" value={formatDate(selectedCase.updated_at)} />
              <Field label="closed_at" value={formatDate(selectedCase.closed_at)} />
            </div>

            <div className="card detail-section" style={{ marginTop: 16 }}>
              <h4>Editar caso</h4>
              <div className="filters-grid">
                <div className="form-row">
                  <label htmlFor="edit-title">Título</label>
                  <input id="edit-title" value={editForm.title} onChange={e => setEditForm(f => ({ ...f, title: e.target.value }))} />
                </div>
                <div className="form-row">
                  <label htmlFor="edit-priority">Prioridad</label>
                  <select id="edit-priority" value={editForm.priority} onChange={e => setEditForm(f => ({ ...f, priority: e.target.value }))}>
                    {PRIORITY_OPTIONS.map(p => <option key={p} value={p}>{p}</option>)}
                  </select>
                </div>
                <div className="form-row">
                  <label htmlFor="edit-status">Estado</label>
                  <select id="edit-status" value={editForm.status} onChange={e => setEditForm(f => ({ ...f, status: e.target.value }))}>
                    {STATUS_OPTIONS.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
                <div className="form-row">
                  <label htmlFor="edit-assigned">Asignación</label>
                  <input id="edit-assigned" value={editForm.assigned_to} onChange={e => setEditForm(f => ({ ...f, assigned_to: e.target.value }))} />
                </div>
              </div>
              <div className="form-row">
                <label htmlFor="edit-description">Descripción</label>
                <textarea id="edit-description" rows="3" value={editForm.description} onChange={e => setEditForm(f => ({ ...f, description: e.target.value }))} />
              </div>
              <button className="button" onClick={handleUpdateCase}>Guardar cambios</button>
            </div>

            <div className="card detail-section">
              <h4>Formulario de cierre</h4>
              <p className="section-help">Cerrar un caso registra una conclusión operativa, pero no genera fraude confirmado automático.</p>
              {selectedCase.status === 'CLOSED' ? (
                <button className="button success" onClick={handleReopenCase}>Reabrir caso</button>
              ) : (
                <>
                  <div className="form-row">
                    <label htmlFor="close-conclusion">Conclusión</label>
                    <textarea id="close-conclusion" rows="3" value={closeConclusion} onChange={e => setCloseConclusion(e.target.value)} />
                  </div>
                  <button className="button danger" onClick={handleCloseCase}>Cerrar caso</button>
                </>
              )}
              {closeError && <div className="status-banner status-error">{closeError}</div>}
            </div>
          </>
        )}
      </div>

      {selectedCase && (
        <div className="card">
          <h3>Comentarios</h3>
          {comments.length === 0 && <div className="empty-state">No hay comentarios registrados.</div>}
          {comments.length > 0 && (
            <div className="table-scroll">
              <table className="table small">
                <thead><tr><th>Usuario</th><th>Comentario</th><th>Fecha</th></tr></thead>
                <tbody>
                  {comments.map(c => (
                    <tr key={c.id}>
                      <td>{c.user_id || '-'}</td>
                      <td>{c.comment_text || '-'}</td>
                      <td>{formatDate(c.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="form-row" style={{ marginTop: 14 }}>
            <label htmlFor="new-comment">Agregar comentario</label>
            <textarea id="new-comment" rows="3" value={commentText} onChange={e => setCommentText(e.target.value)} />
          </div>
          <button className="button" onClick={handleAddComment}>Agregar comentario</button>
          {commentError && <div className="status-banner status-error">{commentError}</div>}
        </div>
      )}

      {selectedCase && (
        <div className="card">
          <h3>Historial</h3>
          {history.length === 0 && <div className="empty-state">No hay historial registrado.</div>}
          {history.length > 0 && (
            <div className="table-scroll">
              <table className="table small">
                <thead><tr><th>Acción</th><th>Valor anterior</th><th>Valor nuevo</th><th>Usuario</th><th>Fecha</th></tr></thead>
                <tbody>
                  {history.map(h => (
                    <tr key={h.id}>
                      <td>{h.action || '-'}</td>
                      <td>{h.old_value || '-'}</td>
                      <td>{h.new_value || '-'}</td>
                      <td>{h.changed_by || '-'}</td>
                      <td>{formatDate(h.changed_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {showCreate && (
        <div className="modal-backdrop">
          <div className="modal-panel">
            <div className="modal-header">
              <h3>Nuevo caso</h3>
              <button className="button button-secondary" onClick={() => setShowCreate(false)}>Cerrar</button>
            </div>
            <form onSubmit={handleCreateCase}>
              <div className="filters-grid">
                <div className="form-row"><label htmlFor="new-source-run">source_run</label><input id="new-source-run" value={caseForm.source_run} onChange={e => setCaseForm(f => ({ ...f, source_run: e.target.value }))} /></div>
                <div className="form-row"><label htmlFor="new-origin-type">origin_type</label><select id="new-origin-type" value={caseForm.origin_type} onChange={e => setCaseForm(f => ({ ...f, origin_type: e.target.value }))}>{ORIGIN_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}</select></div>
                <div className="form-row"><label htmlFor="new-origin-ref">origin_ref_id</label><input id="new-origin-ref" value={caseForm.origin_ref_id} onChange={e => setCaseForm(f => ({ ...f, origin_ref_id: e.target.value }))} /></div>
                <div className="form-row"><label htmlFor="new-summary-alert">summary_alert_id</label><input id="new-summary-alert" value={caseForm.summary_alert_id} onChange={e => setCaseForm(f => ({ ...f, summary_alert_id: e.target.value }))} /></div>
                <div className="form-row"><label htmlFor="new-transaction">transaction_id</label><input id="new-transaction" value={caseForm.transaction_id} onChange={e => setCaseForm(f => ({ ...f, transaction_id: e.target.value }))} /></div>
                <div className="form-row"><label htmlFor="new-scoring-run">scoring_run_id</label><input id="new-scoring-run" value={caseForm.scoring_run_id} onChange={e => setCaseForm(f => ({ ...f, scoring_run_id: e.target.value }))} /></div>
                <div className="form-row"><label htmlFor="new-customer">customer_hash</label><input id="new-customer" value={caseForm.customer_hash} onChange={e => setCaseForm(f => ({ ...f, customer_hash: e.target.value }))} /></div>
                <div className="form-row"><label htmlFor="new-priority">priority</label><select id="new-priority" value={caseForm.priority} onChange={e => setCaseForm(f => ({ ...f, priority: e.target.value }))}>{PRIORITY_OPTIONS.map(p => <option key={p} value={p}>{p}</option>)}</select></div>
                <div className="form-row"><label htmlFor="new-assigned">assigned_to</label><input id="new-assigned" value={caseForm.assigned_to} onChange={e => setCaseForm(f => ({ ...f, assigned_to: e.target.value }))} /></div>
              </div>
              <div className="form-row"><label htmlFor="new-title">title</label><input id="new-title" value={caseForm.title} onChange={e => setCaseForm(f => ({ ...f, title: e.target.value }))} /></div>
              <div className="form-row"><label htmlFor="new-description">description</label><textarea id="new-description" rows="4" value={caseForm.description} onChange={e => setCaseForm(f => ({ ...f, description: e.target.value }))} /></div>
              {createError && <div className="status-banner status-error">{createError}</div>}
              <button className="button" type="submit">Crear caso</button>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
