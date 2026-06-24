import React, { useEffect, useMemo, useState } from 'react'
import KPICard from '../components/KPICard'
import { getImportAlertTraceabilitySummary } from '../services/api'

const STATUS_COLORS = {
  IMPORTED: 'var(--success)',
  AVAILABLE: 'var(--success)',
  SUCCESS: 'var(--success)',
  COMPLETED: 'var(--success)',
  FAILED: 'var(--danger)',
  MISSING: 'var(--danger)',
  PENDING: 'var(--warning)',
  PARTIAL: 'var(--warning)',
  RUNNING: 'var(--warning)',
  DERIVADO: 'var(--warning)',
}

function StatusBadge({ value }) {
  if (!value) return <span style={{ color: 'var(--text-muted)' }}>—</span>
  const color = STATUS_COLORS[value] || 'var(--text-muted)'
  return <span style={{ color, fontWeight: 600, fontSize: '0.82rem' }}>{value}</span>
}

function PendingCell({ label = 'Sin datos' }) {
  return (
    <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.82rem' }}>
      {label}
    </span>
  )
}

function DateCell({ iso }) {
  if (!iso) return <span style={{ color: 'var(--text-muted)' }}>—</span>
  // Show only the date part (YYYY-MM-DD) to keep table compact
  const date = iso.slice(0, 10)
  const time = iso.slice(11, 16)
  return (
    <span title={iso} style={{ fontSize: '0.8rem' }}>
      {date}
      {time && <span style={{ color: 'var(--text-muted)', marginLeft: '4px' }}>{time}</span>}
    </span>
  )
}

function UserCell({ name }) {
  if (!name) return <span style={{ color: 'var(--text-muted)' }}>—</span>
  return <span style={{ fontSize: '0.82rem' }}>{name}</span>
}

export default function Traceability() {
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getImportAlertTraceabilitySummary()
      .then(data => { setRows(Array.isArray(data) ? data : []); setLoading(false) })
      .catch(e => { setError(e?.message || 'Error al cargar datos de trazabilidad'); setLoading(false) })
  }, [])

  const kpis = useMemo(() => ({
    datasets: new Set(rows.map(r => r.dataset_id)).size,
    prepRuns: rows.filter(r => r.preprocessing_run_id !== null && r.preprocessing_run_id !== undefined).length,
    totalDetailedAlerts: rows.reduce((s, r) => s + (r.detailed_alert_count || 0), 0),
    confirmedByReview: rows.reduce(
      (s, r) => s + (r.detailed_confirmed_by_review_count || 0) + (r.grouped_confirmed_by_review_count || 0), 0
    ),
  }), [rows])

  if (loading) {
    return (
      <div className="content-area">
        <p style={{ color: 'var(--text-muted)', padding: '2rem' }}>Cargando trazabilidad...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="content-area">
        <p style={{ color: 'var(--danger)', padding: '2rem' }}>{error}</p>
      </div>
    )
  }

  return (
    <div className="content-area">
      <div className="page-header" style={{ marginBottom: '1.5rem' }}>
        <h1 style={{ marginBottom: '0.5rem' }}>Trazabilidad: Importación → Alertas</h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '860px' }}>
          Este cuadro permite rastrear cada dataset importado hasta las alertas generadas por el motor
          de reglas, mostrando la relación entre importación, preprocesamiento, reglas, alertas y
          revisión humana.
        </p>
      </div>

      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '1.5rem' }}>
        <KPICard title="Datasets importados" value={kpis.datasets} />
        <KPICard title="Runs de preprocesamiento" value={kpis.prepRuns} />
        <KPICard title="Total alertas detalladas" value={kpis.totalDetailedAlerts} />
        <KPICard title="Alertas confirmadas por analista" value={kpis.confirmedByReview} />
      </div>

      <div style={{
        background: 'rgba(45,140,255,0.07)',
        border: '1px solid rgba(45,140,255,0.3)',
        borderRadius: 'var(--radius-sm)',
        padding: '0.5rem 1rem',
        marginBottom: '0.75rem',
        fontSize: '0.82rem',
        color: 'var(--text-muted)',
      }}>
        El seguimiento del usuario (Importado por / Preprocesado por) está disponible para
        operaciones realizadas después de esta actualización. Las operaciones previas muestran "—".
      </div>

      {rows.some(r => r.rule_run_status === 'DERIVADO') && (
        <div style={{
          background: 'rgba(247,185,85,0.1)',
          border: '1px solid var(--warning)',
          borderRadius: 'var(--radius-sm)',
          padding: '0.6rem 1rem',
          marginBottom: '1rem',
          fontSize: '0.85rem',
          color: 'var(--warning)',
        }}>
          <strong>DERIVADO:</strong> Algunas filas muestran alertas derivadas del archivo CSV porque
          no existe un registro en la tabla de ejecuciones de reglas, pero los artefactos sí están
          disponibles. Los conteos provienen de <em>artifact_registry.row_count</em>.
        </div>
      )}

      <div className="card table-card">
        <div className="table-scroll">
          <table className="table" style={{ fontSize: '0.82rem' }}>
            <thead>
              <tr>
                {/* Dataset */}
                <th>Dataset</th>
                <th>Archivo</th>
                <th>Estado</th>
                <th>Registros</th>
                <th>Importado por</th>
                <th>Fecha importación</th>
                {/* Preprocessing */}
                <th>Run Preproc.</th>
                <th>Estado Preproc.</th>
                <th>Reg. procesados</th>
                <th>Preprocesado por</th>
                <th>Fecha preproc.</th>
                {/* Rule run */}
                <th>Run Reglas</th>
                <th>Estado Reglas</th>
                <th>Fecha reglas</th>
                <th>Alertas Det.</th>
                <th>Alertas Agrup.</th>
                {/* Review */}
                <th>Conf. por revisión (Det.)</th>
                <th>Conf. por revisión (Agrup.)</th>
                {/* Artifacts */}
                <th>CSV Preproc.</th>
                <th>CSV Alertas</th>
                <th>CSV Resumen</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td colSpan={21} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                    No existen datasets importados.
                  </td>
                </tr>
              )}
              {rows.map((row, i) => {
                const hasPrepRun = row.preprocessing_run_id !== null && row.preprocessing_run_id !== undefined
                const hasRuleRun = row.rule_run_id !== null && row.rule_run_id !== undefined
                const isDerived = row.rule_run_status === 'DERIVADO'
                const cfDet = row.detailed_confirmed_by_review_count || 0
                const cfAgr = row.grouped_confirmed_by_review_count || 0
                return (
                  <tr key={i}>
                    {/* Dataset */}
                    <td style={{ fontWeight: 500 }}>{row.dataset_name || '—'}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{row.dataset_filename || '—'}</td>
                    <td><StatusBadge value={row.dataset_status} /></td>
                    <td>{row.dataset_total_records ?? '—'}</td>
                    <td><UserCell name={row.dataset_uploaded_by} /></td>
                    <td><DateCell iso={row.dataset_created_at} /></td>
                    {/* Preprocessing */}
                    <td>
                      {hasPrepRun
                        ? <span>Run #{row.preprocessing_run_id}</span>
                        : <PendingCell label="Sin preprocesamiento" />}
                    </td>
                    <td>
                      {hasPrepRun
                        ? <StatusBadge value={row.preprocessing_run_status} />
                        : <PendingCell />}
                    </td>
                    <td>
                      {hasPrepRun ? (row.preprocessing_processed_records ?? '—') : <PendingCell />}
                    </td>
                    <td>
                      {hasPrepRun ? <UserCell name={row.preprocessing_executed_by} /> : <PendingCell />}
                    </td>
                    <td>
                      {hasPrepRun ? <DateCell iso={row.preprocessing_started_at} /> : <PendingCell />}
                    </td>
                    {/* Rule run */}
                    <td>
                      {hasRuleRun
                        ? <span>Run #{row.rule_run_id}</span>
                        : isDerived
                          ? <PendingCell label="Sin registro" />
                          : <PendingCell label="Sin reglas" />}
                    </td>
                    <td>
                      {(hasRuleRun || isDerived)
                        ? <StatusBadge value={row.rule_run_status} />
                        : <PendingCell />}
                    </td>
                    <td>
                      {hasRuleRun ? <DateCell iso={row.rule_run_created_at} /> : <PendingCell />}
                    </td>
                    <td>{row.detailed_alert_count ?? 0}</td>
                    <td>{row.grouped_alert_count ?? 0}</td>
                    {/* Review */}
                    <td style={{ color: cfDet > 0 ? 'var(--warning)' : undefined, fontWeight: cfDet > 0 ? 600 : undefined }}>
                      {cfDet}
                    </td>
                    <td style={{ color: cfAgr > 0 ? 'var(--warning)' : undefined, fontWeight: cfAgr > 0 ? 600 : undefined }}>
                      {cfAgr}
                    </td>
                    {/* Artifacts */}
                    <td><StatusBadge value={row.artifact_preprocessed_csv} /></td>
                    <td><StatusBadge value={row.artifact_rule_alerts_csv} /></td>
                    <td><StatusBadge value={row.artifact_rule_summary_csv} /></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
