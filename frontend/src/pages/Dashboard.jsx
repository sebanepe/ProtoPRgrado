import React, {useEffect, useMemo, useState} from 'react'
import { Link } from 'react-router-dom'
import KPICard from '../components/KPICard'
import { getDashboardOverview } from '../services/api'

const DEFAULT_OVERVIEW = {
  source_run: 'preprocessed_run_26',
  anomaly_run: 'run_26',
  total_transactions: null,
  active_alerts: null,
  average_risk_score: null,
  active_model: null,
  review_distribution: {},
  alerts_evolution: [],
  recent_alerts: [],
  warnings: []
}

const numberFormatter = new Intl.NumberFormat('es-BO')
const percentFormatter = new Intl.NumberFormat('es-BO', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

function formatNumber(value){
  if(value === null || value === undefined || Number.isNaN(Number(value))) return '--'
  return numberFormatter.format(Number(value))
}

function formatRisk(value){
  if(value === null || value === undefined || Number.isNaN(Number(value))) return '--'
  return Number(value).toFixed(2)
}

function formatPercent(value){
  if(value === null || value === undefined || Number.isNaN(Number(value))) return '--'
  return `${percentFormatter.format(Number(value) * 100)} %`
}

function formatDate(value){
  if(!value) return '--'
  const parsed = new Date(value)
  if(Number.isNaN(parsed.getTime())) return String(value)
  return parsed.toLocaleString('es-BO', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

function SimpleLineChart({points=[]}){
  if(!points || points.length===0) {
    return <div style={{height:96, display:'grid', placeItems:'center', color:'var(--text-muted)'}}>No existen alertas para graficar.</div>
  }
  const max = Math.max(...points.map(p=>Number(p.count) || 0), 1)
  const w = 420
  const h = 100
  const pad = 12
  const step = points.length > 1 ? (w - 2 * pad) / (points.length - 1) : 0
  const path = points.map((p,i)=> {
    const x = points.length > 1 ? pad + i * step : w / 2
    const y = h - pad - ((Number(p.count) || 0) / max) * (h - 2 * pad)
    return `${i===0?'M':'L'} ${x} ${y}`
  }).join(' ')
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Evolucion de alertas">
      <path d={path} stroke="var(--accent-blue)" fill="none" strokeWidth={2.5} strokeLinecap="round" />
      {points.map((p,i)=> {
        const x = points.length > 1 ? pad + i * step : w / 2
        const y = h - pad - ((Number(p.count) || 0) / max) * (h - 2 * pad)
        return <circle key={`${p.date}-${i}`} cx={x} cy={y} r="3" fill="var(--accent-blue)" />
      })}
    </svg>
  )
}

function ReviewDistribution({distribution={}}){
  const confirmed = Number(distribution.confirmed_fraud || 0)
  const dismissed = Number(distribution.dismissed || 0)
  const total = confirmed + dismissed
  if(total <= 0){
    return <div style={{height:96, display:'grid', placeItems:'center', color:'var(--text-muted)', textAlign:'center'}}>Sin revisiones humanas suficientes.</div>
  }
  const confirmedPct = Math.round((confirmed / total) * 100)
  const dismissedPct = 100 - confirmedPct
  return (
    <div style={{display:'grid', gap:10}}>
      <div>
        <div style={{display:'flex', justifyContent:'space-between', gap:12}}>
          <span>Confirmed Fraud</span>
          <strong>{confirmedPct}%</strong>
        </div>
        <div style={{height:8, background:'var(--bg-card-soft)', borderRadius:4, overflow:'hidden', marginTop:6}}>
          <div style={{width:`${confirmedPct}%`, height:'100%', background:'var(--danger)'}} />
        </div>
      </div>
      <div>
        <div style={{display:'flex', justifyContent:'space-between', gap:12}}>
          <span>Dismissed</span>
          <strong>{dismissedPct}%</strong>
        </div>
        <div style={{height:8, background:'var(--bg-card-soft)', borderRadius:4, overflow:'hidden', marginTop:6}}>
          <div style={{width:`${dismissedPct}%`, height:'100%', background:'var(--accent-blue)'}} />
        </div>
      </div>
    </div>
  )
}

export default function Dashboard(){
  const [overview, setOverview] = useState(DEFAULT_OVERVIEW)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(()=>{
    let mounted = true
    setLoading(true)
    setError(null)
    getDashboardOverview({ source_run: 'preprocessed_run_26', anomaly_run: 'run_26' })
      .then(data=>{ if(mounted) setOverview({...DEFAULT_OVERVIEW, ...(data || {})}) })
      .catch(()=>{ if(mounted) setError('No se pudo cargar el resumen real del dashboard.') })
      .finally(()=>{ if(mounted) setLoading(false) })
    return ()=>{ mounted=false }
  },[])

  const activeModel = overview.active_model
  const modelValue = activeModel
    ? `${activeModel.model_name || 'Isolation Forest'} / ${activeModel.run_id || overview.anomaly_run}`
    : 'Sin modelo activo'
  const modelDetail = activeModel
    ? `${formatNumber(activeModel.anomaly_count)} anomalias (${formatPercent(activeModel.anomaly_rate)})`
    : ''

  const warnings = useMemo(()=> overview.warnings || [], [overview.warnings])

  return (
    <div>
      <div className="header"><h2>Panel</h2></div>
      {loading && <div className="card" style={{marginBottom:16, color:'var(--text-muted)'}}>Cargando metricas reales...</div>}
      {error && <div className="card" style={{marginBottom:16, color:'var(--danger)'}}>{error}</div>}
      {!error && warnings.length > 0 && (
        <div className="card" style={{marginBottom:16, color:'var(--text-muted)'}}>
          {warnings.join(' ')}
        </div>
      )}

      <div className="kpi-grid">
        <KPICard title="Transacciones analizadas" value={formatNumber(overview.total_transactions)} />
        <KPICard title="Alertas activas" value={formatNumber(overview.active_alerts)} />
        <KPICard title="Riesgo promedio" value={formatRisk(overview.average_risk_score)} />
        <KPICard title="Modelo activo" value={modelValue} />
      </div>
      {modelDetail && <div style={{marginTop:8, color:'var(--text-muted)', fontSize:13}}>Modelo no supervisado: {modelDetail}</div>}

      <div className="dashboard-chart-grid">
        <div className="card">
          <h3>Evolucion de alertas</h3>
          <SimpleLineChart points={overview.alerts_evolution || []} />
        </div>
        <div className="card">
          <h3>Proporcion de revision</h3>
          <ReviewDistribution distribution={overview.review_distribution || {}} />
        </div>
      </div>

      <div className="card" style={{marginTop:16}}>
        <h3>Alertas recientes</h3>
        <div className="table-scroll">
          <table className="table dashboard-alerts-table">
            <thead><tr><th>ID Alerta</th><th>Regla</th><th>Cliente</th><th>Score</th><th>Riesgo</th><th>Estado</th><th>Fecha</th><th>Accion</th></tr></thead>
            <tbody>
              {(overview.recent_alerts || []).map(a=> (
                <tr key={a.alert_id}>
                  <td>{a.alert_id}</td>
                  <td>{a.rule_code || '--'}</td>
                  <td>{a.customer_hash || '--'}</td>
                  <td>{formatRisk(a.risk_score)}</td>
                  <td>{a.risk_level || '--'}</td>
                  <td>{a.status || '--'}</td>
                  <td>{formatDate(a.created_at)}</td>
                  <td><Link to="/rules-alerts">Ver detalle</Link></td>
                </tr>
              ))}
              {(overview.recent_alerts || []).length===0 && <tr><td colSpan={8}>No hay alertas recientes.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
