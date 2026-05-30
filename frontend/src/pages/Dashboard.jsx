import React, {useEffect, useState} from 'react'
import KPICard from '../components/KPICard'
import { getDashboardSummary } from '../services/api'

function SimpleLineChart({points=[]}){
  // points: [{date, count}]
  if(!points || points.length===0) return <div style={{height:80, display:'grid', placeItems:'center', color:'var(--text-muted)'}}>No data</div>
  const max = Math.max(...points.map(p=>p.count))
  const w = 300, h=80, pad=10
  const step = (w-2*pad)/(points.length-1)
  const path = points.map((p,i)=> `${i===0?'M':'L'} ${pad + i*step} ${h - pad - (p.count/max)*(h-2*pad)}`).join(' ')
  return <svg width={w} height={h}><path d={path} stroke="var(--accent-blue)" fill="none" strokeWidth={2.5} strokeLinecap="round" /></svg>
}

function SimplePie({fraud=0, normal=100}){
  const total = fraud+normal || 1
  const fraudPct = fraud/total
  const radius = 40, cx=50, cy=50
  const a = 0, b = fraudPct * Math.PI*2
  const x1 = cx + radius * Math.cos(a)
  const y1 = cy + radius * Math.sin(a)
  const x2 = cx + radius * Math.cos(b)
  const y2 = cy + radius * Math.sin(b)
  const large = fraudPct>0.5?1:0
  const d = `M ${cx} ${cy} L ${x1} ${y1} A ${radius} ${radius} 0 ${large} 1 ${x2} ${y2} Z`
  return (
    <svg width={100} height={100} viewBox="0 0 100 100">
      <circle cx={cx} cy={cy} r={radius} fill="var(--bg-card-soft)" />
      <path d={d} fill="var(--danger)" />
      <circle cx={cx} cy={cy} r={radius-18} fill="var(--bg-card)" />
    </svg>
  )
}

export default function Dashboard(){
  const [summary, setSummary] = useState({transactions:0, alerts:0, risk:0, model:'--', alertTrend:[], fraudRatio:{fraud:0, normal:100}, recentAlerts:[]})
  useEffect(()=>{
    let mounted = true
    getDashboardSummary().then(s=>{ if(mounted) setSummary(s) }).catch(()=>{})
    return ()=>{ mounted=false }
  },[])

  return (
    <div>
      <div className="header"><h2>Panel</h2></div>
      <div className="kpi-grid">
        <KPICard title="Transacciones analizadas" value={summary.transactions} />
        <KPICard title="Alertas activas" value={summary.alerts} />
        <KPICard title="Riesgo promedio" value={summary.risk} />
        <KPICard title="Modelo activo" value={summary.model} />
      </div>

      <div style={{display:'flex', gap:16, marginTop:16}}>
        <div style={{flex:1}} className="card">
          <h3>Evolución de alertas</h3>
          <SimpleLineChart points={summary.alertTrend||[]} />
        </div>
        <div style={{width:220}} className="card">
          <h3>Proporción fraude</h3>
          <div style={{display:'flex', alignItems:'center', gap:12}}>
            <SimplePie fraud={summary.fraudRatio?.fraud} normal={summary.fraudRatio?.normal} />
            <div>
              <div><strong>Fraude:</strong> {summary.fraudRatio?.fraud}%</div>
              <div><strong>Normal:</strong> {summary.fraudRatio?.normal}%</div>
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{marginTop:16}}>
        <h3>Alertas recientes</h3>
        <table className="table">
          <thead><tr><th>ID Alerta</th><th>Tx</th><th>Puntuación</th><th>Canal</th><th>Monto</th><th>Estado</th></tr></thead>
          <tbody>
            {(summary.recentAlerts||[]).map(a=> (
              <tr key={a.alert_id}>
                <td>{a.alert_id}</td>
                <td>{a.transaction_id}</td>
                <td>{a.score}</td>
                <td>{a.channel}</td>
                <td>{a.amount}</td>
                <td>{a.status}</td>
              </tr>
            ))}
            {(summary.recentAlerts||[]).length===0 && <tr><td colSpan={6}>No hay alertas recientes</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  )
}
