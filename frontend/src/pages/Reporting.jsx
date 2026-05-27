import React, { useEffect, useState } from 'react'
import { getReportingSummary } from '../services/api'

export default function Reporting(){
  const [summary, setSummary] = useState(null)
  useEffect(()=>{ getReportingSummary().then(r=> setSummary(r)).catch(()=>{}) },[])
  return (
    <div>
      <div className="header"><h2>Analítica Avanzada</h2></div>
      <div className="card">
        {!summary ? (
          <div>No hay datos de reporte disponibles.</div>
        ) : (
          <div>
            <div>Total alertas: {summary.total_alerts}</div>
            <div>Pérdida potencial: {summary.potential_loss}</div>
          </div>
        )}
      </div>
    </div>
  )
}
