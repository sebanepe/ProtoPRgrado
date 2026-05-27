import React, { useEffect, useState } from 'react'
import { runBatchScoring, getModelConfig } from '../services/api'

export default function BatchScoring(){
  const [running, setRunning] = useState(false)
  const [summary, setSummary] = useState(null)
  const [models, setModels] = useState([])

  useEffect(()=>{ getModelConfig().then(c=> setSummary(c)).catch(()=>{}) },[])

  const run = async ()=>{
    setRunning(true)
    try{
      // For now send empty transactions to backend which will create none but exercise endpoint
      const res = await runBatchScoring({ transactions: [] })
      alert('Scoring ejecutado')
    }catch(e){ alert('Error al ejecutar scoring') }
    setRunning(false)
  }

  return (
    <div>
      <div className="header"><h2>Scoring por Lotes</h2></div>
      <div className="card">
        <div>Modelo activo: {summary && summary.active_model_id ? summary.active_model_id : '—'}</div>
        <div>Umbral: {summary && summary.alert_threshold ? summary.alert_threshold : '—'}</div>
        <div style={{marginTop:10}}>
          <button className="button" onClick={run} disabled={running}>{running? 'Ejecutando...':'Ejecutar scoring'}</button>
        </div>
      </div>
    </div>
  )
}
