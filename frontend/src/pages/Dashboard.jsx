import React, {useEffect, useState} from 'react'
import KPICard from '../components/KPICard'
import { health, listAlerts, listModels } from '../services/api'

export default function Dashboard(){
  const [kpi, setKpi] = useState({transactions:0, alerts:0, risk:0, model:'--'})
  useEffect(()=>{
    health().then(h=>{})
    listAlerts().then(data=> setKpi(k=> ({...k, alerts: (data?.length||0)})))
    listModels().then(models=> setKpi(k=> ({...k, model: models && models.length? models[0].model_name : '--'}))).catch(()=>{})
  },[])

  return (
    <div>
      <div className="header"><h2>Dashboard</h2></div>
      <div className="kpi-grid">
        <KPICard title="Transacciones analizadas" value={kpi.transactions} />
        <KPICard title="Alertas activas" value={kpi.alerts} />
        <KPICard title="Riesgo promedio" value={kpi.risk} />
        <KPICard title="Modelo activo" value={kpi.model} />
      </div>
      <div className="card">
        <h3>Resumen rápido</h3>
        <p>Usa el menú para importar datos, preprocesar y entrenar modelos.</p>
      </div>
    </div>
  )
}
