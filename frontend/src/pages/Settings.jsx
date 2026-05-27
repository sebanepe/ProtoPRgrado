import React, { useEffect, useState } from 'react'
import { getModelConfig, setModelConfig, listModels } from '../services/api'

export default function Settings(){
  const [config, setConfig] = useState(null)
  const [models, setModels] = useState([])
  const [form, setForm] = useState({ active_model_id: null, alert_threshold: 0.7, updated_by: '' })

  useEffect(()=>{
    async function load(){
      const cfg = await getModelConfig()
      const mlist = await listModels()
      setConfig(cfg)
      setModels(mlist)
      if(cfg){ setForm({ active_model_id: cfg.active_model_id, alert_threshold: cfg.alert_threshold ?? 0.7, updated_by: cfg.updated_by ?? '' }) }
    }
    load()
  },[])

  const save = async ()=>{
    // validate threshold
    if (form.alert_threshold < 0 || form.alert_threshold > 1) {
      alert('Umbral debe estar entre 0 y 1')
      return
    }
    const payload = { ...form }
    const res = await setModelConfig(payload)
    setConfig(res)
    alert('Configuración guardada')
  }

  return (
    <div>
      <div className="header"><h2>Configuración</h2></div>
      <div className="card">
        <h3>Configuración del modelo</h3>
        <div style={{marginBottom:10}}>
          <label>Modelo activo: </label>
          <select value={form.active_model_id ?? ''} onChange={e=> setForm({...form, active_model_id: e.target.value ? Number(e.target.value) : null})}>
            <option value="">-- ninguno --</option>
            {models.map(m=> <option key={m.id} value={m.id}>{m.model_name} {m.version?`(${m.version})`:''}</option>)}
          </select>
        </div>
        <div style={{marginBottom:10}}>
          <label>Umbral de alerta: </label>
          <input type="number" step="0.01" min="0" max="1" value={form.alert_threshold} onChange={e=> setForm({...form, alert_threshold: Number(e.target.value)})} />
        </div>
        <div style={{marginBottom:10}}>
          <label>Actualizado por: </label>
          <input value={form.updated_by} onChange={e=> setForm({...form, updated_by: e.target.value})} />
        </div>
        <div>
          <button className="button" onClick={save}>Guardar</button>
        </div>
      </div>
    </div>
  )
}
