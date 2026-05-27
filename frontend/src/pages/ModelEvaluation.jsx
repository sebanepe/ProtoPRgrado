import React, { useEffect, useState } from 'react'
import Table from '../components/Table'
import { getModelResults, activateModel, exportModelResults } from '../services/api'

export default function ModelEvaluation(){
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(()=>{
    async function load(){
      setLoading(true)
      const res = await getModelResults()
      setModels(res||[])
      setLoading(false)
    }
    load()
  },[])

  const cols = [
    { key:'model_name', title:'Nombre' },
    { key:'roc_auc', title:'ROC AUC' },
    { key:'recall', title:'Recall' },
    { key:'precision', title:'Precision' },
    { key:'f1_score', title:'F1-Score' },
    { key:'is_active', title:'Estado' }
  ]

  const onActivate = async (id)=>{
    try{ await activateModel(id); alert('Modelo activado'); const res = await getModelResults(); setModels(res) }catch(e){ alert('Error al activar') }
  }

  const onExport = async (id)=>{
    try{
      const blob = await exportModelResults(id)
      const url = window.URL.createObjectURL(new Blob([blob]))
      const a = document.createElement('a'); a.href = url; a.download = `model_${id}_results.csv`; document.body.appendChild(a); a.click(); a.remove()
    }catch(e){ alert('Export error') }
  }

  return (
    <div>
      <div className="header"><h2>Evaluación de Modelos</h2></div>
      {loading ? <div>Loading...</div> : (
        <div>
          <Table columns={cols} data={models.map(m=> ({...m, is_active: m.is_active? 'Active':'Inactive'}))} />
          <div style={{marginTop:10}}>
            {models.map(m => (
              <div key={m.id} style={{display:'inline-block', marginRight:10}}>
                <button className="button" onClick={()=> onActivate(m.id)}>Activar</button>
                <button className="button" onClick={()=> onExport(m.id)} style={{marginLeft:6}}>Exportar resultados</button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
