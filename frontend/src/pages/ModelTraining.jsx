import React, { useEffect, useState } from 'react'
import { listDatasets, trainModel } from '../services/api'

export default function ModelTraining(){
  const [datasets, setDatasets] = useState([])
  const [datasetId, setDatasetId] = useState(null)
  const [algo, setAlgo] = useState('random_forest')
  const [status, setStatus] = useState(null)

  useEffect(()=>{ listDatasets().then(d=> setDatasets(d||[])).catch(()=>{}) },[])

  const run = async ()=>{
    setStatus('running')
    try{
      const res = await trainModel({ dataset_id: datasetId, algorithm: algo })
      setStatus('completed')
      alert('Entrenamiento finalizado')
    }catch(e){
      setStatus('error')
      alert('Error al entrenar')
    }
  }

  return (
    <div>
      <div className="header"><h2>Entrenamiento de Modelos</h2></div>
      <div className="card">
        <div style={{marginBottom:10}}>
          <label>Dataset: </label>
          <select value={datasetId||''} onChange={e=> setDatasetId(e.target.value?Number(e.target.value):null)}>
            <option value="">-- seleccionar --</option>
            {datasets.map(d=> <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
        </div>
        <div style={{marginBottom:10}}>
          <label>Algoritmo: </label>
          <select value={algo} onChange={e=> setAlgo(e.target.value)}>
            <option value="logistic">Logistic Regression</option>
            <option value="random_forest">Random Forest</option>
            <option value="gbm">Gradient Boosting</option>
            <option value="isolation_forest">Isolation Forest</option>
          </select>
        </div>
        <div>
          <button className="button" onClick={run} disabled={status==='running'}>Entrenar modelo</button>
          {status==='running' && <span style={{marginLeft:10}}>Entrenando...</span>}
          {status==='completed' && <span style={{marginLeft:10}}>Finalizado</span>}
        </div>
      </div>
    </div>
  )
}
