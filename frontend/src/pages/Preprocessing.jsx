import React, {useState, useEffect} from 'react'
import { runPreprocessing, listDatasets, previewDataset, deleteDataset, listPreprocessingRuns, previewPreprocessingRun } from '../services/api'

export default function Preprocessing(){
  const [msg,setMsg] = useState('')
  const [datasets, setDatasets] = useState([])
  const [preview, setPreview] = useState(null)
  const [runs, setRuns] = useState([])
  const [runPreview, setRunPreview] = useState(null)
  const [loadingDatasets, setLoadingDatasets] = useState(false)

  const run = async ()=>{
    setMsg('Ejecutando...')
    try{
      const res = await runPreprocessing()
      setMsg('Listo: '+ JSON.stringify(res))
      await loadRuns()
    }catch(e){ setMsg('Error: '+ (e?.message||e)) }
  }

  const load = async ()=>{
    setLoadingDatasets(true)
    try{
      const ds = await listDatasets()
      setDatasets(ds)
    }catch(e){ console.error(e); }
    setLoadingDatasets(false)
  }

  const handlePreview = async (id)=>{
    setPreview({ loading: true })
    try{
      const p = await previewDataset(id, 20)
      setPreview({ loading: false, data: p })
    }catch(e){
      const errMsg = e?.response?.data?.detail || e?.message || String(e)
      setPreview({ loading: false, error: errMsg })
    }
  }

  const handleDelete = async (id)=>{
    if(!window.confirm('¿Borrar dataset #' + id + '? Esta acción no se puede deshacer.')) return
    try{
      await deleteDataset(id)
      await load()
      setMsg('Dataset ' + id + ' borrado')
    }catch(e){ setMsg('Error borrando dataset: ' + (e?.response?.data?.detail || e?.message || String(e))) }
  }

  useEffect(()=>{ load() }, [])

  const loadRuns = async ()=>{
    try{
      const r = await listPreprocessingRuns()
      setRuns(r||[])
    }catch(e){ console.error('loadRuns', e) }
  }

  useEffect(()=>{ loadRuns() }, [])

  const handleRunPreview = async (id)=>{
    setRunPreview({ loading: true })
    try{
      const p = await previewPreprocessingRun(id)
      setRunPreview({ loading: false, data: p })
    }catch(e){ setRunPreview({ loading: false, error: e?.response?.data?.detail || e?.message || String(e) }) }
  }

  return (
    <div>
      <div className="header"><h2>Preprocesamiento</h2>
        <div style={{display:'flex',gap:8}}>
          <button className="button" onClick={run}>Ejecutar preprocesamiento</button>
          <button className="button" onClick={load}>Refrescar datasets</button>
        </div>
      </div>

      <div className="card">
        <h3>Datasets importados</h3>
        {loadingDatasets && <div>Cargando...</div>}
        {!loadingDatasets && datasets.length===0 && <div>No hay datasets importados.</div>}
        {!loadingDatasets && datasets.length>0 && (
          <table className="table">
            <thead><tr><th>Id</th><th>Nombre</th><th>Archivo</th><th>Total</th><th>Válidos</th><th>Inválidos</th><th>Acciones</th></tr></thead>
            <tbody>
              {datasets.map(d=> (
                <tr key={d.id}>
                  <td>{d.id}</td>
                  <td>{d.name}</td>
                  <td>{d.original_filename}</td>
                  <td>{d.total_records}</td>
                  <td>{d.valid_records}</td>
                  <td>{d.invalid_records}</td>
                  <td style={{display:'flex',gap:8}}>
                    <button className="button" onClick={()=>handlePreview(d.id)}>Previsualizar</button>
                    <button className="button danger" onClick={()=>handleDelete(d.id)}>Borrar</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

          <div style={{marginTop:20}}>
            <h3>Ejecuciones de preprocesamiento</h3>
            {runs.length===0 && <div>No hay ejecuciones aún.</div>}
            {runs.length>0 && (
              <table className="table">
                <thead><tr><th>Id</th><th>Estado</th><th>Total</th><th>Procesados</th><th>Removidos</th><th>Acciones</th></tr></thead>
                <tbody>
                  {runs.map(rr=> (
                    <tr key={rr.id}>
                      <td>{rr.id}</td>
                      <td>{rr.status}</td>
                      <td>{rr.total_records}</td>
                      <td>{rr.processed_records}</td>
                      <td>{rr.removed_records}</td>
                      <td><button className="button" onClick={()=>handleRunPreview(rr.id)}>Previsualizar run</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

        {preview && preview.loading && <div>Obteniendo previsualización...</div>}
        {preview && preview.error && <div>Error: {preview.error}</div>}
        {preview && preview.data && (
          <div style={{marginTop:12}}>
            <h4>Previsualización: {preview.data.dataset_id}</h4>
            <table className="table small">
              <thead>
                <tr>{preview.data.columns.map(c=> <th key={c}>{c}</th>)}</tr>
              </thead>
              <tbody>
                {preview.data.rows.map((r,idx)=> (
                  <tr key={idx}>{preview.data.columns.map(c=> <td key={c}>{String(r[c] ?? '')}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {runPreview && runPreview.loading && <div>Obteniendo previsualización del run...</div>}
        {runPreview && runPreview.error && <div>Error: {runPreview.error}</div>}
        {runPreview && runPreview.data && (
          <div style={{marginTop:12}}>
            <h4>Run {runPreview.data.run.id} — Estado: {runPreview.data.run.status}</h4>
            <div style={{display:'flex',gap:20}}>
              <div style={{flex:1}}>
                <h5>Antes (DB)</h5>
                <table className="table small">
                  <thead><tr><th>transaction_id</th><th>amount</th><th>type</th><th>location</th><th>datetime</th><th>is_fraud</th></tr></thead>
                  <tbody>
                    {runPreview.data.before.map((r,idx)=> (
                      <tr key={idx}><td>{r.transaction_id}</td><td>{r.amount}</td><td>{r.transaction_type}</td><td>{r.location}</td><td>{String(r.transaction_datetime)}</td><td>{String(r.is_fraud)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{flex:1}}>
                <h5>Después (Procesado)</h5>
                <table className="table small">
                  <thead>
                    <tr>{(runPreview.data.after.length>0 ? Object.keys(runPreview.data.after[0]).slice(0,8) : []).map(c=> <th key={c}>{c}</th>)}</tr>
                  </thead>
                  <tbody>
                    {runPreview.data.after.map((r,idx)=> (
                      <tr key={idx}>{Object.keys(r).slice(0,8).map(c=> <td key={c}>{String(r[c] ?? '')}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

      </div>
      <div style={{marginTop:10}} className="card">{msg || 'Presiona para ejecutar SMOTE y transformar datos'}</div>
    </div>
  )
}
