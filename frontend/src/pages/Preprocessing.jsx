import React, {useState, useEffect} from 'react'
import { runPreprocessing, listDatasets, previewDataset, deleteDataset } from '../services/api'

export default function Preprocessing(){
  const [msg,setMsg] = useState('')
  const [datasets, setDatasets] = useState([])
  const [preview, setPreview] = useState(null)
  const [loadingDatasets, setLoadingDatasets] = useState(false)

  const run = async ()=>{
    setMsg('Ejecutando...')
    try{
      const res = await runPreprocessing()
      setMsg('Listo: '+ JSON.stringify(res))
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

      </div>
      <div style={{marginTop:10}} className="card">{msg || 'Presiona para ejecutar SMOTE y transformar datos'}</div>
    </div>
  )
}
