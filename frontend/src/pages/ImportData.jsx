import React, {useState, useEffect, useRef} from 'react'
import { importDatasetBackground, getDatasetStatus, listDatasets } from '../services/api'

export default function ImportData(){
  const [file, setFile] = useState(null)
  const [msg, setMsg] = useState('')
  const [uploading, setUploading] = useState(false)
  const [datasetId, setDatasetId] = useState(null)
  const [status, setStatus] = useState(null)
  const [progress, setProgress] = useState(0)
  const pollRef = useRef(null)

  useEffect(()=>{
    return ()=>{ if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const startPolling = (id)=>{
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = setInterval(async ()=>{
      try{
        const s = await getDatasetStatus(id)
        setStatus(s.status)
        setProgress(s.progress_percent || 0)
        setMsg(`Processed ${s.processed_rows}/${s.total_rows} rows (${s.progress_percent || 0}%)`)
        if (s.status === 'COMPLETED' || s.status === 'FAILED'){
          clearInterval(pollRef.current)
          pollRef.current = null
        }
      }catch(e){
        console.error('poll error', e)
      }
    }, 3000)
  }

  const onSubmit = async (e)=>{
    e.preventDefault()
    setMsg('')
    if (!file) return setMsg('Selecciona un archivo')
    setUploading(true)
    try{
      const res = await importDatasetBackground(file)
      if (res && res.accepted){
        setMsg('Procesamiento iniciado en segundo plano')
        setDatasetId(res.dataset_id)
        setStatus('PROCESSING')
        startPolling(res.dataset_id)
        setFile(null)
      } else {
        setMsg('Error al iniciar el procesamiento')
      }
    }catch(err){ setMsg('Error: '+ (err?.response?.data?.detail || err?.message || err)) }
    finally{ setUploading(false) }
  }

  return (
    <div>
      <div className="header"><h2>Importar datos</h2></div>
      <div className="card">
        <form onSubmit={onSubmit}>
          <div className="form-row">
            <input type="file" onChange={e=>setFile(e.target.files[0])} />
            {file && <div style={{marginTop:8,fontSize:13}}>Archivo: {file.name}</div>}
          </div>
          <div style={{textAlign:'right'}}>
            <button className="button" disabled={uploading} type="submit">{uploading ? 'Subiendo...' : 'Subir'}</button>
          </div>
        </form>
        {msg && <div style={{marginTop:12}} className="card">{msg}</div>}
        {datasetId && <div style={{marginTop:8}}>Dataset id: {datasetId} — status: {status} — {progress}%</div>}
      </div>
    </div>
  )
}
