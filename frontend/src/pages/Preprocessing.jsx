import React, {useState, useEffect, useRef} from 'react'
import { runPreprocessing, listDatasets, previewDataset, deleteDataset, listPreprocessingRuns, previewPreprocessingRun, downloadPreprocessingRun, deletePreprocessingRun, downloadPreprocessingRunReport } from '../services/api'

export default function Preprocessing(){
  const [msg,setMsg] = useState('')
  const [showResultModal, setShowResultModal] = useState(false)
  const [resultDetails, setResultDetails] = useState(null)
  const [showResultDetailsJson, setShowResultDetailsJson] = useState(false)
  const [datasets, setDatasets] = useState([])
  const [preview, setPreview] = useState(null)
  const [runs, setRuns] = useState([])
  const [runPreview, setRunPreview] = useState(null)
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  const runsPollRef = useRef(null)

  const isActiveRun = (run) => ['PENDING', 'RUNNING'].includes(String(run?.status || '').toUpperCase())

  const stopRunsPolling = () => {
    if (runsPollRef.current) {
      clearInterval(runsPollRef.current)
      runsPollRef.current = null
    }
  }

  const startRunsPolling = () => {
    if (runsPollRef.current) return
    runsPollRef.current = setInterval(async () => {
      const latestRuns = await loadRuns()
      if (!latestRuns.some(isActiveRun)) {
        stopRunsPolling()
        setMsg((current) => current.includes('preprocesamiento') || current.includes('Ejecutando') ? 'Preprocesamiento completado' : current)
      }
    }, 3000)
  }

  const run = async ()=>{
    setMsg('Ejecutando...')
    try{
      const res = await runPreprocessing()
      setMsg('Preprocesamiento en ejecución. Actualizando estado...')
      setResultDetails(res)
      setShowResultDetailsJson(false)
      setShowResultModal(true)
      await loadRuns()
      startRunsPolling()
    }catch(e){ setMsg('Error: '+ (e?.message||e)) }
  }

  const handleRunDataset = async (datasetId) => {
    setMsg('Ejecutando preprocesamiento para dataset ' + datasetId + '...')
    try{
      // call backend with selected dataset id so processing is scoped
      const res = await runPreprocessing(datasetId)
      setMsg('Preprocesamiento en ejecución. Actualizando estado...')
      setResultDetails(res)
      setShowResultDetailsJson(false)
      setShowResultModal(true)
      await loadRuns()
      startRunsPolling()
    }catch(e){ setMsg('Error: '+ (e?.response?.data?.detail || e?.message || String(e))) }
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
      if ((r || []).some(isActiveRun)) {
        startRunsPolling()
      } else {
        stopRunsPolling()
      }
      return r || []
    }catch(e){ console.error('loadRuns', e) }
    return []
  }

  useEffect(()=>{ loadRuns() }, [])
  useEffect(()=> stopRunsPolling, [])

  const handleRunPreview = async (id)=>{
    setRunPreview({ loading: true })
    try{
      const p = await previewPreprocessingRun(id)
      setRunPreview({ loading: false, data: p })
    }catch(e){ setRunPreview({ loading: false, error: e?.response?.data?.detail || e?.message || String(e) }) }
  }

  const handleDownloadRun = async (id) => {
    try{
      const blob = await downloadPreprocessingRun(id)
      const url = window.URL.createObjectURL(new Blob([blob], { type: 'text/csv' }))
      const a = document.createElement('a')
      a.href = url
      a.download = `preprocessed_run_${id}.csv`
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    }catch(e){
      alert('Error descargando el CSV: ' + (e?.response?.data?.detail || e?.message || String(e)))
    }
  }

  const handleViewRunReport = async (id) => {
    try{
      const blob = await downloadPreprocessingRunReport(id)
      const url = window.URL.createObjectURL(new Blob([blob], { type: 'text/markdown' }))
      const tab = window.open('', '_blank', 'noopener,noreferrer')
      if (tab) {
        tab.location.href = url
        tab.focus()
      }
      window.setTimeout(() => window.URL.revokeObjectURL(url), 60000)
    }catch(e){
      alert('Error abriendo el reporte: ' + (e?.response?.data?.detail || e?.message || String(e)))
    }
  }

  const handleDeleteRun = async (id)=>{
    if(!window.confirm('¿Borrar run #' + id + '? Esta acción eliminará los archivos procesados.')) return
    try{
      await deletePreprocessingRun(id)
      await loadRuns()
      setMsg('Run ' + id + ' borrado')
    }catch(e){ setMsg('Error borrando run: ' + (e?.response?.data?.detail || e?.message || String(e))) }
  }

  return (
    <div>
      <div className="header"><h2>Preprocesamiento</h2>
        <div style={{display:'flex',gap:8}}>
          <button className="button" onClick={load}>Refrescar datasets</button>
        </div>
      </div>

      <div className="card">
        <h3>Datasets importados</h3>
        {loadingDatasets && <div>Cargando...</div>}
        {!loadingDatasets && datasets.length===0 && <div>No hay datasets importados.</div>}
        {!loadingDatasets && datasets.length>0 && (
          <div className="table-scroll">
            <table className="table preprocessing-table">
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
                    <td style={{display:'flex',gap:8,whiteSpace:'nowrap'}}>
                      <button className="button" onClick={()=>handlePreview(d.id)}>Previsualizar</button>
                      <button className="button" onClick={()=>handleRunDataset(d.id)}>Procesar</button>
                      <button className="button danger" onClick={()=>handleDelete(d.id)}>Borrar</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

          <div style={{marginTop:20}}>
            <h3>Ejecuciones de preprocesamiento</h3>
            {runs.length===0 && <div>No hay ejecuciones aún.</div>}
            {runs.length>0 && (
              <div className="table-scroll">
                <table className="table preprocessing-table">
                  <thead><tr><th>Id</th><th>Estado</th><th>Total</th><th>Procesados</th><th>Removidos</th><th>Acciones</th></tr></thead>
                  <tbody>
                    {runs.map(rr=> (
                      <tr key={rr.id}>
                        <td>{rr.id}</td>
                        <td>{rr.status}</td>
                        <td>{rr.total_records}</td>
                        <td>{rr.processed_records}</td>
                        <td>{rr.removed_records}</td>
                        <td style={{display:'flex',gap:8,whiteSpace:'nowrap'}}>
                          <button className="button" onClick={()=>handleRunPreview(rr.id)}>Previsualizar run</button>
                          <button className="button" onClick={()=>handleViewRunReport(rr.id)}>Ver reporte</button>
                          <button className="button" onClick={()=>handleDownloadRun(rr.id)}>Descargar CSV</button>
                          <button className="button danger" onClick={()=>handleDeleteRun(rr.id)}>Borrar run</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

        {preview && preview.loading && <div>Obteniendo previsualización...</div>}
        {preview && preview.error && <div>Error: {preview.error}</div>}
        {preview && preview.data && (
          <div style={{marginTop:12}}>
            <h4>Previsualización: {preview.data.dataset_id}</h4>
            <div className="table-scroll">
              <table className="table small preprocessing-table">
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
          </div>
        )}

        {runPreview && runPreview.loading && <div>Obteniendo previsualización del run...</div>}
        {runPreview && runPreview.error && <div>Error: {runPreview.error}</div>}
        {runPreview && runPreview.data && (
          <div style={{marginTop:12}}>
            <h4>Run {runPreview.data.run.id} — Estado: {runPreview.data.run.status}</h4>
            <div style={{display:'flex',gap:20}}>
              <div style={{flex:1, minWidth:0}}>
                <h5>Antes (DB)</h5>
                <div className="table-scroll">
                  <table className="table small preprocessing-table">
                    <thead><tr><th>transaction_id</th><th>amount</th><th>type</th><th>location</th><th>datetime</th><th>is_fraud</th></tr></thead>
                    <tbody>
                      {runPreview.data.before.map((r,idx)=> (
                        <tr key={idx}><td>{r.transaction_id}</td><td>{r.amount}</td><td>{r.transaction_type}</td><td>{r.location}</td><td>{String(r.transaction_datetime)}</td><td>{String(r.is_fraud)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <div style={{flex:1, minWidth:0}}>
                <h5>Después (Procesado)</h5>
                <div className="table-scroll">
                  <table className="table small preprocessing-table">
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
          </div>
        )}

      </div>
      <div style={{marginTop:10}} className="card">{msg || 'Presiona para ejecutar limpieza, normalización y anonimización'}</div>

      {showResultModal && (
        <div className="modal-backdrop" onClick={()=>setShowResultModal(false)}>
          <div className="modal-panel" onClick={e=>e.stopPropagation()}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
              <h3 style={{margin:0}}>Resultado del preprocesamiento</h3>
              <div>
                <button className="button" onClick={()=>{setShowResultModal(false)}}>Cerrar</button>
              </div>
            </div>
            <div style={{marginTop:12}}>
              <div><strong>Estado:</strong> {resultDetails?.status ?? 'Desconocido'}</div>
              <div style={{marginTop:6}}><strong>Resumen:</strong> {resultDetails?.summary ? `before:${resultDetails.summary.before}, after:${resultDetails.summary.after_clean ?? resultDetails.summary.after}` : 'No disponible'}</div>
              <div style={{marginTop:10}}>
                <button className="button" onClick={()=>setShowResultDetailsJson(s=>!s)}>{showResultDetailsJson ? 'Ocultar detalles' : 'Ver detalles'}</button>
              </div>
              {showResultDetailsJson && (
                <pre className="code-panel">{JSON.stringify(resultDetails, null, 2)}</pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
