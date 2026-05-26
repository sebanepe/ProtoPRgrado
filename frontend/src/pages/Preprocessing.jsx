import React, {useState, useEffect} from 'react'
import { runPreprocessing, listDatasets, previewDataset, deleteDataset, listPreprocessingRuns, previewPreprocessingRun, getPreprocessingRunStages, downloadPreprocessingRun, deletePreprocessingRun } from '../services/api'

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

  const run = async ()=>{
    setMsg('Ejecutando...')
    try{
      const res = await runPreprocessing()
      setMsg('Listo')
      setResultDetails(res)
      setShowResultDetailsJson(false)
      setShowResultModal(true)
      await loadRuns()
    }catch(e){ setMsg('Error: '+ (e?.message||e)) }
  }

  const handleRunDataset = async (datasetId) => {
    setMsg('Ejecutando preprocesamiento para dataset ' + datasetId + '...')
    try{
      // call backend with selected dataset id so processing is scoped
      const res = await runPreprocessing(datasetId)
      setMsg('Listo')
      setResultDetails(res)
      setShowResultDetailsJson(false)
      setShowResultModal(true)
      await loadRuns()
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
    }catch(e){ console.error('loadRuns', e) }
  }

  useEffect(()=>{ loadRuns() }, [])

  const handleRunPreview = async (id)=>{
    setRunPreview({ loading: true })
    try{
      const p = await previewPreprocessingRun(id)
      // fetch inferred stages for UI; if that fails, try to infer from preview 'after' rows
      let stages = null
      try{
        const s = await getPreprocessingRunStages(id)
        stages = s && s.stages ? s.stages : null
      }catch(_){ stages = null }

      if(!stages && p && p.after && Array.isArray(p.after) && p.after.length>0){
        // infer basic stages from the previewed processed rows
        const sample = p.after[0]
        const hasAmountScaled = Object.prototype.hasOwnProperty.call(sample, 'amount_scaled')
        const dummy_like = Object.keys(sample).some(c => c.includes('_') && !['is_fraud','fraud_label_reason'].includes(c))
        stages = {
          limpieza: 'COMPLETED',
          normalizacion: hasAmountScaled ? 'COMPLETED' : 'COMPLETED',
          codificacion: dummy_like ? 'COMPLETED' : 'PENDING',
          smote: 'NOT_APPLIED',
        }
      }

      setRunPreview({ loading: false, data: p, stages })
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
                    <button className="button" onClick={()=>handleRunDataset(d.id)}>Procesar</button>
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
                      <td style={{display:'flex',gap:8}}>
                        <button className="button" onClick={()=>handleRunPreview(rr.id)}>Previsualizar run</button>
                        <button className="button" onClick={()=>handleDownloadRun(rr.id)}>Descargar CSV</button>
                        <button className="button danger" onClick={()=>handleDeleteRun(rr.id)}>Borrar run</button>
                      </td>
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
              <div style={{width:240}}>
                <h5>Etapas</h5>
                {!runPreview.stages && <div style={{fontSize:12,color:'#666'}}>No disponible</div>}
                {runPreview.stages && (
                  <div style={{display:'flex',flexDirection:'column',gap:8}}>
                    {Object.entries(runPreview.stages).map(([k,v])=> (
                      <div key={k} style={{display:'flex',justifyContent:'space-between'}}>
                        <div style={{textTransform:'capitalize'}}>{k.replace('_',' ')}</div>
                        <div style={{fontWeight:600}}>{v}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

      </div>
      <div style={{marginTop:10}} className="card">{msg || 'Presiona para ejecutar SMOTE y transformar datos'}</div>

      {showResultModal && (
        <div style={{position:'fixed',left:0,top:0,right:0,bottom:0,background:'rgba(0,0,0,0.4)',display:'flex',alignItems:'center',justifyContent:'center',zIndex:9999}} onClick={()=>setShowResultModal(false)}>
          <div style={{width:700,maxWidth:'95%',background:'#fff',padding:20,borderRadius:6,boxShadow:'0 8px 30px rgba(0,0,0,0.2)'}} onClick={e=>e.stopPropagation()}>
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
                <pre style={{marginTop:10,maxHeight:360,overflow:'auto',background:'#f7f7f7',padding:10,borderRadius:4}}>{JSON.stringify(resultDetails, null, 2)}</pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
