import React, {useState} from 'react'
import { importDataset } from '../services/api'

export default function ImportData(){
  const [file, setFile] = useState(null)
  const [msg, setMsg] = useState('')
  const [uploading, setUploading] = useState(false)
  const onSubmit = async (e)=>{
    e.preventDefault()
    setMsg('')
    if (!file) return setMsg('Selecciona un archivo')
    setUploading(true)
    try{
      const res = await importDataset(file)
      setMsg('Importado correctamente')
      setFile(null)
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
      </div>
    </div>
  )
}
