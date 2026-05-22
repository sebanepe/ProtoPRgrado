import React, {useState} from 'react'
import { importDataset } from '../services/api'

export default function ImportData(){
  const [file, setFile] = useState(null)
  const [msg, setMsg] = useState('')
  const onSubmit = async (e)=>{
    e.preventDefault()
    if (!file) return setMsg('Selecciona un archivo')
    try{
      const res = await importDataset(file)
      setMsg('Importado OK: '+ JSON.stringify(res))
    }catch(err){ setMsg('Error: '+ (err?.message||err)) }
  }
  return (
    <div>
      <div className="header"><h2>Importar datos</h2></div>
      <div className="card">
        <form onSubmit={onSubmit}>
          <div className="form-row">
            <input type="file" onChange={e=>setFile(e.target.files[0])} />
          </div>
          <div style={{textAlign:'right'}}><button className="button">Subir</button></div>
        </form>
        {msg && <div style={{marginTop:12}} className="card">{msg}</div>}
      </div>
    </div>
  )
}
