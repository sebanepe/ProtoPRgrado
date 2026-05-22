import React, {useState} from 'react'
import { runPreprocessing } from '../services/api'

export default function Preprocessing(){
  const [msg,setMsg] = useState('')
  const run = async ()=>{
    setMsg('Ejecutando...')
    try{
      const res = await runPreprocessing()
      setMsg('Listo: '+ JSON.stringify(res))
    }catch(e){ setMsg('Error: '+ (e?.message||e)) }
  }
  return (
    <div>
      <div className="header"><h2>Preprocesamiento</h2>
        <div><button className="button" onClick={run}>Ejecutar preprocesamiento</button></div>
      </div>
      <div className="card">{msg || 'Presiona para ejecutar SMOTE y transformar datos'}</div>
    </div>
  )
}
