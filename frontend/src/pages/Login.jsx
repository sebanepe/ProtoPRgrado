import React, {useState} from 'react'
import { login as apiLogin, register as apiRegister } from '../services/api'

export default function Login(){
  const [email,setEmail] = useState('')
  const [password,setPassword] = useState('')
  const [showPassword,setShowPassword] = useState(false)
  const [error,setError] = useState(null)
  const [loading,setLoading] = useState(false)
  const [isRegistering, setIsRegistering] = useState(false)
  const [fullName, setFullName] = useState('')
  const [role, setRole] = useState('FRAUD_ANALYST')

  const submit = async (e)=>{
    e.preventDefault()
    setError(null)
    setLoading(true)
    try{
      const res = await apiLogin({ email, password })
      localStorage.setItem('user', JSON.stringify(res))
      // Force navigation + reload so App re-evaluates logged state
      window.location.href = '/dashboard'
    }catch(err){
      setError('Credenciales inválidas o error de conexión')
    }finally{ setLoading(false) }
  }

  const doRegister = async (e)=>{
    e.preventDefault()
    setError(null)
    setLoading(true)
    try{
      const payload = { full_name: fullName, email, password, role }
      const res = await apiRegister(payload)
      // after register, auto-login and redirect
      localStorage.setItem('user', JSON.stringify(res))
      window.location.href = '/dashboard'
    }catch(err){
      setError('Error al crear usuario: ' + (err?.response?.data?.detail || err.message || ''))
    }finally{ setLoading(false) }
  }

  return (
    <div style={{display:'flex',height:'100vh'}}>
      <div style={{flex:1,display:'flex',alignItems:'center',justifyContent:'center',background:'linear-gradient(135deg,#0f1724 0%,#1f2937 100%)',color:'#fff'}}>
        <div style={{maxWidth:420,padding:24}}>
          <h1>Sistema de Detección de Fraude</h1>
          <p style={{opacity:0.9}}>Plataforma de detección y análisis de fraude — accede para revisar alertas y operar modelos.</p>
        </div>
      </div>
      <div style={{width:420,display:'flex',alignItems:'center',justifyContent:'center'}}>
        <div style={{width:360}} className="card">
          <h3>{isRegistering ? 'Crear usuario' : 'Ingresar al Sistema'}</h3>
          <form onSubmit={isRegistering ? doRegister : submit}>
            {isRegistering && <div className="form-row"><input className="input" placeholder="Nombre completo" value={fullName} onChange={e=>setFullName(e.target.value)} required/></div>}
            <div className="form-row"><input className="input" placeholder="Email" type="email" value={email} onChange={e=>setEmail(e.target.value)} required/></div>
            <div className="form-row" style={{display:'flex',gap:8}}>
              <input className="input" placeholder="Contraseña" type={showPassword ? 'text' : 'password'} value={password} onChange={e=>setPassword(e.target.value)} required style={{flex:1}} />
              <button type="button" aria-label={showPassword ? 'Ocultar contraseña' : 'Mostrar contraseña'} onClick={()=>setShowPassword(s=>!s)} style={{padding:'6px 10px',borderRadius:6,border:'1px solid #ddd',background:'#f3f4f6',cursor:'pointer'}}>
                {showPassword ? 'Ocultar' : 'Mostrar'}
              </button>
            </div>
            {isRegistering && (
              <div className="form-row">
                <label style={{display:'block',marginBottom:6}}>Rol</label>
                <select className="input" value={role} onChange={e=>setRole(e.target.value)}>
                  <option value="FRAUD_ANALYST">Analista de Fraude</option>
                  <option value="DATA_SCIENTIST">Científico de Datos</option>
                </select>
              </div>
            )}
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:12}}>
              <a href="#" onClick={(e)=>{e.preventDefault(); alert('Función de recuperación no configurada')}}>{isRegistering ? '¿Ya tienes cuenta? Iniciar sesión' : 'Recuperar contraseña'}</a>
              <div style={{display:'flex',gap:8}}>
                <button type="button" className="button" onClick={()=>{ setIsRegistering(s=>!s); setError(null); }}>{isRegistering ? 'Volver al login' : 'Crear usuario'}</button>
                <button className="button" disabled={loading} type="submit">{loading ? (isRegistering ? 'Creando...' : 'Ingresando...') : (isRegistering ? 'Crear cuenta' : 'Ingresar al Sistema')}</button>
              </div>
            </div>
            {error && <div style={{color:'red',marginBottom:8}}>{error}</div>}
          </form>
          <div style={{fontSize:12,opacity:0.7,marginTop:12}}>© {new Date().getFullYear()} Sistema de Detección — Soporte: soporte@example.com</div>
        </div>
      </div>
    </div>
  )
}
