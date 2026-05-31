import React, {useState} from 'react'
import { login as apiLogin, register as apiRegister } from '../services/api'

export default function Login({ checking = false }){
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
      // Try history pushState (works in tests). In real browser reload to let App re-evaluate auth.
      try { window.history.pushState({}, '', '/dashboard') } catch (e) { /* ignore */ }
      const isTest = (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'test') || (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_TEST === 'true')
      if (!isTest) {
        try { window.location.reload() } catch (e) { /* ignore */ }
      }
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
      try { window.history.pushState({}, '', '/dashboard') } catch (e) { /* ignore */ }
      const isTest2 = (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'test') || (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_TEST === 'true')
      if (!isTest2) {
        try { window.location.reload() } catch (e) { /* ignore */ }
      }
    }catch(err){
      setError('Error al crear usuario: ' + (err?.response?.data?.detail || err.message || ''))
    }finally{ setLoading(false) }
  }

  return (
    <div className="login-shell">
      <div className="login-hero">
        <div className="login-hero-panel">
          <h1>Sistema de Detección y monitoreo de Fraude</h1>
          <p>Plataforma de detección y análisis de fraude — accede para revisar alertas y operar modelos.</p>
        </div>
      </div>
      <div className="login-panel">
        <div className="card login-card">
          <h3>{isRegistering ? 'Crear usuario' : 'Ingresar al Sistema'}</h3>
          <form onSubmit={isRegistering ? doRegister : submit}>
            {checking && (
              <div className="login-status">Validando sesión...</div>
            )}
            {isRegistering && <div className="form-row"><input className="input" placeholder="Nombre completo" value={fullName} onChange={e=>setFullName(e.target.value)} required/></div>}
            <div className="form-row"><input className="input" placeholder="Email" type="email" value={email} onChange={e=>setEmail(e.target.value)} required/></div>
            <div className="form-row" style={{display:'flex',gap:8}}>
              <input className="input" placeholder="Contraseña" type={showPassword ? 'text' : 'password'} value={password} onChange={e=>setPassword(e.target.value)} required style={{flex:1}} />
              <button type="button" className="button button-secondary password-toggle" aria-label={showPassword ? 'Ocultar contraseña' : 'Mostrar contraseña'} onClick={()=>setShowPassword(s=>!s)}>
                {showPassword ? 'Ocultar' : 'Mostrar'}
              </button>
            </div>
            {isRegistering && (
              <div className="form-row">
                <label>Rol</label>
                <select className="input" value={role} onChange={e=>setRole(e.target.value)}>
                  <option value="FRAUD_ANALYST">Analista de Fraude</option>
                  <option value="DATA_SCIENTIST">Científico de Datos</option>
                </select>
              </div>
            )}
            <div className="login-links-row">
              <a href="#" onClick={(e)=>{e.preventDefault(); alert('Función de recuperación no configurada')}}>{isRegistering ? '¿Ya tienes cuenta? Iniciar sesión' : 'Recuperar contraseña'}</a>
                <div className="login-switchers">
                <button type="button" className="button" onClick={()=>{ setIsRegistering(s=>!s); setError(null); }} disabled={checking}>{isRegistering ? 'Volver al login' : 'Crear usuario'}</button>
                <button className="button" disabled={loading || checking} type="submit">{checking ? 'Validando...' : (loading ? (isRegistering ? 'Creando...' : 'Ingresando...') : (isRegistering ? 'Crear cuenta' : 'Ingresar al Sistema'))}</button>
              </div>
            </div>
            {error && <div className="error-text">{error}</div>}
          </form>
          <div className="login-footer">© {new Date().getFullYear()} Sistema de Detección — Soporte: soporte@example.com</div>
        </div>
      </div>
    </div>
  )
}
