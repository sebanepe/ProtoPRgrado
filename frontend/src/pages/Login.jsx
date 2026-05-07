import React, {useState} from 'react'

export default function Login(){
  const [user,setUser] = useState('')
  const [pass,setPass] = useState('')
  const submit = (e)=>{ e.preventDefault(); localStorage.setItem('user', JSON.stringify({username:user})); window.location.reload() }
  return (
    <div style={{display:'flex',height:'100vh',alignItems:'center',justifyContent:'center'}}>
      <div style={{width:340}} className="card">
        <h3>Login (simulado)</h3>
        <form onSubmit={submit}>
          <div className="form-row"><input className="input" placeholder="Usuario" value={user} onChange={e=>setUser(e.target.value)} /></div>
          <div className="form-row"><input className="input" placeholder="Password" type="password" value={pass} onChange={e=>setPass(e.target.value)} /></div>
          <div style={{textAlign:'right'}}><button className="button">Login</button></div>
        </form>
      </div>
    </div>
  )
}
