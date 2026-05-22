import React from 'react'
import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Panel' },
  { to: '/import', label: 'Importar datos' },
  { to: '/preprocessing', label: 'Preprocesamiento' },
  { to: '/models', label: 'Modelos' },
  { to: '/alerts', label: 'Alertas' },
  { to: '/users', label: 'Usuarios' },
  { to: '/settings', label: 'Configuración' }
]

export default function Sidebar() {
  const logout = () => { localStorage.removeItem('user'); window.location.reload(); }
  return (
    <aside className="sidebar">
      <h2>Sistema de Detección de Fraude</h2>
      <nav>
        {links.map(l => (
          <NavLink key={l.to} to={l.to} className={({isActive})=> isActive? 'nav-link active':'nav-link'}>{l.label}</NavLink>
        ))}
      </nav>
      <div style={{marginTop:20}}>
        <button className="button" onClick={logout}>Cerrar sesión</button>
      </div>
    </aside>
  )
}
