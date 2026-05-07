import React from 'react'
import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Dashboard' },
  { to: '/import', label: 'Import Data' },
  { to: '/preprocessing', label: 'Preprocessing' },
  { to: '/models', label: 'Models' },
  { to: '/alerts', label: 'Alerts' },
  { to: '/users', label: 'Users' },
  { to: '/settings', label: 'Settings' }
]

export default function Sidebar() {
  const logout = () => { localStorage.removeItem('user'); window.location.reload(); }
  return (
    <aside className="sidebar">
      <h2>Fraud System</h2>
      <nav>
        {links.map(l => (
          <NavLink key={l.to} to={l.to} className={({isActive})=> isActive? 'nav-link active':'nav-link'}>{l.label}</NavLink>
        ))}
      </nav>
      <div style={{marginTop:20}}>
        <button className="button" onClick={logout}>Logout</button>
      </div>
    </aside>
  )
}
