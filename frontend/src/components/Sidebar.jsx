import React from 'react'
import { NavLink } from 'react-router-dom'

const sections = [
  { title: 'Dashboard', items: [{ to: '/', label: 'Dashboard' }] },
  { title: 'Fase A: Data Pipeline', items: [{ to: '/import', label: 'Importación de Datos' }, { to: '/preprocessing', label: 'Preprocesamiento' }] },
  { title: 'Fase B: Alertas del Sistema', items: [{ to: '/rules-alerts', label: 'Reglas y Alertas' }] },
  { title: 'Fase C: Modelos', items: [{ to: '/models/unsupervised', label: 'No Supervisados' }, { to: '/models/supervised', label: 'Supervisados' }, { to: '/models/evaluation', label: 'Evaluación de Modelos' }] },
  { title: 'Fase D: Monitoreo', items: [{ to: '/monitoring/scoring', label: 'Scoring por Lotes' }, { to: '/monitoring/cases', label: 'Manejo de Casos' }] },
  { title: 'Administración', items: [{ to: '/admin/users', label: 'Usuarios' }] },
]

export default function Sidebar() {
  const logout = () => { localStorage.removeItem('user'); window.location.reload() }
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-mark">FP</div>
        <h2>Sistema de Detección y Monitoreo de Fraude</h2>
      </div>
      <nav>
        {sections.map((section) => (
          <div key={section.title} className="menu-section">
            <div className="menu-title">{section.title}</div>
            {section.items.map((item) => (
              <NavLink key={item.to} to={item.to} className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
                {item.label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
      <div className="sidebar-footer">
        <button className="button button-secondary" onClick={logout}>Cerrar sesión</button>
      </div>
    </aside>
  )
}
