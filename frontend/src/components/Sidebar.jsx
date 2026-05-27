import React from 'react'
import { NavLink } from 'react-router-dom'

const sections = [
  { title: 'Dashboard', items: [{to:'/', label:'Panel'}] },
  { title: 'Data Pipeline', items: [{to:'/import', label:'Importación de Datos'}, {to:'/preprocessing', label:'Preprocesamiento'}] },
  { title: 'Modelos', items: [{to:'/models/training', label:'Entrenamiento'}, {to:'/models/evaluation', label:'Evaluación de Modelos'}, {to:'/models', label:'Modelo Activo'}] },
  { title: 'Monitoreo', items: [{to:'/monitoring/scoring', label:'Scoring por Lotes'}, {to:'/alerts', label:'Alertas'}, {to:'/monitoring/cases', label:'Manejo de casos'}] },
  { title: 'Reportes', items: [{to:'/reporting', label:'Analítica Avanzada'}, {to:'/reporting/export', label:'Exportación de Resultados'}] },
  { title: 'Administración', items: [{to:'/users', label:'Usuarios'}, {to:'/settings', label:'Configuración'}] }
]

export default function Sidebar() {
  const logout = () => { localStorage.removeItem('user'); window.location.reload(); }
  return (
    <aside className="sidebar">
      <h2>Sistema de Detección de Fraude</h2>
      <nav>
        {sections.map(s => (
          <div key={s.title} className="menu-section">
            <div className="menu-title">{s.title}</div>
            {s.items.map(i => (
              <NavLink key={i.to} to={i.to} className={({isActive})=> isActive? 'nav-link active':'nav-link'}>{i.label}</NavLink>
            ))}
          </div>
        ))}
      </nav>
      <div style={{marginTop:20}}>
        <button className="button" onClick={logout}>Cerrar sesión</button>
      </div>
    </aside>
  )
}
