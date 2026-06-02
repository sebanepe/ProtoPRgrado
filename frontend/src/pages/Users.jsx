import React, { useEffect, useState, useMemo } from 'react'
import { getUsers, createUser, updateUser, activateUser, deactivateUser, getRoles } from '../services/api'

const ROLE_LABELS = {
  ADMIN: 'Administrador',
  FRAUD_ANALYST: 'Analista de Fraude',
  DATA_SCIENTIST: 'Científico de Datos',
}

const ROLE_DESCRIPTIONS = {
  ADMIN: ['Control completo del sistema', 'Gestión de usuarios', 'Acceso a todas las fases', 'Administración del sistema'],
  FRAUD_ANALYST: ['Consulta de dashboard', 'Revisión de alertas', 'Registro de revisión humana', 'Consulta de scoring', 'Gestión de casos'],
  DATA_SCIENTIST: ['Importación de datasets', 'Ejecución de preprocesamiento', 'Configuración de reglas', 'Entrenamiento de modelos', 'Evaluación de modelos', 'Análisis comparativos'],
}

const ALLOWED_ROLES = ['ADMIN', 'FRAUD_ANALYST', 'DATA_SCIENTIST']

function formatDate(iso) {
  if (!iso) return '—'
  try { return new Date(iso).toLocaleDateString('es-BO') } catch { return iso }
}

const EMPTY_FORM = { username: '', email: '', full_name: '', role_id: '', password: '', is_active: true }

export default function Users() {
  const [users, setUsers] = useState([])
  const [roles, setRoles] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [search, setSearch] = useState('')
  const [filterRole, setFilterRole] = useState('')
  const [filterStatus, setFilterStatus] = useState('')
  const [modal, setModal] = useState(null) // null | 'create' | 'edit'
  const [editingUser, setEditingUser] = useState(null)
  const [form, setForm] = useState(EMPTY_FORM)
  const [formErrors, setFormErrors] = useState({})
  const [saving, setSaving] = useState(false)
  const [actionMsg, setActionMsg] = useState(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [usersData, rolesData] = await Promise.all([getUsers(), getRoles()])
      setUsers(Array.isArray(usersData) ? usersData : [])
      setRoles(Array.isArray(rolesData) ? rolesData.filter(r => ALLOWED_ROLES.includes(r.code)) : [])
    } catch (e) {
      setError('No se pudo cargar la lista de usuarios.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const filtered = useMemo(() => {
    let list = users
    if (search) {
      const q = search.toLowerCase()
      list = list.filter(u =>
        (u.username || '').toLowerCase().includes(q) ||
        (u.email || '').toLowerCase().includes(q) ||
        (u.full_name || '').toLowerCase().includes(q)
      )
    }
    if (filterRole) {
      list = list.filter(u => u.role === filterRole)
    }
    if (filterStatus !== '') {
      const active = filterStatus === 'true'
      list = list.filter(u => u.is_active === active)
    }
    return list
  }, [users, search, filterRole, filterStatus])

  const summary = useMemo(() => ({
    total: users.length,
    active: users.filter(u => u.is_active).length,
    inactive: users.filter(u => !u.is_active).length,
    admin: users.filter(u => u.role === 'ADMIN').length,
    analyst: users.filter(u => u.role === 'FRAUD_ANALYST').length,
    scientist: users.filter(u => u.role === 'DATA_SCIENTIST').length,
  }), [users])

  const openCreate = () => {
    setForm(EMPTY_FORM)
    setFormErrors({})
    setEditingUser(null)
    setModal('create')
  }

  const openEdit = (user) => {
    setForm({
      username: user.username || '',
      email: user.email || '',
      full_name: user.full_name || '',
      role_id: user.role_id ? String(user.role_id) : '',
      password: '',
      is_active: user.is_active,
    })
    setFormErrors({})
    setEditingUser(user)
    setModal('edit')
  }

  const closeModal = () => { setModal(null); setEditingUser(null) }

  const validateForm = () => {
    const errs = {}
    if (!form.username.trim()) errs.username = 'El username es requerido.'
    if (!form.email.trim()) errs.email = 'El email es requerido.'
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) errs.email = 'Formato de email inválido.'
    if (!form.full_name.trim()) errs.full_name = 'El nombre completo es requerido.'
    if (!form.role_id) errs.role_id = 'El rol es requerido.'
    if (modal === 'create') {
      if (!form.password) errs.password = 'La contraseña es requerida.'
      else if (form.password.length < 8) errs.password = 'Mínimo 8 caracteres.'
    } else if (modal === 'edit' && form.password && form.password.length < 8) {
      errs.password = 'Mínimo 8 caracteres.'
    }
    setFormErrors(errs)
    return Object.keys(errs).length === 0
  }

  const handleSave = async () => {
    if (!validateForm()) return
    setSaving(true)
    try {
      const payload = {
        username: form.username.trim(),
        email: form.email.trim(),
        full_name: form.full_name.trim(),
        role_id: Number(form.role_id),
        is_active: form.is_active,
      }
      if (modal === 'create' || (modal === 'edit' && form.password.trim())) {
        payload.password = form.password
      }
      if (modal === 'create') {
        await createUser(payload)
        showMsg('Usuario creado correctamente.')
      } else {
        await updateUser(editingUser.id, payload)
        showMsg('Usuario actualizado correctamente.')
      }
      closeModal()
      await load()
    } catch (e) {
      const detail = e?.response?.data?.detail || 'Error al guardar el usuario.'
      setFormErrors({ _global: detail })
    } finally {
      setSaving(false)
    }
  }

  const handleToggleStatus = async (user) => {
    if (!user.is_active) {
      await doActivate(user)
    } else {
      if (!window.confirm(`¿Desactivar a ${user.full_name}?`)) return
      await doDeactivate(user)
    }
  }

  const doActivate = async (user) => {
    try {
      await activateUser(user.id)
      showMsg(`Usuario ${user.full_name} activado.`)
      await load()
    } catch (e) {
      showMsg(e?.response?.data?.detail || 'Error al activar usuario.', true)
    }
  }

  const doDeactivate = async (user) => {
    try {
      await deactivateUser(user.id)
      showMsg(`Usuario ${user.full_name} desactivado.`)
      await load()
    } catch (e) {
      showMsg(e?.response?.data?.detail || 'Error al desactivar usuario.', true)
    }
  }

  const showMsg = (msg, isError = false) => {
    setActionMsg({ text: msg, error: isError })
    setTimeout(() => setActionMsg(null), 4000)
  }

  const roleName = (code) => ROLE_LABELS[code] || code

  const roleIdByCode = (code) => {
    const r = roles.find(r => r.code === code)
    return r ? String(r.id) : ''
  }

  return (
    <div>
      <div className="header">
        <div>
          <h2>Usuarios</h2>
          <p style={{ color: '#6b7280', marginTop: 4, marginBottom: 0 }}>
            Administración de cuentas, roles y permisos de acceso al sistema.
          </p>
          <p style={{ color: '#9ca3af', fontSize: 13, marginTop: 4 }}>
            Los roles controlan qué módulos puede utilizar cada usuario dentro del prototipo.
          </p>
        </div>
        <button className="button button-primary" onClick={openCreate} data-testid="btn-nuevo-usuario">
          + Nuevo usuario
        </button>
      </div>

      {actionMsg && (
        <div className="card" style={{ background: actionMsg.error ? '#fef2f2' : '#f0fdf4', border: `1px solid ${actionMsg.error ? '#fca5a5' : '#86efac'}`, color: actionMsg.error ? '#dc2626' : '#15803d', marginBottom: 12 }}>
          {actionMsg.text}
        </div>
      )}

      {/* Resumen */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 12, marginBottom: 20 }}>
        {[
          { label: 'Total usuarios', value: summary.total },
          { label: 'Activos', value: summary.active },
          { label: 'Inactivos', value: summary.inactive },
          { label: 'Administradores', value: summary.admin },
          { label: 'Analistas de Fraude', value: summary.analyst },
          { label: 'Científicos de Datos', value: summary.scientist },
        ].map(({ label, value }) => (
          <div key={label} className="card" style={{ padding: '14px 16px', textAlign: 'center' }}>
            <div style={{ fontSize: 26, fontWeight: 700 }}>{value}</div>
            <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Filtros */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div style={{ flex: '1 1 200px' }}>
            <label style={{ fontSize: 12, color: '#6b7280', display: 'block', marginBottom: 4 }}>Buscar</label>
            <input
              className="form-input"
              placeholder="Username, email o nombre..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              data-testid="filter-search"
            />
          </div>
          <div style={{ flex: '0 0 200px' }}>
            <label style={{ fontSize: 12, color: '#6b7280', display: 'block', marginBottom: 4 }}>Rol</label>
            <select className="form-input" value={filterRole} onChange={e => setFilterRole(e.target.value)} data-testid="filter-role">
              <option value="">Todos los roles</option>
              <option value="ADMIN">Administrador</option>
              <option value="FRAUD_ANALYST">Analista de Fraude</option>
              <option value="DATA_SCIENTIST">Científico de Datos</option>
            </select>
          </div>
          <div style={{ flex: '0 0 160px' }}>
            <label style={{ fontSize: 12, color: '#6b7280', display: 'block', marginBottom: 4 }}>Estado</label>
            <select className="form-input" value={filterStatus} onChange={e => setFilterStatus(e.target.value)} data-testid="filter-status">
              <option value="">Todos</option>
              <option value="true">Activo</option>
              <option value="false">Inactivo</option>
            </select>
          </div>
          {(search || filterRole || filterStatus) && (
            <button className="button button-secondary" onClick={() => { setSearch(''); setFilterRole(''); setFilterStatus('') }}>
              Limpiar
            </button>
          )}
        </div>
      </div>

      {/* Tabla */}
      <div className="card" style={{ padding: 0, overflow: 'hidden', marginBottom: 20 }}>
        {loading ? (
          <div style={{ padding: 32, textAlign: 'center', color: '#9ca3af' }}>Cargando usuarios...</div>
        ) : error ? (
          <div style={{ padding: 32, textAlign: 'center', color: '#dc2626' }}>{error}</div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                {['Nombre', 'Usuario', 'Email', 'Rol', 'Estado', 'Fecha creación', 'Acciones'].map(h => (
                  <th key={h} style={{ padding: '10px 14px', textAlign: 'left', fontSize: 12, fontWeight: 600, color: '#374151', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr><td colSpan={7} style={{ padding: 32, textAlign: 'center', color: '#9ca3af' }}>No se encontraron usuarios.</td></tr>
              ) : filtered.map(u => (
                <tr key={u.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: '10px 14px', fontWeight: 500 }}>{u.full_name}</td>
                  <td style={{ padding: '10px 14px', color: '#6b7280', fontFamily: 'monospace', fontSize: 13 }}>{u.username || '—'}</td>
                  <td style={{ padding: '10px 14px', fontSize: 13 }}>{u.email}</td>
                  <td style={{ padding: '10px 14px' }}>
                    <RoleBadge code={u.role} />
                  </td>
                  <td style={{ padding: '10px 14px' }}>
                    <StatusBadge active={u.is_active} />
                  </td>
                  <td style={{ padding: '10px 14px', fontSize: 12, color: '#9ca3af' }}>{formatDate(u.created_at)}</td>
                  <td style={{ padding: '10px 14px', whiteSpace: 'nowrap' }}>
                    <button
                      className="button button-secondary"
                      style={{ fontSize: 12, padding: '4px 10px', marginRight: 6 }}
                      onClick={() => openEdit(u)}
                      data-testid={`btn-edit-${u.id}`}
                    >
                      Editar
                    </button>
                    <button
                      className="button button-secondary"
                      style={{ fontSize: 12, padding: '4px 10px', background: u.is_active ? '#fef2f2' : '#f0fdf4', color: u.is_active ? '#dc2626' : '#15803d', borderColor: u.is_active ? '#fca5a5' : '#86efac' }}
                      onClick={() => handleToggleStatus(u)}
                      data-testid={`btn-toggle-${u.id}`}
                    >
                      {u.is_active ? 'Desactivar' : 'Activar'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Panel de roles */}
      <div>
        <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Roles del sistema</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 16 }}>
          {ALLOWED_ROLES.map(code => (
            <div key={code} className="card" data-testid={`role-card-${code}`}>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>
                <RoleBadge code={code} />
              </div>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#374151', lineHeight: 1.7 }}>
                {ROLE_DESCRIPTIONS[code].map(d => <li key={d}>{d}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Modal crear / editar */}
      {modal && (
        <div
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onClick={e => { if (e.target === e.currentTarget) closeModal() }}
        >
          <div className="card" style={{ width: '100%', maxWidth: 480, margin: 16, maxHeight: '90vh', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
              <h3 style={{ margin: 0 }}>{modal === 'create' ? 'Nuevo usuario' : 'Editar usuario'}</h3>
              <button onClick={closeModal} style={{ background: 'none', border: 'none', fontSize: 20, cursor: 'pointer', color: '#6b7280' }}>×</button>
            </div>

            {formErrors._global && (
              <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', color: '#dc2626', padding: '8px 12px', borderRadius: 6, marginBottom: 12, fontSize: 13 }}>
                {formErrors._global}
              </div>
            )}

            <FormField label="Username *" error={formErrors.username}>
              <input
                className="form-input"
                value={form.username}
                onChange={e => setForm(f => ({ ...f, username: e.target.value }))}
                data-testid="input-username"
              />
            </FormField>

            <FormField label="Email *" error={formErrors.email}>
              <input
                className="form-input"
                type="email"
                value={form.email}
                onChange={e => setForm(f => ({ ...f, email: e.target.value }))}
                data-testid="input-email"
              />
            </FormField>

            <FormField label="Nombre completo *" error={formErrors.full_name}>
              <input
                className="form-input"
                value={form.full_name}
                onChange={e => setForm(f => ({ ...f, full_name: e.target.value }))}
                data-testid="input-full-name"
              />
            </FormField>

            <FormField label="Rol *" error={formErrors.role_id}>
              <select
                className="form-input"
                value={form.role_id}
                onChange={e => setForm(f => ({ ...f, role_id: e.target.value }))}
                data-testid="input-role"
              >
                <option value="">Seleccionar rol...</option>
                {roles.map(r => (
                  <option key={r.id} value={String(r.id)}>{ROLE_LABELS[r.code] || r.name}</option>
                ))}
              </select>
            </FormField>

            <FormField label={modal === 'create' ? 'Contraseña *' : 'Nueva contraseña (dejar vacío para no cambiar)'} error={formErrors.password}>
              <input
                className="form-input"
                type="password"
                value={form.password}
                onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                data-testid="input-password"
                autoComplete="new-password"
              />
            </FormField>

            <FormField label="Estado" error={null}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none' }}>
                <input
                  type="checkbox"
                  checked={form.is_active}
                  onChange={e => setForm(f => ({ ...f, is_active: e.target.checked }))}
                  data-testid="input-is-active"
                />
                <span style={{ fontSize: 13 }}>Cuenta activa</span>
              </label>
            </FormField>

            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', marginTop: 8 }}>
              <button className="button button-secondary" onClick={closeModal} disabled={saving}>Cancelar</button>
              <button className="button button-primary" onClick={handleSave} disabled={saving} data-testid="btn-guardar">
                {saving ? 'Guardando...' : 'Guardar'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function FormField({ label, error, children }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ fontSize: 13, fontWeight: 500, display: 'block', marginBottom: 4 }}>{label}</label>
      {children}
      {error && <div style={{ color: '#dc2626', fontSize: 12, marginTop: 3 }}>{error}</div>}
    </div>
  )
}

function RoleBadge({ code }) {
  const colors = {
    ADMIN: { bg: '#eff6ff', color: '#1d4ed8', border: '#bfdbfe' },
    FRAUD_ANALYST: { bg: '#fef3c7', color: '#92400e', border: '#fde68a' },
    DATA_SCIENTIST: { bg: '#f0fdf4', color: '#15803d', border: '#bbf7d0' },
  }
  const s = colors[code] || { bg: '#f3f4f6', color: '#374151', border: '#d1d5db' }
  return (
    <span style={{ background: s.bg, color: s.color, border: `1px solid ${s.border}`, borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 500 }}>
      {ROLE_LABELS[code] || code}
    </span>
  )
}

function StatusBadge({ active }) {
  return (
    <span style={{ background: active ? '#f0fdf4' : '#fef2f2', color: active ? '#15803d' : '#dc2626', border: `1px solid ${active ? '#86efac' : '#fca5a5'}`, borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 500 }}>
      {active ? 'Activo' : 'Inactivo'}
    </span>
  )
}
