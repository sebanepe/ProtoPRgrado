import React from 'react'
import { cleanup, render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../services/api')

import Users from '../pages/Users'
import * as api from '../services/api'

const ROLES = [
  { id: 1, code: 'ADMIN', name: 'Administrador', description: null },
  { id: 2, code: 'FRAUD_ANALYST', name: 'Analista de Fraude', description: null },
  { id: 3, code: 'DATA_SCIENTIST', name: 'Científico de Datos', description: null },
]

const USERS = [
  { id: 1, username: 'admin', email: 'admin@sistema.local', full_name: 'Administrador', role: 'ADMIN', role_id: 1, is_active: true, created_at: '2026-01-01T00:00:00Z', updated_at: null },
  { id: 2, username: 'analista1', email: 'analista1@sistema.local', full_name: 'Analista Uno', role: 'FRAUD_ANALYST', role_id: 2, is_active: true, created_at: '2026-02-01T00:00:00Z', updated_at: null },
  { id: 3, username: 'cientifico1', email: 'cient1@sistema.local', full_name: 'Científico Uno', role: 'DATA_SCIENTIST', role_id: 3, is_active: false, created_at: '2026-03-01T00:00:00Z', updated_at: null },
]

function setupMocks(overrides = {}) {
  api.getUsers.mockResolvedValue(overrides.users ?? USERS)
  api.getRoles.mockResolvedValue(overrides.roles ?? ROLES)
  api.createUser.mockResolvedValue({ ...USERS[0], id: 99 })
  api.updateUser.mockResolvedValue(USERS[1])
  api.activateUser.mockResolvedValue({ ...USERS[2], is_active: true })
  api.deactivateUser.mockResolvedValue({ ...USERS[1], is_active: false })
}

function renderUsers() {
  return render(<MemoryRouter><Users /></MemoryRouter>)
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('Pantalla Usuarios', () => {
  it('renderiza pantalla Usuarios con título correcto', async () => {
    setupMocks()
    renderUsers()
    expect(screen.getByText('Usuarios')).toBeTruthy()
    await waitFor(() => expect(api.getUsers).toHaveBeenCalled())
  })

  it('carga usuarios desde la API', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => expect(api.getUsers).toHaveBeenCalledTimes(1))
    // Analista Uno only appears in the user table, not in filters or role panels
    await waitFor(() => screen.getByText('Analista Uno'))
  })

  it('muestra resumen de usuarios con cards correctas', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    expect(screen.getByText('Total usuarios')).toBeTruthy()
    expect(screen.getByText('Activos')).toBeTruthy()
    expect(screen.getByText('Inactivos')).toBeTruthy()
    expect(screen.getByText('Administradores')).toBeTruthy()
    expect(screen.getByText('Analistas de Fraude')).toBeTruthy()
    expect(screen.getByText('Científicos de Datos')).toBeTruthy()
  })

  it('muestra tabla con encabezados correctos', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    // Some header texts also appear in filters; use getAllByText to allow duplicates
    expect(screen.getAllByText('Nombre').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Usuario').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Email').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Rol').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Estado').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Acciones').length).toBeGreaterThanOrEqual(1)
  })

  it('muestra usuarios en la tabla', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    expect(screen.getByText('analista1')).toBeTruthy()
    expect(screen.getByText('analista1@sistema.local')).toBeTruthy()
  })

  it('no muestra password ni password_hash en la tabla', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    const tableContent = document.body.textContent
    expect(tableContent).not.toContain('password_hash')
    expect(tableContent).not.toContain('password_hash')
  })

  it('filtra por búsqueda de username', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    const searchInput = screen.getByTestId('filter-search')
    fireEvent.change(searchInput, { target: { value: 'analista1' } })
    await waitFor(() => {
      // The admin email should not be visible in the filtered results
      expect(screen.queryByText('admin@sistema.local')).toBeNull()
      expect(screen.getByText('Analista Uno')).toBeTruthy()
    })
  })

  it('filtra por rol', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    const roleFilter = screen.getByTestId('filter-role')
    fireEvent.change(roleFilter, { target: { value: 'ADMIN' } })
    await waitFor(() => {
      expect(screen.queryByText('Analista Uno')).toBeNull()
    })
  })

  it('filtra por estado activo/inactivo', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    const statusFilter = screen.getByTestId('filter-status')
    fireEvent.change(statusFilter, { target: { value: 'false' } })
    await waitFor(() => {
      expect(screen.queryByText('Analista Uno')).toBeNull()
      expect(screen.getByText('Científico Uno')).toBeTruthy()
    })
  })

  it('el filtro de roles solo muestra tres opciones válidas', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    const roleFilter = screen.getByTestId('filter-role')
    const options = Array.from(roleFilter.querySelectorAll('option')).map(o => o.value)
    expect(options).toContain('ADMIN')
    expect(options).toContain('FRAUD_ANALYST')
    expect(options).toContain('DATA_SCIENTIST')
    expect(options).not.toContain('CONSULTA')
    expect(options).not.toContain('VIEWER')
  })

  it('no muestra rol Consulta ni Viewer en filtros', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    const filterText = screen.getByTestId('filter-role').textContent
    expect(filterText.toLowerCase()).not.toContain('consulta')
    expect(filterText.toLowerCase()).not.toContain('viewer')
    expect(filterText.toLowerCase()).not.toContain('read only')
  })

  it('abre modal al hacer clic en Nuevo usuario', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    fireEvent.click(screen.getByTestId('btn-nuevo-usuario'))
    await waitFor(() => screen.getByText('Nuevo usuario'))
    expect(screen.getByTestId('input-username')).toBeTruthy()
    expect(screen.getByTestId('input-email')).toBeTruthy()
    expect(screen.getByTestId('input-password')).toBeTruthy()
  })

  it('valida campos requeridos en creación', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    fireEvent.click(screen.getByTestId('btn-nuevo-usuario'))
    await waitFor(() => screen.getByTestId('btn-guardar'))
    fireEvent.click(screen.getByTestId('btn-guardar'))
    await waitFor(() => {
      expect(screen.getByText('El username es requerido.')).toBeTruthy()
    })
  })

  it('llama createUser con payload correcto al guardar', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    fireEvent.click(screen.getByTestId('btn-nuevo-usuario'))
    await waitFor(() => screen.getByTestId('input-username'))

    fireEvent.change(screen.getByTestId('input-username'), { target: { value: 'nuevousuario' } })
    fireEvent.change(screen.getByTestId('input-email'), { target: { value: 'nuevo@test.local' } })
    fireEvent.change(screen.getByTestId('input-full-name'), { target: { value: 'Nuevo Usuario' } })
    fireEvent.change(screen.getByTestId('input-role'), { target: { value: '2' } })
    fireEvent.change(screen.getByTestId('input-password'), { target: { value: 'Temporal123' } })

    fireEvent.click(screen.getByTestId('btn-guardar'))
    await waitFor(() => expect(api.createUser).toHaveBeenCalledTimes(1))
    const callArg = api.createUser.mock.calls[0][0]
    expect(callArg.username).toBe('nuevousuario')
    expect(callArg.email).toBe('nuevo@test.local')
    expect(callArg.password).toBe('Temporal123')
    expect(callArg).not.toHaveProperty('password_hash')
  })

  it('no envía password en edición si está vacío', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    fireEvent.click(screen.getByTestId('btn-edit-2'))
    await waitFor(() => screen.getByTestId('input-password'))

    // leave password empty
    fireEvent.change(screen.getByTestId('input-full-name'), { target: { value: 'Analista Editado' } })
    fireEvent.click(screen.getByTestId('btn-guardar'))

    await waitFor(() => expect(api.updateUser).toHaveBeenCalledTimes(1))
    const callArg = api.updateUser.mock.calls[0][1]
    expect(callArg).not.toHaveProperty('password')
  })

  it('llama updateUser al editar usuario', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    fireEvent.click(screen.getByTestId('btn-edit-2'))
    await waitFor(() => screen.getByTestId('input-full-name'))
    fireEvent.change(screen.getByTestId('input-full-name'), { target: { value: 'Nombre Nuevo' } })
    fireEvent.click(screen.getByTestId('btn-guardar'))
    await waitFor(() => expect(api.updateUser).toHaveBeenCalledWith(2, expect.objectContaining({ full_name: 'Nombre Nuevo' })))
  })

  it('llama deactivateUser al desactivar usuario', async () => {
    setupMocks()
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    fireEvent.click(screen.getByTestId('btn-toggle-2'))
    await waitFor(() => expect(api.deactivateUser).toHaveBeenCalledWith(2))
  })

  it('llama activateUser al activar usuario inactivo', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Científico Uno'))
    fireEvent.click(screen.getByTestId('btn-toggle-3'))
    await waitFor(() => expect(api.activateUser).toHaveBeenCalledWith(3))
  })

  it('muestra panel de roles del sistema con tres cards', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    expect(screen.getByTestId('role-card-ADMIN')).toBeTruthy()
    expect(screen.getByTestId('role-card-FRAUD_ANALYST')).toBeTruthy()
    expect(screen.getByTestId('role-card-DATA_SCIENTIST')).toBeTruthy()
    expect(screen.getByText('Roles del sistema')).toBeTruthy()
  })

  it('el panel de roles no muestra un rol llamado Consulta ni Viewer', async () => {
    setupMocks()
    renderUsers()
    await waitFor(() => screen.getByText('Analista Uno'))
    // The roles panel must not have a card for "Consulta" or "Viewer" as role names
    expect(screen.queryByTestId('role-card-CONSULTA')).toBeNull()
    expect(screen.queryByTestId('role-card-VIEWER')).toBeNull()
    expect(screen.queryByTestId('role-card-READ_ONLY')).toBeNull()
    // No role options in filters
    const filterText = screen.getByTestId('filter-role').textContent
    expect(filterText.toLowerCase()).not.toContain('viewer')
    expect(filterText.toLowerCase()).not.toContain('read only')
  })
})
