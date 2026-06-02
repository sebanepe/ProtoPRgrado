import React from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import Sidebar from '../components/Sidebar'

function renderSidebar(initialPath = '/admin/users') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Sidebar />
    </MemoryRouter>
  )
}

afterEach(() => cleanup())

describe('Sidebar – navegación de Administración', () => {
  it('muestra el ítem Usuarios en la sección Administración', () => {
    renderSidebar()
    expect(screen.getByText('Administración')).toBeTruthy()
    const link = screen.getByRole('link', { name: 'Usuarios' })
    expect(link).toBeTruthy()
    expect(link.getAttribute('href')).toBe('/admin/users')
  })

  it('no muestra Configuración en el sidebar', () => {
    renderSidebar()
    expect(screen.queryByText('Configuración')).toBeNull()
  })

  it('no muestra Settings en el sidebar', () => {
    renderSidebar()
    expect(screen.queryByText('Settings')).toBeNull()
  })

  it('no muestra Parámetros en el sidebar', () => {
    renderSidebar()
    expect(screen.queryByText('Parámetros')).toBeNull()
  })

  it('Administración solo contiene Usuarios', () => {
    renderSidebar()
    // Find the Administration section by its title
    const sections = document.querySelectorAll('.menu-section')
    const adminSection = Array.from(sections).find(s => s.textContent.includes('Administración'))
    expect(adminSection).toBeTruthy()
    const links = adminSection.querySelectorAll('a')
    expect(links.length).toBe(1)
    expect(links[0].textContent.trim()).toBe('Usuarios')
  })

  it('muestra todas las secciones principales', () => {
    renderSidebar()
    // "Dashboard" appears as both section title and nav link; use getAllByText
    expect(screen.getAllByText('Dashboard').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Fase A: Data Pipeline')).toBeTruthy()
    expect(screen.getByText('Fase B: Alertas del Sistema')).toBeTruthy()
    expect(screen.getByText('Fase C: Modelos')).toBeTruthy()
    expect(screen.getByText('Fase D: Monitoreo')).toBeTruthy()
    expect(screen.getByText('Administración')).toBeTruthy()
  })
})
