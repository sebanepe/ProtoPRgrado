import React from 'react'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import Sidebar from '../../../src/components/Sidebar'

describe('Sidebar component', () => {
  it('renders navigation with Administracion limited to Usuarios', () => {
    render(
      <MemoryRouter>
        <Sidebar />
      </MemoryRouter>
    )

    expect(screen.getAllByText('Dashboard').length).toBeGreaterThan(0)
    expect(screen.getByText('Fase A: Data Pipeline')).toBeTruthy()
    expect(screen.getByText('Importación de Datos')).toBeTruthy()
    expect(screen.getByText('Preprocesamiento')).toBeTruthy()
    expect(screen.getByText('Fase B: Alertas del Sistema')).toBeTruthy()
    expect(screen.getByText('Reglas y Alertas')).toBeTruthy()
    expect(screen.getByText('Fase C: Modelos')).toBeTruthy()
    expect(screen.getByText('No Supervisados')).toBeTruthy()
    expect(screen.getByText('Supervisados')).toBeTruthy()
    expect(screen.getByText('Evaluación de Modelos')).toBeTruthy()
    expect(screen.getByText('Fase D: Monitoreo')).toBeTruthy()
    expect(screen.getByText('Scoring por Lotes')).toBeTruthy()
    expect(screen.getByText('Manejo de Casos')).toBeTruthy()
    expect(screen.getByText('Administración')).toBeTruthy()
    expect(screen.getByText('Usuarios')).toBeTruthy()

    expect(screen.queryByText('Configuracion')).toBeNull()
    expect(screen.queryByText('Configuración')).toBeNull()
    expect(screen.queryByText('ConfiguraciÃ³n')).toBeNull()
    expect(screen.queryByText('Settings')).toBeNull()
  })
})
