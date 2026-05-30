import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Sidebar from '../../../src/components/Sidebar'
import { MemoryRouter } from 'react-router-dom'

describe('Sidebar component', () => {
  it('renders the reorganized navigation structure', () => {
    render(
      <MemoryRouter>
        <Sidebar />
      </MemoryRouter>
    )

    expect(screen.getByText('Sistema de Detección y Monitoreo de Fraude')).toBeTruthy()
    expect(screen.getByText('Dashboard')).toBeTruthy()
    expect(screen.getByText('Panel')).toBeTruthy()
    expect(screen.getByText('Fase A: Data Pipeline')).toBeTruthy()
    expect(screen.getByText('Importación de Datos')).toBeTruthy()
    expect(screen.getByText('Preprocesamiento')).toBeTruthy()
    expect(screen.getByText('Fase B: Alertas del Sistema')).toBeTruthy()
    expect(screen.getByText('Reglas y Alertas')).toBeTruthy()
    expect(screen.getByText('Fase C: Modelos')).toBeTruthy()
    expect(screen.getByText('Entrenamiento')).toBeTruthy()
    expect(screen.getByText('Evaluación de Modelos')).toBeTruthy()
    expect(screen.getByText('Modelo Activo')).toBeTruthy()
    expect(screen.getByText('Fase D: Monitoreo')).toBeTruthy()
    expect(screen.getByText('Scoring por Lotes')).toBeTruthy()
    expect(screen.getByText('Manejo de Casos')).toBeTruthy()
    expect(screen.getByText('Administración')).toBeTruthy()
    expect(screen.getByText('Usuarios')).toBeTruthy()
    expect(screen.getByText('Configuración')).toBeTruthy()

    expect(screen.queryByText('Alertas (Legacy)')).toBeNull()
    expect(screen.queryByText('Reportes')).toBeNull()
    expect(screen.queryByText('Analítica Avanzada')).toBeNull()
    expect(screen.queryByText('Exportación de Resultados')).toBeNull()
  })
})
