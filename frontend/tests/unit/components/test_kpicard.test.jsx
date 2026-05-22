import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import KPICard from '../../../src/components/KPICard'

describe('KPICard component', () => {
  it('renders title and value', () => {
    render(<KPICard title="Test" value="42" />)
    // título debe mostrarse en la tarjeta KPI
    expect(screen.getByText('Test')).toBeTruthy()
    // valor numérico debe mostrarse correctamente
    expect(screen.getByText('42')).toBeTruthy()
  })
})
