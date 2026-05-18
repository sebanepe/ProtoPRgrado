import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import KPICard from '../../../src/components/KPICard'

describe('KPICard component', () => {
  it('renders title and value', () => {
    render(<KPICard title="Test" value="42" />)
    expect(screen.getByText('Test')).toBeTruthy()
    expect(screen.getByText('42')).toBeTruthy()
  })
})
