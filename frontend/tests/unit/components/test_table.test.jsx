import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Table from '../../../src/components/Table'

const columns = [{ key: 'a', title: 'A' }, { key: 'b', title: 'B' }]
const rows = [{ a: 1, b: 2 }, { a: 3, b: 4 }]

describe('Table component', () => {
  it('renders table headers and cells', () => {
    render(<Table columns={columns} data={rows} />)
    // encabezado de columna debe mostrarse
    expect(screen.getByText('A')).toBeTruthy()
    // celda con valor 1 debe renderizarse
    expect(screen.getByText('1')).toBeTruthy()
  })
})
