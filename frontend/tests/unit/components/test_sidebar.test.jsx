import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Sidebar from '../../../src/components/Sidebar'
import { MemoryRouter } from 'react-router-dom'

describe('Sidebar component', () => {
  it('renders links', () => {
    render(
      <MemoryRouter>
        <Sidebar />
      </MemoryRouter>
    )
    // Sidebar contains a link to Dashboard by default
    expect(screen.getByText(/Dashboard/i)).toBeTruthy()
  })
})
