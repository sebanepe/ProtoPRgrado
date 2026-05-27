import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Users from '../../src/pages/Users'

describe('Users page', ()=>{
  it('renders users header', ()=>{
    render(<Users />)
    const matches = screen.getAllByText(/Usuarios/i)
    expect(matches.length).toBeGreaterThan(0)
  })
})
