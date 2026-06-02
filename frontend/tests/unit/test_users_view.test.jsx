import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('../../src/services/api')
import * as api from '../../src/services/api'

import Users from '../../src/pages/Users'

beforeEach(() => {
  api.getUsers.mockResolvedValue([])
  api.getRoles.mockResolvedValue([])
})

describe('Users page', () => {
  it('renders users header', async () => {
    render(<MemoryRouter><Users /></MemoryRouter>)
    await waitFor(() => {
      const matches = screen.getAllByText(/Usuarios/i)
      expect(matches.length).toBeGreaterThan(0)
    })
  })
})
