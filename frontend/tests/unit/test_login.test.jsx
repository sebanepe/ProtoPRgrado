import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import Login from '../../src/pages/Login'

vi.mock('../../src/services/api', () => ({
  login: vi.fn(async (creds) => ({ id: 1, email: creds.email, token: 't' }))
}))

describe('Login page', () => {
  it('renders form and submits', async () => {
    render(<Login />)
    const email = screen.getByPlaceholderText(/Email/i)
    const pass = screen.getByPlaceholderText(/Contraseña/i)
    const btn = screen.getByRole('button', { name: /Ingresar al Sistema/i })

    fireEvent.change(email, { target: { value: 'test@example.com' } })
    fireEvent.change(pass, { target: { value: 'secret' } })
    fireEvent.click(btn)

    await waitFor(() => expect(window.location.href).toContain('/dashboard'))
  })
})
