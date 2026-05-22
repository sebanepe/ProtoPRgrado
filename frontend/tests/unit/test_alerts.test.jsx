import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { vi, describe, it, beforeEach, expect } from 'vitest'
import Alerts from '../../src/pages/Alerts'

vi.mock('../../src/services/api', () => ({ listAlerts: vi.fn() }))
import { listAlerts } from '../../src/services/api'

describe('Alerts page', () => {
  beforeEach(() => {
    listAlerts.mockResolvedValue([{ alert_id:1, risk_level:'HIGH', risk_score:0.9 }])
  })

  it('renders alert rows fetched from API', async () => {
    render(<Alerts />)
    await waitFor(() => expect(listAlerts).toHaveBeenCalled())
    // debe mostrar la fila con nivel HIGH devuelta por la API
    expect(screen.getByText(/HIGH/i)).toBeTruthy()
  })
})
