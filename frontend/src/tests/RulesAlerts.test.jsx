import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
vi.mock('../services/api')
import RulesAlerts from '../pages/RulesAlerts'
import * as api from '../services/api'

const run = {
  run_id: 'preprocessed_run_26',
  filename: 'preprocessed_run_26.csv',
  size_bytes: 1024,
  created_at: '2026-05-29T12:00:00Z',
  has_alerts: true,
  has_summary: true,
  has_report: true
}

const runWithoutSummary = {
  run_id: 'preprocessed_run_1',
  filename: 'preprocessed_run_1.csv',
  size_bytes: 256,
  created_at: '2026-05-01T12:00:00Z',
  has_alerts: false,
  has_summary: false,
  has_report: false
}

const summaryPage1 = {
  run_id: 'preprocessed_run_26',
  page: 1,
  page_size: 20,
  total_items: 25,
  total_pages: 2,
  items: [
    {
      summary_alert_id: '26-S-000001',
      customer_hash: 'cust-1',
      rule_code: 'RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY',
      rule_name: 'Double country card present same day',
      risk_level: 'HIGH',
      max_risk_score: 85,
      count_transactions: 2,
      countries_detected: 'BO|AR',
      representative_transaction_id: 'tx-001',
      status: 'NEW',
      window_start: '2026-04-12T21:56:11.649000+00:00',
      window_end: '2026-04-12T22:16:11.649000+00:00',
      created_at: '2026-04-12T22:20:11.649000+00:00'
    }
  ]
}

const summaryPage2 = {
  run_id: 'preprocessed_run_26',
  page: 2,
  page_size: 20,
  total_items: 25,
  total_pages: 2,
  items: [
    {
      summary_alert_id: '26-S-000021',
      customer_hash: 'cust-2',
      rule_code: 'RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY',
      rule_name: 'Double country card present same day',
      risk_level: 'HIGH',
      max_risk_score: 85,
      count_transactions: 3,
      countries_detected: 'BO|CL',
      status: 'IN_REVIEW'
    }
  ]
}

beforeEach(() => {
  vi.clearAllMocks()
  api.getPreprocessedRuns.mockResolvedValue([runWithoutSummary, run])
  api.analyzeRules.mockResolvedValue({ status: 'COMPLETED' })
  api.getRulesMetrics.mockResolvedValue({
    run_id: 'preprocessed_run_26',
    total_alerts: 0,
    total_summary_alerts: 0,
    alerts_by_rule: {},
    alerts_by_risk_level: {},
    alerts_by_mcc: {},
    alerts_by_country: {},
    top_customers: []
  })
  api.getSummaryFilterOptions.mockResolvedValue({
    rule_code: ['RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY', 'RULE_GAMBLING_MCC'],
    risk_level: ['HIGH'],
    status: ['NEW', 'IN_REVIEW'],
    country_code: ['BO', 'AR'],
    merchant_rubro_proxy: ['5944', '6011'],
    customer_hash: ['cust-1', 'cust-2']
  })
  api.getRulesReport.mockResolvedValue({ report: '# Report' })
  api.getRulesAlerts.mockImplementation((_runId, params = {}) => {
    if (params.transaction_id === 'tx-001') {
      return Promise.resolve({
        run_id: 'preprocessed_run_26',
        page: 1,
        page_size: 50,
        total_items: 1,
        total_pages: 1,
        items: [
          {
            alert_id: '26-000001',
            transaction_id: 'tx-001',
            customer_hash: 'cust-1',
            transaction_datetime: '2026-04-12T21:56:11.649000+00:00',
            amount: 10.0,
            country_code: 'BO',
            pos_entry_mode: '7',
            merchant_rubro_proxy: '5944',
            rule_code: 'RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY',
            rule_name: 'Double country card present same day',
            risk_level: 'HIGH',
            risk_score: 0.95,
            status: 'NEW'
          }
        ]
      })
    }
    return Promise.resolve({ run_id: 'preprocessed_run_26', page: 1, page_size: 50, total_items: 0, total_pages: 0, items: [] })
  })
  api.getRuleSummaryTransactions.mockResolvedValue({
    run_id: 'preprocessed_run_26',
    alert_id: '26-S-000001',
    total_transactions: 3,
    items: [
      {
        transaction_id: 'tx-001',
        transaction_datetime: '2026-04-14T00:07:46.000000+00:00',
        amount: 123.45,
        country_code: 'BO',
        pos_entry_mode: '7',
        merchant_rubro_proxy: '5944',
        merchant_name: 'Mercado Uno',
        has_pinblock: 0,
        risk_score: 85,
        customer_hash: 'cust-1',
        masked_card: '469826******8047'
      },
      {
        transaction_id: 'tx-002',
        transaction_datetime: '2026-04-14T00:15:00.000000+00:00',
        amount: 88,
        country_code: 'AR',
        pos_entry_mode: '7',
        merchant_rubro_proxy: '5944',
        merchant_name: 'Mercado Dos',
        has_pinblock: 1,
        risk_score: 91,
        customer_hash: 'cust-1',
        masked_card: '469826******8047'
      },
      {
        transaction_id: 'tx-003',
        transaction_datetime: '2026-04-14T00:30:00.000000+00:00',
        amount: 59.5,
        country_code: 'BO',
        pos_entry_mode: '5',
        merchant_rubro_proxy: '5944',
        merchant_name: null,
        has_pinblock: 0,
        risk_score: 95,
        customer_hash: 'cust-1',
        masked_card: null
      }
    ]
  })
  api.getRuleAlertDetail.mockResolvedValue({
    alert_id: '26-000001',
    transaction_id: 'tx-1',
    customer_hash: 'cust-1',
    transaction_datetime: '2026-04-12T21:56:11.649000+00:00',
    amount: 10.0,
    country_code: 'BO',
    pos_entry_mode: '7',
    merchant_rubro_proxy: '5944',
    risk_score: 0.95,
    alert_reason: 'Double country card present same day',
    created_at: '2026-04-12T22:20:11.649000+00:00'
  })
  api.getCustomerCardLookup.mockResolvedValue({
    customer_hash: 'cust-1',
    masked_card: null,
    last4: null,
    available: false
  })
  api.getAlertReviewHistory.mockResolvedValue({ history: [] })
  api.getSummaryAlertReviewHistory.mockResolvedValue({ history: [] })
  api.updateAlertReviewStatus.mockResolvedValue({})
  api.updateSummaryAlertReviewStatus.mockResolvedValue({})
})

describe('RulesAlerts page', () => {
  const firstFilter = (label) => screen.getAllByLabelText(label)[0]

  it('selects latest run with summary by default and loads filter options for it', async () => {
    api.getRulesSummary.mockResolvedValue(summaryPage1)

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    await waitFor(() => expect(api.getSummaryFilterOptions).toHaveBeenCalledWith('preprocessed_run_26'))
    expect(api.getSummaryFilterOptions).not.toHaveBeenCalledWith('preprocessed_run_1')
  })

  it('does not load summary filter options when selected run has no summary', async () => {
    api.getPreprocessedRuns.mockResolvedValue([runWithoutSummary])
    api.getRulesSummary.mockResolvedValue(summaryPage1)

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    await waitFor(() => expect(api.getSummaryFilterOptions).not.toHaveBeenCalled())
  })

  it('shows RULE_GAMBLING_MCC in rule filter when provided by summary filter options', async () => {
    api.getRulesSummary.mockResolvedValue(summaryPage1)

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getSummaryFilterOptions).toHaveBeenCalledWith('preprocessed_run_26'))

    const ruleSelect = firstFilter('Regla')
    const values = Array.from(ruleSelect.querySelectorAll('option')).map((option) => option.value)
    expect(values).toContain('RULE_GAMBLING_MCC')
  })

  it('sends select-based summary filters and preserves them when paginating', async () => {
    api.getRulesSummary.mockImplementation((_runId, params = {}) => {
      if (params.page === 2) {
        return Promise.resolve(summaryPage2)
      }
      return Promise.resolve(summaryPage1)
    })

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('button', { name: /Analizar y Generar Alertas/ })[0])

    await waitFor(() => expect(api.getRulesSummary).toHaveBeenCalled())

    fireEvent.change(firstFilter('Regla'), { target: { value: 'RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY' } })
    fireEvent.change(firstFilter('Nivel de Riesgo'), { target: { value: 'HIGH' } })
    fireEvent.change(firstFilter('Estado'), { target: { value: 'IN_REVIEW' } })
    fireEvent.change(firstFilter('País'), { target: { value: 'BO' } })
    fireEvent.change(firstFilter('MCC/Rubro'), { target: { value: '6011' } })
    fireEvent.change(screen.getAllByPlaceholderText('Hash del cliente')[0], { target: { value: 'cust-1' } })
    fireEvent.click(screen.getAllByRole('button', { name: /Aplicar Filtros/ })[0])

    await waitFor(() => {
      const lastCall = api.getRulesSummary.mock.calls.at(-1)
      expect(lastCall[0]).toBe('preprocessed_run_26')
      expect(lastCall[1]).toEqual(expect.objectContaining({
        page: 1,
        rule_code: 'RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY',
        risk_level: 'HIGH',
        status: 'IN_REVIEW',
        country_code: 'BO',
        merchant_rubro_proxy: '6011',
        customer_hash: 'cust-1'
      }))
    })

    fireEvent.click(screen.getByRole('button', { name: /Siguiente →/ }))

    await waitFor(() => {
      const lastCall = api.getRulesSummary.mock.calls.at(-1)
      expect(lastCall[1]).toEqual(expect.objectContaining({ page: 2, status: 'IN_REVIEW', merchant_rubro_proxy: '6011' }))
    })
  })

  it('clears merchant_rubro_proxy when filters are reset', async () => {
    api.getRulesSummary.mockResolvedValue(summaryPage1)

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('button', { name: /Analizar y Generar Alertas/ })[0])

    await waitFor(() => expect(api.getRulesSummary).toHaveBeenCalled())

    fireEvent.change(firstFilter('MCC/Rubro'), { target: { value: '6011' } })
    fireEvent.click(screen.getAllByRole('button', { name: /Aplicar Filtros/ })[0])

    await waitFor(() => {
      const lastCall = api.getRulesSummary.mock.calls.at(-1)
      expect(lastCall[1]).toEqual(expect.objectContaining({ merchant_rubro_proxy: '6011', page: 1 }))
    })

    fireEvent.click(screen.getAllByRole('button', { name: /Limpiar Filtros/ })[0])

    await waitFor(() => {
      const lastCall = api.getRulesSummary.mock.calls.at(-1)
      expect(lastCall[1]).toEqual(expect.not.objectContaining({ merchant_rubro_proxy: '6011' }))
      expect(lastCall[1]).toEqual(expect.objectContaining({ page: 1 }))
    })
  })

  it('applies only selected rule_code without sending empty filters', async () => {
    api.getRulesSummary.mockResolvedValue(summaryPage1)

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('button', { name: /Analizar y Generar Alertas/ })[0])

    await waitFor(() => expect(api.getRulesSummary).toHaveBeenCalled())

    fireEvent.change(firstFilter('Regla'), { target: { value: 'RULE_GAMBLING_MCC' } })
    fireEvent.click(screen.getAllByRole('button', { name: /Aplicar Filtros/ })[0])

    await waitFor(() => {
      const lastCall = api.getRulesSummary.mock.calls.at(-1)
      expect(lastCall[0]).toBe('preprocessed_run_26')
      expect(lastCall[1]).toEqual({
        page: 1,
        page_size: 20,
        rule_code: 'RULE_GAMBLING_MCC'
      })
    })
  })

  it('renders summary rows when API returns filtered results', async () => {
    api.getRulesSummary.mockResolvedValue(summaryPage1)

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('button', { name: /Analizar y Generar Alertas/ })[0])

    await waitFor(() => {
      const matchingCells = screen.getAllByRole('cell', { name: 'RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY' })
      expect(matchingCells.length).toBeGreaterThan(0)
    })
  })

  it('formats dates and loads masked card lookup in the detail modal', async () => {
    api.getRulesSummary.mockResolvedValue(summaryPage1)
    api.getCustomerCardLookup.mockResolvedValue({
      customer_hash: 'cust-1',
      masked_card: '469826******8047',
      last4: '8047',
      available: true
    })

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('button', { name: /Analizar y Generar Alertas/ })[0])

    await waitFor(() => expect(screen.getAllByRole('button', { name: /Ver Detalle/ }).length).toBeGreaterThan(0))
    fireEvent.click(screen.getAllByRole('button', { name: /Ver Detalle/ })[0])

    await waitFor(() => expect(api.getRulesAlerts).toHaveBeenCalledWith('preprocessed_run_26', expect.objectContaining({ transaction_id: 'tx-001' })))
    await waitFor(() => expect(screen.getAllByText('2026-04-12 21:56:11').length).toBeGreaterThan(0))
    expect(screen.getAllByText('2026-04-12 22:16:11').length).toBeGreaterThan(0)
    expect(screen.getAllByText('2026-04-12 22:20:11').length).toBeGreaterThan(0)
    expect(screen.getByText('Ver tarjeta asociada')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /Ver tarjeta asociada/ }))

    await waitFor(() => {
      expect(api.getCustomerCardLookup).toHaveBeenCalledWith('cust-1')
      expect(screen.getByText('469826******8047')).toBeTruthy()
    })

    expect(screen.queryByText('4698261234568047')).toBeNull()
    expect(screen.queryByText('is_fraud')).toBeNull()
    expect(screen.queryByText('confirmed_fraud')).toBeNull()
  })

  it('shows no disponible when customer card mapping is missing', async () => {
    api.getRulesSummary.mockResolvedValue(summaryPage1)
    api.getCustomerCardLookup.mockResolvedValue({
      customer_hash: 'cust-1',
      masked_card: null,
      last4: null,
      available: false
    })

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('button', { name: /Analizar y Generar Alertas/ })[0])

    await waitFor(() => expect(screen.getAllByRole('button', { name: /Ver Detalle/ }).length).toBeGreaterThan(0))
    fireEvent.click(screen.getAllByRole('button', { name: /Ver Detalle/ })[0])

    fireEvent.click(screen.getByRole('button', { name: /Ver tarjeta asociada/ }))

    await waitFor(() => {
      expect(api.getCustomerCardLookup).toHaveBeenCalledWith('cust-1')
      expect(screen.getByText('No disponible')).toBeTruthy()
    })
  })

  it('loads, renders, and hides grouped transactions from the detail modal', async () => {
    api.getRulesSummary.mockResolvedValue(summaryPage1)

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('button', { name: /Analizar y Generar Alertas/ })[0])

    await waitFor(() => expect(screen.getAllByRole('button', { name: /Ver Detalle/ }).length).toBeGreaterThan(0))
    fireEvent.click(screen.getAllByRole('button', { name: /Ver Detalle/ })[0])

    expect(screen.getByRole('button', { name: /Ver transacciones detectadas/ })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /Ver transacciones detectadas/ }))

    await waitFor(() => {
      expect(api.getRuleSummaryTransactions).toHaveBeenCalledWith('preprocessed_run_26', '26-S-000001')
      expect(screen.getByText(/Estas son las transacciones que componen la alerta agrupada/)).toBeTruthy()
      expect(screen.getByText('2026-04-14 00:07:46', { selector: 'td' })).toBeTruthy()
      expect(screen.getAllByText('469826******8047').length).toBeGreaterThanOrEqual(2)
      expect(screen.queryByText('4698261234568047')).toBeNull()
    })

    fireEvent.click(screen.getByRole('button', { name: /Ocultar transacciones/ }))
    await waitFor(() => {
      expect(screen.queryByText(/Estas son las transacciones que componen la alerta agrupada/)).toBeNull()
    })
  })

  it('shows loading while fetching grouped transactions', async () => {
    api.getRulesSummary.mockResolvedValue(summaryPage1)

    let resolveTransactions
    const pendingTransactions = new Promise((resolve) => {
      resolveTransactions = resolve
    })
    api.getRuleSummaryTransactions.mockReturnValue(pendingTransactions)

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('button', { name: /Analizar y Generar Alertas/ })[0])
    await waitFor(() => expect(screen.getAllByRole('button', { name: /Ver Detalle/ }).length).toBeGreaterThan(0))
    fireEvent.click(screen.getAllByRole('button', { name: /Ver Detalle/ })[0])
    fireEvent.click(screen.getByRole('button', { name: /Ver transacciones detectadas/ }))

    await waitFor(() => {
      expect(screen.getByText(/Cargando transacciones detectadas/)).toBeTruthy()
    })

    resolveTransactions({
      run_id: 'preprocessed_run_26',
      alert_id: '26-S-000001',
      total_transactions: 1,
      items: [
        {
          transaction_id: 'tx-001',
          transaction_datetime: '2026-04-14T00:07:46.000000+00:00',
          amount: 123.45,
          country_code: 'BO',
          pos_entry_mode: '7',
          merchant_rubro_proxy: '5944',
          risk_score: 85,
          customer_hash: 'cust-1',
          masked_card: '469826******8047'
        }
      ]
    })

    await waitFor(() => expect(screen.getByText('469826******8047')).toBeTruthy())
  })

  it('shows an empty message when grouped transactions are missing', async () => {
    api.getRulesSummary.mockResolvedValue(summaryPage1)
    api.getRuleSummaryTransactions.mockResolvedValue({
      run_id: 'preprocessed_run_26',
      alert_id: '26-S-000001',
      total_transactions: 0,
      items: []
    })

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('button', { name: /Analizar y Generar Alertas/ })[0])
    await waitFor(() => expect(screen.getAllByRole('button', { name: /Ver Detalle/ }).length).toBeGreaterThan(0))
    fireEvent.click(screen.getAllByRole('button', { name: /Ver Detalle/ })[0])
    fireEvent.click(screen.getByRole('button', { name: /Ver transacciones detectadas/ }))

    await waitFor(() => {
      expect(screen.getByText('No hay transacciones detectadas para esta alerta agrupada.')).toBeTruthy()
    })
  })

  it('shows a controlled message when grouped transactions are unauthorized', async () => {
    api.getRulesSummary.mockResolvedValue(summaryPage1)
    api.getRuleSummaryTransactions.mockRejectedValue({ response: { status: 403, data: { detail: 'Forbidden' } } })

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('button', { name: /Analizar y Generar Alertas/ })[0])
    await waitFor(() => expect(screen.getAllByRole('button', { name: /Ver Detalle/ }).length).toBeGreaterThan(0))
    fireEvent.click(screen.getAllByRole('button', { name: /Ver Detalle/ })[0])
    fireEvent.click(screen.getByRole('button', { name: /Ver transacciones detectadas/ }))

    await waitFor(() => {
      expect(screen.getByText('No autorizado para ver las transacciones detectadas.')).toBeTruthy()
    })
  })

  it('shows a no-results message for filtered empty summaries', async () => {
    api.getRulesSummary.mockResolvedValue({
      run_id: 'preprocessed_run_26',
      page: 1,
      page_size: 20,
      total_items: 25,
      total_pages: 2,
      items: summaryPage1.items
    })

    render(<RulesAlerts />)

    await waitFor(() => expect(api.getPreprocessedRuns).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('button', { name: /Analizar y Generar Alertas/ })[0])

    await waitFor(() => expect(api.getRulesSummary).toHaveBeenCalled())

    fireEvent.change(firstFilter('Estado'), { target: { value: 'IN_REVIEW' } })
    api.getRulesSummary.mockResolvedValueOnce({
      run_id: 'preprocessed_run_26',
      page: 1,
      page_size: 20,
      total_items: 0,
      total_pages: 0,
      items: []
    })
    fireEvent.click(screen.getAllByRole('button', { name: /Aplicar Filtros/ })[0])

    await waitFor(() => {
      expect(screen.getByText('No hay coincidencias para los filtros seleccionados.')).toBeTruthy()
    })
  })
})
