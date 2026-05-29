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
      status: 'NEW'
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
  api.getRulesAlerts.mockResolvedValue({ run_id: 'preprocessed_run_26', page: 1, page_size: 50, total_items: 0, total_pages: 0, items: [] })
  api.getRuleAlertDetail.mockResolvedValue({ alert_id: '26-000001' })
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
