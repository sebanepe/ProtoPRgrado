import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { vi, describe, it, beforeEach, expect } from 'vitest'
import RulesAlerts from '../../src/pages/RulesAlerts'

vi.mock('../../src/services/api', () => ({
  getPreprocessedRuns: vi.fn(),
  analyzeRules: vi.fn(),
  getRulesSummary: vi.fn(),
  getSummaryFilterOptions: vi.fn(),
  getRulesReport: vi.fn(),
  getRulesMetrics: vi.fn(),
  getRulesAlerts: vi.fn(),
  getRuleAlertDetail: vi.fn()
}))

import {
  getPreprocessedRuns,
  analyzeRules,
  getRulesSummary,
  getSummaryFilterOptions,
  getRulesReport,
  getRulesMetrics,
  getRulesAlerts,
  getRuleAlertDetail
} from '../../src/services/api'

describe('RulesAlerts page', () => {
  beforeEach(() => {
    // Mock runs list
    getPreprocessedRuns.mockResolvedValue([
      {
        run_id: 'preprocessed_run_26',
        filename: 'preprocessed_run_26.csv',
        size_bytes: 50000000,
        has_alerts: true,
        has_summary: true,
        has_report: true,
        created_at: '2026-05-29T00:00:00Z'
      }
    ])

    // Mock metrics
    getRulesMetrics.mockResolvedValue({
      run_id: 'preprocessed_run_26',
      total_alerts: 73377,
      total_summary_alerts: 16845,
      alerts_by_rule: {
        'RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY': 27554,
        'RULE_DOUBLE_COUNTRY_CARD_ABSENT_CONTEXTUAL': 5701,
        'RULE_CONTEXTUAL_HIGH_RISK_MCC_WITH_SIGNAL': 16590
      },
      alerts_by_risk_level: {
        HIGH: 61234,
        MEDIUM: 12143
      },
      alerts_by_mcc: {
        JEWELRY: 1200,
        GAMBLING: 350
      },
      alerts_by_country: {
        BO: 64000,
        US: 2500
      },
      top_customers: [
        { customer_hash: 'cust_abc123', alert_count: 5 }
      ]
    })

    // Mock summary
    getRulesSummary.mockResolvedValue({
      run_id: 'preprocessed_run_26',
      page: 1,
      page_size: 50,
      total_items: 16845,
      total_pages: 337,
      items: [
        {
          summary_alert_id: 'summary_001',
          customer_hash: 'cust_abc123',
          rule_code: 'RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY',
          rule_name: 'Doble País - Tarjeta Presente',
          risk_level: 'HIGH',
          max_risk_score: 0.85,
          count_transactions: 5,
          countries_detected: 'BO, US',
          status: 'NEW',
          merchant_rubro_proxy: 'JEWELRY',
          window_start: '2026-05-28 10:00:00',
          window_end: '2026-05-29 10:00:00',
          representative_transaction_id: 'tx_001'
        }
      ]
    })
    getSummaryFilterOptions.mockResolvedValue({
      rule_code: ['RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY'],
      risk_level: ['HIGH'],
      status: ['NEW', 'IN_REVIEW'],
      country_code: ['BO'],
      merchant_rubro_proxy: ['JEWELRY'],
      customer_hash: ['cust_abc123']
    })

    // Mock report
    getRulesReport.mockResolvedValue({
      run_id: 'preprocessed_run_26',
      report: '# Reporte de Reglas\n\nTotal alertas: 73377'
    })

    // Mock alerts detail
    getRulesAlerts.mockResolvedValue({
      run_id: 'preprocessed_run_26',
      page: 1,
      page_size: 50,
      total_items: 1,
      total_pages: 1,
      items: [
        {
          alert_id: 'alert_001',
          transaction_id: 'tx_001',
          transaction_datetime: '2026-05-28 10:30:00',
          amount: 1200.50,
          country_code: 'BO',
          pos_entry_mode: '5',
          merchant_rubro_proxy: 'JEWELRY',
          risk_score: 0.85,
          alert_reason: 'Transacción en país diferente'
        }
      ]
    })

    getRuleAlertDetail.mockResolvedValue({
      alert_id: 'alert_001',
      transaction_id: 'tx_001',
      transaction_datetime: '2026-05-28 10:30:00',
      amount: 1200.5,
      country_code: 'BO',
      pos_entry_mode: '5',
      merchant_rubro_proxy: 'JEWELRY',
      risk_score: 0.85,
      alert_reason: 'Transacción en país diferente',
      status: 'NEW'
    })

    // Mock analyze
    analyzeRules.mockResolvedValue({ status: 'COMPLETED' })
  })

  it('renders the RulesAlerts page', async () => {
    render(<RulesAlerts />)
    await waitFor(() => expect(screen.getByText(/Reglas y Alertas/i)).toBeTruthy())
  })

  it('calls getPreprocessedRuns on mount', async () => {
    render(<RulesAlerts />)
    await waitFor(() => expect(getPreprocessedRuns).toHaveBeenCalled())
  })

  it('displays preprocessed runs', async () => {
    render(<RulesAlerts />)
    const runs = screen.queryAllByText(/preprocessed_run_26/i)
    expect(runs.length > 0).toBeTruthy()
  })

  it('shows analyze button', async () => {
    render(<RulesAlerts />)
    const buttons = screen.getAllByText(/Analizar y Generar Alertas/i)
    expect(buttons.length > 0).toBeTruthy()
  })

  it('calls analyzeRules when analyze button is clicked', async () => {
    render(<RulesAlerts />)
    const buttons = screen.getAllByText(/Analizar y Generar Alertas/i)
    const analyzeButton = buttons.find(btn => btn.tagName === 'BUTTON')
    fireEvent.click(analyzeButton)
    await waitFor(() => expect(analyzeRules).toHaveBeenCalled())
  })

  it('loads metrics after analysis', async () => {
    render(<RulesAlerts />)
    const buttons = screen.getAllByText(/Analizar y Generar Alertas/i)
    const analyzeButton = buttons.find(btn => btn.tagName === 'BUTTON')
    fireEvent.click(analyzeButton)
    await waitFor(() => expect(getRulesMetrics).toHaveBeenCalled())
  })

  it('renders metrics cards', async () => {
    render(<RulesAlerts />)
    const buttons = screen.getAllByText(/Analizar y Generar Alertas/i)
    const analyzeButton = buttons.find(btn => btn.tagName === 'BUTTON')
    fireEvent.click(analyzeButton)
    await waitFor(() => expect(getRulesMetrics).toHaveBeenCalled(), { timeout: 3000 })
    // Metrics should be displayed after API call
    const metricsLabel = screen.queryAllByText(/Total Alertas Detalladas/i)
    expect(metricsLabel.length > 0).toBe(true)
  })

  it('renders summary table', async () => {
    render(<RulesAlerts />)
    const buttons = screen.getAllByText(/Analizar y Generar Alertas/i)
    const analyzeButton = buttons.find(btn => btn.tagName === 'BUTTON')
    fireEvent.click(analyzeButton)
    await waitFor(() => expect(getRulesSummary).toHaveBeenCalled())
    await waitFor(() => {
      const ruleElements = screen.queryAllByText(/RULE_DOUBLE_COUNTRY_CARD_PRESENT_SAME_DAY/i)
      return expect(ruleElements.length > 0).toBeTruthy()
    }, { timeout: 3000 })
  })

  it('has pagination controls', async () => {
    render(<RulesAlerts />)
    const buttons = screen.getAllByText(/Analizar y Generar Alertas/i)
    const analyzeButton = buttons.find(btn => btn.tagName === 'BUTTON')
    fireEvent.click(analyzeButton)
    await waitFor(() => expect(screen.getByText(/Anterior/)).toBeTruthy())
    expect(screen.getByText(/Siguiente/)).toBeTruthy()
  })

  it('shows filter section', async () => {
    render(<RulesAlerts />)
    const filterElements = screen.queryAllByText(/Filtros de Resumen/i)
    expect(filterElements.length > 0).toBeTruthy()
  })

  it('allows applying filters', async () => {
    render(<RulesAlerts />)
    const filterButtons = screen.queryAllByText(/Aplicar Filtros/i)
    expect(filterButtons.length > 0).toBeTruthy()
  })

  it('shows warning about non-confirmed fraud', async () => {
    render(<RulesAlerts />)
    const warnings = screen.queryAllByText(/Las alertas generadas no representan fraude confirmado/i)
    expect(warnings.length > 0).toBeTruthy()
  })

  it('does not show is_fraud or confirmed_fraud fields', async () => {
    render(<RulesAlerts />)
    const buttons = screen.getAllByText(/Analizar y Generar Alertas/i)
    const analyzeButton = buttons.find(btn => btn.tagName === 'BUTTON')
    fireEvent.click(analyzeButton)
    await waitFor(() => expect(getRulesSummary).toHaveBeenCalled())
    // Check that these fields are NOT in the summary columns
    expect(screen.queryByRole('cell', { name: /is_fraud/i })).toBeNull()
    expect(screen.queryByRole('cell', { name: /confirmed_fraud/i })).toBeNull()
  })

  it('can view detail when clicking Ver Detalle', async () => {
    render(<RulesAlerts />)
    const buttons = screen.getAllByText(/Analizar y Generar Alertas/i)
    const analyzeButton = buttons.find(btn => btn.tagName === 'BUTTON')
    fireEvent.click(analyzeButton)
    await waitFor(() => expect(screen.queryAllByText(/Ver Detalle/i).length > 0).toBeTruthy())
    const detailButtons = screen.getAllByText(/Ver Detalle/i)
    fireEvent.click(detailButtons[0])
    await waitFor(() => expect(screen.getByText(/Detalle de Alerta Agrupada/i)).toBeTruthy())
  })

  it('shows detail alert information without fraud labels', async () => {
    render(<RulesAlerts />)
    const buttons = screen.getAllByText(/Analizar y Generar Alertas/i)
    const analyzeButton = buttons.find(btn => btn.tagName === 'BUTTON')
    fireEvent.click(analyzeButton)
    await waitFor(() => expect(screen.queryAllByText(/Ver Detalle/i).length > 0).toBeTruthy())
    const detailButtons = screen.getAllByText(/Ver Detalle/i)
    fireEvent.click(detailButtons[0])
    await waitFor(() => {
      const custElements = screen.queryAllByText(/cust_abc123/i)
      return expect(custElements.length > 0).toBeTruthy()
    })
    expect(screen.queryByText(/is_fraud/i)).toBeNull()
  })

  it('shows report modal when viewing report', async () => {
    render(<RulesAlerts />)
    const reportButtons = screen.getAllByText(/Ver Reporte/i)
    fireEvent.click(reportButtons[0])
    await waitFor(() => expect(screen.getByText(/Reporte de Reglas/i)).toBeTruthy())
  })

  it('renders disclaimer in detail modal', async () => {
    render(<RulesAlerts />)
    const buttons = screen.getAllByText(/Analizar y Generar Alertas/i)
    const analyzeButton = buttons.find(btn => btn.tagName === 'BUTTON')
    fireEvent.click(analyzeButton)
    await waitFor(() => expect(screen.queryAllByText(/Ver Detalle/i).length > 0).toBeTruthy())
    const detailButtons = screen.getAllByText(/Ver Detalle/i)
    fireEvent.click(detailButtons[0])
    await waitFor(() => {
      const disclaimers = screen.queryAllByText(/No constituye fraude confirmado/i)
      return expect(disclaimers.length > 0).toBeTruthy()
    })
  })
})
