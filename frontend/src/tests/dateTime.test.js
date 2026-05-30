import { describe, it, expect } from 'vitest'
import { formatDateTime } from '../utils/dateTime'

describe('formatDateTime', () => {
  it('formats ISO timestamps into a readable UTC string', () => {
    expect(formatDateTime('2026-04-12T21:56:11.649000+00:00')).toBe('2026-04-12 21:56:11')
  })

  it('returns the original value for invalid inputs', () => {
    expect(formatDateTime('not-a-date')).toBe('not-a-date')
    expect(formatDateTime(null)).toBeNull()
    expect(formatDateTime(undefined)).toBeUndefined()
  })
})
