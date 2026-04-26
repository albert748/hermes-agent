import { describe, it, expect } from 'vitest'
import { toolTrailLabel, formatToolCall, compactPreview } from '../text'

describe('formatToolCall and toolTrailLabel', () => {
  it('toolTrailLabel extracts correct names', () => {
    expect(toolTrailLabel('web_search')).toBe('Web Search')
    expect(toolTrailLabel('my_custom_tool')).toBe('My Custom Tool')
  })

  it('formatToolCall empty context returns label only', () => {
    const result = formatToolCall('test_tool', '')
    expect(result).toBe('Test Tool')
  })

  it('formatToolCall short context is fully visible', () => {
    const ctx = 'A'.repeat(64)
    const result = formatToolCall('web_search', ctx)
    expect(result).toContain(ctx)
  })

  it('formatToolCall context at 128 chars is fully visible', () => {
    const ctx = 'B'.repeat(128)
    const result = formatToolCall('search', ctx)
    expect(result).toContain(ctx)
    // No truncation marker since compactPreview allows 128 chars
  })

  it('formatToolCall context over 128 chars is truncated', () => {
    const ctx = 'C'.repeat(200)
    const result = formatToolCall('test_tool', ctx)
    expect(result).toContain('…')
    // truncated at max=128: first 127 chars + '…' (before ws collapse)
    expect(result.length).toBeLessThanOrEqual(
      'Test Tool'.length + 3 + 127 + 3  // label + ("…" + 127 chars + ") "
    )
  })

  it('compactPreview truncation at 128', () => {
    const s = 'D'.repeat(200)
    const preview = compactPreview(s, 128)
    expect(preview).toContain('…')
    expect(preview.length).toBeLessThanOrEqual(128)
    // exact: first 127 chars + '…' = 128
    expect(preview.length).toBe(128)
  })
})
