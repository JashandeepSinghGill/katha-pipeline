'use client'

import { useState } from 'react'
import type { ResolveBody } from '@/lib/api'
import { romanize } from '@/lib/roman'
import { ENTITY_TYPES, type EntityType, type Flag } from '@/lib/types'

export function FlagPanel({
  flag,
  onResolve,
}: {
  flag: Flag
  onResolve: (body: ResolveBody) => Promise<void>
}) {
  const [custom, setCustom] = useState('')
  const [remember, setRemember] = useState(false)
  const [entityType, setEntityType] = useState<EntityType>(
    flag.kind === 'proper_noun' ? 'person' : 'other',
  )
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async (body: ResolveBody) => {
    setBusy(true)
    setError(null)
    try {
      await onResolve(body)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const accept = (text: string) =>
    run({ action: 'accept', text, remember: remember ? { entity_type: entityType } : undefined })

  return (
    <div className="flag">
      <div className="flag-head">
        <span className="pill">{flag.kind}</span>
        <span className="gurmukhi" style={{ fontSize: 16, lineHeight: 1.4 }}>{flag.span_text}</span>
        <span style={{ color: 'var(--muted)' }}>{flag.reason}</span>
        {flag.ledger_id && (
          <span className="pill done" title="The top candidate comes from the correction ledger">
            ledger
          </span>
        )}
      </div>
      <div className="flag-candidates">
        {flag.candidates.map((c) => (
          <button key={c} disabled={busy} onClick={() => accept(c)} className="candidate">
            <span className="gurmukhi" style={{ fontSize: 17, lineHeight: 1.4 }}>{c}</span>
            <span style={{ color: 'var(--muted)', fontSize: 12 }}>{romanize(c)}</span>
          </button>
        ))}
      </div>
      <div className="flag-actions">
        <input
          className="gurmukhi field"
          placeholder="custom Gurmukhi…"
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          style={{ fontSize: 16, flex: 1, minWidth: 180 }}
        />
        <button disabled={busy || !custom.trim()} onClick={() => accept(custom)}>Apply</button>
        <button disabled={busy} onClick={() => run({ action: 'reject' })}>Reject</button>
        <label style={{ fontSize: 13, display: 'flex', gap: 6, alignItems: 'center' }}>
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
          Remember as
          <select
            className="field"
            value={entityType}
            onChange={(e) => setEntityType(e.target.value as EntityType)}
            disabled={!remember}
          >
            {ENTITY_TYPES.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>
      </div>
      {error && <div style={{ color: 'var(--warn)', fontSize: 13, marginTop: 6 }}>{error}</div>}
    </div>
  )
}
