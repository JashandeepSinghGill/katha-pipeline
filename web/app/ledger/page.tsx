'use client'

import { useCallback, useEffect, useState } from 'react'
import { createLedger, fetchLedger, patchLedger } from '@/lib/api'
import { romanize } from '@/lib/roman'
import { ENTITY_TYPES, type EntityType, type LedgerEntry } from '@/lib/types'

const splitForms = (s: string) => s.split(',').map((f) => f.trim()).filter(Boolean)

export default function Ledger() {
  const [q, setQ] = useState('')
  const [includeRetired, setIncludeRetired] = useState(false)
  const [rows, setRows] = useState<LedgerEntry[]>([])
  const [error, setError] = useState<string | null>(null)
  const [draft, setDraft] = useState({ canonical: '', forms: '', type: 'person' as EntityType, gloss: '' })
  const [editing, setEditing] = useState<LedgerEntry | null>(null)
  const [editingForms, setEditingForms] = useState('')

  const load = useCallback(
    () => fetchLedger(q, includeRetired).then(setRows).catch((e) => setError(String(e))),
    [q, includeRetired],
  )
  useEffect(() => { load() }, [load])

  const create = () =>
    createLedger({
      canonical_gurmukhi: draft.canonical,
      surface_forms: splitForms(draft.forms),
      entity_type: draft.type,
      english_gloss: draft.gloss || undefined,
    })
      .then(() => { setDraft({ canonical: '', forms: '', type: 'person', gloss: '' }); return load() })
      .catch((e) => setError(String(e)))

  const startEdit = (r: LedgerEntry) => {
    setEditing(r)
    setEditingForms(r.surface_forms.join(', '))
  }

  const save = () => {
    if (!editing) return
    patchLedger(editing.id, {
      canonical_gurmukhi: editing.canonical_gurmukhi,
      surface_forms: splitForms(editingForms),
      entity_type: editing.entity_type,
      english_gloss: editing.english_gloss,
      confidence_threshold: editing.confidence_threshold,
    })
      .then(() => { setEditing(null); return load() })
      .catch((e) => setError(String(e)))
  }

  const toggleRetired = (r: LedgerEntry) =>
    patchLedger(r.id, { retired: !r.retired }).then(load).catch((e) => setError(String(e)))

  return (
    <>
      <h2>Correction ledger</h2>
      <p style={{ color: 'var(--muted)', fontSize: 13 }}>
        Every entry is a canonical Gurmukhi spelling plus the misrecognitions the ASR has produced for it.
        A flagged span that fuzzy-matches a misrecognition is suggested; if the surrounding words also
        resemble the context the entry was learned in, it is applied automatically (and can be reverted).
      </p>
      {error && <p className="pill failed">{error}</p>}

      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 12 }}>
        <input
          className="field"
          placeholder="search Gurmukhi, romanized, or a misrecognition"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          style={{ flex: 1 }}
        />
        <label style={{ fontSize: 13 }}>
          <input type="checkbox" checked={includeRetired} onChange={(e) => setIncludeRetired(e.target.checked)} />
          {' '}show retired
        </label>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        <input
          className="gurmukhi field"
          placeholder="canonical Gurmukhi"
          value={draft.canonical}
          onChange={(e) => setDraft({ ...draft, canonical: e.target.value })}
          style={{ fontSize: 16, lineHeight: 1.4 }}
        />
        <input
          className="gurmukhi field"
          placeholder="misrecognitions, comma-separated"
          value={draft.forms}
          onChange={(e) => setDraft({ ...draft, forms: e.target.value })}
          style={{ fontSize: 16, lineHeight: 1.4, flex: 1, minWidth: 220 }}
        />
        <select className="field" value={draft.type} onChange={(e) => setDraft({ ...draft, type: e.target.value as EntityType })}>
          {ENTITY_TYPES.map((t) => <option key={t}>{t}</option>)}
        </select>
        <input
          className="field"
          placeholder="English gloss"
          value={draft.gloss}
          onChange={(e) => setDraft({ ...draft, gloss: e.target.value })}
        />
        <button disabled={!draft.canonical.trim()} onClick={create}>Add</button>
      </div>

      <table>
        <thead>
          <tr>
            <th>Canonical</th><th>Misrecognitions</th><th>Type</th><th>Gloss</th>
            <th>Threshold</th><th>Used</th><th></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => editing?.id === r.id ? (
            <tr key={r.id}>
              <td>
                <input
                  className="gurmukhi field"
                  value={editing.canonical_gurmukhi}
                  onChange={(e) => setEditing({ ...editing, canonical_gurmukhi: e.target.value })}
                  style={{ fontSize: 16, lineHeight: 1.4, width: '100%' }}
                />
              </td>
              <td>
                <input
                  className="gurmukhi field"
                  value={editingForms}
                  onChange={(e) => setEditingForms(e.target.value)}
                  style={{ fontSize: 16, lineHeight: 1.4, width: '100%' }}
                />
              </td>
              <td>
                <select className="field" value={editing.entity_type} onChange={(e) => setEditing({ ...editing, entity_type: e.target.value as EntityType })}>
                  {ENTITY_TYPES.map((t) => <option key={t}>{t}</option>)}
                </select>
              </td>
              <td>
                <input
                  className="field"
                  value={editing.english_gloss ?? ''}
                  onChange={(e) => setEditing({ ...editing, english_gloss: e.target.value || null })}
                />
              </td>
              <td>
                <input
                  className="field"
                  type="number" step="0.05" min="0" max="1"
                  value={editing.confidence_threshold}
                  onChange={(e) => setEditing({ ...editing, confidence_threshold: Number(e.target.value) })}
                  style={{ width: 80 }}
                />
              </td>
              <td>{r.times_applied}</td>
              <td>
                <button onClick={save}>Save</button>{' '}
                <button onClick={() => setEditing(null)}>Cancel</button>
              </td>
            </tr>
          ) : (
            <tr key={r.id} style={{ opacity: r.retired ? 0.5 : 1 }}>
              <td>
                <div className="gurmukhi" style={{ fontSize: 17, lineHeight: 1.5 }}>{r.canonical_gurmukhi}</div>
                <div style={{ color: 'var(--muted)', fontSize: 12 }}>{r.romanized}</div>
              </td>
              <td className="gurmukhi" style={{ fontSize: 15, lineHeight: 1.5 }}>
                {r.surface_forms.map((f) => (
                  <div key={f}>{f} <span style={{ color: 'var(--muted)', fontSize: 12 }}>{romanize(f)}</span></div>
                ))}
              </td>
              <td>{r.entity_type}</td>
              <td>{r.english_gloss ?? '—'}</td>
              <td>
                {r.confidence_threshold.toFixed(2)}
                {!r.embedding_model && (
                  <div style={{ color: 'var(--muted)', fontSize: 11 }} title="No context has been learned yet, so this entry can only suggest, never auto-apply">
                    no context yet
                  </div>
                )}
              </td>
              <td>{r.times_applied}</td>
              <td>
                <button onClick={() => startEdit(r)}>Edit</button>{' '}
                <button onClick={() => toggleRetired(r)}>{r.retired ? 'Restore' : 'Retire'}</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}
