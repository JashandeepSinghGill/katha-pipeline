'use client'

import { useState } from 'react'
import { FlagPanel } from '@/components/FlagPanel'
import type { ResolveBody } from '@/lib/api'
import { highlight } from '@/lib/highlight'
import { romanize } from '@/lib/roman'
import type { Segment } from '@/lib/types'

const clock = (ms: number) => {
  const total = Math.floor(ms / 1000)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

const roleColour: Record<Segment['speaker_role'], string> = {
  reciter: 'var(--reciter)',
  kathakar: 'var(--kathakar)',
  unknown: 'var(--unknown)',
}

export function SegmentList({
  segments,
  activeId,
  onPlay,
  onSave,
  onResolve,
  onRevert,
}: {
  segments: Segment[]
  activeId: number | null
  onPlay: (segment: Segment) => void
  onSave: (id: number, text: string) => Promise<void>
  onResolve: (flagId: number, body: ResolveBody) => Promise<void>
  onRevert: (correctionId: number) => Promise<void>
}) {
  const [editingId, setEditingId] = useState<number | null>(null)
  const [draft, setDraft] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [revertError, setRevertError] = useState<string | null>(null)

  const startEdit = (s: Segment) => {
    setEditingId(s.id)
    setDraft(s.text_pa)
    setSaveError(null)
  }

  const cancelEdit = () => {
    setEditingId(null)
    setSaveError(null)
  }

  // Keep the editor open until the save resolves — closing it eagerly is how a
  // failed PATCH silently loses the admin's correction. On failure the draft and
  // the editor stay put so they can retry; on success we close it.
  const save = async (id: number) => {
    setSaving(true)
    setSaveError(null)
    try {
      await onSave(id, draft)
      setEditingId(null)
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const revert = (id: number) => {
    setRevertError(null)
    onRevert(id).catch((e) => setRevertError(e instanceof Error ? e.message : String(e)))
  }

  return (
    <div>
      {revertError && <p className="pill failed">{revertError}</p>}
      {segments.map((s) => {
        const colour = roleColour[s.speaker_role]
        const openFlags = s.flags.filter((f) => f.status === 'open')
        const resolvedFlags = s.flags.filter((f) => f.status !== 'open' && f.status !== 'auto_applied')
        const activeCorrections = s.corrections.filter((c) => !c.reverted)
        return (
          <div
            key={s.id}
            style={{
              padding: '12px 14px',
              margin: '8px 0',
              borderRadius: 8,
              borderLeft: `3px solid ${colour}`,
              background: activeId === s.id ? 'var(--panel)' : 'transparent',
            }}
          >
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', fontSize: 13 }}>
              <button onClick={() => onPlay(s)}>▶ {clock(s.start_ms)}</button>
              <span style={{ color: colour }}>{s.speaker_role}</span>
              {s.timing_source === 'asr' && (
                <span className="pill failed" title="Forced alignment did not map to this clip; span is only as tight as the ASR sentence chunk">
                  loose timing
                </span>
              )}
              {s.edited && <span className="pill">edited</span>}
              {openFlags.length > 0 && (
                <span className="pill flagged">{openFlags.length} flag{openFlags.length > 1 ? 's' : ''}</span>
              )}
            </div>

            {editingId === s.id ? (
              <>
                <textarea
                  className="gurmukhi"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  style={{
                    width: '100%', marginTop: 8, background: 'var(--bg)',
                    color: 'var(--ink)', border: '1px solid var(--line)',
                    borderRadius: 6, padding: 8,
                  }}
                  rows={2}
                />
                <button onClick={() => save(s.id)} disabled={saving}>
                  {saving ? 'Saving…' : 'Save'}
                </button>
                <button onClick={cancelEdit} disabled={saving} style={{ marginLeft: 6 }}>
                  Cancel
                </button>
                {saveError && (
                  <div style={{ color: 'var(--warn)', fontSize: 13, marginTop: 6 }}>
                    Save failed: {saveError}
                  </div>
                )}
              </>
            ) : (
              <div
                className="gurmukhi"
                onDoubleClick={() => startEdit(s)}
                title="Double-click to correct"
                style={{ marginTop: 4 }}
              >
                {highlight(s.text_pa, openFlags.map((f) => f.span_text))}
              </div>
            )}

            <div style={{ color: 'var(--muted)', fontSize: 13 }}>{romanize(s.text_pa)}</div>

            {activeCorrections.map((c) => (
              <div key={c.id} className="correction">
                <span className={`pill ${c.source === 'auto' ? 'done' : ''}`}>
                  {c.source === 'auto' ? 'auto-fixed' : 'corrected'}
                </span>
                <span className="gurmukhi" style={{ fontSize: 15, lineHeight: 1.4 }}>
                  {c.before_text} → {c.after_text}
                </span>
                {c.context_similarity !== null && (
                  <span style={{ color: 'var(--muted)', fontSize: 12 }}>
                    sim {c.context_similarity.toFixed(2)}
                  </span>
                )}
                <button onClick={() => revert(c.id)}>Revert</button>
              </div>
            ))}

            {openFlags.map((f) => (
              <FlagPanel key={f.id} flag={f} onResolve={(body) => onResolve(f.id, body)} />
            ))}

            {resolvedFlags.length > 0 && (
              <div style={{ color: 'var(--muted)', fontSize: 12, marginTop: 4 }}>
                {resolvedFlags.length} flag{resolvedFlags.length > 1 ? 's' : ''} resolved
                {' '}({resolvedFlags.map((f) => f.status).join(', ')})
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
