'use client'

import { useState } from 'react'
import { romanize } from '@/lib/roman'
import type { Segment } from '@/lib/types'

const clock = (ms: number) => {
  const total = Math.floor(ms / 1000)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

export function SegmentList({
  segments,
  activeId,
  onPlay,
  onSave,
}: {
  segments: Segment[]
  activeId: number | null
  onPlay: (segment: Segment) => void
  onSave: (id: number, text: string) => void
}) {
  const [editingId, setEditingId] = useState<number | null>(null)
  const [draft, setDraft] = useState('')

  return (
    <div>
      {segments.map((s) => {
        const colour = s.speaker_role === 'reciter' ? 'var(--reciter)' : 'var(--kathakar)'
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
                <button onClick={() => { onSave(s.id, draft); setEditingId(null) }}>Save</button>
                <button onClick={() => setEditingId(null)} style={{ marginLeft: 6 }}>Cancel</button>
              </>
            ) : (
              <div
                className="gurmukhi"
                onDoubleClick={() => { setEditingId(s.id); setDraft(s.text_pa) }}
                title="Double-click to correct"
                style={{ marginTop: 4 }}
              >
                {s.text_pa}
              </div>
            )}

            <div style={{ color: 'var(--muted)', fontSize: 13 }}>{romanize(s.text_pa)}</div>
          </div>
        )
      })}
    </div>
  )
}
