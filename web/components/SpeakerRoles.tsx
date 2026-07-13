'use client'

import type { Clip, Segment } from '@/lib/types'

export function SpeakerRoles({
  clip,
  segments,
  onChange,
}: {
  clip: Clip
  segments: Segment[]
  onChange: (speakerId: string | null) => void
}) {
  const speakers = [...new Set(segments.map((s) => s.speaker_id))].sort()

  return (
    <div style={{ margin: '16px 0', padding: 12, border: '1px solid var(--line)', borderRadius: 8 }}>
      <div style={{ color: 'var(--muted)', fontSize: 13, marginBottom: 8 }}>
        Which speaker is the reciter? (Guessed from airtime — verify by ear.)
      </div>
      {speakers.map((id) => (
        <button
          key={id}
          onClick={() => onChange(id)}
          style={{
            marginRight: 8,
            borderColor: clip.reciter_speaker_id === id ? 'var(--reciter)' : undefined,
            color: clip.reciter_speaker_id === id ? 'var(--reciter)' : undefined,
          }}
        >
          Speaker {id} {clip.reciter_speaker_id === id && '· reciter'}
        </button>
      ))}
      <button onClick={() => onChange(null)}>Neither / unclear</button>
    </div>
  )
}
