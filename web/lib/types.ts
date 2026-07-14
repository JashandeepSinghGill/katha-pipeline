export type Word = { text: string; start_ms: number; end_ms: number }

export type Segment = {
  id: number
  idx: number
  speaker_id: string
  speaker_role: 'reciter' | 'kathakar' | 'unknown'
  start_ms: number
  end_ms: number
  text_pa: string
  align_score: number | null
  timing_source: 'aligned' | 'asr'
  edited: boolean
  words: Word[]
}

// Columns that actually exist on the `clips` table — this is what
// `GET /api/clips/{id}` returns (`SELECT * FROM clips`).
export interface Clip {
  id: number
  title: string
  duration_seconds: number | null
  reciter_speaker_id: string | null
}

// `GET /api/clips` additionally joins `jobs` and computes segment/loose
// counts. Those fields only exist on the list endpoint, never on a single
// clip fetch — keep them off the base `Clip` type.
export interface ClipListItem extends Clip {
  stage: 'prepare' | 'asr' | 'align' | 'assemble' | 'complete'
  status: 'queued' | 'running' | 'failed' | 'done'
  attempts: number
  last_error: string | null
  segment_count: number
  loose_count: number
}
