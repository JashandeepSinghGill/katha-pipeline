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

export type Clip = {
  id: number
  title: string
  duration_seconds: number | null
  reciter_speaker_id: string | null
  stage: 'prepare' | 'asr' | 'align' | 'assemble' | 'complete'
  status: 'queued' | 'running' | 'failed' | 'done'
  attempts: number
  last_error: string | null
  segment_count: number
  loose_count: number
}
