export type Word = { text: string; start_ms: number; end_ms: number }

export type FlagStatus = 'open' | 'accepted' | 'rejected' | 'auto_applied' | 'stale'
export type EntityType = 'person' | 'place' | 'date' | 'book' | 'title' | 'term' | 'other'
export const ENTITY_TYPES: EntityType[] = ['person', 'place', 'date', 'book', 'title', 'term', 'other']

export type Flag = {
  id: number
  span_text: string
  kind: string
  reason: string
  candidates: string[]
  ledger_id: number | null
  status: FlagStatus
  resolved_text: string | null
}

export type Correction = {
  id: number
  flag_id: number | null
  ledger_id: number | null
  source: 'auto' | 'admin'
  before_text: string
  after_text: string
  context_similarity: number | null
  reverted: boolean
}

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
  flags: Flag[]
  corrections: Correction[]
}

// The explicit column allowlist `GET /api/clips/{id}` selects (repo.get_clip).
// Deliberately narrower than the full `clips` row: source_path (a server
// filesystem path) and sha256 are excluded so they never reach the browser.
export interface Clip {
  id: number
  title: string
  duration_seconds: number | null
  reciter_speaker_id: string | null
}

// `GET /api/clips` additionally joins `jobs` and computes segment/loose/flag
// counts. Those fields only exist on the list endpoint, never on a single
// clip fetch — keep them off the base `Clip` type.
export interface ClipListItem extends Clip {
  stage: 'prepare' | 'asr' | 'align' | 'assemble' | 'correct' | 'complete'
  status: 'queued' | 'running' | 'failed' | 'done'
  attempts: number
  last_error: string | null
  segment_count: number
  loose_count: number
  open_flags: number
  auto_applied: number
}

export type LedgerEntry = {
  id: number
  canonical_gurmukhi: string
  romanized: string
  surface_forms: string[]
  entity_type: EntityType
  english_gloss: string | null
  embedding_model: string | null
  confidence_threshold: number
  times_applied: number
  retired: boolean
  last_used_at: string | null
}

export type SettingSpec = {
  key: string
  kind: 'int' | 'float' | 'bool' | 'str'
  default: string
  minimum: number | null
  maximum: number | null
  help: string
}
