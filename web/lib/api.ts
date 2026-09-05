import type { Clip, ClipListItem, EntityType, LedgerEntry, Segment, SettingSpec } from './types'

const API = process.env.NEXT_PUBLIC_API ?? 'http://localhost:8000'

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    // FastAPI puts the human-readable reason in `detail`; surface it so a 409
    // ("span no longer present") reads as a sentence, not a status code.
    let detail = ''
    try {
      const body = await res.json()
      detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail ?? '')
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${init?.method ?? 'GET'} ${path} → ${res.status}${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<T>
}

export const fetchClips = () => json<{ clips: ClipListItem[] }>('/api/clips').then((r) => r.clips)

export const fetchClip = (id: number) =>
  json<{ clip: Clip; segments: Segment[] }>(`/api/clips/${id}`)

export const patchReciter = (id: number, reciter_speaker_id: string | null) =>
  json(`/api/clips/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ reciter_speaker_id }),
  })

export const patchSegment = (id: number, text_pa: string) =>
  json(`/api/segments/${id}`, { method: 'PATCH', body: JSON.stringify({ text_pa }) })

// The retry endpoint 404s if the clip doesn't exist and also reports whether the
// job was actually requeued (it may already be queued/running).
export const retryClip = (id: number) =>
  json<{ ok: boolean; requeued: boolean }>(`/api/clips/${id}/retry`, { method: 'POST' })

export const audioUrl = (clipId: number) => `${API}/api/audio/${clipId}`

// --- correction review -------------------------------------------------------

export type ResolveBody =
  | { action: 'reject' }
  | { action: 'accept'; text: string; remember?: { entity_type: EntityType; english_gloss?: string } }

export const resolveFlag = (id: number, body: ResolveBody) =>
  json<{ ok: boolean; status: string; text_pa?: string }>(`/api/flags/${id}/resolve`, {
    method: 'POST',
    body: JSON.stringify(body),
  })

export const revertCorrection = (id: number) =>
  json<{ ok: boolean; text_pa: string }>(`/api/corrections/${id}/revert`, { method: 'POST' })

export const fetchLedger = (q: string, includeRetired: boolean) =>
  json<{ entries: LedgerEntry[] }>(
    `/api/ledger?q=${encodeURIComponent(q)}&include_retired=${includeRetired}`,
  ).then((r) => r.entries)

export const createLedger = (body: {
  canonical_gurmukhi: string
  surface_forms: string[]
  entity_type: EntityType
  english_gloss?: string
}) => json<{ ok: boolean; id: number }>('/api/ledger', { method: 'POST', body: JSON.stringify(body) })

export type LedgerPatch = Partial<
  Pick<LedgerEntry, 'canonical_gurmukhi' | 'surface_forms' | 'entity_type' | 'english_gloss' | 'confidence_threshold' | 'retired'>
>

export const patchLedger = (id: number, body: LedgerPatch) =>
  json(`/api/ledger/${id}`, { method: 'PATCH', body: JSON.stringify(body) })

export const fetchSettings = () =>
  json<{ values: Record<string, string>; specs: SettingSpec[] }>('/api/settings')

export const putSettings = (values: Record<string, string>) =>
  json<{ ok: boolean; values: Record<string, string> }>('/api/settings', {
    method: 'PUT',
    body: JSON.stringify({ values }),
  })
