import type { Clip, ClipListItem, Segment } from './types'

const API = process.env.NEXT_PUBLIC_API ?? 'http://localhost:8000'

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) throw new Error(`${init?.method ?? 'GET'} ${path} → ${res.status}`)
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
