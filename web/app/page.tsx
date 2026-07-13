'use client'

import { useEffect, useState } from 'react'
import { fetchClips, retryClip } from '@/lib/api'
import type { Clip } from '@/lib/types'

function duration(seconds: number | null): string {
  if (!seconds) return '—'
  const m = Math.floor(seconds / 60)
  return `${m}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}

export default function Dashboard() {
  const [clips, setClips] = useState<Clip[]>([])
  const [error, setError] = useState<string | null>(null)

  const load = () => fetchClips().then(setClips).catch((e) => setError(String(e)))

  useEffect(() => {
    load()
    const timer = setInterval(load, 5000) // stages are minutes long; 5s is plenty
    return () => clearInterval(timer)
  }, [])

  if (error) return <p className="pill failed">{error}</p>

  return (
    <table>
      <thead>
        <tr>
          <th>Clip</th><th>Length</th><th>Stage</th>
          <th>Segments</th><th>Loose timings</th><th></th>
        </tr>
      </thead>
      <tbody>
        {clips.map((c) => (
          <tr key={c.id}>
            <td>
              {c.status === 'done'
                ? <a href={`/clips/${c.id}`}>{c.title}</a>
                : c.title}
              {c.last_error && <div className="pill failed">{c.last_error}</div>}
            </td>
            <td>{duration(c.duration_seconds)}</td>
            <td><span className={`pill ${c.status}`}>{c.stage} · {c.status}</span></td>
            <td>{c.segment_count}</td>
            <td>{c.loose_count > 0 ? <span className="pill failed">{c.loose_count}</span> : '—'}</td>
            <td>
              {c.status === 'failed' && (
                <button onClick={() => retryClip(c.id).then(load)}>Retry</button>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
