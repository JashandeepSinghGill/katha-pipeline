'use client'

import { use, useCallback, useEffect, useRef, useState } from 'react'
import { SegmentList } from '@/components/SegmentList'
import { SpeakerRoles } from '@/components/SpeakerRoles'
import { audioUrl, fetchClip, patchReciter, patchSegment } from '@/lib/api'
import type { Clip, Segment } from '@/lib/types'

export default function Review({ params }: { params: Promise<{ id: string }> }) {
  const clipId = Number(use(params).id)
  const audio = useRef<HTMLAudioElement>(null)
  const stopAt = useRef<number | null>(null)

  const [clip, setClip] = useState<Clip | null>(null)
  const [segments, setSegments] = useState<Segment[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)

  const load = useCallback(
    () => fetchClip(clipId).then((r) => { setClip(r.clip); setSegments(r.segments) }),
    [clipId],
  )
  useEffect(() => { load() }, [load])

  // Play exactly the segment's span and stop at its end (spec §7: "auto-plays only that span").
  const play = (segment: Segment) => {
    const el = audio.current
    if (!el) return
    setActiveId(segment.id)
    stopAt.current = segment.end_ms / 1000
    el.currentTime = segment.start_ms / 1000
    void el.play()
  }

  const onTimeUpdate = () => {
    const el = audio.current
    if (el && stopAt.current !== null && el.currentTime >= stopAt.current) {
      el.pause()
      stopAt.current = null
    }
  }

  if (!clip) return <p>Loading…</p>

  return (
    <>
      <h2>{clip.title}</h2>

      <audio
        ref={audio}
        src={audioUrl(clipId)}
        onTimeUpdate={onTimeUpdate}
        controls
        preload="metadata"
        style={{ width: '100%', position: 'sticky', top: 0, zIndex: 1 }}
      />

      <SpeakerRoles
        clip={clip}
        segments={segments}
        onChange={(id) => patchReciter(clipId, id).then(load)}
      />

      <SegmentList
        segments={segments}
        activeId={activeId}
        onPlay={play}
        onSave={(id, text) => patchSegment(id, text).then(load)}
      />
    </>
  )
}
