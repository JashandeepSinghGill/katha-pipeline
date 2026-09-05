import type { ReactNode } from 'react'

/** Wrap the first occurrence of each span in <mark>. Overlapping spans are dropped. */
export function highlight(text: string, spans: string[]): ReactNode[] {
  const ranges: [number, number][] = []
  for (const s of spans) {
    if (!s) continue
    const at = text.indexOf(s)
    if (at < 0) continue
    const end = at + s.length
    if (ranges.some(([a, b]) => at < b && end > a)) continue
    ranges.push([at, end])
  }
  ranges.sort((x, y) => x[0] - y[0])
  const out: ReactNode[] = []
  let cursor = 0
  ranges.forEach(([a, b], i) => {
    if (a > cursor) out.push(text.slice(cursor, a))
    out.push(<mark key={i}>{text.slice(a, b)}</mark>)
    cursor = b
  })
  if (cursor < text.length) out.push(text.slice(cursor))
  return out
}
