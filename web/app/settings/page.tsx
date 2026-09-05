'use client'

import { useEffect, useState } from 'react'
import { fetchSettings, putSettings } from '@/lib/api'
import type { SettingSpec } from '@/lib/types'

export default function SettingsPage() {
  const [specs, setSpecs] = useState<SettingSpec[]>([])
  const [values, setValues] = useState<Record<string, string>>({})
  const [saved, setSaved] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    fetchSettings()
      .then((r) => { setSpecs(r.specs); setValues(r.values); setSaved(r.values) })
      .catch((e) => setError(String(e)))
  }, [])

  const dirty = Object.keys(values).filter((k) => values[k] !== saved[k])

  const save = async () => {
    setBusy(true)
    setError(null)
    try {
      const r = await putSettings(Object.fromEntries(dirty.map((k) => [k, values[k]])))
      setValues(r.values)
      setSaved(r.values)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <h2>Settings</h2>
      <p style={{ color: 'var(--muted)', fontSize: 13 }}>
        These live in the database and take effect on the worker&apos;s next run. Paths, API keys and the
        LLM backend are environment variables (see <code>.env</code>).
      </p>
      {error && <p className="pill failed">{error}</p>}
      <table>
        <tbody>
          {specs.map((s) => (
            <tr key={s.key}>
              <td style={{ width: 280 }}>
                <code>{s.key}</code>
                <div style={{ color: 'var(--muted)', fontSize: 12 }}>{s.help}</div>
              </td>
              <td>
                {s.kind === 'bool' ? (
                  <select
                    className="field"
                    value={values[s.key] ?? s.default}
                    onChange={(e) => setValues({ ...values, [s.key]: e.target.value })}
                  >
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                ) : (
                  <input
                    className="field"
                    type={s.kind === 'str' ? 'text' : 'number'}
                    step={s.kind === 'float' ? 'any' : 1}
                    min={s.minimum ?? undefined}
                    max={s.maximum ?? undefined}
                    value={values[s.key] ?? s.default}
                    onChange={(e) => setValues({ ...values, [s.key]: e.target.value })}
                    style={{ width: s.kind === 'str' ? 320 : 120 }}
                  />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 12 }}>
        <button disabled={busy || dirty.length === 0} onClick={save}>
          {busy ? 'Saving…' : dirty.length ? `Save (${dirty.length})` : 'Save'}
        </button>
      </div>
    </>
  )
}
