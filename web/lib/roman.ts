import pkg from 'gurmukhi-utils'

const { toEnglish } = pkg as { toEnglish: (t: string) => string }

/** Romanized Punjabi is the review aid, per spec §3.3: easier to scan and to type
 *  than Gurmukhi. Gurmukhi remains the canonical stored form — never write this back. */
export function romanize(gurmukhi: string): string {
  try {
    return toEnglish(gurmukhi)
  } catch {
    return ''
  }
}
