/**
 * Words for the square-kufic decor. Single words only, drawn from what the site keeps
 * returning to; mixed Turkish / French / English the way the interests are. Every word must
 * be spellable with the glyphs in src/lib/kufic.ts (Turkish dotless I is fine; no dotted
 * capitals or other diacritics).
 */
export const kuficWords = [
  'KUBBE', // dome: Kalemkubbe, the Istanbul mosques
  'HARF', // letter: hat, Ottoman linguistics
  'AKINTI', // current: The Current, the Bosphorus current
  'BOSPHORE', // the 1819 map
  'DENGE', // equilibrium: game theory
  'ODAK', // focal point: coordination, beauty contests
  'QUORUM', // distributed systems
  'ABSURDE', // Camus
  'APOLLON', // the archetype
  'ZEITGEIST', // the monthly record
  'KADER', // fate
  'NOMOS', // law, order
] as const;

/** Hero composition: four words that stack into a near-square block. */
export const heroWords = ['KUBBE', 'DENGE', 'AKINTI', 'HARF'];

/** Deterministic pick per article, so a post always wears the same words. */
export function wordsFor(slug: string, n = 2): string[] {
  let h = 2166136261;
  for (const c of slug) h = Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0;
  const out: string[] = [];
  for (let i = 0; out.length < n; i++) {
    const w = kuficWords[(h + i * 7) % kuficWords.length];
    if (!out.includes(w)) out.push(w);
  }
  return out;
}
