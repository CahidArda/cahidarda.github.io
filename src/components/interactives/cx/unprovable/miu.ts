/**
 * Hofstadter's MIU system (Gödel, Escher, Bach, 1979, ch. I). Axiom MI, four one-way rules.
 * `apply` acts at the first (leftmost) match, which is what the playground buttons do;
 * `successors` applies every rule at every position, for the reachability check.
 */
export type Rule = 1 | 2 | 3 | 4;

/** Hofstadter's wording, verbatim (GEB ch. I, "The MU-puzzle"). */
export const RULE_TEXT: Record<Rule, string> = {
  1: 'RULE I: If you possess a string whose last letter is I, you can add on a U at the end.',
  2: 'RULE II: Suppose you have Mx. Then you may add Mxx to your collection.',
  3: 'RULE III: If III occurs in one of the strings in your collection, you may make a new string with U in place of III.',
  4: 'RULE IV: If UU occurs inside one of your strings, you can drop it.',
};

export const RULE_SHORT: Record<Rule, string> = {
  1: 'ends in I: add U',
  2: 'double after M',
  3: 'III becomes U',
  4: 'delete UU',
};

export const iCount = (s: string) => {
  let n = 0;
  for (const c of s) if (c === 'I') n++;
  return n;
};

/** Index of the first match the rule acts on, or -1 when the rule does not apply. */
export function matchAt(s: string, r: Rule): number {
  if (r === 1) return s.endsWith('I') ? s.length - 1 : -1;
  if (r === 2) return s.length > 1 ? 1 : -1;
  if (r === 3) return s.indexOf('III');
  return s.indexOf('UU');
}

export function apply(s: string, r: Rule, at = matchAt(s, r)): string | null {
  if (at < 0) return null;
  if (r === 1) return s.endsWith('I') ? s + 'U' : null;
  if (r === 2) return 'M' + s.slice(1) + s.slice(1);
  if (r === 3) return s.slice(at, at + 3) === 'III' ? s.slice(0, at) + 'U' + s.slice(at + 3) : null;
  return s.slice(at, at + 2) === 'UU' ? s.slice(0, at) + s.slice(at + 2) : null;
}

export function successors(s: string): string[] {
  const out = new Set<string>();
  const a = apply(s, 1);
  if (a) out.add(a);
  const b = apply(s, 2);
  if (b) out.add(b);
  for (let i = 0; i + 3 <= s.length; i++) {
    const c = apply(s, 3, i);
    if (c) out.add(c);
  }
  for (let i = 0; i + 2 <= s.length; i++) {
    const d = apply(s, 4, i);
    if (d) out.add(d);
  }
  return [...out];
}

/** Every string reachable from MI in at most `depth` rule applications. */
export function reachable(depth: number): Set<string> {
  const seen = new Set<string>(['MI']);
  let frontier = ['MI'];
  for (let d = 0; d < depth; d++) {
    const next: string[] = [];
    for (const s of frontier)
      for (const t of successors(s))
        if (!seen.has(t)) {
          seen.add(t);
          next.push(t);
        }
    frontier = next;
  }
  return seen;
}
