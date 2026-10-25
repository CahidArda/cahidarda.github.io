/**
 * Tiny assertion kit for plate sweep scripts (see .claude/skills/verify-interactives/SKILL.md).
 * No deps; run sweeps with `node --experimental-strip-types <file>.ts` (Node 22) or `node` (23+).
 * Failures are grouped by tag so a 4-million-check sweep prints one line per kind of bug.
 */
export type Pt = { x: number; y: number };
export type Box = { x0: number; y0: number; x1: number; y1: number; name: string };

let checks = 0;
const fails = new Map<string, { n: number; first: string }>();

/** Record one assertion. `tag` names the kind of bug; `detail` (lazy) shows the first case. */
export function ok(cond: boolean, tag: string, detail: () => string = () => '') {
  checks++;
  if (cond) return;
  const f = fails.get(tag);
  if (f) f.n++;
  else fails.set(tag, { n: 1, first: detail() });
}

/** Relative-or-absolute closeness: |a − b| ≤ tol · max(1, |b|). */
export const near = (a: number, b: number, tol = 1e-6) =>
  Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

/** Inclusive numeric range, e.g. range(5, 30, 0.5) for a slider (min, max, step). */
export function range(min: number, max: number, step: number) {
  const out: number[] = [];
  const n = Math.round((max - min) / step);
  for (let i = 0; i <= n; i++) out.push(+(min + i * step).toFixed(10));
  return out;
}

/** Approximate box of an SVG text label. `w` = average glyph width in em (serif ≈ 0.5, mono 0.6). */
export function textBox(
  name: string,
  x: number,
  y: number,
  text: string,
  size: number,
  anchor: 'start' | 'middle' | 'end' = 'start',
  baseline: 'alphabetic' | 'central' = 'alphabetic',
  w = 0.56,
): Box {
  const len = [...text].length * size * w;
  const x0 = anchor === 'start' ? x : anchor === 'middle' ? x - len / 2 : x - len;
  const y0 = baseline === 'central' ? y - size * 0.5 : y - size * 0.78;
  return { name, x0, y0, x1: x0 + len, y1: y0 + size };
}

export const overlap = (a: Box, b: Box, pad = 0) =>
  a.x0 < b.x1 + pad && b.x0 < a.x1 + pad && a.y0 < b.y1 + pad && b.y0 < a.y1 + pad;

/** Is the box inside a W × H drawing with margin m? */
export const inside = (b: Box, W: number, H: number, m = 0) =>
  b.x0 >= m && b.y0 >= m && b.x1 <= W - m && b.y1 <= H - m;

/** Does segment p→q pass through box b shrunk by `pad`? Sampled every half unit. */
export function segHitsBox(p: Pt, q: Pt, b: Box, pad = 0) {
  const n = Math.ceil(Math.hypot(q.x - p.x, q.y - p.y) / 0.5) + 1;
  for (let i = 0; i <= n; i++) {
    const x = p.x + ((q.x - p.x) * i) / n;
    const y = p.y + ((q.y - p.y) * i) / n;
    if (x > b.x0 + pad && x < b.x1 - pad && y > b.y0 + pad && y < b.y1 - pad) return true;
  }
  return false;
}

/** Assert no two label boxes overlap. */
export function noCollisions(boxes: Box[], tag: string, ctx: string, pad = 1) {
  for (let i = 0; i < boxes.length; i++)
    for (let j = i + 1; j < boxes.length; j++)
      ok(
        !overlap(boxes[i], boxes[j], pad),
        `${tag}: '${boxes[i].name}' vs '${boxes[j].name}'`,
        () => ctx,
      );
}

/** Print the summary and exit 1 on any failure. Call once at the end. */
export function report(extra: string[] = []) {
  console.log(`${checks} checks`);
  for (const line of extra) console.log(line);
  if (!fails.size) return console.log('all passed');
  console.log(`\nFAILED (${fails.size} kinds):`);
  for (const [k, v] of fails) console.log(`  ${k}  ×${v.n}  e.g. ${v.first}`);
  process.exit(1);
}
