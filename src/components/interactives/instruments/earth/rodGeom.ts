/**
 * Episode 2 (a shared unit): lay rods end to end along the hilltop precinct, from the
 * minaret's foot M toward the channel (100 m of flat paving, site.ts FLAT_X). Count whole
 * rods, read the remainder on the last rod's divisions, and see what a rod that is a little
 * short does to the total.
 * Units: metres. The precinct is FLAT_X long; the rods are the Egyptian royal cubit (52.5 cm,
 * 7 palms, 28 digits) or the Ottoman architect's arşın (75.774 cm, 24 parmak).
 * Screen: SVG viewBox units, y down.
 */
import { SITE } from './site.ts';

export const UNITS = {
  cubit: { id: 'cubit', name: 'Egyptian royal cubit', len: 0.525, sub: 28, subName: 'digits', subShort: 'digits', major: 4 },
  arsin: { id: 'arsin', name: 'Ottoman architect’s arşın', len: 0.75774, sub: 24, subName: 'parmak', subShort: 'parmak', major: 6 },
} as const;
export type UnitId = keyof typeof UNITS;
export const SPAN = SITE.FLAT_X; // 100 m: M to the start of the channel

/** Rod error, mm per rod (negative = rod short). */
export const ROD_ERR = { MIN: -3, MAX: 3, STEP: 0.5 };

/**
 * Measuring SPAN with a rod whose true length is len + err: the reader counts n whole rods
 * and reads r subdivisions on the last one (assuming the rod is the standard length).
 * The measured length is then (n + r/sub)·len, the truth is SPAN.
 */
export function measure(unit: UnitId, errMm: number) {
  const u = UNITS[unit];
  const real = u.len + errMm / 1000;
  const nExact = SPAN / real;
  const n = Math.floor(nExact);
  const rem = SPAN - n * real; // m left after n real rods
  const subs = (rem / real) * u.sub; // in the rod's own (stretched or shrunk) divisions
  const subsRead = Math.round(subs * 2) / 2; // read to half a division
  const measured = (n + subsRead / u.sub) * u.len;
  return { u, real, n, rem, subs, subsRead, measured, error: measured - SPAN };
}

/**
 * Overview: the whole precinct with the rods laid so far (blocks of ten when rods are thin).
 * Zoom: the last few rods before the end mark and one more rod laid past it, whose divisions
 * are read at the end mark. All rods drawn at their REAL length (err applied), so a short rod
 * visibly needs more of itself.
 */
export function rodScene(W: number, unit: UnitId, errMm: number, laid: number, read: number) {
  const m = measure(unit, errMm);
  const { u, real, n } = m;
  const H = 230;
  const pad = 16;
  const k = (W - 2 * pad) / SPAN; // overview px per metre
  const oy = 52; // overview strip top
  const oh = 14;
  const shown = Math.max(0, Math.min(n, laid));
  const head = pad + shown * real * k;
  const endX = pad + SPAN * k;
  const block = real * k < 3 ? 10 : 1; // draw blocks of 10 rods when one rod is under 3 px
  const blocks: { x0: number; x1: number; odd: boolean }[] = [];
  for (let i = 0; i < shown; i += block) {
    const j = Math.min(shown, i + block);
    blocks.push({ x0: pad + i * real * k, x1: pad + j * real * k, odd: (i / block) % 2 === 1 });
  }
  // zoom window: the end of the last whole rod and the extra rod laid past the end mark
  const zx0w = n * real - 0.45 * real;
  const zx1w = n * real + 1.12 * real;
  const zk = (W - 2 * pad) / (zx1w - zx0w);
  const zX = (xw: number) => pad + (xw - zx0w) * zk;
  const zy = 128;
  const zh = 26;
  const zRods: { x0: number; x1: number; i: number; extra: boolean }[] = [];
  const visible = laid >= n ? n + 1 : shown; // the extra rod appears once all whole rods are down
  for (let i = Math.max(0, n - 5); i < visible; i++) {
    const a = i * real;
    const b = a + real;
    if (b < zx0w || a > zx1w) continue;
    zRods.push({ x0: zX(Math.max(a, zx0w)), x1: zX(Math.min(b, zx1w)), i, extra: i === n });
  }
  const extraX0 = zX(n * real);
  const divs: { x: number; major: boolean; label?: string }[] = [];
  for (let d = 0; d <= u.sub; d++) {
    const xw = n * real + (d * real) / u.sub;
    if (xw > zx1w) break;
    const major = d % u.major === 0;
    divs.push({ x: zX(xw), major, label: major ? String(d) : undefined });
  }
  const pointer = zX(n * real + (read * real) / u.sub);
  const endZ = zX(SPAN);
  const readOk = Math.abs(read - m.subsRead) < 1e-9;
  return { ...m, W, H, pad, k, oy, oh, shown, head, endX, blocks, zx0w, zx1w, zk, zy, zh, zRods, extraX0, divs, pointer, endZ, readOk, visible };
}
