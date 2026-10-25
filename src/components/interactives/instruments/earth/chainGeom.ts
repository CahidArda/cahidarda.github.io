/**
 * Episode 4a: Gunter's chain along the cut M → A (site.ts): 200 m of hilltop, then the road
 * down at 1 in 8 to the quay. Horizontal distance 480 m.
 * A chain is 66 ft = 20.1168 m = 100 links, with brass tags every 10 links.
 * Two ways to lay it on the slope: on the surface (reads the slope length) or "stepped"
 * (held level, the end dropped with a plumb line: reads the horizontal length).
 * Screen: SVG viewBox units, y down; section drawn with heights ×VE (said on the drawing).
 */
import { CREST_Z, LINE_H, LINE_SURFACE, SITE, groundZ } from './site.ts';

export const CHAIN_M = 20.1168;
export const LINKS = 100;
export const LINK_M = CHAIN_M / LINKS; // 0.201168 m = 7.92 in
export type Lay = 'surface' | 'stepped';

/** Length the chain measures along the cut, M to A. */
export const lineLength = (lay: Lay) => (lay === 'stepped' ? LINE_H : LINE_SURFACE);
export function chainCount(lay: Lay) {
  const L = lineLength(lay);
  const n = Math.floor(L / CHAIN_M);
  const linksExact = (L - n * CHAIN_M) / LINK_M;
  const links = Math.round(linksExact);
  return { L, n, linksExact, links, measured: (n + links / LINKS) * CHAIN_M };
}
/** Horizontal distance from a reading on the slope: D·cos θ (for the surface lay). */
export const SLOPE_COS = (SITE.SHORE_X - SITE.CREST_X) / Math.hypot(SITE.SHORE_X - SITE.CREST_X, CREST_Z - SITE.QUAY_Z);

/**
 * Where chain i (0-based) starts and ends along the ground, as horizontal x (m).
 * Surface lay: each chain covers CHAIN_M of ground surface. Stepped: CHAIN_M of horizontal.
 */
export function chainEnds(lay: Lay, count: number) {
  const ends: number[] = [0];
  let x = 0;
  for (let i = 0; i < count; i++) {
    if (lay === 'stepped') x += CHAIN_M;
    else {
      // walk CHAIN_M along the surface profile (piecewise linear)
      let left = CHAIN_M;
      while (left > 1e-9) {
        const seg = x < SITE.FLAT_X ? SITE.FLAT_X : x < SITE.CREST_X ? SITE.CREST_X : SITE.SHORE_X + 1e6;
        const z0 = groundZ(x);
        const z1 = groundZ(Math.min(seg, SITE.SHORE_X));
        const run = Math.min(seg, SITE.SHORE_X) - x;
        const slope = run > 0 ? (z0 - z1) / run : 0;
        const per = Math.sqrt(1 + slope * slope); // surface m per horizontal m
        const can = seg > SITE.SHORE_X ? Infinity : run * per;
        if (left <= can) {
          x += left / per;
          left = 0;
        } else {
          x = Math.min(seg, SITE.SHORE_X);
          left -= can;
        }
      }
    }
    ends.push(x);
  }
  return ends;
}

export function chainScene(W: number, lay: Lay, laid: number) {
  const H = 230;
  const pad = 18;
  const VE = 3;
  const k = (W - 2 * pad) / (SITE.SHORE_X + 20);
  const kz = k * VE;
  const top = 40;
  const X = (x: number) => pad + x * k;
  const Y = (z: number) => top + (SITE.HILL_Z - z) * kz;
  const ground = [0, SITE.FLAT_X, SITE.CREST_X, SITE.SHORE_X].map((x) => ({ x: X(x), y: Y(groundZ(x)) }));
  const ends = chainEnds(lay, laid).filter((x) => x <= SITE.SHORE_X + 1e-6);
  const segs = ends.slice(1).map((x1, i) => {
    const x0 = ends[i];
    if (lay === 'surface') return { pts: chainPath(x0, x1), drop: null as null | { x: number; y0: number; y1: number }, i };
    // stepped: held level at the higher end's ground height, the lower end plumbed down
    const zHi = groundZ(x0);
    const zLo = groundZ(x1);
    const pts = [
      { x: X(x0), y: Y(zHi) },
      { x: X(x1), y: Y(zHi) },
    ];
    return { pts, drop: zHi - zLo > 0.05 ? { x: X(x1), y0: Y(zHi), y1: Y(zLo) } : null, i };
  });
  function chainPath(x0: number, x1: number) {
    const xs = [x0, ...[SITE.FLAT_X, SITE.CREST_X].filter((b) => b > x0 && b < x1), x1];
    return xs.map((x) => ({ x: X(x), y: Y(groundZ(x)) }));
  }
  const pins = ends.map((x) => ({ x: X(x), y: Y(groundZ(x)) }));
  const A = { x: X(SITE.SHORE_X), y: Y(SITE.QUAY_Z) };
  const M = { x: X(0), y: Y(SITE.HILL_Z) };
  return { W, H, pad, VE, k, X, Y, ground, segs, pins, A, M };
}

/** The last, partial chain, enlarged: links and brass tags, the end at A. */
export function linksScene(W: number, lay: Lay, read: number) {
  const c = chainCount(lay);
  const H = 120;
  const pad = 16;
  const kl = (W - 2 * pad) / LINKS; // px per link
  const y = 56;
  const endX = pad + c.linksExact * kl;
  const pointer = pad + read * kl;
  const tags = Array.from({ length: 11 }, (_, i) => ({ x: pad + i * 10 * kl, n: i * 10 }));
  return { ...c, W, H, pad, kl, y, endX, pointer, tags };
}
