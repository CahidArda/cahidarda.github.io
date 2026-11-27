/**
 * Penrose kites and darts by deflation of Robinson half-tiles, starting from a "sun" of five
 * kites. Half-kite (A, B, C): A the kite's 72 degree tip, B a side corner, C the 144 degree
 * corner, axis AC. Half-dart (A, B, C): A the dart's tip, B a side corner, C the reflex corner,
 * axis AC. The rules below were chosen by brute force over all ways to cut each half-tile into
 * smaller ones, keeping the only choice whose deflated patches pair every half into a whole
 * tile, meet edge to edge, and show exactly the seven legal vertex stars (sun, star, ace,
 * deuce, jack, queen, king).
 */
export type Pt = { x: number; y: number };
export type Half = { kind: 'K' | 'D'; a: Pt; b: Pt; c: Pt };
export type Tile = {
  kind: 'kite' | 'dart';
  pts: Pt[]; // A, B1, C, B2
  cx: number;
  cy: number;
  dir: number; // axis direction A -> C, in twentieths of a turn (0..19)
};

const PHI = (1 + Math.sqrt(5)) / 2;
const G = 1 / PHI; // 0.618...
const G2 = 1 - G; // 0.382...
const at = (a: Pt, b: Pt, t: number): Pt => ({
  x: a.x + (b.x - a.x) * t,
  y: a.y + (b.y - a.y) * t,
});

function deflate(h: Half): Half[] {
  const { a, b, c } = h;
  if (h.kind === 'K') {
    const p = at(a, c, G);
    const q = at(a, b, G2);
    return [
      { kind: 'D', a, b: p, c: q },
      { kind: 'K', a: b, b: c, c: p },
      { kind: 'K', a: b, b: q, c: p },
    ];
  }
  const p = at(a, b, G);
  return [
    { kind: 'K', a, b: p, c },
    { kind: 'D', a: b, b: c, c: p },
  ];
}

function sun(r: number): Half[] {
  const out: Half[] = [];
  const o = { x: 0, y: 0 };
  for (let k = 0; k < 5; k++) {
    const th = (k * 2 * Math.PI) / 5 - Math.PI / 2;
    const c = { x: r * Math.cos(th), y: r * Math.sin(th) };
    for (const s of [1, -1]) {
      const t = th + (s * Math.PI) / 5;
      out.push({ kind: 'K', a: o, b: { x: r * Math.cos(t), y: r * Math.sin(t) }, c });
    }
  }
  return out;
}

const key = (p: Pt) => `${Math.round(p.x * 1e4)},${Math.round(p.y * 1e4)}`;

/** Whole kites and darts after `n` deflations of a sun of radius r (unpaired halves dropped). */
export function penrose(n: number, r: number): Tile[] {
  let hs = sun(r);
  for (let i = 0; i < n; i++) hs = hs.flatMap(deflate);
  const byAxis = new Map<string, Half[]>();
  for (const h of hs) {
    const k = `${h.kind}|${key(h.a)}|${key(h.c)}`;
    const l = byAxis.get(k);
    if (l) l.push(h);
    else byAxis.set(k, [h]);
  }
  const tiles: Tile[] = [];
  for (const l of byAxis.values()) {
    if (l.length !== 2) continue;
    const [h1, h2] = l;
    const pts = [h1.a, h1.b, h1.c, h2.b];
    const cx = pts.reduce((s, p) => s + p.x, 0) / 4;
    const cy = pts.reduce((s, p) => s + p.y, 0) / 4;
    const ang = Math.atan2(h1.c.y - h1.a.y, h1.c.x - h1.a.x);
    const dir = ((Math.round(ang / (Math.PI / 10)) % 20) + 20) % 20;
    tiles.push({ kind: h1.kind === 'K' ? 'kite' : 'dart', pts, cx, cy, dir });
  }
  return tiles;
}

/**
 * How well a copy of the tiling shifted by (dx, dy) lines up inside the square view
 * |x|, |y| < half: of the copy's tiles that land in the view (and come from the covered part of
 * the patch, |c| < cover), the share that land exactly on a tile of the same kind pointing the
 * same way. `match[i]` says whether tile i of the copy lines up.
 */
export function overlap(tiles: Tile[], dx: number, dy: number, half: number, cover = 0.88) {
  const idx = new Set<string>();
  const q = (v: number) => Math.round(v * 500);
  for (const t of tiles) idx.add(`${t.kind}${t.dir}|${q(t.cx)},${q(t.cy)}`);
  let inside = 0;
  let hit = 0;
  const match: boolean[] = [];
  for (const t of tiles) {
    const x = t.cx + dx;
    const y = t.cy + dy;
    const ok = Math.abs(x) < half && Math.abs(y) < half && Math.hypot(t.cx, t.cy) < cover;
    let m = false;
    if (ok) {
      inside++;
      m = idx.has(`${t.kind}${t.dir}|${q(x)},${q(y)}`);
      if (m) hit++;
    }
    match.push(m);
  }
  return { share: inside ? hit / inside : 0, inside, hit, match };
}

/**
 * Candidate shifts: move the tile nearest the centre onto every other tile of the same kind
 * and direction within `reach`, and score each over the view. Best first.
 */
export function bestShifts(tiles: Tile[], half: number, reach: number) {
  let c0 = tiles[0];
  for (const t of tiles) if (Math.hypot(t.cx, t.cy) < Math.hypot(c0.cx, c0.cy)) c0 = t;
  const cands: { dx: number; dy: number; share: number }[] = [];
  for (const t of tiles) {
    if (t === c0 || t.kind !== c0.kind || t.dir !== c0.dir) continue;
    const dx = t.cx - c0.cx;
    const dy = t.cy - c0.cy;
    if (Math.hypot(dx, dy) > reach) continue;
    cands.push({ dx, dy, share: overlap(tiles, dx, dy, half).share });
  }
  return cands.sort((p, q) => q.share - p.share);
}
