/**
 * The stable reference view of the latitude post, pure (no React): a cut through the Earth along
 * the observer's meridian. Every procedure plate draws it with `latRef(spec, W)` and the SAME
 * layout (the Earth's size and place depend on the width only), so the reader sees each
 * instrument attacking the same angle.
 *
 * Frame. Screen coordinates, y DOWN, viewBox units = CSS px of the measured width W. Angles in
 * degrees, 0° = +x, counter-clockwise on screen (the `polar` convention of ../shared). North up,
 * the equator horizontal, the observer on the right-hand rim at latitude φ.
 * World (fixed for a plate's state, never moved by an instrument control):
 *   sun   parallel rays from the right at the sun's declination δ (screen direction δ),
 *   star  parallel rays from Polaris, nearly along the axis (screen direction 90° − pole offset).
 * Quantities: h (sky) the body's altitude above your horizon, at you; φ (red) your latitude at
 * the centre; δ (brass) the sun's declination at the centre. Two-city mode (Eratosthenes) adds a
 * second observer and draws θ, the angle between the two places, at the centre and at you.
 * An optional measured sight (red, dashed) shows the reader's altitude against the true one.
 */

const RAD = Math.PI / 180;

/* ───────────────────────────── layout helpers (shared by every sky plate) ───────────────────────────── */

export type Pt = { x: number; y: number };
export type Seg = { id: string; a: Pt; b: Pt };
export type Label = {
  id: string;
  text: string;
  x: number;
  y: number;
  anchor: 'start' | 'middle' | 'end';
  size: number;
  /** false if no candidate position was free (the sweep fails on it). */
  placed: boolean;
};
export type Box = { x0: number; y0: number; x1: number; y1: number };

export const P = (cx: number, cy: number, r: number, deg: number): Pt => ({
  x: cx + r * Math.cos(deg * RAD),
  y: cy - r * Math.sin(deg * RAD),
});
/** Text box estimate (central baseline): serif ≈ 0.55 em per glyph, a little generous. */
export function labelBox(l: Pick<Label, 'x' | 'y' | 'text' | 'size' | 'anchor'>): Box {
  const w = [...l.text].length * l.size * 0.56;
  const x0 = l.anchor === 'start' ? l.x : l.anchor === 'middle' ? l.x - w / 2 : l.x - w;
  return { x0, y0: l.y - l.size * 0.55, x1: x0 + w, y1: l.y + l.size * 0.55 };
}
export const boxesOverlap = (a: Box, b: Box, pad = 1) =>
  a.x0 < b.x1 + pad && b.x0 < a.x1 + pad && a.y0 < b.y1 + pad && b.y0 < a.y1 + pad;
export function segHitsBox(a: Pt, b: Pt, bx: Box, pad = 0) {
  const n = Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / 0.5) + 1;
  for (let i = 0; i <= n; i++) {
    const x = a.x + ((b.x - a.x) * i) / n;
    const y = a.y + ((b.y - a.y) * i) / n;
    if (x > bx.x0 - pad && x < bx.x1 + pad && y > bx.y0 - pad && y < bx.y1 + pad) return true;
  }
  return false;
}
const inBounds = (b: Box, W: number, H: number, m = 2) =>
  b.x0 >= m && b.y0 >= m && b.x1 <= W - m && b.y1 <= H - m;

/** Greedy label placement: first candidate inside the drawing, off every line, off every label. */
export function placer(W: number, H: number, segs: Seg[], arcs: Pt[][] = [], soft: string[] = [], gap = 6) {
  const placed: Label[] = [];
  const free = (l: Omit<Label, 'placed'>) => {
    const b = labelBox(l);
    if (!inBounds(b, W, H)) return false;
    // small word labels may cross the faint decorative rays (`soft`): their paper halo cuts them
    // the reader's dashed sight line never blocks a label (the paper halo cuts it)
    const hard = (l.size >= 14 ? segs : segs.filter((s) => !soft.includes(s.id))).filter((s) => s.id !== 'sight');
    if (hard.some((s) => segHitsBox(s.a, s.b, b, 1))) return false;
    if (arcs.some((pts) => pts.some((p, i) => i > 0 && segHitsBox(pts[i - 1], p, b, 0.5))))
      return false;
    return !placed.some((o) => boxesOverlap(labelBox(o), b, gap));
  };
  const put = (cands: Omit<Label, 'placed'>[]) => {
    const hit = cands.find(free);
    const l = { ...(hit ?? cands[0]), placed: !!hit };
    placed.push(l);
    return l;
  };
  return { put, placed };
}
/** Points along an arc (for label/arc collision checks), same convention as arcPath. */
export function arcPts(cx: number, cy: number, r: number, from: number, to: number, n = 24) {
  return Array.from({ length: n + 1 }, (_, i) => P(cx, cy, r, from + ((to - from) * i) / n));
}

/** One drawing size for both views of a step plate, from the measured column width. */
export const viewH = (W: number) => Math.round(Math.max(290, Math.min(400, W * 0.86)));

/* ───────────────────────────── the reference ───────────────────────────── */

export type RefSpec = {
  /** Your latitude, degrees (north +). */
  phi: number;
  /** Where the light comes from. Sun: its declination. Star: Polaris at altitude `alt` (deg) above
   *  your north horizon (φ plus or minus its small circle round the pole). */
  src: { kind: 'sun'; dec: number } | { kind: 'star'; alt: number };
  /** Draw h (false on days the body is not measurable). */
  visible?: boolean;
  /** A second observer (Eratosthenes' Syene), latitude in degrees. */
  second?: { phi: number; name: string };
  /** Name of the observer's label (default "you"). */
  you?: string;
  /** The reader's measured altitude, degrees (drawn as a red dashed sight line), or null. */
  sight?: number | null;
  /** Label of the light source (default "sunlight" / "Polaris"). */
  srcName?: string;
};

export type RefScene = ReturnType<typeof latRef>;

export function latRef(spec: RefSpec, W: number) {
  const H = viewH(W);
  const phi = spec.phi;
  const star = spec.src.kind === 'star';
  const two = !!spec.second;
  const dec = spec.src.kind === 'sun' ? spec.src.dec : 0;
  // screen direction from you towards the light's source
  const dir = spec.src.kind === 'sun' ? spec.src.dec : phi + 90 - spec.src.alt;
  const bearing: 'S' | 'N' = star ? 'N' : phi >= dec ? 'S' : 'N';
  const visible = spec.visible ?? true;
  // the zenith line (1.32 Re from the centre) must fit above and below, with the sun's side open
  const Re = Math.min((H / 2 - 12) / 1.32, (W - 40) / 2.55);
  const cx = Re * 1.32 + 10;
  const cy = H / 2;
  const zLen = Re * 0.32;
  const L = Re * 0.62;
  const obs = P(cx, cy, Re, phi);
  const zen = P(cx, cy, Re + zLen, phi);
  const horDir = bearing === 'S' ? phi - 90 : phi + 90; // horizon towards the body
  const horSun = P(obs.x, obs.y, L, horDir);
  const horFar = P(obs.x, obs.y, L * 0.7, horDir + 180);
  const ux = Math.cos(dir * RAD);
  const uy = -Math.sin(dir * RAD);
  // a ray back towards the source from p, clipped to the drawing (4 units in)
  const toSrc = (p: Pt) => {
    let t = Infinity;
    if (ux > 1e-9) t = Math.min(t, (W - 4 - p.x) / ux);
    if (ux < -1e-9) t = Math.min(t, (4 - p.x) / ux);
    if (uy < -1e-9) t = Math.min(t, (4 - p.y) / uy);
    if (uy > 1e-9) t = Math.min(t, (H - 4 - p.y) / uy);
    return { x: p.x + ux * t, y: p.y + uy * t };
  };
  const sunFrom = toSrc(obs);
  // other parallel rays across the beam, ending where they meet the Earth
  const rays = (star ? [-0.55, 0.55] : [-0.62, 0.62]).map((o) => {
    const n = { x: Math.sin(dir * RAD), y: Math.cos(dir * RAD) }; // screen normal of the beam
    const off = o * Re;
    const along = Math.sqrt(Math.max(Re * Re - off * off, 0));
    const hit = { x: cx + n.x * off + ux * along, y: cy + n.y * off + uy * along };
    return { a: toSrc(hit), b: hit, o };
  });
  const sub = P(cx, cy, Re, dec); // where the sun is overhead
  const phiArcR = Re * 0.4;
  const decArcR = Re * 0.68;
  const hArcR = L * 0.55;
  // h arc: from the horizon (body side) to the body, the short way
  const hFrom = bearing === 'S' ? horDir : dir;
  const hTo = bearing === 'S' ? dir : horDir;
  const drawnH = Math.abs(hTo - hFrom);
  const drawnPhi = Math.atan2(-(obs.y - cy), obs.x - cx) / RAD;
  const drawnDec = Math.atan2(-(sunFrom.y - obs.y), sunFrom.x - obs.x) / RAD;

  // two cities: Syene, its zenith, its ray; θ at the centre and at you (between zenith and sun)
  const obs2 = two ? P(cx, cy, Re, spec.second!.phi) : null;
  const zen2 = two ? P(cx, cy, Re + zLen, spec.second!.phi) : null;
  const sun2 = obs2 ? toSrc(obs2) : null;
  const thetaArcR = Re * 0.86;
  const zArcR = zLen * 0.95;

  // the reader's sight line at you (red, dashed): the measured altitude from the same horizon
  const sightDir =
    spec.sight == null ? null : bearing === 'S' ? horDir + spec.sight : horDir - spec.sight;
  const sightEnd = sightDir == null ? null : P(obs.x, obs.y, L * 1.05, sightDir);

  const segs: Seg[] = [
    { id: 'equator', a: { x: cx - Re - 8, y: cy }, b: { x: cx + Re + 14, y: cy } },
    { id: 'axis', a: { x: cx, y: cy - Re - 8 }, b: { x: cx, y: cy + Re + 8 } },
    { id: 'zenith', a: { x: cx, y: cy }, b: zen },
    ...(star || two ? [] : [{ id: 'decline', a: { x: cx, y: cy }, b: sub }]),
    ...(two ? [] : [{ id: 'horizon', a: horFar, b: horSun }]),
    { id: 'sunray', a: sunFrom, b: obs },
    ...rays.map((r, i) => ({ id: `ray${i}`, a: r.a, b: r.b })),
    ...(two ? [{ id: 'zenith2', a: { x: cx, y: cy }, b: zen2! }, { id: 'sunray2', a: sun2!, b: obs2! }] : []),
    ...(sightEnd ? [{ id: 'sight', a: obs, b: sightEnd }] : []),
  ];
  const arcs = [
    ...(two ? [] : [arcPts(cx, cy, phiArcR, 0, phi)]),
    ...(star || two ? [] : [arcPts(cx, cy, decArcR, 0, dec)]),
    ...(visible && !two ? [arcPts(obs.x, obs.y, hArcR, hFrom, hTo)] : []),
    ...(two ? [arcPts(cx, cy, thetaArcR, spec.second!.phi, phi), arcPts(obs.x, obs.y, zArcR, dir, phi)] : []),
    arcPts(cx, cy, Re, 0, 360, 96),
  ];
  const pl = placer(W, H, segs, arcs, ['ray0', 'ray1']);
  const S = 15;
  const s11 = 11;
  const lab = (id: string, text: string, p: Pt, size: number, anchor: Label['anchor'] = 'middle') => ({ id, text, x: p.x, y: p.y, size, anchor });
  /** Candidates on rings around p: preferred angles first, then every 20°. Anchor by side. */
  const ring = (id: string, text: string, p: Pt, size: number, rs: number[], pref: number[]) => {
    const out: ReturnType<typeof lab>[] = [];
    const angs = [...pref, ...Array.from({ length: 18 }, (_, i) => i * 20)];
    for (const r of rs)
      for (const a of angs) {
        const q = P(p.x, p.y, r, a);
        const c = Math.cos(a * RAD);
        out.push(lab(id, text, q, size, size >= S ? 'middle' : c > 0.35 ? 'start' : c < -0.35 ? 'end' : 'middle'));
      }
    return out;
  };
  /** Angle-label candidates along an arc about (x0, y0): inside, on and outside the arc. */
  const along = (id: string, text: string, x0: number, y0: number, r: number, from: number, to: number) => {
    const out: ReturnType<typeof lab>[] = [];
    const span = to - from;
    const fr = [0.5, 0.35, 0.65, 0.2, 0.8];
    for (const dr of [11, -11, 20, -18, 30, -28, -40])
      for (const f of fr) out.push(lab(id, text, P(x0, y0, r + dr, from + span * f), S));
    // a tiny angle: just outside its ends
    for (const dr of [0, 10, -10]) for (const e of [-12, 12, -20, 20]) out.push(lab(id, text, P(x0, y0, r + dr, (e < 0 ? Math.min(from, to) : Math.max(from, to)) + e), S));
    return out;
  };
  /** A centre angle measured from the equator: keep its label on the value's side of it. */
  const sameSide = (cands: ReturnType<typeof lab>[], v: number, r: number) => {
    const sg = v >= 0 ? -1 : 1;
    const extra: ReturnType<typeof lab>[] = [];
    for (const dy of [10, 14, 20, 26, 32]) for (const dx of [0, 8, -8, 16, -16, 26, -26, 36, -36, 48]) extra.push(lab(cands[0].id, cands[0].text, { x: cx + r + dx, y: cy + sg * dy }, S));
    return [...cands, ...extra].filter((c) => (v >= 0 ? c.y <= cy - 4 : c.y >= cy + 4));
  };
  // most constrained first
  if (two) {
    const p2 = spec.second!.phi;
    pl.put(along('theta', 'θ', cx, cy, thetaArcR, p2, phi).filter((c) => Math.hypot(c.x - cx, c.y - cy) < Re - 8));
    pl.put(along('z', 'θ', obs.x, obs.y, zArcR, dir, phi));
    pl.put(ring('you', spec.you ?? 'you', obs, s11, [16, 24, 32, 42], [phi + 150, phi + 180, phi + 210, phi + 120]));
    pl.put(ring('second', spec.second!.name, obs2!, s11, [16, 24, 32, 42], [p2 - 150, p2 - 180, p2 - 120, p2 - 90]));
  } else {
    if (visible) pl.put(along('h', 'h', obs.x, obs.y, hArcR, hFrom, hTo));
    if (!star) pl.put(sameSide(along('dec', 'δ', cx, cy, decArcR, 0, dec), dec, decArcR));
    pl.put(sameSide(along('phi', 'φ', cx, cy, phiArcR, 0, phi), phi, phiArcR));
    pl.put(ring('you', spec.you ?? 'you', obs, s11, [14, 22, 30], [phi + 180, phi + 150, phi + 210]));
  }
  pl.put(ring('zenith', 'zenith', zen, s11, [8, 14, 22, 32], [phi, phi + 30, phi - 30]));
  if (!two) pl.put(ring('horizon', 'horizon', horSun, s11, [8, 14, 22, 32], [horDir - 60, horDir + 60, horDir]));
  pl.put([
    lab('pole', 'N pole', { x: cx + 5, y: cy - Re - 10 }, s11, 'start'),
    lab('pole', 'N pole', { x: cx - 5, y: cy - Re - 10 }, s11, 'end'),
    ...ring('pole', 'N pole', { x: cx, y: cy - Re - 4 }, s11, [10, 18, 28], [150, 30, 200, 340]),
  ]);
  pl.put([
    lab('equator', 'equator', { x: cx - Re + 8, y: cy - 8 }, s11, 'start'),
    lab('equator', 'equator', { x: cx - Re + 8, y: cy + 9 }, s11, 'start'),
    lab('equator', 'equator', { x: cx + Re + 4, y: cy + 9 }, s11, 'start'),
    lab('equator', 'equator', { x: cx + Re + 4, y: cy - 8 }, s11, 'start'),
    lab('equator', 'equator', { x: cx - 6, y: cy + 9 }, s11, 'end'),
  ]);
  const srcText = spec.srcName ?? (star ? 'Polaris' : 'sunlight');
  pl.put(
    star
      ? [
          lab('sun', srcText, { x: W - 6, y: 12 }, s11, 'end'),
          lab('sun', srcText, { x: cx + Re * 0.75, y: 12 }, s11, 'start'),
          lab('sun', srcText, { x: 8, y: 12 }, s11, 'start'),
          ...[0.3, 0.6, 0.9].map((f) => lab('sun', srcText, { x: 8 + (W - 16) * f, y: 12 }, s11, 'middle')),
        ]
      : [
          lab('sun', srcText, { x: W - 6, y: 12 }, s11, 'end'),
          lab('sun', srcText, { x: W - 6, y: H - 10 }, s11, 'end'),
          lab('sun', srcText, { x: W - 6, y: cy - Re - 10 }, s11, 'end'),
          lab('sun', srcText, { x: W - 6, y: cy + Re + 10 }, s11, 'end'),
          ...[0.25, 0.75, 0.5, 0.1, 0.9].map((f) => lab('sun', srcText, { x: W - 6, y: 12 + (H - 22) * f }, s11, 'end')),
        ],
  );
  const labels = Object.fromEntries(pl.placed.map((l) => [l.id, l])) as Record<string, Label>;
  return {
    W, H, Re, cx, cy, zLen, L, obs, zen, horDir, horSun, horFar, sunFrom, rays, sub,
    phiArcR, decArcR, hArcR, hFrom, hTo, drawnH, drawnPhi, drawnDec, segs, labels, phi, dec,
    dir, bearing, star, two, visible, obs2, zen2, sun2, thetaArcR, zArcR, sightDir, sightEnd,
    arcs,
  };
}
