/**
 * Earth plates: pure geometry. Every drawn coordinate that carries meaning is computed here,
 * from the plate's parameters, so the drawings and the sweep script (scripts/earth-sweep.ts)
 * use the same numbers. No React, no DOM.
 *
 * Conventions: world angles in radians, maths convention (0 = +x, counter-clockwise, y up).
 * Screen coordinates are SVG viewBox units (y down).
 */

export const RAD = Math.PI / 180;
export const deg = (r: number) => r / RAD;

export type Pt = { x: number; y: number };
/** An axis-aligned text box, for collision and bounds checks. */
export type Box = { x0: number; y0: number; x1: number; y1: number; name: string };

/** Rough box of a text label. `w` = average glyph width as a fraction of the font size. */
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
/** Same, for text rotated by ±90° about its anchor point. */
export function vTextBox(name: string, x: number, y: number, text: string, size: number, w = 0.56): Box {
  const len = [...text].length * size * w;
  return { name, x0: x - size * 0.5, y0: y - len / 2, x1: x + size * 0.5, y1: y + len / 2 };
}
export const overlap = (a: Box, b: Box, pad = 0) =>
  a.x0 < b.x1 + pad && b.x0 < a.x1 + pad && a.y0 < b.y1 + pad && b.y0 < a.y1 + pad;
/** Does segment p→q pass through box b grown by `pad`? Sampled every half unit. */
export function segHitsBox(p: Pt, q: Pt, b: Box, pad = 0) {
  const n = Math.ceil(Math.hypot(q.x - p.x, q.y - p.y) / 0.5) + 1;
  for (let i = 0; i <= n; i++) {
    const x = p.x + ((q.x - p.x) * i) / n;
    const y = p.y + ((q.y - p.y) * i) / n;
    if (x > b.x0 - pad && x < b.x1 + pad && y > b.y0 - pad && y < b.y1 + pad) return true;
  }
  return false;
}

/* ─────────────────────────── triangulation ─────────────────────────── */

/** Height from two elevation angles α (far station) and β (near), baseline b between them. */
export const heightFrom = (b: number, a: number, be: number) =>
  (b * Math.sin(a) * Math.sin(be)) / Math.sin(be - a);
/** Perpendicular distance of C from the baseline AB, angles α at A and β at B. */
export const widthFrom = (b: number, a: number, be: number) =>
  (b * Math.sin(a) * Math.sin(be)) / Math.sin(a + be);

/** Where the ray from P at angle tp meets the ray from Q at angle tq (both forward), or null. */
export function rayHit(P: Pt, tp: number, Q: Pt, tq: number): Pt | null {
  const d1 = { x: Math.cos(tp), y: Math.sin(tp) };
  const d2 = { x: Math.cos(tq), y: Math.sin(tq) };
  const den = d1.x * d2.y - d1.y * d2.x;
  if (Math.abs(den) < 1e-12) return null;
  const wx = Q.x - P.x;
  const wy = Q.y - P.y;
  const s = (wx * d2.y - wy * d2.x) / den;
  const u = (wx * d1.y - wy * d1.x) / den;
  if (s <= 0 || u <= 0) return null;
  return { x: P.x + s * d1.x, y: P.y + s * d1.y };
}

/**
 * The region where the two sight lines can cross when each angle may be off by ±e: the
 * intersection of two wedges, a quadrilateral whose corners are the four extreme crossings.
 * Its vertical extent [lo, hi] is the range of the measured height (or width). `bounded` is
 * false when some pair of perturbed lines no longer meets above the ground: then there is no
 * upper limit at all.
 */
export function errorRegion(P: Pt, tp: number, Q: Pt, tq: number, e: number) {
  const order: [number, number][] = [
    [-1, -1],
    [-1, 1],
    [1, 1],
    [1, -1],
  ];
  const hits = order.map(([s1, s2]) => rayHit(P, tp + s1 * e, Q, tq + s2 * e));
  const ok = hits.filter((h): h is Pt => h !== null && h.y > 0);
  const bounded = ok.length === 4;
  const ys = ok.map((h) => h.y);
  const lo = ys.length ? Math.min(...ys) : 0;
  const hi = bounded ? Math.max(...ys) : Infinity;
  return { corners: bounded ? (hits as Pt[]) : null, lo, hi, bounded };
}

export const H_TRUE = 76; // Süleymaniye minaret, m (round figure)
export const W_TRUE = 700; // Bosphorus at its narrowest, m (round figure)
export const MIN_Y_MAX = 115; // the minaret drawing shows heights up to this many metres

export type Seg = { x1: number; y1: number; x2: number; y2: number };

/** Every meaningful coordinate of the minaret drawing. */
export function minaretLayout(W: number, dB: number, b: number, epsRad: number) {
  const narrow = W < 520;
  const right = W - (narrow ? 44 : 78); // the minaret's foot
  const kx = (right - 24) / 300; // A can be at most 300 m from the foot
  const ky = (narrow ? 200 : 300) / MIN_Y_MAX;
  const k = Math.min(kx, ky); // ONE scale, px per metre, both axes
  const top = 12; // y of MIN_Y_MAX
  const g = top + MIN_Y_MAX * k; // ground line
  const Hs = Math.round(g + 66);
  const fx = right;
  const dA = dB + b;
  const ax = fx - dA * k;
  const bx = fx - dB * k;
  const al = Math.atan(H_TRUE / dA);
  const be = Math.atan(H_TRUE / dB);
  const h = heightFrom(b, al, be);
  const tip = { x: fx, y: g - h * k };
  const toS = (p: Pt) => ({ x: fx + p.x * k, y: g - p.y * k });
  const reg = errorRegion({ x: -dA, y: 0 }, al, { x: -dB, y: 0 }, be, epsRad);
  const quad = reg.corners ? reg.corners.map(toS) : null;
  // the range bracket, beside the minaret, from lo to hi (clipped at the top of the drawing)
  const yTopLimit = 4;
  const loY = g - reg.lo * k;
  const hiYraw = reg.bounded ? g - reg.hi * k : -Infinity;
  const clippedTop = hiYraw < yTopLimit;
  const hiY = Math.max(yTopLimit, hiYraw);
  const dimOff = narrow ? 14 : 22; // h dimension, right of the minaret
  const brOff = narrow ? 27 : 42; // range bracket, further right
  const range = { x: fx + brOff, y1: loY, y2: hiY, clippedTop };
  const rangeText = !reg.bounded
    ? `${reg.lo.toFixed(0)} m to no limit`
    : `${reg.lo.toFixed(1)} to ${reg.hi.toFixed(1)} m`;
  const rlSize = narrow ? 10 : 11;
  const rlHalf = ([...rangeText].length * rlSize * 0.52) / 2 + 3;
  // centred on the bracket, unless that would push it out of the drawing
  const rangeLabel = {
    x: fx + brOff + (narrow ? 9 : 12),
    y: Math.min(Math.max((loY + hiY) / 2, rlHalf), g - rlHalf),
    text: rangeText,
    size: rlSize,
  };
  // angle arcs. The minaret's half-width (balconies) bounds what fits in front of it.
  const span = bx - ax;
  const mw = Math.max(4, H_TRUE * k * 0.07) * 1.05 + 2;
  const room = fx - mw - bx; // free ground between B and the minaret
  const rB = Math.max(10, Math.min(narrow ? 22 : 28, room * 0.45));
  // α's arc stays left of B when there is room; otherwise it reaches round β's arc
  const rA = span - 6 >= (narrow ? 26 : 32) ? Math.min(narrow ? 40 : 48, span - 6) : span + rB + 10;
  // Angle labels: try positions in order, take the first that is clear of both sight lines,
  // the minaret, the ground and the other label. The last resort is open sky above A's sight
  // line, with a leader back to the arc.
  const tipS = { x: fx, y: g - H_TRUE * k };
  const minBox: Box = { name: 'minaret', x0: fx - mw, x1: fx + mw, y0: tipS.y, y1: g };
  const lines: [Pt, Pt][] = [
    [{ x: ax, y: g }, tipS],
    [{ x: bx, y: g }, tipS],
  ];
  const fits = (b: Box, others: Box[]) =>
    b.y1 < g - 1 &&
    b.x0 > 2 &&
    b.y0 > 2 &&
    !lines.some(([p, q]) => segHitsBox(p, q, b, 1.5)) &&
    !overlap(b, minBox, 2) &&
    !others.some((o) => overlap(o, b, 2));
  const place = (cands: Pt[], fallbacks: Pt[], others: Box[], name: string) => {
    for (const c of cands) {
      const b = textBox(name, c.x, c.y, name, 14, 'middle', 'central');
      if (fits(b, others)) return { ...c, leader: false };
    }
    for (const c of fallbacks) {
      const b = textBox(name, c.x, c.y, name, 14, 'middle', 'central');
      if (fits(b, others)) return { ...c, leader: true };
    }
    return { ...fallbacks[0], leader: true };
  };
  const along = (cx: number, r: number, ang: number) => ({ x: cx + r * Math.cos(ang), y: g - r * Math.sin(ang) });
  const aboveA = (x: number, dy: number) => ({ x, y: g - (x - ax) * Math.tan(al) - dy });
  const alEnd = along(ax, rA, al);
  const alLabel = place(
    [
      { x: alEnd.x - 11 * Math.sin(al), y: alEnd.y - 11 * Math.cos(al) },
      along(ax, rA + 10, al / 2),
      along(ax, rA - 12, al / 2),
    ],
    [aboveA(ax + rA, 26), aboveA(ax + rA, 40)],
    [],
    'α',
  );
  const alBox = textBox('α', alLabel.x, alLabel.y, 'α', 14, 'middle', 'central');
  const bis = be / 2;
  const beCands: Pt[] = [];
  for (const rr of [rB + 9, rB + 4, rB - 6, rB + 16, rB + 24]) beCands.push(along(bx, rr, bis));
  const beEnd = along(bx, rB, be);
  beCands.push({ x: beEnd.x - 10 * Math.sin(be), y: beEnd.y - 10 * Math.cos(be) });
  const beLabel = place(
    beCands,
    [aboveA(bx - 4, 30), aboveA(bx + 14, 30), aboveA(bx - 4, 46), aboveA(bx + 24, 44)],
    [alBox],
    'β',
  );
  const beArcMid = along(bx, rB, bis);
  const alArcMid = along(ax, rA, al / 2);
  const close = span < 34;
  const nameA = { x: close ? ax - 3 : ax, y: g + 22, anchor: close ? ('end' as const) : ('middle' as const) };
  const nameB = { x: close ? bx + 3 : bx, y: g + 22, anchor: close ? ('start' as const) : ('middle' as const) };
  const baseDim = { x1: ax, x2: bx, y: g + 40 };
  const baseLabel = { x: (ax + bx) / 2, y: g + 57, text: `b = ${b} m` };
  const hLabelText = `${h.toFixed(1)} m`;
  const boxes: Box[] = [
    textBox('α', alLabel.x, alLabel.y, 'α', 14, 'middle', 'central'),
    textBox('β', beLabel.x, beLabel.y, 'β', 14, 'middle', 'central'),
    textBox('A', nameA.x, nameA.y, 'A', 13, nameA.anchor),
    textBox('B', nameB.x, nameB.y, 'B', 13, nameB.anchor),
    textBox('b label', baseLabel.x, baseLabel.y, baseLabel.text, 11, 'middle', 'alphabetic', 0.62),
    vTextBox('h label', fx + dimOff, (g + tip.y) / 2, hLabelText, 11, 0.62),
    vTextBox('range label', rangeLabel.x, rangeLabel.y, rangeText, rangeLabel.size, 0.52),
  ];
  return {
    W,
    Hs,
    narrow,
    k,
    g,
    fx,
    ax,
    bx,
    al,
    be,
    h,
    tip,
    reg,
    quad,
    range,
    rangeLabel,
    rA,
    rB,
    alLabel,
    beLabel,
    alArcMid,
    beArcMid,
    nameA,
    nameB,
    baseDim,
    baseLabel,
    dimOff,
    hLabelText,
    sightA: { x1: ax, y1: g, x2: tip.x, y2: tip.y } as Seg,
    sightB: { x1: bx, y1: g, x2: tip.x, y2: tip.y } as Seg,
    boxes,
  };
}

/** Every meaningful coordinate of the Bosphorus (plan view) drawing. */
export function straitLayout(W: number, b: number, epsRad: number) {
  const narrow = W < 520;
  const Hs = narrow ? 330 : 350;
  const nearY = Hs - 62;
  const k = (Hs - 150) / W_TRUE; // one scale, px per metre
  const farY = nearY - W_TRUE * k;
  const ax = narrow ? 60 : Math.round(W * 0.26);
  const bx = ax + b * k;
  const al = Math.PI / 2;
  const be = Math.atan(W_TRUE / b);
  const w = widthFrom(b, al, be);
  const C = { x: ax + w * Math.cos(al) * k, y: nearY - w * k };
  const toS = (p: Pt) => ({ x: ax + p.x * k, y: nearY - p.y * k });
  const reg = errorRegion({ x: 0, y: 0 }, al, { x: b, y: 0 }, Math.PI - be, epsRad);
  const quad = reg.corners ? reg.corners.map(toS) : null;
  const yTopLimit = 4;
  const loY = nearY - reg.lo * k;
  const hiYraw = reg.bounded ? nearY - reg.hi * k : -Infinity;
  const clippedTop = hiYraw < yTopLimit;
  const hiY = Math.max(yTopLimit, hiYraw);
  const dimOff = narrow ? 18 : 24; // w dimension, left of A's line
  const brOff = narrow ? 34 : 46; // range bracket, further left
  const range = { x: ax - brOff, y1: loY, y2: hiY, clippedTop };
  const rangeText = !reg.bounded
    ? `${reg.lo.toFixed(0)} m to no limit`
    : `${reg.lo.toFixed(0)} to ${reg.hi.toFixed(0)} m`;
  const rangeLabel = { x: ax - brOff - (narrow ? 9 : 12), y: (loY + hiY) / 2, text: rangeText, size: narrow ? 10 : 11 };
  const rB = narrow ? 24 : 28;
  const beLabel = {
    x: bx + (rB + 9) * Math.cos(Math.PI - be / 2),
    y: nearY - (rB + 9) * Math.sin(Math.PI - be / 2),
  };
  const wText = `${w.toFixed(0)} m`;
  const baseLabel = { x: (ax + bx) / 2, y: nearY + 44, text: `b = ${b} m` };
  const boxes: Box[] = [
    textBox('β', beLabel.x, beLabel.y, 'β', 14, 'middle', 'central'),
    textBox('A', ax - 8, nearY + 4, 'A', 13, 'end'),
    textBox('B', bx + 8, nearY + 4, 'B', 13, 'start'),
    textBox('C', ax + 8, C.y - 8, 'C', 13, 'start'),
    textBox('b label', baseLabel.x, baseLabel.y, baseLabel.text, 11, 'middle', 'alphabetic', 0.62),
    vTextBox('w label', ax - dimOff, (nearY + C.y) / 2, wText, 11, 0.62),
    vTextBox('range label', rangeLabel.x, rangeLabel.y, rangeText, rangeLabel.size, 0.52),
    textBox('Europe', W - 10, farY - 12, 'Europe', 12, 'end'),
    textBox('Asia', W - 10, nearY + 24, 'Asia', 12, 'end'),
  ];
  return {
    W,
    Hs,
    narrow,
    k,
    nearY,
    farY,
    ax,
    bx,
    al,
    be,
    w,
    C,
    reg,
    quad,
    range,
    rangeLabel,
    rB,
    beLabel,
    dimOff,
    wText,
    baseLabel,
    boxes,
  };
}

/* ─────────────────────────── A-frame level ─────────────────────────── */

export const AFRAME_H_CM = 50; // the drawn frame's apex-to-crossbar drop stands for 50 cm

export function aframeLayout(W: number, thetaDeg: number) {
  const narrow = W < 520;
  const s = Math.min(1, W / 470);
  const Hs = narrow ? 300 : 330;
  const cx = W / 2;
  const gy = Hs - 72;
  const Hf = 215 * s + (narrow ? 25 : 0);
  const L = 118 * s;
  const h = Hf * 0.55; // apex to crossbar, px
  const t = thetaDeg * RAD;
  /** frame coordinates (x along the feet, y up the frame) to screen */
  const P = (x: number, y: number) => ({
    x: cx + x * Math.cos(t) - y * Math.sin(t),
    y: gy - (x * Math.sin(t) + y * Math.cos(t)),
  });
  const beamHalf = Math.min(W * 0.44, 270);
  const beam = [P(-beamHalf, 0), P(beamHalf, 0), P(beamHalf, -20), P(-beamHalf, -20)];
  const apex = P(0, Hf);
  const cb = h / Hf;
  const cl = P(-L * cb, Hf - h);
  const cr = P(L * cb, Hf - h);
  const mark = P(0, Hf - h);
  // Where a truly vertical string from the apex crosses the crossbar. Going straight down
  // from the apex, in frame coordinates, runs along (−sin t, −cos t): the crossing sits at
  // frame x = −h·tan t, on the DOWNHILL side.
  const xOff = -h * Math.tan(t);
  const hit = P(xOff, Hf - h);
  // x dimension: from the centre mark to the string, drawn just below the crossbar (toward the
  // feet). DimLine's offset is along the screen-left normal of mark→hit, which points down the
  // frame when the hit is to the left (θ > 0) and up when it is to the right.
  const dimOffset = thetaDeg > 0 ? 22 : -22;
  // x label at the outer end of the x dimension, beyond the string (never on it)
  const xLabelPt = P(xOff + Math.sign(xOff || 1) * 14, Hf - h - 22);
  const hDim = { a: P(-L * 0.75 - 26, Hf), b: P(-L * 0.75 - 26, Hf - h) };
  const hLabel = { x: (hDim.a.x + hDim.b.x) / 2 - 9, y: (hDim.a.y + hDim.b.y) / 2 };
  // θ arc about the beam's centre, between true level and the top of the beam, just outside
  // the feet, on the side where the beam dips below the level line (open air above it there).
  const dipLeft = thetaDeg >= 0;
  const thetaR = L + 16;
  const thetaFrom = dipLeft ? 180 : 0;
  const thetaTo = dipLeft ? 180 + thetaDeg : thetaDeg;
  const thetaLabel = {
    x: dipLeft ? cx - thetaR - 4 : cx + thetaR + 4,
    y: gy - 4,
    anchor: dipLeft ? ('end' as const) : ('start' as const),
  };
  const levelText = narrow ? 'level' : 'true level';
  const levelLabel = {
    x: dipLeft ? cx - beamHalf - 6 : cx + beamHalf + 6,
    y: gy - 7,
    anchor: dipLeft ? ('start' as const) : ('end' as const),
  };
  const status = { x: cx, y: Hs - 18 };
  const level = Math.abs(thetaDeg) < 0.15;
  const statusText = level
    ? 'level: the string falls on the mark'
    : `tilted ${Math.abs(thetaDeg).toFixed(1)}°: the string misses the mark`;
  const boxes: Box[] = [
    textBox('h', hLabel.x, hLabel.y, 'h', 14, 'end', 'central'),
    textBox('apex', apex.x + 10, apex.y - 8, 'apex', 12),
    textBox('status', status.x, status.y, statusText, narrow ? 12 : 13, 'middle', 'alphabetic', 0.48),
    textBox('true level', levelLabel.x, levelLabel.y, levelText, narrow ? 11 : 12, levelLabel.anchor, 'alphabetic', 0.48),
  ];
  if (Math.abs(thetaDeg) >= 0.6) {
    boxes.push(textBox('x', xLabelPt.x, xLabelPt.y, 'x', 14, 'middle', 'central'));
    boxes.push(textBox('θ', thetaLabel.x, thetaLabel.y, 'θ', 14, thetaLabel.anchor));
  }
  return {
    W,
    Hs,
    narrow,
    s,
    cx,
    gy,
    Hf,
    L,
    h,
    t,
    P,
    beamHalf,
    beam,
    apex,
    cl,
    cr,
    mark,
    xOff,
    hit,
    dimOffset,
    xLabelPt,
    hDim,
    hLabel,
    thetaR,
    thetaFrom,
    thetaTo,
    thetaLabel,
    levelLabel,
    levelText,
    status,
    level,
    statusText,
    boxes,
  };
}

/** Intersection of the string (p + s·d) with segment a→b, as a parameter on a→b. */
export function hitParam(p: Pt, d: Pt, a: Pt, b: Pt) {
  const ex = b.x - a.x;
  const ey = b.y - a.y;
  const den = d.x * ey - d.y * ex;
  if (Math.abs(den) < 1e-9) return 0.5;
  return (d.x * (a.y - p.y) - d.y * (a.x - p.x)) / -den;
}

/* ─────────────────────────── havayi terazi on a cord ─────────────────────────── */

export const CORD = { SPAN: 6, NEAR: 1.2, DROP: 0.3, MIN: 1.3, MAX: 1.9 };

export function cordLayout(W: number, far: number) {
  const { SPAN, NEAR, DROP } = CORD;
  const narrow = W < 520;
  const Hs = narrow ? 330 : 350;
  const S = Math.min(W * 0.36, 250);
  const xL = W / 2 - S;
  const xR = W / 2 + S;
  const ky = (Hs - 120) / 2.1; // px per metre, vertical (exaggerated, said in the caption)
  const gL = Hs - 52;
  const gR = gL + DROP * ky;
  const yAt = (pole: 'L' | 'R', m: number) => (pole === 'L' ? gL : gR) - m * ky;
  const a = { x: xL, y: yAt('L', NEAR) };
  const b = { x: xR, y: yAt('R', far) };
  const tiltDrawn = Math.atan2(a.y - b.y, b.x - a.x);
  const tiltReal = Math.atan2(far - (NEAR + DROP), SPAN);
  const side = narrow ? 84 : 116;
  const th = (side * Math.sqrt(3)) / 2;
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  const ux = Math.cos(tiltDrawn);
  const uy = -Math.sin(tiltDrawn);
  const nx = -uy;
  const ny = ux;
  const drop = 10;
  const top = { x: mid.x + nx * drop, y: mid.y + ny * drop };
  const h1 = { x: top.x - ux * side * 0.5, y: top.y - uy * side * 0.5 };
  const h2 = { x: top.x + ux * side * 0.5, y: top.y + uy * side * 0.5 };
  const tip = { x: top.x + nx * th, y: top.y + ny * th };
  const err = far - (NEAR + DROP);
  const level = Math.abs(err) < 0.006;
  const poleTop = (g: number) => g - 2.05 * ky;
  const statusText = level
    ? `level: the ground falls ${(DROP * 100).toFixed(0)} cm between the poles`
    : `cord ${err > 0 ? 'rises' : 'falls'} ${Math.abs(tiltReal / RAD).toFixed(2)}°: keep sliding`;
  const status = { x: W / 2, y: Hs - 12 };

  // the enlarged-tip detail sits right of the triangle, under the cord, above the status line
  const r = narrow ? 30 : 40;
  const dx = Math.min(xR - r - 14, tip.x + r + (narrow ? 26 : 44));
  const cordY = a.y + ((b.y - a.y) * (dx - xL)) / (xR - xL);
  const dc = { x: dx, y: Math.min(Math.max(cordY + r + 16, tip.y - r * 0.2), status.y - 30 - r) };
  // magnification: as large as fits the worst tilt of the slider range, at most 3×
  const worst = Math.max(
    ...[CORD.MIN, CORD.MAX].map((f) => {
      const t = Math.atan2((f - NEAR - DROP) * ky, 2 * S);
      return Math.abs(th * Math.sin(t));
    }),
  );
  const mag = Math.min(3, (r * 0.8) / worst);
  const boxes: Box[] = [
    textBox('near pole', xL, poleTop(gL) - 8, 'near pole', 12, 'middle'),
    textBox('far pole', xR, poleTop(gR) - 8, 'far pole', 12, 'middle'),
    textBox('status', status.x, status.y, statusText, narrow ? 12 : 13, 'middle', 'alphabetic', 0.48),
    textBox('detail', dc.x, dc.y + r + 14, 'the tip, enlarged', 11, 'middle', 'alphabetic', 0.48),
  ];
  return {
    W,
    Hs,
    narrow,
    S,
    xL,
    xR,
    ky,
    gL,
    gR,
    a,
    b,
    tiltDrawn,
    tiltReal,
    side,
    th,
    ux,
    uy,
    nx,
    ny,
    drop,
    top,
    h1,
    h2,
    tip,
    err,
    level,
    statusText,
    status,
    detail: { r, c: dc, mag },
    poleTop,
    boxes,
  };
}

/* ─────────────────────────── arcsecond chart ─────────────────────────── */

export const ARCSEC = Math.PI / 180 / 3600;
export const EARTH_R = 6_371_000;
export const CHART = { X0: 0, X1: 6, Y0: -6, Y1: 4 }; // log10 metres

export function chartLayout(W: number, logD: number, epsArcsec: number) {
  const narrow = W < 520;
  const H = narrow ? 300 : 330;
  const m = { l: narrow ? 50 : 60, r: narrow ? 12 : 20, t: 18, b: 40 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const { X0, X1, Y0, Y1 } = CHART;
  const x = (ld: number) => m.l + ((ld - X0) / (X1 - X0)) * iw;
  const y = (le: number) => m.t + (1 - (le - Y0) / (Y1 - Y0)) * ih;
  const d = 10 ** logD;
  const miss = d * Math.tan(epsArcsec * ARCSEC);
  const drop = (d * d) / (2 * EARTH_R);
  const ex = x(logD);
  const missLe = Math.log10(miss);
  const dropLe = Math.log10(drop);
  const missOn = missLe >= Y0 && missLe <= Y1;
  const dropOn = dropLe >= Y0 && dropLe <= Y1;
  const ey = y(missLe);
  const cy = y(dropLe);
  const right = logD > 4.6;
  const missLabel = { x: ex + (right ? -9 : 9), y: ey + (ey < m.t + 30 ? 16 : -10), anchor: right ? ('end' as const) : ('start' as const) };
  const crossover = 2 * EARTH_R * Math.tan(epsArcsec * ARCSEC); // where d·tanε = d²/2R
  return { W, H, narrow, m, iw, ih, x, y, d, miss, drop, ex, ey, cy, missOn, dropOn, missLabel, crossover };
}

/* ─────────────────────────── theodolite ─────────────────────────── */

/**
 * Lowest point of the telescope (alidade coordinates, metres of the model) at tilt v, and the
 * top of the alidade plate it must clear. Parts: tube r .055 from x −.28 to .32, objective
 * r .07 at .32 to .42, hood r .072 at .42 to .45, eyepiece r .035 at −.39 to −.27, eye ring
 * r .045 at −.38 to −.34, focus ring r .064 at −.145 to −.095. Trunnion at y = .12.
 */
export function scopeClearance(vDeg: number) {
  const parts = [
    { x0: -0.28, x1: 0.32, r: 0.055 },
    { x0: 0.32, x1: 0.42, r: 0.07 },
    { x0: 0.42, x1: 0.45, r: 0.072 },
    { x0: -0.39, x1: -0.27, r: 0.035 },
    { x0: -0.38, x1: -0.34, r: 0.045 },
    { x0: -0.145, x1: -0.095, r: 0.064 },
  ];
  const v = vDeg * RAD;
  let low = Infinity;
  let high = -Infinity;
  for (const p of parts)
    for (const x of [p.x0, p.x1])
      for (const s of [-1, 1]) {
        const y = 0.12 + x * Math.sin(v) + s * p.r * Math.cos(v);
        low = Math.min(low, y);
        high = Math.max(high, y);
      }
  return { low, high, plateTop: -0.26 + 0.015 };
}
