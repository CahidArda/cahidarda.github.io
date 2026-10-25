/**
 * Plate 3 geometry, pure (no React) so a sweep script can check it.
 *
 * World frame: units = frame size, origin = the index-mirror pivot, y UP, angles in degrees
 * (0° = +x, counter-clockwise). The world is fixed: the sun stands at altitude SUN_ALT up to
 * the left, its light travels down-right, and the horizon is straight ahead to the left.
 * The instrument is a schematic sextant: index mirror on the pivot, horizon mirror below-left,
 * telescope running right from the horizon mirror to the eye (the eye looks left, at the
 * horizon). The reader turns the index mirror by θ (0 = mirrors parallel) and can roll the
 * whole instrument by ρ about the pivot. The sunlight ray that is traced always passes the
 * pivot, so the sun and its ray never move.
 */
export const SUN_ALT = 36.8; // Istanbul, noon on 25 October 2026 (see Plate 4)
export const THETA_MIN = 12;
export const THETA_MAX = 26;
export const ROLL_MIN = -5;
export const ROLL_MAX = 5;
export const H_HALF = 0.09; // half length of the horizon mirror
export const I_HALF = 0.22; // half length of the index mirror
export const SUN_DIST = 0.9;
export const BEAM_HALF = 0.24; // half width of the drawn beam of sunlight

const RAD = Math.PI / 180;
export type V2 = { x: number; y: number };
const norm = (v: V2): V2 => {
  const l = Math.hypot(v.x, v.y) || 1;
  return { x: v.x / l, y: v.y / l };
};
export const dot = (a: V2, b: V2) => a.x * b.x + a.y * b.y;
export const rot = (v: V2, deg: number): V2 => {
  const c = Math.cos(deg * RAD);
  const s = Math.sin(deg * RAD);
  return { x: v.x * c - v.y * s, y: v.x * s + v.y * c };
};
export const reflect = (d: V2, n: V2): V2 => {
  const k = 2 * dot(d, n);
  return { x: d.x - k * n.x, y: d.y - k * n.y };
};
export const ang = (v: V2) => Math.atan2(v.y, v.x) / RAD;
/** Smallest signed difference a − b in degrees, in (−180, 180]. */
export const dAng = (a: number, b: number) => {
  let d = (a - b) % 360;
  if (d > 180) d -= 360;
  if (d <= -180) d += 360;
  return d;
};

/** Instrument at no roll: horizon mirror centre, its normal, telescope. */
const H0: V2 = { x: -0.17, y: -0.3 };
const U0 = norm(H0); // pivot → horizon mirror
const NH0 = norm({ x: 1 - U0.x, y: -U0.y }); // sends light arriving along U0 out along +x
const EYE_X = 0.6; // eye at (EYE_X, H0.y) in the instrument frame

export function doubleReflection(theta: number, roll: number) {
  const H = rot(H0, roll);
  const nH = rot(NH0, roll);
  // index mirror: parallel to the horizon mirror at θ = 0, turned clockwise by θ
  const nI0 = rot(nH, -theta);
  const toSun = rot({ x: -1, y: 0 }, -SUN_ALT); // up and to the left, fixed in the world
  const sun: V2 = { x: toSun.x * SUN_DIST, y: toSun.y * SUN_DIST };
  const d1: V2 = { x: -toSun.x, y: -toSun.y }; // sunlight travels down-right
  // light must reach the silvered (front) face of the index mirror
  const nI = dot(d1, nI0) < 0 ? nI0 : { x: -nI0.x, y: -nI0.y };
  const d2 = reflect(d1, nI);
  // Of all the parallel rays from the sun, trace the one that the index mirror sends to the
  // centre of the horizon mirror: back from H along −d2 to the index mirror's line (through
  // the pivot, normal nI).  Q = H − b·d2 with Q·nI = 0.
  const b = dot(H, nI) / dot(d2, nI);
  const Q: V2 = { x: H.x - b * d2.x, y: H.y - b * d2.y };
  const tI = { x: -nI.y, y: nI.x };
  const qAlong = dot(Q, tI); // where on the index mirror the light lands
  const onIndex = Math.abs(qAlong) <= I_HALF && b > 0;
  // the traced ray starts on the line across the sun's disc, perpendicular to its light
  const sAlong = dot({ x: Q.x - sun.x, y: Q.y - sun.y }, toSun); // ≤ 0: Q is below the sun
  const start: V2 = { x: Q.x - toSun.x * sAlong, y: Q.y - toSun.y * sAlong };
  const startOffset = dot({ x: start.x - sun.x, y: start.y - sun.y }, { x: -toSun.y, y: toSun.x });
  const P = H;
  const d3 = reflect(d2, nH);
  // telescope line (rolls with the instrument)
  const telFrom = rot({ x: H0.x + 0.12, y: H0.y }, roll);
  const eye = rot({ x: EYE_X, y: H0.y }, roll);
  const telDir = rot({ x: 1, y: 0 }, roll);
  const D = dAng(ang(d3), ang(d1)); // deviation, CCW positive
  // where the eye sees the sun's image: elevation of −d3 (+ = above the horizon)
  const imageVsHorizon = -dAng(ang(d3), 0);
  const imageVsTelescope = dAng(ang(d3), ang(telDir));
  const outLen = Math.hypot(eye.x - P.x, eye.y - P.y) + 0.05;
  const out: V2 = { x: P.x + d3.x * outLen, y: P.y + d3.y * outLen };
  return {
    theta, roll, H, nH, nI, tI, sun, toSun, d1, d2, d3, Q, qAlong, onIndex, start, startOffset, P,
    telFrom, eye, telDir, D, imageVsHorizon, imageVsTelescope, out,
    aligned: Math.abs(imageVsHorizon) < 0.3,
  };
}
export type DR = ReturnType<typeof doubleReflection>;

export type Pt = { x: number; y: number };
export type Label = { id: string; x: number; y: number; text: string; size: number; anchor: 'start' | 'middle' | 'end'; baseline?: 'central' | 'alphabetic' };

/** Everything drawn on the plate, in screen px, for a drawing W px wide. */
export function drScene(theta: number, roll: number, W: number) {
  const g = doubleReflection(theta, roll);
  const narrow = W < 520;
  const S = Math.min((W - 40) / 1.47, 280); // px per unit
  const ox = W / 2 + 0.06 * S; // content spans about −0.8 … 0.68 units
  const oy = 10 + 0.76 * S; // the beam of sunlight reaches 0.73 units up
  const H = Math.round(oy + 0.5 * S + 18);
  const P2 = (v: V2): Pt => ({ x: ox + v.x * S, y: oy - v.y * S });
  const seg = (c: V2, n: V2, half: number) => {
    const t = { x: -n.y, y: n.x };
    return [P2({ x: c.x + t.x * half, y: c.y + t.y * half }), P2({ x: c.x - t.x * half, y: c.y - t.y * half })] as const;
  };
  const RAD_ = Math.PI / 180;
  const pa = (c: Pt, r: number, deg: number): Pt => ({ x: c.x + r * Math.cos(deg * RAD_), y: c.y - r * Math.sin(deg * RAD_) });
  const sun = P2(g.sun);
  const sunR = narrow ? 9 : 11;
  const idx = seg({ x: 0, y: 0 }, g.nI, I_HALF);
  const hor = seg(g.H, g.nH, H_HALF);
  const pivot = P2({ x: 0, y: 0 });
  const Q = P2(g.Q);
  const P = P2(g.P);
  const start = P2(g.start);
  const out = P2(g.out);
  const eye = P2(g.eye);
  const telFrom = P2(g.telFrom);
  // parallel sunlight: a beam across the sun's direction, fixed in the world
  const perp = { x: -g.toSun.y, y: g.toSun.x };
  const beam = [-0.24, -0.16, -0.08, 0, 0.08, 0.16, 0.24].map((o) => {
    const a0 = { x: g.sun.x + perp.x * o, y: g.sun.y + perp.y * o };
    return [P2(a0), P2({ x: a0.x + g.d1.x * 0.22, y: a0.y + g.d1.y * 0.22 })] as const;
  });
  // θ: between the index mirror and a dashed line through the pivot parallel to the horizon mirror
  const tH = { x: -g.nH.y, y: g.nH.x };

  const angTH = ang(tH);
  const parLen = 0.12;
  // drawn on the far side of the pivot (the vertical angle), clear of the light
  // (the arc sits far enough out that the gap between its two sides fits the θ label)
  const thR = Math.max(S * 0.17, 9 / Math.sin((theta / 2) * (Math.PI / 180)) - 12);
  const thetaArc = { cx: pivot.x, cy: pivot.y, r: thR, from: angTH + 180 - theta, to: angTH + 180 };
  // D: at the horizon mirror, between the incoming sunlight (dashed copy) and the outgoing light
  const pl = thR / S + 0.04;
  const parallel = [P2({ x: tH.x * parLen, y: tH.y * parLen }), P2({ x: -tH.x * pl, y: -tH.y * pl })] as const;
  // dashed extension of the index mirror's line, so both sides of θ reach the arc
  const idxExt = [pivot, pa(pivot, thR + 8, angTH + 180 - theta)] as const;
  const dR = S * 0.26;
  const ghost = [P, P2({ x: g.P.x + g.d1.x * 0.32, y: g.P.y + g.d1.y * 0.32 })] as const;
  const dArc = { cx: P.x, cy: P.y, r: dR, from: ang(g.d1), to: ang(g.d3) };
  // light from the horizon is horizontal in the world: a dashed reference through P
  const horizonLine = [P, { x: Math.max(eye.x, out.x) + 6, y: P.y }] as const;
  const hR = S * 0.36;
  const hArc = { cx: P.x, cy: P.y, r: hR, from: ang(g.d1), to: 0 };
  const labels: Label[] = [
    { id: 'sun', x: sun.x + sunR + 6, y: sun.y - 4, text: 'sun (fixed)', size: 12, anchor: 'start' },
    { id: 'idx', x: Math.max(idx[0].x, idx[1].x) + 8, y: Math.min(idx[0].y, idx[1].y) + 4, text: 'index mirror', size: 12, anchor: 'start' },
    { id: 'hor', x: Math.min(hor[0].x, hor[1].x) - 6, y: Math.max(hor[0].y, hor[1].y) + 14, text: 'horizon mirror', size: 12, anchor: 'middle' },
    { id: 'eye', x: eye.x, y: eye.y + 22, text: 'eye', size: 12, anchor: 'middle' },
    { id: 'h', ...pa(P, hR + 12, ang(g.d1) / 2 - 4), text: 'h', size: 14, anchor: 'middle', baseline: 'central' },
    { id: 'theta', ...pa(pivot, thetaArc.r + 12, angTH + 180 - theta / 2), text: 'θ', size: 14, anchor: 'middle', baseline: 'central' },
    { id: 'D', ...pa(P, dR - 12, (ang(g.d1) + ang(g.d3)) / 2), text: 'D', size: 14, anchor: 'middle', baseline: 'central' },
  ];
  return { g, narrow, W, H, S, P2, sun, sunR, idx, hor, pivot, Q, P, start, out, eye, telFrom, beam, parallel, thetaArc, idxExt, ghost, dArc, hArc, horizonLine, labels };
}
export type Scene = ReturnType<typeof drScene>;

/** Approximate text box (px) for collision checks: serif ~0.5 em per char. */
export function labelBox(l: Label) {
  const w = l.text.length * l.size * 0.5;
  const x0 = l.anchor === 'start' ? l.x : l.anchor === 'end' ? l.x - w : l.x - w / 2;
  const y0 = l.baseline === 'central' ? l.y - l.size * 0.5 : l.y - l.size * 0.8;
  return { x0, x1: x0 + w, y0, y1: y0 + l.size };
}

/** Reading for which the sun's image sits on the horizon (roll does not matter). */
export const alignedTheta = SUN_ALT / 2;
