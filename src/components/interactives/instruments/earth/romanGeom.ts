/**
 * Episode 3: Roman right angles and gradients.
 *
 * GROMA (plan, to scale, north up). The groma stands at A on the near quay. Its cross has four
 * plumb lines; opposite strings are S apart. The cross is turned to ψ (degrees, maths
 * convention: 0° = east, 90° = north). Pair 1 sights along ψ, pair 2 along ψ − 90°.
 * B is 200 m due north of A (site.ts), the far quay runs north-south 700 m east of A.
 * Wind swings the strings sideways by `sway` mm (the far string of the pair, the worst case),
 * which turns the sight line by atan(sway / S).
 *
 * CHOROBATES (side view). A bench ~20 Roman feet long on the channel bed, which falls
 * 2.5 cm in 100 m (site.ts). The reader shims the downhill leg; the bench then tilts by
 * (GROUND_FALL − shim) / LEG_SPAN. It LOOKS level while that tilt is under the tilt the water
 * groove can show (TILT_TOL): it is level only "as far as the eye can tell". The sight along
 * the top then meets a staff 100 m away, and the reading carries the residual tilt with it.
 * Screen: SVG viewBox units, y down. World: metres unless named mm.
 */
import { SITE, CREST_Z } from './site.ts';

export const RAD = Math.PI / 180;

/* ───────────────────────── groma ───────────────────────── */

export const GROMA = {
  S: 0.9, // m between opposite plumb lines (illustrative; reconstructions differ)
  PSI_MIN: 0,
  PSI_MAX: 180,
  PSI_STEP: 0.1,
  PSI_START: 35,
  TOL: 0.15, // deg: the strings cover the pole
  POLE_MIN: -150,
  POLE_MAX: 250,
  POLE_STEP: 1,
  SWAY_MAX: 15, // mm
  B_Y: SITE.B_Y,
  WIDTH: SITE.WIDTH,
};

/** Where pair 2's sight line meets the far quay (m north of A), or null if it misses. */
export function farHit(psiDeg: number, swayMm = 0) {
  const dir = (psiDeg - 90) * RAD + Math.atan(swayMm / 1000 / GROMA.S);
  if (Math.cos(dir) <= 1e-6) return null;
  return GROMA.WIDTH * Math.tan(dir);
}
/** Is pair 1 on B (due north)? */
export const onB = (psi: number) => Math.abs(psi - 90) <= GROMA.TOL;
/** Is the far pole hidden behind pair 2's strings (within half a pole's width, 0.5 m)? */
export function poleHidden(psi: number, poleY: number) {
  const y = farHit(psi);
  return y !== null && Math.abs(y - poleY) <= 0.5;
}
/** The wind's miss on the far shore (m) for a sway in mm, and as an angle (arcmin). */
export const windMiss = (swayMm: number) => GROMA.WIDTH * Math.tan(Math.atan(swayMm / 1000 / GROMA.S));
export const windAngleMin = (swayMm: number) => (Math.atan(swayMm / 1000 / GROMA.S) / RAD) * 60;

export function gromaScene(W: number, psi: number, poleY: number, sway: number) {
  const pad = 18;
  const H = Math.round(Math.min(360, Math.max(300, W * 0.72)));
  // plan scale: 700 m across plus margins; north range −150..250 m
  const k = Math.min((W - 2 * pad - 40) / GROMA.WIDTH, (H - 70) / (GROMA.POLE_MAX - GROMA.POLE_MIN));
  const ax = pad + 30;
  const ay = 36 + GROMA.POLE_MAX * k; // y of A (north = up)
  const P = (x: number, y: number) => ({ x: ax + x * k, y: ay - y * k });
  const A = P(0, 0);
  const B = P(0, GROMA.B_Y);
  const farX = ax + GROMA.WIDTH * k;
  const pole = P(GROMA.WIDTH, poleY);
  const hit0 = farHit(psi);
  const hitW = farHit(psi, sway);
  const ray = (y: number | null) => (y === null ? null : P(GROMA.WIDTH, Math.max(GROMA.POLE_MIN - 30, Math.min(GROMA.POLE_MAX + 30, y))));
  // pair 1 sight line toward B: drawn to the top edge along ψ
  const d1 = { x: Math.cos(psi * RAD), y: -Math.sin(psi * RAD) };
  const t1 = Math.min(...[d1.y < 0 ? (A.y - 20) / -d1.y : Infinity, d1.x > 0 ? (W - pad - A.x) / d1.x : Infinity, d1.x < 0 ? (A.x - 6) / -d1.x : Infinity]);
  const sight1 = { x: A.x + d1.x * t1, y: A.y + d1.y * t1 };
  // the cross, enlarged in an inset circle below-right of A
  const inset = { cx: ax + Math.min(150, (farX - ax) * 0.42), cy: ay + Math.min(70, (H - ay) - 46), r: 38 };
  if (inset.cy + inset.r > H - 26) inset.cy = H - 26 - inset.r;
  const arm = inset.r * 0.78;
  const strings = [0, 90, 180, 270].map((a) => {
    const t = (psi + a) * RAD;
    return { x: inset.cx + arm * Math.cos(t), y: inset.cy - arm * Math.sin(t), pair: a % 180 === 0 ? 1 : 2 };
  });
  return { W, H, pad, k, A, B, farX, pole, hit0, hitW, ray0: ray(hit0), rayW: ray(hitW), sight1, inset, strings, P };
}

/* ───────────────────────── chorobates ───────────────────────── */

export const CHORO = {
  LEN: 5.9, // m, "about twenty feet" (Roman foot ≈ 0.296 m)
  LEG_SPAN: 5.6, // m between the legs
  TOP: 1.0, // m, height of the top edge above the feet when level
  GROOVE: 1.48, // m, the water groove: 5 Roman feet
  MENISCUS: 0.001, // m: the water's edge judged to about 1 mm (an assumption, said in the caption)
  SHIM_MIN: 0,
  SHIM_MAX: 10, // mm
  SHIM_STEP: 0.5,
  SHIM_START: 9,
  RUN: SITE.CREST_X - SITE.FLAT_X, // 100 m of channel
  FALL: SITE.HILL_Z - CREST_Z, // 0.025 m
};
/** The smallest tilt the water groove shows: about 0.68 per thousand. */
export const TILT_TOL = CHORO.MENISCUS / CHORO.GROOVE;
/** The bed falls this much between the legs (mm). */
export const BED_FALL_MM = (CHORO.FALL / CHORO.RUN) * CHORO.LEG_SPAN * 1000;
/** Tilt of the bench (rad, + = dips toward the staff, downhill) for a shim under the downhill leg. */
export const tiltOf = (shimMm: number) => (BED_FALL_MM - shimMm) / 1000 / CHORO.LEG_SPAN;
export const looksLevel = (shimMm: number) => Math.abs(tiltOf(shimMm)) < TILT_TOL;
/** Distance from the bench's downhill leg to the staff at the end of the channel (m). */
export const SIGHT_D = CHORO.RUN - CHORO.LEG_SPAN;
/** The bed's true fall between the leg and the staff (m). */
export const TRUE_DROP = (CHORO.FALL * SIGHT_D) / CHORO.RUN;
/** Height of the top edge above the bed at the downhill leg (m): the instrument height. */
export const instHeight = (shimMm: number) => CHORO.TOP + shimMm / 1000;
/**
 * The staff stands on the bed at the end. The sight along the top edge leaves at the
 * instrument height and runs at the bench's tilt (dipping toward the staff when tilt > 0),
 * so it meets the staff at: instrument height + the bed's drop − distance · tilt.
 */
export const staffReading = (shimMm: number) => instHeight(shimMm) + TRUE_DROP - SIGHT_D * Math.tan(tiltOf(shimMm));
/** The surveyor's arithmetic: fall = reading − instrument height (m, over SIGHT_D). */
export const measuredDrop = (shimMm: number) => staffReading(shimMm) - instHeight(shimMm);
/** Largest error in that fall from a tilt the water cannot show (m). */
export const DROP_UNC = SIGHT_D * TILT_TOL;

/** The bench up close, tilt exaggerated ×EX (said on the drawing). */
export function benchScene(W: number, shimMm: number) {
  const H = 220;
  const pad = 20;
  const EX = 60; // tilt exaggeration
  const L = W - 2 * pad - 20;
  const kx = L / CHORO.LEN;
  const footY = H - 56;
  const topH = 70; // px from feet to top edge
  const t = Math.atan(Math.tan(tiltOf(shimMm)) * EX); // drawn tilt, + = right end down
  const x0 = pad + 10;
  const legL = x0 + ((CHORO.LEN - CHORO.LEG_SPAN) / 2) * kx;
  const legR = legL + CHORO.LEG_SPAN * kx;
  const bedSlope = Math.atan((BED_FALL_MM / 1000 / CHORO.LEG_SPAN) * EX);
  const bedY = (x: number) => footY + (x - legL) * Math.tan(bedSlope);
  const shimPx = Math.max(0, shimMm) * (kx / 1000) * EX;
  // the bench rotates about its left foot
  const pivot = { x: legL, y: bedY(legL) };
  const R = (x: number, y: number) => {
    const dx = x - pivot.x;
    const dy = y - pivot.y;
    return { x: pivot.x + dx * Math.cos(t) - dy * Math.sin(t), y: pivot.y + dx * Math.sin(t) + dy * Math.cos(t) };
  };
  const topL = R(x0, pivot.y - topH);
  const topR = R(x0 + CHORO.LEN * kx, pivot.y - topH);
  const footR = R(legR, pivot.y);
  // plumb lines hang from two crosspieces, truly vertical; the marks tilt with the bench
  const bobs = [0.25, 0.75].map((f) => {
    const hang = R(x0 + CHORO.LEN * kx * f, pivot.y - topH + 6);
    const mark = R(x0 + CHORO.LEN * kx * f, pivot.y - topH + 46);
    return { hang, mark, bob: { x: hang.x, y: hang.y + 40 } };
  });
  // water in the groove: level surface; rims follow the bench
  const gL = R(x0 + (CHORO.LEN * kx) / 2 - (CHORO.GROOVE * kx) / 2, pivot.y - topH);
  const gR = R(x0 + (CHORO.LEN * kx) / 2 + (CHORO.GROOVE * kx) / 2, pivot.y - topH);
  const tilt = tiltOf(shimMm);
  const status = looksLevel(shimMm)
    ? 'looks level: water touches both rims'
    : `tilted ${(Math.abs(tilt) * 1000).toFixed(2)} per thousand: ${tilt > 0 ? 'raise' : 'lower'} the right leg`;
  return { W, H, EX, pad, kx, footY, legL, legR, bedY, shimPx, pivot, topL, topR, footR, bobs, gL, gR, status, t };
}

/**
 * The long sight to the staff, side view. Horizontal: 100 m of channel to scale. Vertical:
 * the bench and staff are drawn at a fixed size (heights above the bed NOT to scale); the
 * bed's fall and the sight line's departure from level are drawn ×VE (said on the drawing),
 * both on the same scale, so the reading's error is visible beside the true fall.
 */
export function sightScene(W: number, shimMm: number, showBand: boolean) {
  const H = 250;
  const pad = 22;
  const VE = 250;
  const kx = (W - 2 * pad - 34) / CHORO.RUN;
  const kv = kx * VE; // px per metre for the fall and the error
  const X = (x: number) => pad + x * kx;
  const legX = X(CHORO.LEG_SPAN);
  const staffX = X(CHORO.RUN);
  const levelY = 128; // screen y of the true level line through the instrument top
  const gap = 56; // px from the instrument top down to the bed at the leg (not to scale)
  const bed0 = { x: X(0), y: levelY + gap - (CHORO.FALL * CHORO.LEG_SPAN / CHORO.RUN) * kv };
  const bedEnd = { x: staffX, y: levelY + gap + TRUE_DROP * kv };
  const legBed = { x: legX, y: levelY + gap };
  const t = tiltOf(shimMm);
  const hitRaw = levelY + SIGHT_D * Math.tan(t) * kv;
  const TOPY = 16;
  const hit = { x: staffX, y: Math.max(TOPY, hitRaw), clipped: hitRaw < TOPY };
  const read = staffReading(shimMm);
  const drop = measuredDrop(shimMm);
  // the staff's own scale: reading r (m above the bed at the staff) → screen y
  const r0 = instHeight(shimMm) + TRUE_DROP; // the reading where a truly level line meets the staff
  const yOfR = (r: number) => levelY - (r - r0) * kv;
  const band = showBand ? { y0: levelY - DROP_UNC * kv, y1: levelY + DROP_UNC * kv } : null;
  // staff ticks every cm of reading, labelled every 2 cm, around the hit
  const ticks: { y: number; cm: number; label: boolean }[] = [];
  const base = Math.round(read * 100);
  for (let c = base - 12; c <= base + 12; c++) {
    const y = yOfR(c / 100);
    if (y < 16 || y > bedEnd.y - 4) continue;
    ticks.push({ y, cm: c, label: c % 2 === 0 });
  }
  const status = `staff reads ${(read * 100).toFixed(1)} cm · instrument height ${(instHeight(shimMm) * 100).toFixed(1)} cm`;
  return { W, H, pad, VE, kx, kv, X, legX, staffX, levelY, gap, bed0, bedEnd, legBed, hit, read, drop, yOfR, band, ticks, status };
}
