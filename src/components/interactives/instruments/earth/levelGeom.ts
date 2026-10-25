/**
 * Episode 1 (plumb line, A-frame, havayi terazi): the numbers and the plumb-line-on-a-wall
 * drawing. The A-frame and the cord drawings are ./geometry.ts (aframeLayout, cordLayout).
 * Screen: SVG viewBox units, y down. World: metres, the plumb line is truly vertical.
 */
import { AFRAME_H_CM, CORD, RAD, textBox, type Box } from './geometry.ts';
import { SITE, SLOPE_RAD, CREST_Z } from './site.ts';

/** The A-frame can stand on three surfaces of the site. Angles in degrees, + = falls to the right. */
export const SURFACES = [
  { id: 'hill', name: 'on the hilltop', theta: 0 },
  { id: 'road', name: 'on the road', theta: -SLOPE_RAD / RAD }, // falls to the right, toward the quay
  { id: 'quay', name: 'on the quay', theta: 0 },
] as const;
export type SurfaceId = (typeof SURFACES)[number]['id'];

/** The miss on the crossbar for a frame whose apex is h above the crossbar, tilted θ. */
export const missMm = (thetaDeg: number) => AFRAME_H_CM * 10 * Math.tan(Math.abs(thetaDeg) * RAD);
/** Slope from the reading: θ = arctan(x / h). */
export const slopeFromMiss = (xMm: number) => Math.atan(xMm / (AFRAME_H_CM * 10)) / RAD;
/** Egyptian seked: palms of run per cubit (7 palms) of rise. */
export const seked = (thetaDeg: number) => 7 / Math.tan(Math.abs(thetaDeg) * RAD);
/** The road: falls (CREST_Z − QUAY_Z) over (SHORE_X − CREST_X): exactly 1 in 8. */
export const ROAD_RUN_PER_RISE = (SITE.SHORE_X - SITE.CREST_X) / (CREST_Z - SITE.QUAY_Z);
/** On the cord: the ground's fall between the poles, from the two readings. */
export const cordFall = (far: number) => far - CORD.NEAR;
export const CORD_LEVEL_TOL = 0.005; // m: the bob sits on the mark within 5 mm at the far pole
export const cordLevel = (far: number) => Math.abs(far - CORD.NEAR - CORD.DROP) < CORD_LEVEL_TOL;

/* ── truing a wall with a plumb line ── */

export const LEAN = { MIN: -20, MAX: 20, STEP: 0.5, START: 12, TOL: 0.5 }; // mm per metre
export const WALL_H = 1.0; // m, the course being checked
export const GAP_TOP_MM = 30; // the bracket holds the line 30 mm off the wall's top edge

/**
 * The wall face leans out by `lean` mm per metre (+ = top leans toward the line). The line
 * hangs truly vertical from a bracket on the top edge. Gap at the top is fixed by the bracket;
 * the gap at the foot shows the lean: gapFoot = gapTop + lean·H (top leaning toward the line
 * means the foot is further away). Drawn with a horizontal exaggeration EX (said on the plate).
 */
export function wallScene(W: number, lean: number) {
  const H = 260;
  const EX = 4; // horizontal exaggeration of gaps and lean
  const top = 34;
  const foot = H - 54;
  const hPx = foot - top; // px per WALL_H
  const kx = (hPx / (WALL_H * 1000)) * EX; // px per mm, horizontal
  const lineX = Math.min(W * 0.6, W - 112);
  const gapTop = GAP_TOP_MM;
  const gapFoot = gapTop + lean * WALL_H;
  const faceTop = { x: lineX - gapTop * kx, y: top };
  const faceFoot = { x: lineX - gapFoot * kx, y: foot };
  const wallBack = Math.max(16, Math.min(faceTop.x, faceFoot.x) - Math.min(140, W * 0.35));
  const plumb = Math.abs(lean) <= LEAN.TOL;
  const status = plumb
    ? 'plumb: the line runs parallel to the face'
    : `leans ${Math.abs(lean).toFixed(1)} mm per metre ${lean > 0 ? 'out' : 'in'}: tap the stone`;
  const labels = {
    top: { x: lineX + 10, y: top + 14, text: `top ${gapTop.toFixed(0)} mm` },
    foot: { x: lineX + 10, y: foot - 22, text: `foot ${gapFoot.toFixed(1)} mm` },
    status: { x: W / 2, y: H - 12, text: status },
  };
  const boxes: Box[] = [
    textBox('top gap', labels.top.x, labels.top.y, labels.top.text, 11, 'start', 'alphabetic', 0.55),
    textBox('foot gap', labels.foot.x, labels.foot.y, labels.foot.text, 11, 'start', 'alphabetic', 0.55),
    textBox('status', labels.status.x, labels.status.y, labels.status.text, 12, 'middle', 'alphabetic', 0.48),
  ];
  return { W, H, EX, top, foot, kx, lineX, gapTop, gapFoot, faceTop, faceFoot, wallBack, plumb, labels, boxes };
}

/* ── the crossbar, enlarged: where the string crosses the scale ── */

/** Marks every 5 mm, numbered every 2 cm, from −10 to +10 cm; the string at x = h·tan θ. */
export function crossbarScene(W: number, thetaDeg: number) {
  const H = 170;
  const cx = W / 2;
  const half = Math.min(W / 2 - 18, 260);
  const kmm = half / 100; // px per mm (±100 mm shown)
  const y = 86;
  const x = missMm(thetaDeg) * (thetaDeg < 0 ? 1 : -1); // + = right of centre on screen
  const sx = cx + x * kmm;
  const ticks: { x: number; major: boolean; label?: string }[] = [];
  for (let mm = -100; mm <= 100; mm += 5) {
    const major = mm % 20 === 0;
    ticks.push({ x: cx + mm * kmm, major, label: major ? `${Math.abs(mm / 10)}` : undefined });
  }
  const read = `${(Math.round(missMm(thetaDeg) * 2) / 2).toFixed(1)} mm`;
  const readLabel = { x: Math.min(Math.max(sx, 40), W - 40), y: y - 34, text: `x = ${read}` };
  return { W, H, cx, half, kmm, y, x, sx, ticks, read, readLabel };
}
