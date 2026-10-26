/**
 * Plate 2 (Polaris and the kamal), pure numbers and drawn coordinates, swept by
 * `kamalGeom.sweep.ts`. Used by KamalProcedure.tsx.
 *
 * World. Polaris sits P_OFF = 0.64° from the celestial pole in 2026 (0.66° in 2018, about 0.45°
 * soon after 2100: Wikipedia, Polaris; linear interpolation) and circles it once a sidereal day.
 * Its hour angle `lha` (deg, 0 = above the pole, 90 = west of it) sets where it is on that circle.
 * Its true altitude: sin a = sin φ cos p + cos φ sin p cos(lha). Refraction and dip are left out
 * (a few arcminutes, under the kamal's resolution); the sweep checks the first-order rule
 * φ ≈ a − p cos(lha) against this exact formula.
 * The kamal: a board BOARD_H tall held square to the line of sight, the eye level with its lower
 * edge, on a cord of length d: tan θ = h / d.
 *
 * Drawings. Screen coordinates, y DOWN, viewBox units = CSS px of the measured width W.
 * Sky view (steps 1-2): facing north, the sea horizon at y = horizonY, altitude up at k px/deg
 * (fixed per place). Side view (step 3): eye, cord and board to one scale. Circle view (step 4):
 * the pole and Polaris's circle magnified CIRCLE_MAG times (said so on the plate).
 */
import { viewH } from './latRefGeom.ts';

const RAD = Math.PI / 180;

export const P_OFF = 0.64; // deg, Polaris from the pole, 2026
export const BOARD_H = 10; // cm
export const D_MIN = 8; // cm, cord length slider
export const D_MAX = 60;
export const D_STEP = 0.05;
/** The board's top edge "touches" Polaris within this, degrees (about the star's twinkle and a steady hand). */
export const TOUCH = 0.15;

export type Port = { id: string; name: string; lat: number };
/** Latitudes to 0.01°: Kozhikode (Calicut), Hormuz island, Istanbul (Sultanahmet). */
export const PORTS: Port[] = [
  { id: 'cal', name: 'Calicut', lat: 11.25 },
  { id: 'hor', name: 'Hormuz', lat: 27.06 },
  { id: 'ist', name: 'Istanbul', lat: 41.01 },
];

/** Polaris's true altitude (deg) at latitude φ and hour angle lha. */
export function polarisAlt(phi: number, lha: number, p = P_OFF) {
  const s = Math.sin(phi * RAD) * Math.cos(p * RAD) + Math.cos(phi * RAD) * Math.sin(p * RAD) * Math.cos(lha * RAD);
  return Math.asin(s) / RAD;
}
/** Board angle for a cord length (cm). */
export const thetaOf = (d: number) => Math.atan(BOARD_H / d) / RAD;
/** Cord length that frames an angle. */
export const cordFor = (theta: number) => BOARD_H / Math.tan(theta * RAD);
/** The board can frame Polaris at all: above the horizon and within the cord's range. */
export const catchable = (alt: number) => alt > thetaOf(D_MAX) && alt < thetaOf(D_MIN);
/** Latitude from the measured angle, first order in p. */
export const latOf = (theta: number, lha: number) => theta - P_OFF * Math.cos(lha * RAD);

export const where = (lha: number) => {
  const a = ((lha % 360) + 360) % 360;
  const names = ['above the pole', 'up and to the west', 'west of the pole', 'down and to the west', 'below the pole', 'down and to the east', 'east of the pole', 'up and to the east'];
  return names[Math.round(a / 45) % 8];
};

/* ───────────────────────────── steps 1-2: the northern sky ───────────────────────────── */

export function skyScene(port: Port, lha: number, d: number, W: number) {
  const H = viewH(W);
  const alt = polarisAlt(port.lat, lha);
  const top = Math.max(22, port.lat + 14); // degrees shown above the horizon (fixed per port)
  const horizonY = H - 52;
  const k = (horizonY - 34) / top; // px per degree
  const cx = W * 0.5;
  const star = { x: cx + Math.sin(lha * RAD) * P_OFF * k * -1, y: horizonY - alt * k };
  // the pole: seen facing north, west is to the LEFT; Polaris west of the pole (lha 90) sits left
  const pole = { x: cx, y: horizonY - port.lat * k };
  const theta = thetaOf(d);
  // a board taller than the drawn sky is clipped at the top (and marked), not drawn off-plate
  const boardTopTrue = horizonY - theta * k;
  const boardTop = Math.max(boardTopTrue, 4);
  const boardW = Math.max(46, Math.min(120, W * 0.24));
  const board = { x0: cx - boardW / 2, x1: cx + boardW / 2, y0: boardTop, y1: horizonY };
  const gap = theta - alt; // + = top edge above Polaris
  const touching = alt > 0 && Math.abs(gap) <= TOUCH;
  const ticks = [] as { y: number; a: number }[];
  const step = top > 40 ? 10 : 5;
  for (let a = step; a < top; a += step) ticks.push({ y: horizonY - a * k, a });
  return { W, H, alt, top, horizonY, k, cx, star, pole, theta, board, boardTopTrue, clipped: boardTopTrue < 4, gap, touching, ticks };
}
export function skySay(g: ReturnType<typeof skyScene>) {
  if (g.alt <= 0) return 'Polaris is below the horizon here';
  if (g.touching) return 'top edge on Polaris';
  return `top edge ${Math.abs(g.gap).toFixed(1)}° ${g.gap > 0 ? 'above' : 'below'} Polaris`;
}

/* ───────────────────────────── step 3: side view of the kamal ───────────────────────────── */

/**
 * Side view (step 3, no slider on that step): drawn to one scale for the CURRENT cord, chosen so
 * the board is tall enough to label. Labels: "eye" under the eye, d under the cord, h right of
 * the board, θ just outside its arc.
 */
export function sideScene(d: number, W: number) {
  const H = viewH(W);
  const pad = 26;
  const eye = { x: pad + 10, y: H - 86 };
  const k = Math.min((W - 2 * pad - 100) / d, (eye.y - 40) / BOARD_H); // px per cm
  const boardX = eye.x + d * k;
  const boardTop = { x: boardX, y: eye.y - BOARD_H * k };
  const theta = thetaOf(d);
  const arcR = Math.min(70, d * k * 0.45);
  // knots every 10 cm along the cord
  const marks: { x: number; cm: number }[] = [];
  for (let cm = 10; cm < d; cm += 10) marks.push({ x: eye.x + cm * k, cm });
  const labels = {
    eye: { x: eye.x - 6, y: eye.y + 22, text: 'eye', size: 11, anchor: 'start' as const },
    d: { x: (eye.x + boardX) / 2, y: eye.y + 22, text: `d = ${d.toFixed(1)} cm`, size: 13, anchor: 'middle' as const },
    h: { x: boardX + 8, y: (eye.y + boardTop.y) / 2, text: `board ${BOARD_H} cm`, size: 13, anchor: 'start' as const },
    theta: { x: eye.x + arcR + 8, y: eye.y - Math.sin((theta * Math.PI) / 360) * arcR, text: 'θ', size: 14, anchor: 'start' as const },
  };
  // the sight line on to the star, past the board
  const far = { x: eye.x + (W - pad - eye.x), y: eye.y - (W - pad - eye.x) * Math.tan(theta * RAD) };
  const farClip = far.y < 26 ? { x: eye.x + (eye.y - 26) / Math.tan(theta * RAD), y: 26 } : far;
  return { W, H, k, eye, boardX, boardTop, theta, arcR, marks, far: farClip, pad, labels };
}

/* ───────────────────────────── step 4: Polaris's circle, magnified ───────────────────────────── */

export const CIRCLE_MAG = 60;
export function circleScene(port: Port, lha: number, W: number) {
  const H = viewH(W);
  const cx = W / 2;
  const cy = H / 2 - 4;
  const r = Math.min(W * 0.3, H * 0.32); // drawn radius of Polaris's circle
  const kd = r / P_OFF; // px per degree, magnified
  const star = { x: cx - Math.sin(lha * RAD) * r, y: cy - Math.cos(lha * RAD) * r };
  const corr = -P_OFF * Math.cos(lha * RAD); // degrees to add to the measured altitude
  return { W, H, cx, cy, r, kd, star, corr, mag: kd };
}

/* ───────────────────────────── step 5: cord length against latitude ───────────────────────────── */

export const CURVE_LAT_MIN = 0;
export const CURVE_LAT_MAX = 50;
export function curveScene(port: Port, W: number) {
  const H = viewH(W);
  const x0 = 48;
  const x1 = W - 16;
  const y0 = 36;
  const y1 = H - 56;
  const kx = (x1 - x0) / (CURVE_LAT_MAX - CURVE_LAT_MIN);
  const ky = (y1 - y0) / D_MAX;
  const xOf = (lat: number) => x0 + (lat - CURVE_LAT_MIN) * kx;
  const yOf = (d: number) => y1 - d * ky;
  const pts: { x: number; y: number }[] = [];
  for (let lat = thetaOf(D_MAX); lat <= CURVE_LAT_MAX + 1e-9; lat += 0.25) pts.push({ x: xOf(lat), y: yOf(cordFor(lat)) });
  // one knot per degree: their spacing on the cord, at the port's latitude
  const here = port.lat > thetaOf(D_MAX) ? { x: xOf(port.lat), y: yOf(cordFor(port.lat)), perDeg: cordFor(port.lat - 0.5) - cordFor(port.lat + 0.5) } : null;
  // the curve falls to the right, so the label goes above-right of the point on the left half
  // (the curve is lower there) and below-left on the right half (the curve is higher there)
  const right = here ? here.x > (x0 + x1) / 2 : false;
  const hereLabel = here
    ? { x: here.x + (right ? -10 : 10), y: here.y + (right ? 18 : -14), text: `${port.name}: ${cordFor(port.lat).toFixed(1)} cm`, size: 12, anchor: (right ? 'end' : 'start') as 'end' | 'start' }
    : null;
  const yTicks = [0, 20, 40, 60].map((d) => ({ y: yOf(d), d }));
  const xTicks = [0, 10, 20, 30, 40, 50].map((l) => ({ x: xOf(l), l }));
  return { W, H, x0, x1, y0, y1, pts, here, hereLabel, xTicks, yTicks, xOf, yOf, unit: { x: x0 + 10, y: y0 + 4, text: 'cm', size: 11, anchor: 'start' as const } };
}
