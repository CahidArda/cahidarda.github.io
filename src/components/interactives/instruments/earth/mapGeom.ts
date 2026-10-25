/**
 * Episode 4b: the map measurer (chartometer / opisometer) on a chart of the site.
 * The chart is the site's plan (site.ts) printed at 1:SCALE. The wheel rolls along the road
 * (ROAD, world m); the dial reads the rolled length in inches on the outer ring (one turn of
 * the needle = 30 in, dots at half inches) and in cm on the inner ring (one turn = 99 cm), as
 * on the box of the instrument in the photo.
 * Screen: SVG viewBox units, y down; the chart is drawn to the page, not to its printed size.
 */
import { ROAD_LEN, ROAD_PTS, LINE_H, roadAt, SITE } from './site.ts';

export const SCALE = 5000; // 1:5,000
export const IN_M = 0.0254;
export const DIAL_IN = 30; // inches per turn of the needle (outer ring)
export const DIAL_CM = 99; // cm per turn (inner ring)
/** Chart length (inches) of a world length (m). */
export const chartIn = (m: number) => m / SCALE / IN_M;
/** World metres per chart inch: 127 m at 1:5,000. */
export const M_PER_IN = SCALE * IN_M;
export const ROAD_IN = chartIn(ROAD_LEN);
export const STRAIGHT_IN = chartIn(LINE_H);
export const READ_STEP = 0.05; // in: the half-inch dots let you judge to about a twentieth
export const readOk = (r: number) => Math.abs(r - ROAD_IN) <= READ_STEP / 2 + 1e-9;
/** The box's own example: 1:250,000, 1 in ≈ 3.95 statute miles, 1 cm = 2.5 km. */
export const BOX = { scale: 250_000, miPerIn: (250_000 * IN_M) / 1609.344, kmPerCm: 250_000 / 100_000 };

/** The chart (plan of the site) and the wheel's position after rolling fraction f of the road. */
export function chartScene(W: number, f: number) {
  const pad = 14;
  const x0w = -40;
  const x1w = SITE.SHORE_X + 260;
  const y0w = -150;
  const y1w = 110;
  const k = (W - 2 * pad) / (x1w - x0w);
  const H = Math.round((y1w - y0w) * k + 2 * pad + 40);
  const P = (x: number, y: number) => ({ x: pad + (x - x0w) * k, y: pad + 18 + (y1w - y) * k });
  const road = ROAD_PTS.filter((_, i) => i % 4 === 0 || i === ROAD_PTS.length - 1).map((r) => P(r.p.x, r.p.y));
  const at = roadAt(f);
  const wheel = P(at.p.x, at.p.y);
  const done = ROAD_PTS.slice(0, at.i).filter((_, i) => i % 4 === 0).map((r) => P(r.p.x, r.p.y)).concat([wheel]);
  const shoreX = P(SITE.SHORE_X, 0).x;
  const M = P(0, 0);
  const A = P(SITE.SHORE_X, 0);
  // scale bar: 0 to 2 inches of chart = 254 m, drawn in world metres
  const bar = { x0: P(0, -125).x, x1: P(2 * M_PER_IN, -125).x, y: P(0, -125).y, ticks: [0, 0.5, 1, 1.5, 2].map((i) => P(i * M_PER_IN, -125).x) };
  return { W, H, k, P, road, wheel, dir: at.dir, done, shoreX, M, A, bar };
}

/** The dial: needle angle for a rolled chart length (in), both rings. */
export function dialScene(W: number, rolledIn: number, read: number | null) {
  const H = 222;
  const cx = W / 2;
  const cy = 112;
  const R = Math.min(84, W / 2 - 24);
  const ang = (inch: number) => 90 - (inch / DIAL_IN) * 360; // 0 at the top, clockwise
  const needle = ang(rolledIn);
  const outer = Array.from({ length: DIAL_IN * 2 }, (_, i) => ({ a: ang(i / 2), major: i % 2 === 0, label: i % 6 === 0 ? String(i / 2) : undefined }));
  const inner = Array.from({ length: DIAL_CM }, (_, i) => ({ a: 90 - (i / DIAL_CM) * 360, label: i % 11 === 0 ? String(i) : undefined }));
  const readAng = read === null ? null : ang(read);
  return { W, H, cx, cy, R, needle, outer, inner, readAng };
}
