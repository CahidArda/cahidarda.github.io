/**
 * Episode 1, the sundial: pure geometry of the shadow plate (SundialPlate.tsx), swept by
 * time.sweep.ts.
 *
 * World: a horizontal dial at Istanbul (φ = 41°), x EAST, y NORTH, z UP, in units of the
 * gnomon's height. The sun's direction at hour angle H (degrees, + = afternoon), declination δ:
 *     E = −cos δ sin H,  N = cos φ sin δ − sin φ cos δ cos H,  U = sin φ sin δ + cos φ cos δ cos H.
 * A point P casts its shadow at P − (P_z / U)(E, N).
 *
 * Two dials on one plate:
 *  - 'unequal': a vertical pin of height 1 at the origin. The day from sunrise to sunset is cut
 *    into 12 temporal hours; hour line k joins the shadow-tip positions at temporal hour k on
 *    every day of the year (so it is computed, not drawn by hand). The hour is read from where
 *    the TIP falls, so when the tip runs off the plate there is no reading.
 *  - 'equal': a style (a slanted edge) through the origin, parallel to the Earth's axis: it
 *    rises to the north at φ. Its shadow turns 15° of hour angle per hour whatever the date;
 *    on the plate the hour line for H sits at tan θ = sin φ tan H from the noon line.
 *
 * Screen: y DOWN, viewBox units = CSS px of the measured width W. The world never moves with
 * the controls except through the true time and date.
 */
import { PHI, cosd, declination, halfDay, sind } from './astro.ts';

const R = Math.PI / 180;
export type Mode = 'unequal' | 'equal';
export type V2 = { x: number; y: number };

/** Plate extent in gnomon heights. */
export const PX0 = -4;
export const PX1 = 4;
export const PY0 = -1.25;
export const PY1 = 3.75;
/** Time slider (local solar hours): covers the longest day at 41° (sunrise 4:32). */
export const T_MIN = 3;
export const T_MAX = 21;
/** The style rises at φ and its tip stands 1 unit high, like the pin. */
export const STYLE_FOOT = Math.cos(PHI * R) / Math.sin(PHI * R);

export function sunVec(Hdeg: number, dec: number, phi = PHI) {
  return {
    E: -cosd(dec) * sind(Hdeg),
    N: cosd(phi) * sind(dec) - sind(phi) * cosd(dec) * cosd(Hdeg),
    U: sind(phi) * sind(dec) + cosd(phi) * cosd(dec) * cosd(Hdeg),
  };
}
/** Shadow of a point (x, y, z) on the dial at hour angle H, or null when the sun is down. */
export function shadowOf(p: { x: number; y: number; z: number }, Hdeg: number, dec: number): V2 | null {
  const s = sunVec(Hdeg, dec);
  if (s.U <= 1e-9) return null;
  return { x: p.x - (p.z / s.U) * s.E, y: p.y - (p.z / s.U) * s.N };
}
export const PIN_TOP = { x: 0, y: 0, z: 1 };
export const STYLE_TIP = { x: 0, y: STYLE_FOOT, z: 1 };

/** Sunrise/sunset (local solar hours) and the length of one temporal hour (minutes). */
export function dayOf(day: number) {
  const dec = declination(day);
  const H0 = halfDay(dec);
  const rise = 12 - H0 / 15;
  const set = 12 + H0 / 15;
  return { dec, H0, rise, set, hourMin: ((set - rise) * 60) / 12 };
}
/** Temporal hour k (0..12) as local solar time on a day. */
export const temporalToTime = (k: number, day: number) => {
  const d = dayOf(day);
  return d.rise + (k * (d.set - d.rise)) / 12;
};
/** Local solar time as temporal hours since sunrise (may be <0 or >12 at night). */
export const timeToTemporal = (t: number, day: number) => {
  const d = dayOf(day);
  return (12 * (t - d.rise)) / (d.set - d.rise);
};
/** The equal-hour line direction on the plate for clock hour t (unit vector, x east y north). */
export function equalDir(t: number) {
  const H = 15 * (t - 12);
  const x = sind(PHI) * sind(H);
  const y = cosd(H);
  const l = Math.hypot(x, y);
  return { x: x / l, y: y / l };
}

/** Clip the ray from o along unit d to the plate rectangle; returns the exit distance. */
function rayExit(o: V2, d: V2) {
  let t = Infinity;
  if (d.x > 1e-12) t = Math.min(t, (PX1 - o.x) / d.x);
  if (d.x < -1e-12) t = Math.min(t, (PX0 - o.x) / d.x);
  if (d.y > 1e-12) t = Math.min(t, (PY1 - o.y) / d.y);
  if (d.y < -1e-12) t = Math.min(t, (PY0 - o.y) / d.y);
  return t;
}
export const onPlate = (p: V2, m = 0) => p.x >= PX0 + m && p.x <= PX1 - m && p.y >= PY0 + m && p.y <= PY1 - m;

/** Clip a polyline to the plate (split into runs). */
function clipRuns(pts: (V2 | null)[]) {
  const runs: V2[][] = [];
  let cur: V2[] = [];
  for (const p of pts) {
    if (p && onPlate(p)) cur.push(p);
    else if (cur.length) {
      runs.push(cur);
      cur = [];
    }
  }
  if (cur.length) runs.push(cur);
  return runs.filter((r) => r.length > 1);
}

const SOLSTICE_JUN = 172;
const SOLSTICE_DEC = 355;
/** Days from the winter to the summer solstice (δ from −23.44° to +23.44°). */
const YEAR_SWEEP = Array.from({ length: 61 }, (_, i) => SOLSTICE_DEC + ((SOLSTICE_JUN + 365 - SOLSTICE_DEC) * i) / 60).map((d) => ((d - 1) % 365) + 1);

/** Hour lines in plate units. */
export function hourLines(mode: Mode) {
  if (mode === 'unequal') {
    return Array.from({ length: 11 }, (_, i) => i + 1).map((k) => {
      const pts = YEAR_SWEEP.map((day) => shadowOf(PIN_TOP, 15 * (temporalToTime(k, day) - 12), declination(day)));
      return { key: k, runs: clipRuns(pts) };
    });
  }
  // equal: whole clock hours whose line leaves the origin onto the plate, 4:00 to 20:00
  return Array.from({ length: 17 }, (_, i) => i + 4).map((h) => {
    const d = equalDir(h);
    const e = rayExit({ x: 0, y: 0 }, d);
    return { key: h, runs: [[{ x: 0, y: 0 }, { x: d.x * e, y: d.y * e }]] };
  });
}

/** The shadow tip's path through one day (sunrise to sunset), clipped. */
export function dayCurve(day: number, mode: Mode) {
  const d = dayOf(day);
  const p = mode === 'unequal' ? PIN_TOP : STYLE_TIP;
  const pts: (V2 | null)[] = [];
  for (let i = 0; i <= 160; i++) {
    const t = d.rise + ((d.set - d.rise) * i) / 160;
    pts.push(shadowOf(p, 15 * (t - 12), d.dec));
  }
  return clipRuns(pts);
}

const ORD = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th'];

/** Everything the plate needs at (mode, day, t). */
export function sundial(mode: Mode, day: number, t: number) {
  const d = dayOf(day);
  const H = 15 * (t - 12);
  const up = t > d.rise && t < d.set && sunVec(H, d.dec).U > 0;
  const tipW = up ? shadowOf(mode === 'unequal' ? PIN_TOP : STYLE_TIP, H, d.dec) : null;
  const root: V2 = mode === 'unequal' ? { x: 0, y: 0 } : { x: 0, y: 0 };
  // the drawn shadow: from the root to the tip, clipped at the plate edge
  let shadowEnd: V2 | null = null;
  let tipOff = false;
  if (tipW) {
    if (onPlate(tipW)) shadowEnd = tipW;
    else {
      const len = Math.hypot(tipW.x - root.x, tipW.y - root.y);
      const dir = { x: (tipW.x - root.x) / len, y: (tipW.y - root.y) / len };
      const e = rayExit(root, dir);
      shadowEnd = { x: root.x + dir.x * e, y: root.y + dir.y * e };
      tipOff = true;
    }
  }
  const k = timeToTemporal(t, day); // temporal hours since sunrise
  // readings
  let read: number | null = null;
  let say: string;
  if (!up) {
    say = 'No sun, no shadow: the dial says nothing.';
  } else if (mode === 'unequal') {
    if (tipOff) say = 'The tip of the shadow runs off the dial: too early or late to read.';
    else {
      const kr = Math.round(k);
      if (Math.abs(k - kr) < 0.04 && kr > 0 && kr < 12) say = `The tip is on line ${kr}: the end of the ${ORD[kr - 1]} hour.`;
      else {
        const idx = Math.min(11, Math.max(0, Math.floor(k)));
        say = `The tip is in the ${ORD[idx]} hour of the day.`;
      }
      // read as if the hours were equal: the 6th hour ends at noon, so hour k ↔ 6:00 + k
      read = 6 + k;
    }
  } else {
    read = t;
    const h0 = Math.floor(t);
    const m = Math.round((t - h0) * 60);
    say = m === 0 || m === 60 ? `The shadow lies on the ${m === 60 ? h0 + 1 : h0} line.` : `The shadow lies between the ${h0} and ${h0 + 1} lines.`;
  }
  return { mode, day, t, H, ...d, up, tipW, shadowEnd, tipOff, k, read, say };
}
export type Sundial = ReturnType<typeof sundial>;

/* ───────────────────────────── screen layout ───────────────────────────── */

export const PAD = 14;
export function dialFrame(W: number) {
  const k = (W - 2 * PAD) / (PX1 - PX0);
  const top = 24; // room for the header note
  const H = Math.round(top + (PY1 - PY0) * k + 40);
  const toS = (p: V2) => ({ x: PAD + (p.x - PX0) * k, y: top + (PY1 - p.y) * k });
  return { k, H, top, toS, x0: PAD, x1: W - PAD, y0: top, y1: top + (PY1 - PY0) * k };
}

export type DLabel = { id: string; x: number; y: number; text: string; size: number; anchor: 'start' | 'middle' | 'end' };
export const dlBox = (l: DLabel) => {
  const w = [...l.text].length * l.size * 0.6;
  const x0 = l.anchor === 'start' ? l.x : l.anchor === 'middle' ? l.x - w / 2 : l.x - w;
  return { x0, x1: x0 + w, y0: l.y - l.size * 0.55, y1: l.y + l.size * 0.45, name: l.id };
};

/** Hour numerals: at the outer end of each hour line, pulled inside the plate. */
export function hourLabels(mode: Mode, W: number) {
  const f = dialFrame(W);
  const size = W < 400 ? 9.5 : 10.5;
  const out: DLabel[] = [];
  for (const line of hourLines(mode)) {
    if (!line.runs.length) continue;
    const run = line.runs.reduce((a, b) => (b.length > a.length ? b : a));
    // the end farthest from the gnomon
    const a = run[0];
    const b = run[run.length - 1];
    const far = Math.hypot(a.x, a.y) > Math.hypot(b.x, b.y) ? a : b;
    const near = far === a ? run[Math.min(1, run.length - 1)] : run[Math.max(0, run.length - 2)];
    const s = f.toS(far);
    const n = f.toS(near);
    const len = Math.hypot(s.x - n.x, s.y - n.y) || 1;
    // step inward along the line so the numeral sits inside the plate
    const inX = (n.x - s.x) / len;
    const inY = (n.y - s.y) / len;
    const text = mode === 'unequal' ? String(line.key) : String(line.key);
    let x = s.x + inX * 10;
    let y = s.y + inY * 10;
    const w = text.length * size * 0.6;
    x = Math.max(f.x0 + w / 2 + 2, Math.min(f.x1 - w / 2 - 2, x));
    y = Math.max(f.y0 + size * 0.6 + 2, Math.min(f.y1 - size * 0.5 - 2, y));
    out.push({ id: `h${line.key}`, x, y, text, size, anchor: 'middle' });
  }
  return out;
}
