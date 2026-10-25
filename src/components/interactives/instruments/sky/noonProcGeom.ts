/**
 * Latitude by a noon sight, as a procedure: the pure numbers and the drawn coordinates of every
 * view in NoonSightProcedure.tsx (no React), so `noonProcGeom.sweep.ts` can check them.
 *
 * Numbers. The sun's declination δ and distance come from the same USNO "approximate solar
 * coordinates" as the rest of the sky post (`sunPosition` in ./noonGeom.ts), evaluated at the
 * place's local noon in GMT (12 h − longitude/15; the equation of time, at most about 16 min,
 * moves δ by under 0.2′ and is ignored). The world is exact: the sun's centre stands at
 * h_true = 90° − |φ − δ| at noon. What the sextant reads is what a navigator on a deck reads:
 * the sun's LOWER edge on the SEA horizon, so
 *     reading = h_true + refraction(apparent) − semi-diameter + dip.
 * The corrections undo that (dip: Bowditch 1.76′√(eye height in m); refraction: Bennett 1982,
 * R = cot(h + 7.31/(h + 4.4)) arcminutes at 10 °C and 101 kPa; semi-diameter: USNO 0.2666°/R).
 * So an exact reading gives back h_true and φ to rounding, and a reading off by x′ gives φ off by
 * x′ (one nautical mile per arcminute).
 *
 * Drawings. Screen coordinates, y DOWN, viewBox units = CSS px of the measured width W. Angles
 * in degrees, 0° = +x, counter-clockwise on screen (the `polar` convention of ../shared).
 * Reference view: a cut through the Earth along the observer's meridian, north up, sunlight
 * from the right. It is the WORLD at the true values and does not change between steps.
 * The eyepiece reuses `sextantView` (./sextantView.ts), the same function as the 3D sextant.
 */
import { sunPosition, DOY_PRESET, YEAR } from './noonGeom.ts';
import { FIELD_R, ON_TOL, sextantView } from './sextantView.ts';
import { armAngle } from './sextantGeom.ts';

export { DOY_PRESET, FIELD_R, ON_TOL, armAngle };

const RAD = Math.PI / 180;

/* ───────────────────────────── places and constants ───────────────────────────── */

export type Place = { id: string; name: string; lat: number; lon: number };
/** Latitudes/longitudes to 0.01° (Istanbul: the Sultanahmet reference used across the series). */
export const PLACES: Place[] = [
  { id: 'ist', name: 'Istanbul', lat: 41.0136, lon: 28.955 },
  { id: 'sin', name: 'Singapore', lat: 1.29, lon: 103.85 },
  { id: 'cpt', name: 'Cape Town', lat: -33.92, lon: 18.42 },
  { id: 'tro', name: 'Tromsø', lat: 69.65, lon: 18.96 },
];
/** Height of the observer's eye above the sea, metres (a small ship's deck). */
export const EYE_HEIGHT = 3;
/** Lowest noon altitude the plate lets you measure (Bennett's formula and sights get poor lower). */
export const MIN_ALT = 5;
/** Arm reading slider range (degrees) and the time slider (local solar hours). */
export const READ_MIN = 0;
export const READ_MAX = 95;
/** The arm clamps at whole degrees; the micrometer drum adds 0 to 59.9 minutes (0.1′ steps). */
export const DRUM_STEP = 0.1;
export const readingOf = (deg: number, drum: number) => deg + drum / 60;
export const T_MIN = 9;
export const T_MAX = 15;
export const T_STEP = 1 / 60;
export const T_START = 10;
/** The time slider snaps to noon within this many hours (3 min): the sun has stopped climbing. */
export const NOON_SNAP = 3 / 60;
export const READ_START = 0;
/** Exact reading split into arm degrees + drum minutes (rounded to the drum's 0.1′). */
export function splitReading(r: number) {
  let deg = Math.floor(r);
  let min = Math.round((r - deg) * 600) / 10;
  if (min >= 60) {
    deg += 1;
    min = 0;
  }
  return { deg, min };
}

/** Dip of the sea horizon, arcminutes (Bowditch, ch. 22: D = 1.76′ √h, h in metres). */
export const dipArcmin = (m: number) => 1.76 * Math.sqrt(m);
/** Atmospheric refraction, arcminutes, from the APPARENT altitude in degrees (Bennett 1982). */
export const refractionArcmin = (ha: number) =>
  Math.max(0, 1 / Math.tan((ha + 7.31 / (ha + 4.4)) * RAD)); // the formula dips below 0 near 90°

/** GMT hours of local noon at a longitude (east positive). */
export const noonUtc = (lon: number) => 12 - lon / 15;
/** The Date of local noon on day-of-year `doy` of 2026 at a longitude. */
export const noonDate = (doy: number, lon: number) =>
  new Date(Date.UTC(YEAR, 0, doy) + noonUtc(lon) * 3600e3);

/** δ (deg) and the sun's semi-diameter (deg) at a moment (USNO approximate solar coordinates). */
export function sunAt(date: Date) {
  const { dec } = sunPosition(date);
  const D = date.getTime() / 86400000 + 2440587.5 - 2451545.0;
  const g = (357.529 + 0.98560028 * D) * RAD;
  const R = 1.00014 - 0.01671 * Math.cos(g) - 0.00014 * Math.cos(2 * g);
  return { dec, sd: 0.2666 / R };
}

/* ───────────────────────────── the sight ───────────────────────────── */

export type Sight = {
  place: Place;
  doy: number;
  utc: number; // GMT hours of local noon
  dec: number; // δ, deg
  sd: number; // sun's semi-diameter, deg
  dip: number; // deg
  hTrue: number; // centre, geometric, deg
  hApp: number; // centre, as refraction lifts it, deg
  refr: number; // deg, at hApp
  hs: number; // the exact sextant reading: lower edge on the sea horizon, deg
  bearing: 'S' | 'N'; // where the noon sun stands
  visible: boolean; // high enough to measure today
};

export function sight(doy: number, place: Place, eye = EYE_HEIGHT): Sight {
  const utc = noonUtc(place.lon);
  const { dec, sd } = sunAt(noonDate(doy, place.lon));
  const hTrue = 90 - Math.abs(place.lat - dec);
  let hApp = hTrue;
  for (let i = 0; i < 40; i++) hApp = hTrue + refractionArcmin(Math.max(hApp, -1)) / 60;
  const refr = hApp - hTrue;
  const dip = dipArcmin(eye) / 60;
  return {
    place,
    doy,
    utc,
    dec,
    sd,
    dip,
    hTrue,
    hApp,
    refr,
    hs: hApp - sd + dip,
    bearing: place.lat >= dec ? 'S' : 'N',
    visible: hTrue >= MIN_ALT,
  };
}

/** The navigator's working from a reading: corrections, h, φ, and the miss in nautical miles. */
export function reduce(s: Sight, reading: number) {
  const dip = -s.dip;
  const semi = s.sd;
  // refraction at the apparent altitude of the centre (reading − dip + semi-diameter)
  const refr = -refractionArcmin(reading - s.dip + s.sd) / 60;
  const corr = dip + semi + refr;
  const h = reading + corr;
  const phi = s.bearing === 'S' ? 90 - h + s.dec : h - 90 + s.dec;
  const miss = (phi - s.place.lat) * 60; // arcminutes = nautical miles, + = too far north
  return { dip, semi, refr, corr, h, phi, miss, missKm: Math.abs(miss) * 1.852 };
}

/** Sun's altitude and azimuth (deg) at local solar time t (hours). Azimuth from SOUTH, west +. */
export function sunAltAz(lat: number, dec: number, t: number) {
  const H = 15 * (t - 12) * RAD;
  const p = lat * RAD;
  const d = dec * RAD;
  const alt = Math.asin(Math.sin(p) * Math.sin(d) + Math.cos(p) * Math.cos(d) * Math.cos(H)) / RAD;
  const az = Math.atan2(Math.sin(H), Math.cos(H) * Math.sin(p) - Math.tan(d) * Math.cos(p)) / RAD;
  return { alt, az };
}
/** Snap the time to noon once the sun has stopped climbing (within NOON_SNAP). */
export const snapTime = (t: number) => (Math.abs(t - 12) <= NOON_SNAP + 1e-9 ? 12 : t);

/* ───────────────────────────── formatting ───────────────────────────── */

/** Degrees and decimal minutes, the navigator's way: 41°00.8′ (N/S optional). */
export function dm(v: number, hemi?: [string, string]) {
  const a = Math.abs(v);
  let d = Math.floor(a);
  let m = Math.round((a - d) * 600) / 10;
  if (m >= 60) {
    d += 1;
    m = 0;
  }
  const s = `${d}°${m.toFixed(1).padStart(4, '0')}′`;
  if (hemi) return `${s} ${v >= 0 ? hemi[0] : hemi[1]}`;
  return v < 0 ? `−${s}` : s;
}
/** Signed arcminutes: +16.2′ / −3.0′. */
export const am = (deg: number) => `${deg < 0 ? '−' : '+'}${Math.abs(deg * 60).toFixed(1)}′`;
/** "09:52" from hours. */
export function hhmm(h: number) {
  const m = Math.round((((h % 24) + 24) % 24) * 60);
  return `${String(Math.floor(m / 60) % 24).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
}

/* ───────────────────────────── layout helpers ───────────────────────────── */

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
const boxesOverlap = (a: Box, b: Box, pad = 1) =>
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
function placer(W: number, H: number, segs: Seg[], arcs: Pt[][] = [], soft: string[] = []) {
  const placed: Label[] = [];
  const free = (l: Omit<Label, 'placed'>) => {
    const b = labelBox(l);
    if (!inBounds(b, W, H)) return false;
    // small word labels may cross the faint decorative rays (`soft`): their paper halo cuts them
    const hard = l.size >= 14 ? segs : segs.filter((s) => !soft.includes(s.id));
    if (hard.some((s) => segHitsBox(s.a, s.b, b, 1))) return false;
    if (arcs.some((pts) => pts.some((p, i) => i > 0 && segHitsBox(pts[i - 1], p, b, 0.5))))
      return false;
    return !placed.some((o) => boxesOverlap(labelBox(o), b, 6));
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

/** One drawing size for both views, from the measured column width. */
export const viewH = (W: number) => Math.round(Math.max(290, Math.min(400, W * 0.86)));

/* ───────────────────────────── reference: the Earth cut along your meridian ───────────────────────────── */

export function refScene(s: Sight, W: number) {
  const H = viewH(W);
  const phi = s.place.lat;
  const dec = s.dec;
  // the zenith line (1.32 Re from the centre) must fit above and below, with the sun's side open
  const Re = Math.min((H / 2 - 12) / 1.32, (W - 40) / 2.55);
  const cx = Re * 1.32 + 10;
  const cy = H / 2;
  const zLen = Re * 0.32;
  const L = Re * 0.62;
  const obs = P(cx, cy, Re, phi);
  const zen = P(cx, cy, Re + zLen, phi);
  const horDir = s.bearing === 'S' ? phi - 90 : phi + 90; // horizon towards the sun
  const horSun = P(obs.x, obs.y, L, horDir);
  const horFar = P(obs.x, obs.y, L * 0.7, horDir + 180);
  // a ray back towards the sun from p, clipped to the drawing (4 units in)
  const toSun = (p: Pt) => {
    const ux = Math.cos(dec * RAD);
    const uy = -Math.sin(dec * RAD);
    let t = (W - 4 - p.x) / ux;
    if (uy < 0) t = Math.min(t, (4 - p.y) / uy);
    if (uy > 0) t = Math.min(t, (H - 4 - p.y) / uy);
    return { x: p.x + ux * t, y: p.y + uy * t };
  };
  const sunFrom = toSun(obs);
  // other parallel rays across the beam, ending where they meet the Earth
  const rays = [-0.62, 0.62].map((o) => {
    const n = { x: Math.sin(dec * RAD), y: Math.cos(dec * RAD) }; // screen normal of the beam
    const off = o * Re;
    const along = Math.sqrt(Math.max(Re * Re - off * off, 0));
    const hit = { x: cx + n.x * off + Math.cos(dec * RAD) * along, y: cy + n.y * off - Math.sin(dec * RAD) * along };
    return { a: toSun(hit), b: hit, o };
  });
  const sub = P(cx, cy, Re, dec); // where the sun is overhead
  const phiArcR = Re * 0.4;
  const decArcR = Re * 0.68;
  const hArcR = L * 0.55;
  // h arc: from the horizon (sun side) to the sun, the short way
  const hFrom = s.bearing === 'S' ? horDir : dec;
  const hTo = s.bearing === 'S' ? dec : horDir;
  const drawnH = Math.abs(hTo - hFrom);
  const drawnPhi = Math.atan2(-(obs.y - cy), obs.x - cx) / RAD;
  const drawnDec = Math.atan2(-(sunFrom.y - obs.y), sunFrom.x - obs.x) / RAD;

  const segs: Seg[] = [
    { id: 'equator', a: { x: cx - Re - 8, y: cy }, b: { x: cx + Re + 14, y: cy } },
    { id: 'axis', a: { x: cx, y: cy - Re - 8 }, b: { x: cx, y: cy + Re + 8 } },
    { id: 'zenith', a: { x: cx, y: cy }, b: zen },
    { id: 'decline', a: { x: cx, y: cy }, b: sub },
    { id: 'horizon', a: horFar, b: horSun },
    { id: 'sunray', a: sunFrom, b: obs },
    ...rays.map((r, i) => ({ id: `ray${i}`, a: r.a, b: r.b })),
  ];
  const arcs = [
    arcPts(cx, cy, phiArcR, 0, phi),
    arcPts(cx, cy, decArcR, 0, dec),
    arcPts(obs.x, obs.y, hArcR, hFrom, hTo),
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
    for (const dr of [11, -11, 20, -18, 30])
      for (const f of fr) out.push(lab(id, text, P(x0, y0, r + dr, from + span * f), S));
    // a tiny angle: just outside its ends
    for (const dr of [0, 10, -10]) for (const e of [-12, 12, -20, 20]) out.push(lab(id, text, P(x0, y0, r + dr, (e < 0 ? Math.min(from, to) : Math.max(from, to)) + e), S));
    return out;
  };
  // most constrained first
  pl.put(along('h', 'h', obs.x, obs.y, hArcR, hFrom, hTo));
  pl.put(along('phi', 'φ', cx, cy, phiArcR, 0, phi));
  pl.put(along('dec', 'δ', cx, cy, decArcR, 0, dec));
  pl.put(ring('you', 'you', obs, s11, [14, 22, 30], [phi + 180, phi + 150, phi + 210]));
  pl.put(ring('zenith', 'zenith', zen, s11, [8, 14, 22, 32], [phi, phi + 30, phi - 30]));
  pl.put(ring('horizon', 'horizon', horSun, s11, [8, 14, 22, 32], [horDir - 60, horDir + 60, horDir]));
  pl.put([
    lab('pole', 'N pole', { x: cx + 5, y: cy - Re - 10 }, s11, 'start'),
    lab('pole', 'N pole', { x: cx - 5, y: cy - Re - 10 }, s11, 'end'),
    ...ring('pole', 'N pole', { x: cx, y: cy - Re - 4 }, s11, [10, 18, 28], [150, 30]),
  ]);
  pl.put([
    lab('equator', 'equator', { x: cx - Re + 8, y: cy - 8 }, s11, 'start'),
    lab('equator', 'equator', { x: cx - Re + 8, y: cy + 9 }, s11, 'start'),
    lab('equator', 'equator', { x: cx + Re + 4, y: cy + 9 }, s11, 'start'),
    lab('equator', 'equator', { x: cx + Re + 4, y: cy - 8 }, s11, 'start'),
  ]);
  pl.put([
    lab('sun', 'sunlight', { x: W - 6, y: 12 }, s11, 'end'),
    lab('sun', 'sunlight', { x: W - 6, y: H - 10 }, s11, 'end'),
    lab('sun', 'sunlight', { x: W - 6, y: cy - Re - 10 }, s11, 'end'),
    lab('sun', 'sunlight', { x: W - 6, y: cy + Re + 10 }, s11, 'end'),
    ...[0.25, 0.75, 0.5, 0.1, 0.9].map((f) => lab('sun', 'sunlight', { x: W - 6, y: 12 + (H - 22) * f }, s11, 'end')),
  ]);
  const labels = Object.fromEntries(pl.placed.map((l) => [l.id, l])) as Record<string, Label>;
  return {
    W, H, Re, cx, cy, zLen, L, obs, zen, horDir, horSun, horFar, sunFrom, rays, sub,
    phiArcR, decArcR, hArcR, hFrom, hTo, drawnH, drawnPhi, drawnDec, segs, labels, phi, dec,
  };
}

/* ───────────────────────────── step 1: the sky to the south (or north) ───────────────────────────── */

/** Panorama facing the noon sun: azimuth ±AZ_SPAN across the width, altitude 0..90 up. */
export const AZ_SPAN = 110;
export function skyScene(s: Sight, t: number, W: number) {
  const H = viewH(W);
  const pad = 14;
  const horizonY = H - 46;
  const top = 30;
  const ky = (horizonY - top) / 90;
  const kx = (W - 2 * pad) / (2 * AZ_SPAN);
  const mid = W / 2;
  const facingN = s.bearing === 'N';
  // x for an azimuth (from south, west +): facing south west is on the right; facing north, left
  const xOf = (az: number) => {
    let rel = facingN ? az - 180 : az;
    rel = ((((rel + 180) % 360) + 360) % 360) - 180;
    return mid + (facingN ? -rel : rel) * kx;
  };
  const yOf = (alt: number) => horizonY - alt * ky;
  const path: Pt[][] = [];
  let run: Pt[] = [];
  for (let tt = 3; tt <= 21.0001; tt += 0.05) {
    const { alt, az } = sunAltAz(s.place.lat, s.dec, tt);
    const x = xOf(az);
    if (alt < -1 || x < pad || x > W - pad || (run.length && Math.abs(x - run[run.length - 1].x) > W / 3)) {
      if (run.length > 1) path.push(run);
      run = [];
      if (alt < -1 || x < pad || x > W - pad) continue;
    }
    run.push({ x, y: yOf(Math.max(alt, 0)) });
  }
  if (run.length > 1) path.push(run);
  const now = sunAltAz(s.place.lat, s.dec, t);
  const noon = sunAltAz(s.place.lat, s.dec, 12);
  const sun = { x: xOf(now.az), y: yOf(now.alt) };
  const sunNoon = { x: xOf(noon.az), y: yOf(noon.alt) };
  // rate of climb, arcminutes per minute of time (central difference)
  const d = 1 / 120;
  const rate =
    ((sunAltAz(s.place.lat, s.dec, t + d).alt - sunAltAz(s.place.lat, s.dec, t - d).alt) * 60) / (2 * d * 60);
  const compass = facingN
    ? [ { t: 'E', az: -90 + 180 }, { t: 'N', az: 180 }, { t: 'W', az: 90 + 180 } ]
    : [ { t: 'E', az: -90 }, { t: 'S', az: 0 }, { t: 'W', az: 90 } ];
  const marks = compass.map((c) => ({ ...c, x: xOf(c.az) }));
  return { W, H, pad, horizonY, top, ky, kx, mid, path, sun, sunNoon, now, noon, rate, marks, sunR: Math.max(5, s.sd * ky * 3) };
}

/* ───────────────────────────── step 2: the eyepiece ───────────────────────────── */

/** Eyepiece of the 2D view: sea horizon in the clear half, the sun's image in the mirror half. */
export function eyeScene(s: Sight, reading: number, W: number) {
  const H = viewH(W);
  const R = Math.min(W * 0.36, H * 0.36);
  const cx = W / 2;
  const cy = H / 2 - 6;
  const k = R / FIELD_R; // px per degree, the sun's disc at true size
  // the same optics function as the 3D sextant: the sun stands (hApp + dip) above the SEA horizon
  const v = sextantView(reading, 0, s.hApp + s.dip);
  const gap = v.gap; // sun's centre above the horizon in the field, degrees
  const limb = gap - s.sd; // lower edge above the horizon
  const sunR = s.sd * k;
  const sunY = cy - gap * k;
  const horY = cy - v.horizonEl * k;
  const visible = Math.abs(gap) <= FIELD_R + s.sd;
  const status: 'on' | 'above' | 'below' = Math.abs(limb) <= ON_TOL ? 'on' : limb > 0 ? 'above' : 'below';
  return { W, H, R, cx, cy, k, gap, limb, sunR, sunY, horY, visible, status, view: v };
}

/* ───────────────────────────── step 3: the arc and the drum ───────────────────────────── */

export function readScene(reading: number, W: number) {
  const H = viewH(W);
  const deg = Math.floor(reading + 1e-9);
  const min = (reading - deg) * 60;
  // a window of the arc, 1° = kx px, the index moves over it
  const span = 3.5;
  const kx = (W - 40) / (2 * span);
  const x0 = W / 2 - (reading - (deg + 0.5)) * 0; // window centred on deg + 0.5
  const center = deg + 0.5;
  const xOf = (r: number) => W / 2 + (r - center) * kx;
  const arcY = 52;
  const ticks: { x: number; major: boolean; label?: string }[] = [];
  for (let r = Math.ceil((center - span) * 6) / 6; r <= center + span + 1e-9; r += 1 / 6) {
    const rr = Math.round(r * 6) / 6;
    const major = Math.abs(rr - Math.round(rr)) < 1e-6;
    ticks.push({ x: xOf(rr), major, label: major ? `${Math.round(rr)}` : undefined });
  }
  const index = { x: xOf(reading), y: arcY };
  // the micrometer drum: 60 divisions, turned so the minutes sit under the fixed index at the top
  const drumR = Math.min(W * 0.2, (H - 100) / 2 - 24);
  const drum = { cx: W / 2, cy: (H + 100) / 2 + 6, r: drumR, turn: -min * 6 }; // degrees on screen
  void x0;
  return { W, H, deg, min, kx, arcY, ticks, index, drum, xOf, span, center };
}

/* ───────────────────────────── step 6: the result on a latitude scale ───────────────────────────── */

export function resultScene(s: Sight, phi: number, W: number) {
  const H = viewH(W);
  const miss = (phi - s.place.lat) * 60;
  const half = Math.max(12, Math.ceil((Math.abs(miss) * 1.6) / 6) * 6); // arcminutes each side
  const top = 34;
  const bot = H - 34;
  const k = (bot - top) / (2 * half); // px per arcminute
  const yOf = (lat: number) => (top + bot) / 2 - (lat - s.place.lat) * 60 * k;
  const x = W * 0.36;
  const step = half <= 12 ? 2 : half <= 30 ? 6 : half <= 60 ? 10 : 30;
  const ticks: { y: number; v: number; major: boolean }[] = [];
  const base = s.place.lat * 60;
  for (let m = Math.ceil((base - half) / step) * step; m <= base + half + 1e-9; m += step)
    ticks.push({ y: yOf(m / 60), v: m / 60, major: Math.round(m) % (step * 2) === 0 });
  return { W, H, top, bot, k, x, half, yTrue: yOf(s.place.lat), yMeas: yOf(phi), miss, ticks, yOf };
}

/** Almanac rows for step 5: δ at whole GMT hours around your noon, and the value at your noon. */
export function almanac(s: Sight) {
  const h0 = Math.floor(s.utc);
  const rows = [h0 - 1, h0, h0 + 1, h0 + 2].map((h) => {
    const date = new Date(Date.UTC(YEAR, 0, s.doy) + h * 3600e3);
    return { h, label: `${hhmm(h)} GMT`, dec: sunAt(date).dec };
  });
  return { rows, at: { label: `${hhmm(s.utc)} GMT`, dec: s.dec } };
}

/* ───────────────────────────── step 4: the corrections, to scale ───────────────────────────── */

/**
 * The corrections as a running total along one arcminute axis, one row per term so nothing can
 * collide: Hs at 0, then dip, semi-diameter, refraction as bars from the running total, then h.
 * Left column: names and values; right: the bars, to scale.
 */
export function corrScene(s: Sight, reading: number, W: number) {
  const H = viewH(W);
  const r = reduce(s, reading);
  const terms = [
    { id: 'dip', name: 'dip', d: r.dip },
    { id: 'semi', name: 'semi-diameter', d: r.semi },
    { id: 'refr', name: 'refraction', d: r.refr },
  ];
  const tot = [0];
  for (const t of terms) tot.push(tot[tot.length - 1] + t.d * 60);
  const lo = Math.min(0, ...tot);
  const hi = Math.max(0, ...tot);
  const colX = 12;
  const x0 = Math.min(118, W * 0.4); // bars start here
  const x1 = W - 16;
  const span = Math.max(hi - lo, 1);
  const k = (x1 - x0) / span; // px per arcminute
  const xOf = (m: number) => x0 + (m - lo) * k;
  const top = 44;
  const rowH = Math.min(46, (H - top - 30) / 5);
  const rows = [
    { id: 'hs', name: 'Hs', value: dm(reading), role: 'ink' as Role4, from: 0, to: 0 },
    ...terms.map((t, i) => ({ id: t.id, name: t.name, value: am(t.d), role: 'soft' as Role4, from: tot[i], to: tot[i + 1] })),
    { id: 'h', name: 'h', value: dm(r.h), role: 'sky' as Role4, from: tot[3], to: tot[3] },
  ].map((row, i) => ({ ...row, y: top + rowH * (i + 0.5), xa: xOf(row.from), xb: xOf(row.to), nx: colX }));
  const unit = span > 40 ? 10 : span > 12 ? 5 : 1;
  const grid: { x: number; v: number }[] = [];
  for (let m = Math.ceil(lo / unit) * unit; m <= hi + 1e-9; m += unit) grid.push({ x: xOf(m), v: m });
  return { W, H, k, xOf, rows, grid, unit, x0, x1, top, rowH, r, tot };
}
type Role4 = 'ink' | 'soft' | 'sky';

/* ───────────────────────────── status lines (bottom of the views; swept for width) ───────────────────────────── */

export function eyeSay(g: ReturnType<typeof eyeScene>) {
  if (g.status === 'on') return 'lower edge on the horizon';
  if (!g.visible) return g.gap > 0 ? 'sun above the field: swing the arm up' : 'sun below the field: swing the arm down';
  return `lower edge ${Math.abs(g.limb * 60).toFixed(1)}′ ${g.status === 'above' ? 'above' : 'below'} the horizon`;
}
export function skySay(k: ReturnType<typeof skyScene>, t: number) {
  if (k.now.alt <= 0) return 'the sun is below the horizon';
  if (t === 12) return `noon: ${dm(k.now.alt)}, highest`;
  return `${hhmm(t)}: ${dm(k.now.alt)}, ${k.rate >= 0 ? 'rising' : 'sinking'} ${Math.abs(k.rate).toFixed(1)}′/min`;
}
export function resultSay(miss: number) {
  if (Math.abs(miss) < 0.05) return 'on the spot';
  return `${Math.abs(miss).toFixed(1)} nmi (${(Math.abs(miss) * 1.852).toFixed(1)} km) too far ${miss > 0 ? 'north' : 'south'}`;
}
