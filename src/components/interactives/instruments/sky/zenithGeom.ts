/**
 * Plate 10 (the digital zenith camera), pure numbers and drawn coordinates, swept by
 * `zenithGeom.sweep.ts`. Used by ZenithCameraProcedure.tsx.
 *
 * World. The point straight overhead (along the plumb line) has a declination equal to your
 * ASTRONOMICAL latitude. A zenith camera is levelled with tilt meters, photographs the stars
 * round the zenith, and matches them against a star catalogue: the declination of the image's
 * centre is φ. Here: Istanbul, GPS (geodetic) latitude PHI_GPS; the plumb line is deflected by an
 * ILLUSTRATIVE DEFLECT arcseconds to the north (real deflections are several arcseconds; the plate
 * says so), so the astronomical latitude is PHI_GPS + DEFLECT.
 * Stars: a seeded field (mulberry32) within ±FIELD arcseconds of the zenith, offsets in arcseconds
 * (north +, east +). The plumb line is the "true" zenith; an unlevelled camera's centre sits
 * `tilt` arcseconds north of it. The catalogue overlay is drawn for the reader's latitude guess.
 *
 * Drawings. Screen coordinates, y DOWN, north up, viewBox units = CSS px of the measured width W.
 */
import { viewH } from './latRefGeom.ts';

export const PHI_GPS = 41.0136; // Istanbul (Sultanahmet), the series' reference point
export const DEFLECT = 4.0; // arcsec, illustrative
export const PHI_ASTRO = PHI_GPS + DEFLECT / 3600;
export const FIELD = 600; // arcsec, half the field shown
export const TILT_MIN = -20;
export const TILT_MAX = 20;
export const TILT_STEP = 0.5;
export const TILT_TOL = 0.5;
/** Latitude guess slider: arcseconds past 41°00′. */
export const G_MIN = 0;
export const G_MAX = 120;
export const G_STEP = 0.5;
export const MATCH_TOL = 0.5; // arcsec
export const BASE = 41; // 41°00′00″
export const guessLat = (g: number) => BASE + g / 3600;
export const truthG = (PHI_ASTRO - BASE) * 3600;

function mulberry32(a: number) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = mulberry32(2010);
/** Catalogue stars: declination offsets from the TRUE astronomical zenith (arcsec), east offsets, size. */
export const STARS = Array.from({ length: 22 }, () => ({
  n: (rnd() * 2 - 1) * FIELD * 0.9,
  e: (rnd() * 2 - 1) * FIELD * 0.9,
  m: 1.5 + rnd() * 2.5,
}));

export function fieldScene(tilt: number, g: number | null, W: number) {
  const H = viewH(W);
  const size = Math.min(W - 40, H - 70);
  const cx = W / 2;
  const cy = 20 + size / 2;
  const k = size / (2 * FIELD); // px per arcsec
  // image centre = zenith + tilt (north); a star at offset n from the true zenith appears at n − tilt
  const stars = STARS.map((s) => ({ x: cx + s.e * k, y: cy - (s.n - tilt) * k, r: s.m }));
  // catalogue circles for the guessed latitude: a star at declination φ_astro + n is expected at
  // (φ_astro + n − guess) from the centre
  const shift = g === null ? null : (PHI_ASTRO - guessLat(g)) * 3600;
  const cat = shift === null ? [] : STARS.map((s) => ({ x: cx + s.e * k, y: cy - (s.n + shift) * k }));
  return { W, H, size, cx, cy, k, stars, cat, shift };
}

export function bubbleScene(tilt: number, W: number) {
  const H = viewH(W);
  const cx = W / 2;
  const cy = H * 0.42;
  const len = Math.min(W - 60, 300);
  const k = len / 2 / (TILT_MAX + 4); // px per arcsec
  return { W, H, cx, cy, len, k, bubble: { x: cx + tilt * k, y: cy } };
}

/** "41°00′53.0″" */
export function dms1(v: number) {
  const a = Math.abs(v);
  let d = Math.floor(a);
  let m = Math.floor((a - d) * 60);
  let s = Math.round(((a - d) * 3600 - m * 60) * 10) / 10;
  if (s >= 60) {
    s = 0;
    m += 1;
  }
  if (m >= 60) {
    m = 0;
    d += 1;
  }
  return `${d}°${String(m).padStart(2, '0')}′${s.toFixed(1).padStart(4, '0')}″`;
}
