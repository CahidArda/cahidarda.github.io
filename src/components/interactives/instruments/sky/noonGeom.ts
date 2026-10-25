/**
 * Noon-sight geometry and the sun's declination, pure (no React) so a sweep script can check it.
 * Declination: USNO "approximate solar coordinates" (about 1′ near the year 2000).
 * Screen coordinates, y down; angles in degrees, 0° = +x, counter-clockwise on screen.
 */
const RAD = Math.PI / 180;
export const YEAR = 2026;
export const ISTANBUL = 41.0136;
export const DOY_PRESET = 298; // 25 October 2026
export const H_MAX = 89;
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** Day of year (1..365) → Date at local noon in Istanbul (≈ 09:50 UTC). */
export const dateOf = (doy: number) => new Date(Date.UTC(YEAR, 0, doy, 9, 50));
export const dateLabel = (doy: number) => {
  const d = dateOf(doy);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
};

/** Solar declination and ecliptic longitude in degrees (USNO approximate solar coordinates). */
export function sunPosition(date: Date) {
  const D = date.getTime() / 86400000 + 2440587.5 - 2451545.0;
  const g = (357.529 + 0.98560028 * D) * RAD;
  const q = 280.459 + 0.98564736 * D;
  const Ldeg = q + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g);
  const e = 23.439 - 0.00000036 * D;
  const dec = Math.asin(Math.sin(e * RAD) * Math.sin(Ldeg * RAD)) / RAD;
  return { dec, lambda: ((Ldeg % 360) + 360) % 360, eps: e };
}
export const declination = (date: Date) => sunPosition(date).dec;

export const dms = (v: number, pos = 'N', neg = 'S') => {
  const a = Math.abs(v);
  let d = Math.floor(a);
  let m = Math.round((a - d) * 60);
  if (m === 60) {
    d += 1;
    m = 0;
  }
  return `${d}°${String(m).padStart(2, '0')}′${pos ? ` ${v >= 0 ? pos : neg}` : ''}`;
};
/** Signed degrees-minutes with a real minus sign, for the worked equation. */
export const sdm = (v: number) => `${v < 0 ? '−' : ''}${dms(Math.abs(v), '')}`;

/**
 * Lowest noon altitude the plate allows for a declination: the formula is for an observer north
 * of the sun, and latitude cannot pass 90° N, so h ≥ δ (plus a margin that keeps the drawing
 * readable near the pole).
 */
export const hMin = (dec: number) => Math.max(10, Math.ceil(dec + 4));
export const clampH = (h: number, dec: number) => Math.min(H_MAX, Math.max(hMin(dec), h));

export function noonScene(h: number, dec: number, W: number) {
  const narrow = W < 520;
  const phi = 90 - h + dec;
  const Re = narrow ? Math.min(W * 0.28, 100) : 112;
  const zLen = narrow ? 62 : 80;
  const cx = narrow ? W * 0.38 : W * 0.36;
  const cy = Re + zLen + 18; // room for the zenith line when you stand at the pole
  const HH = Math.round(cy + Re + 26);
  const P = (x: number, y: number, r: number, deg: number) => ({ x: x + r * Math.cos(deg * RAD), y: y - r * Math.sin(deg * RAD) });
  const obs = P(cx, cy, Re, phi);
  const zen = P(cx, cy, Re + zLen, phi);
  const L = narrow ? 58 : 76;
  const horS = P(obs.x, obs.y, L, phi - 90); // horizon towards the south (the sun's side)
  const horN = P(obs.x, obs.y, L * 0.7, phi + 90);
  const sunAt = P(obs.x, obs.y, L * 1.5, dec);
  // drawn angles, measured back from the drawn points
  const drawnH = (Math.atan2(-(sunAt.y - obs.y), sunAt.x - obs.x) - Math.atan2(-(horS.y - obs.y), horS.x - obs.x)) / RAD;
  const drawnPhi = Math.atan2(-(obs.y - cy), obs.x - cx) / RAD;
  const hArcR = L * 0.6;
  const phiArcR = Re * 0.42;
  const decArcR = Re * 0.68;
  const labels = {
    phi: { ...P(cx, cy, phiArcR + 12, phi / 2), text: 'φ' },
    dec: { ...P(cx, cy, decArcR + 11, dec / 2), text: 'δ' },
    // a low sun leaves no room inside the arc: put h just under the horizon line instead
    h: { ...(h >= 24 ? P(obs.x, obs.y, hArcR + 12, phi - 90 + h / 2) : P(obs.x, obs.y, hArcR + 2, phi - 90 - 11)), text: 'h' },
    zenith: { x: zen.x + 6, y: zen.y, text: 'zenith' },
    horizon: { x: horS.x + 4, y: horS.y + 14, text: 'horizon' },
    equator: { x: cx - Re + 6, y: cy - 6, text: 'equator' },
    pole: { x: cx + 4, y: cy - Re - 12, text: 'N pole' },
  };
  return { narrow, HH, phi, Re, cx, cy, obs, zen, L, horS, horN, sunAt, drawnH, drawnPhi, hArcR, phiArcR, decArcR, labels };
}
