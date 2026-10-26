/**
 * Spherical astronomy for the astrolabe post. Pure, degrees in and out.
 *
 * Conventions (used by every plate in this folder):
 *   φ  latitude (Istanbul plate: 41° N)        δ  declination        h  altitude
 *   H  hour angle, degrees, 0 on the upper (south) meridian, + to the WEST (afternoon)
 *   λ  the sun's ecliptic longitude            α  right ascension    ε  obliquity
 *   LST (local sidereal time, degrees) = H + α. On the astrolabe it is the rete's turn.
 *   Local apparent (sun) time t = 12 h + H / 15°.
 * Sun position: USNO "approximate solar coordinates" (about 1′), https://aa.usno.navy.mil/faq/sun_approx
 */
export const PHI = 41;
export const EPS = 23.44;
export const YEAR = 2026;
const R = Math.PI / 180;
export const sind = (d: number) => Math.sin(d * R);
export const cosd = (d: number) => Math.cos(d * R);
export const tand = (d: number) => Math.tan(d * R);
export const asind = (x: number) => Math.asin(Math.max(-1, Math.min(1, x))) / R;
export const acosd = (x: number) => Math.acos(Math.max(-1, Math.min(1, x))) / R;
export const atan2d = (y: number, x: number) => Math.atan2(y, x) / R;
/** Wrap to (−180, 180]. */
export const wrap180 = (a: number) => {
  const x = ((((a + 180) % 360) + 360) % 360) - 180;
  return x === -180 ? 180 : x;
};
export const wrap360 = (a: number) => ((a % 360) + 360) % 360;

/** JD at 12:00 UT on day-of-year `doy` of YEAR (2026-01-01 00:00 UT = JD 2461041.5). */
export const jdOf = (doy: number) => 2461041.5 + (doy - 1) + 0.5;

/** The sun's ecliptic longitude λ (degrees, 0..360) on day `doy`, at noon. */
export function sunLambda(doy: number) {
  const D = jdOf(doy) - 2451545.0;
  const g = 357.529 + 0.98560028 * D;
  const q = 280.459 + 0.98564736 * D;
  return wrap360(q + 1.915 * sind(g) + 0.02 * sind(2 * g));
}
/** Declination of a point on the ecliptic. */
export const eclDec = (lam: number) => asind(sind(EPS) * sind(lam));
/** Right ascension of a point on the ecliptic (0..360). */
export const eclRA = (lam: number) => wrap360(atan2d(cosd(EPS) * sind(lam), cosd(lam)));

/** Altitude of a body at declination δ, hour angle H, from latitude φ. */
export const altitude = (dec: number, H: number, phi = PHI) =>
  asind(sind(phi) * sind(dec) + cosd(phi) * cosd(dec) * cosd(H));

/** Azimuth measured from SOUTH, + towards WEST (degrees, −180..180). */
export function azimuthS(dec: number, H: number, phi = PHI) {
  // local frame: x west, y south, z up
  const x = cosd(dec) * sind(H);
  const y = sind(phi) * cosd(dec) * cosd(H) - cosd(phi) * sind(dec);
  return atan2d(x, y);
}

/** cos H for altitude h; |value| > 1 means the sun never reaches h that day. */
export const cosHour = (h: number, dec: number, phi = PHI) =>
  (sind(h) - sind(phi) * sind(dec)) / (cosd(phi) * cosd(dec));
/** |H| (0..180) for altitude h, or null when the sun never reaches h that day. */
export function hourAngle(h: number, dec: number, phi = PHI) {
  const c = cosHour(h, dec, phi);
  if (c > 1 + 1e-12 || c < -1 - 1e-12) return null;
  return acosd(c);
}
/** Half-day arc (sunset hour angle, centre of the sun on the horizon). */
export const halfDay = (dec: number, phi = PHI) => acosd(-tand(phi) * tand(dec));
export const noonAlt = (dec: number, phi = PHI) => 90 - phi + dec;

/** Sun time (hours) from hour angle. */
export const timeOf = (H: number) => 12 + H / 15;
/** Hour angle from sun time. */
export const hourOf = (t: number) => (t - 12) * 15;

/** "8:46" (wraps into 0..24); minutes rounded. */
export function hm(hours: number) {
  let t = ((hours % 24) + 24) % 24;
  let h = Math.floor(t);
  let m = Math.round((t - h) * 60);
  if (m === 60) {
    h = (h + 1) % 24;
    m = 0;
  }
  t = h;
  return `${t}:${String(m).padStart(2, '0')}`;
}
/** Signed minutes, e.g. "+12 min", "−3 min", "0 min". */
export const dmin = (hours: number) => {
  const m = Math.round(hours * 60);
  return `${m > 0 ? '+' : m < 0 ? '−' : ''}${Math.abs(m)} min`;
};
/** Degrees with one decimal and a real minus sign. */
export const deg1 = (v: number) => `${v < -0.049 ? '−' : ''}${Math.abs(v).toFixed(1)}°`;
export const deg0 = (v: number) => `${v < -0.49 ? '−' : ''}${Math.abs(v).toFixed(0)}°`;
export const num3 = (v: number) => `${v < -0.0005 ? '−' : ''}${Math.abs(v).toFixed(3)}`;

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
export function dayLabel(doy: number) {
  let d = Math.round(doy);
  for (let m = 0; m < 12; m++) {
    if (d <= MDAYS[m]) return `${d} ${MONTHS[m]}`;
    d -= MDAYS[m];
  }
  return '31 Dec';
}
/** Day-of-year of a month (0-based) and day. */
export const doyOf = (m: number, d: number) => MDAYS.slice(0, m).reduce((a, b) => a + b, 0) + d;

const SIGNS = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'];
/** "7° Scorpio" style: the sign and degree, as the back of an astrolabe gives them. */
export function signOf(lam: number) {
  const l = wrap360(lam);
  const i = Math.floor(l / 30) % 12;
  return { sign: SIGNS[i], deg: l - 30 * i, i };
}
export const SIGN_NAMES = SIGNS;

/** Stars on the rete. J2000 positions (precession since then is under 0.4°, invisible here). */
export const STARS = [
  { id: 'vega', name: 'Vega', ra: 279.23, dec: 38.78 },
  { id: 'altair', name: 'Altair', ra: 297.7, dec: 8.87 },
  { id: 'aldebaran', name: 'Aldebaran', ra: 68.98, dec: 16.51 },
  { id: 'arcturus', name: 'Arcturus', ra: 213.92, dec: 19.18 },
  { id: 'capella', name: 'Capella', ra: 79.17, dec: 46.0 },
  { id: 'regulus', name: 'Regulus', ra: 152.09, dec: 11.97 },
  { id: 'spica', name: 'Spica', ra: 201.3, dec: -11.16 },
] as const;

/** Sun for a date + sun time: λ, δ, α, H, h, LST. */
export function sunAt(doy: number, t: number, phi = PHI) {
  const lam = sunLambda(doy);
  const dec = eclDec(lam);
  const ra = eclRA(lam);
  const H = hourOf(t);
  return { lam, dec, ra, H, h: altitude(dec, H, phi), lst: wrap360(H + ra) };
}

/** Moments for "when you step outside" (sun time). All have the sun up at 41° N all year. */
export const MOMENTS = [
  { id: 'm1', label: 'mid-morning', t: 9 + 10 / 60 },
  { id: 'm2', label: 'late morning', t: 10 + 40 / 60 },
  { id: 'm3', label: 'early afternoon', t: 13 + 25 / 60 },
  { id: 'm4', label: 'late afternoon', t: 15 + 5 / 60 },
] as const;
export type MomentId = (typeof MOMENTS)[number]['id'];
export const DOY_DEFAULT = 299; // 26 October

/** Equation of time on day `doy` (minutes, apparent − mean sun time), from the same USNO
 *  formulae: EoT = q/15 − α/15 (hours). Good to about a minute; enough for this plate. */
export function eotMinutes(doy: number) {
  const D = jdOf(doy) - 2451545.0;
  const q = wrap360(280.459 + 0.98564736 * D);
  const ra = eclRA(sunLambda(doy));
  return wrap180(q - ra) * 4;
}
/** Istanbul: 28.98° E, clocks on UTC+3 (45° E), so mean sun time runs 64 min behind the clock. */
export const IST_LON = 28.978;
export const ZONE_LON = 45;
export const lonMinutes = (ZONE_LON - IST_LON) * 4;
/** Sidereal rate from the USNO GMST formula (factor 1.0027379 per unit of solar time). */
export const SIDEREAL_RATE = 1.0027379;

/** Font size for a one-line italic note centred under a drawing W wide: the base size, shrunk
 *  (down to 9.5) when the text would run off the edges (serif italic ≈ 0.5 em per character). */
export const noteSize = (text: string, W: number, base = 12) =>
  Math.max(9.5, Math.min(base, (W - 16) / (Math.max(1, [...text].length) * 0.5)));
