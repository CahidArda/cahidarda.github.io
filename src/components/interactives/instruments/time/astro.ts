/** Small spherical-astronomy helpers for the time post. Degrees in, degrees out. */
export const PHI = 41; // Istanbul's latitude, the one the 1753 and 1860 boards were made for
const R = Math.PI / 180;
export const sind = (d: number) => Math.sin(d * R);
export const cosd = (d: number) => Math.cos(d * R);
export const acosd = (x: number) => Math.acos(Math.max(-1, Math.min(1, x))) / R;

/** Approximate solar declination for day-of-year n (1..365), good to about half a degree. */
export const declination = (n: number) => 23.44 * Math.sin((2 * Math.PI * (n - 81)) / 365);

/** Half-day arc H0 (degrees): hour angle of sunset, with the centre of the sun on the horizon. */
export const halfDay = (dec: number, phi = PHI) => acosd(-Math.tan(phi * R) * Math.tan(dec * R));

/** Noon altitude of the sun. */
export const noonAltitude = (dec: number, phi = PHI) => 90 - phi + dec;

/** Hour angle (degrees, >= 0) for altitude h, or null if the sun never reaches h that day. */
export function hourAngle(h: number, dec: number, phi = PHI) {
  const c = (sind(h) - sind(phi) * sind(dec)) / (cosd(phi) * cosd(dec));
  if (c > 1 || c < -1) return null;
  return acosd(c);
}

/** Sun's altitude (degrees) at hour angle H (degrees, 0 = noon) on a day of declination dec. */
export const altitude = (Hdeg: number, dec: number, phi = PHI) =>
  Math.asin(
    Math.max(-1, Math.min(1, sind(phi) * sind(dec) + cosd(phi) * cosd(dec) * cosd(Hdeg))),
  ) / R;

/** Altitude at local solar time t (hours, 12 = noon). */
export const altitudeAt = (t: number, dec: number, phi = PHI) => altitude(15 * (t - 12), dec, phi);

/** Sunrise and sunset, local solar hours (centre of the sun on the horizon, no refraction). */
export const sunTimes = (dec: number, phi = PHI) => {
  const h0 = halfDay(dec, phi) / 15;
  return { rise: 12 - h0, set: 12 + h0 };
};

const MONTHS =['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
export function dayLabel(n: number) {
  let d = Math.round(n);
  for (let m = 0; m < 12; m++) {
    if (d <= DAYS[m]) return `${d} ${MONTHS[m]}`;
    d -= DAYS[m];
  }
  return '31 Dec';
}

/** "8:46" from hours (wraps into 0..24). */
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
