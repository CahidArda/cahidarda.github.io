/** Pure maths of the alaturka plate (swept by a script). Hours are local solar time. */
import { declination, halfDay, hm } from './astro';

export const sunsetOf = (n: number) => 12 + halfDay(declination(n)) / 15;
export const sunriseOf = (n: number) => 12 - halfDay(declination(n)) / 15;
/** Reading of a 12-hour alaturka dial (0..12, where 0 shows as 12): hours since sunset. */
export const alaturka = (t: number, n: number) => (((t - sunsetOf(n)) % 12) + 12) % 12;
export const hm12 = (v: number) => hm(v < 1 ? v + 12 : v);
export const hm12f = (v: number) => hm12(v % 12);
/** Hour-hand angle on a 12-hour dial, maths convention (12 at the top, clockwise). */
export const dialAngle = (h: number) => 90 - ((h % 12) / 12) * 360;
/** Hours since the previous sunset (0..24): what the alaturka day counts. */
export const sinceSunset = (t: number, n: number) => {
  const d = t - sunsetOf(n);
  return d >= 0 ? d : d + 24;
};
