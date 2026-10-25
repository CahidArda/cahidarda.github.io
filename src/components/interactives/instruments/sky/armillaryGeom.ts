/**
 * Geometry for the "why the sun's height changes" plate: the celestial sphere drawn as an
 * armillary, in an orthographic view. Pure (no React) so a sweep script can check it.
 *
 * Equatorial frame: x towards the March equinox, z towards the north celestial pole. The sun at
 * ecliptic longitude λ sits at (cos λ, sin λ cos ε, sin λ sin ε), so sin δ = sin ε · sin λ.
 * Screen: y down. Points on the far side of the sphere are drawn dashed.
 */
import { sunPosition, dateOf } from './noonGeom.ts';

const RAD = Math.PI / 180;
type V3 = [number, number, number];
export type Pt = { x: number; y: number };

const VIEW_EL = 18; // degrees above the equator
const VIEW_AZ = 270; // the viewer faces the December solstice; the equinoxes sit on the rim
const v: V3 = [Math.cos(VIEW_EL * RAD) * Math.cos(VIEW_AZ * RAD), Math.cos(VIEW_EL * RAD) * Math.sin(VIEW_AZ * RAD), Math.sin(VIEW_EL * RAD)];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / l, a[1] / l, a[2] / l];
};
const UP = unit([-v[2] * v[0], -v[2] * v[1], 1 - v[2] * v[2]]); // z minus its part along v
const RIGHT = cross(UP, v);

export const eclPoint = (lam: number, eps: number): V3 => [
  Math.cos(lam * RAD),
  Math.sin(lam * RAD) * Math.cos(eps * RAD),
  Math.sin(lam * RAD) * Math.sin(eps * RAD),
];
export const eqPoint = (ra: number): V3 => [Math.cos(ra * RAD), Math.sin(ra * RAD), 0];
/** Point on the hour circle at right ascension ra, declination d. */
export const hourPoint = (ra: number, d: number): V3 => [
  Math.cos(d * RAD) * Math.cos(ra * RAD),
  Math.cos(d * RAD) * Math.sin(ra * RAD),
  Math.sin(d * RAD),
];

export function armillaryScene(doy: number, W: number) {
  const narrow = W < 520;
  const R = Math.min(narrow ? W * 0.36 : W * 0.3, 160);
  const HH = Math.round(2 * R + 84);
  const cx = W / 2;
  const cy = R + 26;
  const P = (p: V3): Pt & { front: boolean } => ({
    x: cx + R * dot(p, RIGHT),
    y: cy - R * dot(p, UP),
    front: dot(p, v) >= -1e-9,
  });
  /** Split a sampled 3D curve into front (solid) and back (dashed) polylines. */
  const curve = (f: (t: number) => V3, t0: number, t1: number, n = 120) => {
    const front: string[] = [];
    const back: string[] = [];
    let cur: string[] = [];
    let curFront: boolean | null = null;
    for (let i = 0; i <= n; i++) {
      const p = P(f(t0 + ((t1 - t0) * i) / n));
      if (curFront !== null && p.front !== curFront) {
        cur.push(`${p.x.toFixed(1)} ${p.y.toFixed(1)}`);
        (curFront ? front : back).push('M ' + cur.join(' L '));
        cur = [];
      }
      curFront = p.front;
      cur.push(`${p.x.toFixed(1)} ${p.y.toFixed(1)}`);
    }
    if (cur.length > 1) (curFront ? front : back).push('M ' + cur.join(' L '));
    return { front: front.join(' '), back: back.join(' ') };
  };

  const { dec, lambda, eps } = sunPosition(dateOf(doy));
  const sun3 = eclPoint(lambda, eps);
  const ra = Math.atan2(sun3[1], sun3[0]) / RAD;
  const sun = P(sun3);
  const foot = P(eqPoint(ra)); // where the sun's hour circle meets the equator
  const equator = curve((t) => eqPoint(t), 0, 360);
  const ecliptic = curve((t) => eclPoint(t, eps), 0, 360);
  const decArc = curve((t) => hourPoint(ra, t), 0, dec, 40);
  const lamArc = curve((t) => eclPoint(t, eps), 0, lambda, 90);
  const epsArc = curve((t) => hourPoint(270, t), 0, -eps, 30); // at the December solstice, facing us
  const pole = P([0, 0, 1]);
  const south = P([0, 0, -1]);
  const march = P([1, 0, 0]);
  const sept = P([-1, 0, 0]);
  const june = P(eclPoint(90, eps));
  const dec21 = P(eclPoint(270, eps));
  const decMid = P(hourPoint(ra, dec / 2));
  const epsMid = P(hourPoint(270, -eps / 2));

  // labels: fixed ones first, then δ and the sun's name, each taking the first free spot
  type Lab = { id: string; x: number; y: number; text: string; size: number; anchor: 'start' | 'middle' | 'end' };
  const box = (l: Lab) => {
    const w = l.text.length * l.size * 0.52;
    const x0 = l.anchor === 'start' ? l.x : l.anchor === 'end' ? l.x - w : l.x - w / 2;
    return { x0, x1: x0 + w, y0: l.y - l.size * 0.8, y1: l.y + l.size * 0.2 };
  };
  const hit = (a: Lab, b: Lab) => {
    const p = box(a);
    const q = box(b);
    return p.x0 < q.x1 + 2 && q.x0 < p.x1 + 2 && p.y0 < q.y1 + 2 && q.y0 < p.y1 + 2;
  };
  const fits = (l: Lab) => {
    const b = box(l);
    return b.x0 >= 4 && b.x1 <= W - 4 && b.y0 >= 4 && b.y1 <= HH - 4;
  };
  const legendY = HH - 26; // two rows, bottom left
  const labels: Lab[] = [
    { id: 'pole', x: pole.x + 8, y: pole.y - 6, text: 'N celestial pole', size: 11, anchor: 'start' },
    { id: 'march', x: march.x + 6, y: march.y + 4, text: 'Mar 20', size: 11, anchor: 'start' },
    { id: 'sept', x: sept.x - 6, y: sept.y + 4, text: 'Sep 23', size: 11, anchor: 'end' },
    { id: 'eps', x: epsMid.x - 7, y: epsMid.y + 4, text: 'ε 23.44°', size: 12, anchor: 'end' },
    { id: 'legend-eq', x: 36, y: legendY, text: 'celestial equator', size: 11, anchor: 'start' },
    { id: 'legend-ecl', x: 36, y: legendY + 15, text: "ecliptic, the sun's path", size: 11, anchor: 'start' },
  ];
  const sunBox: Lab = { id: 'sun-disc', x: sun.x, y: sun.y + 7, text: 'oo', size: 18, anchor: 'middle' }; // ±9 px
  // when the sun passes a fixed label (an equinox date, ε) that label steps aside: the sun's
  // own label and the date slider say the same thing at that moment
  for (let i = labels.length - 1; i >= 0; i--) if (hit(labels[i], sunBox)) labels.splice(i, 1);
  const place = (cands: Lab[]) => {
    const l = cands.find((c) => fits(c) && !labels.some((o) => hit(c, o)) && (c.id === 'sun' || !hit(c, sunBox)));
    if (l) labels.push(l);
    return l;
  };
  const showDec = Math.abs(dec) > 1.5;
  if (showDec)
    place([12, -12, 22, -22, 34, -34].map((dx) => ({ id: 'dec', x: decMid.x + dx, y: decMid.y + 5, text: 'δ', size: 15, anchor: dx > 0 ? 'start' : 'end' })));
  const up = dec >= 0 ? -1 : 1;
  place([
    { id: 'sun', x: sun.x, y: sun.y + (up < 0 ? -13 : 21), text: 'sun', size: 12, anchor: 'middle' },
    { id: 'sun', x: sun.x + 13, y: sun.y + 4, text: 'sun', size: 12, anchor: 'start' },
    { id: 'sun', x: sun.x - 13, y: sun.y + 4, text: 'sun', size: 12, anchor: 'end' },
    { id: 'sun', x: sun.x, y: sun.y + (up < 0 ? 21 : -13), text: 'sun', size: 12, anchor: 'middle' },
  ]);
  const legend = { x: [10, 30] as const, eq: legendY - 4, ecl: legendY + 11 };
  return {
    narrow, R, HH, cx, cy, dec, lambda, eps, ra, sun, foot, equator, ecliptic, decArc, lamArc, epsArc,
    pole, south, march, sept, june, dec21, legend, showDec, labels,
    // drawn δ, measured back from the drawing's 3D source (for the sweep)
    drawnDec: Math.asin(sun3[2]) / RAD,
  };
}
