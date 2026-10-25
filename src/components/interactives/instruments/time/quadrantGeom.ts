/**
 * Pure geometry of the sine-quadrant plate, so the drawing and the numbers come from one place
 * (and can be swept by a script). Board coordinates: apex at (AX, AY), y down, the gridded
 * quarter has radius R = 60 parts. The thread at altitude h hangs at maths angle (h − 90°).
 *
 * The board only handles magnitudes 0..60. The muvakkit's rule for the date term
 * sin φ sin δ: subtract it from sin h when the sun is north of the equator, ADD its size when
 * the sun is south (sin h = sin φ sin δ + cos φ cos δ cos H, so cos H grows when δ < 0). On the
 * drawing that is a brass segment starting at the thread's reading and running left
 * (subtract) or right (add); it never leaves the grid, because the sum is at most cos φ cos δ.
 */
import { PHI, acosd, cosd, declination, noonAltitude, sind } from './astro';

export const W = 440;
export const H = 440;
export const BAND = 22;
export const AX = 30 + BAND;
export const AY = 22 + BAND;
export const R = 350;
export const U = R / 60;

export const polarPt = (r: number, deg: number) => ({
  x: AX + r * Math.cos((deg * Math.PI) / 180),
  y: AY - r * Math.sin((deg * Math.PI) / 180),
});
/** Where the thread meets radius r, for altitude h. */
export const threadAt = (h: number, r = R) => polarPt(r, h - 90);

/** Highest altitude the slider may take on a day: the noon altitude, to 0.1°, rounded down. */
export const maxAltitude = (day: number) => Math.floor(noonAltitude(declination(day)) * 10) / 10;

export function quadrant(hIn: number, day: number) {
  const dec = declination(day);
  const hMax = maxAltitude(day);
  const h = Math.max(0, Math.min(hIn, hMax));
  const sinh = sind(h);
  const a = sind(PHI) * sind(dec); // the date term, signed
  const b = cosd(PHI) * cosd(dec); // the divisor
  const diff = sinh - a; // what the brass segment ends on
  const cosH = Math.max(-1, Math.min(1, diff / b));
  const Hdeg = acosd(cosH);
  const tip = threadAt(h);
  const xRead = AX + 60 * sinh * U; // 60·sin h on the top scale (same x as the rim crossing)
  const xDiffRaw = AX + 60 * diff * U;
  const diffClipped = diff < 0; // sun below the "6 o'clock" altitude: past the apex
  const xDiff = Math.max(AX, xDiffRaw);
  const xA = AX + 60 * Math.abs(a) * U; // the date term as a length from the apex
  const xAbs = AX + 60 * Math.abs(diff) * U; // |difference|: where the bead must sit
  // the brass shift on its row: from the sine reading to the difference (or, past the apex,
  // the segment a − sin h, whose length is |difference|)
  const shift: [number, number] = diff >= 0 ? [xRead, xDiff] : [xRead, xA];
  const xCos = AX + 60 * Math.abs(cosH) * U; // 60·|cos H| on the top scale
  // read H back: the rim point under 60·|cos H| sits at (90° − H') from the left edge
  const Hp = acosd(Math.abs(cosH)); // = H, or 180° − H when cos H < 0
  const rimH = threadAt(90 - Hp);
  const theta = 90 - Hp; // dividing thread, from the left edge
  const bead = polarPt(60 * b * U, theta - 90);
  const divTip = threadAt(theta);
  return { dec, hMax, h, sinh, a, b, diff, cosH, Hdeg, tip, xRead, xDiff, xDiffRaw, diffClipped, xA, xAbs, shift, xCos, Hp, rimH, theta, bead, divTip };
}

/** Approximate text box (SVG units) for collision checks: mono ~0.62em per char. */
export const textBox = (x: number, y: number, s: string, size: number, anchor: 'start' | 'middle' | 'end') => {
  const w = s.length * size * 0.62;
  const x0 = anchor === 'start' ? x : anchor === 'middle' ? x - w / 2 : x - w;
  return { x0, x1: x0 + w, y0: y - size * 0.8, y1: y + size * 0.25 };
};

/** Every label the plate draws, with its anchor, so the sweep can check overlaps. */
export function quadrantLabels(q: ReturnType<typeof quadrant>) {
  // keep each label inside the quarter at its own height
  const clampX = (x: number, s: string, size: number, y: number) => {
    const w = s.length * size * 0.62;
    const right = AX + Math.sqrt(R * R - (y - AY) ** 2) - 6;
    return Math.max(AX + 4 + w / 2, Math.min(right - w / 2, x));
  };
  const sinTxt = (60 * q.sinh).toFixed(1);
  const dTxt =
    q.diff >= 0
      ? `${q.a >= 0 ? '−' : '+'} ${(60 * Math.abs(q.a)).toFixed(1)}`
      : `${(60 * q.a).toFixed(1)} − ${(60 * q.sinh).toFixed(1)}`;
  const cTxt = `${(60 * Math.abs(q.cosH)).toFixed(1)}`;
  const midD = (q.shift[0] + q.shift[1]) / 2;
  return {
    sin: { x: clampX(q.xRead, sinTxt, 11.5, AY + 17), y: AY + 17, s: sinTxt, size: 11.5 },
    date: { x: clampX(midD, dTxt, 11, AY + 47), y: AY + 47, s: dTxt, size: 11 },
    cos: { x: clampX(q.xCos, cTxt, 11.5, AY + 64), y: AY + 64, s: cTxt, size: 11.5 },
  };
}
/** Rows: both readings are dots on the top scale; the date shift runs along its own row. */
export const ROW = { sin: AY, date: AY + 31, cos: AY };
