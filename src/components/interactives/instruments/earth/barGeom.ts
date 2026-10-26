/**
 * Episode 6: the baseline and temperature.
 *
 * Plain bar: a steel bar of nominal length L0 = 10 ft (3.048 m), true at T0 = 15 °C, grows by
 * α_steel·L0·ΔT. Laid N times along the baseline, the error is N times that.
 *
 * Colby's compensation bar (principle; dimensions illustrative): an iron bar and a brass bar of
 * the same length lie side by side a gap s apart, fixed together at their centres. Steel
 * tongues cross both ends, pivoted on each bar. Warming moves the brass end out by
 * α_b·(L/2)·ΔT and the iron end by α_i·(L/2)·ΔT, so each tongue tilts. A point on the tongue
 * beyond the iron bar, a distance x outside it, moves by
 *   u(x) = (L/2)·ΔT·(α_i − x·(α_b − α_i)/s)
 * which is zero at x* = s·α_i / (α_b − α_i). With α_b : α_i about 3 : 2, x* = 2s.
 * The two dots at x* on the two tongues then stay the same distance apart: the bar's length.
 * Coefficients: iron 12×10⁻⁶ /°C, brass 18×10⁻⁶ /°C (ratio 3:2), steel 11.5×10⁻⁶ /°C.
 * Screen: SVG viewBox units, y down; expansions drawn ×EX (said on the drawing).
 */
export const ALPHA = { steel: 11.5e-6, iron: 12e-6, brass: 18e-6 };
export const L0 = 3.048; // m, 10 ft
export const T0 = 15; // °C, the bar's standard temperature
export const T = { MIN: -5, MAX: 40, STEP: 1, START: 30 };
export const BASELINE = 200; // m: the site's baseline A–B
export const BARS = Math.round(BASELINE / L0); // 66 bar lengths (65.6, rounded for the drawing)
export const X_STAR = ALPHA.iron / (ALPHA.brass - ALPHA.iron); // in gap widths: 2

/** Error of the whole baseline with plain steel bars at temperature t (m, + = bars too long, base read short). */
export const plainErr = (t: number) => BASELINE * ALPHA.steel * (t - T0);
/** The baseline as read with plain bars (m): the bars are longer, so fewer fit: reads short. */
export const plainRead = (t: number) => BASELINE / (1 + ALPHA.steel * (t - T0));
/** Movement of a point on the tongue, x gap-widths beyond the iron bar (m), per end. */
export const tongueMove = (x: number, t: number) => (L0 / 2) * (t - T0) * (ALPHA.iron - x * (ALPHA.brass - ALPHA.iron));
/** The compensated bar's length (between the two dots at X_STAR): constant. */
export const compLen = (t: number) => L0 + 2 * tongueMove(X_STAR, t);

/** The compensation bar, drawn from above, with expansions ×EX. */
export function compScene(W: number, t: number, dotX: number) {
  const H = 210;
  const pad = 24;
  const EX = 400; // expansion exaggeration
  const kx = (W - 2 * pad - 130) / L0;
  const cx = W / 2;
  const s = 26; // px between the bars' centre lines (the gap)
  const yBrass = 70;
  const yIron = yBrass + s;
  const dT = t - T0;
  const half = (L0 / 2) * kx;
  const eB = (L0 / 2) * ALPHA.brass * dT * kx * EX;
  const eI = (L0 / 2) * ALPHA.iron * dT * kx * EX;
  const ends = [-1, 1].map((side) => {
    const b = { x: cx + side * (half + eB), y: yBrass };
    const i = { x: cx + side * (half + eI), y: yIron };
    // the tongue continues past the iron bar; the point x gap-widths beyond it
    const ux = (i.x - b.x) / s;
    const at = (xg: number) => ({ x: i.x + ux * xg * s, y: yIron + xg * s });
    const dot = at(X_STAR);
    const probe = at(dotX);
    const tip = at(2.6);
    return { side, b, i, dot, probe, tip };
  });
  const rest = [-1, 1].map((side) => cx + side * half); // where the dots sit at T0
  const probeMove = tongueMove(dotX, t);
  return { W, H, pad, EX, kx, cx, s, yBrass, yIron, half, ends, rest, probeMove };
}

/** The baseline with N plain bars: where the read end lands vs the true end, ×EX2. */
export function baseScene(W: number, t: number, compensated: boolean) {
  const H = 120;
  const pad = 20;
  const EX2 = 500;
  const k = (W - 2 * pad - 70) / BASELINE;
  const y = 54;
  const err = compensated ? 0 : BASELINE - plainRead(t); // m, + = the read base falls short; drawn ×EX2
  const trueEnd = pad + BASELINE * k;
  const readEnd = trueEnd - err * k * EX2;
  return { W, H, pad, EX2, k, y, err, trueEnd, readEnd };
}
