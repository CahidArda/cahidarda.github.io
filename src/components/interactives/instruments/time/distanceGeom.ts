/** Pure layout of the clock-error plate (swept by a script). */
export const C_LIGHT = 299_792_458; // m/s, exact by definition of the metre
export const EARTH_EQ = 40_075_017 / 86_400; // m/s: equator circumference over a solar day ≈ 463.8
export const LMIN = -9.5;
export const LMAX = 3.3;
export const D0 = -3;
export const D1 = 12;
export const ROWS = { ship: 46, radio: 112, axis: 146 };

export function distanceLayout(s: number, W: number) {
  const narrow = W < 460;
  const m = { l: 14, r: 14 };
  const iw = W - m.l - m.r;
  const x = (d: number) => m.l + ((Math.log10(Math.max(d, 10 ** D0)) - D0) / (D1 - D0)) * iw;
  const size = narrow ? 12 : 13;
  /** Rough width of a row's text block (serif label above, mono value below). */
  const textW = (label: string, value: string) =>
    Math.max(label.length * (size - 1) * 0.5, value.length * size * 0.62);
  /** Put the text on the side of the end-point where it fits; prefer the side that does
   *  not cover `avoid` (another row's drop line). */
  const row = (y: number, d: number, label: string, value: string, avoid?: number) => {
    const xe = x(d);
    const w = textW(label, value);
    const fitsR = xe + 8 + w <= W - 2;
    const fitsL = xe - 8 - w >= 2;
    const coversR = avoid !== undefined && avoid > xe && avoid < xe + 8 + w;
    const coversL = avoid !== undefined && avoid < xe && avoid > xe - 8 - w;
    let right = fitsR;
    if (fitsR && fitsL) right = coversR && !coversL ? false : !(xe > m.l + iw * 0.62 && !coversL);
    if (!fitsR && !fitsL) right = xe < W / 2;
    const tx = right ? xe + 8 : xe - 8;
    const anchor: 'start' | 'end' = right ? 'start' : 'end';
    return { xe, y, off: d < 10 ** D0, tx, anchor, label, value, size, w };
  };
  return { narrow, m, iw, x, row };
}

/** The ship's dashed drop line, broken where it would run through the radio row's text. */
export function dropSegments(xe: number, yFrom: number, yTo: number, block?: { x0: number; x1: number; y0: number; y1: number }) {
  if (!block || xe < block.x0 - 3 || xe > block.x1 + 3) return [[yFrom, yTo]];
  return [
    [yFrom, block.y0 - 3],
    [block.y1 + 3, yTo],
  ].filter(([a, b]) => b > a);
}
/** Text block of a row (label at y−10, value at y+14). */
export const rowBlock = (r: { tx: number; anchor: 'start' | 'end'; w: number; y: number; size: number }) => {
  const x0 = r.anchor === 'start' ? r.tx : r.tx - r.w;
  return { x0, x1: x0 + r.w, y0: r.y - 10 - r.size * 0.8, y1: r.y + 14 + r.size * 0.25 };
};
