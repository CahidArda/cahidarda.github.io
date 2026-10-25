/**
 * Vitruvius's hodometer (De architectura 10.9): pure numbers and the plate's layout.
 * Distances in Roman feet; screen in SVG viewBox units, y down.
 */
export const WHEEL_FT = 4;
export const TEETH = 400;
export const VITRUVIUS_REV = 12.5;
export const REV = Math.PI * WHEEL_FT;
export const MILE_FT = 5000;
export function hodometer(ft: number) {
  const revs = ft / REV;
  const teeth = Math.floor(revs);
  const pebbles = Math.floor(teeth / TEETH);
  const counted = pebbles * MILE_FT; // what the bowl says
  const wheelDeg = (revs % 1) * 360;
  const drumDeg = ((teeth % TEETH) / TEETH) * 360;
  return { revs, teeth, pebbles, counted, wheelDeg, drumDeg };
}

/** Layout of the wheel, the drum and the bowl at drawing width W (kept on the plate at 326 px). */
export function hodoLayout(W: number) {
  const H = 220;
  const wc = { x: Math.min(110, W * 0.26), y: 120, r: Math.min(80, W * 0.2) };
  const dc = { x: wc.x + wc.r + Math.min(90, W * 0.22), y: 70, r: 44 };
  const ground = wc.y + wc.r;
  // the bowl and its label (26 below it) sit above the ground line
  const bowl = { x: Math.min(dc.x + Math.min(110, W * 0.26), W - 56), y: Math.min(160, ground - 40) };
  return { H, wc, dc, bowl, ground };
}
