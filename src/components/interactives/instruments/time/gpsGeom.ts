/**
 * Episode 9, GPS (time is distance): geometry for GpsPlate.tsx (swept by time.sweep.ts).
 *
 * A flat 2D picture around the receiver (you, at the origin; x east, y north, metres). A GPS
 * satellite is about 20,200 km up (Ashby), so within a few kilometres of you the sphere of points
 * at its measured range is a straight line, perpendicular to the direction to the satellite (the
 * sag over ±20 km is (20 km)²/(2·20 200 km) ≈ 10 m, under one pixel at that zoom; checked by the
 * sweep). Satellite i is in direction u_i; its line is { x : u_i·x = −δ_i }, where δ_i is how
 * much LONGER the measured range is than the truth (a longer range puts you farther away from
 * the satellite, i.e. shifted against u_i).
 *   δ_i = c·(b − b̂) + c·e_i
 * b: the receiver's clock error (s), b̂: the receiver's estimate of it (the 2D stand-in for the
 * fourth satellite's job), e_i: satellite i's own clock error (only the "uncorrected relativity"
 * step uses it: 38 µs a day, GPS World / Ashby).
 */
export const C = 299_792_458; // m/s, exact
export const ORBIT_KM = 20_200;
export const REL_PER_DAY = 38.6e-6; // s/day (Ashby: 4.4647 × 10⁻¹⁰ × 86 400)
export const B_MAX = 3e-6; // receiver clock error slider, ± seconds
export const B_STEP = 1e-8; // 10 ns
/** The lines "meet" when the triangle they form is under this size (m): 3 ns of light. */
export const MEET_TOL = 1;
export const SATS = [
  { id: 'A', az: 40 },
  { id: 'B', az: 150 },
  { id: 'C', az: 265 },
].map((s) => ({ ...s, u: { x: Math.cos((s.az * Math.PI) / 180), y: Math.sin((s.az * Math.PI) / 180) } }));

export type GpsState = { n: number; b: number; bHat: number; relHours: number };
export const offsets = (s: GpsState) =>
  SATS.slice(0, s.n).map((_, i) => C * (s.b - s.bHat) + (i === 0 ? C * REL_PER_DAY * (s.relHours / 24) : 0));

/** Intersection of lines i and j (u·x = −δ). */
export function meet(i: number, j: number, d: number[]) {
  const a = SATS[i].u;
  const b = SATS[j].u;
  const det = a.x * b.y - a.y * b.x;
  return { x: (-d[i] * b.y + d[j] * a.y) / det, y: (-a.x * d[j] + b.x * d[i]) / det };
}
export function fix(s: GpsState) {
  const d = offsets(s);
  if (s.n < 2) return { d, pts: [] as { x: number; y: number }[], size: Infinity, at: null as null | { x: number; y: number } };
  const pts = s.n === 2 ? [meet(0, 1, d)] : [meet(0, 1, d), meet(1, 2, d), meet(0, 2, d)];
  let size = 0;
  for (const p of pts) for (const q of pts) size = Math.max(size, Math.hypot(p.x - q.x, p.y - q.y));
  const at = { x: pts.reduce((a, p) => a + p.x, 0) / pts.length, y: pts.reduce((a, p) => a + p.y, 0) / pts.length };
  return { d, pts, size, at };
}

/** Zoom window half-width (m): fixed at 1.5 km unless something drawn needs more room. */
export function halfWidth(s: GpsState) {
  const f = fix(s);
  let need = 0;
  for (const p of f.pts) need = Math.max(need, Math.abs(p.x), Math.abs(p.y));
  for (const d of f.d) need = Math.max(need, Math.abs(d));
  const base = 1500;
  if (need * 1.35 <= base) return base;
  // round up to 1, 2, 5 × 10^k so the scale bar stays a round number
  const raw = need * 1.35;
  const k = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 5, 10]) if (m * k >= raw) return m * k;
  return raw;
}

/** The line u·x = −δ clipped to the square |x|,|y| ≤ hw (endpoints), or null if it misses. */
export function clipLine(u: { x: number; y: number }, delta: number, hw: number) {
  const p0 = { x: -delta * u.x, y: -delta * u.y };
  const t = { x: -u.y, y: u.x };
  let lo = -Infinity;
  let hi = Infinity;
  for (const [pc, tc] of [
    [p0.x, t.x],
    [p0.y, t.y],
  ]) {
    if (Math.abs(tc) < 1e-12) {
      if (Math.abs(pc) > hw) return null;
      continue;
    }
    const a = (-hw - pc) / tc;
    const b = (hw - pc) / tc;
    lo = Math.max(lo, Math.min(a, b));
    hi = Math.min(hi, Math.max(a, b));
  }
  if (lo >= hi) return null;
  return [
    { x: p0.x + lo * t.x, y: p0.y + lo * t.y },
    { x: p0.x + hi * t.x, y: p0.y + hi * t.y },
  ];
}

/** Scale bar length (m) for a window. */
export const scaleBar = (hw: number) => {
  const raw = hw / 2;
  const k = 10 ** Math.floor(Math.log10(raw));
  for (const m of [5, 2, 1]) if (m * k <= raw) return m * k;
  return k;
};
export const fmtM = (m: number) => (m >= 1000 ? `${+(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`);
