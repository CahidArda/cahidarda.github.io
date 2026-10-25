/**
 * Episode 8, the caesium clock: numbers for AtomicPlate.tsx (swept by time.sweep.ts).
 * Since 1967 the second is 9,192,631,770 periods of the caesium-133 ground-state hyperfine
 * transition (BIPM, 13th CGPM, Resolution 1). The clock tunes a microwave signal (made from a
 * quartz oscillator) until the most atoms flip, then holds it there.
 *
 * The resonance curve here is ILLUSTRATIVE: a Lorentzian of half-width GAMMA Hz (the plate says
 * so). How well the peak is held is set by real clocks' published figures:
 *   1955 Essen and Parry, NPL: about 1 s in 300 years (Science Museum Group)
 *   2014 NIST-F2 fountain: 1 s in about 300 million years (NIST via ScienceDaily)
 *   2024 JILA strontium optical clock: 8.1 × 10⁻¹⁹ systematic uncertainty (Aeppli et al.)
 */
export const F_CS = 9_192_631_770;
export const GAMMA = 150; // Hz, illustrative half-width
export const DET_MIN = -1000;
export const DET_MAX = 1000;
/** "On the peak" within this many Hz (a third of the half-width: 90% of the atoms flip). */
export const PEAK_TOL = 50;
export const flipFraction = (det: number) => 1 / (1 + (det / GAMMA) ** 2);
/** A free-running quartz (no lock) is off by this many Hz at 9.19 GHz: 1 part in 10⁷, the
 *  order of a good 1950s quartz oscillator (illustrative). */
export const QUARTZ_OFFSET = 919;
export const YEAR = 365.25;
export type Era = 'essen' | 'f2' | 'optical';
export const ERAS: { id: Era; label: string; perDay: number; say: string }[] = [
  { id: 'essen', label: '1955 caesium', perDay: 1 / (300 * YEAR), say: '1 s in 300 years' },
  { id: 'f2', label: '2014 fountain', perDay: 1 / (300e6 * YEAR), say: '1 s in 300 million years' },
  { id: 'optical', label: '2024 optical', perDay: 8.1e-19 * 86400, say: '8.1 × 10⁻¹⁹: under 1 s in the age of the universe' },
];
/** Seconds a day for a frequency offset (Hz) from F_CS. */
export const errFromDet = (det: number) => (86400 * det) / F_CS;
/** The 10-digit cycle count, grouped: "9 192 631 770". */
export const grouped = (n: number) => Math.round(n).toLocaleString('en-US').replace(/,/g, ' ');

/** Curve layout. */
export function curveFrame(W: number) {
  const H = 210;
  const m = { l: 18, r: 18, t: 34, b: 44 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const x = (det: number) => m.l + ((det - DET_MIN) / (DET_MAX - DET_MIN)) * iw;
  const y = (p: number) => m.t + ih - p * ih;
  return { H, m, iw, ih, x, y };
}
