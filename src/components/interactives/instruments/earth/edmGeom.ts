/**
 * Episode 7: distance by light (EDM). The instrument at A sends light modulated at f to a
 * reflector at C and back. The phase of the returning modulation, compared with the outgoing,
 * gives the fraction of a "half wavelength" λ/2 = c / (2f); a second, coarse frequency fixes
 * how many whole half wavelengths N fit.  D = (N + Δφ/360°) · λ/2.
 * Fine: 10 MHz → λ/2 ≈ 14.99 m. Coarse: 100 kHz → λ/2 ≈ 1499 m (one cycle covers the strait).
 * Air: the speed of light in air is c/n; n − 1 ≈ 2.8×10⁻⁴ (standard air, an approximation).
 * The distance is the strait at the narrows, A to C, 700 m (site.ts).
 */
import { SITE } from './site.ts';

export const C0 = 299_792_458; // m/s
export const N_AIR = 1.00028; // standard air (approx.)
export const F_FINE = 10e6;
export const F_COARSE = 100e3;
export const halfLambda = (f: number, n = N_AIR) => C0 / n / f / 2;
export const D_TRUE = SITE.WIDTH; // 700 m
/** Phase (degrees, 0..360) of the returned modulation at frequency f. */
export const phaseOf = (f: number, D = D_TRUE) => {
  const cyc = D / halfLambda(f);
  return (cyc - Math.floor(cyc)) * 360;
};
export const N_TRUE = Math.floor(D_TRUE / halfLambda(F_FINE)); // 46
export const PHI_FINE = phaseOf(F_FINE); // degrees
export const PHI_COARSE = phaseOf(F_COARSE);
/** The coarse estimate of the distance (m), from the coarse phase alone. */
export const coarseD = (phiDeg: number) => (phiDeg / 360) * halfLambda(F_COARSE);
export const PHASE_STEP = 0.5; // degrees, the phase meter's dial
export const phaseOk = (read: number, truth: number) => {
  const d = Math.abs(((read - truth + 540) % 360) - 180);
  return d <= PHASE_STEP / 2 + 1e-9;
};
export const distance = (N: number, phiDeg: number) => (N + phiDeg / 360) * halfLambda(F_FINE);
/** Error from assuming the wrong air: the ppm term. 1 °C of air is about 1 ppm. */
export const PPM_PER_C = 1.0;

/** The phase meter: outgoing and returning sine waves and the phase dial. */
export function phaseScene(W: number, phiTrue: number, phiRead: number) {
  const H = 190;
  const pad = 16;
  const wy = [52, 100]; // outgoing, returning wave centre lines
  const amp = 14;
  const wl = Math.max(80, (W - 2 * pad) / 2.5); // px per cycle
  const wave = (y: number, shift: number) => {
    const pts: string[] = [];
    for (let x = pad; x <= W - pad; x += 2) {
      const ph = ((x - pad) / wl) * 2 * Math.PI - (shift * Math.PI) / 180;
      pts.push(`${x === pad ? 'M' : 'L'} ${x} ${(y - amp * Math.sin(ph)).toFixed(2)}`);
    }
    return pts.join(' ');
  };
  // the reader shifts the reference wave by phiRead; when it matches, the two coincide
  const out = wave(wy[0], phiRead);
  const ret = wave(wy[1], phiTrue);
  const peakOut = pad + (phiRead / 360) * wl + wl / 4;
  const peakRet = pad + (phiTrue / 360) * wl + wl / 4;
  return { W, H, pad, wy, amp, wl, out, ret, peakOut, peakRet };
}

/** Plan of the strait with the beam and the fine half-wavelength "tape" marks. */
export function beamScene(W: number, nShown: number) {
  const H = 120;
  const pad = 20;
  const k = (W - 2 * pad) / D_TRUE;
  const y = 60;
  const hl = halfLambda(F_FINE);
  const marks = Array.from({ length: Math.min(N_TRUE, nShown) + 1 }, (_, i) => pad + i * hl * k);
  const A = pad;
  const Cx = pad + D_TRUE * k;
  return { W, H, pad, k, y, marks, A, Cx, hl };
}
