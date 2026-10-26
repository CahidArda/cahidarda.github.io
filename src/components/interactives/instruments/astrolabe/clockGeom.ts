/**
 * Plate 7, the astrolabe dial in a clock: the rete's state after `days` evenings at the same
 * sun time (swept by astrolabe.sweep.ts). Rates: USNO GMST formula (SIDEREAL_RATE sidereal per
 * solar day).
 */
import { SIDEREAL_RATE, eclRA, sunLambda, wrap360 } from './astro.ts';

const START = 80; // 21 March
const CLOCK_T = 21; // 21:00 sun time every evening

export const clockState = (days: number) => {
  const doy = ((START - 1 + days) % 365) + 1;
  const lam = sunLambda(doy);
  const H = (CLOCK_T - 12) * 15;
  const lst = wrap360(H + eclRA(lam));
  const lst0 = wrap360(H + eclRA(sunLambda(START)));
  const gain = days * (24 * 60) * (SIDEREAL_RATE - 1); // minutes of sidereal gain at the same sun time
  // the drawn rete follows the real sun, so its turn differs from the mean gain (gain/4 degrees)
  // by the equation of time, a few degrees; unwrap it around the mean so 365 days reads 360°, not 0°
  const mean = gain / 4;
  const turned = mean + (wrap360(lst - lst0 - mean + 180) - 180);
  return { doy, lam, H, lst, turned, gain };
};
