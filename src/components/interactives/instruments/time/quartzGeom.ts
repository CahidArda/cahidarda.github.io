/**
 * Episode 7, the quartz watch: numbers for QuartzPlate.tsx (swept by time.sweep.ts).
 * A tuning-fork crystal rings at 32,768 Hz = 2¹⁵; a chain of 15 flip-flops, each halving the
 * frequency, turns that into one pulse a second (Wikipedia, Quartz clock). The fork is cut so its
 * frequency peaks near 25 °C and falls off as a parabola: Δf/f = −0.04 ppm/°C² · (T − 25)²
 * (Wikipedia, Crystal oscillator). Seconds per day = 86 400 · Δf/f.
 */
export const F0 = 32768;
export const STAGES = 15;
export const TURNOVER = 25;
export const PARABOLIC = -0.04e-6; // per °C²
export const T_MIN = -10;
export const T_MAX = 50;
/** Frequency after k halvings. */
export const stageFreq = (k: number) => F0 / 2 ** k;
export const fracShift = (tempC: number) => PARABOLIC * (tempC - TURNOVER) ** 2;
export const errPerDay = (tempC: number) => 86400 * fracShift(tempC);
export const hzShift = (tempC: number) => F0 * fracShift(tempC);
/** The fork's tine deflection (drawing only): rings at F0, shown slowed down a lot. */
export const SHOW_SLOW = 4096; // shown 4096 times slower: 8 Hz on screen
/** Seiko Astron 35SQ (1969): ±5 s a month (IEEE Milestone). */
export const ASTRON_PER_DAY = 5 / 30;

/** Divider chain layout: STAGES+1 boxes (the crystal, then each halving) in rows. */
export function chainLayout(W: number) {
  const cols = W < 560 ? 4 : 8;
  const gap = 6;
  const bw = (W - 24 - gap * (cols - 1)) / cols;
  const bh = 34;
  const items = Array.from({ length: STAGES + 1 }, (_, k) => {
    const r = Math.floor(k / cols);
    const c = k % cols;
    return { k, x: 12 + c * (bw + gap), y: 112 + r * (bh + 8), w: bw, h: bh, f: stageFreq(k) };
  });
  const rows = Math.ceil((STAGES + 1) / cols);
  return { items, bw, bh, H: 112 + rows * (bh + 8) + 34, cols };
}
/** Hz as text: "32 768 Hz", "4.0 Hz", "1 Hz". */
export const hzText = (f: number) =>
  f >= 1000 ? `${Math.round(f).toLocaleString('en-US').replace(/,/g, ' ')} Hz` : f >= 10 ? `${Math.round(f)} Hz` : `${f} Hz`;
