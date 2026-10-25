/** Number formatting for the time post: explicit units, never exponent notation. */

/** A duration in seconds as a short human string: "1 h 12 min", "2 min 28 s", "4.3 s", "38 µs". */
export function fmtSec(sIn: number, digits = 2): string {
  const s = Math.abs(sIn);
  if (s === 0) return '0 s';
  if (s >= 3600) {
    const h = Math.floor(s / 3600);
    const m = Math.round((s - h * 3600) / 60);
    return m === 60 ? `${h + 1} h` : m ? `${h} h ${m} min` : `${h} h`;
  }
  if (s >= 600) return `${Math.round(s / 60)} min`;
  if (s >= 60) {
    const m = Math.floor(s / 60);
    const r = Math.round(s - m * 60);
    return r === 60 ? `${m + 1} min` : r ? `${m} min ${r} s` : `${m} min`;
  }
  const units: [number, string][] = [
    [1, 's'],
    [1e-3, 'ms'],
    [1e-6, 'µs'],
    [1e-9, 'ns'],
    [1e-12, 'ps'],
    [1e-15, 'fs'],
  ];
  for (const [u, name] of units) {
    if (s >= u * 0.9995) return `${sig(s / u, digits)} ${name}`;
  }
  return `${sig(s / 1e-15, digits)} fs`;
}

/** `digits` significant figures, but never exponent notation and never more than 2 decimals
 *  for values >= 1 (so 3.0 → "3", 12.34 → "12", 0.0625 → "0.063"). */
export function sig(v: number, digits = 2): string {
  if (v === 0) return '0';
  const mag = Math.floor(Math.log10(Math.abs(v)));
  const dec = Math.max(0, digits - 1 - mag);
  const s = v.toFixed(Math.min(dec, 6));
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

/** A distance in metres: "11.4 km", "463 m", "30 cm", "3 mm". */
export function fmtDist(m: number): string {
  const a = Math.abs(m);
  if (a >= 1000) return `${sig(a / 1000, a >= 100000 ? 4 : 3)} km`;
  if (a >= 1) return `${sig(a, 3)} m`;
  if (a >= 0.01) return `${sig(a * 100, 2)} cm`;
  return `${sig(a * 1000, 2)} mm`;
}

/** "9:07" from hours (wraps into 0..24), rounded to the minute. */
export function hm(hours: number): string {
  const t = ((hours % 24) + 24) % 24;
  let h = Math.floor(t);
  let m = Math.round((t - h) * 60);
  if (m === 60) {
    h = (h + 1) % 24;
    m = 0;
  }
  return `${h}:${String(m).padStart(2, '0')}`;
}

/** Signed with a real minus sign. */
export const signed = (v: number, f: (x: number) => string) => `${v < 0 ? '−' : '+'}${f(Math.abs(v))}`;

const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
/** "1.1 × 10⁻¹⁰" (two significant figures), written out, never "e-10". */
export function sci(v: number, digits = 2): string {
  if (v === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = v / 10 ** e;
  const ms = sig(m, digits);
  if (ms === '10') return `1 × 10${String(e + 1).replace(/./g, (c) => SUP[c])}`;
  return `${ms} × 10${String(e).replace(/./g, (c) => SUP[c])}`;
}
