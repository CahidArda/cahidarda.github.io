/**
 * The time post's STABLE REFERENCE drawing, shared by every episode plate (TimeRef.tsx draws it).
 * One layout for all nine plates, so the reader sees each device attack the same problem:
 *
 *   row 1  THE SUN       the sun's height over Istanbul through the chosen day, the sun at the
 *                        TRUE local solar time (a world object: the instrument never moves it).
 *   row 2  THE CLOCK     a 24 h line with the true time (ink) and what the device says (red),
 *                        and the gap as a duration (and, at sea or for GPS, as a distance).
 *   row 3  ERROR A DAY   a log gauge from 10⁴ s to 10⁻¹⁴ s a day: every device so far as a band,
 *                        the current one in red, so each episode visibly moves the mark right.
 *
 * Screen coordinates, y DOWN, viewBox units = CSS px of the measured width W. Pure: no React.
 */
import { PHI, altitudeAt, sunTimes } from './astro.ts';
import { fmtDist, fmtSec, hm } from './fmt.ts';

/** Each episode's device on the gauge: the error per day it could hold (seconds), with the
 *  source of the numbers. `oom`: an order-of-magnitude estimate, drawn hollow. */
export type Device = { short: string; lo: number; hi: number; oom?: boolean; note: string };
export const DEVICES: Device[] = [
  { short: 'sundial', lo: 300, hi: 1800, oom: true, note: 'order of magnitude: a soft shadow edge, read to a few minutes' },
  { short: 'water clock', lo: 300, hi: 1800, oom: true, note: 'order of magnitude' },
  { short: 'sine quadrant', lo: 60, hi: 300, oom: true, note: 'per reading, not a drift; my estimate from reading the board to half a part' },
  { short: 'verge clock', lo: 900, hi: 7200, note: 'one to two hours a day at first, 15 minutes at best (Wikipedia, Verge escapement)' },
  { short: 'pendulum clock', lo: 1, hi: 60, note: 'under a minute a day (Huygens, NIST) to about a second (Graham, 1721, NIST)' },
  { short: 'H4 at sea', lo: 5 / 81.2, hi: 5 / 81.2, note: '5 s in 81 days on the Jamaica trial (Wikipedia, John Harrison)' },
  { short: 'quartz watch', lo: 5 / 30, hi: 15 / 30, note: '5 s a month (Seiko Astron) to 15 s a month (typical)' },
  { short: 'atomic clocks', lo: 8.1e-19 * 86400, hi: 1 / (300 * 365.25), note: '1 s in 300 years (1955) to 8.1 × 10⁻¹⁹ (2024)' },
  { short: 'GPS needs', lo: 3e-9, hi: 3e-9, note: '3 ns of clock error is about 1 m of range (Ashby)' },
];

export const GAUGE_HI = 1e4; // left end, s/day
export const GAUGE_LO = 1e-14; // right end

export type RefInput = {
  /** Episode index 0..8: which device is current on the gauge. */
  ep: number;
  /** True local solar time now, hours (0..24). */
  t: number;
  /** The sun's declination that day (degrees). */
  dec: number;
  /** Date label for the sky row. */
  dateText: string;
  /** What the device reads, hours; null = it gives no reading (night, cloud, an empty bowl). */
  read: number | null;
  /** |device − true| in seconds, exact (the 24 h line cannot show 38 µs; the label can). */
  gapSec: number | null;
  /** Why there is a gap, e.g. "set at noon, 6 h ago". */
  gapNote?: string;
  /** If set, the gap also as a distance: metres per second of clock error. */
  metresPerSec?: number;
  distWord?: string;
  /** A live error-per-day value for the current device (a dot on its band), s/day. */
  rate?: number | null;
  /** Which row the current step is about. */
  hi?: 'sky' | 'clock' | 'gauge' | null;
  /** Text for a missing reading. */
  noRead?: string;
};

export type Lbl = { id: string; x: number; y: number; text: string; size: number; anchor: 'start' | 'middle' | 'end'; kind: 'label' | 'num' | 'note' };

export const labelW = (l: Pick<Lbl, 'text' | 'size' | 'kind'>) =>
  [...l.text].length * l.size * (l.kind === 'label' ? 0.78 * 0.72 : l.kind === 'num' ? 0.6 : 0.5);
export function lblBox(l: Lbl) {
  const w = labelW(l);
  const x0 = l.anchor === 'start' ? l.x : l.anchor === 'middle' ? l.x - w / 2 : l.x - w;
  const s = l.kind === 'label' ? l.size * 0.78 : l.size;
  return { x0, x1: x0 + w, y0: l.y - s * 0.6, y1: l.y + s * 0.4, name: l.id };
}

const PAD = 12;
export const REF_H = 258;

export function refScene(inp: RefInput, W: number) {
  const narrow = W < 420;
  const x0 = PAD;
  const x1 = W - PAD;
  const iw = x1 - x0;
  const xT = (t: number) => x0 + (t / 24) * iw;

  // ── row 1: the sun ──
  const skyTop = 30;
  const horizonY = 88;
  const ky = (horizonY - skyTop) / 75;
  const path: { x: number; y: number }[] = [];
  for (let i = 0; i <= 192; i++) {
    const t = (i / 192) * 24;
    const a = altitudeAt(t, inp.dec, PHI);
    path.push({ x: xT(t), y: horizonY - Math.max(a, -6) * ky });
  }
  const alt = altitudeAt(inp.t, inp.dec, PHI);
  const sun = { x: xT(inp.t), y: horizonY - Math.max(alt, -6) * ky, up: alt > 0 };
  const { rise, set } = sunTimes(inp.dec, PHI);

  // ── row 2: true time vs the device ──
  const axisY = 140;
  const trueX = xT(((inp.t % 24) + 24) % 24);
  const readX = inp.read === null ? null : xT(((inp.read % 24) + 24) % 24);
  const hourTicks = [0, 6, 12, 18, 24].map((h) => ({ h, x: xT(h) }));

  // ── row 3: the gauge ──
  const gaugeY = 234;
  const lg = (v: number) => Math.log10(v);
  const xG = (v: number) =>
    x0 + ((lg(GAUGE_HI) - lg(Math.min(GAUGE_HI, Math.max(GAUGE_LO, v)))) / (lg(GAUGE_HI) - lg(GAUGE_LO))) * iw;
  const TICKS: [number, string][] = narrow
    ? [[60, '1 min'], [1, '1 s'], [1e-3, '1 ms'], [1e-6, '1 µs'], [1e-9, '1 ns'], [1e-12, '1 ps']]
    : [[3600, '1 h'], [60, '1 min'], [1, '1 s'], [1e-3, '1 ms'], [1e-6, '1 µs'], [1e-9, '1 ns'], [1e-12, '1 ps']];
  const gTicks = TICKS.map(([v, s]) => ({ v, s, x: xG(v) }));
  const bands = DEVICES.slice(0, inp.ep + 1).map((d, i) => {
    let a = xG(d.hi);
    let b = xG(d.lo);
    if (b - a < 4) {
      const m = (a + b) / 2;
      a = m - 2;
      b = m + 2;
    }
    return { i, a, b, cur: i === inp.ep, oom: !!d.oom, short: d.short };
  });
  const rateX = inp.rate == null || !(inp.rate > 0) ? null : xG(inp.rate);

  // ── labels ──
  const L: Lbl[] = [];
  L.push({ id: 'sunHead', x: x0, y: 14, text: `the sun over Istanbul, ${inp.dateText}`, size: 11.5, anchor: 'start', kind: 'note' });
  L.push({ id: 'clockHead', x: x0, y: 106, text: 'what time is it?', size: 11.5, anchor: 'start', kind: 'note' });
  L.push({ id: 'gaugeHead', x: x0, y: 204, text: 'error per day, better to the right', size: 11.5, anchor: 'start', kind: 'note' });
  // sun label: beside the sun, inside the plate
  {
    if (sun.up) {
      const s = `sun ${alt.toFixed(0)}° high`;
      const w = labelW({ text: s, size: 11, kind: 'note' });
      const right = sun.x + 10 + w <= x1;
      const y = Math.max(skyTop + 2, Math.min(horizonY - 8, sun.y - 8));
      L.push({ id: 'sun', x: right ? sun.x + 10 : sun.x - 10, y, text: s, size: 11, anchor: right ? 'start' : 'end', kind: 'note' });
    } else {
      // night: say so in the top corner on the sun's side, clear of the day's path
      const left = sun.x < (x0 + x1) / 2;
      L.push({ id: 'sun', x: left ? x0 : x1, y: skyTop + 4, text: 'night: the sun is below the horizon', size: 11, anchor: left ? 'start' : 'end', kind: 'note' });
    }
  }
  // true / device labels over the axis
  const trueTxt = `true ${hm(inp.t)}`;
  const readTxt = inp.read === null ? '' : `device ${hm(inp.read)}`;
  const tw = labelW({ text: trueTxt, size: 11, kind: 'num' });
  const rw = labelW({ text: readTxt, size: 11, kind: 'num' });
  const yTop = axisY - 14;
  if (readX === null) {
    // only the true time on the line; the missing reading is said on the gap line
    const left = trueX - 6 - tw >= x0;
    L.push({ id: 'true', x: left ? trueX - 6 : trueX + 6, y: yTop, text: trueTxt, size: 11, anchor: left ? 'end' : 'start', kind: 'num' });
  } else {
    const readFirst = readX < trueX;
    // keep both on the line: left label ends at its tick, right label starts at its tick
    const lx = Math.min(readFirst ? readX : trueX, x1 - tw - rw - 16);
    const rx = Math.max(readFirst ? trueX : readX, x0 + tw + rw + 16);
    const leftTxt = readFirst ? readTxt : trueTxt;
    const rightTxt = readFirst ? trueTxt : readTxt;
    const lw = readFirst ? rw : tw;
    let ltx = Math.max(x0 + lw, lx - 5);
    let rtx = Math.min(x1 - (readFirst ? tw : rw), rx + 5);
    if (rtx - ltx < 8) {
      const mid = (ltx + rtx) / 2;
      ltx = mid - 4;
      rtx = mid + 4;
      if (ltx - lw < x0) {
        ltx = x0 + lw;
        rtx = ltx + 8;
      }
      if (rtx + (readFirst ? tw : rw) > x1) {
        rtx = x1 - (readFirst ? tw : rw);
        ltx = rtx - 8;
      }
    }
    L.push({ id: readFirst ? 'read' : 'true', x: ltx, y: yTop, text: leftTxt, size: 11, anchor: 'end', kind: 'num' });
    L.push({ id: readFirst ? 'true' : 'read', x: rtx, y: yTop, text: rightTxt, size: 11, anchor: 'start', kind: 'num' });
  }
  // the gap line under the axis
  let gapTxt = '';
  if (inp.gapSec === null) gapTxt = inp.read === null ? `no reading: ${inp.noRead ?? 'the device gives no time now'}` : '';
  else if (inp.gapSec === 0) gapTxt = 'device and true time agree';
  else {
    gapTxt = `off by ${fmtSec(inp.gapSec)}`;
    if (inp.metresPerSec) gapTxt += ` = ${fmtDist(inp.gapSec * inp.metresPerSec)}${inp.distWord ? ` ${inp.distWord}` : ''}`;
  }
  if (inp.gapNote && gapTxt && inp.gapSec !== null) gapTxt += ` (${inp.gapNote})`;
  const gapSize = narrow ? 10.5 : 11.5;
  // wrap into two lines when it does not fit
  const lines: string[] = [];
  if (labelW({ text: gapTxt, size: gapSize, kind: 'note' }) <= iw) lines.push(gapTxt);
  else {
    const k = gapTxt.indexOf(' (');
    if (k > 0) lines.push(gapTxt.slice(0, k), gapTxt.slice(k + 1));
    else lines.push(gapTxt);
  }
  lines.forEach((s, i) =>
    L.push({ id: `gap${i}`, x: x0, y: 172 + i * 13, text: s, size: gapSize, anchor: 'start', kind: 'note' }),
  );
  // current device label over its band
  const cur = bands[bands.length - 1];
  {
    const d = DEVICES[inp.ep];
    const s = d.short;
    const w = labelW({ text: s, size: 11, kind: 'note' });
    const mid = (cur.a + cur.b) / 2;
    const x = Math.max(x0, Math.min(x1 - w, mid - w / 2));
    L.push({ id: 'dev', x, y: gaugeY - 11, text: s, size: 11, anchor: 'start', kind: 'note' });
  }
  return {
    W,
    H: REF_H,
    x0,
    x1,
    skyTop,
    horizonY,
    path,
    sun,
    rise: xT(rise),
    set: xT(set),
    noonX: xT(12),
    axisY,
    trueX,
    readX,
    hourTicks,
    gaugeY,
    gTicks,
    bands,
    rateX,
    labels: L,
    gapLines: lines.length,
  };
}
export type RefScene = ReturnType<typeof refScene>;
