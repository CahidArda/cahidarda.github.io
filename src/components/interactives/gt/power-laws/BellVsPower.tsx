import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  GT,
  Seg,
  Slider,
  Specimen,
  T,
  clamp,
  useHoverPreview,
  useReducedMotion,
} from '../shared';
import { copyingModel, fitSlope, gauss, rng } from './model';

/**
 * Two quantities on one chart, both measured in "multiples of the average":
 *  - heights of 100,000 men (illustrative normal: mean 175 cm, sd 7 cm)
 *  - in-links of 100,000 pages grown by the copying model (mean in-links is 1)
 * Toggle linear vs log-log. On log-log the power law is a straight line; the bell is a spike at 1x.
 *
 * Motion: the axes morph through a Box-Cox family (x^λ, λ from 1 to 0, i.e. linear to log) with
 * the axis range interpolated too, so every mark stays near the plot during the switch and the
 * tween ends exactly on its target (no exponential tail, no final snap). A change of p glides the
 * points, the fitted line and the readout together on one clock.
 */
const N = 100_000;
const SD = 7 / 175; // height sd as a fraction of the mean
const KMAX = 30; // exact counts k = 1..30 for the linear view
const RATIO = 1.5; // log-bin width factor
const X_HI = 1e5; // right end of the log x axis (top page is at most ~55k)

type Mode = 'linear' | 'loglog';

/** Fixed log-bin slots [lo, hi) on integers, identical for every p so points can glide. */
const SLOTS = (() => {
  const out: { lo: number; hi: number; x: number }[] = [];
  let lo = 1;
  while (lo <= X_HI) {
    const hi = Math.max(lo + 1, Math.ceil(lo * RATIO));
    out.push({ lo, hi, x: Math.sqrt(lo * (hi - 1)) });
    lo = hi;
  }
  return out;
})();

/** What the chart shows for one p, all in log10 of "how common, relative to k = 1". */
type View = {
  ex: number[]; // exact counts, k = 1..KMAX
  exOn: number[]; // 1 if that k occurs
  bn: number[]; // log bins
  bnOn: number[];
  slope: number;
  icpt: number;
  lmax: number; // log10 of the most-linked page, in multiples of the average
};

const cache = new Map<number, View>();
function simulate(pct: number): View {
  const hit = cache.get(pct);
  if (hit) return hit;
  const p = pct / 100;
  const { indeg } = copyingModel(N, p, 7);
  let max = 0;
  let total = 0;
  const exact = new Array(KMAX + 1).fill(0);
  const counts = new Array(SLOTS.length).fill(0);
  for (let i = 0; i < N; i++) {
    const v = indeg[i];
    total += v;
    if (v > max) max = v;
    if (v === 0) continue;
    if (v <= KMAX) exact[v]++;
    let s = 0;
    while (SLOTS[s].hi <= v) s++;
    counts[s]++;
  }
  const mean = total / N;
  const peak = exact[1] / N; // k = 1 is the most common non-zero value
  const lp = Math.log10(peak);
  const ex = exact.slice(1).map((c) => (c ? Math.log10(c / N) - lp : 0));
  const exOn = exact.slice(1).map((c) => (c ? 1 : 0));
  const bins = SLOTS.map((s, i) => ({
    x: s.x,
    y: counts[i] / N / (s.hi - s.lo),
    count: counts[i],
  }));
  const bn = bins.map((q) => (q.count ? Math.log10(q.y) - lp : 0));
  const bnOn = bins.map((q) => (q.count ? 1 : 0));
  // fit on the clean part of the tail (enough counts, past the head where random links dominate)
  const c = p / (1 - p);
  const fit = fitSlope(bins.filter((q) => q.count >= 10 && q.x >= Math.max(3, 8 * c)));
  const v: View = {
    ex,
    exOn,
    bn,
    bnOn,
    slope: fit.slope,
    icpt: fit.intercept - lp,
    lmax: Math.log10(max / mean),
  };
  cache.set(pct, v);
  return v;
}

const lerp = (a: number, b: number, s: number) => a + (b - a) * s;
const mixArr = (a: number[], aOn: number[], b: number[], bOn: number[], s: number) =>
  a.map((v, i) => (aOn[i] <= 0 ? b[i] : bOn[i] <= 0 ? v : lerp(v, b[i], s)));
const mixView = (a: View, b: View, s: number): View => ({
  ex: mixArr(a.ex, a.exOn, b.ex, b.exOn, s),
  exOn: a.exOn.map((v, i) => lerp(v, b.exOn[i], s)),
  bn: mixArr(a.bn, a.bnOn, b.bn, b.bnOn, s),
  bnOn: a.bnOn.map((v, i) => lerp(v, b.bnOn[i], s)),
  slope: lerp(a.slope, b.slope, s),
  icpt: lerp(a.icpt, b.icpt, s),
  lmax: lerp(a.lmax, b.lmax, s),
});
const easeInOut = (k: number) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);
const now = () => (typeof performance === 'undefined' ? 0 : performance.now());

/**
 * Fixed-duration glide from whatever is on screen to `target`. Retargeting mid-flight starts from
 * the current mix, and the last frame lands exactly on the target, so nothing ever snaps.
 */
function useGlide<S>(target: S, mix: (a: S, b: S, s: number) => S, reduced: boolean, ms: number) {
  const g = useRef<{ from: S; to: S; t0: number }>({ from: target, to: target, t0: -Infinity });
  const [clock, setClock] = useState(0);
  const prog = (tm: number) => (reduced ? 1 : easeInOut(clamp((tm - g.current.t0) / ms, 0, 1)));
  if (g.current.to !== target) {
    const tm = now();
    g.current = { from: mix(g.current.from, g.current.to, prog(tm)), to: target, t0: tm };
  }
  const { t0 } = g.current;
  useEffect(() => {
    if (reduced || t0 === -Infinity) return;
    let raf = 0;
    const tick = () => {
      const tm = now();
      setClock(tm);
      if (tm - t0 < ms) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [t0, reduced, ms]);
  const s = prog(Math.max(clock, g.current.t0));
  return s >= 1 ? g.current.to : mix(g.current.from, g.current.to, s);
}

const tallest = (() => {
  const r = rng(11);
  let m = 0;
  for (let i = 0; i < N; i++) m = Math.max(m, 1 + SD * gauss(r));
  return m;
})();

const r2 = (v: number) => Math.round(v * 100) / 100;
const bell = (x: number) => Math.exp(-((x - 1) ** 2) / (2 * SD * SD));
/** Box-Cox: (v^λ - 1) / λ, which is ln v at λ = 0 (expm1 keeps it smooth as λ → 0). */
const boxCox = (v: number, lam: number) => {
  const L = Math.log(Math.max(v, 1e-12));
  return lam === 0 ? L : Math.expm1(lam * L) / lam;
};
/** Axis at morph t (0 = linear [0, linHi], 1 = log [logLo, logHi]) → fraction 0..1. */
function axis(t: number, linHi: number, logLo: number, logHi: number) {
  const lam = 1 - t;
  const lo = logLo * t;
  const hi = Math.exp(lerp(Math.log(linHi), Math.log(logHi), t));
  const f0 = boxCox(lo, lam);
  const f1 = boxCox(hi, lam);
  return (v: number) => (boxCox(v, lam) - f0) / (f1 - f0);
}
/** Round to two significant figures so the readout settles instead of ticking. */
const sig2 = (v: number) => {
  const m = Math.pow(10, Math.max(0, Math.floor(Math.log10(v)) - 1));
  return Math.round(v / m) * m;
};

export default function BellVsPower() {
  const reduced = useReducedMotion();
  const [sel, setSel] = useState<Mode>('loglog');
  const [mode, bindMode] = useHoverPreview<Mode>(sel);
  const [pct, setPct] = useState(20);
  const t = useGlide(mode === 'loglog' ? 1 : 0, lerp, reduced, 750);
  const view = useGlide(simulate(pct), mixView, reduced, 450);
  const target = simulate(pct);
  const alpha = -target.slope;
  const pred = 1 + 1 / (1 - pct / 100);
  const topPage = sig2(Math.pow(10, view.lmax));
  // labels belong to a settled view: fade in at the end of the switch, out at the start
  const logOp = clamp((t - 0.75) / 0.25, 0, 1);
  const linOp = clamp((0.25 - t) / 0.25, 0, 1);

  // phones and desktops get their own drawing, chosen by CSS (no re-layout after hydration)
  const chart = (narrow: boolean) => {
    const W = narrow ? 370 : 600;
    const H = narrow ? 330 : 300;
    const M = { l: 56, r: 12, t: 14, b: 38 }; // room for 10⁻⁶ ticks beside the rotated axis title
    const PW = W - M.l - M.r;
    const PH = H - M.t - M.b;
    const X0 = M.l;
    const Y0 = H - M.b;

    const LY = [1e-7, Math.pow(10, 0.3)];
    const ax = axis(t, 20, 0.5, X_HI);
    const ay = axis(t, 1.08, LY[0], LY[1]);
    // rounded so server and browser floating point agree at hydration
    const sx = (x: number) => r2(X0 + ax(x) * PW);
    const sy = (y: number) => r2(Y0 - ay(y) * PH);
    // settled projections for labels
    const axL = axis(1, 20, 0.5, X_HI);
    const ayL = axis(1, 1.08, LY[0], LY[1]);
    const sxLog = (x: number) => r2(X0 + axL(x) * PW);
    const syLog = (y: number) => r2(Y0 - ayL(y) * PH);
    const sxLin = (x: number) => r2(X0 + (x / 20) * PW);
    const syLin = (y: number) => r2(Y0 - (y / 1.08) * PH);
    const inX = (px: number) => px >= X0 - 0.5 && px <= W - M.r + 0.5;
    const inY = (py: number) => py >= M.t - 0.5 && py <= Y0 + 0.5;

    // bell curve (heights)
    let bellPath = '';
    for (let i = 0; i <= 240; i++) {
      const x = 0.75 + (0.5 * i) / 240;
      bellPath += `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(bell(x)).toFixed(1)}`;
    }

    // fitted power law, sampled so it bends with the axes and is straight once they are log
    const fitY = (x: number) => Math.pow(10, view.icpt + view.slope * Math.log10(x));
    const lx1 = Math.log10(X_HI);
    let fitPath = '';
    for (let i = 0; i <= 60; i++) {
      const x = Math.pow(10, (lx1 * i) / 60);
      fitPath += `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(fitY(x)).toFixed(1)}`;
    }
    const fitOp = clamp((t - 0.4) / 0.5, 0, 1);

    const xTicksLin = [0, 5, 10, 15, 20];
    const xTicksLog = [1, 10, 100, 1000, 10000, 100000];
    const yTicksLin = [0, 0.5, 1];
    const yTicksLog = [1, 1e-2, 1e-4, 1e-6];
    const fmtX = (v: number) => (v >= 1000 ? `${v / 1000}k` : String(v));
    const fmtYLog = (v: number) => (v === 1 ? '1' : `10${sup(Math.log10(v))}`);
    const tickOpLin = 1 - t;
    const tickOpLog = t;

    // most-linked page: a marker on the x axis (log view), an arrow off the right edge (linear)
    const topX = Math.pow(10, view.lmax);
    const topPx = sxLog(topX);
    const clipId = narrow ? 'gt10-2-clip-n' : 'gt10-2-clip';
    const sq = (x: number, y: number, op: number, key: string) => {
      const px = sx(x);
      const py = sy(y);
      if (op <= 0.01 || !inX(px) || !inY(py)) return null;
      return (
        <rect
          key={key}
          x={px - 3}
          y={py - 3}
          width={6}
          height={6}
          style={{ fill: GT.a, opacity: r2(op) }}
        />
      );
    };
    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full"
        role="img"
        aria-label="Distribution of heights and of in-links"
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={M.l} y={M.t} width={PW} height={PH} />
          </clipPath>
        </defs>
        {/* grid + ticks: both sets ride the morphing axes and cross-fade */}
        {(
          [
            [xTicksLin, tickOpLin, 'l'],
            [xTicksLog, tickOpLog, 'g'],
          ] as const
        ).map(([ticks, op, k]) =>
          op <= 0.01
            ? null
            : ticks.map((v) => {
                const px = sx(v);
                if (!inX(px)) return null;
                return (
                  <g key={`x${k}${v}`} style={{ opacity: r2(op) }}>
                    <line x1={px} x2={px} y1={M.t} y2={Y0} style={{ stroke: GT.line }} />
                    <T x={px} y={Y0 + 12} size={10}>
                      {fmtX(v)}
                    </T>
                  </g>
                );
              }),
        )}
        {(
          [
            [yTicksLin, tickOpLin, 'l'],
            [yTicksLog, tickOpLog, 'g'],
          ] as const
        ).map(([ticks, op, k]) =>
          op <= 0.01
            ? null
            : ticks.map((v) => {
                const py = sy(v);
                if (!inY(py)) return null;
                return (
                  <g key={`y${k}${v}`} style={{ opacity: r2(op) }}>
                    <line x1={M.l} x2={W - M.r} y1={py} y2={py} style={{ stroke: GT.line }} />
                    <T x={M.l - 6} y={py} anchor="end" size={10}>
                      {k === 'g' ? fmtYLog(v) : String(v)}
                    </T>
                  </g>
                );
              }),
        )}
        <line x1={M.l} x2={M.l} y1={M.t} y2={Y0} style={{ stroke: GT.lineStrong }} />
        <line x1={M.l} x2={W - M.r} y1={Y0} y2={Y0} style={{ stroke: GT.lineStrong }} />
        <T x={M.l + PW / 2} y={H - 8} size={10} color={GT.inkSoft}>
          value, in multiples of the average
        </T>
        <g transform={`translate(10 ${M.t + PH / 2}) rotate(-90)`}>
          <T x={0} y={0} size={10} color={GT.inkSoft}>
            how common
          </T>
        </g>

        <g clipPath={`url(#${clipId})`}>
          {fitOp > 0.01 && (
            <path
              d={fitPath}
              strokeDasharray="5 4"
              style={{ fill: 'none', stroke: GT.eq, strokeWidth: 1.5, opacity: r2(fitOp) }}
            />
          )}
          <path d={bellPath} style={{ fill: 'none', stroke: GT.b, strokeWidth: 2.2 }} />
          {view.ex.map((ly, i) => sq(i + 1, Math.pow(10, ly), view.exOn[i] * (1 - t), `e${i}`))}
          {view.bn.map((ly, i) => sq(SLOTS[i].x, Math.pow(10, ly), view.bnOn[i] * t, `b${i}`))}
        </g>

        {/* log-log annotations, attached to their marks */}
        {logOp > 0 && (
          <g style={{ opacity: r2(logOp) }}>
            <T x={sxLog(1.3)} y={syLog(1e-5)} anchor="start" size={10} color={GT.b} weight={600}>
              heights
            </T>
            <T x={sxLog(1.3)} y={syLog(1e-5) + 13} anchor="start" size={10} color={GT.b}>
              everyone near 1×
            </T>
            <T
              x={sxLog(narrow ? 25 : 40)}
              y={syLog(fitY(narrow ? 25 : 40)) - 16}
              anchor="start"
              size={10}
              color={GT.a}
              weight={600}
            >
              links: a straight line
            </T>
            <T
              x={sxLog(narrow ? 200 : 400)}
              y={syLog(fitY(narrow ? 200 : 400)) - 14}
              anchor="start"
              size={10}
              color={GT.eq}
            >
              slope −{alpha.toFixed(2)}
              {narrow ? '' : ` (theory −${pred.toFixed(2)})`}
            </T>
            <path d={`M${topPx},${Y0 - 1} l-4,-7 h8 z`} style={{ fill: GT.a, stroke: 'none' }} />
            <T
              x={topPx}
              y={Y0 - 16}
              anchor={topPx > W - M.r - 60 ? 'end' : 'middle'}
              size={10}
              color={GT.a}
            >
              most-linked page
            </T>
          </g>
        )}
        {/* linear annotations */}
        {linOp > 0 && (
          <g style={{ opacity: r2(linOp) }}>
            <T x={sxLin(1.6)} y={syLin(0.92)} anchor="start" size={10} color={GT.b}>
              <tspan style={{ fontWeight: 600 }}>heights</tspan>: everyone near 1×
            </T>
            <T x={sxLin(3.6)} y={syLin(0.2)} anchor="start" size={10} color={GT.a}>
              <tspan style={{ fontWeight: 600 }}>links</tspan>: a thin tail that keeps going
            </T>
            <T x={W - M.r - 4} y={Y0 - 16} anchor="end" size={10} color={GT.a}>
              most-linked page: way off to the right →
            </T>
          </g>
        )}
      </svg>
    );
  };

  return (
    <Specimen
      code="GT-10.2"
      title="bell vs power"
      kicker="100,000 men · 100,000 pages"
      hint="switch the axes; drag p to tilt the line"
    >
      <div className="hidden sm:block">{chart(false)}</div>
      <div className="sm:hidden">{chart(true)}</div>
      <div className="mt-3 grid items-end gap-x-8 gap-y-4 sm:grid-cols-[auto_minmax(0,1fr)]">
        <Seg<Mode>
          label="axes"
          value={mode}
          onChange={setSel}
          bind={bindMode}
          options={[
            { value: 'linear', label: 'linear' },
            { value: 'loglog', label: 'log-log' },
          ]}
        />
        <div className="max-w-sm">
          <Slider
            label="p · random links"
            value={pct}
            min={5}
            max={50}
            step={5}
            onChange={setPct}
            format={(v) => (v / 100).toFixed(2)}
          />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-x-10 gap-y-3 border-t border-line pt-4">
        <BigStat
          label="tallest man"
          value={`${tallest.toFixed(2)}×`}
          color={GT.b}
          sub="the average height"
        />
        <BigStat
          label="most-linked page"
          value={`${topPage.toLocaleString('en-US')}×`}
          color={GT.a}
          sub="the average page"
        />
      </div>
    </Specimen>
  );
}

function sup(n: number) {
  const map: Record<string, string> = {
    '-': '⁻',
    '0': '⁰',
    '1': '¹',
    '2': '²',
    '3': '³',
    '4': '⁴',
    '5': '⁵',
    '6': '⁶',
    '7': '⁷',
    '8': '⁸',
    '9': '⁹',
  };
  return String(Math.round(n))
    .split('')
    .map((c) => map[c] ?? c)
    .join('');
}
