import { useEffect, useRef, useState } from 'react';
import {
  Caption,
  GT,
  Seg,
  Specimen,
  T,
  clamp,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { COLOR } from './palette';
import { CURVES, HOUR_STEPS, maxN, sup } from './model';

/**
 * CX-01.2 "a faster computer". For each growth rate, the biggest n a billion-steps-a-second
 * machine finishes in one hour, and the same for a machine 1000× faster. Bars on one log axis
 * of n. On view the machine upgrades by itself; the toggle (with hover preview) switches back.
 * Every bar and number is drawn from one eased clock t in [0, 1].
 */
type Mode = 'today' | 'fast';
const ROWS = CURVES.map((c) => {
  const a = maxN(c, HOUR_STEPS);
  const b = maxN(c, HOUR_STEPS * 1000);
  return { c, a, b, la: Math.log10(a), lb: Math.log10(b) };
});
const LX = 16; // axis: 1 .. 10^16
const MS = 1000;
const ease = (k: number) => (1 - Math.cos(Math.PI * k)) / 2;
const r2 = (v: number) => Math.round(v * 100) / 100;

function gain(a: number, b: number, exp: boolean) {
  if (exp) return `+${b - a}`;
  return `×${Math.round(b / a)}`;
}

function fmtN(v: number, narrow: boolean): string {
  const big: [number, string, string][] = [
    [1e12, 'trillion', 'T'],
    [1e9, 'billion', 'B'],
    [1e6, 'million', 'M'],
  ];
  for (const [d, word, s] of big) {
    if (v >= d) {
      const x = v / d;
      const t =
        x >= 100
          ? Math.round(x).toLocaleString('en-US')
          : x >= 10
            ? String(Math.round(x))
            : x.toFixed(1);
      return narrow ? `${t}${s}` : `${t} ${word}`;
    }
  }
  return Math.round(v).toLocaleString('en-US');
}

function Chart({ t, narrow }: { t: number; narrow: boolean }) {
  const W = narrow ? 360 : 640;
  const L = narrow ? 60 : 72;
  const R = narrow ? 304 : 566;
  const RH = narrow ? 30 : 32;
  const Tp = 22;
  const B = Tp + RH * ROWS.length;
  const H = B + 36;
  const fs = 10;
  const bh = 16;
  const sx = (l: number) => L + (l / LX) * (R - L);
  const cw = 0.62 * (fs + 1); // mono glyph width

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="block w-full"
      role="img"
      aria-label="Biggest input each algorithm finishes in one hour, today and on a computer 1000 times faster"
    >
      {/* axis */}
      {[0, 2, 4, 6, 8, 10, 12, 14, 16].map((e) => (
        <g key={e}>
          <line x1={sx(e)} x2={sx(e)} y1={Tp - 6} y2={B} style={{ stroke: GT.line }} />
          {(!narrow || e % 4 === 0) && (
            <T x={sx(e)} y={B + 12} size={fs - 1}>
              {e === 0 ? '1' : e === 2 ? '100' : `10${sup(e)}`}
            </T>
          )}
        </g>
      ))}
      <line x1={L} x2={R} y1={B} y2={B} style={{ stroke: GT.lineStrong }} />
      <T x={(L + R) / 2} y={H - 6} size={fs} color={GT.inkSoft}>
        {narrow ? 'biggest n done in an hour' : 'biggest n finished in one hour (log scale)'}
      </T>
      <T x={W - 4} y={Tp - 10} anchor="end" size={fs - 1}>
        gain
      </T>

      {ROWS.map(({ c, a, b, la, lb }, i) => {
        const y = Tp + i * RH + (RH - bh) / 2;
        const l = la + (lb - la) * t;
        const x1 = r2(sx(l));
        const v = Math.pow(10, l);
        const exp = !c.poly;
        const label = exp ? String(Math.round(a + (b - a) * t)) : fmtN(v, narrow);
        // inside the bar when even the short (today) bar has room for the longest label
        const longest = (exp ? String(b) : fmtN(b, narrow)).length * cw;
        const inside = sx(la) - L > longest + 12;
        return (
          <g key={c.id}>
            <T x={L - 8} y={y + bh / 2} anchor="end" size={fs + 1} color={COLOR[c.id]} weight={600}>
              {c.label}
            </T>
            <rect
              x={L}
              y={y}
              width={r2(x1 - L)}
              height={bh}
              style={{ fill: COLOR[c.id], opacity: 0.85 }}
            />
            {/* where today's machine stopped: notches above and below, clear of the label */}
            {t > 0.001 && (
              <path
                d={`M${r2(sx(la))},${y - 1} l-3.5,-5 h7 z M${r2(sx(la))},${y + bh + 1} l-3.5,5 h7 z`}
                style={{ fill: GT.ink, opacity: r2(Math.min(1, t * 3)) }}
              />
            )}
            <T
              x={inside ? x1 - 5 : x1 + 5}
              y={y + bh / 2}
              anchor={inside ? 'end' : 'start'}
              size={fs + 1}
              color={inside ? GT.paperRaised : COLOR[c.id]}
              weight={600}
            >
              {label}
            </T>
            <g style={{ opacity: r2(t) }}>
              <T
                x={W - 4}
                y={y + bh / 2}
                anchor="end"
                size={fs + 2}
                color={COLOR[c.id]}
                weight={700}
              >
                {gain(a, b, exp)}
              </T>
            </g>
          </g>
        );
      })}
    </svg>
  );
}

export default function Faster() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [sel, setSel] = useState<Mode>('today');
  const [mode, bind] = useHoverPreview<Mode>(sel);
  const touched = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // auto: upgrade the machine shortly after the figure comes into view
  useEffect(() => {
    if (!inView || touched.current) return;
    touched.current = true;
    if (reduced) {
      setSel('fast');
      return;
    }
    timer.current = setTimeout(() => setSel('fast'), 900);
  }, [inView, reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  // fixed-duration eased glide toward the active mode; retargeting starts from where it is
  const goal = mode === 'fast' ? 1 : 0;
  const [t, setT] = useState(0);
  const tRef = useRef(0);
  useEffect(() => {
    if (reduced) {
      tRef.current = goal;
      setT(goal);
      return;
    }
    const from = tRef.current;
    if (from === goal) return;
    const dur = MS * Math.abs(goal - from);
    const t0 = performance.now();
    let raf = 0;
    const tick = () => {
      const k = clamp((performance.now() - t0) / dur, 0, 1);
      const v = from + (goal - from) * ease(k);
      tRef.current = v;
      setT(v);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [goal, reduced]);

  const choose = (m: Mode) => {
    touched.current = true;
    if (timer.current) clearTimeout(timer.current);
    setSel(m);
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-01.2"
      title="a faster computer"
      kicker="one hour of computing"
      hint="switch the machine; hover to preview"
      footer="10⁹ vs 10¹² steps a second"
    >
      <div className="hidden sm:block">
        <Chart t={t} narrow={false} />
      </div>
      <div className="sm:hidden">
        <Chart t={t} narrow />
      </div>
      <div className="mt-3 border-t border-line pt-3">
        <Seg<Mode>
          label="computer"
          value={mode}
          onChange={choose}
          bind={bind}
          options={[
            { value: 'today', label: 'a billion steps/s' },
            { value: 'fast', label: '1000× faster' },
          ]}
        />
      </div>
      <Caption>
        The same hour on a machine 1000 times faster: an n³ algorithm handles 10 times more data. A
        2ⁿ algorithm handles 10 more items, and n! only 2.
      </Caption>
    </Specimen>
  );
}
