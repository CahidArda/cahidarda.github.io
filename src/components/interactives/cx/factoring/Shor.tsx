import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Reserve } from '../../gt/Reserve';
import {
  BigStat,
  GT,
  Seg,
  Specimen,
  T,
  clamp,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';

/**
 * Shor's algorithm on a small number, in three steps.
 *  1. f(x) = a^x mod N repeats with a period r (top row: x = 0 .. Q-1, the bars where f(x) = 1
 *     are marked; they sit r apart).
 *  2. The quantum Fourier transform adds one wave e^(2πi k x / Q) for every x with the same
 *     f(x). Their sum (bottom row, |sum| / number of waves) starts flat (every answer equally
 *     likely) and collapses into spikes at k = j·Q/r: the wrong answers cancel. The peak spacing
 *     gives r.
 *  3. gcd(a^(r/2) − 1, N) and gcd(a^(r/2) + 1, N) are the factors.
 * Q = 128 keeps the picture legible (a real run uses Q ≥ N²). Every number below is checked
 * by scratchpad/shor/verify.py: 15/7 → r 4, 48 50, 3 5; 21/2 → r 6, 7 9, 7 3; 35/2 → r 12,
 * 63 65, 7 5.
 */
const Q = 128;

type Case = { id: '15' | '21' | '35'; N: number; a: number };
const CASES: Case[] = [
  { id: '15', N: 15, a: 7 },
  { id: '21', N: 21, a: 2 },
  { id: '35', N: 35, a: 2 },
];

function powmod(b: number, e: number, m: number) {
  let r = 1;
  b %= m;
  while (e > 0) {
    if (e & 1) r = (r * b) % m;
    b = (b * b) % m;
    e >>= 1;
  }
  return r;
}
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const sup = (n: number) => String(n).replace(/\d/g, (d) => SUP[Number(d)]);
const gcd = (x: number, y: number): number => (y === 0 ? x : gcd(y, x % y));

function solve(c: Case) {
  const f = Array.from({ length: Q }, (_, x) => powmod(c.a, x, c.N));
  let r = 1;
  while (f[r] !== 1) r++;
  const half = c.a ** (r / 2); // small enough to be exact for these cases
  const lo = half - 1;
  const hi = half + 1;
  const p = gcd(lo, c.N);
  const q = gcd(hi, c.N);
  const xs = f.map((v, x) => (v === 1 ? x : -1)).filter((x) => x >= 0); // 0, r, 2r, ...
  return { ...c, f, r, half, lo, hi, p, q, xs };
}
const SOLVED = CASES.map(solve);

/* the clock, in seconds */
const T_BARS = 1.4; // top row sweeps in
const T_BRACKET = 1.5; // the period bracket fades in
const T_WAVES = 2.0; // waves start adding
const D_WAVES = 4.0;
const T_PEAKS = T_WAVES + D_WAVES; // bottom bracket
const T_R = T_PEAKS + 0.3;
const T_POW = T_R + 0.45;
const T_GCD = T_POW + 0.45;
const END = T_GCD + 0.5;

const ramp = (t: number, t0: number, d = 0.35) => clamp((t - t0) / d, 0, 1);

/** How many waves are in the sum at time t (fractional: the newest one fades in). */
function wavesAt(t: number, M: number) {
  const p = clamp((t - T_WAVES) / D_WAVES, 0, 1);
  return M * p ** 1.8;
}

type Geo = {
  W: number;
  H: number;
  l: number;
  r: number;
  y1t: number; // top row: top, baseline
  y1b: number;
  y2t: number; // bottom row: top, baseline
  y2b: number;
  fs: number;
  narrow: boolean;
};
const WIDE: Geo = {
  W: 640,
  H: 340,
  l: 10,
  r: 10,
  y1t: 30,
  y1b: 112,
  y2t: 182,
  y2b: 296,
  fs: 10,
  narrow: false,
};
const NARROW: Geo = {
  W: 360,
  H: 340,
  l: 6,
  r: 6,
  y1t: 30,
  y1b: 108,
  y2t: 180,
  y2b: 296,
  fs: 9,
  narrow: true,
};

function Pow({ a, e, fs }: { a: ReactNode; e: ReactNode; fs: number }) {
  return (
    <>
      {a}
      <tspan dy={-4} fontSize={fs - 2}>
        {e}
      </tspan>
      <tspan dy={4}> </tspan>
    </>
  );
}

/** A bracket under a span, ends ticked up, with a label starting at its left end. */
function Bracket({
  x0,
  x1,
  y,
  color,
  label,
  fs,
  opacity,
}: {
  x0: number;
  x1: number;
  y: number;
  color: string;
  label: ReactNode;
  fs: number;
  opacity: number;
}) {
  return (
    <g style={{ opacity }}>
      <path
        d={`M${x0} ${y - 4} V${y} H${x1} V${y - 4}`}
        style={{ fill: 'none', stroke: color, strokeWidth: 1.4 }}
      />
      <T x={x0} y={y + 10} anchor="start" size={fs} color={color} weight={600}>
        {label}
      </T>
    </g>
  );
}

export default function Shor() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [sel, setSel] = useState<Case['id']>('15');
  const [active, bind, previewing] = useHoverPreview(sel);
  const [t, setT] = useState(0);
  const started = useRef(false);

  useEffect(() => {
    if (reduced) {
      setT(END);
      return;
    }
    if (!inView || started.current) return;
    const id = setTimeout(() => {
      started.current = true;
      setT(0.0001);
    }, 400);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  useFrame(
    (dt) => setT((v) => Math.min(END, v + dt)),
    inView && !reduced && started.current && t < END,
  );

  const pick = (id: Case['id']) => {
    started.current = true;
    setSel(id);
    setT(reduced ? END : 0);
  };

  const c = SOLVED.find((s) => s.id === active)!;
  // a hovered tab previews that case finished; the committed one plays on its own clock
  const tt = previewing ? END : t;
  const M = c.xs.length;
  const n = wavesAt(tt, M);

  const chart = (g: Geo, cls: string) => {
    const X0 = g.l;
    const X1 = g.W - g.r;
    const sx = (v: number) => X0 + (v / Q) * (X1 - X0);
    const step = (X1 - X0) / Q;
    const bw = Math.max(1.4, step * 0.68);
    const r2 = (v: number) => Math.round(v * 100) / 100;

    // top row: bars sweep in left to right
    const barH = (v: number) => ((v / (c.N - 1)) * 0.86 + 0.14) * (g.y1b - g.y1t);
    const grow = (x: number) => (tt >= END ? 1 : ramp(tt, (x / Q) * (T_BARS - 0.2), 0.2));

    // bottom row: |Σ w_j e^(2πi k x_j / Q)| / Σ w_j, sampled finely in k
    const ws = c.xs.map((_, j) => clamp(n - j, 0, 1));
    const wsum = ws.reduce((s, w) => s + w, 0);
    const S = g.narrow ? 720 : 1024;
    const plotH = g.y2b - g.y2t;
    let area = '';
    let line = '';
    if (wsum > 1e-4) {
      for (let i = 0; i <= S; i++) {
        const k = (i / S) * Q;
        let re = 0;
        let im = 0;
        for (let j = 0; j < c.xs.length; j++) {
          if (ws[j] === 0) break;
          const ph = (2 * Math.PI * k * c.xs[j]) / Q;
          re += ws[j] * Math.cos(ph);
          im += ws[j] * Math.sin(ph);
        }
        const amp = Math.min(1, Math.hypot(re, im) / wsum);
        const px = r2(sx(k));
        const py = r2(g.y2b - amp * plotH);
        line += `${i ? 'L' : 'M'}${px} ${py}`;
      }
      area = `${line}L${r2(sx(Q))} ${g.y2b}L${r2(sx(0))} ${g.y2b}Z`;
    }
    // the newest wave, drawn faintly while only a few are in (later ones are too fine to see)
    const newest = Math.min(c.xs.length - 1, Math.floor(n));
    const waveOpacity = tt >= END ? 0 : clamp(1.2 - n / 5, 0, 1) * (wsum > 1e-4 ? 1 : 0);
    let wave = '';
    if (waveOpacity > 0) {
      const xw = c.xs[newest];
      for (let i = 0; i <= S; i++) {
        const k = (i / S) * Q;
        const v = (1 + Math.cos((2 * Math.PI * k * xw) / Q)) / 2;
        wave += `${i ? 'L' : 'M'}${r2(sx(k))} ${r2(g.y2b - v * plotH)}`;
      }
    }

    const fs = g.fs;
    const aN = `${c.a}`;
    const showBracket1 = tt >= END ? 1 : ramp(tt, T_BRACKET);
    const showBracket2 = tt >= END ? 1 : ramp(tt, T_PEAKS);
    const spacing = Q / c.r;
    const spacingTxt = Number.isInteger(spacing) ? `${spacing}` : `≈ ${spacing.toFixed(1)}`;

    return (
      <svg
        viewBox={`0 0 ${g.W} ${g.H}`}
        className={cls}
        role="img"
        aria-label={`Top: ${c.a} to the power x mod ${c.N} repeats every ${c.r} steps. Bottom: adding one wave per repeat gives peaks every ${Q}/${c.r}, which reveals the period ${c.r}.`}
      >
        {/* ── 1: the repeating function ── */}
        <T x={X0} y={14} anchor="start" size={fs + 1} color={GT.ink} weight={600}>
          <tspan style={{ fill: GT.eq }}>1 </tspan>
          <Pow a={aN} e="x" fs={fs + 1} />
          {`mod ${c.N}, for x = 0 to ${Q - 1}`}
        </T>
        <line x1={X0} x2={X1} y1={g.y1b} y2={g.y1b} style={{ stroke: GT.lineStrong }} />
        {c.f.map((v, x) => {
          const h = barH(v) * grow(x);
          if (h <= 0) return null;
          const one = v === 1;
          return (
            <rect
              key={x}
              x={sx(x + 0.5) - bw / 2}
              y={g.y1b - h}
              width={bw}
              height={h}
              style={{ fill: one ? GT.eq : GT.muted, opacity: one ? 1 : 0.55 }}
            />
          );
        })}
        {/* a dot under each x whose wave has been added below */}
        {c.xs.map((x, j) =>
          ws[j] > 0 ? (
            <circle
              key={x}
              cx={sx(x + 0.5)}
              cy={g.y1b + 5}
              r={g.narrow ? 1.6 : 2}
              style={{ fill: GT.eq, opacity: ws[j] }}
            />
          ) : null,
        )}
        <Bracket
          x0={sx(0.5)}
          x1={sx(c.r + 0.5)}
          y={g.y1b + 14}
          color={GT.eq}
          fs={fs}
          opacity={showBracket1}
          label={`repeats every r = ${c.r}`}
        />
        <T x={X1} y={g.y1b + 24} anchor="end" size={fs - 1}>
          {g.narrow ? 'value = 1 in orange' : 'orange: the x where the value is 1'}
        </T>

        {/* ── 2: the interference ── */}
        <T x={X0} y={g.y2t - 22} anchor="start" size={fs + 1} color={GT.ink} weight={600}>
          <tspan style={{ fill: GT.eq }}>2 </tspan>
          {g.narrow ? 'one wave per orange x, summed' : 'add one wave for every orange x'}
        </T>
        <line
          x1={X0}
          x2={X1}
          y1={g.y2t}
          y2={g.y2t}
          style={{ stroke: GT.line, strokeDasharray: '3 4' }}
        />
        <line x1={X0} x2={X1} y1={g.y2b} y2={g.y2b} style={{ stroke: GT.lineStrong }} />
        {wave && (
          <path
            d={wave}
            style={{ fill: 'none', stroke: GT.eq, strokeWidth: 1, opacity: 0.55 * waveOpacity }}
          />
        )}
        {area && <path d={area} style={{ fill: GT.aSoft, opacity: 0.55 }} />}
        {line && (
          <path
            d={line}
            style={{ fill: 'none', stroke: GT.a, strokeWidth: 1.6, strokeLinejoin: 'round' }}
          />
        )}
        <T x={X1} y={g.y2t - 8} anchor="end" size={fs - 1}>
          {wsum < 1e-4
            ? ' '
            : tt >= T_PEAKS
              ? g.narrow
                ? 'right answers add up'
                : 'the right answers add up, the rest cancel'
              : `${Math.max(1, Math.ceil(n))} of ${M} waves`}
        </T>
        <Bracket
          x0={sx(0)}
          x1={sx(spacing)}
          y={g.y2b + 14}
          color={GT.a}
          fs={fs}
          opacity={showBracket2}
          label={`peaks every ${Q} / ${c.r} ${spacingTxt.startsWith('≈') ? spacingTxt : `= ${spacingTxt}`}`}
        />
        <T x={X1} y={g.y2b + 24} anchor="end" size={fs - 1}>
          {g.narrow ? 'what you measure →' : 'what the measurement reads, 0 to 127 →'}
        </T>
      </svg>
    );
  };

  const stats = SOLVED.map((s) => {
    const show = (t0: number) => ({
      opacity: tt >= END ? 1 : ramp(tt, t0),
      transition: 'none',
    });
    const [p, q] = [s.p, s.q].sort((u, v) => u - v);
    return (
      <div key={s.id} className="grid grid-cols-3 gap-3 sm:gap-4">
        <div style={show(T_R)}>
          <BigStat
            label="the period"
            value={s.r}
            color={GT.a}
            size="sm"
            sub={`${Q} / peak spacing`}
          />
        </div>
        <div style={show(T_POW)}>
          <BigStat
            label={`${s.a}${sup(s.r / 2)} ± 1`}
            value={`${s.lo}, ${s.hi}`}
            size="sm"
            sub={`${s.a}${sup(s.r / 2)} = ${s.half}`}
          />
        </div>
        <div style={show(T_GCD)}>
          <BigStat
            label={`gcd with ${s.N}`}
            value={`${p} × ${q}`}
            color={GT.opt}
            size="sm"
            sub={`gcd(${s.lo}, ${s.N}) = ${s.p}, gcd(${s.hi}, ${s.N}) = ${s.q}`}
          />
        </div>
      </div>
    );
  });

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-08.3"
      title="finding the period"
      kicker={`N = ${c.N}, a = ${c.a}`}
      hint={
        <Seg
          label="N"
          options={CASES.map((k) => ({ value: k.id, label: k.id }))}
          value={active}
          onChange={pick}
          bind={bind}
        />
      }
      footer={reduced ? 'Shor 1994' : 'Shor 1994 · click again to replay'}
    >
      {chart(WIDE, 'hidden w-full sm:block')}
      {chart(NARROW, 'block w-full sm:hidden')}
      <Reserve className="mt-4" active={SOLVED.findIndex((s) => s.id === active)} items={stats} />
    </Specimen>
  );
}
