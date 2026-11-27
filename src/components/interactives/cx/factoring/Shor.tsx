import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Reserve } from '../../gt/Reserve';
import {
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
 * CX-08.4 · how a quantum computer factors 15, in five steps (Shor 1994).
 *  1. the cycle: 7^x mod 15 for x = 0..31 is 1, 7, 4, 13, 1, ... so the period r = 4.
 *  2. why r helps: 7^4 - 1 = 48 x 50 is a multiple of 15, so gcd(48, 15) = 3, gcd(50, 15) = 5.
 *  3. registers: A has 5 qubits and holds every x = 0..31 at once (1/32 each); B has 4 qubits,
 *     and one run puts 7^x mod 15 into it for every x together. Reading a register gives one
 *     number at random.
 *  4. measure: reading B gives 7, which leaves only x = 1, 5, ..., 29 in A (equal gaps). The
 *     flat 1/32 bars collapse to 1/8 on those eight; reading x now says nothing about the gap.
 *  5. interference: the Fourier transform adds one wave per surviving x. The bars are the
 *     real measurement odds |Σ w_j e^(2πi k x_j / Q)|² / (Q Σ w_j²) while the waves are added
 *     one by one; at the end they are 1/4 at k = 0, 8, 16, 24 and 0 elsewhere. It reads 8:
 *     8/32 = 1/4, so r = 4 (24 also gives r = 4; 0 or 16 mean run again).
 * Q = 32 keeps every slot legible (a real run uses Q ≥ N² = 225). Every number is checked by
 * scratchpad/shor2/verify.py.
 */
const N = 15;
const A = 7;
const Q = 32;
const OUT = 7; // what register B reads in step 4 (not 4, which looks like r)
const READ = 8; // what the final measurement reads in step 5

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
const gcd = (x: number, y: number): number => (y === 0 ? x : gcd(y, x % y));

const F = Array.from({ length: Q }, (_, x) => powmod(A, x, N));
const R = F.indexOf(1, 1); // 4
const CYCLE = F.slice(0, R); // 1, 7, 4, 13
const HALF = A ** (R / 2); // 49
const P1 = gcd(HALF - 1, N); // 3
const P2 = gcd(HALF + 1, N); // 5
const SURV = F.map((v, x) => (v === OUT ? x : -1)).filter((x) => x >= 0); // 1, 5, ..., 29
const SPACING = Q / R; // 8

/* the clock: how long each step animates, then how long auto-play lingers on it */
const DUR = [3.7, 4.2, 3.6, 2.6, 5.2];
const DWELL = [3.2, 4.6, 4.4, 5.6, 0];
const LAST = 4;
const ramp = (t: number, t0: number, d = 0.35) => clamp((t - t0) / d, 0, 1);

/* step 1: cells fill in one by one */
const cellAt = (x: number) => 0.3 + x * 0.095;
/* step 3: register A goes from 0 to all x, then one run fills register B */
const T_REG = 0.15; // both registers appear, all qubits 0
const T_A = 0.7; // A holds every x: its odds rise to 1/32
const T_B = 1.7; // one run computes 7^x mod 15 into B for every x
const T_HINT = 2.7; // reading gives one number
/* step 4 */
const T_DIM = 1.25; // the output is read
const T_COMB = 1.75; // the x-odds rise
/* step 5 */
const T_W0 = 0.3; // waves start
const D_W = 4.0; // all eight added
const T_PEAKS = T_W0 + D_W + 0.1;
const T_READ = T_PEAKS + 0.45;

type Step = 0 | 1 | 2 | 3 | 4;
const STEPS: { label: string; caption: ReactNode }[] = [
  {
    label: '1 cycle',
    caption: (
      <>
        To factor 15, pick a number at random, say 7 (2, 4, 8, 11 or 13 work as well), and take its
        powers mod 15: 1, 7, 4, 13, then back to 1. The pattern repeats every <b>r = 4</b> steps.
      </>
    ),
  },
  {
    label: '2 why r',
    caption: (
      <>
        We know 15 (it is the number we want to split), just not its factors. r = 4 means 7⁴ leaves
        remainder 1 when divided by 15, so 7⁴ − 1 = 2400 is a multiple of 15. Split it as 48 × 50:
        15 goes into the product but into neither half, so its factors are shared between them, and
        a gcd pulls each one out. The catch: for a 617-digit number, finding r is the hard part.
      </>
    ),
  },
  {
    label: '3 qubits',
    caption: (
      <>
        A quantum computer keeps numbers in registers of qubits. Register A has 5 qubits, so it
        holds 0 to 31, and unlike plain bits it can hold all 32 at once, each with chance 1/32. One
        run then puts 7<sup>x</sup> mod 15 into register B for every x together. Reading a register
        gives one ordinary number, at random, and changes what is left.
      </>
    ),
  },
  {
    label: '4 read',
    caption: (
      <>
        Read register B. It gives one number, say 7. That changes register A too: only the x that
        give 7 stay (1, 5, 9, …, 29), with equal gaps, and the bars are their chances now. Reading A
        would give one of them at random, which says nothing about the gap.
      </>
    ),
  },
  {
    label: '5 waves',
    caption: (
      <>
        The quantum Fourier transform makes the surviving x interfere: their waves cancel everywhere
        except at multiples of 32 / r = 8. Measuring now gives 0, 8, 16 or 24, a 1 in 4 chance each;
        here 8. Divide by 32 and reduce: 8/32 = 1/4 and 24/32 = 3/4 both show <b>r = 4</b>{' '}
        underneath. 16/32 = 1/2 and 0/32 hide it, so you run again.
      </>
    ),
  },
];

/** Measurement odds over the 32 slots for a step at time t (bars of steps 3 to 5). */
function odds(view: Step, t: number): { p: number[]; n: number } {
  const comb = Array.from({ length: Q }, (_, k) => (F[k] === OUT ? 1 / SURV.length : 0));
  if (view === 2) {
    const g = ramp(t, T_A, 0.5);
    return { p: comb.map(() => g / Q), n: 0 };
  }
  if (view === 3) {
    // the flat 1/32 collapses onto the x that give 7
    const g = ramp(t, T_COMB, 0.5);
    return { p: comb.map((v) => (1 - g) / Q + g * v), n: 0 };
  }
  // step 5: add the waves one at a time (the newest fades in)
  const n = clamp((t - T_W0) / D_W, 0, 1) * SURV.length;
  const w = SURV.map((_, j) => clamp(n - j, 0, 1));
  const w2 = w.reduce((s, v) => s + v * v, 0);
  const p = comb.map((c, k) => {
    let pw = 1 / Q;
    if (w2 > 1e-9) {
      let re = 0;
      let im = 0;
      for (let j = 0; j < SURV.length; j++) {
        if (w[j] === 0) break;
        const ph = (2 * Math.PI * k * SURV[j]) / Q;
        re += w[j] * Math.cos(ph);
        im += w[j] * Math.sin(ph);
      }
      pw = (re * re + im * im) / (Q * w2);
    }
    // the first wave carries the comb over to the flat line, the rest interfere
    return (1 - w[0]) * c + w[0] * pw;
  });
  return { p, n };
}

type Geo = {
  W: number;
  H: number;
  X0: number;
  X1: number;
  rows: number;
  cellH: number;
  rowTop: number[];
  panelTop: number;
  chartTop: number;
  chartBase: number;
  regY: number[]; // step 3: register A row, register B row, the reading note (centres)
  fs: number;
  narrow: boolean;
};
const WIDE: Geo = {
  W: 640,
  H: 316,
  X0: 10,
  X1: 630,
  rows: 1,
  cellH: 28,
  rowTop: [24],
  panelTop: 110,
  chartTop: 148,
  chartBase: 272,
  regY: [148, 188, 226],
  fs: 10,
  narrow: false,
};
const NARROW: Geo = {
  W: 360,
  H: 354,
  X0: 6,
  X1: 354,
  rows: 2,
  cellH: 26,
  rowTop: [24, 72],
  panelTop: 156,
  chartTop: 194,
  chartBase: 310,
  regY: [192, 230, 266],
  fs: 9.5,
  narrow: true,
};

function Sup({ base, exp, fs }: { base: ReactNode; exp: ReactNode; fs: number }) {
  return (
    <>
      {base}
      <tspan dy={-fs * 0.4} fontSize={fs * 0.72}>
        {exp}
      </tspan>
      <tspan dy={fs * 0.4}>{'\u00a0'}</tspan>
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

const fade = (on: boolean) => ({
  opacity: on ? 1 : 0,
  transition: 'opacity 260ms ease',
  pointerEvents: 'none' as const,
});

export default function Shor() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [st, setSt] = useState<{ step: Step; t: number }>({ step: 0, t: 0 });
  const [view, bind, previewing] = useHoverPreview<Step>(st.step);
  const started = useRef(false);
  const auto = useRef(true);
  const [, kick] = useState(0);

  // auto-play once on scroll-in, a beat after the figure appears
  useEffect(() => {
    if (reduced) {
      auto.current = false;
      setSt((s) => ({ ...s, t: DUR[s.step] }));
      return;
    }
    if (!inView || started.current) return;
    const id = setTimeout(() => {
      started.current = true;
      kick((v) => v + 1);
    }, 400);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  const running =
    inView &&
    !reduced &&
    started.current &&
    !previewing &&
    (st.t < DUR[st.step] || (auto.current && st.step < LAST));
  useFrame((dt) => {
    setSt((s) => {
      const t = s.t + dt;
      if (auto.current && s.step < LAST && t >= DUR[s.step] + DWELL[s.step]) {
        return { step: (s.step + 1) as Step, t: 0 };
      }
      if (!auto.current || s.step === LAST) return { step: s.step, t: Math.min(t, DUR[s.step]) };
      return { step: s.step, t };
    });
  }, running);

  const pick = (s: Step) => {
    auto.current = false;
    started.current = true;
    setSt({ step: s, t: reduced ? DUR[s] : 0 });
  };

  // a hovered tab plays that step from the start on its own clock; leaving it returns to the
  // committed step, which keeps its place
  const [pv, setPv] = useState<{ step: Step; t: number }>({ step: 0, t: 0 });
  const hoverOther = previewing && view !== st.step;
  useEffect(() => {
    if (hoverOther) setPv({ step: view, t: 0 });
  }, [hoverOther, view]);
  useFrame(
    (dt) => setPv((v) => ({ ...v, t: Math.min(v.t + dt, DUR[v.step]) })),
    inView && !reduced && hoverOther && pv.step === view && pv.t < DUR[view],
  );
  const tt = hoverOther
    ? reduced || pv.step !== view
      ? reduced
        ? DUR[view]
        : 0
      : pv.t
    : Math.min(st.t, DUR[view]);
  const { p: bars, n: waves } = view >= 2 ? odds(view, tt) : { p: [], n: 0 };

  const chart = (g: Geo, cls: string) => {
    const { X0, X1, fs } = g;
    const perRow = Q / g.rows;
    const cw = (X1 - X0) / perRow;
    const cell = (x: number) => {
      const row = Math.floor(x / perRow);
      const col = x % perRow;
      return { x: X0 + col * cw, y: g.rowTop[row], cx: X0 + (col + 0.5) * cw, row };
    };
    const lastRow = g.rows - 1;
    const rowBottom = (row: number) => g.rowTop[row] + g.cellH;

    // strip state
    // step 1 fills the cells one by one; step 3 fills them all at once (one run)
    const shown = (x: number) =>
      view === 0 ? ramp(tt, cellAt(x), 0.14) : view === 2 ? ramp(tt, T_B, 0.5) : 1;
    let current = -1; // the cell just filled (step 1), for the ring token
    if (view === 0 && tt < cellAt(Q - 1) + 0.3) {
      for (let x = 0; x < Q; x++) if (tt >= cellAt(x)) current = x;
    }
    const pulse =
      view === 2
        ? Math.sin(Math.PI * clamp((tt - T_B + 0.2) / 1.0, 0, 1))
        : view === 3
          ? Math.sin(Math.PI * clamp(tt / 1.0, 0, 1))
          : 0;
    const dim = view === 3 ? ramp(tt, T_DIM, 0.4) : view === 4 ? 1 : 0;
    const survTicks = view >= 3 ? dim : 0;
    // every x under the strip (register A) in steps 3 and 4, every 4th x otherwise
    const allX = view === 2 ? ramp(tt, T_A, 0.4) : view === 3 ? 1 - dim : 0;
    const wsDot = (j: number) => (view === 4 ? clamp(waves - j, 0, 1) : 0);

    // the bracket of r in the strip, on the last row
    const b0 = lastRow * perRow;
    // the "equal gaps" bracket spans the first two survivors on that row (1 and 5 when B reads 7)
    const [gap0, gap1] = SURV.filter((x) => x >= b0);
    const rBracketOn = view === 0 ? ramp(tt, cellAt(R) + 0.1) : view === 2 ? 0 : 1;

    // the bar chart (steps 3 and 4)
    const bw = Math.max(3, cw * (g.rows === 1 ? 0.62 : 0.31));
    const kx = (k: number) => X0 + ((k + 0.5) / Q) * (X1 - X0);
    const ph = g.chartBase - g.chartTop;
    const barH = (p: number) => (p / 0.25) * ph;
    const peaksOn = view === 4 ? ramp(tt, T_PEAKS) : 0;
    const readOn = view === 4 ? ramp(tt, T_READ) : 0;
    const combLab = view === 3 ? ramp(tt, T_COMB + 0.3) : 0;

    // registers (step 3): qubit boxes start at 0, then hold 0 and 1 at once
    const aOne = view === 2 ? ramp(tt, T_A, 0.4) : 0;
    const bOne = view === 2 ? ramp(tt, T_B, 0.4) : 0;
    const hintOn = view === 2 ? ramp(tt, T_HINT, 0.4) : 0;
    const qs = g.narrow ? 20 : 22;
    const qGap = 4;
    const regX = X0 + (g.narrow ? 16 : 20);
    const regText = regX + 5 * (qs + qGap) + 6;
    const register = (
      name: string,
      n: number,
      y: number,
      color: string,
      one: number,
      text: ReactNode,
    ) => (
      <g>
        <T x={X0} y={y} anchor="start" size={fs + 4} weight={700} color={color}>
          {name}
        </T>
        {Array.from({ length: n }, (_, i) => {
          const x = regX + i * (qs + qGap);
          return (
            <g key={i}>
              <rect
                x={x}
                y={y - qs / 2}
                width={qs}
                height={qs}
                style={{ fill: GT.paper, stroke: color, strokeWidth: 1.2 }}
              />
              <T x={x + qs / 2} y={y - (qs / 4.4) * one} size={fs} weight={600} color={color}>
                0
              </T>
              <g style={{ opacity: one }}>
                <T x={x + qs / 2} y={y + qs / 4.4} size={fs} weight={600} color={color}>
                  1
                </T>
              </g>
            </g>
          );
        })}
        <g style={{ opacity: one }}>
          <T x={regText} y={y} anchor="start" size={fs} color={GT.inkSoft}>
            {text}
          </T>
        </g>
      </g>
    );

    // ring (step 1)
    const ringCy = (g.panelTop + g.H) / 2 + 2;
    const ringR = g.narrow ? 56 : 60;
    const nodeR = 17;
    const ang = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / R;
    const nodeXY = (i: number) => ({
      x: g.W / 2 + ringR * Math.cos(ang(i)),
      y: ringCy + ringR * Math.sin(ang(i)),
    });
    const ringDone = view === 0 ? ramp(tt, cellAt(Q - 1) + 0.25) : 1;

    // gcd lines (step 2)
    const L = g.narrow ? [186, 208, 230, 252, 274, 322] : [140, 164, 188, 212, 236, 284];
    const lineOn = (i: number) => (view === 1 ? ramp(tt, 0.25 + i * 0.6) : 0);
    const fsEq = g.narrow ? 11.5 : 13;

    const panelTitle = [
      'the cycle: times 7 each step, mod 15',
      'if you know r',
      'two registers of qubits',
      'register A: chance of each x',
      tt < T_PEAKS
        ? `adding one wave per x: ${Math.max(1, Math.ceil(waves))} of ${SURV.length}`
        : g.narrow
          ? 'after the transform'
          : 'after the Fourier transform',
    ][view];
    const status =
      view === 2
        ? tt < T_B
          ? ''
          : g.narrow
            ? 'B in the cells, A below'
            : 'register B in the cells, register A below'
        : view === 3
          ? tt < T_DIM
            ? 'reading register B'
            : `register B reads ${OUT}`
          : '';

    return (
      <svg
        viewBox={`0 0 ${g.W} ${g.H}`}
        className={cls}
        role="img"
        aria-label={`7 to the power x mod 15 repeats 1, 7, 4, 13 with period r = 4. gcd(48, 15) = 3 and gcd(50, 15) = 5. A quantum computer holds every x from 0 to 31 at once in register A (5 qubits) and computes 7 to the x mod 15 into register B (4 qubits) for all of them in one run. Reading register B gives 7, leaving x = 1, 5, ..., 29; the Fourier transform turns that into peaks at 0, 8, 16, 24, and a reading of 8 gives r = 4.`}
      >
        <defs>
          <marker
            id={`shor-arrow-${g.narrow ? 'n' : 'w'}`}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M0 0 L10 5 L0 10 z" style={{ fill: GT.muted }} />
          </marker>
        </defs>
        {/* ── the strip: 7^x mod 15 for x = 0 .. 31 ── */}
        <T x={X0} y={10} anchor="start" size={fs + 1} color={GT.ink} weight={600}>
          <Sup base="7" exp="x" fs={fs + 1} />
          {`mod ${N}, for x = 0 to ${Q - 1}`}
        </T>
        <T x={X1} y={10} anchor="end" size={fs} color={dim > 0.5 ? GT.b : GT.muted} weight={600}>
          {status}
        </T>
        {g.rowTop.map((y, row) => (
          <rect
            key={`pulse${row}`}
            x={X0 - 2}
            y={y - 2}
            width={X1 - X0 + 4}
            height={g.cellH + 4}
            style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2, opacity: pulse }}
          />
        ))}
        {F.map((v, x) => {
          const c = cell(x);
          const s = shown(x);
          const one = v === 1;
          const hit = v === OUT;
          const fadeOut = hit ? 1 : 1 - 0.72 * dim;
          const isCur = x === current;
          return (
            <g key={x} style={{ opacity: fadeOut }}>
              <rect
                x={c.x + 1}
                y={c.y}
                width={cw - 2}
                height={g.cellH}
                style={{
                  fill: isCur ? GT.ink : GT.paper,
                  stroke: isCur ? GT.ink : GT.line,
                  strokeWidth: 1,
                }}
              />
              {hit && dim > 0 && (
                <rect
                  x={c.x + 1}
                  y={c.y}
                  width={cw - 2}
                  height={g.cellH}
                  style={{ fill: GT.bSoft, stroke: GT.b, strokeWidth: 1.2, opacity: dim }}
                />
              )}
              {one && !isCur && (
                <rect
                  x={c.x + 1}
                  y={c.y + g.cellH - 3}
                  width={cw - 2}
                  height={3}
                  style={{ fill: GT.eq, opacity: s }}
                />
              )}
              <g style={{ opacity: s }}>
                <T
                  x={c.cx}
                  y={c.y + g.cellH / 2}
                  size={fs + 1}
                  weight={one || (hit && dim > 0.5) ? 700 : 500}
                  color={isCur ? GT.paper : hit && dim > 0.5 ? GT.b : one ? GT.eq : GT.ink}
                >
                  {v}
                </T>
              </g>
              {/* a dot under each x whose wave is in the sum (step 4) */}
              {hit && view === 4 && wsDot(SURV.indexOf(x)) > 0 && (
                <circle
                  cx={c.cx}
                  cy={rowBottom(c.row) + 4}
                  r={2}
                  style={{ fill: GT.a, opacity: wsDot(SURV.indexOf(x)) }}
                />
              )}
            </g>
          );
        })}
        {/* x labels under the strip: every 4th x, then every x (register A), then the survivors */}
        {Array.from({ length: Q / R }, (_, i) => i * R).map((x) => {
          const c = cell(x);
          const on = view <= 1 ? shown(x) : view === 2 ? 1 - allX : 0;
          return (
            <g key={`t${x}`} style={{ opacity: on }}>
              <T x={c.cx} y={rowBottom(c.row) + 12} size={fs - 1}>
                {x}
              </T>
            </g>
          );
        })}
        {F.map((_, x) => {
          const c = cell(x);
          return (
            <g key={`all${x}`} style={{ opacity: allX }}>
              <T x={c.cx} y={rowBottom(c.row) + 12} size={fs - 1} color={GT.b}>
                {x}
              </T>
            </g>
          );
        })}
        {SURV.map((x) => {
          const c = cell(x);
          return (
            <g key={`s${x}`} style={{ opacity: survTicks }}>
              <T x={c.cx} y={rowBottom(c.row) + 12} size={fs - 1} color={GT.b} weight={600}>
                {x}
              </T>
            </g>
          );
        })}
        <Bracket
          x0={cell(view >= 3 ? gap0 : b0).cx}
          x1={cell(view >= 3 ? gap1 : b0 + R).cx}
          y={rowBottom(lastRow) + 22}
          color={view >= 3 ? GT.b : GT.eq}
          fs={fs}
          opacity={rBracketOn}
          label={view >= 3 ? 'equal gaps' : `r = ${R}: back to 1`}
        />

        {/* ── the panel ── */}
        <T x={X0} y={g.panelTop} anchor="start" size={fs + 1} color={GT.ink} weight={600}>
          {panelTitle}
        </T>
        <g style={fade(view === 4)}>
          <T x={X1} y={g.panelTop} anchor="end" size={fs + 0.5} color={GT.a} weight={700}>
            <tspan style={{ opacity: readOn }}>
              {g.narrow
                ? `reads ${READ}, so r = ${R}`
                : `reads ${READ}: ${READ}/${Q} = 1/${R}, so r = ${R}`}
            </tspan>
          </T>
        </g>

        {/* step 1: the cycle as a ring */}
        <g style={fade(view === 0)}>
          {CYCLE.map((_, i) => {
            const a0 = ang(i) + (nodeR + 3) / ringR;
            const a1 = ang(i + 1) - (nodeR + 6) / ringR;
            const p0 = { x: g.W / 2 + ringR * Math.cos(a0), y: ringCy + ringR * Math.sin(a0) };
            const p1 = { x: g.W / 2 + ringR * Math.cos(a1), y: ringCy + ringR * Math.sin(a1) };
            const am = (ang(i) + ang(i + 1)) / 2;
            return (
              <g key={`arc${i}`}>
                <path
                  d={`M${p0.x} ${p0.y} A${ringR} ${ringR} 0 0 1 ${p1.x} ${p1.y}`}
                  markerEnd={`url(#shor-arrow-${g.narrow ? 'n' : 'w'})`}
                  style={{ fill: 'none', stroke: GT.muted, strokeWidth: 1.3 }}
                />
                <T
                  x={g.W / 2 + (ringR + 16) * Math.cos(am)}
                  y={ringCy + (ringR + 16) * Math.sin(am)}
                  size={fs - 0.5}
                >
                  ×7
                </T>
              </g>
            );
          })}
          {CYCLE.map((v, i) => {
            const { x, y } = nodeXY(i);
            const on = current >= 0 && F[current] === v;
            const one = v === 1;
            return (
              <g key={`n${v}`}>
                <circle
                  cx={x}
                  cy={y}
                  r={nodeR}
                  style={{
                    fill: on ? GT.ink : GT.paper,
                    stroke: one ? GT.eq : GT.lineStrong,
                    strokeWidth: one ? 2 : 1.2,
                  }}
                />
                <T
                  x={x}
                  y={y + 0.5}
                  size={fs + 3}
                  weight={700}
                  color={on ? GT.paper : one ? GT.eq : GT.ink}
                >
                  {v}
                </T>
              </g>
            );
          })}
          <g style={{ opacity: ringDone }}>
            <T x={g.W / 2} y={ringCy + 1} size={22} font="display" color={GT.eq}>
              r = {R}
            </T>
          </g>
        </g>

        {/* step 2: from r to the factors */}
        <g style={fade(view === 1)}>
          <g style={{ opacity: lineOn(0) }}>
            <T x={g.W / 2} y={L[0]} size={fsEq} color={GT.ink}>
              {`r = ${R}, so `}
              <Sup base="7" exp="4" fs={fsEq} />
              {`mod ${N} = 1`}
            </T>
          </g>
          <g style={{ opacity: lineOn(1) }}>
            <T x={g.W / 2} y={L[1]} size={fsEq} color={GT.ink}>
              {`so ${N} divides `}
              <Sup base="7" exp="4" fs={fsEq} />
              {`− 1 = ${A ** R - 1}`}
            </T>
          </g>
          <g style={{ opacity: lineOn(2) }}>
            <T x={g.W / 2} y={L[2]} size={fsEq} color={GT.ink}>
              <Sup base="7" exp="4" fs={fsEq} />
              {'− 1 = ('}
              <Sup base="7" exp="2" fs={fsEq} />
              {'− 1)('}
              <Sup base="7" exp="2" fs={fsEq} />
              {`+ 1) = ${HALF - 1} × ${HALF + 1}`}
            </T>
          </g>
          <g style={{ opacity: lineOn(3) }}>
            <T x={g.W / 2} y={L[3]} size={fsEq} color={GT.inkSoft}>
              {`${N} divides ${HALF - 1} × ${HALF + 1} but neither alone: it splits`}
            </T>
          </g>
          <g style={{ opacity: lineOn(4) }}>
            <T x={g.W / 2} y={L[4]} size={fsEq} color={GT.ink}>
              {`gcd(${HALF - 1}, ${N}) = `}
              <tspan style={{ fill: GT.opt, fontWeight: 700 }}>{P1}</tspan>
              <tspan dx={fsEq * 2}>{`gcd(${HALF + 1}, ${N}) = `}</tspan>
              <tspan style={{ fill: GT.opt, fontWeight: 700 }}>{P2}</tspan>
            </T>
          </g>
          <g style={{ opacity: lineOn(5) }}>
            <T x={g.W / 2} y={L[5]} size={g.narrow ? 30 : 34} font="display" color={GT.opt}>
              {`${N} = ${P1} × ${P2}`}
            </T>
          </g>
        </g>

        {/* step 3: the two registers */}
        <g style={fade(view === 2)}>
          {register(
            'A',
            5,
            g.regY[0],
            GT.b,
            aOne,
            g.narrow
              ? '5 qubits: x = 0 to 31, all at once'
              : '5 qubits: every x from 0 to 31, all at once',
          )}
          {register(
            'B',
            4,
            g.regY[1],
            GT.ink,
            bOne,
            <>
              {'4 qubits: '}
              <Sup base="7" exp="x" fs={fs} />
              {g.narrow ? 'mod 15 for each x' : 'mod 15 for every x, in one run'}
            </>,
          )}
          <g style={{ opacity: hintOn }}>
            <T x={X0} y={g.regY[2]} anchor="start" size={fs} color={GT.muted}>
              {g.narrow
                ? 'reading one gives a single number, at random'
                : 'reading a register gives one number, at random, and changes it'}
            </T>
          </g>
        </g>

        {/* steps 3 to 5: the odds of each reading */}
        <g style={fade(view >= 2)}>
          <line
            x1={X0}
            x2={X1}
            y1={g.chartBase}
            y2={g.chartBase}
            style={{ stroke: GT.lineStrong }}
          />
          {bars.map((p, k) => {
            const h = barH(p);
            if (h < 0.05) return null;
            return (
              <rect
                key={k}
                x={kx(k) - bw / 2}
                y={g.chartBase - h}
                width={bw}
                height={h}
                style={{
                  fill: view <= 3 ? GT.b : GT.a,
                  opacity: view === 4 && readOn > 0 && k !== READ ? 1 - 0.45 * readOn : 1,
                }}
              />
            );
          })}
          {/* odds labels */}
          {view === 2 && (
            <g style={{ opacity: aOne }}>
              <T x={X1} y={g.chartBase - barH(1 / Q) - 9} anchor="end" size={fs - 1} color={GT.b}>
                {`1/${Q} each`}
              </T>
            </g>
          )}
          {view === 3 &&
            SURV.map((x) => (
              <g key={`c${x}`} style={{ opacity: combLab }}>
                <T x={kx(x)} y={g.chartBase - barH(1 / 8) - 8} size={fs - 1} color={GT.b}>
                  1/8
                </T>
              </g>
            ))}
          {view === 4 &&
            Array.from({ length: R }, (_, i) => i * SPACING).map((k) => (
              <g key={`p${k}`} style={{ opacity: peaksOn }}>
                <T x={kx(k)} y={g.chartTop - 8} size={fs - 1} color={GT.a}>
                  1/4
                </T>
              </g>
            ))}
          {/* the reading */}
          {view === 4 && (
            <path
              d={`M${kx(READ) - 5} ${g.chartTop - 26} L${kx(READ) + 5} ${g.chartTop - 26} L${kx(READ)} ${g.chartTop - 19} Z`}
              style={{ fill: GT.ink, opacity: readOn }}
            />
          )}
          {/* readings along the axis */}
          {view === 2 &&
            Array.from({ length: Q / R }, (_, i) => i * R).map((x) => (
              <g key={`r${x}`} style={{ opacity: aOne }}>
                <T x={kx(x)} y={g.chartBase + 11} size={fs - 1} color={GT.b}>
                  {x}
                </T>
              </g>
            ))}
          {view === 3 &&
            SURV.map((x) => (
              <g key={`a${x}`} style={{ opacity: combLab }}>
                <T x={kx(x)} y={g.chartBase + 11} size={fs - 1} color={GT.b}>
                  {x}
                </T>
              </g>
            ))}
          {view === 4 &&
            Array.from({ length: R }, (_, i) => i * SPACING).map((k) => (
              <g key={`k${k}`}>
                <T
                  x={kx(k)}
                  y={g.chartBase + 11}
                  size={fs - 1}
                  color={peaksOn > 0 ? GT.a : GT.muted}
                  weight={k === READ && readOn > 0 ? 700 : 400}
                >
                  {k}
                </T>
              </g>
            ))}
          {view === 4 && (
            <Bracket
              x0={kx(0)}
              x1={kx(SPACING)}
              y={g.chartBase + 26}
              color={GT.a}
              fs={fs}
              opacity={peaksOn}
              label={`every ${Q} / r = ${SPACING}`}
            />
          )}
        </g>
      </svg>
    );
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-08.4"
      title="a quantum computer factors 15"
      hint={
        <Seg
          options={STEPS.map((s, i) => ({ value: String(i), label: s.label }))}
          value={String(view)}
          onChange={(v) => pick(Number(v) as Step)}
          bind={(v) => bind(Number(v) as Step)}
        />
      }
      footer="Shor 1994 · 32 slots here, a real run uses at least 15² = 225"
    >
      {chart(WIDE, 'hidden w-full select-none sm:block')}
      {chart(NARROW, 'block w-full select-none sm:hidden')}
      <Reserve
        className="mt-3"
        active={view}
        items={STEPS.map((s, i) => (
          <p key={i} className="text-[0.85rem] leading-snug text-ink-soft">
            {s.caption}
          </p>
        ))}
      />
    </Specimen>
  );
}
