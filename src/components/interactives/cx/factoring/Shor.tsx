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
  useTween,
} from '../../gt/shared';

/**
 * CX-08.4 · how a quantum computer factors 15, in six steps (Shor 1994).
 *  1. the cycle: 7^x mod 15 for x = 0..31 is 1, 7, 4, 13, 1, ... so the period r = 4.
 *  2. why r helps: 7^4 - 1 = 48 x 50 is a multiple of 15, so gcd(48, 15) = 3, gcd(50, 15) = 5.
 *  3. X, Y: two registers linked by Y = 7^X mod 15. X holds every x = 0..31 at once (1/32
 *     each), so Y holds 1, 7, 4, 13 (1/4 each). A chip shows the contrast: read X first and
 *     you get one x (6), and Y is then sure to be 7^6 mod 15 = 4, so nothing is left to find.
 *  4. read Y: it gives 7, which leaves x = 1, 5, ..., 29 in X (1/8 each, equal gaps). A dashed
 *     ghost shows another run (Y = 4 leaves 2, 6, ..., 30): same gaps, random start.
 *  5. arrows (the Fourier transform): for a reading k, each surviving x adds a unit arrow
 *     turned k·x/32 of a full turn (drawn from a common centre, "×8" where they coincide) and
 *     the blue arrow is their average; the chance of k is |sum|² / (8 · 32) = |average|² / 4.
 *     That is 1/4 at k = 0, 8, 16, 24 and 0 elsewhere. Auto-play sweeps k once round (0 to 31,
 *     then on to 8) and stops at 8; hover or tap a bar to pick k. The dashed arrows are the
 *     Y = 4 run (x = 2, 6, ...): every arrow turned by k/32, same length, same chance.
 *  6. read X: it reads 8. 8/32 = 1/4, so r = 4 (24/32 = 3/4 too; 0 and 16/32 = 1/2 mean run
 *     again), which is what step 2 needed: 15 = 3 × 5.
 * Q = 32 keeps every slot legible (a real run uses Q ≥ N² = 225). Every number is checked by
 * scratchpad/shor4/verify.py.
 */
const N = 15;
const A = 7;
const Q = 32;
const OUT = 7; // what Y reads in step 4 (not 4, which looks like r)
const OTHER = 4; // the other run in steps 4 and 5 (x = 2, 6, ..., 30)
const XFIRST = 6; // the x the contrast reads first in step 3
const READ = 8; // what the final reading of X gives in step 6

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
const GHOST = F.map((v, x) => (v === OTHER ? x : -1)).filter((x) => x >= 0); // 2, 6, ..., 30
const SPACING = Q / R; // 8
const YFIRST = F[XFIRST]; // 4

/** The arrows of a reading k (continuous): head-to-tail sum of e^(2πi k x / Q) over xs. */
function sumAt(k: number, xs: number[]) {
  let re = 0;
  let im = 0;
  for (const x of xs) {
    const a = (2 * Math.PI * k * x) / Q;
    re += Math.cos(a);
    im += Math.sin(a);
  }
  return { re, im };
}
/* the chance of each reading k after the transform: |sum|² / (8 · 32) */
const PK = Array.from({ length: Q }, (_, k) => {
  const { re, im } = sumAt(k, SURV);
  return (re * re + im * im) / (SURV.length * Q);
});
const wrap = (d: number) => ((((d + Q / 2) % Q) + Q) % Q) - Q / 2;

/* the clock: how long each step animates, then how long auto-play lingers on it */
const DUR = [3.7, 4.2, 3.2, 3.3, 7.0, 3.1];
const DWELL = [3.2, 4.6, 4.4, 5.4, 4.0, 0];
const LAST = 5;
const ramp = (t: number, t0: number, d = 0.35) => clamp((t - t0) / d, 0, 1);

/* step 1: cells fill in one by one */
const cellAt = (x: number) => 0.3 + x * 0.095;
/* step 3 */
const T3_X = 0.25; // X holds every x: its chances rise to 1/32
const T3_EQ = 0.9; // the equation linking them
const T3_Y = 1.3; // one run fills Y
const T3_CHIP = 2.3; // "what if you read X first?"
/* step 4 */
const T4_READ = 0.3; // Y is read
const T_DIM = 0.9; // the strip keeps only the 7s
const T_COMB = 1.4; // X collapses to 1/8 on the survivors
const T4_GHOST = 2.3; // another run, another start
/* step 5: k sweeps once round (0 to 31, then 0 to 8) at a steady pace and eases into 8 */
const T5_SW = 0.55;
const SLOT = 0.13; // seconds per k
const K_LIN = Q + READ - 4; // 36: steady up to here, then slow down over 4 slots
const T5_END = T5_SW + K_LIN * SLOT + 8 * SLOT;
const T5_GHOST = T5_END + 0.15;
function sweepK(t: number) {
  const t1 = T5_SW + K_LIN * SLOT;
  let k: number;
  if (t <= t1) k = (t - T5_SW) / SLOT;
  else {
    const u = clamp((t - t1) / (8 * SLOT), 0, 1);
    k = K_LIN + 4 * (2 * u - u * u);
  }
  // linger a little on each whole k (never stopping), so each reading's arrows can be seen
  const n = Math.floor(k);
  const f = k - n;
  return n + 0.15 * f + 0.85 * f * f * f * (f * (6 * f - 15) + 10);
}
/* step 6 */
const T6_READ = 0.3;
const T6_FRAC = 0.9;
const T6_R = 1.5;
const T6_BACK = 2.2;

type Step = 0 | 1 | 2 | 3 | 4 | 5;
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
    label: '3 X, Y',
    caption: (
      <>
        A quantum computer has two registers, X and Y. X holds every x from 0 to 31 at once, each
        with chance 1/32. One run computes Y = 7<sup>X</sup> mod 15 for all of them together, so Y
        holds 1, 7, 4 and 13, each with chance 1/4. Reading a register gives one number, at random.
      </>
    ),
  },
  {
    label: '4 read Y',
    caption: (
      <>
        Read Y: it says 7. Now X keeps only the x that give 7: 1, 5, 9, …, 29, each with chance 1/8,
        and equal gaps. The start is random, though: another run might read 4 and leave 2, 6, 10 and
        so on. One x from one run says nothing about the gap.
      </>
    ),
  },
  {
    label: '5 arrows',
    caption: (
      <>
        The Fourier transform as arrows: for a reading k, each x left in X adds an arrow turned
        k·x/32 of a full turn, and the blue arrow is their average. Lined up, k is likely; fanned
        out, they cancel. At k = 8 neighbours differ by a full turn, so all eight agree. Another
        start turns every arrow alike, so the chances stay put. Hover or tap a bar.
      </>
    ),
  },
  {
    label: '6 read X',
    caption: (
      <>
        Now read X. It lands on a peak, here 8. Divide by 32 and reduce: 8/32 = 1/4, and the bottom
        number is <b>r = 4</b> (24/32 = 3/4 gives 4 too; 0 and 16/32 = 1/2 hide it, so run again).
        That r is what step 2 needed: 15 = 3 × 5.
      </>
    ),
  },
];

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
  mainX1: number; // the main chart (x or k) runs X0..mainX1
  sideX0: number; // the side box (Y, the arrows, the result) runs sideX0..X1
  arrow: { cx: number; cy: number; u: number };
  eqSize: number;
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
  chartTop: 150,
  chartBase: 272,
  mainX1: 400,
  sideX0: 440,
  arrow: { cx: 535, cy: 212, u: 70 },
  eqSize: 24,
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
  chartBase: 300,
  mainX1: 204,
  sideX0: 222,
  arrow: { cx: 288, cy: 252, u: 52 },
  eqSize: 17,
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
      <tspan dy={fs * 0.4}>{' '}</tspan>
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

/** A straight arrow with a drawn head (markers scale with the stroke, these do not). */
function Arrow({
  x1,
  y1,
  x2,
  y2,
  color,
  width,
  head,
  dash,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  width: number;
  head: number;
  dash?: string;
}) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len < 0.5) return null;
  const ux = dx / len;
  const uy = dy / len;
  const h = Math.min(head, len * 0.6);
  const bx = x2 - ux * h;
  const by = y2 - uy * h;
  const w = h * 0.55;
  return (
    <g>
      <line
        x1={x1}
        y1={y1}
        x2={bx + ux * 0.5}
        y2={by + uy * 0.5}
        style={{ stroke: color, strokeWidth: width, strokeDasharray: dash, strokeLinecap: 'round' }}
      />
      {!dash && (
        <path
          d={`M${x2} ${y2} L${bx - uy * w} ${by + ux * w} L${bx + uy * w} ${by - ux * w} Z`}
          style={{ fill: color }}
        />
      )}
    </g>
  );
}

const fade = (on: boolean) => ({
  opacity: on ? 1 : 0,
  transition: 'opacity 260ms ease',
  pointerEvents: on ? undefined : ('none' as const),
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
    !(previewing && view !== st.step) &&
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

  // step 3: "what if you read X first?" (hover previews it, a tap or click keeps it)
  const [xfHover, setXfHover] = useState(false);
  const [xfPin, setXfPin] = useState(false);
  // step 5: pick a reading k by hovering or tapping its bar
  const [kHover, setKHover] = useState<number | null>(null);
  const [kPin, setKPin] = useState<number | null>(null);
  const [explored, setExplored] = useState(false);
  useEffect(() => {
    setXfHover(false);
    setXfPin(false);
    setKHover(null);
    setKPin(null);
    setExplored(false);
  }, [view, st.step]);
  const xf = useTween(view === 2 && (xfHover || xfPin) ? 1 : 0, reduced, 7);

  // k: the sweep's clock, or the picked bar, reached by a short glide the shorter way round
  const kRaw = view === 4 ? sweepK(tt) : READ;
  const kPicked = kHover ?? kPin;
  const kTarget = kPicked ?? ((clamp(kRaw, 0, Q + READ) % Q) + Q) % Q;
  const [kGlide, setKGlide] = useState<number | null>(null);
  const kShown = useRef(kTarget);
  const kTargetRef = useRef(kTarget);
  kTargetRef.current = kTarget;
  const firstPick = useRef(true);
  useEffect(() => {
    if (firstPick.current) {
      firstPick.current = false;
      return;
    }
    if (!reduced) setKGlide(kShown.current);
  }, [kPicked, reduced]);
  useFrame(
    (dt) =>
      setKGlide((g) => {
        if (g === null) return null;
        const d = wrap(kTargetRef.current - g);
        if (Math.abs(d) < 0.01) return null;
        return g + d * Math.min(1, dt * 9);
      }),
    kGlide !== null && !reduced,
  );
  const kd = (((kGlide ?? kTarget) % Q) + Q) % Q;
  kShown.current = kd;
  const kInt = Math.round(kd) % Q;

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
      view === 0 ? ramp(tt, cellAt(x), 0.14) : view === 2 ? ramp(tt, T3_Y, 0.5) : 1;
    let current = -1; // the cell just filled (step 1), for the ring token
    if (view === 0 && tt < cellAt(Q - 1) + 0.3) {
      for (let x = 0; x < Q; x++) if (tt >= cellAt(x)) current = x;
    }
    const pulse =
      view === 2
        ? Math.sin(Math.PI * clamp((tt - T3_Y + 0.2) / 1.0, 0, 1))
        : view === 3
          ? Math.sin(Math.PI * clamp(tt / 1.0, 0, 1))
          : 0;
    // which cells stay lit, and how far the rest fade
    const dim = view === 2 ? xf : view === 3 ? ramp(tt, T_DIM, 0.4) : view >= 4 ? 1 : 0;
    const isHit = (x: number) => (view === 2 ? x === XFIRST : F[x] === OUT);
    const survTicks = view >= 3 ? dim : 0;
    // every x under the strip (X holds them all) in steps 3 and 4, every 4th x otherwise
    const allX = view === 2 ? ramp(tt, T3_X, 0.4) : view === 3 ? 1 - dim : 0;

    // the bracket of r in the strip, on the last row
    const b0 = lastRow * perRow;
    // the "equal gaps" bracket spans the first two survivors on that row (1 and 5 when Y reads 7)
    const [gap0, gap1] = SURV.filter((x) => x >= b0);
    const rBracketOn =
      view === 0 ? ramp(tt, cellAt(R) + 0.1) : view === 1 ? 1 : view >= 3 ? dim : 0;

    // ── the panel: main chart over 0..31 on the left, side box on the right ──
    const mW = g.mainX1 - X0;
    const kx = (k: number) => X0 + ((k + 0.5) / Q) * mW;
    const slot = mW / Q;
    const bw = Math.max(3, slot * 0.6);
    const ph = g.chartBase - g.chartTop;
    const barH = (p: number) => Math.min(ph, (p / 0.25) * ph); // the main chart tops out at 1/4
    const sideW = X1 - g.sideX0;
    const yx = (i: number) => g.sideX0 + ((i + 0.5) / R) * sideW;
    const ybw = (sideW / R) * 0.46;
    const yH = (p: number) => p * ph; // the Y chart tops out at 1
    const sideMid = (g.sideX0 + X1) / 2;

    // step 3: X holds every x; Y holds the four values; the contrast reads X first
    const xOn = view === 2 ? ramp(tt, T3_X, 0.5) : 0;
    const yOn = view === 2 ? ramp(tt, T3_Y, 0.5) : view === 3 ? 1 : 0;
    const eqOn = view === 2 ? ramp(tt, T3_EQ, 0.4) : 0;
    const chipOn = view === 2 ? ramp(tt, T3_CHIP, 0.4) : 0;
    // step 4: Y reads 7, X collapses, another run is ghosted
    const yRead = view === 3 ? ramp(tt, T4_READ, 0.5) : 0;
    const comb = view === 3 ? ramp(tt, T_COMB, 0.5) : view === 4 ? 1 - ramp(tt, 0.05, 0.45) : 0;
    const ghostOn = view === 3 ? ramp(tt, T4_GHOST, 0.4) : 0;
    const xBars = Array.from({ length: Q }, (_, x) => {
      if (view === 2) {
        const base = xOn / Q;
        return (1 - xf) * base + xf * (x === XFIRST ? 1 : 0);
      }
      if (view === 3) return (1 - comb) / Q + comb * (F[x] === OUT ? 1 / SURV.length : 0);
      if (view === 4) return F[x] === OUT ? 1 / SURV.length : 0;
      return 0;
    });
    const yBars = CYCLE.map((v) => {
      const base = yOn / R;
      if (view === 2) return (1 - xf) * base + xf * (v === YFIRST ? 1 : 0);
      return (1 - yRead) * base + yRead * (v === OUT ? 1 : 0);
    });
    // step 5: the k chart fills in as the sweep passes; step 6 reads it
    const fill = (k: number) =>
      view === 5 || explored || kPicked !== null ? 1 : clamp((kRaw - k + 0.3) / 1.5, 0, 1);
    const kBars = PK.map((p, k) => (view >= 4 ? p * fill(k) : 0));
    // (only where the arrows mostly agree: a fan over a fan would just be noise)
    const ghost5 =
      view === 4
        ? (kPicked !== null || explored ? 1 : ramp(tt, T5_GHOST, 0.5)) *
          clamp(
            (Math.hypot(sumAt(kd, SURV).re, sumAt(kd, SURV).im) / SURV.length - 0.4) * 2.5,
            0,
            1,
          )
        : 0;
    const readOn = view === 5 ? ramp(tt, T6_READ) : 0;
    const fracOn = view === 5 ? ramp(tt, T6_FRAC) : 0;
    const rOn = view === 5 ? ramp(tt, T6_R) : 0;
    const backOn = view === 5 ? ramp(tt, T6_BACK) : 0;

    // the arrows of the current k: one per x from a common centre, and their average (blue),
    // whose length is 1 when all eight line up and 0 when they fan out evenly
    const { cx, cy, u: AL } = g.arrow;
    const spokes = (xs: number[]) =>
      xs.map((x) => {
        const a = (2 * Math.PI * kd * x) / Q;
        return { x: cx + AL * Math.cos(a), y: cy - AL * Math.sin(a), a };
      });
    const main = spokes(SURV);
    const other = spokes(GHOST);
    const avg = sumAt(kd, SURV);
    const avgEnd = {
      x: cx + (AL * avg.re) / SURV.length,
      y: cy - (AL * avg.im) / SURV.length,
    };
    // arrows that coincide get a count at their tip ("×8" when all line up)
    const stacks: { x: number; y: number; a: number; n: number }[] = [];
    for (const p of main) {
      const s = stacks.find((q) => Math.hypot(q.x - p.x, q.y - p.y) < 1.5);
      if (s) s.n++;
      else stacks.push({ ...p, n: 1 });
    }
    const chance = PK[kInt];
    const chanceText = chance > 0.2 ? '1/4' : '0';

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
      'X: every x at once',
      'X: only x that give 7 stay',
      'chance of each reading k',
      'read X, divide by 32',
    ][view];
    const status =
      view === 2
        ? tt < T3_Y
          ? ''
          : g.narrow
            ? 'Y in the cells, X below'
            : 'Y in the cells, X below them'
        : view === 3
          ? tt < T_DIM
            ? 'reading Y'
            : `Y reads ${OUT}`
          : '';
    const tri = (x: number, y: number, on: number) => (
      <path
        d={`M${x - 5} ${y} L${x + 5} ${y} L${x} ${y + 7} Z`}
        style={{ fill: GT.ink, opacity: on }}
      />
    );
    // the current k on the axis; the fixed ticks step aside when it comes close
    const near = (k: number) => Math.abs(wrap(kd - k)) < (g.narrow ? 2.2 : 1.5);

    return (
      <svg
        viewBox={`0 0 ${g.W} ${g.H}`}
        className={cls}
        role="img"
        aria-label={`7 to the power x mod 15 repeats 1, 7, 4, 13 with period r = 4, and gcd(48, 15) = 3, gcd(50, 15) = 5. A quantum computer has two registers linked by Y = 7 to the X mod 15: X holds every x from 0 to 31 at once, so Y holds 1, 7, 4 and 13. Reading Y gives 7 and leaves x = 1, 5, ..., 29 in X. For each reading k every surviving x adds an arrow turned k x / 32 of a turn; they line up only at k = 0, 8, 16 and 24, each with chance 1/4. Reading X gives 8, and 8/32 = 1/4 gives r = 4.`}
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
          const hit = isHit(x);
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
            </g>
          );
        })}
        {/* x labels under the strip: every 4th x, then every x (X), then the survivors */}
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
          const on = allX * (view === 2 && x !== XFIRST ? 1 - 0.72 * xf : 1);
          return (
            <g key={`all${x}`} style={{ opacity: on }}>
              <T
                x={c.cx}
                y={rowBottom(c.row) + 12}
                size={fs - 1}
                color={GT.b}
                weight={view === 2 && x === XFIRST && xf > 0.5 ? 700 : 400}
              >
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

        {/* ── steps 3 to 6: the main chart (x, then k) ── */}
        <g style={fade(view >= 2)}>
          <line
            x1={X0}
            x2={g.mainX1}
            y1={g.chartBase}
            y2={g.chartBase}
            style={{ stroke: GT.lineStrong }}
          />
          {/* step 5: the current k, a band behind its bar */}
          {view === 4 && (
            <rect
              x={kx(kd > Q - 0.5 ? kd - Q : kd) - slot / 2}
              y={g.chartTop - 4}
              width={slot}
              height={ph + 4}
              style={{ fill: GT.aSoft }}
            />
          )}
          {/* X: chances of each x (steps 3, 4; fading out at the start of 5) */}
          {xBars.map((p, x) => {
            const h = barH(p);
            if (h < 0.05) return null;
            return (
              <rect
                key={`xb${x}`}
                x={kx(x) - bw / 2}
                y={g.chartBase - h}
                width={bw}
                height={h}
                style={{ fill: GT.b, opacity: view === 4 ? comb : 1 }}
              />
            );
          })}
          {/* step 4: another run (Y = 13) leaves the same comb, shifted */}
          {view === 3 &&
            GHOST.map((x) => (
              <rect
                key={`gb${x}`}
                x={kx(x) - bw / 2}
                y={g.chartBase - barH(1 / SURV.length)}
                width={bw}
                height={barH(1 / SURV.length)}
                style={{
                  fill: 'none',
                  stroke: GT.muted,
                  strokeWidth: 1,
                  strokeDasharray: '2 2',
                  opacity: ghostOn,
                }}
              />
            ))}
          {/* k: chances of each reading after the transform (steps 5, 6) */}
          {kBars.map((p, k) => {
            const h = barH(p);
            if (h < 0.05) return null;
            return (
              <rect
                key={`kb${k}`}
                x={kx(k) - bw / 2}
                y={g.chartBase - h}
                width={bw}
                height={h}
                style={{
                  fill: GT.a,
                  opacity: view === 5 && k !== READ ? 1 - 0.5 * readOn : 1,
                }}
              />
            );
          })}

          {/* labels over the bars */}
          {view === 2 && (
            <>
              <g style={{ opacity: xOn * (1 - xf) }}>
                <T
                  x={X0}
                  y={g.chartBase - barH(1 / Q) - 9}
                  anchor="start"
                  size={fs - 1}
                  color={GT.b}
                >
                  {`1/${Q} each`}
                </T>
              </g>
              <g style={{ opacity: xf }}>
                <T x={kx(XFIRST)} y={g.chartTop - 8} size={fs - 1} color={GT.b} weight={700}>
                  1
                </T>
                {tri(kx(XFIRST), g.chartTop - 26, 1)}
              </g>
            </>
          )}
          {view === 3 &&
            SURV.map((x) => (
              <g key={`c${x}`} style={{ opacity: ramp(tt, T_COMB + 0.3) }}>
                <T x={kx(x)} y={g.chartBase - barH(1 / SURV.length) - 8} size={fs - 1} color={GT.b}>
                  1/8
                </T>
              </g>
            ))}
          {view >= 4 &&
            Array.from({ length: R }, (_, i) => i * SPACING).map((k) => (
              <g key={`p${k}`} style={{ opacity: fill(k) }}>
                <T x={kx(k)} y={g.chartTop - 8} size={fs - 1} color={GT.a}>
                  1/4
                </T>
              </g>
            ))}
          {view === 5 && tri(kx(READ), g.chartTop - 26, readOn)}

          {/* the axis: x below in steps 3, 4; k below in steps 5, 6 */}
          {view === 2 &&
            Array.from({ length: Q / R }, (_, i) => i * R).map((x) => (
              <g key={`r${x}`} style={{ opacity: xOn }}>
                <T x={kx(x)} y={g.chartBase + 11} size={fs - 1} color={GT.b}>
                  {x}
                </T>
              </g>
            ))}
          {view === 3 &&
            SURV.map((x) => (
              <g key={`a${x}`} style={{ opacity: comb }}>
                <T x={kx(x)} y={g.chartBase + 11} size={fs - 1} color={GT.b}>
                  {x}
                </T>
              </g>
            ))}
          {view >= 4 &&
            Array.from({ length: R }, (_, i) => i * SPACING).map((k) => (
              <g key={`k${k}`} style={{ opacity: view === 4 && near(k) ? 0 : 1 }}>
                <T
                  x={kx(k)}
                  y={g.chartBase + 11}
                  size={fs - 1}
                  color={view === 5 || fill(k) > 0.5 ? GT.a : GT.muted}
                  weight={view === 5 && k === READ && readOn > 0 ? 700 : 400}
                >
                  {k}
                </T>
              </g>
            ))}
          {view === 4 && (
            <T x={kx(kInt)} y={g.chartBase + 11} size={fs - 1} color={GT.a} weight={700}>
              {kInt}
            </T>
          )}
          {/* step 6: each peak over 32, and what it says about r */}
          {view === 5 &&
            Array.from({ length: R }, (_, i) => i * SPACING).map((k) => {
              const f = k / Q;
              const den = f === 0 ? 1 : Q / gcd(k, Q);
              const ok = den === R;
              return (
                <g key={`f${k}`}>
                  <g style={{ opacity: fracOn }}>
                    <T x={kx(k)} y={g.chartBase + 24} size={fs - 1} color={GT.inkSoft}>
                      {k === 0 ? '0' : `${k / gcd(k, Q)}/${den}`}
                    </T>
                  </g>
                  <g style={{ opacity: rOn }}>
                    <T
                      x={k === 0 ? X0 : kx(k)}
                      anchor={k === 0 ? 'start' : 'middle'}
                      y={g.chartBase + 37}
                      size={fs - 1}
                      color={ok ? GT.eq : GT.muted}
                      weight={ok ? 700 : 400}
                    >
                      {ok ? `r = ${den}` : 'again'}
                    </T>
                  </g>
                </g>
              );
            })}

          {/* hit areas: hover or tap a bar to pick its k (step 5) */}
          {view === 4 &&
            Array.from({ length: Q }, (_, k) => (
              <rect
                key={`hit${k}`}
                x={kx(k) - slot / 2}
                y={g.chartTop - 12}
                width={slot}
                height={ph + 28}
                style={{ fill: 'transparent', cursor: 'pointer' }}
                onPointerEnter={(e) => {
                  if (e.pointerType !== 'touch') {
                    setKHover(k);
                    setExplored(true);
                  }
                }}
                onPointerLeave={() => setKHover(null)}
                onClick={() => {
                  setKPin(k);
                  setExplored(true);
                }}
              />
            ))}
        </g>

        {/* step 3: the equation and the contrast, in the empty top of the X chart */}
        <g style={fade(view === 2)}>
          <g style={{ opacity: eqOn * (1 - xf) }}>
            <T
              x={g.mainX1}
              y={g.chartTop + (g.narrow ? 22 : 30)}
              anchor="end"
              size={g.eqSize}
              font="display"
              color={GT.ink}
            >
              {'Y = 7'}
              <tspan dy={-g.eqSize * 0.4} fontSize={g.eqSize * 0.72} style={{ fill: GT.b }}>
                X
              </tspan>
              <tspan dy={g.eqSize * 0.4}>{` mod ${N}`}</tspan>
            </T>
          </g>
          <g style={{ opacity: eqOn * xf }}>
            <T
              x={g.mainX1}
              y={g.chartTop + (g.narrow ? 22 : 30)}
              anchor="end"
              size={g.eqSize}
              font="display"
              color={GT.ink}
            >
              {'Y = 7'}
              <tspan dy={-g.eqSize * 0.4} fontSize={g.eqSize * 0.72} style={{ fill: GT.b }}>
                {XFIRST}
              </tspan>
              <tspan dy={g.eqSize * 0.4}>{` mod ${N} = `}</tspan>
              <tspan style={{ fill: GT.ink }}>{YFIRST}</tspan>
            </T>
          </g>
          <g
            style={{
              opacity: chipOn,
              cursor: 'pointer',
              pointerEvents: chipOn > 0.5 && view === 2 ? undefined : 'none',
            }}
            onPointerEnter={(e) => {
              if (e.pointerType !== 'touch') setXfHover(true);
            }}
            onPointerLeave={() => setXfHover(false)}
            onClick={() => setXfPin((v) => !v)}
          >
            {(() => {
              const label = 'what if you read X first?';
              const w = label.length * (fs - 0.5) * 0.6 + 16;
              const y = g.chartTop + (g.narrow ? 46 : 58);
              return (
                <>
                  <rect
                    x={g.mainX1 - w}
                    y={y - 10}
                    width={w}
                    height={20}
                    style={{
                      fill: GT.bSoft,
                      fillOpacity: xf,
                      stroke: GT.b,
                      strokeWidth: 1,
                      strokeDasharray: xf > 0.5 ? undefined : '3 2',
                    }}
                  />
                  <T x={g.mainX1 - w / 2} y={y + 0.5} size={fs - 0.5} color={GT.b} weight={600}>
                    {label}
                  </T>
                  <g style={{ opacity: xf }}>
                    <T x={g.mainX1} y={y + 22} anchor="end" size={fs - 1} color={GT.inkSoft}>
                      {g.narrow ? 'the trick is lost' : 'one x, one Y: the trick is lost'}
                    </T>
                  </g>
                </>
              );
            })()}
          </g>
        </g>

        {/* step 4: the other run, in words */}
        <g style={fade(view === 3)}>
          <g style={{ opacity: ghostOn }}>
            <T x={X0} y={g.chartTop + 4} anchor="start" size={fs - 1} color={GT.muted}>
              {`dashed: a run where Y reads ${OTHER}`}
            </T>
            <T x={X0} y={g.chartTop + 18} anchor="start" size={fs - 1} color={GT.muted}>
              same gaps, another start
            </T>
          </g>
        </g>

        {/* ── the side box ── */}
        {/* steps 3, 4: Y, chance of each value */}
        <g style={fade(view === 2 || view === 3)}>
          <T x={g.sideX0} y={g.panelTop} anchor="start" size={fs + 1} color={GT.ink} weight={600}>
            {view === 3 ? `Y reads ${OUT}` : 'Y: chance of each'}
          </T>
          <line
            x1={g.sideX0}
            x2={X1}
            y1={g.chartBase}
            y2={g.chartBase}
            style={{ stroke: GT.lineStrong }}
          />
          {CYCLE.map((v, i) => {
            const h = yH(yBars[i]);
            const read = view === 3 ? (v === OUT ? yRead : 0) : v === YFIRST ? xf : 0;
            const quarter = view === 2 ? yOn * (1 - xf) : 1 - yRead;
            return (
              <g key={`y${v}`}>
                {h > 0.05 && (
                  <rect
                    x={yx(i) - ybw / 2}
                    y={g.chartBase - h}
                    width={ybw}
                    height={h}
                    style={{ fill: GT.ink }}
                  />
                )}
                <g style={{ opacity: quarter }}>
                  <T x={yx(i)} y={g.chartBase - yH(1 / R) - 8} size={fs - 1} color={GT.ink}>
                    1/4
                  </T>
                </g>
                <g style={{ opacity: read }}>
                  <T x={yx(i)} y={g.chartTop - 8} size={fs - 1} color={GT.ink} weight={700}>
                    1
                  </T>
                </g>
                {tri(yx(i), g.chartTop - 26, view === 3 && v === OUT ? yRead : 0)}
                <T
                  x={yx(i)}
                  y={g.chartBase + 11}
                  size={fs}
                  color={GT.ink}
                  weight={read > 0.5 ? 700 : 400}
                >
                  {v}
                </T>
              </g>
            );
          })}
        </g>

        {/* step 5: the arrows of the current k */}
        <g style={fade(view === 4)}>
          <T x={g.sideX0} y={g.panelTop} anchor="start" size={fs + 2} color={GT.a} weight={700}>
            {`k = ${kInt}`}
          </T>
          <T x={X1} y={g.panelTop} anchor="end" size={fs} color={GT.a} weight={600}>
            <tspan style={{ opacity: clamp(1 - Math.abs(kd - Math.round(kd)) * 5, 0, 1) }}>
              {`chance ${chanceText}`}
            </tspan>
          </T>
          <g style={{ opacity: ghost5 }}>
            {other.map((p, j) => (
              <g key={`o${j}`}>
                <line
                  x1={cx}
                  y1={cy}
                  x2={p.x}
                  y2={p.y}
                  style={{ stroke: GT.muted, strokeWidth: 1, strokeDasharray: '3 2' }}
                />
                <circle cx={p.x} cy={p.y} r={1.8} style={{ fill: GT.muted }} />
              </g>
            ))}
            <T x={sideMid} y={g.H - 8} size={fs - 1} color={GT.muted}>
              {`dashed: Y reads ${OTHER}`}
            </T>
          </g>
          <g style={{ opacity: 0.4 }}>
            <Arrow
              x1={cx}
              y1={cy}
              x2={avgEnd.x}
              y2={avgEnd.y}
              color={GT.a}
              width={g.narrow ? 7 : 9}
              head={g.narrow ? 14 : 17}
            />
          </g>
          {main.map((p, j) => (
            <Arrow
              key={`m${j}`}
              x1={cx}
              y1={cy}
              x2={p.x}
              y2={p.y}
              color={GT.b}
              width={1.3}
              head={g.narrow ? 6 : 7}
            />
          ))}
          {stacks
            .filter((q) => q.n > 1)
            .map((q) => (
              <T
                key={`n${Math.round(q.a * 100)}`}
                x={cx + (AL + 13) * Math.cos(q.a)}
                y={cy - (AL + 11) * Math.sin(q.a)}
                size={fs - 1}
                color={GT.b}
                weight={700}
              >
                {`×${q.n}`}
              </T>
            ))}
          <circle cx={cx} cy={cy} r={2.2} style={{ fill: GT.ink }} />
        </g>

        {/* step 6: the reading, the fraction, r */}
        <g style={fade(view === 5)}>
          <g style={{ opacity: fracOn }}>
            <T
              x={sideMid}
              y={g.chartTop + (g.narrow ? 4 : 2)}
              size={g.narrow ? 18 : 24}
              font="display"
              color={GT.ink}
            >
              {`${READ}/${Q} = 1/${R}`}
            </T>
          </g>
          <g style={{ opacity: rOn }}>
            <T
              x={sideMid}
              y={g.chartTop + (g.narrow ? 46 : 52)}
              size={g.narrow ? 28 : 34}
              font="display"
              color={GT.eq}
            >
              {`r = ${R}`}
            </T>
          </g>
          <g style={{ opacity: backOn }}>
            <T x={sideMid} y={g.chartTop + (g.narrow ? 82 : 94)} size={fs - 1} color={GT.muted}>
              step 2 then gives
            </T>
            <T
              x={sideMid}
              y={g.chartTop + (g.narrow ? 101 : 116)}
              size={g.narrow ? 17 : 22}
              font="display"
              color={GT.opt}
            >
              {`${N} = ${P1} × ${P2}`}
            </T>
          </g>
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
