import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  GT,
  Seg,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
  useTween,
} from '../shared';
import { cdfFromDensity, nudge, presetDensity, shares, type Cdf, type Preset } from './logic';
import { useCompact } from '../useCompact';
import { Reserve } from '../Reserve';

const H = 250;
const BASE = 190; // baseline of the density curve
const TOP = 52; // highest point of the curve
// rounded so server and client print identical attribute strings (no hydration mismatch)
const GRID = 200;
const STEP = 0.01;
const TURN_MS = 190;
const STEPS_PER_TURN = 2; // each round moves up to 2 grid steps, so the race is short but calm
/** Distance per second a candidate covers in the race (grid steps per turn / turn time). */
const RACE_SPEED = (STEP * STEPS_PER_TURN) / (TURN_MS / 1000);

/**
 * One candidate's turn: up to STEPS_PER_TURN small best-response steps, never jumping over
 * the rival (a candidate who reaches the rival's spot stops there and they share the voters).
 * No leapfrogging means the shading never flips sides while they converge.
 */
function turnMove(pos: number[], who: number, F: Cdf): number[] {
  let p = pos;
  for (let k = 0; k < STEPS_PER_TURN; k++) {
    const q = nudge(p, who, STEP, F);
    if (q[who] === p[who]) break;
    const o = p[1 - who];
    const crossed = (p[who] - o) * (q[who] - o) < 0;
    if (crossed || Math.abs(q[who] - o) < 1e-9) {
      const next = p.slice();
      next[who] = o;
      // only join the rival if sharing the spot actually beats staying
      if (shares(next, F)[who] > shares(p, F)[who] + 1e-9) p = next;
      break;
    }
    p = q;
  }
  return p;
}

/**
 * One round: both candidates make their best small move at the same time, from the same
 * picture. They close in on each other smoothly, so the vote split drifts steadily toward
 * 50/50 instead of see-sawing. If the moves would cross, they meet where the split is even.
 */
function round(pos: number[], F: Cdf): number[] {
  // Together: they split the voters 50/50 and walk on side by side toward the median,
  // instead of one stepping out and the other rejoining (which looks like a bounce).
  if (Math.abs(pos[0] - pos[1]) < 1e-9) {
    const x = pos[0];
    let best = x;
    for (let k = 1; k <= STEPS_PER_TURN; k++) {
      for (const d of [-STEP * k, STEP * k]) {
        const y = Math.round((x + d) / STEP) * STEP;
        if (y < 0 || y > 1) continue;
        if (
          Math.abs(F(y) - 0.5) < Math.abs(F(best) - 0.5) - 1e-9 &&
          Math.abs(y - x) <= Math.abs(best - x) + STEP + 1e-9
        )
          best = y;
      }
    }
    return [best, best];
  }
  const a = turnMove(pos, 0, F)[0];
  const b = turnMove(pos, 1, F)[1];
  const before = Math.sign(pos[0] - pos[1]);
  const after = Math.sign(a - b);
  if (before !== 0 && after !== before) {
    const meet = Math.abs(F(a) - 0.5) <= Math.abs(F(b) - 0.5) ? a : b;
    return [meet, meet];
  }
  return [a, b];
}
const START: [number, number] = [0.1, 0.9];

const fmtPct = (s: number) => `${Math.round(s * 100)}`;

const PRESETS: { value: Preset; label: string }[] = [
  { value: 'even', label: 'even' },
  { value: 'lopsided', label: 'lopsided' },
  { value: 'polarized', label: 'polarized' },
];

const NOTES: Record<Preset, string> = {
  even: 'Voters spread evenly: both candidates end at the centre.',
  lopsided:
    'Most voters lean left, with a long tail to the right. The candidates stop at the median, not the mean.',
  polarized:
    'Two camps and an empty middle. The candidates still meet in the middle, where few voters actually are.',
};

function Candidate({ x, color, letter }: { x: number; color: string; letter: string }) {
  return (
    <g transform={`translate(${x} 0)`}>
      <line x1={0} y1={BASE} x2={0} y2={BASE + 14} style={{ stroke: color, strokeWidth: 2 }} />
      <rect
        x={-11}
        y={BASE + 14}
        width={22}
        height={20}
        style={{ fill: GT.paperRaised, stroke: color, strokeWidth: 2 }}
      />
      <text
        x={0}
        y={BASE + 24.5}
        textAnchor="middle"
        dominantBaseline="middle"
        style={{ fill: color, fontFamily: GT.display, fontSize: 13, fontWeight: 600 }}
      >
        {letter}
      </text>
    </g>
  );
}

export default function Ballot() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [boxRef, compact] = useCompact();
  const W = compact ? 400 : 640;
  const X0 = compact ? 14 : 30;
  const X1 = W - X0;
  const fs = compact ? 12 : 10.5;
  const px = (p: number) => Math.round((X0 + p * (X1 - X0)) * 100) / 100;
  const svgRef = useRef<SVGSVGElement | null>(null);

  const [sel, setSel] = useState<Preset>('lopsided');
  const [preset, bindPreset] = useHoverPreview(sel);
  const distOf = (p: Preset) => {
    const dens = presetDensity(p, GRID);
    return { dens, ...cdfFromDensity(dens) };
  };
  // the race always runs on the committed crowd; the drawing follows the previewed one
  const distSel = useMemo(() => distOf(sel), [sel]);
  const dist = useMemo(() => (preset === sel ? distSel : distOf(preset)), [preset, sel, distSel]);

  const [pos, setPos] = useState<number[]>([...START]);
  const [show, setShow] = useState<number[]>([...START]);
  const [running, setRunning] = useState(false);
  const [turn, setTurn] = useState(0);
  const dragging = useRef<number | null>(null);
  const demoDone = useRef(false);
  const touched = useRef(false); // the reader took over: no auto-demo

  // Drawn positions follow the game at a steady speed (the race's own pace), so the
  // candidates glide continuously instead of hopping one step and waiting. Big jumps
  // (a rerun, a preview) still close quickly.
  useFrame(
    (dt) =>
      setShow((s) =>
        s.map((x, i) => {
          const d = pos[i] - x;
          const v = Math.max(RACE_SPEED, Math.abs(d) * 6);
          return Math.abs(d) <= v * dt ? pos[i] : x + Math.sign(d) * v * dt;
        }),
      ),
    !reduced && show.some((x, i) => Math.abs(x - pos[i]) > 1e-4),
  );
  useEffect(() => {
    if (reduced) setShow(pos);
  }, [reduced, pos]);

  const settle = (F: Cdf = distSel.F) => {
    // reduced motion: jump straight to where the race ends
    let p = [...START];
    for (let k = 0; k < 2000; k++) {
      const q = round(p, F);
      if (q.every((v, i) => v === p[i])) break;
      p = q;
    }
    return p;
  };

  const race = () => {
    touched.current = true;
    setTurn(0);
    if (reduced) {
      const p = settle();
      setPos(p);
      setShow(p);
      return;
    }
    setPos([...START]);
    setRunning(true);
  };

  useEffect(() => {
    if (demoDone.current || !inView) return;
    if (reduced) {
      demoDone.current = true;
      const p = settle();
      setPos(p);
      setShow(p);
      return;
    }
    const id = setTimeout(() => {
      demoDone.current = true;
      if (!touched.current) setRunning(true);
    }, 900);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  // replay the race whenever the committed crowd changes
  const firstSel = useRef(true);
  useEffect(() => {
    if (firstSel.current) {
      firstSel.current = false;
      return;
    }
    race();
  }, [sel]);

  useEffect(() => {
    if (!running || !inView) return;
    const id = setTimeout(() => {
      const next = round(pos, distSel.F);
      if (next.every((v, i) => v === pos[i])) {
        setRunning(false);
        return;
      }
      setPos(next);
      setTurn((t) => t + 1);
    }, TURN_MS);
    return () => clearTimeout(id);
  }, [running, turn, pos, inView, distSel]);

  const toX = (e: React.PointerEvent<SVGSVGElement>) => {
    const p = svgPoint(svgRef.current!, e.clientX, e.clientY);
    return clamp((p.x - X0) / (X1 - X0), 0, 1);
  };
  const place = (who: number, x: number) => {
    const next = pos.slice();
    next[who] = Math.round(x / STEP) * STEP;
    setPos(next);
    setShow(next);
  };
  const onDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const x = toX(e);
    const d = show.map((s) => Math.abs(s - x));
    const who = d[0] <= d[1] ? 0 : 1;
    if (d[who] > 0.12) return;
    touched.current = true;
    setRunning(false);
    dragging.current = who;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    place(who, x);
  };
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragging.current === null || !svgRef.current) return;
    place(dragging.current, toX(e));
  };
  const onUp = () => (dragging.current = null);
  const onKey = (who: number) => (e: React.KeyboardEvent) => {
    const d = e.key === 'ArrowLeft' ? -0.02 : e.key === 'ArrowRight' ? 0.02 : 0;
    if (!d) return;
    e.preventDefault();
    touched.current = true;
    setRunning(false);
    place(who, clamp(pos[who] + d, 0, 1));
  };

  // geometry of the density curve
  const peak = Math.max(...dist.dens);
  const yOf = (d: number) => Math.round((BASE - (d / peak) * (BASE - TOP)) * 100) / 100;
  const pts = dist.dens.map((d, i) => [px((i + 0.5) / GRID), yOf(d)] as const);
  const curve =
    `M ${px(0)} ${yOf(dist.dens[0])} ` +
    pts.map(([x, y]) => `L ${x} ${y}`).join(' ') +
    ` L ${px(1)} ${yOf(dist.dens[GRID - 1])}`;
  const area = (lo: number, hi: number) => {
    const inside = pts.filter(([x]) => x >= px(lo) && x <= px(hi));
    const yAt = (p: number) => yOf(dist.dens[Math.min(GRID - 1, Math.floor(p * GRID))]);
    return (
      `M ${px(lo)} ${BASE} L ${px(lo)} ${yAt(lo)} ` +
      inside.map(([x, y]) => `L ${x} ${y}`).join(' ') +
      ` L ${px(hi)} ${yAt(hi)} L ${px(hi)} ${BASE} Z`
    );
  };

  // previewing another crowd shows where its race ends (what a click would play out to);
  // readouts, candidates and the split all read these, so nothing lags behind the preview
  const pvPos = useMemo(
    () => (preset === sel ? null : settle(dist.F)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [preset, sel, dist],
  );
  const P = pvPos ?? pos;
  const S = pvPos ?? show;

  // readouts follow the game state; the drawing follows the gliding positions
  const shRaw = shares(P, dist.F);
  const shA = useTween(shRaw[0], reduced, 6);
  const sh = [shA, 1 - shA];
  const together = Math.abs(P[0] - P[1]) < 1e-6;
  // who stands on the left; when they meet, keep the side they had before meeting so the
  // two halves keep their colours (each still takes the voters on its own side: 50/50)
  const lastLeft = useRef(0);
  const sideNow = S[0] <= S[1] ? 0 : 1;
  if (!together) lastLeft.current = sideNow;
  const left = together ? lastLeft.current : sideNow;
  // Draw the two boxes so they never overlap: as the candidates close in, the boxes stop
  // when they touch and sit side by side (no jump, no bounce when they finally meet).
  const BOXGAP = 24;
  const drawX = (() => {
    const xs = [px(S[0]), px(S[1])];
    const lo = left;
    const hi = 1 - lo;
    if (xs[hi] - xs[lo] < BOXGAP) {
      const m = (xs[0] + xs[1]) / 2;
      xs[lo] = m - BOXGAP / 2;
      xs[hi] = m + BOXGAP / 2;
    }
    return xs.map((x) => Math.round(x * 100) / 100);
  })();
  const mid = (P[0] + P[1]) / 2;
  const colors = [GT.a, GT.b];
  const softs = [GT.aSoft, GT.bSoft];
  const close = Math.abs(dist.mean - dist.median) < 0.02;

  return (
    <Specimen
      code="GT-03.2"
      title="the ballot"
      kicker="two candidates · one left-right axis"
      rootRef={rootRef}
      hint="pick a crowd to run the race (again to rerun), or drag a candidate"
      footer={`median ${dist.median.toFixed(2)} · mean ${dist.mean.toFixed(2)}`}
    >
      <div ref={boxRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full touch-none select-none"
          role="img"
          aria-label={`Voter density, ${preset}. Candidate A at ${S[0].toFixed(2)}, B at ${S[1].toFixed(
            2,
          )}. Median voter at ${dist.median.toFixed(2)}.`}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
        >
          {/* each candidate's voters */}
          <path d={area(0, mid)} style={{ fill: softs[left] }} />
          <path d={area(mid, 1)} style={{ fill: softs[1 - left] }} />
          <path d={curve} style={{ fill: 'none', stroke: GT.ink, strokeWidth: 1.5 }} />
          <line x1={X0} y1={BASE} x2={X1} y2={BASE} style={{ stroke: GT.ink, strokeWidth: 1.2 }} />

          {/* mean (dashed) and median (solid accent) */}
          <line
            x1={px(dist.mean)}
            y1={TOP - 14}
            x2={px(dist.mean)}
            y2={BASE}
            style={{ stroke: GT.muted, strokeWidth: 1.2, strokeDasharray: '4 3' }}
          />
          <line
            x1={px(dist.median)}
            y1={TOP - 30}
            x2={px(dist.median)}
            y2={BASE}
            style={{ stroke: GT.eq, strokeWidth: 1.8 }}
          />
          <T x={px(dist.median)} y={TOP - 38} color={GT.eq} size={fs + 0.5} weight={600}>
            median voter
          </T>
          {!close && (
            <T x={px(dist.mean) + 5} y={TOP - 14} anchor="start" size={fs}>
              mean
            </T>
          )}
          <line
            x1={px(mid)}
            y1={BASE - 4}
            x2={px(mid)}
            y2={BASE + 6}
            style={{ stroke: GT.ink, strokeWidth: 1, strokeDasharray: '2 2' }}
          />

          {/* candidates */}
          {[1, 0].map((k) => (
            <g
              key={k}
              tabIndex={0}
              role="slider"
              aria-label={`candidate ${k === 0 ? 'A' : 'B'}`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(P[k] * 100)}
              onKeyDown={onKey(k)}
              style={{ cursor: 'grab', outline: 'none' }}
            >
              <Candidate x={drawX[k]} color={colors[k]} letter={k === 0 ? 'A' : 'B'} />
            </g>
          ))}

          <T x={X0} y={BASE + 50} anchor="start" size={fs}>
            ← left
          </T>
          <T x={X1} y={BASE + 50} anchor="end" size={fs}>
            right →
          </T>
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-[6.5rem_6.5rem] items-end gap-x-4 gap-y-4 sm:grid-cols-[6.5rem_6.5rem_minmax(0,1fr)]">
        <BigStat label="votes for A" value={fmtPct(sh[0])} unit="%" color={GT.a} />
        <BigStat
          label="votes for B"
          value={fmtPct(1 - Math.round(sh[0] * 100) / 100)}
          unit="%"
          color={GT.b}
        />
        <div className="col-span-2 flex flex-wrap items-center gap-2 sm:col-span-1 sm:justify-end">
          <Seg
            options={PRESETS}
            value={preset}
            onChange={(v) => (v === sel ? race() : setSel(v))}
            bind={bindPreset}
            label="voters"
          />
        </div>
      </div>
      <Reserve
        items={PRESETS.map((x) => (
          <Caption key={x.value}>{NOTES[x.value]}</Caption>
        ))}
        active={PRESETS.findIndex((x) => x.value === preset)}
      />
    </Specimen>
  );
}
