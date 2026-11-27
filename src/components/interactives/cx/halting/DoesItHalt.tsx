import { useEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { useBoost } from '../../gt/braess/useBoost';
import { useCompact } from '../../gt/useCompact';
import { collatz, niceTicks, stoppingTimes } from './collatz';

/**
 * The Collatz program: halve n if even, else 3n + 1, stop at 1. Top: the chosen start's
 * whole trip (value against step). Bottom: steps to reach 1 for every start 1..1000, which
 * doubles as the picker (drag across it; it snaps to the famous starts).
 */
const MAXN = 1000;
const ST = stoppingTimes(MAXN);
const SNAPS = [27, 97, 703, 871];

const PT = 26;
const PB = 190;
const SPT = 244;
const SPB = 330;
const H = SPB + 40;
const SMAX = 180;
const sy = (s: number) => SPB - (s / SMAX) * (SPB - SPT);
const fmt = (v: number) => Math.round(v).toLocaleString('en-US');

export default function DoesItHalt() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(500);
  const W = compact ? 380 : 520;
  const PL = compact ? 46 : 56;
  const PR = W - (compact ? 14 : 20);
  const sx = (v: number) => PL + ((v - 1) / (MAXN - 1)) * (PR - PL);
  const [svgRef, k] = useBoost(W, 9, 8.5);
  const [sel, setSel] = useState(27);
  const [hover, setHover] = useState<number | null>(null);
  const n = hover ?? sel;
  const seq = useMemo(() => collatz(n), [n]);
  const S = seq.length - 1;
  const peak = Math.max(...seq);
  const peakAt = seq.indexOf(peak);

  // p = how many steps of the committed trip are drawn (continuous); SSR shows it whole
  const [p, setP] = useState(Infinity);
  const pRef = useRef(Infinity);
  const [demo, setDemo] = useState(false);
  const started = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dragging = useRef(false);

  const startDraw = (delay: number) => {
    pRef.current = 0;
    setP(0);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setDemo(true), delay);
  };
  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    if (!reduced) startDraw(500);
  }, [inView, reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  // the reduced-motion setting can arrive after the first effect: show the whole trip
  useEffect(() => {
    if (!reduced) return;
    if (timer.current) clearTimeout(timer.current);
    setDemo(false);
    pRef.current = Infinity;
    setP(Infinity);
  }, [reduced]);
  const selS = ST[sel] || 0;
  const speed = Math.max(40, selS / 2.4); // steps per second
  useFrame((dt) => {
    const v = Math.min(selS, pRef.current + Math.max(0, dt) * speed);
    pRef.current = v;
    setP(v);
    if (v >= selS) {
      pRef.current = Infinity;
      setP(Infinity);
      setDemo(false);
    }
  }, demo && inView);

  const take = () => {
    if (timer.current) clearTimeout(timer.current);
    setDemo(false);
    pRef.current = Infinity;
    setP(Infinity);
  };
  const replay = () => {
    setHover(null);
    setSel(27);
    if (reduced) return take();
    startDraw(1000);
  };

  const pick = (e: RPointerEvent) => {
    if (!svgRef.current) return null;
    const pt = svgPoint(svgRef.current, e.clientX, e.clientY);
    if (pt.y < SPT - 24 && !dragging.current) return null;
    for (const s of SNAPS) if (Math.abs(sx(s) - pt.x) < 5) return s;
    return Math.round(clamp(1 + ((pt.x - PL) / (PR - PL)) * (MAXN - 1), 1, MAXN));
  };

  // drawn portion (only the committed trip animates; a hovered preview shows whole)
  const drawn = hover !== null ? S : Math.min(S, p);
  const xMax = Math.max(S, 1);
  const yMax = Math.max(peak, 4);
  const px = (i: number) => PL + (i / xMax) * (PR - PL);
  const py = (v: number) => PB - (v / yMax) * (PB - PT);
  let path = '';
  const whole = Math.floor(drawn);
  for (let i = 0; i <= whole; i++)
    path += `${i ? 'L' : 'M'}${px(i).toFixed(2)} ${py(seq[i]).toFixed(2)}`;
  if (drawn > whole && whole < S) {
    const f = drawn - whole;
    const x = px(whole + f);
    const y = py(seq[whole] + (seq[whole + 1] - seq[whole]) * f);
    path += `L${x.toFixed(2)} ${y.toFixed(2)}`;
  }
  const headI = Math.min(S, drawn);
  const hx = px(headI);
  const hi = Math.floor(headI);
  const hy = py(hi >= S ? seq[S] : seq[hi] + (seq[hi + 1] - seq[hi]) * (headI - hi));
  const peakSoFar = Math.max(...seq.slice(0, whole + 1));
  const finished = drawn >= S;

  const xt = niceTicks(xMax, 4);
  const yt = niceTicks(yMax, 4);
  const halo = {
    stroke: GT.paperRaised,
    strokeWidth: 4,
    strokeLinejoin: 'round' as const,
    paintOrder: 'stroke' as const,
  };

  // scatter dots as one path (1000 tiny squares)
  const dots = useMemo(() => {
    let d = '';
    for (let i = 1; i <= MAXN; i++)
      d += `M${(sx(i) - 1.1).toFixed(1)} ${(sy(ST[i]) - 1.1).toFixed(1)}h2.2v2.2h-2.2z`;
    return d;
  }, [compact]);
  const selX = sx(n);
  const selY = sy(ST[n]);
  const show871 = Math.abs(n - 871) > 140;
  const labRight = selX < PR - 120;

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-09.2"
      title="does it halt?"
      kicker="halve if even, else 3n + 1"
      hint="drag along the dots to pick a start; it snaps to 27, 97, 703 and 871"
      footer="checked below 2^71 (Barina, 2025)"
    >
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full touch-pan-y select-none"
          role="img"
          aria-label={`The Collatz sequence starting at ${n}: ${S} steps to reach 1, peak ${fmt(peak)}. Below, the number of steps for every start from 1 to 1000.`}
          onPointerMove={(e) => {
            const v = pick(e);
            if (dragging.current) {
              if (v !== null) setSel(v);
            } else if (e.pointerType !== 'touch') {
              setHover(v);
              if (v !== null && demo) take();
            }
          }}
          onPointerLeave={() => setHover(null)}
          onPointerDown={(e) => {
            const v = pick(e);
            if (v === null) return;
            take();
            dragging.current = true;
            (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
            setSel(v);
            setHover(null);
          }}
          onPointerUp={() => (dragging.current = false)}
          onPointerCancel={() => (dragging.current = false)}
        >
          {/* trip chart */}
          {yt.map((v) => (
            <g key={`y${v}`}>
              <line x1={PL} x2={PR} y1={py(v)} y2={py(v)} style={{ stroke: GT.line }} />
              <T x={PL - 6} y={py(v)} anchor="end" size={9 * k}>
                {fmt(v)}
              </T>
            </g>
          ))}
          {xt.map((v) => (
            <T key={`x${v}`} x={px(v)} y={PB + 12 * k} size={9 * k}>
              {fmt(v)}
            </T>
          ))}
          <line x1={PL} y1={PB} x2={PR} y2={PB} style={{ stroke: GT.lineStrong }} />
          <T x={PL} y={PT - 14} anchor="start" size={9 * k} color={GT.ink}>
            {`starting at ${fmt(n)}: value after each step`}
          </T>
          <path
            d={path}
            style={{ fill: 'none', stroke: GT.a, strokeWidth: 1.8, strokeLinejoin: 'round' }}
          />
          {/* peak and finish, labelled where they are */}
          {whole >= peakAt && peak > n && (
            <g>
              <circle cx={px(peakAt)} cy={py(peak)} r={3.5} style={{ fill: GT.b }} />
              <g style={halo}>
                <T
                  x={px(peakAt) + (peakAt / xMax < 0.6 ? 8 : -8)}
                  y={py(peak) + 2}
                  anchor={peakAt / xMax < 0.6 ? 'start' : 'end'}
                  size={9.5 * k}
                  color={GT.b}
                  weight={700}
                >
                  {`peak ${fmt(peak)}`}
                </T>
              </g>
            </g>
          )}
          {finished ? (
            <g>
              <line
                x1={px(S)}
                x2={px(S)}
                y1={py(1)}
                y2={py(1) - 26 * k}
                style={{ stroke: GT.opt, strokeWidth: 1 }}
              />
              <circle cx={px(S)} cy={py(1)} r={4} style={{ fill: GT.opt }} />
              <g style={halo}>
                <T
                  x={px(S) - 4}
                  y={py(1) - 32 * k}
                  anchor="end"
                  size={9.5 * k}
                  color={GT.opt}
                  weight={700}
                >
                  {`halts: 1 at step ${S}`}
                </T>
              </g>
            </g>
          ) : (
            <circle cx={hx} cy={hy} r={3.5} style={{ fill: GT.a }} />
          )}

          {/* every start 1..1000 */}
          <T x={PL} y={SPT - 18} anchor="start" size={9 * k} color={GT.ink}>
            steps to reach 1, for every start from 1 to 1,000
          </T>
          {[0, 50, 100, 150].map((v) => (
            <g key={`s${v}`}>
              <line x1={PL} x2={PR} y1={sy(v)} y2={sy(v)} style={{ stroke: GT.line }} />
              <T x={PL - 6} y={sy(v)} anchor="end" size={9 * k}>
                {v}
              </T>
            </g>
          ))}
          {[1, 250, 500, 750, 1000].map((v) => (
            <T key={`sx${v}`} x={sx(v)} y={SPB + 12 * k} size={9 * k}>
              {fmt(v)}
            </T>
          ))}
          <path d={dots} style={{ fill: GT.muted, opacity: 0.75 }} />
          {show871 && (
            <g style={halo}>
              <T x={sx(871) - 7} y={sy(178)} anchor="end" size={9 * k} color={GT.ink}>
                871: 178 steps
              </T>
            </g>
          )}
          <line
            x1={selX}
            x2={selX}
            y1={SPT - 6}
            y2={SPB}
            style={{ stroke: GT.a, strokeWidth: 1, strokeDasharray: '2 3' }}
          />
          <rect
            x={selX - 4}
            y={selY - 4}
            width={8}
            height={8}
            style={{ fill: GT.a, stroke: GT.paperRaised, strokeWidth: 1.5 }}
          />
          <g style={halo}>
            <T
              x={selX + (labRight ? 9 : -9)}
              y={selY < SPT + 20 ? selY + 12 : selY - 10}
              anchor={labRight ? 'start' : 'end'}
              size={9.5 * k}
              color={GT.a}
              weight={700}
            >
              {`${fmt(n)}: ${ST[n]} steps`}
            </T>
          </g>
          {/* generous hit area for the picker */}
          <rect
            x={PL - 10}
            y={SPT - 24}
            width={PR - PL + 20}
            height={SPB - SPT + 30}
            style={{ fill: 'transparent', cursor: 'ew-resize' }}
          />
        </svg>
      </div>

      <div className="mt-4 flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
        <div className="flex flex-wrap gap-x-8 gap-y-3">
          <BigStat label="start" value={fmt(n)} color={GT.a} />
          <BigStat label="steps so far" value={Math.floor(Math.min(drawn, S))} />
          <BigStat label="highest value" value={fmt(peakSoFar)} color={GT.b} />
        </div>
        <Btn primary onClick={replay}>
          replay
        </Btn>
      </div>
      <Caption>
        Every start here halts, but you only learn that by running it to the end. If a run has not
        stopped yet, you cannot tell whether it is slow or will run forever.
      </Caption>
    </Specimen>
  );
}
