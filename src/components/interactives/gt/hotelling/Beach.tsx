import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  Chip,
  GT,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import { meanWalk, nudge, shares } from './logic';

/* One coordinate system: the beach runs from x=X0 (position 0) to x=X1 (position 1). */
const W = 640;
const H = 200;
const X0 = 30;
const X1 = 610;
const SAND = 150; // y of the sand line (feet)
// rounded so server and client print identical attribute strings (no hydration mismatch)
const px = (p: number) => Math.round((X0 + p * (X1 - X0)) * 100) / 100;

const N = 60;
const PEOPLE = Array.from({ length: N }, (_, i) => (i + 0.5) / N);
const OPT: [number, number] = [0.25, 0.75];
const STEP = 0.025;
const TURN_MS = 380;

const fmtPct = (s: number) => `${Math.round(s * 1000) / 10}`;
const metres = (w: number) => Math.round(w * 1000);

/** Which cart each beachgoer walks to (co-located carts split the crowd one by one). */
function owner(pos: number[], i: number): number {
  const x = PEOPLE[i];
  const da = Math.abs(pos[0] - x);
  const db = Math.abs(pos[1] - x);
  if (Math.abs(da - db) < 1e-9) return i % 2;
  return da < db ? 0 : 1;
}

function Cart({ x, color, letter }: { x: number; color: string; letter: string }) {
  // a little ice-cream cart: box, umbrella, two wheels. Bottom of the wheels sits on SAND.
  return (
    <g transform={`translate(${x} 0)`}>
      <line x1={0} y1={SAND - 40} x2={0} y2={SAND - 22} style={{ stroke: GT.ink, strokeWidth: 1.2 }} />
      <path
        d={`M -15 ${SAND - 38} Q 0 ${SAND - 54} 15 ${SAND - 38} Z`}
        style={{ fill: color, stroke: GT.ink, strokeWidth: 1 }}
      />
      <rect
        x={-12}
        y={SAND - 24}
        width={24}
        height={16}
        style={{ fill: GT.paperRaised, stroke: color, strokeWidth: 2 }}
      />
      <text
        x={0}
        y={SAND - 15.5}
        textAnchor="middle"
        dominantBaseline="middle"
        style={{ fill: color, fontFamily: GT.display, fontSize: 13, fontWeight: 600 }}
      >
        {letter}
      </text>
      <circle cx={-7} cy={SAND - 4} r={3.5} style={{ fill: GT.paperRaised, stroke: GT.ink, strokeWidth: 1.2 }} />
      <circle cx={7} cy={SAND - 4} r={3.5} style={{ fill: GT.paperRaised, stroke: GT.ink, strokeWidth: 1.2 }} />
    </g>
  );
}

function Ghost({ x }: { x: number }) {
  return (
    <g transform={`translate(${x} 0)`} style={{ opacity: 0.9 }}>
      <rect
        x={-14}
        y={SAND - 26}
        width={28}
        height={26}
        style={{ fill: 'none', stroke: GT.opt, strokeWidth: 1.2, strokeDasharray: '3 3' }}
      />
    </g>
  );
}

type Mode = 'idle' | 'compete' | 'spread';

export default function Beach() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const svgRef = useRef<SVGSVGElement | null>(null);

  // target positions (the game state) and displayed positions (glide toward the target)
  const [pos, setPos] = useState<number[]>([...OPT]);
  const [show, setShow] = useState<number[]>([...OPT]);
  const [mode, setMode] = useState<Mode>('idle');
  const [turn, setTurn] = useState(0);
  const [last, setLast] = useState<string>('Carts at the quarter marks: nobody walks more than 250 m.');
  const dragging = useRef<number | null>(null);
  const demoDone = useRef(false);

  // glide displayed carts toward the state
  useFrame(
    (dt) => {
      setShow((s) => s.map((x, i) => x + (pos[i] - x) * Math.min(1, dt * 9)));
    },
    !reduced && show.some((x, i) => Math.abs(x - pos[i]) > 1e-4),
  );
  useEffect(() => {
    if (reduced) setShow(pos);
  }, [reduced, pos]);

  // auto-demo once on first view: hold the optimum, then let them compete
  useEffect(() => {
    if (demoDone.current || !inView) return;
    demoDone.current = true;
    if (reduced) {
      // static poster: the punchline (both at the centre, optimum as ghosts)
      setPos([0.5, 0.5]);
      setShow([0.5, 0.5]);
      setLast('Both carts end up side by side at the centre.');
      return;
    }
    const id = setTimeout(() => setMode('compete'), 1400);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  // the competition: carts take turns stepping wherever wins them more customers
  useEffect(() => {
    if (mode !== 'compete' || !inView) return;
    const id = setTimeout(() => {
      const who = turn % 2;
      const next = nudge(pos, who, STEP);
      const moved = next[who] !== pos[who];
      const other = nudge(next, 1 - who, STEP);
      const stuck = !moved && other[1 - who] === next[1 - who];
      const name = who === 0 ? 'A' : 'B';
      if (stuck) {
        setMode('idle');
        setLast('Neither cart can gain by moving. They stay put, side by side.');
        return;
      }
      if (moved) {
        const s = shares(next)[who];
        setLast(`${name} edges ${next[who] > pos[who] ? 'right' : 'left'} and now serves ${fmtPct(s)}% of the beach.`);
      }
      setPos(next);
      setTurn((t) => t + 1);
    }, TURN_MS);
    return () => clearTimeout(id);
  }, [mode, turn, pos, inView]);

  const stop = () => setMode('idle');

  const compete = () => {
    setTurn(0);
    setMode('compete');
  };
  const spread = () => {
    setMode('idle');
    setPos([...OPT]);
    if (reduced) setShow([...OPT]);
    setLast('Back at the quarter marks: the shortest average walk two carts can give.');
  };

  const onDown = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const p = svgPoint(svg, e.clientX, e.clientY);
    const x = clamp((p.x - X0) / (X1 - X0), 0, 1);
    const d0 = Math.abs(show[0] - x);
    const d1 = Math.abs(show[1] - x);
    const who = d0 <= d1 ? 0 : 1;
    if (Math.min(d0, d1) > 0.12) return;
    stop();
    dragging.current = who;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    moveTo(who, x);
  };
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (dragging.current === null || !svg) return;
    const p = svgPoint(svg, e.clientX, e.clientY);
    moveTo(dragging.current, clamp((p.x - X0) / (X1 - X0), 0, 1));
  };
  const moveTo = (who: number, x: number) => {
    const next = pos.slice();
    next[who] = Math.round(x * 200) / 200;
    setPos(next);
    setShow(next);
    const s = shares(next)[who];
    setLast(`${who === 0 ? 'A' : 'B'} at ${Math.round(next[who] * 1000)} m serves ${fmtPct(s)}% of the beach.`);
  };
  const onUp = () => {
    dragging.current = null;
  };
  const onKey = (who: number) => (e: React.KeyboardEvent) => {
    const d = e.key === 'ArrowLeft' ? -0.025 : e.key === 'ArrowRight' ? 0.025 : 0;
    if (!d) return;
    e.preventDefault();
    stop();
    moveTo(who, clamp(pos[who] + d, 0, 1));
  };

  const sh = shares(show);
  const walk = meanWalk(show, PEOPLE);
  const owners = PEOPLE.map((_, i) => owner(show, i));
  const colors = [GT.a, GT.b];
  const together = Math.abs(show[0] - show[1]) < 1e-3;
  const boundary = (show[0] + show[1]) / 2;

  return (
    <Specimen
      code="GT-03.1"
      title="the beach"
      kicker="1 km of sand · 60 sunbathers"
      rootRef={rootRef}
      hint="drag a cart (or focus it and use ← →), or press a button"
      footer={`turn ${turn}`}
    >
      <div className="grid grid-cols-3 gap-3">
        <BigStat label="cart A" value={fmtPct(sh[0])} unit="%" color={GT.a} size="md" />
        <BigStat label="cart B" value={fmtPct(sh[1])} unit="%" color={GT.b} size="md" />
        <BigStat
          label="avg walk"
          value={metres(walk)}
          unit="m"
          color={walk > 0.126 ? GT.eq : GT.opt}
          sub="best possible 125 m"
          align="right"
        />
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="mt-3 block w-full touch-none select-none"
        role="img"
        aria-label={`Beach with two ice-cream carts. A at ${Math.round(show[0] * 1000)} metres, B at ${Math.round(
          show[1] * 1000,
        )} metres. Average walk ${metres(walk)} metres.`}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
      >
        {/* sand */}
        <rect x={X0} y={SAND} width={X1 - X0} height={14} style={{ fill: GT.warn, opacity: 0.18 }} />
        <line x1={X0} y1={SAND} x2={X1} y2={SAND} style={{ stroke: GT.ink, strokeWidth: 1.2 }} />

        {/* walk arcs (behind everything) */}
        {PEOPLE.map((x, i) => {
          const c = show[owners[i]];
          const x1 = px(x);
          const x2 = px(c);
          const h = Math.round(Math.abs(x2 - x1) * 34) / 100;
          const mx = Math.round((x1 + x2) * 50) / 100;
          const y = SAND - 22;
          const cy = Math.round((y - h) * 100) / 100;
          return (
            <path
              key={`w${i}`}
              d={`M ${x1} ${y} Q ${mx} ${cy} ${x2} ${SAND - 30}`}
              style={{ fill: 'none', stroke: colors[owners[i]], strokeWidth: 1, opacity: 0.32 }}
            />
          );
        })}

        {/* optimum ghosts */}
        {OPT.map((p) => (
          <Ghost key={p} x={px(p)} />
        ))}
        <T x={px(0.25)} y={SAND + 32} color={GT.opt} size={9.5}>
          best for walkers
        </T>
        <T x={px(0.75)} y={SAND + 32} color={GT.opt} size={9.5}>
          best for walkers
        </T>

        {/* market boundary */}
        {!together && (
          <line
            x1={px(boundary)}
            y1={SAND - 6}
            x2={px(boundary)}
            y2={SAND + 14}
            style={{ stroke: GT.muted, strokeWidth: 1, strokeDasharray: '2 2' }}
          />
        )}

        {/* people */}
        {PEOPLE.map((x, i) => {
          const c = colors[owners[i]];
          const cx = px(x);
          return (
            <g key={`p${i}`}>
              <circle cx={cx} cy={SAND - 18} r={2.6} style={{ fill: c }} />
              <line x1={cx} y1={SAND - 15} x2={cx} y2={SAND - 5} style={{ stroke: c, strokeWidth: 1.6 }} />
              <line x1={cx} y1={SAND - 5} x2={cx - 2} y2={SAND} style={{ stroke: c, strokeWidth: 1.2 }} />
              <line x1={cx} y1={SAND - 5} x2={cx + 2} y2={SAND} style={{ stroke: c, strokeWidth: 1.2 }} />
            </g>
          );
        })}

        {/* carts (B first so A sits on top when they share a spot) */}
        {[1, 0].map((k) => (
          <g
            key={`c${k}`}
            tabIndex={0}
            role="slider"
            aria-label={`cart ${k === 0 ? 'A' : 'B'}`}
            aria-valuemin={0}
            aria-valuemax={1000}
            aria-valuenow={Math.round(pos[k] * 1000)}
            onKeyDown={onKey(k)}
            style={{ cursor: 'grab', outline: 'none' }}
            transform={together ? `translate(${k === 0 ? -9 : 9} 0)` : undefined}
          >
            <Cart x={px(show[k])} color={colors[k]} letter={k === 0 ? 'A' : 'B'} />
          </g>
        ))}

        {/* axis */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={px(t)} y1={SAND + 14} x2={px(t)} y2={SAND + 19} style={{ stroke: GT.muted }} />
            {(t === 0 || t === 0.5 || t === 1) && (
              <T x={px(t)} y={SAND + 32} size={9.5}>
                {t === 0 ? '0' : t === 1 ? '1 km' : '500 m'}
              </T>
            )}
          </g>
        ))}
      </svg>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Btn primary onClick={compete}>
          let them compete
        </Btn>
        <Btn onClick={spread}>spread them out</Btn>
        <span className="ml-auto flex flex-wrap gap-3">
          <Chip color={GT.a}>walks to A</Chip>
          <Chip color={GT.b}>walks to B</Chip>
          <Chip color={GT.opt}>best for walkers</Chip>
        </span>
      </div>
      <Caption>{last}</Caption>
    </Specimen>
  );
}
