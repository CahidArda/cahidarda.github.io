import { useEffect, useMemo, useRef, useState } from 'react';
import { BigStat, Btn, GT, Specimen, T, svgPoint, useInView, useReducedMotion } from '../shared';
import { ADJ, EDGES, H, NODES, SEEDS, W, clusterDensity, runAll } from './network';
import { prefersReduced } from './motion';

// NCM 19.2-19.3: a new behaviour spreads round by round. A node switches once at least a
// fraction q of its neighbours have. It sweeps the loose town and the thin chains, then
// stops at a group whose members keep more than 1 - q of their friends inside it.

const STEP_MS = 750;
const R = 11;
const Q0 = 0.3;

// tipping strip geometry (its own small SVG under the network)
const CW = 600;
const CH = 120;
const CP = { l: 34, r: 12, t: 14, b: 26 };
const QMIN = 0.05;
const QMAX = 0.9;
const cx = (q: number) => CP.l + ((q - QMIN) / (QMAX - QMIN)) * (CW - CP.l - CP.r);
const cy = (n: number) => CP.t + (1 - n / NODES.length) * (CH - CP.t - CP.b);

const fmt = (q: number) => q.toFixed(2);

export default function IdeaSpreads() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [q, setQ] = useState(Q0);
  const [seeds, setSeeds] = useState<number[]>(SEEDS);
  const [shownRound, setShownRound] = useState(99); // SSR / reduced motion: the stalled end
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const started = useRef(false);
  const chartRef = useRef<SVGSVGElement | null>(null);

  const when = useMemo(() => runAll(seeds, q), [seeds, q]);
  const lastRound = Math.max(0, ...when);
  const r = Math.min(shownRound, lastRound);
  const stalled = r >= lastRound;
  const adopted = when.map((w) => w >= 0 && w <= r);
  const nAdopted = adopted.filter(Boolean).length;
  const holdouts = stalled ? NODES.filter((n) => when[n.id] < 0).map((n) => n.id) : [];
  const density = holdouts.length ? clusterDensity(holdouts) : null;

  // the tipping curve for the current seeds: final adopters for q = 0.05 .. 0.90
  const curve = useMemo(() => {
    const pts: [number, number][] = [];
    for (let k = 5; k <= 90; k++) {
      const qq = k / 100;
      pts.push([qq, runAll(seeds, qq).filter((w) => w >= 0).length]);
    }
    return pts;
  }, [seeds]);
  const curvePath =
    curve.map(([qq, n], i) => (i === 0 ? `M${cx(qq)},${cy(n)}` : `H${cx(qq)}V${cy(n)}`)).join('') +
    `H${cx(QMAX)}`;

  // auto-play once on view
  useEffect(() => {
    if (reduced || prefersReduced() || started.current || !inView) return;
    started.current = true;
    setShownRound(0);
    setPlaying(true);
  }, [reduced, inView]);

  useEffect(() => {
    if (!playing || !inView) return;
    if (shownRound >= lastRound) {
      setPlaying(false);
      return;
    }
    const id = setTimeout(() => setShownRound((x) => x + 1), shownRound === 0 ? 600 : STEP_MS);
    return () => clearTimeout(id);
  }, [playing, shownRound, lastRound, inView]);

  const restart = () => {
    if (reduced) {
      setShownRound(99);
      setPlaying(false);
    } else {
      setShownRound(0);
      setPlaying(true);
    }
  };
  const changeQ = (v: number) => {
    setQ(v);
    started.current = true;
    restart();
  };
  const toggleSeed = (id: number) => {
    started.current = true;
    setSeeds((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
    restart();
  };
  const toggle = () => {
    if (stalled) restart();
    else setPlaying((p) => !p);
  };

  const onChart = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!chartRef.current) return;
    if (e.type === 'pointermove' && e.buttons !== 1) return;
    const p = svgPoint(chartRef.current, e.clientX, e.clientY);
    const t = QMIN + ((p.x - CP.l) / (CW - CP.l - CP.r)) * (QMAX - QMIN);
    const v = Math.round(Math.max(QMIN, Math.min(QMAX, t)) * 20) / 20;
    if (Math.abs(v - q) > 1e-9) changeQ(v);
  };

  const hv = hover !== null ? hover : null;
  const hvInfo =
    hv !== null
      ? (() => {
          const nb = ADJ[hv];
          const k = nb.filter((j) => adopted[j]).length;
          return { k, d: nb.length, p: nb.length ? k / nb.length : 0 };
        })()
      : null;

  return (
    <Specimen
      code="GT-08.2"
      title="a new idea spreads"
      kicker="NCM 19.2 · adopt when p ≥ q"
      rootRef={rootRef}
      hint="drag q, or click any person to make them an early adopter"
      footer={`${seeds.length} seed${seeds.length === 1 ? '' : 's'} · round ${r}`}
    >
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto block w-full"
        style={{ maxHeight: 400 }}
        role="img"
        aria-label={`A network of ${NODES.length} people. ${nAdopted} have adopted the new behaviour at threshold ${fmt(q)}.`}
      >
        {/* the blocking cluster, drawn as a soft territory behind everything */}
        {holdouts.length > 0 && (
          <g style={{ opacity: 0.9 }}>
            {holdouts.map((i) => (
              <circle key={i} cx={NODES[i].x} cy={NODES[i].y} r={27} style={{ fill: GT.aSoft }} />
            ))}
          </g>
        )}
        {EDGES.map(([a, b]) => {
          const both = adopted[a] && adopted[b];
          return (
            <line
              key={`${a}-${b}`}
              x1={NODES[a].x}
              y1={NODES[a].y}
              x2={NODES[b].x}
              y2={NODES[b].y}
              style={{
                stroke: both ? GT.eq : GT.lineStrong,
                strokeWidth: both ? 2.2 : 1.2,
                opacity: both ? 0.85 : 0.7,
                transition: 'stroke 300ms',
              }}
            />
          );
        })}
        {NODES.map((n) => {
          const on = adopted[n.id];
          const seed = seeds.includes(n.id);
          const fresh = on && when[n.id] === r && r > 0 && !stalled;
          const hold = holdouts.includes(n.id);
          return (
            <g
              key={n.id}
              role="button"
              tabIndex={0}
              aria-label={`person ${n.id + 1}${seed ? ', early adopter' : ''}${on ? ', adopted' : ''}`}
              onClick={() => toggleSeed(n.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  toggleSeed(n.id);
                }
              }}
              onMouseEnter={() => setHover(n.id)}
              onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(n.id)}
              onBlur={() => setHover(null)}
              style={{ cursor: 'pointer', outline: 'none' }}
            >
              <circle cx={n.x} cy={n.y} r={R + 9} style={{ fill: 'transparent' }} />
              {fresh && (
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={R + 6}
                  className="animate-ping"
                  style={{
                    fill: 'none',
                    stroke: GT.eq,
                    strokeWidth: 2,
                    transformBox: 'fill-box',
                    transformOrigin: 'center',
                  }}
                />
              )}
              {seed && (
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={R + 4.5}
                  style={{ fill: 'none', stroke: GT.eq, strokeWidth: 1.5 }}
                />
              )}
              <circle
                cx={n.x}
                cy={n.y}
                r={R}
                style={{
                  fill: on ? GT.eq : GT.paperRaised,
                  stroke: on ? GT.eq : hold ? GT.a : GT.inkSoft,
                  strokeWidth: hover === n.id ? 3 : 1.6,
                  transition: 'fill 300ms, stroke 300ms',
                }}
              />
            </g>
          );
        })}
        <T x={40} y={22} anchor="start" size={13} color={GT.muted}>
          the town
        </T>
        <T x={490} y={20} anchor="middle" size={13} color={GT.muted}>
          a triangle
        </T>
        <T
          x={505}
          y={354}
          anchor="middle"
          size={12}
          color={holdouts.includes(19) ? GT.a : GT.muted}
        >
          a tight-knit group
        </T>
      </svg>

      {/* status: fixed height so the figure never reflows while it plays */}
      <div className="mt-1 h-[2.2rem] font-mono text-[0.7rem] leading-snug text-ink-soft sm:h-[1.2rem]">
        {hvInfo && hv !== null ? (
          <span>
            person {hv + 1}: {hvInfo.k} of {hvInfo.d} friends adopted = {Math.round(hvInfo.p * 100)}
            %{' '}
            {adopted[hv] ? '(adopted)' : hvInfo.p >= q - 1e-9 ? '≥ q, switches next' : '< q, stays'}
          </span>
        ) : stalled && holdouts.length ? (
          <span>
            stuck: the shaded group keeps {density !== null ? Math.round(density * 100) : ''}% of
            its ties inside, more than 1 − q = {Math.round((1 - q) * 100)}%
          </span>
        ) : stalled ? (
          <span>everyone switched</span>
        ) : (
          <span>round {r}: a person switches once at least q of their friends have</span>
        )}
      </div>

      {/* one supporting element: the threshold, as a draggable "how far it gets" strip */}
      <div className="mt-3 grid items-end gap-x-6 gap-y-3 border-t border-line pt-4 sm:grid-cols-[auto_minmax(0,480px)] sm:justify-between">
        <div className="flex items-end justify-between gap-4 sm:block">
          <BigStat
            label="adopted"
            value={
              <span>
                {nAdopted}
                <span className="text-muted">/{NODES.length}</span>
              </span>
            }
            color={GT.eq}
            sub={`at q = ${fmt(q)}`}
          />
          <div className="flex gap-2 sm:mt-3">
            <Btn primary onClick={toggle}>
              <span className="inline-block w-[6ch] text-center">
                {playing && !stalled ? 'pause' : stalled ? 'replay' : 'play'}
              </span>
            </Btn>
            <Btn
              onClick={() => {
                setSeeds(SEEDS);
                setQ(Q0);
                restart();
              }}
            >
              reset
            </Btn>
          </div>
        </div>
        <div className="min-w-0">
          <div className="label mb-1" style={{ color: GT.eq }}>
            drag to set q · final adopters for every q
          </div>
          <svg
            ref={chartRef}
            viewBox={`0 0 ${CW} ${CH}`}
            className="block w-full touch-none select-none"
            style={{ cursor: 'ew-resize' }}
            onPointerDown={onChart}
            onPointerMove={onChart}
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
                e.preventDefault();
                changeQ(Math.min(QMAX, Math.round((q + 0.05) * 20) / 20));
              } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
                e.preventDefault();
                changeQ(Math.max(QMIN, Math.round((q - 0.05) * 20) / 20));
              }
            }}
            role="slider"
            aria-label="threshold q"
            aria-valuemin={QMIN}
            aria-valuemax={QMAX}
            aria-valuenow={q}
          >
            {[0, NODES.length].map((n) => (
              <g key={n}>
                <line
                  x1={CP.l}
                  x2={CW - CP.r}
                  y1={cy(n)}
                  y2={cy(n)}
                  style={{ stroke: GT.line, strokeDasharray: '2 3' }}
                />
                <T x={CP.l - 6} y={cy(n)} anchor="end" size={13}>
                  {n}
                </T>
              </g>
            ))}
            {[0.25, 0.5, 0.75].map((t) => (
              <T key={t} x={cx(t)} y={CH - 8} size={13}>
                {t.toFixed(2)}
              </T>
            ))}
            <T x={CW - CP.r} y={CH - 8} anchor="end" size={13}>
              q →
            </T>
            <path d={curvePath} style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2.2 }} />
            <line
              x1={cx(q)}
              x2={cx(q)}
              y1={CP.t - 6}
              y2={CH - CP.b}
              style={{ stroke: GT.ink, strokeWidth: 1.5 }}
            />
            <circle
              cx={cx(q)}
              cy={cy(curve.find(([qq]) => Math.abs(qq - q) < 1e-6)?.[1] ?? 0)}
              r={6}
              style={{ fill: GT.ink }}
            />
          </svg>
        </div>
      </div>
    </Specimen>
  );
}
