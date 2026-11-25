import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  GT,
  Slider,
  Specimen,
  cssVar,
  mulberry32,
  useFrame,
  useInView,
  useReducedMotion,
  useThemeVersion,
} from '../shared';
import { happy, makeGrid, stats, sweep, type Grid } from './schelling';
import { pct, prefersReduced, useSharedT } from './sharedT';

// Schelling's city: 40 x 40 lots, 10% empty, two kinds in equal numbers. Every frame a few
// unsatisfied residents move to a random empty lot. Hollow squares are unsatisfied.

const N = 40;
const SEED0 = 1971;
const newGrid = (seed: number) => makeGrid(N, mulberry32(seed));

// SSR poster: the initial mixed city as two compact paths (one per kind)
function posterPaths(g: Grid) {
  let a = '';
  let b = '';
  for (let i = 0; i < g.length; i++) {
    if (!g[i]) continue;
    const d = `M${i % N} ${(i / N) | 0}h.86v.86h-.86z`;
    if (g[i] === 1) a += d;
    else b += d;
  }
  return { a, b };
}

export default function City() {
  const reduced = useReducedMotion();
  const theme = useThemeVersion();
  const [rootRef, inView] = useInView();
  const [t, setT] = useSharedT();
  const [seed, setSeed] = useState(SEED0);
  const gridRef = useRef<Grid>(newGrid(SEED0));
  const rndRef = useRef(mulberry32(SEED0 + 1));
  const [mounted, setMounted] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [moves, setMoves] = useState(0);
  const [st, setSt] = useState(() => stats(gridRef.current, N, t));
  const started = useRef(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [px, setPx] = useState(400);

  const poster = useMemo(() => posterPaths(newGrid(SEED0)), []);

  const draw = () => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const W = px;
    if (cv.width !== Math.round(W * dpr)) {
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(W * dpr);
    }
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const ca = cssVar('--gt-a');
    const cb = cssVar('--gt-b');
    ctx.fillStyle = cssVar('--paper-raised');
    ctx.fillRect(0, 0, W, W);
    const cs = W / N;
    const g = gridRef.current;
    const tt = tRef.current;
    const inset = Math.max(0.5, cs * 0.07);
    for (let i = 0; i < g.length; i++) {
      const v = g[i];
      if (!v) continue;
      const x = (i % N) * cs;
      const y = ((i / N) | 0) * cs;
      const col = v === 1 ? ca : cb;
      if (happy(g, N, i, tt)) {
        ctx.fillStyle = col;
        ctx.fillRect(x + inset, y + inset, cs - 2 * inset, cs - 2 * inset);
      } else {
        ctx.strokeStyle = col;
        ctx.lineWidth = Math.max(1, cs * 0.14);
        const o = inset + ctx.lineWidth / 2;
        ctx.strokeRect(x + o, y + o, cs - 2 * o, cs - 2 * o);
      }
    }
  };
  const tRef = useRef(t);
  tRef.current = t;

  // size the canvas to its box
  useEffect(() => {
    setMounted(true);
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setPx(Math.max(160, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const refresh = () => {
    setSt(stats(gridRef.current, N, tRef.current));
    draw();
  };

  const settle = () => {
    for (let i = 0; i < 400; i++)
      if (sweep(gridRef.current, N, tRef.current, rndRef.current) === 0) break;
  };

  // auto-play once on view; reduced motion jumps to the settled city
  useEffect(() => {
    if (!mounted || started.current || !inView) return;
    started.current = true;
    if (reduced || prefersReduced()) {
      settle();
      refresh();
    } else setPlaying(true);
  }, [mounted, inView, reduced]);

  useEffect(() => {
    if (mounted) refresh();
  }, [px, theme, mounted, t]);

  // when t changes, start moving again
  useEffect(() => {
    if (!started.current) return;
    if (reduced) {
      settle();
      refresh();
    } else setPlaying(true);
  }, [t]);

  useFrame(
    () => {
      const g = gridRef.current;
      const rnd = rndRef.current;
      const movers: number[] = [];
      const empties: number[] = [];
      for (let i = 0; i < g.length; i++) {
        if (!g[i]) empties.push(i);
        else if (!happy(g, N, i, tRef.current)) movers.push(i);
      }
      if (movers.length === 0) {
        setPlaying(false);
        refresh();
        return;
      }
      const k = Math.min(movers.length, Math.max(3, Math.ceil(movers.length * 0.06)));
      let moved = 0;
      for (let n = 0; n < k; n++) {
        const j = Math.floor(rnd() * movers.length);
        const from = movers[j];
        movers[j] = movers[movers.length - 1];
        movers.pop();
        if (!g[from] || happy(g, N, from, tRef.current)) continue;
        const e = Math.floor(rnd() * empties.length);
        const to = empties[e];
        g[to] = g[from];
        g[from] = 0;
        empties[e] = from;
        moved++;
      }
      setMoves((m) => m + moved);
      refresh();
    },
    playing && inView && mounted,
  );

  const reset = (s: number) => {
    setSeed(s);
    gridRef.current = newGrid(s);
    rndRef.current = mulberry32(s + 1);
    setMoves(0);
    if (reduced) settle();
    else setPlaying(true);
    refresh();
  };

  const settled = mounted && st.unhappy === 0;

  return (
    <Specimen
      code="GT-09.2"
      title="the city"
      kicker={`${N} × ${N} lots · 10% empty`}
      rootRef={rootRef}
      hint="drag the bar; watch the city re-sort itself"
      footer={`${moves.toLocaleString('en-US')} moves${settled ? ' · settled' : ''}`}
    >
      <div className="grid grid-cols-[minmax(0,1fr)] gap-5 sm:grid-cols-[minmax(0,440px)_minmax(0,1fr)] sm:gap-8">
        <div ref={wrapRef} className="relative mx-auto w-full max-w-[440px] sm:mx-0">
          {mounted ? (
            <canvas
              ref={canvasRef}
              style={{ width: '100%', aspectRatio: '1 / 1', display: 'block' }}
              role="img"
              aria-label={`A city of two kinds of residents, ${pct(st.similarity)} of neighbours alike on average`}
            />
          ) : (
            <svg
              viewBox={`0 0 ${N} ${N}`}
              className="block w-full"
              role="img"
              aria-label="A mixed city"
            >
              <rect width={N} height={N} style={{ fill: GT.paperRaised }} />
              <path d={poster.a} style={{ fill: GT.a }} />
              <path d={poster.b} style={{ fill: GT.b }} />
            </svg>
          )}
        </div>

        <div className="flex min-w-0 flex-col gap-4 sm:pt-1">
          <BigStat
            label="you got"
            value={pct(st.similarity)}
            size="lg"
            color={GT.eq}
            sub={<span className="tabular-nums">neighbours alike, for a bar of {pct(t)}</span>}
          />
          <div
            className="h-[1rem] font-mono text-[0.68rem] tabular-nums"
            style={{ color: st.unhappy > 0 ? GT.warn : GT.opt }}
          >
            {st.unhappy > 0
              ? `${pct(st.unhappy)} unhappy (hollow), moving`
              : 'everyone is satisfied'}
          </div>
          <Slider
            label="each resident needs"
            value={t}
            min={0}
            max={0.9}
            step={0.01}
            onChange={setT}
            format={pct}
            color={GT.eq}
          />
          <div className="flex flex-wrap gap-2">
            <Btn primary onClick={() => setPlaying((p) => !p)} disabled={settled}>
              <span className="inline-block w-[5ch] text-center">{playing ? 'pause' : 'play'}</span>
            </Btn>
            <Btn onClick={() => reset(seed)}>reset</Btn>
            <Btn onClick={() => reset(seed + 1)}>new city</Btn>
          </div>
        </div>
      </div>
    </Specimen>
  );
}
