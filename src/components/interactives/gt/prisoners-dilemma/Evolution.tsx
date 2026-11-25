import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
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
} from '../shared';
import { EVO_IDS, STRATS, evolve } from './pd';
import { useCompact } from '../useCompact';
import { Reserve } from '../Reserve';

const GENS = 650;
const H = 250;
const X0 = 40;
const Y0 = 214;
const Y1 = 18;
const sy = (v: number) => Y0 - v * (Y0 - Y1);
const STEP = 2; // sample every other generation for the paths
const PLAY_SECONDS = 7;

// EVO_IDS order: alld, allc, tft, pavlov (stacked bottom to top)
const COLOR: Record<string, string> = {
  alld: GT.b,
  allc: GT.alt,
  tft: GT.a,
  pavlov: GT.opt,
};

type Noise = '0' | '0.01' | '0.05';
const NOISES: { value: Noise; label: string }[] = [
  { value: '0', label: 'no mistakes' },
  { value: '0.01', label: '1 in 100' },
  { value: '0.05', label: '1 in 20' },
];
const CAPTION: Record<Noise, string> = {
  '0': 'Defectors feed on the unconditional cooperators, then starve once Tit for Tat takes over. Peace holds for good.',
  '0.01':
    'Tit for Tat clears out the defectors, then loses to Pavlov, which recovers from a slip in two rounds instead of feuding. Pavlov in turn is slowly undercut by defectors, and the cycle repeats.',
  '0.05':
    'Cooperation blooms for a moment, but with this many accidents every cooperator loses too much to feuds and exploitation. Defection takes over for good.',
};

function bands(shares: number[][], gx: (g: number) => number) {
  const n = EVO_IDS.length;
  const out: string[] = [];
  for (let k = 0; k < n; k++) {
    const top: string[] = [];
    const bot: string[] = [];
    for (let g = 0; g <= GENS; g += STEP) {
      let below = 0;
      for (let j = 0; j < k; j++) below += shares[g][j];
      const x = gx(g).toFixed(1);
      top.push(`${x},${sy(below + shares[g][k]).toFixed(1)}`);
      bot.push(`${x},${sy(below).toFixed(1)}`);
    }
    out.push(`M${top.join('L')}L${bot.reverse().join('L')}Z`);
  }
  return out;
}

export default function Evolution() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [sel, setSel] = useState<Noise>('0.01');
  const [noise, bind] = useHoverPreview(sel);
  const [boxRef, compact] = useCompact();
  const W = compact ? 400 : 720;
  const X1 = W - 10;
  const gx = (g: number) => X0 + (g / GENS) * (X1 - X0);
  const runs = useMemo(() => {
    const r = {} as Record<
      Noise,
      {
        shares: number[][];
        mean: number[];
        paths: string[];
        labels: { k: number; g: number; y: number }[];
      }
    >;
    for (const n of NOISES) {
      const e = evolve(Number(n.value), GENS);
      // one direct label per strategy, in the middle of its widest thick stretch
      const HW = 35;
      const labels: { k: number; g: number; y: number }[] = [];
      EVO_IDS.forEach((_, k) => {
        let best = -1;
        let bg = 0;
        for (let g = HW; g <= GENS - HW; g += 2) {
          let m = 1;
          for (let h = -HW; h <= HW; h += 5) m = Math.min(m, e.shares[g + h][k]);
          if (m > best) {
            best = m;
            bg = g;
          }
        }
        if (best < 0.12) return;
        let below = 0;
        for (let j = 0; j < k; j++) below += e.shares[bg][j];
        const y = below + e.shares[bg][k] / 2;
        // skip a label that would sit on top of one already placed
        const clash = labels.some(
          (l) => Math.abs(gx(l.g) - gx(bg)) < 110 && Math.abs(sy(l.y) - sy(y)) < 16,
        );
        if (!clash) labels.push({ k, g: bg, y });
      });
      r[n.value] = { ...e, paths: bands(e.shares, gx), labels };
    }
    return r;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [W]);
  const run = runs[noise];

  const [progress, setProgress] = useState(1); // SSR + reduced motion: the whole story
  const [playing, setPlaying] = useState(false);
  const [cursor, setCursor] = useState<number | null>(null);
  const started = useRef(false);
  const svgRef = useRef<SVGSVGElement>(null);

  const play = () => {
    if (reduced || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    setProgress(0);
    setPlaying(true);
  };
  useEffect(() => {
    if (reduced || !inView || started.current) return;
    started.current = true;
    play();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced, inView]);
  useFrame(
    (dt) => setProgress((p) => clamp(p + Math.max(0, dt) / PLAY_SECONDS, 0, 1)),
    playing && inView,
  );
  useEffect(() => {
    if (!reduced) return;
    setPlaying(false);
    setProgress(1);
  }, [reduced]);
  useEffect(() => {
    if (playing && progress >= 1) setPlaying(false);
  }, [playing, progress]);

  const choose = (v: Noise) => {
    setSel(v);
    play();
  };

  const head = Math.round(progress * GENS);
  const g = cursor === null ? head : Math.min(cursor, head);
  const fbar = run.mean[g];

  const onMove = (e: React.PointerEvent) => {
    if (!svgRef.current) return;
    const pt = svgPoint(svgRef.current, e.clientX, e.clientY);
    setCursor(Math.round(clamp((pt.x - X0) / (X1 - X0), 0, 1) * GENS));
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-02.4"
      title="survival of the fittest"
      kicker="replicator dynamics"
      hint="choose a mistake rate, then hover the chart to scrub through generations"
    >
      <div ref={boxRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full touch-none select-none"
          role="img"
          aria-label="Share of each strategy in the population over 650 generations"
          onPointerMove={onMove}
          onPointerDown={onMove}
          onPointerLeave={() => setCursor(null)}
        >
          <defs>
            <clipPath id="gt02-evo-clip">
              <rect x={X0} y={0} width={(X1 - X0) * progress} height={H} />
            </clipPath>
          </defs>
          <rect
            x={X0}
            y={Y1}
            width={X1 - X0}
            height={Y0 - Y1}
            style={{ fill: GT.paper, stroke: GT.lineStrong }}
          />
          <g clipPath="url(#gt02-evo-clip)">
            {run.paths.map((d, k) => (
              <path key={EVO_IDS[k]} d={d} style={{ fill: COLOR[EVO_IDS[k]], opacity: 0.88 }} />
            ))}
            {/* direct labels on the bands */}
            {run.labels.map(({ k, g: lg, y }) => {
              const x = gx(lg);
              const anchor = x < X0 + 70 ? 'start' : x > X1 - 70 ? 'end' : 'middle';
              return (
                <T
                  key={'l' + k}
                  x={anchor === 'start' ? x - 10 : anchor === 'end' ? x + 10 : x}
                  y={sy(y)}
                  anchor={anchor}
                  size={11}
                  weight={600}
                  color={GT.paper}
                >
                  {STRATS[EVO_IDS[k]].name}
                </T>
              );
            })}
          </g>
          {[0, 0.5, 1].map((v) => (
            <T key={v} x={X0 - 6} y={sy(v)} anchor="end" size={11}>
              {`${v * 100}%`}
            </T>
          ))}
          {[0, 200, 400, 600].map((v) => (
            <T key={v} x={gx(v)} y={Y0 + 14} size={11}>
              {v}
            </T>
          ))}
          <T x={(X0 + X1) / 2} y={Y0 + 30} size={11}>
            generation
          </T>
          <line
            x1={gx(g)}
            y1={Y1}
            x2={gx(g)}
            y2={Y0}
            style={{ stroke: GT.ink, strokeWidth: 1.5 }}
          />
          <T
            x={gx(g) > X1 - 60 ? gx(g) - 4 : gx(g) + 4}
            y={Y1 - 8}
            anchor={gx(g) > X1 - 60 ? 'end' : 'start'}
            size={10}
            color={GT.ink}
          >
            {`gen ${g}`}
          </T>
        </svg>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
        <BigStat
          label="average payoff"
          value={fbar.toFixed(2)}
          unit="/round"
          color={fbar > 2 ? GT.opt : GT.b}
          sub="3 = all cooperate, 1 = all defect"
        />
        <div className="flex flex-wrap items-center gap-2">
          <Seg label="mistakes" options={NOISES} value={noise} onChange={choose} bind={bind} />
          <Btn onClick={play} disabled={reduced}>
            replay
          </Btn>
        </div>
      </div>
      <Reserve
        items={NOISES.map((n) => (
          <Caption key={n.value}>{CAPTION[n.value]}</Caption>
        ))}
        active={NOISES.findIndex((n) => n.value === noise)}
      />
    </Specimen>
  );
}
