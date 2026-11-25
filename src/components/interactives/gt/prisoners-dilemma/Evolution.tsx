import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  Chip,
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

const GENS = 650;
const W = 500;
const H = 260;
const X0 = 46;
const X1 = 490;
const Y0 = 228;
const Y1 = 14;
const gx = (g: number) => X0 + (g / GENS) * (X1 - X0);
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

function bands(shares: number[][]) {
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
  const runs = useMemo(() => {
    const r = {} as Record<Noise, { shares: number[][]; mean: number[]; paths: string[] }>;
    for (const n of NOISES) {
      const e = evolve(Number(n.value), GENS);
      r[n.value] = { ...e, paths: bands(e.shares) };
    }
    return r;
  }, []);
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
  const x = run.shares[g];
  const lead = x.indexOf(Math.max(...x));
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
      footer={`generation ${g}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Seg label="mistakes" options={NOISES} value={noise} onChange={choose} bind={bind} />
        <Btn onClick={play} disabled={reduced}>
          replay
        </Btn>
      </div>

      <div className="mt-4 flex flex-wrap gap-x-10 gap-y-4">
        <BigStat
          label="average payoff"
          value={fbar.toFixed(2)}
          unit="/round"
          color={fbar > 2 ? GT.opt : GT.b}
          sub="3 = everyone cooperates, 1 = everyone defects"
        />
        <BigStat
          label="most common"
          value={STRATS[EVO_IDS[lead]].name}
          color={COLOR[EVO_IDS[lead]]}
          sub={`${Math.round(x[lead] * 100)}% of the population`}
        />
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="mt-3 block w-full touch-none select-none"
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
        <T x={(X0 + X1) / 2} y={Y0 + 26} size={11}>
          generation
        </T>
        <line x1={gx(g)} y1={Y1} x2={gx(g)} y2={Y0} style={{ stroke: GT.ink, strokeWidth: 1.5 }} />
      </svg>

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {EVO_IDS.map((id, k) => (
          <Chip key={id} color={COLOR[id]}>
            {`${STRATS[id].name} ${Math.round(x[k] * 100)}%`}
          </Chip>
        ))}
      </div>
      <Caption>{CAPTION[noise]}</Caption>
    </Specimen>
  );
}
