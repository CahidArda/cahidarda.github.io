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
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../shared';
import { useNarrow } from './useNarrow';

/**
 * Four tight circles of six friends (strong ties) joined by four acquaintances (weak ties).
 * News spreads along every tie; a strong tie passes it on in 1 round, a weak tie in 3 (you
 * see acquaintances less often). Cut the weak ties and the news stays home. Cut as many
 * strong ties and nothing changes: inside a circle there is always another way round.
 */
/** Round to 0.01 so server and client trig agree (avoids hydration mismatches). */
const r2 = (v: number) => Math.round(v * 100) / 100;
type Center = { x: number; y: number; name: string; lx: number; ly: number };
type Node = { x: number; y: number; c: number };
const PER = 6;
function layout(W: number, H: number, CENTERS: Center[], RAD: number) {
  const nodes: Node[] = [];
  CENTERS.forEach((ctr, c) => {
    for (let i = 0; i < PER; i++) {
      const a = (2 * Math.PI * i) / PER;
      nodes.push({ x: r2(ctr.x + RAD * Math.cos(a)), y: r2(ctr.y + RAD * Math.sin(a)), c });
    }
  });
  return { W, H, CENTERS, nodes };
}
const WIDE = layout(
  600,
  380,
  [
    { x: 135, y: 105, name: 'A', lx: 30, ly: 40 },
    { x: 465, y: 105, name: 'B', lx: 570, ly: 40 },
    { x: 135, y: 280, name: 'C', lx: 30, ly: 345 },
    { x: 465, y: 280, name: 'D', lx: 570, ly: 345 },
  ],
  54,
);
// phones: the same four circles, packed closer so labels stay legible
const NARROW = layout(
  380,
  400,
  [
    { x: 100, y: 118, name: 'A', lx: 18, ly: 28 },
    { x: 280, y: 118, name: 'B', lx: 362, ly: 28 },
    { x: 100, y: 298, name: 'C', lx: 18, ly: 385 },
    { x: 280, y: 298, name: 'D', lx: 362, ly: 385 },
  ],
  50,
);
const N_PEOPLE = 4 * PER;
// node index helper: circle c, slot i (slot 0 points right, 1.5 down, 3 left, 4.5 up)
const id = (c: number, i: number) => c * PER + i;

type Tie = { a: number; b: number; weak: boolean };
const strong: Tie[] = [];
for (let c = 0; c < 4; c++)
  for (let i = 0; i < PER; i++) {
    strong.push({ a: id(c, i), b: id(c, (i + 1) % PER), weak: false });
    if (i < PER) strong.push({ a: id(c, i), b: id(c, (i + 2) % PER), weak: false });
  }
// de-duplicate (i, i+2) pairs
const seen = new Set<string>();
const strongTies = strong.filter((t) => {
  const k = `${Math.min(t.a, t.b)}-${Math.max(t.a, t.b)}`;
  if (seen.has(k)) return false;
  seen.add(k);
  return true;
});
const weakTies: Tie[] = [
  { a: id(0, 0), b: id(1, 3), weak: true }, // A - B
  { a: id(1, 1), b: id(3, 5), weak: true }, // B - D
  { a: id(3, 3), b: id(2, 0), weak: true }, // D - C
  { a: id(2, 4), b: id(0, 2), weak: true }, // C - A
];
// four strong ties to cut for comparison: one per circle, away from the weak-tie ends
const strongCut = new Set([
  `${id(0, 3)}-${id(0, 4)}`,
  `${id(1, 4)}-${id(1, 5)}`,
  `${id(2, 1)}-${id(2, 2)}`,
  `${id(3, 1)}-${id(3, 2)}`,
]);
const key = (t: Tie) => `${Math.min(t.a, t.b)}-${Math.max(t.a, t.b)}`;

type Cut = 'none' | 'weak' | 'strong';
const ROUND = 0.42; // seconds per round
const WEAK_DELAY = 3;

/** Earliest round each person hears the news, and the tie it came over. */
function spread(source: number, cut: Cut) {
  const ties = [
    ...strongTies.filter((t) => !(cut === 'strong' && strongCut.has(key(t)))),
    ...(cut === 'weak' ? [] : weakTies),
  ];
  const n = N_PEOPLE;
  const time = new Array(n).fill(Infinity);
  const via: (Tie | null)[] = new Array(n).fill(null);
  time[source] = 0;
  const done = new Array(n).fill(false);
  for (let it = 0; it < n; it++) {
    let u = -1;
    for (let i = 0; i < n; i++) if (!done[i] && (u < 0 || time[i] < time[u])) u = i;
    if (u < 0 || time[u] === Infinity) break;
    done[u] = true;
    for (const t of ties) {
      const v = t.a === u ? t.b : t.b === u ? t.a : -1;
      if (v < 0) continue;
      const nt = time[u] + (t.weak ? WEAK_DELAY : 1);
      if (nt < time[v]) {
        time[v] = nt;
        via[v] = t;
      }
    }
  }
  return { time, via, end: Math.max(...time.filter((x) => x < Infinity)) };
}

export default function WeakTies() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView<HTMLDivElement>();
  const [sel, setSel] = useState<Cut>('none');
  const [cut, bindCut] = useHoverPreview<Cut>(sel);
  const [source, setSource] = useState(id(0, 4));
  const [clock, setClock] = useState(99);
  const clockRef = useRef(99);
  clockRef.current = clock;
  const [playing, setPlaying] = useState(false);
  const started = useRef(false);
  const narrow = useNarrow(rootRef);
  const { W, H, CENTERS, nodes } = narrow ? NARROW : WIDE;

  const res = useMemo(() => spread(source, cut), [source, cut]);

  const play = () => {
    if (reduced) {
      setClock(99);
      return;
    }
    setClock(0);
    setPlaying(true);
  };
  useEffect(() => {
    if (started.current || !inView || reduced) return;
    started.current = true;
    play();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, reduced]);
  // replay when the cut or the source changes (hovering a tab previews it)
  useEffect(() => {
    if (started.current) play();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cut, source]);

  useFrame(
    (dt) => {
      const nc = clockRef.current + dt / ROUND;
      setClock(nc);
      if (nc > res.end + 0.6) setPlaying(false);
    },
    playing && inView && !reduced,
  );

  const heard = res.time.filter((t) => t <= clock).length;
  const circles = new Set(
    nodes.map((nd, i) => (res.time[i] <= clock ? nd.c : -1)).filter((c) => c >= 0),
  ).size;
  const usedVia = new Set(res.via.filter((t, i) => t && res.time[i] <= clock).map((t) => key(t!)));

  const tieLine = (t: Tie, i: number) => {
    const a = nodes[t.a];
    const b = nodes[t.b];
    const isCut =
      (cut === 'weak' && t.weak) || (cut === 'strong' && !t.weak && strongCut.has(key(t)));
    const carried = usedVia.has(key(t)) && !isCut;
    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2;
    return (
      <g key={i}>
        <line
          x1={a.x}
          y1={a.y}
          x2={b.x}
          y2={b.y}
          strokeDasharray={isCut ? '3 4' : t.weak ? '7 4' : undefined}
          style={{
            stroke: carried ? GT.eq : t.weak ? GT.alt : GT.ink,
            strokeWidth: t.weak ? (carried ? 2.2 : 1.6) : carried ? 4 : 3,
            opacity: isCut ? 0.25 : carried ? 1 : t.weak ? 0.9 : 0.35,
          }}
        />
        {isCut && (
          <T x={mx} y={my} size={13} color={GT.b} weight={700}>
            ×
          </T>
        )}
      </g>
    );
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-07.2"
      title="weak ties carry news"
      kicker="24 people · 4 circles"
      hint="click anyone to start a rumour; cut weak or strong ties"
      footer={
        <span className="inline-flex flex-wrap gap-x-3 gap-y-1">
          <Chip color={GT.ink}>close friend</Chip>
          <Chip color={GT.alt}>acquaintance</Chip>
          <Chip color={GT.eq}>has heard</Chip>
        </span>
      }
    >
      <div className="mb-4 grid grid-cols-2 gap-3">
        <BigStat
          label="heard the news"
          value={`${heard} / ${nodes.length}`}
          color={GT.eq}
          size="lg"
        />
        <BigStat label="circles reached" value={`${circles} / 4`} size="lg" />
      </div>
      <div className="mb-3">
        <Seg<Cut>
          label="cut"
          value={sel}
          onChange={setSel}
          bind={bindCut}
          options={[
            { value: 'none', label: 'nothing' },
            { value: 'weak', label: '4 acquaintance ties' },
            { value: 'strong', label: '4 friend ties' },
          ]}
        />
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full"
        role="img"
        aria-label={`${heard} of ${nodes.length} people have heard the news, in ${circles} of 4 circles`}
      >
        {CENTERS.map((c, i) => (
          <T key={i} x={c.lx} y={c.ly} size={22} font="display" color={GT.muted}>
            {c.name}
          </T>
        ))}
        {strongTies.map((t, i) => tieLine(t, i))}
        {weakTies.map((t, i) => tieLine(t, 100 + i))}
        {nodes.map((nd, i) => {
          const on = res.time[i] <= clock;
          const isSrc = i === source;
          return (
            <g key={i} onClick={() => setSource(i)} style={{ cursor: 'pointer' }}>
              <circle cx={nd.x} cy={nd.y} r={15} style={{ fill: 'transparent' }} />
              <circle
                cx={nd.x}
                cy={nd.y}
                r={isSrc ? 11 : 9}
                style={{
                  fill: on ? GT.eq : GT.paperRaised,
                  stroke: on ? GT.eq : GT.ink,
                  strokeWidth: 1.6,
                  transition: 'fill 200ms',
                }}
              />
            </g>
          );
        })}
        <T
          x={nodes[source].x}
          y={nodes[source].y + (nodes[source].y < CENTERS[nodes[source].c].y ? -22 : 24)}
          size={10}
          color={GT.eq}
          weight={600}
        >
          job opening
        </T>
      </svg>
      <div className="mt-3 flex gap-2">
        <Btn primary onClick={play}>
          play again
        </Btn>
      </div>
      <Caption>
        Inside a circle everyone already knows what everyone knows. New information arrives over the
        thin lines, from people you see once in a while.
      </Caption>
    </Specimen>
  );
}
