import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
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
import { Reserve } from '../../gt/Reserve';
import { useBoost } from '../../gt/braess/useBoost';
import { useCompact } from '../../gt/useCompact';

/**
 * A pile of 4 stones; players take 1 or 2 in turn; whoever takes the last stone wins.
 * The whole game tree has 12 positions. You move first (levels 0, 2: "you pick one",
 * an ∃), they reply (levels 1, 3: "every reply", a ∀).
 *
 * "one path": a single winning line (you choose their moves too) is a certificate,
 *   3 moves to check. That is what NP asks for.
 * "every reply": a winning strategy. Taking 2 first fails (they take the last 2), so
 *   you take 1; then both of their replies must be answered. 5 moves to check, and the
 *   count doubles with every pair of turns in a longer game.
 */

type Node = { id: number; pile: number; parent: number | null; depth: number; col: number };
// col: horizontal slot in leaf units (leaves at integer columns, parents at averages)
const NODES: Node[] = [
  { id: 0, pile: 4, parent: null, depth: 0, col: 2.375 },
  { id: 1, pile: 3, parent: 0, depth: 1, col: 1.25 },
  { id: 2, pile: 2, parent: 0, depth: 1, col: 3.5 },
  { id: 3, pile: 2, parent: 1, depth: 2, col: 0.5 },
  { id: 4, pile: 1, parent: 1, depth: 2, col: 2 },
  { id: 5, pile: 1, parent: 2, depth: 2, col: 3 },
  { id: 6, pile: 0, parent: 2, depth: 2, col: 4 },
  { id: 7, pile: 1, parent: 3, depth: 3, col: 0 },
  { id: 8, pile: 0, parent: 3, depth: 3, col: 1 },
  { id: 9, pile: 0, parent: 4, depth: 3, col: 2 },
  { id: 10, pile: 0, parent: 5, depth: 3, col: 3 },
  { id: 11, pile: 0, parent: 7, depth: 4, col: 0 },
];
// a leaf (pile 0) was emptied by the player who moved into it: depth odd = you moved last
const youWin = (n: Node) => n.depth % 2 === 1;

type Mode = 'path' | 'game';
type Stroke = { to: number; at: number; tone: 'try' | 'bad' | 'good' };
const D = 0.42; // seconds per move drawn
const SCRIPT: Record<
  Mode,
  { segs: Stroke[]; marks: { node: number; at: number }[]; fade?: number; end: number }
> = {
  path: {
    segs: [
      { to: 2, at: 0, tone: 'good' },
      { to: 5, at: D, tone: 'good' },
      { to: 10, at: 2 * D, tone: 'good' },
    ],
    marks: [{ node: 10, at: 3 * D }],
    end: 3 * D + 0.4,
  },
  game: {
    segs: [
      { to: 2, at: 0, tone: 'try' },
      { to: 6, at: D, tone: 'bad' },
      { to: 1, at: 2 * D + 0.55, tone: 'good' },
      { to: 3, at: 3 * D + 0.55, tone: 'good' },
      { to: 4, at: 3 * D + 0.55, tone: 'good' },
      { to: 8, at: 4 * D + 0.55, tone: 'good' },
      { to: 9, at: 4 * D + 0.55, tone: 'good' },
    ],
    marks: [
      { node: 6, at: 2 * D },
      { node: 8, at: 5 * D + 0.55 },
      { node: 9, at: 5 * D + 0.55 },
    ],
    fade: 2 * D + 0.2,
    end: 5 * D + 0.95,
  },
};
const CHECK: Record<Mode, number> = { path: 3, game: 5 };

const CAPS: Record<Mode, string[]> = {
  path: [
    'A puzzle (NP): you may pick every move, theirs too. One line that ends in your win is the whole proof: 3 moves to check.',
  ],
  game: [
    'A game: they pick their own moves. Taking 2 first loses (they take the last 2), so take 1. Then every reply needs an answer: the proof is a tree, 5 moves to check.',
  ],
};

export default function GameTree() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(500);
  const W = compact ? 380 : 520;
  const X0 = compact ? 32 : 40;
  const DX = compact ? 64 : 88;
  const LABX = W - 8;
  const Y0 = 30;
  const DY = compact ? 64 : 62;
  const R = compact ? 14 : 15;
  const H = Y0 + 4 * DY + 30;
  const [svgRef, kb] = useBoost(W, 10, 9);
  const k = kb * (compact ? 1.15 : 1);

  const [sel, setSel] = useState<Mode>('game');
  const [mode, bind, previewing] = useHoverPreview(sel);
  const [t, setT] = useState(SCRIPT.game.end); // SSR: the finished strategy
  const tRef = useRef(t);
  const [playing, setPlaying] = useState(false);
  const started = useRef(false);
  const chain = useRef(false); // the first auto-demo plays "path", then "game"
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shown = useRef<Mode>('game');

  const play = (m: Mode, delay: number) => {
    if (timer.current) clearTimeout(timer.current);
    shown.current = m;
    tRef.current = 0;
    setT(0);
    setPlaying(false);
    if (reduced) {
      tRef.current = SCRIPT[m].end;
      setT(SCRIPT[m].end);
      return;
    }
    timer.current = setTimeout(() => setPlaying(true), delay);
  };

  useEffect(() => {
    if (!inView || started.current || reduced) return;
    started.current = true;
    chain.current = true;
    setSel('path');
    play('path', 400);
  }, [inView, reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  // a hover preview (or a committed change) replays that mode from the start
  useEffect(() => {
    if (!started.current && !previewing) return;
    if (shown.current !== mode) play(mode, 120);
  }, [mode]);

  // the reduced-motion setting can arrive after the first effect: show the finished strategy
  useEffect(() => {
    if (!reduced) return;
    if (timer.current) clearTimeout(timer.current);
    chain.current = false;
    setPlaying(false);
    shown.current = 'game';
    setSel('game');
    tRef.current = SCRIPT.game.end;
    setT(SCRIPT.game.end);
  }, [reduced]);
  useFrame((dt) => {
    const end = SCRIPT[shown.current].end;
    const v = Math.min(end, tRef.current + Math.max(0, dt));
    tRef.current = v;
    setT(v);
    if (v >= end) {
      setPlaying(false);
      if (chain.current && shown.current === 'path') {
        chain.current = false;
        timer.current = setTimeout(() => {
          setSel('game');
        }, 900);
      }
    }
  }, playing && inView);

  const replay = () => {
    chain.current = false;
    play(sel, 1000);
  };

  const sc = SCRIPT[mode === shown.current ? mode : shown.current];
  const pos = (n: Node) => ({ x: X0 + n.col * DX, y: Y0 + n.depth * DY });
  const byId = (id: number) => NODES[id];
  const toneColor = { try: GT.ink, bad: GT.b, good: GT.opt } as const;
  const fadeA = sc.fade !== undefined ? 1 - 0.7 * clamp((t - sc.fade) / 0.35, 0, 1) : 1;

  // the segment from a node's parent edge to its circle edge
  const edge = (to: number) => {
    const b = byId(to);
    const a = byId(b.parent!);
    const pa = pos(a);
    const pb = pos(b);
    const len = Math.hypot(pb.x - pa.x, pb.y - pa.y);
    const ux = (pb.x - pa.x) / len;
    const uy = (pb.y - pa.y) / len;
    const rb = b.pile === 0 ? R * 0.85 : R;
    return {
      x1: pa.x + ux * R,
      y1: pa.y + uy * R,
      x2: pb.x - ux * rb,
      y2: pb.y - uy * rb,
      mx: (pa.x + pb.x) / 2,
      my: (pa.y + pb.y) / 2,
      take: a.pile - b.pile,
    };
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-11.2"
      title="why games are harder"
      kicker="take 1 or 2; last stone wins"
      hint="hover a tab to replay it; click to keep it"
      footer="Sipser §8.3"
    >
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label="The game tree of a 4-stone take-1-or-2 game, with the winning line or the winning strategy highlighted"
        >
          {/* level bands: who moves */}
          {[0, 1, 2, 3].map((d) => (
            <g key={d}>
              <T
                x={LABX}
                y={Y0 + d * DY}
                anchor="end"
                size={9.5 * k}
                color={d % 2 === 0 ? GT.a : GT.b}
                weight={700}
              >
                {d % 2 === 0
                  ? compact
                    ? 'you: ∃'
                    : 'you pick one ∃'
                  : compact
                    ? 'they: ∀'
                    : 'they pick any ∀'}
              </T>
            </g>
          ))}
          {/* the full tree, faint */}
          {NODES.filter((n) => n.parent !== null).map((n) => {
            const e = edge(n.id);
            return (
              <g key={`e${n.id}`}>
                <line
                  x1={e.x1}
                  y1={e.y1}
                  x2={e.x2}
                  y2={e.y2}
                  style={{ stroke: GT.lineStrong, strokeWidth: 1.2 }}
                />
                <T
                  x={e.mx + (e.x2 > e.x1 ? 7 : e.x2 < e.x1 ? -7 : 8)}
                  y={e.my - 2}
                  size={8.5 * k}
                  anchor={e.x2 >= e.x1 ? 'start' : 'end'}
                >
                  {`−${e.take}`}
                </T>
              </g>
            );
          })}
          {/* the highlighted proof, drawn along the visible links */}
          {sc.segs.map((s, i) => {
            const e = edge(s.to);
            const u = clamp((t - s.at) / D, 0, 1);
            if (u <= 0) return null;
            const faded = s.tone !== 'good' ? fadeA : 1;
            return (
              <line
                key={`h${i}`}
                x1={e.x1}
                y1={e.y1}
                x2={e.x1 + (e.x2 - e.x1) * u}
                y2={e.y1 + (e.y2 - e.y1) * u}
                opacity={faded}
                style={{ stroke: toneColor[s.tone], strokeWidth: 3.5, strokeLinecap: 'round' }}
              />
            );
          })}
          {/* positions */}
          {NODES.map((n) => {
            const { x, y } = pos(n);
            const onPath = n.id === 0 || sc.segs.some((s) => s.to === n.id && t >= s.at + D);
            const mark = sc.marks.find((m) => m.node === n.id);
            const ma = mark ? clamp((t - mark.at) / 0.25, 0, 1) : 0;
            if (n.pile === 0) {
              const win = youWin(n);
              const c = win ? GT.opt : GT.b;
              const s = R * 0.85;
              const faded = !win ? fadeA : 1;
              return (
                <g key={n.id}>
                  <rect
                    x={x - s * 1.3}
                    y={y - s}
                    width={2.6 * s}
                    height={2 * s}
                    style={{ fill: GT.paperRaised, stroke: c, strokeWidth: 1.5 }}
                  />
                  <rect
                    x={x - s * 1.3}
                    y={y - s}
                    width={2.6 * s}
                    height={2 * s}
                    opacity={ma * faded}
                    style={{ fill: c }}
                  />
                  <T x={x} y={y + 0.5} size={8.5 * k} color={c} weight={700}>
                    {win ? 'win' : 'lose'}
                  </T>
                  <T x={x} y={y + 0.5} size={8.5 * k} color={GT.paperRaised} weight={700}>
                    <tspan opacity={ma * faded}>{win ? 'win' : 'lose'}</tspan>
                  </T>
                </g>
              );
            }
            const mine = n.depth % 2 === 0;
            return (
              <g key={n.id}>
                <circle
                  cx={x}
                  cy={y}
                  r={R}
                  style={{
                    fill: GT.paperRaised,
                    stroke: onPath ? GT.ink : GT.lineStrong,
                    strokeWidth: onPath ? 2 : 1.2,
                  }}
                />
                <T
                  x={x}
                  y={y + 0.5}
                  size={12 * k}
                  font="display"
                  color={mine ? GT.a : GT.b}
                  weight={600}
                >
                  {n.pile}
                </T>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-3 flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
        <BigStat
          label="moves to check"
          value={CHECK[mode]}
          color={mode === 'path' ? GT.a : GT.opt}
          sub={
            mode === 'path'
              ? '40-move puzzle: 40 to check'
              : '40-move game, 2 replies a turn: about a million'
          }
        />
        <div className="flex flex-wrap items-center gap-3">
          <Seg
            options={[
              { value: 'path', label: 'one path (NP)' },
              { value: 'game', label: 'every reply (game)' },
            ]}
            value={mode}
            onChange={(v) => {
              chain.current = false;
              if (v === sel) play(v, 120);
              setSel(v);
            }}
            bind={bind}
          />
          <Btn primary onClick={replay}>
            replay
          </Btn>
        </div>
      </div>
      <Reserve
        items={(['path', 'game'] as Mode[]).map((m) => (
          <Caption key={m}>{CAPS[m][0]}</Caption>
        ))}
        active={mode === 'path' ? 0 : 1}
      />
    </Specimen>
  );
}
