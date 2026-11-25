import { useEffect, useMemo, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  Chip,
  GT,
  Specimen,
  T,
  useInView,
  useReducedMotion,
} from '../shared';
import { STRATS, roundRobin } from './pd';

/* The field: five nice strategies, three that sometimes defect first. */
const IDS = ['tft', 'tf2t', 'grudger', 'pavlov', 'allc', 'random', 'alld', 'joss'];
const N = IDS.length;
const ROUNDS = 200;

// heat-map geometry
const W = 340;
const H = 330;
const MX = 64;
const MY = 58;
const C = 34;

const heat = (v: number) =>
  `color-mix(in srgb, var(--color-gt-a) ${Math.round((v / 5) * 85)}%, var(--color-paper-raised))`;

export default function Tournament() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const rr = useMemo(() => roundRobin(IDS, ROUNDS, 5), []);
  const total = N * N;
  const [played, setPlayed] = useState(total); // SSR + reduced motion: the finished table
  const [started, setStarted] = useState(false);
  const [pick, setPick] = useState<[number, number]>([0, 7]); // TFT vs Joss

  // Auto-play once on view: matches fill in row by row.
  useEffect(() => {
    if (reduced) {
      setPlayed(total);
      return;
    }
    if (!inView || started) return;
    setStarted(true);
    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) setPlayed(0);
  }, [reduced, inView, started, total]);
  useEffect(() => {
    if (played >= total || !inView) return;
    const id = window.setTimeout(() => setPlayed((p) => Math.min(total, p + 1)), 45);
    return () => window.clearTimeout(id);
  }, [played, total, inView]);

  // running totals over the matches played so far
  const shown = (i: number, j: number) => i * N + j < played;
  const partial = IDS.map((_, i) => {
    let s = 0;
    for (let j = 0; j < N; j++) if (shown(i, j)) s += rr.score[i][j];
    return s / N;
  });
  const order = IDS.map((_, i) => i).sort((a, b) => partial[b] - partial[a]);
  const done = played >= total;
  const leader = order[0];
  const niceTop = (() => {
    let k = 0;
    while (k < N && STRATS[IDS[order[k]]].nice) k++;
    return k;
  })();

  const [pi, pj] = pick;
  const m = rr.sample[pi][pj];
  const SHOW = 40;

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-02.3"
      title="round robin"
      kicker={`${N} programs · ${ROUNDS} rounds each`}
      hint="hover or tap a cell to replay the first 40 rounds of that match"
      footer={done ? `top ${niceTop} are all nice` : `${played} / ${total} matches`}
    >
      <div className="flex flex-wrap gap-x-10 gap-y-4">
        <BigStat
          label="leader"
          value={STRATS[IDS[leader]].name}
          color={STRATS[IDS[leader]].nice ? GT.a : GT.b}
          sub={`${partial[leader].toFixed(2)} points per round`}
        />
        <BigStat
          label="nice programs on top"
          value={done ? `${niceTop} of ${N}` : '…'}
          color={GT.a}
          sub="never defect first"
        />
      </div>

      <div className="mt-4 grid gap-5 sm:grid-cols-[minmax(0,340px)_1fr] sm:items-start">
        {/* score matrix */}
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full"
          role="img"
          aria-label="Average score of each row program against each column program"
        >
          <T x={MX + (N * C) / 2} y={12} size={9}>
            {'opponent →'}
          </T>
          {IDS.map((id, j) => (
            <g key={'c' + id} transform={`translate(${MX + j * C + C / 2} ${MY - 8}) rotate(-50)`}>
              <T x={0} y={0} anchor="start" size={9} color={STRATS[id].nice ? GT.a : GT.b}>
                {STRATS[id].short}
              </T>
            </g>
          ))}
          {IDS.map((id, i) => (
            <T
              key={'r' + id}
              x={MX - 6}
              y={MY + i * C + C / 2}
              anchor="end"
              size={9}
              color={STRATS[id].nice ? GT.a : GT.b}
            >
              {STRATS[id].short}
            </T>
          ))}
          {IDS.map((_, i) =>
            IDS.map((__, j) => {
              const on = shown(i, j);
              const v = rr.score[i][j];
              const sel = i === pi && j === pj;
              return (
                <g
                  key={`${i}-${j}`}
                  onMouseEnter={() => setPick([i, j])}
                  onClick={() => setPick([i, j])}
                  style={{ cursor: 'pointer' }}
                >
                  <rect
                    x={MX + j * C}
                    y={MY + i * C}
                    width={C}
                    height={C}
                    style={{
                      fill: on ? heat(v) : GT.paperRaised,
                      stroke: GT.line,
                      strokeWidth: 0.6,
                      transition: 'fill 200ms',
                    }}
                  />
                  {on && (
                    <T
                      x={MX + j * C + C / 2}
                      y={MY + i * C + C / 2 + 1}
                      size={10}
                      color={v > 2.6 ? GT.paper : GT.ink}
                    >
                      {v.toFixed(1)}
                    </T>
                  )}
                  {sel && (
                    <rect
                      x={MX + j * C + 1}
                      y={MY + i * C + 1}
                      width={C - 2}
                      height={C - 2}
                      style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2.5 }}
                    />
                  )}
                </g>
              );
            }),
          )}
          <T x={MX + (N * C) / 2} y={MY + N * C + 14} size={9}>
            row's average points per round against column
          </T>
        </svg>

        {/* leaderboard */}
        <div className="min-w-0">
          <div className="label mb-2">standings</div>
          <ol className="m-0 list-none p-0">
            {order.map((i, rank) => {
              const s = STRATS[IDS[i]];
              const c = s.nice ? GT.a : GT.b;
              return (
                <li
                  key={s.id}
                  className="mb-1.5 grid grid-cols-[1.2rem_8.6rem_1fr_2.4rem] items-center gap-2"
                  style={{ transition: 'all 300ms' }}
                >
                  <span className="font-mono text-[0.68rem] text-muted">{rank + 1}</span>
                  <span
                    className="truncate font-mono text-[0.72rem]"
                    style={{ color: c }}
                    title={s.rule}
                  >
                    {s.name}
                  </span>
                  <span className="relative block h-3 border border-line">
                    <span
                      className="absolute inset-y-0 left-0 block"
                      style={{
                        width: `${(partial[i] / 3) * 100}%`,
                        background: c,
                        transition: 'width 200ms',
                      }}
                    />
                  </span>
                  <span className="text-right font-mono text-[0.72rem] tabular-nums text-ink">
                    {partial[i].toFixed(2)}
                  </span>
                </li>
              );
            })}
          </ol>
          <div className="mt-2">
            <Btn onClick={() => setPlayed(reduced ? total : 0)} disabled={!done || reduced}>
              replay tournament
            </Btn>
          </div>
        </div>
      </div>

      {/* the selected match as two strips of moves */}
      <div className="mt-5 border-t border-line pt-3">
        <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
          <span className="font-mono text-[0.72rem] text-ink">
            {STRATS[IDS[pi]].name} <span className="text-muted">vs</span> {STRATS[IDS[pj]].name}
          </span>
          <span className="font-mono text-[0.68rem] text-muted">
            {m.scoreA.toFixed(2)} : {m.scoreB.toFixed(2)} per round
          </span>
        </div>
        {[m.a, m.b].map((moves, k) => (
          <div key={k} className="mb-1 flex gap-[2px]">
            {moves.slice(0, SHOW).map((mv, r) => (
              <span
                key={r}
                className="block h-4 min-w-0 flex-1"
                style={{ background: mv === 'C' ? GT.a : GT.b, opacity: mv === 'C' ? 0.85 : 1 }}
                title={`round ${r + 1}: ${mv === 'C' ? 'cooperate' : 'defect'}`}
              />
            ))}
          </div>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
        <Chip color={GT.a}>nice / cooperate</Chip>
        <Chip color={GT.b}>defects first sometimes / defect</Chip>
      </div>
      <Caption>
        Tit for Tat can never outscore the opponent in front of it, and neither can any nice program
        win a match by much. They top the table by scoring well against everyone.
      </Caption>
    </Specimen>
  );
}
