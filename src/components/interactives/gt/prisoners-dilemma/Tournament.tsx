import { useEffect, useMemo, useState } from 'react';
import { BigStat, Btn, Caption, GT, Specimen, T, useInView, useReducedMotion } from '../shared';
import { useCompact } from '../useCompact';
import { STRATS, roundRobin } from './pd';

/* The field: five nice strategies, three that sometimes defect first. */
const IDS = ['tft', 'tf2t', 'grudger', 'pavlov', 'allc', 'random', 'alld', 'joss'];
const N = IDS.length;
const ROUNDS = 200;

// standings chart: one row per program, sorted by score, bars grow as matches are played
const RH = 32;
const TOP = 22;
const H = TOP + N * RH + 6;
const MAXV = 3;

export default function Tournament() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [boxRef, compact] = useCompact();
  const W = compact ? 400 : 680;
  const NX = compact ? 112 : 150; // right edge of the names
  const BX = NX + 10; // bars start
  const BW = compact ? 190 : 330; // width of a 3-point bar
  const fs = compact ? 11.5 : 11;
  const bx = (v: number) => BX + (v / MAXV) * BW;

  const rr = useMemo(() => roundRobin(IDS, ROUNDS, 5), []);
  const total = N * N;
  const [played, setPlayed] = useState(total); // SSR + reduced motion: the finished table
  const [started, setStarted] = useState(false);

  // Auto-play once on view: matches are played one by one and the bars grow.
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
    const id = window.setTimeout(() => setPlayed((p) => Math.min(total, p + 1)), 60);
    return () => window.clearTimeout(id);
  }, [played, total, inView]);

  // running totals over the matches played so far (match k = row k / N against column k % N)
  const shown = (i: number, j: number) => i * N + j < played;
  const partial = IDS.map((_, i) => {
    let s = 0;
    for (let j = 0; j < N; j++) if (shown(i, j)) s += rr.score[i][j];
    return s / N;
  });
  const order = IDS.map((_, i) => i).sort((a, b) => partial[b] - partial[a] || a - b);
  const rank = IDS.map((_, i) => order.indexOf(i));
  const done = played >= total;
  const leader = order[0];
  const niceTop = (() => {
    let k = 0;
    while (k < N && STRATS[IDS[order[k]]].nice) k++;
    return k;
  })();
  const nNice = IDS.filter((id) => STRATS[id].nice).length;
  const split = done && niceTop === nNice;
  const ry = (r: number) => TOP + r * RH;

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-02.3"
      title="round robin"
      kicker={`${N} programs · ${ROUNDS} rounds a match`}
      hint="watch the table fill, then replay it"
    >
      <div ref={boxRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full"
          role="img"
          aria-label="Standings: average points per round of each program against the whole field"
        >
          <T x={BX} y={8} anchor="start" size={10}>
            average points per round, against everyone
          </T>
          <line
            x1={BX}
            y1={TOP - 4}
            x2={BX}
            y2={H - 6}
            style={{ stroke: GT.lineStrong, strokeWidth: 1 }}
          />
          {IDS.map((id, i) => {
            const s = STRATS[id];
            const c = s.nice ? GT.a : GT.b;
            const v = partial[i];
            return (
              <g
                key={id}
                style={{
                  transform: `translateY(${ry(rank[i])}px)`,
                  transition: reduced ? undefined : 'transform 350ms ease',
                }}
              >
                <title>{s.rule}</title>
                <T x={NX} y={RH / 2} anchor="end" size={fs} color={c} weight={600}>
                  {s.name}
                </T>
                <rect
                  x={BX}
                  y={RH / 2 - 8}
                  width={Math.max(0, bx(v) - BX)}
                  height={16}
                  style={{ fill: c, opacity: s.nice ? 0.9 : 0.85 }}
                />
                <T x={bx(v) + 6} y={RH / 2} anchor="start" size={fs} color={GT.ink}>
                  {v.toFixed(2)}
                </T>
              </g>
            );
          })}
          {/* the split Axelrod found: every nice program above every other one */}
          {split && (
            <g>
              <line
                x1={0}
                y1={ry(niceTop)}
                x2={W}
                y2={ry(niceTop)}
                style={{ stroke: GT.ink, strokeWidth: 1, strokeDasharray: '4 3' }}
              />
              <T x={W} y={ry(niceTop) - 9} anchor="end" size={10} color={GT.a} weight={600}>
                {compact ? 'nice ↑' : 'nice: never defect first ↑'}
              </T>
              <T x={W} y={ry(niceTop) + 10} anchor="end" size={10} color={GT.b} weight={600}>
                {compact ? 'not nice ↓' : 'defect first sometimes ↓'}
              </T>
            </g>
          )}
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-[minmax(0,1fr)_auto] items-end gap-4">
        <BigStat
          label={done ? 'winner' : 'leading'}
          value={<span className="block truncate">{STRATS[IDS[leader]].name}</span>}
          color={STRATS[IDS[leader]].nice ? GT.a : GT.b}
          sub={`${played} of ${total} matches played`}
        />
        <Btn onClick={() => setPlayed(reduced ? total : 0)} disabled={!done || reduced}>
          replay tournament
        </Btn>
      </div>
      <Caption>
        No nice program can beat the opponent in front of it by much. They top the table by doing
        well against everyone.
      </Caption>
    </Specimen>
  );
}
