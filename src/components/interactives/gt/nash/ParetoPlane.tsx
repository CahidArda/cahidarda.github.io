import { useEffect, useState } from 'react';
import {
  BigStat,
  Caption,
  Chip,
  GT,
  Seg,
  Specimen,
  SvgArrows,
  T,
  useHoverPreview,
} from '../shared';
import {
  GAMES,
  GAME_BY_ID,
  NASH_EVENT,
  equilibria,
  isCell,
  paretoOptimal,
  type Cell,
  type GameId,
} from './games';

const W = 420;
const H = 300;
const X0 = 62;
const X1 = 392;
const Y0 = 262;
const Y1 = 26;

const CELLS: Cell[] = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

const CAPTION: Record<GameId, string> = {
  pd: 'The equilibrium (1, 1) sits inside the green box of outcomes that are better for both. Mutual cooperation is there, and nobody can reach it alone.',
  stag: 'Both corners are equilibria, but only the stag hunt is on the frontier. The hare equilibrium is safe and worse for both.',
  chicken:
    'Both equilibria are on the frontier: they are efficient, just unfair. The crash (0, 0) is the outcome everyone fears.',
  coord:
    'Both equilibria are efficient. The fight is over which one, since each player prefers a different one.',
  pennies:
    'Zero-sum: one player gains exactly what the other loses, so every outcome is efficient and no cell is stable. The mixed equilibrium pays (0, 0).',
};

const fmt = (v: number) => (v < 0 ? `−${-v}` : String(v));

export default function ParetoPlane() {
  const [sel, setSel] = useState<GameId>('pd');
  // pick up the game the sibling widget is showing if it hydrated first
  useEffect(() => {
    const last = (window as unknown as { __gtNashGame?: GameId }).__gtNashGame;
    if (last) setSel(last);
  }, []);
  const [gid, bind] = useHoverPreview(sel);

  useEffect(() => {
    const on = (e: Event) => {
      const d = (e as CustomEvent).detail as { id: GameId; src: string };
      if (d.src !== 'pp') setSel(d.id);
    };
    window.addEventListener(NASH_EVENT, on);
    return () => window.removeEventListener(NASH_EVENT, on);
  }, []);

  const commit = (id: GameId) => {
    setSel(id);
    (window as unknown as { __gtNashGame?: GameId }).__gtNashGame = id;
    window.dispatchEvent(new CustomEvent(NASH_EVENT, { detail: { id, src: 'pp' } }));
  };

  const g = GAME_BY_ID[gid];
  const zeroSum = gid === 'pennies';
  const lo = zeroSum ? -1 : 0;
  const hi = zeroSum ? 1 : 5;
  const px = (v: number) => X0 + ((v - lo) / (hi - lo)) * (X1 - X0);
  const py = (v: number) => Y0 - ((v - lo) / (hi - lo)) * (Y0 - Y1);
  const ticks = zeroSum ? [-1, 0, 1] : [0, 1, 2, 3, 4, 5];

  const eqs = equilibria(g);
  const po = paretoOptimal(g);
  const u = (c: Cell) => g.u[c[0]][c[1]];
  const total = (c: Cell) => u(c)[0] + u(c)[1];
  const bestTotal = Math.max(...CELLS.map(total));
  const eqTotals = zeroSum ? [0] : eqs.map(total);
  const eqLo = Math.min(...eqTotals);
  const eqHi = Math.max(...eqTotals);
  const dominatedEqs = eqs.filter((c) => !isCell(po, c));

  // frontier: Pareto points sorted left to right
  const frontier = [...po].sort((a, b) => u(a)[0] - u(b)[0]);

  const label = (c: Cell) => `${g.rows[c[0]][0]},${g.cols[c[1]][0]}`;

  return (
    <Specimen
      code="GT-01.3"
      title="equilibrium vs everyone's best"
      kicker="payoff plane"
      hint="hover a game to preview it, click to keep it"
      footer={`${dominatedEqs.length} equilibri${dominatedEqs.length === 1 ? 'um' : 'a'} beaten for both`}
    >
      <Seg
        options={GAMES.map((x) => ({ value: x.id, label: x.name }))}
        value={gid}
        onChange={commit}
        bind={bind}
      />
      <div className="mt-4 grid items-center gap-4 sm:grid-cols-[1fr_auto]">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full"
          role="img"
          aria-label={`${g.name}: the four outcomes plotted by blue's and red's payoff`}
        >
          <SvgArrows />
          {/* grid + axes */}
          {ticks.map((t) => (
            <g key={'t' + t}>
              <line
                x1={px(t)}
                y1={Y0}
                x2={px(t)}
                y2={Y1}
                style={{ stroke: GT.line, strokeWidth: 0.6 }}
              />
              <line
                x1={X0}
                y1={py(t)}
                x2={X1}
                y2={py(t)}
                style={{ stroke: GT.line, strokeWidth: 0.6 }}
              />
              <T x={px(t)} y={Y0 + 13} size={10} color={GT.a}>
                {fmt(t)}
              </T>
              <T x={X0 - 8} y={py(t)} anchor="end" size={10} color={GT.b}>
                {fmt(t)}
              </T>
            </g>
          ))}
          <line x1={X0} y1={Y0} x2={X1} y2={Y0} style={{ stroke: GT.lineStrong }} />
          <line x1={X0} y1={Y0} x2={X0} y2={Y1} style={{ stroke: GT.lineStrong }} />
          <T x={(X0 + X1) / 2} y={Y0 + 28} size={10} color={GT.a}>
            blue's payoff →
          </T>
          <g transform={`translate(${X0 - 34} ${(Y0 + Y1) / 2}) rotate(-90)`}>
            <T x={0} y={0} size={10} color={GT.b}>
              red's payoff →
            </T>
          </g>

          {/* "better for both" box above each dominated equilibrium */}
          {dominatedEqs.map((c) => {
            const [a, b] = u(c);
            return (
              <rect
                key={'box' + label(c)}
                x={px(a)}
                y={Y1}
                width={X1 - px(a)}
                height={py(b) - Y1}
                style={{ fill: GT.optSoft, opacity: 0.75 }}
              />
            );
          })}

          {/* Pareto frontier */}
          {frontier.length > 1 && (
            <polyline
              points={frontier.map((c) => `${px(u(c)[0])},${py(u(c)[1])}`).join(' ')}
              style={{ fill: 'none', stroke: GT.opt, strokeWidth: 1.5, strokeDasharray: '5 4' }}
            />
          )}

          {/* arrow from a dominated equilibrium to the best outcome above it */}
          {dominatedEqs.map((c) => {
            const [a, b] = u(c);
            const better = CELLS.filter(
              (d) => u(d)[0] >= a && u(d)[1] >= b && total(d) > a + b,
            ).sort((d1, d2) => total(d2) - total(d1))[0];
            if (!better) return null;
            const [a2, b2] = u(better);
            const dx = px(a2) - px(a);
            const dy = py(b2) - py(b);
            const len = Math.hypot(dx, dy);
            const k1 = 14 / len;
            const k2 = 12 / len;
            return (
              <line
                key={'ar' + label(c)}
                x1={px(a) + dx * k1}
                y1={py(b) + dy * k1}
                x2={px(a2) - dx * k2}
                y2={py(b2) - dy * k2}
                markerEnd="url(#gt-arrow-opt)"
                style={{ stroke: GT.opt, strokeWidth: 2 }}
              />
            );
          })}

          {/* the mixed equilibrium of matching pennies */}
          {zeroSum && (
            <g style={{ transform: `translate(${px(0)}px, ${py(0)}px)` }}>
              <rect
                x={-7}
                y={-7}
                width={14}
                height={14}
                transform="rotate(45)"
                style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2 }}
              />
              <T x={12} y={14} anchor="start" size={10} color={GT.eq} weight={600}>
                mixed Nash
              </T>
            </g>
          )}

          {/* the four outcomes */}
          {CELLS.map((c, i) => {
            const [a, b] = u(c);
            const eq = isCell(eqs, c);
            const opt = isCell(po, c);
            const left = px(a) > X1 - 70;
            // cells that land on the same point share one label (matching pennies)
            const twins = CELLS.filter((d) => u(d)[0] === a && u(d)[1] === b);
            const first = twins[0] === c;
            return (
              <g
                key={i}
                style={{
                  transform: `translate(${px(a)}px, ${py(b)}px)`,
                  transition: 'transform 450ms cubic-bezier(.3,.7,.2,1)',
                }}
              >
                {eq && <circle r={12} style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2.5 }} />}
                <circle
                  r={6}
                  style={{
                    fill: opt ? GT.opt : GT.paperRaised,
                    stroke: opt ? GT.opt : GT.muted,
                    strokeWidth: 2,
                  }}
                />
                <T
                  x={left ? -16 : 16}
                  y={-2}
                  anchor={left ? 'end' : 'start'}
                  size={13}
                  color={GT.ink}
                  weight={600}
                >
                  {first ? twins.map(label).join(' or ') : ''}
                </T>
                {first && (
                  <T x={left ? -16 : 16} y={14} anchor={left ? 'end' : 'start'} size={11}>
                    {`(${fmt(a)}, ${fmt(b)})${eq ? ' Nash' : ''}`}
                  </T>
                )}
              </g>
            );
          })}
        </svg>

        <div className="flex flex-row flex-wrap gap-6 sm:flex-col sm:gap-5">
          <BigStat
            label="equilibrium total"
            value={eqLo === eqHi ? fmt(eqLo) : `${fmt(eqLo)}–${fmt(eqHi)}`}
            color={GT.eq}
            size="lg"
            sub={zeroSum ? 'mixed: 50:50 each' : 'blue + red at Nash'}
          />
          <BigStat
            label="best total"
            value={fmt(bestTotal)}
            color={GT.opt}
            size="lg"
            sub="blue + red, best cell"
          />
        </div>
      </div>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
        <Chip color={GT.eq}>Nash equilibrium</Chip>
        <Chip color={GT.opt}>Pareto-optimal: can't help one without hurting the other</Chip>
      </div>
      <Caption>{CAPTION[gid]}</Caption>
    </Specimen>
  );
}
