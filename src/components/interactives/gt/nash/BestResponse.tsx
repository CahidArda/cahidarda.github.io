import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  GT,
  Seg,
  Specimen,
  SvgArrows,
  T,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import {
  GAMES,
  GAME_BY_ID,
  NASH_EVENT,
  arrows,
  equilibria,
  isCell,
  nextHop,
  type Cell,
  type GameId,
} from './games';
import { Reserve } from '../Reserve';

/* ── geometry: one SVG coordinate system ── */
const W = 420;
const H = 296;
const CW = 128; // cell width
const CH = 88; // cell height
const X0 = 96;
const GX = 52; // horizontal gap (red arrows live here)
const Y0 = 52;
const GY = 50; // vertical gap (blue arrows live here)
const cx = (c: number) => X0 + c * (CW + GX) + CW / 2;
const cy = (r: number) => Y0 + r * (CH + GY) + CH / 2;
const TOKEN_DY = 28;

const EQ_FILL = 'color-mix(in srgb, var(--color-accent) 13%, var(--color-paper-raised))';

/** Starting cell for a game: the first cell that is not an equilibrium. */
function startCell(id: GameId): Cell {
  const eq = equilibria(GAME_BY_ID[id]);
  const order: Cell[] = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  return order.find((c) => !isCell(eq, c)) ?? [0, 0];
}

/** Where a token dropped at `c` comes to rest (null if it never does). */
function restingCell(id: GameId, c: Cell): Cell | null {
  const g = GAME_BY_ID[id];
  let at = c;
  let last: 'row' | 'col' | null = null;
  for (let i = 0; i < 8; i++) {
    const h = nextHop(g, at, last);
    if (!h) return at;
    at = h.to;
    last = h.who;
  }
  return null;
}

const fmt = (v: number) => (v < 0 ? `−${-v}` : String(v));

export default function BestResponse() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [gid, setGid] = useState<GameId>('pd');
  const [cell, setCell] = useState<Cell>(startCell('pd'));
  const [lastWho, setLastWho] = useState<'row' | 'col' | null>(null);
  const [hops, setHops] = useState(0);
  const [touched, setTouched] = useState(false);
  const tourRef = useRef(0); // games visited by the auto-tour
  const pos = useRef({ x: cx(startCell('pd')[1]), y: cy(startCell('pd')[0]) + TOKEN_DY });
  const [, force] = useState(0);

  const g = GAME_BY_ID[gid];
  const eqs = equilibria(g);
  const arr = arrows(g);
  const hop = nextHop(g, cell, lastWho);
  const resting = hop === null;

  // Choose a game (from the tabs or from the sibling widget).
  const choose = (id: GameId, fromUser: boolean, broadcast = true) => {
    if (fromUser) setTouched(true);
    setGid(id);
    const s = startCell(id);
    setCell(reduced ? (restingCell(id, s) ?? s) : s);
    setLastWho(null);
    setHops(0);
    (window as unknown as { __gtNashGame?: GameId }).__gtNashGame = id;
    if (broadcast) window.dispatchEvent(new CustomEvent(NASH_EVENT, { detail: { id, src: 'br' } }));
  };

  useEffect(() => {
    const on = (e: Event) => {
      const d = (e as CustomEvent).detail as { id: GameId; src: string };
      if (d.src !== 'br') choose(d.id, true, false);
    };
    window.addEventListener(NASH_EVENT, on);
    return () => window.removeEventListener(NASH_EVENT, on);
  });

  // Reduced motion: show the poster (token already at rest).
  useEffect(() => {
    if (reduced) setCell((c) => restingCell(gid, c) ?? c);
  }, [reduced, gid]);

  // Hop along improvement arrows; when at rest, the auto-tour moves to the next game.
  useEffect(() => {
    if (!inView || reduced) return;
    if (hop) {
      const id = window.setTimeout(
        () => {
          setCell(hop.to);
          setLastWho(hop.who);
          setHops((h) => h + 1);
        },
        hops === 0 ? 1100 : 950,
      );
      return () => window.clearTimeout(id);
    }
    return undefined;
  }, [hop, inView, reduced, hops]);

  // Auto-tour: PD, stag hunt, hawk-dove, coordination, pennies, then stop.
  const settledOrDizzy = resting || hops >= 6;
  useEffect(() => {
    if (touched || reduced || !inView || !settledOrDizzy) return;
    if (tourRef.current >= GAMES.length - 1) return;
    const id = window.setTimeout(() => {
      tourRef.current += 1;
      choose(GAMES[tourRef.current].id, false);
    }, 2200);
    return () => window.clearTimeout(id);
  });

  // Glide the token in SVG user space.
  const target = { x: cx(cell[1]), y: cy(cell[0]) + TOKEN_DY };
  useFrame(
    (dt) => {
      const p = pos.current;
      const k = Math.min(1, dt * 9);
      p.x += (target.x - p.x) * k;
      p.y += (target.y - p.y) * k;
      force((v) => v + 1);
    },
    !reduced &&
      (Math.abs(pos.current.x - target.x) > 0.3 || Math.abs(pos.current.y - target.y) > 0.3),
  );
  if (reduced) pos.current = { ...target };

  const drop = (c: Cell) => {
    setTouched(true);
    setCell(reduced ? (restingCell(gid, c) ?? c) : c);
    setLastWho(null);
    setHops(0);
  };

  const STATUS = ['someone wants to move', 'circling forever', 'resting: nobody gains alone'];
  const statusIdx = resting ? 2 : gid === 'pennies' && hops > 0 ? 1 : 0;

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-01.1"
      title="best-response arrows"
      kicker="2 players · 2 moves"
      hint="pick a game, then click any cell to drop the token there"
    >
      <div className="grid gap-x-6 gap-y-4 sm:grid-cols-[minmax(0,34rem)_minmax(0,1fr)] sm:items-center">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full"
          role="img"
          aria-label={`${g.name} payoff grid with improvement arrows`}
        >
          <SvgArrows />
          {/* who moves which way, in the matrix corner */}
          <text
            x={X0 - 12}
            y={Y0 - 16}
            textAnchor="end"
            dominantBaseline="middle"
            style={{ fontFamily: GT.mono, fontSize: 10 }}
          >
            <tspan style={{ fill: GT.a }}>blue ↕</tspan>
            <tspan style={{ fill: GT.b }} dx={6}>
              red ↔
            </tspan>
          </text>
          {/* column labels (red player) and row labels (blue player) */}
          {g.cols.map((s, c) => (
            <T key={'c' + c} x={cx(c)} y={Y0 - 16} color={GT.b} size={11}>
              {s}
            </T>
          ))}
          {g.rows.map((s, r) => (
            <T key={'r' + r} x={X0 - 12} y={cy(r)} anchor="end" color={GT.a} size={11}>
              {s}
            </T>
          ))}

          {[0, 1].map((r) =>
            [0, 1].map((c) => {
              const eq = isCell(eqs, [r, c]);
              const here = cell[0] === r && cell[1] === c;
              const [ur, uc] = g.u[r][c];
              return (
                <g
                  key={`${r}${c}`}
                  onClick={() => drop([r, c])}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      drop([r, c]);
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  aria-label={`drop token at ${g.rows[r]}, ${g.cols[c]}`}
                  style={{ cursor: 'pointer', outline: 'none' }}
                >
                  <rect
                    x={cx(c) - CW / 2}
                    y={cy(r) - CH / 2}
                    width={CW}
                    height={CH}
                    style={{
                      fill: eq ? EQ_FILL : GT.paperRaised,
                      stroke: eq ? GT.eq : here ? GT.ink : GT.lineStrong,
                      strokeWidth: eq ? 2.5 : here ? 1.6 : 1,
                      transition: 'fill 250ms, stroke 250ms',
                    }}
                  />
                  {eq && (
                    <T
                      x={cx(c) - CW / 2 + 7}
                      y={cy(r) - CH / 2 + 11}
                      anchor="start"
                      size={9}
                      color={GT.eq}
                      weight={600}
                    >
                      NASH
                    </T>
                  )}
                  <T x={cx(c) - 8} y={cy(r) - 6} anchor="end" font="display" size={30} color={GT.a}>
                    {fmt(ur)}
                  </T>
                  <T x={cx(c)} y={cy(r) - 6} size={22} color={GT.muted} font="display">
                    ,
                  </T>
                  <T
                    x={cx(c) + 8}
                    y={cy(r) - 6}
                    anchor="start"
                    font="display"
                    size={30}
                    color={GT.b}
                  >
                    {fmt(uc)}
                  </T>
                </g>
              );
            }),
          )}

          {/* improvement arrows across the gaps; the one about to be taken stays bold */}
          {arr.map((a, i) => {
            const active =
              hop && a.from[0] === hop.from[0] && a.from[1] === hop.from[1] && a.who === hop.who;
            const op = !hop || active ? 1 : 0.4;
            if (a.who === 'row') {
              const x = cx(a.from[1]);
              const down = a.to[0] > a.from[0];
              const y1 = down ? Y0 + CH + 5 : Y0 + CH + GY - 5;
              const y2 = down ? Y0 + CH + GY - 5 : Y0 + CH + 5;
              return (
                <g key={i} style={{ opacity: op, transition: 'opacity 250ms' }}>
                  <line
                    x1={x}
                    y1={y1}
                    x2={x}
                    y2={y2}
                    markerEnd="url(#gt-arrow-a)"
                    style={{ stroke: GT.a, strokeWidth: 2.5 }}
                  />
                  <T x={x + 12} y={(y1 + y2) / 2 - 4} anchor="start" size={11} color={GT.a}>
                    +{a.gain}
                  </T>
                </g>
              );
            }
            const y = cy(a.from[0]);
            const right = a.to[1] > a.from[1];
            const x1 = right ? X0 + CW + 5 : X0 + CW + GX - 5;
            const x2 = right ? X0 + CW + GX - 5 : X0 + CW + 5;
            return (
              <g key={i} style={{ opacity: op, transition: 'opacity 250ms' }}>
                <line
                  x1={x1}
                  y1={y}
                  x2={x2}
                  y2={y}
                  markerEnd="url(#gt-arrow-b)"
                  style={{ stroke: GT.b, strokeWidth: 2.5 }}
                />
                <T x={(x1 + x2) / 2 - 4} y={y - 13} size={11} color={GT.b}>
                  +{a.gain}
                </T>
              </g>
            );
          })}

          {/* the token */}
          <circle
            cx={pos.current.x}
            cy={pos.current.y}
            r={8}
            style={{ fill: resting ? GT.eq : GT.ink, stroke: GT.paperRaised, strokeWidth: 2 }}
          />
        </svg>

        <div>
          <BigStat
            label="pure equilibria"
            value={eqs.length}
            color={eqs.length ? GT.eq : GT.ink}
            size="lg"
          />
          <Reserve
            className="mt-2 font-mono text-[0.68rem] leading-snug text-muted"
            items={STATUS.map((s, i) => (
              <span key={i}>
                <span className="text-ink">token: </span>
                {s}
              </span>
            ))}
            active={statusIdx}
          />
        </div>
      </div>

      <div className="mt-4">
        <Seg
          options={GAMES.map((x) => ({ value: x.id, label: x.name }))}
          value={gid}
          onChange={(v) => choose(v, true)}
        />
      </div>
      <Reserve
        items={GAMES.map((x) => (
          <Caption key={x.id}>{x.note}</Caption>
        ))}
        active={GAMES.findIndex((x) => x.id === gid)}
      />
    </Specimen>
  );
}
