import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  GT,
  Specimen,
  T,
  clamp,
  lerp,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useBoost } from '../../gt/braess/useBoost';
import { useCompact } from '../../gt/useCompact';
import { RULE_SHORT, RULE_TEXT, apply, iCount, matchAt, type Rule } from './miu';

/**
 * CX-10.1 · the MIU playground. Hofstadter's MU puzzle: start at MI, apply four one-way rules,
 * try to reach MU. The letters are tiles that slide when a rule fires; under them a number line
 * of I-counts 0..19 with every multiple of 3 marked. The marker hops (II doubles, III takes 3
 * away, I and IV leave it) and never lands on a marked square, so MU (I-count 0) stays out of
 * reach. miu.ts checks the invariant by search: all 15,885 strings within 8 steps.
 *
 * One clock (seconds) drives every transition; the drawing is a pure function of it.
 */

const MAXLEN = 20;
const MAXI = MAXLEN - 1;
const DUR = 0.5; // one rule's slide
const DEMO: Rule[] = [2, 2, 2, 3, 3, 4];
const DEMO_GAP = 1.05;
const ROMAN = ['', 'I', 'II', 'III', 'IV'];
const RULES: Rule[] = [1, 2, 3, 4];

type Tile = { id: number; ch: string };
type Move = { id: number; ch: string; from: number | null; to: number; gone?: boolean };
type Anim = { moves: Move[]; c0: number; c1: number; at: number };
type Line = { s: string; rule: Rule | 0 };

const ease = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);

let nextId = 1;
const tilesOf = (s: string): Tile[] => [...s].map((ch) => ({ id: nextId++, ch }));

/** Apply rule r at its first match: the new tiles plus how every tile travels. */
function step(tiles: Tile[], r: Rule): { tiles: Tile[]; moves: Move[] } | null {
  const s = tiles.map((t) => t.ch).join('');
  const out = apply(s, r);
  if (!out || out.length > MAXLEN) return null;
  const a = matchAt(s, r);
  const L = tiles.length;
  const keep = (t: Tile, i: number, j = i): Move => ({ id: t.id, ch: t.ch, from: i, to: j });
  if (r === 1) {
    const u = { id: nextId++, ch: 'U' };
    return {
      tiles: [...tiles, u],
      moves: [...tiles.map((t, i) => keep(t, i)), { ...u, from: null, to: L }],
    };
  }
  if (r === 2) {
    const copies = tiles.slice(1).map((t) => ({ id: nextId++, ch: t.ch }));
    return {
      tiles: [...tiles, ...copies],
      moves: [
        ...tiles.map((t, i) => keep(t, i)),
        ...copies.map((c, k) => ({ ...c, from: k + 1, to: L + k })),
      ],
    };
  }
  if (r === 3) {
    const u = { id: nextId++, ch: 'U' };
    const moves: Move[] = [];
    tiles.forEach((t, i) => {
      if (i < a) moves.push(keep(t, i));
      else if (i < a + 3) moves.push({ ...keep(t, i, a), gone: true });
      else moves.push(keep(t, i, i - 2));
    });
    moves.push({ ...u, from: null, to: a });
    return { tiles: [...tiles.slice(0, a), u, ...tiles.slice(a + 3)], moves };
  }
  const moves: Move[] = [];
  tiles.forEach((t, i) => {
    if (i < a) moves.push(keep(t, i));
    else if (i < a + 2) moves.push({ ...keep(t, i, a), gone: true });
    else moves.push(keep(t, i, i - 2));
  });
  return { tiles: [...tiles.slice(0, a), ...tiles.slice(a + 2)], moves };
}

/** Tiles a rule would touch, for the hover highlight. */
function touched(s: string, r: Rule): number[] {
  const a = matchAt(s, r);
  if (a < 0) return [];
  if (r === 1) return [a];
  if (r === 2) return Array.from({ length: s.length - 1 }, (_, i) => i + 1);
  return Array.from({ length: r === 3 ? 3 : 2 }, (_, i) => a + i);
}

function poster() {
  let tiles = tilesOf('MI');
  const lines: Line[] = [{ s: 'MI', rule: 0 }];
  for (const r of DEMO) {
    tiles = step(tiles, r)!.tiles;
    lines.push({ s: tiles.map((t) => t.ch).join(''), rule: r });
  }
  return { tiles, lines };
}

export default function MiuPlayground() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(500);
  const W = compact ? 360 : 520;
  const X0 = compact ? 14 : 20;
  const TW = compact ? 15.5 : 23; // tile pitch
  const TH = compact ? 22 : 28;
  const TILE_Y = 30;
  const NL = 116; // number line baseline
  const LIST_Y = 168;
  const LH = 17;
  const ROWS = 4;
  const H = LIST_Y + ROWS * LH + 4;
  const [svgRef, k] = useBoost(W, 10, 9);

  const [tiles, setTiles] = useState<Tile[]>(() => tilesOf('MI'));
  const [lines, setLines] = useState<Line[]>([{ s: 'MI', rule: 0 }]);
  const [anim, setAnim] = useState<Anim | null>(null);
  const [clock, setClock] = useState(0);
  const clk = useRef(0);
  const [demo, setDemo] = useState<{ i: number; next: number } | null>(null);
  const [hover, setHover] = useState<Rule | null>(null);
  const started = useRef(false);
  const tilesRef = useRef(tiles);
  tilesRef.current = tiles;

  const s = tiles.map((t) => t.ch).join('');
  const count = iCount(s);

  const fire = (r: Rule) => {
    const res = step(tilesRef.current, r);
    if (!res) return;
    const c0 = iCount(tilesRef.current.map((t) => t.ch).join(''));
    const ns = res.tiles.map((t) => t.ch).join('');
    tilesRef.current = res.tiles;
    setTiles(res.tiles);
    setLines((ls) => [...ls, { s: ns, rule: r }]);
    setAnim(reduced ? null : { moves: res.moves, c0, c1: iCount(ns), at: clk.current });
  };

  const startDemo = (delay: number) => {
    if (reduced) {
      const p = poster();
      tilesRef.current = p.tiles;
      setTiles(p.tiles);
      setLines(p.lines);
      setAnim(null);
      setDemo(null);
      return;
    }
    setDemo({ i: 0, next: clk.current + delay });
  };

  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    startDemo(0.6);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, reduced]);
  useEffect(() => {
    if (reduced && started.current) startDemo(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced]);

  const animating = anim !== null && clock - anim.at < DUR;
  useFrame(
    (dt) => {
      clk.current += dt;
      const now = clk.current;
      if (demo && now >= demo.next) {
        fire(DEMO[demo.i]);
        setDemo(demo.i + 1 < DEMO.length ? { i: demo.i + 1, next: now + DEMO_GAP } : null);
      }
      setClock(now);
      if (anim && now - anim.at >= DUR && !demo) setAnim(null);
    },
    inView && (demo !== null || anim !== null),
  );

  const restart = () => {
    setHover(null);
    const t0 = tilesOf('MI');
    tilesRef.current = t0;
    setTiles(t0);
    setLines([{ s: 'MI', rule: 0 }]);
    setAnim(null);
    startDemo(1.0);
  };

  const userFire = (r: Rule) => {
    setDemo(null);
    fire(r);
  };

  // progress of the current slide
  const e = anim ? ease(clamp((clock - anim.at) / DUR, 0, 1)) : 1;
  const tx = (i: number) => X0 + i * TW;
  const nx = (n: number) => X0 + 6 + (n * (W - 2 * X0 - 12)) / MAXI;

  const moves: Move[] =
    anim && e < 1 ? anim.moves : tiles.map((t, i) => ({ id: t.id, ch: t.ch, from: i, to: i }));

  // the marker on the number line: a hop from the old count to the new one
  const c0 = anim && e < 1 ? anim.c0 : count;
  const c1 = anim && e < 1 ? anim.c1 : count;
  const mx = lerp(nx(c0), nx(c1), e);
  const hop = Math.min(34, Math.abs(nx(c1) - nx(c0)) * 0.35);
  const my = NL - 10 - Math.sin(Math.PI * e) * hop;
  const countNow = e < 0.5 ? c0 : c1; // whole number, changes once mid-hop

  // hover preview of a rule: highlight its tiles, ghost its result on the line
  const pv = hover && !animating ? apply(s, hover, matchAt(s, hover)) : null;
  const pvOk = pv !== null && pv.length <= MAXLEN;
  const hiSet = new Set(hover && pvOk ? touched(s, hover) : []);

  // derivation list, scrolled so the newest line is the last row
  const total = lines.length;
  const off1 = Math.max(0, total - ROWS);
  const off0 = anim && e < 1 ? Math.max(0, total - 1 - ROWS) : off1;
  const off = lerp(off0, off1, e);

  const letterColor = (ch: string) => (ch === 'I' ? GT.a : ch === 'U' ? GT.alt : GT.ink);
  const fsTile = (compact ? 12 : 14) * k;
  const fs = (compact ? 9.5 : 10) * k;

  const capIdx = hover ? hover : demo ? 5 : 0;
  const captions = [
    'Rules I and IV leave the I-count alone. Rule II doubles it and rule III takes away 3. From 1, neither can ever land on a multiple of 3, and MU needs 0.',
    ...RULES.map((r) => `${RULE_TEXT[r]} (Hofstadter)`),
    'Start from MI and apply the rules. Each new string goes on the list; the dot tracks how many I’s it has.',
  ];

  const blocked = (r: Rule) => {
    const out = apply(s, r);
    if (!out) return 'does not apply to this string';
    if (out.length > MAXLEN) return 'too long to draw here';
    return null;
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-10.1"
      title="the MIU playground"
      kicker="Hofstadter's MU puzzle"
      hint="tap a rule (III and IV act on the first match)"
      footer="checked: all 15,885 strings within 8 steps"
    >
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label={`The current string ${s}, with ${count} letters I. A number line of I-counts marks every multiple of 3; the count never lands on one, and MU needs 0.`}
        >
          <T x={X0} y={10} anchor="start" size={(compact ? 9 : 9.5) * k}>
            {'your string'}
          </T>
          {/* tiles */}
          {moves.map((m) => {
            const fromX = m.from === null ? tx(m.to) : tx(m.from);
            const x = lerp(fromX, tx(m.to), e);
            const op = m.gone ? 1 - e : m.from === null ? e : 1;
            const i = m.from ?? m.to;
            const hi = hiSet.has(i) && !m.gone && m.from === m.to;
            return (
              <g key={m.id} opacity={op}>
                <rect
                  x={x + 1}
                  y={TILE_Y}
                  width={TW - 2}
                  height={TH}
                  style={{
                    fill: GT.paper,
                    stroke: hi ? GT.eq : GT.lineStrong,
                    strokeWidth: hi ? 2 : 1,
                  }}
                />
                <T
                  x={x + TW / 2}
                  y={TILE_Y + TH / 2 + 1}
                  size={fsTile}
                  color={letterColor(m.ch)}
                  weight={700}
                >
                  {m.ch}
                </T>
              </g>
            );
          })}

          {/* number line of I-counts */}
          <T x={X0} y={NL - 44} anchor="start" size={(compact ? 9 : 9.5) * k}>
            {'how many I’s'}
          </T>
          <line
            x1={nx(0)}
            x2={nx(MAXI)}
            y1={NL}
            y2={NL}
            style={{ stroke: GT.lineStrong, strokeWidth: 1 }}
          />
          {Array.from({ length: MAXI + 1 }, (_, n) => {
            const bad = n % 3 === 0;
            const x = nx(n);
            return (
              <g key={n}>
                {bad ? (
                  <rect
                    x={x - 5}
                    y={NL - 5}
                    width={10}
                    height={10}
                    style={{ fill: GT.bSoft, stroke: GT.b, strokeWidth: 1.2 }}
                  />
                ) : (
                  <line x1={x} x2={x} y1={NL - 4} y2={NL + 4} style={{ stroke: GT.lineStrong }} />
                )}
                {(!compact || n % 3 === 0 || n === countNow) && (
                  <T
                    x={x}
                    y={NL + 15}
                    size={fs}
                    color={bad ? GT.b : n === countNow ? GT.a : GT.muted}
                    weight={n === countNow ? 700 : 400}
                  >
                    {n}
                  </T>
                )}
              </g>
            );
          })}
          <T x={nx(0) - 4} y={NL + 30} anchor="start" size={fs} color={GT.b} weight={700}>
            {compact ? '◼ multiple of 3 · MU needs 0' : '◼ multiples of 3 · MU needs 0'}
          </T>
          {/* ghost of the hovered rule's result */}
          {pvOk && pv && (
            <circle
              cx={nx(iCount(pv))}
              cy={NL - 10}
              r={5.5}
              style={{ fill: 'none', stroke: GT.a, strokeWidth: 1.5, strokeDasharray: '2 2' }}
            />
          )}
          <circle cx={mx} cy={my} r={6} style={{ fill: GT.a }} />
          <T x={mx} y={my - 13} size={fs} color={GT.a} weight={700}>
            {countNow}
          </T>

          {/* derivation list */}
          <line x1={X0} x2={W - X0} y1={LIST_Y - 12} y2={LIST_Y - 12} style={{ stroke: GT.line }} />
          <clipPath id="miu-list">
            <rect x={0} y={LIST_Y - 10} width={W} height={ROWS * LH + 2} />
          </clipPath>
          <g clipPath="url(#miu-list)">
            {lines.map((ln, i) => {
              const y = LIST_Y + (i - off) * LH + 3;
              if (y < LIST_Y - 2 * LH || y > LIST_Y + (ROWS + 1) * LH) return null;
              const newest = i === total - 1;
              const op = newest && anim && e < 1 ? e : 1;
              return (
                <g key={i} opacity={op}>
                  <T x={X0} y={y} anchor="start" size={fs} color={GT.muted}>
                    {String(i + 1).padStart(2, ' ')}
                  </T>
                  <T
                    x={X0 + 24}
                    y={y}
                    anchor="start"
                    size={fs}
                    color={newest ? GT.ink : GT.inkSoft}
                    weight={newest ? 700 : 400}
                  >
                    {ln.s}
                  </T>
                  <T x={W - X0} y={y} anchor="end" size={fs} color={GT.muted}>
                    {ln.rule ? `rule ${ROMAN[ln.rule]}` : 'axiom'}
                  </T>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {RULES.map((r) => {
          const why = blocked(r);
          return (
            <button
              key={r}
              type="button"
              disabled={why !== null}
              title={why ? `${RULE_TEXT[r]} (${why})` : RULE_TEXT[r]}
              onClick={() => userFire(r)}
              onPointerEnter={(ev) => {
                if (ev.pointerType !== 'touch') setHover(r);
              }}
              onPointerLeave={() => setHover(null)}
              className="border border-line-strong px-2.5 py-1 font-mono text-[0.72rem] text-ink-soft transition-colors hover:border-ink hover:text-ink disabled:opacity-40"
            >
              <span className="text-accent">{ROMAN[r]}</span> {RULE_SHORT[r]}
            </button>
          );
        })}
        <span className="ml-auto">
          <button
            type="button"
            onClick={restart}
            className="border border-ink bg-ink px-3 py-1 font-mono text-[0.72rem] text-paper transition-colors hover:border-accent hover:bg-accent"
          >
            restart
          </button>
        </span>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-4">
        <BigStat label="I-count" value={countNow} size="sm" color={GT.a} />
        <BigStat
          label="remainder ÷ 3"
          value={countNow % 3}
          size="sm"
          color={GT.a}
          sub="always 1 or 2"
        />
        <BigStat label="MU needs" value={0} size="sm" color={GT.b} sub="remainder 0" />
      </div>
      <Reserve
        items={captions.map((c, i) => (
          <Caption key={i}>{c}</Caption>
        ))}
        active={capIdx}
      />
    </Specimen>
  );
}
