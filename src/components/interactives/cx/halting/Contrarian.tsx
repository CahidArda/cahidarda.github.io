import { useEffect, useRef, useState } from 'react';
import {
  Btn,
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

/**
 * Turing's diagonal argument as a table. Rows are programs, columns are inputs (each input
 * is some program's code). A hypothetical predictor H fills every cell with "halt" or
 * "loop". The contrarian D reads the diagonal (each program run on its own code) and does
 * the opposite, so D's row is the flipped diagonal. Then the cell "D run on D" has no
 * consistent value: either verdict makes H wrong.
 *
 * One clock t (seconds) drives everything; the drawing is a pure function of t.
 */

// H's verdicts: 1 = halts, 0 = loops. Columns 0..4 are P1..P5's code, DCOL is D's code.
const V = [
  [1, 1, 1, 1, 1],
  [0, 0, 0, 0, 0],
  [1, 0, 1, 0, 1],
  [0, 1, 1, 0, 0],
  [1, 0, 0, 1, 0],
];
const VD = [1, 0, 0, 1, 1]; // H on (Pi, D's code)
const N = 5;
const DOTS = 5; // index of the "…" row / column
const DROW = 6;

function layout(compact: boolean) {
  const W = compact ? 370 : 520;
  const LX = compact ? 4 : 20;
  const LW = compact ? 30 : 60;
  const CW = compact ? 48 : 60;
  const CH = compact ? 26 : 28;
  const TOP = 44;
  const GAP = 8;
  const colX = (c: number) => LX + LW + c * CW;
  const rowY = (r: number) => TOP + r * CH + (r === DROW ? GAP : 0);
  const DDX = colX(DROW) + CW / 2;
  const DDB = rowY(DROW) + CH;
  const ELBOW = DDB + 16;
  const CARD_T = ELBOW + 14;
  const CARD_H = 82;
  const H = CARD_T + CARD_H + 10;
  const mid = W / 2;
  const CARDS = [
    { x0: LX, x1: mid - 5, says: 1 },
    { x0: mid + 5, x1: W - LX, says: 0 },
  ];
  return { W, LX, LW, CW, CH, TOP, GAP, colX, rowY, DDX, DDB, ELBOW, CARD_T, CARD_H, H, CARDS };
}

// timeline (seconds)
const DROP0 = 0.9;
const DROP_GAP = 0.35;
const DROP_DUR = 0.9;
const Q_AT = DROP0 + DROP_GAP * (N - 1) + DROP_DUR + 0.1; // 3.3
const LINK_AT = Q_AT + 0.45;
const CARD_AT = LINK_AT + 0.65;
const WRONG_AT = CARD_AT + 0.55;
const END = WRONG_AT + 0.45;

const ease = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
const ramp = (t: number, at: number, dur: number) => clamp((t - at) / dur, 0, 1);

const word = (h: number) => (h ? 'halt' : 'loop');
const col = (h: number) => (h ? GT.opt : GT.b);
const soft = (h: number) => (h ? GT.optSoft : GT.bSoft);

const CAPTIONS = [
  'Suppose a program H could predict, for any program and any input, whether it halts. Its verdicts fill this table. Inputs are programs too: any code can be fed to any program.',
  'Build D, the contrarian. Given a program’s code, D asks H what that program does when run on its own code (the diagonal), then does the opposite.',
  'D is a program, so it has a row and a column of its own. What does H say about D run on D?',
  'If H says “halts”, D loops. If H says “loops”, D halts. Either way H is wrong, so no perfect H can exist.',
  ...V.map(
    (r, i) =>
      `D run on P${i + 1}’s code: H says P${i + 1} run on itself ${r[i] ? 'halts' : 'loops'}, so D ${r[i] ? 'loops forever' : 'halts at once'}.`,
  ),
];

export default function Contrarian() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(500);
  const { W, LX, LW, CW, CH, TOP, GAP, colX, rowY, DDX, DDB, ELBOW, CARD_T, CARD_H, H, CARDS } =
    layout(compact);
  const [svgRef, k] = useBoost(W, 10, 9);
  const [t, setT] = useState(0);
  const tRef = useRef(0);
  const [playing, setPlaying] = useState(false);
  const started = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [hoverCell, setHoverCell] = useState<number | null>(null);

  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    if (reduced) {
      tRef.current = END;
      setT(END);
      return;
    }
    timer.current = setTimeout(() => setPlaying(true), 500);
  }, [inView, reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  // the reduced-motion setting can arrive after the first effect: jump to the poster
  useEffect(() => {
    if (!reduced) return;
    if (timer.current) clearTimeout(timer.current);
    setPlaying(false);
    tRef.current = END;
    setT(END);
  }, [reduced]);
  useFrame((dt) => {
    const n = Math.min(END, tRef.current + Math.max(0, dt));
    tRef.current = n;
    setT(n);
    if (n >= END) setPlaying(false);
  }, playing && inView);

  const replay = () => {
    if (timer.current) clearTimeout(timer.current);
    setPlaying(false);
    setHoverCell(null);
    if (reduced) {
      tRef.current = END;
      setT(END);
      return;
    }
    tRef.current = 0;
    setT(0);
    timer.current = setTimeout(() => setPlaying(true), 1000);
  };

  const done = t >= END;
  const phase = t < DROP0 ? 0 : t < Q_AT ? 1 : t < CARD_AT ? 2 : 3;
  const capIdx = done && hoverCell !== null ? 4 + hoverCell : phase;

  const fs = (compact ? 11.5 : 10) * k;
  const cell = (
    key: string,
    x: number,
    y: number,
    h: number,
    opts: { stroke?: string; sw?: number; opacity?: number } = {},
  ) => (
    <g key={key} opacity={opts.opacity ?? 1}>
      <rect
        x={x + 2}
        y={y + 2}
        width={CW - 4}
        height={CH - 4}
        style={{ fill: soft(h), stroke: opts.stroke ?? 'none', strokeWidth: opts.sw ?? 0 }}
      />
      <T x={x + CW / 2} y={y + CH / 2 + 0.5} size={fs} color={col(h)} weight={600}>
        {word(h)}
      </T>
    </g>
  );

  const halo = {
    stroke: GT.paperRaised,
    strokeWidth: 4,
    strokeLinejoin: 'round' as const,
    paintOrder: 'stroke' as const,
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-09.1"
      title="the contrarian"
      kicker="Turing's diagonal argument"
      hint="replay to watch D being built; hover D's row for each step"
      footer="Turing, 1936"
    >
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label="A table of predicted verdicts, the flipped diagonal forming the program D, and the two contradictory answers for D run on D"
        >
          {/* headers */}
          <T x={LX} y={12} anchor="start" size={(compact ? 10.5 : 9.5) * k}>
            {'rows: programs · columns: inputs (program code)'}
          </T>
          {Array.from({ length: 7 }, (_, c) => (
            <T
              key={c}
              x={colX(c) + CW / 2}
              y={TOP - 12}
              size={fs}
              color={c === DROW ? GT.a : GT.ink}
              weight={c === DROW ? 700 : 400}
            >
              {c === DOTS ? '…' : c === DROW ? 'D' : `P${c + 1}`}
            </T>
          ))}
          {Array.from({ length: 7 }, (_, r) => (
            <T
              key={r}
              x={LX + LW - 12}
              y={rowY(r) + CH / 2}
              anchor="end"
              size={fs}
              color={r === DROW ? GT.a : GT.ink}
              weight={r === DROW ? 700 : 400}
            >
              {r === DOTS ? '⋮' : r === DROW ? 'D' : `P${r + 1}`}
            </T>
          ))}

          {/* H's table */}
          {V.map((row, r) => (
            <g key={r}>
              {row.map((h, c) => cell(`${r}-${c}`, colX(c), rowY(r), h, {}))}
              <T x={colX(DOTS) + CW / 2} y={rowY(r) + CH / 2} size={fs}>
                …
              </T>
              {cell(`${r}-d`, colX(DROW), rowY(r), VD[r])}
            </g>
          ))}
          {/* the diagonal: each program run on its own code, outlined as D reads it */}
          {V.map((_, i) => (
            <rect
              key={`dm${i}`}
              x={colX(i) + 2}
              y={rowY(i) + 2}
              width={CW - 4}
              height={CH - 4}
              opacity={ramp(t, 0.15 * i, 0.3)}
              style={{ fill: 'none', stroke: GT.a, strokeWidth: 2 }}
            />
          ))}
          {/* the ⋮ row */}
          {Array.from({ length: 7 }, (_, c) => (
            <T key={c} x={colX(c) + CW / 2} y={rowY(DOTS) + CH / 2} size={fs}>
              {c === DOTS ? '⋱' : '⋮'}
            </T>
          ))}

          {/* D's row: empty slots, filled as the flipped diagonal lands */}
          <line
            x1={LX}
            x2={W - LX}
            y1={rowY(DROW) - GAP / 2}
            y2={rowY(DROW) - GAP / 2}
            style={{ stroke: GT.a, strokeWidth: 1 }}
          />
          {Array.from({ length: N }, (_, i) => {
            const landed = t >= DROP0 + DROP_GAP * i + DROP_DUR;
            const h = 1 - V[i][i];
            return (
              <g
                key={`D${i}`}
                onPointerEnter={() => setHoverCell(i)}
                onPointerLeave={() => setHoverCell(null)}
                style={{ cursor: done ? 'help' : 'default' }}
              >
                <rect
                  x={colX(i) + 2}
                  y={rowY(DROW) + 2}
                  width={CW - 4}
                  height={CH - 4}
                  style={{
                    fill: 'transparent',
                    stroke: landed ? 'none' : GT.lineStrong,
                    strokeDasharray: '3 3',
                  }}
                />
                {landed &&
                  cell(`Dc${i}`, colX(i), rowY(DROW), h, {
                    stroke: done && hoverCell === i ? GT.a : 'none',
                    sw: 2,
                  })}
              </g>
            );
          })}
          <T x={colX(DOTS) + CW / 2} y={rowY(DROW) + CH / 2} size={fs}>
            …
          </T>
          {/* D on D */}
          <rect
            x={colX(DROW) + 2}
            y={rowY(DROW) + 2}
            width={CW - 4}
            height={CH - 4}
            style={{ fill: 'transparent', stroke: GT.lineStrong, strokeDasharray: '3 3' }}
          />
          <g opacity={ramp(t, Q_AT, 0.4)}>
            <rect
              x={colX(DROW) + 2}
              y={rowY(DROW) + 2}
              width={CW - 4}
              height={CH - 4}
              style={{ fill: GT.paperRaised, stroke: GT.a, strokeWidth: 2 }}
            />
            <T
              x={colX(DROW) + CW / 2}
              y={rowY(DROW) + CH / 2 + 0.5}
              size={13 * k}
              color={GT.a}
              weight={700}
            >
              ?
            </T>
          </g>

          {/* the flipped diagonal falling into D's row (drawn on top: a card being carried) */}
          {V.map((row, i) => {
            const u = ramp(t, DROP0 + DROP_GAP * i, DROP_DUR);
            if (u <= 0 || u >= 1) return null;
            const e = ease(u);
            const y = lerp(rowY(i), rowY(DROW), e);
            const s = Math.abs(Math.cos(Math.PI * e));
            const h = e < 0.5 ? row[i] : 1 - row[i];
            const cy = y + CH / 2;
            return (
              <g
                key={`g${i}`}
                transform={`translate(0 ${cy}) scale(1 ${Math.max(0.02, s)}) translate(0 ${-cy})`}
              >
                <rect x={colX(i)} y={y} width={CW} height={CH} style={{ fill: GT.paperRaised }} />
                {cell(`gc${i}`, colX(i), y, h, { stroke: GT.a, sw: 2 })}
              </g>
            );
          })}

          {/* the two branches */}
          {CARDS.map((cd, j) => {
            const cx = (cd.x0 + cd.x1) / 2;
            const p = ramp(t, LINK_AT, 0.6);
            const a = ramp(t, CARD_AT, 0.4);
            const w = ramp(t, WRONG_AT, 0.4);
            const d = cd.says;
            return (
              <g key={j}>
                <path
                  d={`M${DDX} ${DDB} V${ELBOW} H${cx} V${CARD_T}`}
                  pathLength={1}
                  style={{
                    fill: 'none',
                    stroke: GT.a,
                    strokeWidth: 1.5,
                    strokeDasharray: 1,
                    strokeDashoffset: 1 - p,
                  }}
                />
                <g opacity={a}>
                  <rect
                    x={cd.x0}
                    y={CARD_T}
                    width={cd.x1 - cd.x0}
                    height={CARD_H}
                    style={{ fill: GT.paperRaised, stroke: GT.lineStrong }}
                  />
                  <T x={cd.x0 + 12} y={CARD_T + 18} anchor="start" size={fs} color={GT.ink}>
                    {compact ? 'H says D on D will' : 'if H says D on D will'}
                    <tspan style={{ fill: col(d), fontWeight: 700 }}>{` ${word(d)}`}</tspan>
                  </T>
                  <T x={cd.x0 + 12} y={CARD_T + 40} anchor="start" size={fs} color={GT.ink}>
                    {compact ? 'so D will' : 'D does the opposite:'}
                    <tspan style={{ fill: col(1 - d), fontWeight: 700 }}>
                      {compact ? ` ${word(1 - d)}` : ` ${word(1 - d)}s`}
                    </tspan>
                  </T>
                </g>
                <g opacity={w} style={halo}>
                  <T
                    x={cd.x0 + 12}
                    y={CARD_T + 63}
                    anchor="start"
                    size={fs * 1.1}
                    color={GT.eq}
                    weight={700}
                  >
                    so H was wrong
                  </T>
                  <path
                    d={`M${cd.x1 - 30} ${CARD_T + 55} l14 14 M${cd.x1 - 16} ${CARD_T + 55} l-14 14`}
                    style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2.5 }}
                  />
                </g>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-3 flex items-start justify-between gap-4">
        <Reserve
          className="min-w-0 flex-1"
          items={CAPTIONS.map((c, i) => (
            <Caption key={i}>{c}</Caption>
          ))}
          active={capIdx}
        />
        <div className="mt-3 shrink-0">
          <Btn primary onClick={replay}>
            replay
          </Btn>
        </div>
      </div>
    </Specimen>
  );
}
