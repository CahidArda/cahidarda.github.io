import { useEffect, useRef, useState } from 'react';
import {
  Btn,
  Caption,
  GT,
  Seg,
  Specimen,
  T,
  clamp,
  lerp,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useBoost } from '../../gt/braess/useBoost';
import { useCompact } from '../../gt/useCompact';

/**
 * CX-10.2 · one diagonal, three proofs. The table and motion of CX-09.1 (the contrarian),
 * relabelled three ways: Cantor's sequences (1891), Turing's programs (1936), Gödel's formulas
 * (1931); the tour ends on Gödel. The same cells are read down the diagonal and flipped into a new row; then the new
 * row is asked about itself. Every label is a pure function of (proof, t), so a hover preview
 * swaps every word at once and the geometry never moves.
 */

type Proof = 'cantor' | 'godel' | 'turing';

const V = [
  [1, 1, 1, 1, 1],
  [0, 0, 0, 0, 0],
  [1, 0, 1, 0, 1],
  [0, 1, 1, 0, 0],
  [1, 0, 0, 1, 0],
];
const VD = [1, 0, 0, 1, 1];
const N = 5;
const DOTS = 5;
const DROW = 6;

type Card = { a: string; b: string; v: string; ok: boolean };
type Spec = {
  label: string;
  head: [string, string];
  col: (c: number) => string;
  row: (r: number) => string;
  newName: string;
  special: string; // header of the new row's own column
  specialCells: boolean; // does that column hold verdicts?
  word: (h: number) => string; // table cells
  newWord: (h: number) => string; // the new row's cells
  cards: [Card, Card];
  cardsC: [Card, Card];
  caps: [string, string, string, string];
  footer: string;
};

const SPECS: Record<Proof, Spec> = {
  cantor: {
    label: 'Cantor 1891',
    head: ['rows: sequences of 0s and 1s · columns: digits', 'rows: sequences · cols: digits'],
    col: (c) => `#${c + 1}`,
    row: (r) => `s${r + 1}`,
    newName: 'new',
    special: 'listed?',
    specialCells: false,
    word: (h) => String(h),
    newWord: (h) => String(h),
    cards: [
      {
        a: 'is new = row k, for some k?',
        b: 'no: they differ at digit k',
        v: 'not in the list',
        ok: false,
      },
      {
        a: 'this works for any list',
        b: 'flip its diagonal, get a new row',
        v: 'no list has them all',
        ok: true,
      },
    ],
    cardsC: [
      { a: 'is new = row k?', b: 'differs at digit k', v: 'not listed', ok: false },
      { a: 'any list at all', b: 'misses its flip', v: 'none has all', ok: true },
    ],
    caps: [
      'Cantor, 1891: list infinite sequences of 0s and 1s, one per row. Suppose the list holds every such sequence.',
      'Read down the diagonal (digit k of row k) and flip each digit. That builds a new sequence.',
      'Is the new sequence anywhere in the list?',
      'It differs from row k at digit k, for every k. So it is missing, and no list can hold every sequence.',
    ],
    footer: 'Cantor, 1891',
  },
  godel: {
    label: 'Gödel 1931',
    head: [
      'rows: formulas F(x) · columns: code numbers · cell: is F(code) provable?',
      'cell: is F(code) provable?',
    ],
    col: (c) => `#F${c + 1}`,
    row: (r) => `F${r + 1}`,
    newName: 'G',
    special: '#G',
    specialCells: true,
    word: (h) => (h ? 'prov' : 'not'),
    newWord: (h) => (h ? 'true' : 'false'),
    cards: [
      {
        a: 'if G(#G) is provable',
        b: 'it says it is not: contradiction',
        v: 'system inconsistent',
        ok: false,
      },
      {
        a: 'if G(#G) is not provable',
        b: 'that is what it says: it is true',
        v: 'true but unprovable',
        ok: true,
      },
    ],
    cardsC: [
      { a: 'if G(#G) provable', b: 'it says it isn’t', v: 'contradiction', ok: false },
      { a: 'if not provable', b: 'what it says holds', v: 'true, unprovable', ok: true },
    ],
    caps: [
      'Gödel, 1931: rows are formulas F(x) about a number x, columns are the code numbers of formulas. Each cell says whether F(code) can be proved.',
      'Build G(x): “formula number x, fed its own number, is not provable”. Its row is the flipped diagonal: true exactly where the diagonal says not.',
      'G is a formula too, with a code #G. What does G(#G) say?',
      'G(#G) says “G(#G) is not provable”. A consistent system cannot prove it, and then what it says is true.',
    ],
    footer: 'Gödel, 1931',
  },
  turing: {
    label: 'Turing 1936',
    head: ['rows: programs · columns: inputs (program code)', 'rows: programs · cols: code'],
    col: (c) => `P${c + 1}`,
    row: (r) => `P${r + 1}`,
    newName: 'D',
    special: 'D',
    specialCells: true,
    word: (h) => (h ? 'halt' : 'loop'),
    newWord: (h) => (h ? 'halt' : 'loop'),
    cards: [
      {
        a: 'if H says D on D will halt',
        b: 'D does the opposite: loops',
        v: 'so H was wrong',
        ok: false,
      },
      {
        a: 'if H says D on D will loop',
        b: 'D does the opposite: halts',
        v: 'so H was wrong',
        ok: false,
      },
    ],
    cardsC: [
      { a: 'H says D on D: halt', b: 'so D will loop', v: 'H wrong', ok: false },
      { a: 'H says D on D: loop', b: 'so D will halt', v: 'H wrong', ok: false },
    ],
    caps: [
      'Turing, 1936: suppose a program H predicts, for any program and input, whether it halts. Its verdicts fill the table.',
      'Build D: given a program’s code, D asks H about that program run on its own code, then does the opposite.',
      'D is a program too, so it has a column. What happens when D runs on D?',
      'Either verdict makes H wrong, so no perfect H exists.',
    ],
    footer: 'Turing, 1936',
  },
};
const ORDER: Proof[] = ['cantor', 'turing', 'godel'];

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
    { x0: LX, x1: mid - 5 },
    { x0: mid + 5, x1: W - LX },
  ];
  return { W, LX, LW, CW, CH, TOP, GAP, colX, rowY, DDX, DDB, ELBOW, CARD_T, CARD_H, H, CARDS };
}

const DROP0 = 0.8;
const DROP_GAP = 0.3;
const DROP_DUR = 0.8;
const Q_AT = DROP0 + DROP_GAP * (N - 1) + DROP_DUR + 0.1;
const LINK_AT = Q_AT + 0.4;
const CARD_AT = LINK_AT + 0.55;
const WRONG_AT = CARD_AT + 0.5;
const END = WRONG_AT + 0.4;
const HOLD = 1.6; // seconds on the finished table before the demo moves to the next proof

const ease = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
const ramp = (t: number, at: number, dur: number) => clamp((t - at) / dur, 0, 1);
const col = (h: number) => (h ? GT.opt : GT.b);
const soft = (h: number) => (h ? GT.optSoft : GT.bSoft);

export default function OneDiagonal() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(500);
  const { W, LX, LW, CW, CH, TOP, GAP, colX, rowY, DDX, DDB, ELBOW, CARD_T, CARD_H, H, CARDS } =
    layout(compact);
  const [svgRef, k] = useBoost(W, 10, 9);
  const [sel, setSel] = useState<Proof>('cantor');
  const [proof, bindRaw] = useHoverPreview(sel);
  const [t, setT] = useState(0);
  const tRef = useRef(0);
  const [playing, setPlaying] = useState(false);
  const auto = useRef(true); // the demo walks through all three until the reader takes over
  const started = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const finish = () => {
    if (timer.current) clearTimeout(timer.current);
    setPlaying(false);
    tRef.current = END;
    setT(END);
  };

  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    if (reduced) {
      auto.current = false;
      setSel('godel');
      finish();
      return;
    }
    timer.current = setTimeout(() => setPlaying(true), 500);
  }, [inView, reduced]);
  useEffect(() => {
    if (!reduced) return;
    auto.current = false;
    finish();
  }, [reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  useFrame((dt) => {
    const n = Math.min(END, tRef.current + Math.max(0, dt));
    tRef.current = n;
    setT(n);
    if (n >= END) {
      setPlaying(false);
      const i = ORDER.indexOf(sel);
      if (auto.current && i < ORDER.length - 1) {
        timer.current = setTimeout(() => {
          if (!auto.current) return;
          setSel(ORDER[i + 1]);
          tRef.current = 0;
          setT(0);
          setPlaying(true);
        }, HOLD * 1000);
      } else auto.current = false;
    }
  }, playing && inView);

  // the reader takes over: stop the tour, show the chosen proof finished
  const takeOver = () => {
    if (!auto.current) return;
    auto.current = false;
    finish();
  };
  const bind = (v: Proof) => {
    const b = bindRaw(v);
    return {
      ...b,
      onPointerEnter: (e: React.PointerEvent<Element>) => {
        if (e.pointerType !== 'touch') takeOver();
        b.onPointerEnter(e);
      },
    };
  };

  const replay = () => {
    auto.current = false;
    if (timer.current) clearTimeout(timer.current);
    setPlaying(false);
    if (reduced) return finish();
    tRef.current = 0;
    setT(0);
    timer.current = setTimeout(() => setPlaying(true), 1000);
  };

  const S = SPECS[proof];
  const cards = compact ? S.cardsC : S.cards;
  const phase = t < DROP0 ? 0 : t < Q_AT ? 1 : t < CARD_AT ? 2 : 3;
  const capIdx = ORDER.indexOf(proof) * 4 + phase;

  const fs = (compact ? 11.5 : 10) * k;
  const cell = (
    key: string,
    x: number,
    y: number,
    h: number,
    w: (h: number) => string,
    opts: { stroke?: string; sw?: number } = {},
  ) => (
    <g key={key}>
      <rect
        x={x + 2}
        y={y + 2}
        width={CW - 4}
        height={CH - 4}
        style={{ fill: soft(h), stroke: opts.stroke ?? 'none', strokeWidth: opts.sw ?? 0 }}
      />
      <T x={x + CW / 2} y={y + CH / 2 + 0.5} size={fs} color={col(h)} weight={600}>
        {w(h)}
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
      code="CX-10.2"
      title="one diagonal, three proofs"
      kicker="the diagonal trick"
      hint="hover or tap a proof; replay to watch it built"
      footer={S.footer}
    >
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label={`${S.label}: a table whose diagonal is flipped into a new row ${S.newName}, which is then asked about itself`}
        >
          <T x={LX} y={12} anchor="start" size={(compact ? 10.5 : 9.5) * k}>
            {S.head[compact ? 1 : 0]}
          </T>
          {Array.from({ length: 7 }, (_, c) => (
            <T
              key={c}
              x={colX(c) + CW / 2}
              y={TOP - 12}
              size={c === DROW && S.special.length > 3 ? fs * 0.85 : fs}
              color={c === DROW ? GT.a : GT.ink}
              weight={c === DROW ? 700 : 400}
            >
              {c === DOTS ? '…' : c === DROW ? S.special : S.col(c)}
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
              {r === DOTS ? '⋮' : r === DROW ? S.newName : S.row(r)}
            </T>
          ))}

          {V.map((row, r) => (
            <g key={r}>
              {row.map((h, c) => cell(`${r}-${c}`, colX(c), rowY(r), h, S.word))}
              <T x={colX(DOTS) + CW / 2} y={rowY(r) + CH / 2} size={fs}>
                …
              </T>
              {S.specialCells && cell(`${r}-d`, colX(DROW), rowY(r), VD[r], S.word)}
            </g>
          ))}
          {V.map((_, i) => (
            <rect
              key={`dm${i}`}
              x={colX(i) + 2}
              y={rowY(i) + 2}
              width={CW - 4}
              height={CH - 4}
              opacity={ramp(t, 0.12 * i, 0.3)}
              style={{ fill: 'none', stroke: GT.a, strokeWidth: 2 }}
            />
          ))}
          {Array.from({ length: 7 }, (_, c) =>
            c === DROW && !S.specialCells ? null : (
              <T key={c} x={colX(c) + CW / 2} y={rowY(DOTS) + CH / 2} size={fs}>
                {c === DOTS ? '⋱' : '⋮'}
              </T>
            ),
          )}

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
              <g key={`D${i}`}>
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
                {landed && cell(`Dc${i}`, colX(i), rowY(DROW), h, S.newWord)}
              </g>
            );
          })}
          <T x={colX(DOTS) + CW / 2} y={rowY(DROW) + CH / 2} size={fs}>
            …
          </T>
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

          {/* the flipped diagonal falling into the new row */}
          {V.map((row, i) => {
            const u = ramp(t, DROP0 + DROP_GAP * i, DROP_DUR);
            if (u <= 0 || u >= 1) return null;
            const e = ease(u);
            const y = lerp(rowY(i), rowY(DROW), e);
            const s = Math.abs(Math.cos(Math.PI * e));
            const first = e < 0.5;
            const h = first ? row[i] : 1 - row[i];
            const cy = y + CH / 2;
            return (
              <g
                key={`g${i}`}
                transform={`translate(0 ${cy}) scale(1 ${Math.max(0.02, s)}) translate(0 ${-cy})`}
              >
                <rect x={colX(i)} y={y} width={CW} height={CH} style={{ fill: GT.paperRaised }} />
                {cell(`gc${i}`, colX(i), y, h, first ? S.word : S.newWord, {
                  stroke: GT.a,
                  sw: 2,
                })}
              </g>
            );
          })}

          {CARDS.map((cd, j) => {
            const cx = (cd.x0 + cd.x1) / 2;
            const p = ramp(t, LINK_AT, 0.55);
            const a = ramp(t, CARD_AT, 0.4);
            const w = ramp(t, WRONG_AT, 0.4);
            const c = cards[j];
            const vc = c.ok ? GT.opt : GT.eq;
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
                    {c.a}
                  </T>
                  <T x={cd.x0 + 12} y={CARD_T + 40} anchor="start" size={fs} color={GT.ink}>
                    {c.b}
                  </T>
                </g>
                <g opacity={w} style={halo}>
                  <T
                    x={cd.x0 + 12}
                    y={CARD_T + 63}
                    anchor="start"
                    size={fs * 1.1}
                    color={vc}
                    weight={700}
                  >
                    {c.v}
                  </T>
                  {c.ok ? (
                    <path
                      d={`M${cd.x1 - 32} ${CARD_T + 62} l6 7 l12 -14`}
                      style={{ fill: 'none', stroke: vc, strokeWidth: 2.5 }}
                    />
                  ) : (
                    <path
                      d={`M${cd.x1 - 30} ${CARD_T + 55} l14 14 M${cd.x1 - 16} ${CARD_T + 55} l-14 14`}
                      style={{ fill: 'none', stroke: vc, strokeWidth: 2.5 }}
                    />
                  )}
                </g>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <Seg
          options={ORDER.map((p) => ({ value: p, label: SPECS[p].label }))}
          value={proof}
          onChange={(v) => {
            takeOver();
            setSel(v);
          }}
          bind={bind}
        />
        <Btn primary onClick={replay}>
          replay
        </Btn>
      </div>
      <Reserve
        items={ORDER.flatMap((p) => SPECS[p].caps).map((c, i) => (
          <Caption key={i}>{c}</Caption>
        ))}
        active={capIdx}
      />
    </Specimen>
  );
}
