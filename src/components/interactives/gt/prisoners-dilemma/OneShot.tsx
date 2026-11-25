import { useEffect, useState } from 'react';
import {
  BigStat,
  Caption,
  GT,
  Seg,
  Specimen,
  SvgArrows,
  T,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../shared';
import { Reserve } from '../Reserve';
import { PAY, payoff, type Move } from './pd';

/* one SVG coordinate system: the 2x2 matrix, your improvement arrow in each column */
const W = 440;
const H = 290;
const CW = 136;
const CH = 84;
const X0 = 112;
const GX = 36;
const Y0 = 52;
const GY = 50;
const cx = (c: number) => X0 + c * (CW + GX) + CW / 2;
const cy = (r: number) => Y0 + r * (CH + GY) + CH / 2;
const MOVES: Move[] = ['C', 'D'];
const NAME: Record<Move, string> = { C: 'Cooperate', D: 'Defect' };
/** Which payoff letter each cell is, from your side: [you][them]. */
const LETTER: Record<string, string> = { CC: 'R', CD: 'S', DC: 'T', DD: 'P' };

export default function OneShot() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [sel, setSel] = useState<Move>('C');
  const [them, bind] = useHoverPreview(sel);
  const [auto, setAuto] = useState(true);

  // Auto-play: flip what they do, so both columns get their turn.
  useEffect(() => {
    if (!auto || reduced || !inView) return;
    const id = window.setInterval(() => setSel((m) => (m === 'C' ? 'D' : 'C')), 2400);
    return () => window.clearInterval(id);
  }, [auto, reduced, inView]);

  const pick = (m: Move) => {
    setAuto(false);
    setSel(m);
  };
  const col = them === 'C' ? 0 : 1;
  const ifC = payoff('C', them);
  const ifD = payoff('D', them);

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-02.1"
      title="one round"
      kicker={`T=${PAY.T} R=${PAY.R} P=${PAY.P} S=${PAY.S}`}
      hint="pick what they do: your best reply never changes"
    >
      <div className="grid gap-x-6 gap-y-4 sm:grid-cols-[minmax(0,34rem)_minmax(0,1fr)] sm:items-center">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full"
          role="img"
          aria-label="Prisoner's Dilemma payoff matrix: defecting pays you more in both columns"
        >
          <SvgArrows />
          {/* the column they chose */}
          <rect
            x={cx(col) - CW / 2 - 8}
            y={Y0 - 34}
            width={CW + 16}
            height={2 * CH + GY + 42}
            style={{ fill: GT.bSoft, transition: 'x 300ms ease' }}
          />
          {/* how to read a cell */}
          <T x={X0 - 14} y={Y0 - 26} anchor="end" size={10} color={GT.a}>
            you: big
          </T>
          <T x={X0 - 14} y={Y0 - 12} anchor="end" size={10} color={GT.b}>
            them: small
          </T>
          {MOVES.map((m, c) => (
            <g
              key={m}
              onClick={() => pick(m)}
              style={{ cursor: 'pointer' }}
              {...bind(m)}
            >
              <rect
                x={cx(c) - CW / 2}
                y={Y0 - 30}
                width={CW}
                height={24}
                style={{ fill: 'transparent' }}
              />
              <T x={cx(c)} y={Y0 - 18} size={12} color={GT.b} weight={c === col ? 600 : 400}>
                {`they ${NAME[m].toLowerCase()}`}
              </T>
            </g>
          ))}
          {MOVES.map((m, r) => (
            <T key={m} x={X0 - 14} y={cy(r)} anchor="end" size={12} color={GT.a}>
              {`you ${m === 'C' ? 'coop.' : 'defect'}`}
            </T>
          ))}
          {MOVES.map((me, r) =>
            MOVES.map((th, c) => {
              const mine = payoff(me, th);
              const theirs = payoff(th, me);
              const nash = me === 'D' && th === 'D';
              const best = me === 'C' && th === 'C';
              const lit = c === col;
              // the other column is dimmed by colour, not opacity, so it stays legible in dark mode
              const ca = lit ? GT.a : GT.inkSoft;
              const cb = lit ? GT.b : GT.muted;
              return (
                <g key={me + th}>
                  <rect
                    x={cx(c) - CW / 2}
                    y={cy(r) - CH / 2}
                    width={CW}
                    height={CH}
                    style={{
                      fill: GT.paperRaised,
                      stroke: nash ? GT.eq : best ? GT.opt : GT.lineStrong,
                      strokeWidth: nash || best ? 2.5 : 1,
                      strokeDasharray: best ? '6 4' : undefined,
                    }}
                  />
                  <T x={cx(c) - CW / 2 + 8} y={cy(r) - CH / 2 + 11} anchor="start" size={10}>
                    {LETTER[me + th]}
                  </T>
                  <T x={cx(c) - 4} y={cy(r) - 4} anchor="end" font="display" size={38} color={ca}>
                    {mine}
                  </T>
                  <T x={cx(c) + 6} y={cy(r) + 4} anchor="start" font="display" size={20} color={cb}>
                    {theirs}
                  </T>
                  {(nash || best) && (
                    <T
                      x={cx(c)}
                      y={cy(r) + CH / 2 - 12}
                      size={10}
                      color={nash ? GT.eq : GT.opt}
                      weight={600}
                    >
                      {nash ? 'you both land here' : 'better for both'}
                    </T>
                  )}
                </g>
              );
            }),
          )}
          {/* your improvement arrow in each column: Cooperate -> Defect */}
          {[0, 1].map((c) => {
            const gain = payoff('D', MOVES[c]) - payoff('C', MOVES[c]);
            const lit = c === col;
            return (
              <g key={'a' + c} style={{ opacity: lit ? 1 : 0.55, transition: 'opacity 300ms' }}>
                <line
                  x1={cx(c)}
                  y1={Y0 + CH + 6}
                  x2={cx(c)}
                  y2={Y0 + CH + GY - 6}
                  markerEnd="url(#gt-arrow-a)"
                  style={{ stroke: GT.a, strokeWidth: 3 }}
                />
                <T
                  x={cx(c) + 14}
                  y={Y0 + CH + GY / 2}
                  anchor="start"
                  size={13}
                  color={GT.a}
                  weight={600}
                >
                  +{gain}
                </T>
              </g>
            );
          })}
        </svg>

        <div>
          <BigStat label="defecting gains you" value={`+${ifD - ifC}`} color={GT.a} size="lg" />
          <Reserve
            className="mt-1 font-mono text-[0.68rem] leading-snug text-muted"
            items={MOVES.map((m) => (
              <span
                key={m}
              >{`${payoff('D', m)} instead of ${payoff('C', m)} when they ${NAME[m].toLowerCase()}`}</span>
            ))}
            active={col}
          />
          <div className="mt-4">
            <Seg
              label="they"
              options={MOVES.map((m) => ({ value: m, label: NAME[m] }))}
              value={them}
              onChange={pick}
              bind={bind}
            />
          </div>
        </div>
      </div>
      <Caption>
        Defect is your better reply in both columns. They reason the same way, so you both get{' '}
        {PAY.P} instead of {PAY.R}.
      </Caption>
    </Specimen>
  );
}
