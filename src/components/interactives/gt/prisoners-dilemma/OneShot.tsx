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
  useInView,
  useReducedMotion,
} from '../shared';
import { PAY, payoff, type Move } from './pd';

/* one SVG coordinate system */
const W = 440;
const H = 318;
const CW = 136;
const CH = 84;
const X0 = 104;
const GX = 30;
const Y0 = 56;
const GY = 48;
const cx = (c: number) => X0 + c * (CW + GX) + CW / 2;
const cy = (r: number) => Y0 + r * (CH + GY) + CH / 2;
const MOVES: Move[] = ['C', 'D'];
const NAME: Record<Move, string> = { C: 'Cooperate', D: 'Defect' };

// number line for T > R > P > S
const NX0 = 70;
const NX1 = 410;
const NY = 296;
const nx = (v: number) => NX0 + (v / 5) * (NX1 - NX0);

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
      hint="choose what the other prisoner does; your best reply never changes"
      footer="defect dominates"
    >
      <Seg
        label="they"
        options={MOVES.map((m) => ({ value: m, label: NAME[m] }))}
        value={them}
        onChange={pick}
        bind={bind}
      />
      <div className="mt-4 grid items-center gap-4 sm:grid-cols-[1fr_auto]">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full"
          role="img"
          aria-label="Prisoner's Dilemma payoff matrix"
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
          {MOVES.map((m, c) => (
            <g
              key={m}
              onClick={() => pick(m)}
              style={{ cursor: 'pointer' }}
              onMouseEnter={bind(m).onMouseEnter}
              onMouseLeave={bind(m).onMouseLeave}
            >
              <T x={cx(c)} y={Y0 - 18} size={12} color={GT.b} weight={c === col ? 600 : 400}>
                {`they ${NAME[m].toLowerCase()}`}
              </T>
            </g>
          ))}
          {MOVES.map((m, r) => (
            <T key={m} x={X0 - 12} y={cy(r)} anchor="end" size={12} color={GT.a}>
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
              return (
                <g key={me + th} style={{ opacity: lit ? 1 : 0.6, transition: 'opacity 300ms' }}>
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
                  <T x={cx(c) - 6} y={cy(r) - 2} anchor="end" font="display" size={36} color={GT.a}>
                    {mine}
                  </T>
                  <T
                    x={cx(c) + 4}
                    y={cy(r) + 6}
                    anchor="start"
                    font="display"
                    size={20}
                    color={GT.b}
                  >
                    {theirs}
                  </T>
                  <T
                    x={cx(c)}
                    y={cy(r) + CH / 2 - 11}
                    size={10}
                    color={nash ? GT.eq : best ? GT.opt : GT.muted}
                  >
                    {nash
                      ? 'you both land here'
                      : best
                        ? 'better for both'
                        : me === 'C'
                          ? 'sucker'
                          : 'temptation'}
                  </T>
                </g>
              );
            }),
          )}
          {/* your improvement arrow in each column: Cooperate -> Defect */}
          {[0, 1].map((c) => {
            const gain = payoff('D', MOVES[c]) - payoff('C', MOVES[c]);
            const lit = c === col;
            return (
              <g key={'a' + c} style={{ opacity: lit ? 1 : 0.45, transition: 'opacity 300ms' }}>
                <line
                  x1={cx(c)}
                  y1={Y0 + CH + 5}
                  x2={cx(c)}
                  y2={Y0 + CH + GY - 5}
                  markerEnd="url(#gt-arrow-a)"
                  style={{ stroke: GT.a, strokeWidth: lit ? 4 : 2.5 }}
                />
                <T
                  x={cx(c) + 18}
                  y={Y0 + CH + GY / 2 - 6}
                  anchor="start"
                  size={12}
                  color={GT.a}
                  weight={600}
                >
                  +{gain}
                </T>
              </g>
            );
          })}
          {/* T > R > P > S on a number line */}
          <line x1={NX0} y1={NY} x2={NX1} y2={NY} style={{ stroke: GT.lineStrong }} />
          <T x={NX0 - 8} y={NY} anchor="end" size={10}>
            payoff
          </T>
          {(
            [
              ['S', PAY.S, 'sucker'],
              ['P', PAY.P, 'both defect'],
              ['R', PAY.R, 'both coop.'],
              ['T', PAY.T, 'temptation'],
            ] as const
          ).map(([k, v, name]) => (
            <g key={k}>
              <circle cx={nx(v)} cy={NY} r={3.5} style={{ fill: GT.ink }} />
              <T x={nx(v)} y={NY - 12} size={11} color={GT.ink} weight={600}>
                {`${k} = ${v}`}
              </T>
              <T x={nx(v)} y={NY + 13} size={10}>
                {name}
              </T>
            </g>
          ))}
        </svg>

        <div className="flex flex-row flex-wrap gap-6 sm:flex-col sm:gap-5">
          <BigStat
            label={`if they ${NAME[them].toLowerCase()}`}
            value={`+${ifD - ifC}`}
            color={GT.a}
            size="lg"
            sub={`defect ${ifD} beats cooperate ${ifC}`}
          />
          <BigStat
            label="both defect"
            value={
              <span>
                {PAY.P}
                <span className="text-muted"> vs </span>
                <span style={{ color: GT.opt }}>{PAY.R}</span>
              </span>
            }
            color={GT.eq}
            size="md"
            sub="each, vs both cooperating"
          />
        </div>
      </div>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
        <Chip color={GT.a}>your payoff (big)</Chip>
        <Chip color={GT.b}>theirs (small)</Chip>
        <Chip color={GT.eq}>the equilibrium</Chip>
        <Chip color={GT.opt}>what both would prefer</Chip>
      </div>
      <Caption>
        Defecting is your better reply in both columns. They reason the same way, so you both defect
        and get {PAY.P} each, when cooperating would have paid {PAY.R} each.
      </Caption>
    </Specimen>
  );
}
