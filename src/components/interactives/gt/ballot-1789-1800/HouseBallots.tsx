import { useEffect, useRef, useState } from 'react';
import { BigStat, Btn, GT, Specimen, T, useInView, useReducedMotion } from '../shared';
import { BALLOTS, HOUSE_1801, houseCount, houseVote } from './data';

const STEP_MS = 120;
const PAUSE_MS = 1400;

export default function HouseBallots() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  // SSR and reduced motion show the finished record
  const [n, setN] = useState(BALLOTS);
  const [running, setRunning] = useState(false);
  const demo = useRef(false);

  const play = () => {
    if (reduced) {
      setN(BALLOTS);
      return;
    }
    setN(0);
    setRunning(true);
  };

  useEffect(() => {
    if (!inView || demo.current || reduced) return;
    demo.current = true;
    const id = setTimeout(play, 700);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  useEffect(() => {
    if (!running || !inView) return;
    if (n >= BALLOTS) {
      setRunning(false);
      return;
    }
    const id = setTimeout(() => setN((x) => x + 1), n === BALLOTS - 1 ? PAUSE_MS : STEP_MS);
    return () => clearTimeout(id);
  }, [running, n, inView]);

  const { J, B } = houseCount(Math.max(1, n));
  const done = n >= BALLOTS;

  // the House grid: one row per state delegation, one column per ballot.
  // Phones and desktops get their own drawing, chosen by CSS (no re-layout after hydration).
  const chart = (compact: boolean) => {
    const W = compact ? 380 : 640;
    const labW = compact ? 24 : 30;
    const rightW = compact ? 72 : 92;
    const gTop = 20;
    const cw = (W - labW - rightW) / BALLOTS;
    const ch = compact ? 13 : 14;
    const gridB = gTop + HOUSE_1801.length * ch;
    const H = gridB + 22;
    const gx = (c: number) => Math.round((labW + c * cw) * 100) / 100;
    const groups = [
      { from: 0, to: 7, text: 'Jefferson', color: GT.b },
      { from: 8, to: 9, text: 'split', color: GT.inkSoft },
      { from: 10, to: 15, text: 'Burr', color: GT.alt },
    ];
    const rx = gx(BALLOTS) + 8;
    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full select-none"
        role="img"
        aria-label="House ballots 1 to 35: Jefferson 8 states, Burr 6, 2 divided. Ballot 36: Jefferson 10, Burr 4, 2 blank."
      >
        <defs>
          <pattern
            id={compact ? 'ex3-div-n' : 'ex3-div'}
            width="4"
            height="4"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <rect width="4" height="4" style={{ fill: GT.bSoft }} />
            <line x1="0" y1="0" x2="0" y2="4" style={{ stroke: GT.alt, strokeWidth: 1.6 }} />
          </pattern>
        </defs>
        {HOUSE_1801.map((st, r) => {
          const y = gTop + r * ch;
          return (
            <g key={st.id}>
              <T
                x={labW - 5}
                y={y + ch / 2}
                anchor="end"
                size={compact ? 8.5 : 9.5}
                color={GT.inkSoft}
              >
                {st.id}
              </T>
              {Array.from({ length: BALLOTS }, (_, c) => {
                const shown = c < n;
                const v = houseVote(r, c + 1);
                const fill =
                  v === 'J'
                    ? GT.b
                    : v === 'B'
                      ? GT.alt
                      : v === 'div'
                        ? `url(#${compact ? 'ex3-div-n' : 'ex3-div'})`
                        : GT.paper;
                return (
                  <rect
                    key={c}
                    x={gx(c)}
                    y={y + 1}
                    width={Math.max(1, Math.round((cw - 1.4) * 100) / 100)}
                    height={ch - 2}
                    style={{
                      fill: shown ? fill : 'none',
                      stroke: shown && v === 'blank' ? GT.lineStrong : GT.line,
                      strokeWidth: 0.6,
                    }}
                  />
                );
              })}
            </g>
          );
        })}
        {/* direct labels for the three blocs (ballots 1 to 35) */}
        {groups.map((g) => {
          const y0 = gTop + g.from * ch + 1;
          const y1 = gTop + (g.to + 1) * ch - 1;
          return (
            <g key={g.text}>
              <line x1={rx} y1={y0} x2={rx} y2={y1} style={{ stroke: g.color, strokeWidth: 1.5 }} />
              <T
                x={rx + 6}
                y={(y0 + y1) / 2}
                anchor="start"
                size={compact ? 9 : 11}
                color={g.color}
                weight={600}
              >
                {g.text}
              </T>
            </g>
          );
        })}
        {/* ballot axis */}
        {[1, 10, 20, 30].map((b) => (
          <T key={b} x={gx(b - 0.5)} y={gridB + 11} size={9}>
            {b}
          </T>
        ))}
        <T
          x={gx(BALLOTS - 0.5)}
          y={gridB + 11}
          size={9}
          color={done ? GT.ink : GT.muted}
          weight={done ? 700 : 400}
        >
          36
        </T>
        <T x={gx(BALLOTS)} y={gTop - 9} anchor="end" size={9} color={done ? GT.ink : GT.muted}>
          {done ? 'ballot 36 · Feb 17, 1801 →' : 'ballots from Feb 11, 1801'}
        </T>
        {done && (
          <rect
            x={gx(BALLOTS - 1) - 1.5}
            y={gTop - 2}
            width={cw + 1.6}
            height={HOUSE_1801.length * ch + 4}
            style={{ fill: 'none', stroke: GT.ink, strokeWidth: 1.6 }}
          />
        )}
      </svg>
    );
  };

  return (
    <Specimen
      code="EX-01.3"
      title="thirty-six ballots"
      kicker="Feb 1801 · the House"
      rootRef={rootRef}
      hint="one vote per state delegation; press replay"
      footer="9 of 16 states needed"
    >
      <div className="hidden sm:block">{chart(false)}</div>
      <div className="sm:hidden">{chart(true)}</div>

      <p className="mt-3 h-[4rem] overflow-hidden text-[0.85rem] leading-snug text-ink-soft sm:h-[2.6rem]">
        {done
          ? 'Ballot 36: Federalists in Maryland and Vermont abstain, handing both states to Jefferson. Delaware and South Carolina cast blanks.'
          : 'The electors tied Jefferson and Burr at 73, so the House voted by state: eight for Jefferson, one short of nine, ballot after ballot.'}
      </p>

      <div className="mt-3 flex flex-wrap items-end justify-between gap-x-8 gap-y-4 border-t border-line pt-4">
        <div className="flex gap-8">
          <BigStat label="ballot" value={Math.max(1, n)} unit="/ 36" />
          <BigStat label="Jefferson" value={J} unit="states" color={GT.b} />
          <BigStat label="Burr" value={B} unit="states" color={GT.alt} />
        </div>
        <Btn primary onClick={play}>
          replay
        </Btn>
      </div>
    </Specimen>
  );
}
