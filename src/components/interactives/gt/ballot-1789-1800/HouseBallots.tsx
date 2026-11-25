import { useEffect, useRef, useState } from 'react';
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
import {
  BALLOTS,
  HOUSE_1801,
  MAJ_1800,
  N_1800,
  STATES_NEEDED,
  TALLY_1800,
  houseCount,
  houseVote,
  type HouseVote,
} from './data';
import { useCompact } from './useCompact';

const STEP_MS = 120;
const PAUSE_MS = 1400;

const fillOf = (v: HouseVote) =>
  v === 'J' ? GT.b : v === 'B' ? GT.alt : v === 'div' ? 'url(#ex3-div)' : GT.paper;

export default function HouseBallots() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [boxRef, compact] = useCompact();
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

  // ── electoral tally (top) ──
  const W = compact ? 380 : 640;
  const lx = compact ? 70 : 92;
  const rx = W - (compact ? 34 : 40);
  const bx = (v: number) => lx + (v / 80) * (rx - lx);
  const rowT = 20;

  // ── the House grid ──
  const gTop = TALLY_1800.length * rowT + 46;
  const labW = compact ? 26 : 34;
  const cw = (W - labW - 4) / BALLOTS;
  const ch = compact ? 11 : 12;
  const H = gTop + 14 + HOUSE_1801.length * ch + 28;

  return (
    <Specimen
      code="EX-01.3"
      title="thirty-six ballots"
      kicker="1800 electors · Feb 1801 House"
      rootRef={rootRef}
      hint="press replay to rerun the House vote"
      footer={`9 of 16 states needed`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="label">one vote per state delegation</span>
        <Btn primary onClick={play}>
          replay
        </Btn>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3">
        <BigStat
          label="ballot"
          value={Math.max(1, n)}
          unit="/ 36"
          sub={done ? 'Feb 17, 1801' : 'from Feb 11, 1801'}
        />
        <BigStat
          label="Jefferson"
          value={J}
          unit="states"
          color={GT.b}
          sub={`needs ${STATES_NEEDED}`}
        />
        <BigStat
          label="Burr"
          value={B}
          unit="states"
          color={GT.alt}
          sub={done ? 'Delaware, S.C. blank' : 'Md., Vt. split'}
          align="right"
        />
      </div>

      <div ref={boxRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="mt-3 block w-full select-none"
          role="img"
          aria-label={`1800 electoral votes: Jefferson 73, Burr 73, Adams 65, Pinckney 64, Jay 1. House ballots 1 to 35: Jefferson 8 states, Burr 6, 2 divided. Ballot 36: Jefferson 10, Burr 4, 2 blank.`}
        >
          <defs>
            <pattern
              id="ex3-div"
              width="4"
              height="4"
              patternUnits="userSpaceOnUse"
              patternTransform="rotate(45)"
            >
              <rect width="4" height="4" style={{ fill: GT.bSoft }} />
              <line x1="0" y1="0" x2="0" y2="4" style={{ stroke: GT.alt, strokeWidth: 1.6 }} />
            </pattern>
          </defs>

          {/* electoral votes, Dec 1800 */}
          <T x={4} y={8} anchor="start" size={10} color={GT.ink} weight={600}>
            ELECTORAL VOTES, 1800 · {N_1800} electors, 2 votes each
          </T>
          {TALLY_1800.map((c, i) => {
            const y = 20 + i * rowT;
            const col = c.side === 'R' ? (i === 0 ? GT.b : GT.alt) : GT.a;
            const op = c.side === 'F' && i > 2 ? 0.5 : 1;
            return (
              <g key={c.name}>
                <T x={lx - 8} y={y + 7} anchor="end" size={11} color={GT.inkSoft}>
                  {c.name}
                </T>
                <rect
                  x={lx}
                  y={y}
                  width={bx(c.votes) - lx}
                  height={14}
                  style={{ fill: col, opacity: op }}
                />
                <T
                  x={bx(c.votes) + 5}
                  y={y + 7.5}
                  anchor="start"
                  size={11}
                  color={GT.ink}
                  weight={600}
                >
                  {c.votes}
                </T>
              </g>
            );
          })}
          <line
            x1={bx(MAJ_1800)}
            y1={16}
            x2={bx(MAJ_1800)}
            y2={20 + TALLY_1800.length * rowT - 2}
            style={{ stroke: GT.ink, strokeWidth: 1, strokeDasharray: '3 2' }}
          />
          <T x={bx(MAJ_1800) - 4} y={20 + TALLY_1800.length * rowT + 6} anchor="end" size={9.5}>
            majority {MAJ_1800}
          </T>

          {/* House ballots */}
          <T x={4} y={gTop} anchor="start" size={10} color={GT.ink} weight={600}>
            HOUSE OF REPRESENTATIVES · one column per ballot
          </T>
          {HOUSE_1801.map((st, r) => {
            const y = gTop + 12 + r * ch;
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
                  return (
                    <rect
                      key={c}
                      x={Math.round((labW + c * cw) * 100) / 100}
                      y={y + 1}
                      width={Math.max(1, Math.round((cw - 1.4) * 100) / 100)}
                      height={ch - 2}
                      style={{
                        fill: shown ? fillOf(v) : 'none',
                        stroke: shown && v === 'blank' ? GT.lineStrong : GT.line,
                        strokeWidth: 0.6,
                      }}
                    />
                  );
                })}
              </g>
            );
          })}
          {/* ballot axis */}
          {[1, 10, 20, 30, 36].map((b) => (
            <T
              key={b}
              x={labW + (b - 0.5) * cw}
              y={gTop + 12 + HOUSE_1801.length * ch + 9}
              size={9}
            >
              {b}
            </T>
          ))}
          {done && (
            <rect
              x={labW + 35 * cw - 1.5}
              y={gTop + 9}
              width={cw + 1.6}
              height={HOUSE_1801.length * ch + 6}
              style={{ fill: 'none', stroke: GT.ink, strokeWidth: 1.6 }}
            />
          )}
        </svg>
      </div>

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        <Chip color={GT.b}>Jefferson</Chip>
        <Chip color={GT.alt}>Burr</Chip>
        <Chip color={GT.bSoft}>delegation split</Chip>
        <Chip color={GT.paper}>blank</Chip>
      </div>
      <Caption>
        {done
          ? 'Ballot 36: Federalists in Maryland and Vermont abstain, which hands both states to Jefferson. Delaware and South Carolina cast blanks. Jefferson has 10.'
          : 'Eight states for Jefferson, one short. Thirty-five times.'}
      </Caption>
    </Specimen>
  );
}
