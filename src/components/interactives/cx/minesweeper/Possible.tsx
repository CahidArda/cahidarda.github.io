/**
 * CX-12.1 "is this board possible?". Kaye's Figure 4 (6 x 6, sixteen numbers, twenty grey
 * squares). The checker reads each number once against the mines on the board: green when it
 * fits, red when it sees too many. The searcher sweeps through all 2^20 ways to hide mines in
 * the grey squares. Demo: the mines go down in reading order, then the sweep runs. The second
 * tab changes one number so that no placement fits: the checker catches it at once, the
 * searcher needs the whole sweep to say "none".
 */
import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  GT,
  Seg,
  Specimen,
  T,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { BROKEN, FIG4, FIG4_MINES, FIG4_SOLUTION, SPACE, constraints, isHidden } from './kaye';

type Tab = 'kaye' | 'broken';
const BOARDS: Record<Tab, string[]> = { kaye: FIG4, broken: BROKEN };
const TABS: { value: Tab; label: string }[] = [
  { value: 'kaye', label: "Kaye's board" },
  { value: 'broken', label: 'one number changed' },
];
const CAPTION: Record<Tab, string> = {
  kaye: 'Checking a guess is one look at each of the 16 numbers. Finding one blind means trying placements: 2 choices for each grey square, 1,048,576 in all, and exactly one fits.',
  broken:
    'The bottom-right 2 is now a 1. The checker flags it the moment the mines go down. To be sure no other placement works, the searcher has to try all 1,048,576 and find none.',
};

const N = 6;
const CS = 40;
const BX = 30;
const BY = 6;
const W = 300;
const BAR_Y = 286;
const BAR_X0 = 14;
const BAR_X1 = 286;
const H = 324;

// timeline (seconds on one clock)
const T0 = 0.45; // first mine
const DT = 0.28; // between mines
const SWEEP_AT = T0 + FIG4_MINES.length * DT + 0.35;
const SWEEP_LEN = 3.0;
const T_END = SWEEP_AT + SWEEP_LEN;
const SOL_P = FIG4_SOLUTION / SPACE;

type State = { t: number; flags: string[]; user: boolean };
const POSTER: State = { t: T_END, flags: FIG4_MINES, user: false };
const EMPTY: State = { t: 0, flags: [], user: false };

const fmt = (n: number) => n.toLocaleString('en-US');

/** What the figure shows for a tab at clock t (flags from the demo unless the reader took over). */
function view(tab: Tab, s: State) {
  const flags = new Set(s.user ? s.flags : FIG4_MINES.filter((_, i) => s.t >= T0 + i * DT));
  const p = Math.max(0, Math.min(1, (s.t - SWEEP_AT) / SWEEP_LEN));
  const { hidden, nums } = constraints(BOARDS[tab]);
  const status = nums.map((q) => {
    const k = q.nb.filter((i) => flags.has(`${hidden[i].r},${hidden[i].c}`)).length;
    return { ...q, k, st: k === q.v ? 'fit' : k > q.v ? 'over' : 'under' };
  });
  const tried = Math.floor(p * SPACE);
  const found = tab === 'kaye' && p >= SOL_P ? 1 : 0;
  return { flags, p, status, tried, found, fit: status.filter((q) => q.st === 'fit').length };
}

export default function Possible() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [sel, setSel] = useState<Tab>('kaye');
  const [tab, bindTab] = useHoverPreview(sel);
  // one state per tab; the tab not yet played shows its finished poster when previewed
  const [states, setStates] = useState<Record<Tab, State>>({ kaye: EMPTY, broken: POSTER });
  const started = useRef(false);
  const [wait, setWait] = useState(0); // seconds to hold before the clock runs (restart pause)

  useEffect(() => {
    if (reduced) setStates({ kaye: POSTER, broken: POSTER });
  }, [reduced]);
  useEffect(() => {
    if (inView && !started.current && !reduced) {
      started.current = true;
      setWait(0.4);
    }
  }, [inView, reduced]);

  const cur = states[sel];
  const running = started.current && !reduced && inView && cur.t < T_END;
  useFrame((dt) => {
    if (wait > 0) return setWait((w) => Math.max(0, w - dt));
    setStates((s) => ({ ...s, [sel]: { ...s[sel], t: Math.min(T_END, s[sel].t + dt) } }));
  }, running);

  const replay = (t: Tab, pause: number) => {
    setStates((s) => ({ ...s, [t]: EMPTY }));
    if (reduced) setStates((s) => ({ ...s, [t]: POSTER }));
    else setWait(pause);
  };
  const choose = (t: Tab) => {
    if (t === sel) return;
    setSel(t);
    replay(t, 0.5);
  };
  const toggle = (key: string) => {
    setStates((s) => {
      const st = s[sel];
      const now = view(sel, st).flags;
      const next = new Set(now);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return { ...s, [sel]: { ...st, user: true, flags: [...next] } };
    });
  };

  const v = view(tab, states[tab]);
  const board = BOARDS[tab];
  const numColor = (st: string) => (st === 'fit' ? GT.opt : st === 'over' ? GT.b : GT.ink);
  const barX = (p: number) => BAR_X0 + p * (BAR_X1 - BAR_X0);

  return (
    <Specimen
      code="CX-12.1"
      title="is this board possible?"
      kicker="Kaye 2000, fig. 4"
      rootRef={rootRef}
      hint="tap a grey square to place or lift a mine"
      footer={<Btn onClick={() => replay(sel, 1)}>restart</Btn>}
    >
      <style>{`@keyframes cx11pop{from{transform:scale(0)}to{transform:scale(1)}}`}</style>
      <div className="grid items-start gap-x-8 gap-y-4 md:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="mx-auto block w-full max-w-[460px] select-none"
          role="img"
          aria-label={`A 6 by 6 Minesweeper board. ${v.fit} of 16 numbers fit the mines placed. Searcher tried ${fmt(v.tried)} of ${fmt(SPACE)} placements, ${v.found} fit.`}
        >
          {[...board.join('')].map((ch, i) => {
            const r = Math.floor(i / N);
            const c = i % N;
            const x = BX + c * CS;
            const y = BY + r * CS;
            const key = `${r},${c}`;
            if (isHidden(ch)) {
              const on = v.flags.has(key);
              return (
                <g
                  key={key}
                  onClick={() => toggle(key)}
                  style={{ cursor: 'pointer' }}
                  role="button"
                  aria-label={`square ${r + 1},${c + 1}${on ? ', mine' : ''}`}
                >
                  <rect
                    x={x + 1}
                    y={y + 1}
                    width={CS - 2}
                    height={CS - 2}
                    style={{ fill: GT.line, stroke: GT.lineStrong, strokeWidth: 1 }}
                  />
                  {on && (
                    <g
                      style={{
                        transformBox: 'fill-box',
                        transformOrigin: 'center',
                        animation: 'cx11pop 180ms ease-out',
                      }}
                    >
                      <Mine cx={x + CS / 2} cy={y + CS / 2} />
                    </g>
                  )}
                </g>
              );
            }
            const q = v.status.find((s) => s.r === r && s.c === c)!;
            const col = numColor(q.st);
            return (
              <g key={key}>
                <rect
                  x={x + 1}
                  y={y + 1}
                  width={CS - 2}
                  height={CS - 2}
                  style={{
                    fill: q.st === 'fit' ? GT.optSoft : q.st === 'over' ? GT.bSoft : GT.paper,
                    stroke: GT.line,
                    strokeWidth: 1,
                    transition: 'fill 160ms',
                  }}
                />
                <T x={x + CS / 2} y={y + CS / 2 + 1} size={19} weight={600} color={col}>
                  {q.v}
                </T>
              </g>
            );
          })}

          {/* the searcher: one line for every placement, swept left to right */}
          <T x={BAR_X0} y={BAR_Y - 14} anchor="start" size={9.5} color={GT.ink}>
            searcher · every way to fill the 20 grey squares
          </T>
          <rect
            x={BAR_X0}
            y={BAR_Y - 6}
            width={BAR_X1 - BAR_X0}
            height={12}
            style={{ fill: GT.paper, stroke: GT.lineStrong, strokeWidth: 1 }}
          />
          <rect
            x={BAR_X0}
            y={BAR_Y - 6}
            width={barX(v.p) - BAR_X0}
            height={12}
            style={{ fill: GT.lineStrong, opacity: 0.55 }}
          />
          {tab === 'kaye' && (
            <g>
              <line
                x1={barX(SOL_P)}
                y1={BAR_Y - 9}
                x2={barX(SOL_P)}
                y2={BAR_Y + 9}
                style={{ stroke: v.found ? GT.opt : GT.lineStrong, strokeWidth: 2.5 }}
              />
              <T x={barX(SOL_P)} y={BAR_Y + 19} size={9} color={v.found ? GT.opt : GT.muted}>
                the one that fits
              </T>
            </g>
          )}
          {tab === 'broken' && (
            <T x={BAR_X1} y={BAR_Y + 19} anchor="end" size={9} color={v.p >= 1 ? GT.b : GT.muted}>
              {v.p >= 1 ? 'none fit' : 'none yet'}
            </T>
          )}
          <line
            x1={barX(v.p)}
            y1={BAR_Y - 9}
            x2={barX(v.p)}
            y2={BAR_Y + 9}
            style={{ stroke: GT.ink, strokeWidth: 1.5, opacity: v.p > 0 && v.p < 1 ? 1 : 0 }}
          />

          {tab === 'kaye' && (
            <T x={BAR_X1} y={BAR_Y + 19} anchor="end" size={9}>
              1,048,576
            </T>
          )}
        </svg>

        <div className="grid gap-5">
          <Seg options={TABS} value={tab} onChange={choose} bind={bindTab} />
          <div className="grid grid-cols-2 gap-4 md:grid-cols-1">
            <BigStat
              label="checker"
              value={v.fit}
              unit="/ 16 fit"
              color={v.status.some((q) => q.st === 'over') ? GT.b : v.fit === 16 ? GT.opt : GT.ink}
              sub="one look per number"
            />
            <BigStat
              label="searcher"
              value={fmt(v.tried)}
              unit="tried"
              sub={`of 1,048,576 · ${v.found} fit${v.found === 1 ? 's' : ''}`}
            />
          </div>
          <Reserve
            items={TABS.map((t) => (
              <p key={t.value} className="text-[0.85rem] leading-snug text-ink-soft">
                {CAPTION[t.value]}
              </p>
            ))}
            active={tab === 'kaye' ? 0 : 1}
          />
        </div>
      </div>
    </Specimen>
  );
}

function Mine({ cx, cy }: { cx: number; cy: number }) {
  const r = 7.5;
  return (
    <g>
      {[0, 45, 90, 135].map((a) => {
        const dx = Math.cos((a * Math.PI) / 180) * (r + 3.5);
        const dy = Math.sin((a * Math.PI) / 180) * (r + 3.5);
        return (
          <line
            key={a}
            x1={cx - dx}
            y1={cy - dy}
            x2={cx + dx}
            y2={cy + dy}
            style={{ stroke: GT.ink, strokeWidth: 2, strokeLinecap: 'round' }}
          />
        );
      })}
      <circle cx={cx} cy={cy} r={r} style={{ fill: GT.ink }} />
      <circle cx={cx - 2.4} cy={cy - 2.4} r={2} style={{ fill: GT.paper }} />
    </g>
  );
}
