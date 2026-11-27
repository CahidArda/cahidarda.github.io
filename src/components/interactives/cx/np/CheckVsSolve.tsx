import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Specimen,
  T,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { useCompact } from '../../gt/useCompact';
import { CHECK_STEPS, GROUPS, N, PUZZLE, badGroups, solveTrace, type Grid } from './sudoku';

/**
 * CX-03.1 · check vs solve. Two copies of the same 6x6 Sudoku run on ONE clock. Left: a checker
 * reads a proposed answer, group by group (108 steps). Right: a backtracking solver tries digits
 * (3,013 steps). The clock starts at 60 steps a second and speeds up as it runs, the same for
 * both, so the step counts stay a fair comparison. Afterwards the reader can change any digit
 * of the proposed answer and watch the checker catch it.
 */
const TRACE = solveTrace(PUZZLE);
const SOLUTION = TRACE.solution;
const SOLVE_STEPS = TRACE.steps.filter((s) => s.digit > 0).length;
// cumulative "tries" at each trace event (erasing on the way back is free)
const TRY_AT: number[] = (() => {
  let k = 0;
  return TRACE.steps.map((s) => (s.digit > 0 ? ++k : k));
})();

// the shared clock: steps(t) = R0 * TAU * (e^(t/TAU) - 1); steps per second start at R0
const R0 = 60;
const TAU = 3;
const stepsAt = (t: number) => R0 * TAU * (Math.exp(t / TAU) - 1);
const T_SOLVE = TAU * Math.log(1 + SOLVE_STEPS / (R0 * TAU)); // about 8.6 s

// two layouts: wide (desktop) and compact (phones), same structure, bigger marks on phones
const LAYOUT = {
  wide: { W: 520, CELL: 34, GX: [18, 284] },
  compact: { W: 360, CELL: 26, GX: [6, 198] },
};
const GY = 30;

const fmt = (n: number) => Math.floor(n).toLocaleString('en-US');

export default function CheckVsSolve() {
  const [wrapRef, compact] = useCompact(460);
  const { W, CELL, GX } = compact ? LAYOUT.compact : LAYOUT.wide;
  const SIDE = CELL * N;
  const H = GY + SIDE + 8;
  const cellXY = (g: number, i: number) => ({
    x: GX[g] + (i % N) * CELL,
    y: GY + Math.floor(i / N) * CELL,
  });
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  // the proposed answer the checker reads (the reader may edit it)
  const [proposal, setProposal] = useState<Grid>(SOLUTION);
  // race clock (seconds) and whether it runs
  const [t, setT] = useState(0);
  const [running, setRunning] = useState(false);
  // a re-check after an edit: its own step count, at the base rate
  const [recheck, setRecheck] = useState<number | null>(null);
  const started = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tRef = useRef(0);

  const finishAll = () => {
    tRef.current = T_SOLVE;
    setT(T_SOLVE);
    setRunning(false);
  };

  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    if (reduced) {
      finishAll();
      return;
    }
    timer.current = setTimeout(() => setRunning(true), 600);
  }, [inView, reduced]);
  // the reduced-motion hook settles one render after mount: jump to the poster then
  useEffect(() => {
    if (!reduced) return;
    if (timer.current) clearTimeout(timer.current);
    finishAll();
  }, [reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  useFrame((dt0) => {
    const dt = Math.max(0, dt0); // the first rAF can report dt < 0
    if (recheck !== null) {
      const n = Math.min(CHECK_STEPS, recheck + dt * R0 * 1.5);
      setRecheck(n >= CHECK_STEPS ? null : n);
    }
    if (tRef.current < T_SOLVE) {
      const n = Math.min(T_SOLVE, tRef.current + dt);
      tRef.current = n;
      setT(n);
      if (n >= T_SOLVE) setRunning(recheck !== null);
    } else if (recheck === null) setRunning(false);
  }, running && inView);

  const replay = () => {
    if (timer.current) clearTimeout(timer.current);
    setProposal(SOLUTION);
    setRecheck(null);
    if (reduced) {
      finishAll();
      return;
    }
    tRef.current = 0;
    setT(0);
    setRunning(false);
    timer.current = setTimeout(() => setRunning(true), 1000);
  };

  const editCell = (i: number) => {
    if (PUZZLE[i] !== 0) return;
    // the race finishes instantly if the reader starts editing mid-run
    if (tRef.current < T_SOLVE) {
      tRef.current = T_SOLVE;
      setT(T_SOLVE);
    }
    setProposal((p) => {
      const q = p.slice();
      q[i] = (q[i] % N) + 1;
      return q;
    });
    if (reduced) {
      setRecheck(null);
      return;
    }
    setRecheck(0);
    setRunning(true);
  };

  // ── derive both grids from the clock ──
  const steps = stepsAt(t);
  const checkSteps = recheck !== null ? recheck : Math.min(CHECK_STEPS, steps);
  const solveSteps = t >= T_SOLVE ? SOLVE_STEPS : Math.min(SOLVE_STEPS, steps);
  const checkDone = checkSteps >= CHECK_STEPS;
  const solveDone = solveSteps >= SOLVE_STEPS;
  const bad = badGroups(proposal);
  const anyBad = bad.some(Boolean);

  // solver grid: replay the trace up to the current number of tries
  const solveGrid = PUZZLE.slice();
  let cur = -1;
  for (let e = 0; e < TRACE.steps.length && TRY_AT[e] <= solveSteps; e++) {
    const s = TRACE.steps[e];
    solveGrid[s.cell] = s.digit;
    cur = s.cell;
  }
  if (solveDone) cur = -1;

  // checker: which group is being read now, which are finished
  const gIdx = Math.min(GROUPS.length - 1, Math.floor(checkSteps / N));
  const reading = !checkDone && (t > 0 || recheck !== null) ? gIdx : -1;
  const readCell = reading >= 0 ? GROUPS[gIdx][Math.floor(checkSteps) % N] : -1;
  const groupsDone = checkDone ? GROUPS.length : gIdx;
  const flagged = new Set<number>();
  GROUPS.forEach((cells, k) => {
    if (k < groupsDone && bad[k]) cells.forEach((c) => flagged.add(c));
  });

  const caption = [
    'Same puzzle, same clock. The checker reads each row, column and box once.',
    'The checker is done: every row, column and box holds 1 to 6. The solver is still guessing.',
    `Solved, after ${fmt(SOLVE_STEPS)} tries. Checking took ${CHECK_STEPS}. Click a blue digit on the left to break the answer.`,
    'Caught: the red row, column or box repeats a digit. Checking still took 108 steps.',
  ];
  const capIdx = anyBad && checkDone ? 3 : solveDone ? 2 : checkDone && t > 0 ? 1 : 0;

  const groupRect = (k: number) => {
    const cells = GROUPS[k];
    const xs = cells.map((c) => c % N);
    const ys = cells.map((c) => Math.floor(c / N));
    const x0 = Math.min(...xs);
    const y0 = Math.min(...ys);
    return {
      x: GX[0] + x0 * CELL,
      y: GY + y0 * CELL,
      w: (Math.max(...xs) - x0 + 1) * CELL,
      h: (Math.max(...ys) - y0 + 1) * CELL,
    };
  };

  const grid = (g: number, values: Grid) => (
    <g>
      {values.map((v, i) => {
        const { x, y } = cellXY(g, i);
        const given = PUZZLE[i] !== 0;
        let fill: string = GT.paperRaised;
        if (g === 0) {
          if (flagged.has(i)) fill = GT.bSoft;
          else if (checkDone && !anyBad) fill = GT.optSoft;
        } else if (solveDone) fill = GT.optSoft;
        const isCur = g === 1 && i === cur;
        const isRead = g === 0 && i === readCell;
        return (
          <g
            key={i}
            onClick={g === 0 && !given ? () => editCell(i) : undefined}
            style={{ cursor: g === 0 && !given ? 'pointer' : 'default' }}
          >
            <rect x={x} y={y} width={CELL} height={CELL} style={{ fill, stroke: GT.line }} />
            {(isCur || isRead) && (
              <rect
                x={x + 2}
                y={y + 2}
                width={CELL - 4}
                height={CELL - 4}
                style={{ fill: 'none', stroke: GT.eq, strokeWidth: 2 }}
              />
            )}
            {v > 0 && (
              <T
                x={x + CELL / 2}
                y={y + CELL / 2 + 1}
                size={compact ? 16 : 19}
                font="sans"
                color={given ? GT.ink : g === 0 && flagged.has(i) ? GT.b : GT.a}
                weight={given ? 700 : 400}
              >
                {v}
              </T>
            )}
          </g>
        );
      })}
      {/* box borders: 2 rows x 3 columns */}
      {[0, 2, 4, 6].map((r) => (
        <line
          key={`r${r}`}
          x1={GX[g]}
          x2={GX[g] + SIDE}
          y1={GY + r * CELL}
          y2={GY + r * CELL}
          style={{ stroke: GT.ink, strokeWidth: r === 0 || r === 6 ? 2 : 1.5 }}
        />
      ))}
      {[0, 3, 6].map((c) => (
        <line
          key={`c${c}`}
          x1={GX[g] + c * CELL}
          x2={GX[g] + c * CELL}
          y1={GY}
          y2={GY + SIDE}
          style={{ stroke: GT.ink, strokeWidth: c === 3 ? 1.5 : 2 }}
        />
      ))}
    </g>
  );

  const rg = reading >= 0 ? groupRect(reading) : null;

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-03.1"
      title="check vs solve"
      kicker="one clock for both, speeding up"
      hint="click a blue digit on the left to change it"
      footer={
        <Btn primary onClick={replay}>
          replay
        </Btn>
      }
    >
      <div ref={wrapRef}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label="Two copies of a 6 by 6 Sudoku: one being checked, one being solved by trial and error"
        >
          <T x={GX[0]} y={12} anchor="start" size={11} color={GT.ink} weight={700}>
            CHECK a proposed answer
          </T>
          <T x={GX[1]} y={12} anchor="start" size={11} color={GT.ink} weight={700}>
            SOLVE from scratch
          </T>
          {grid(0, proposal)}
          {grid(1, solveGrid)}
          {rg && (
            <rect
              x={rg.x}
              y={rg.y}
              width={rg.w}
              height={rg.h}
              style={{ fill: 'none', stroke: GT.eq, strokeWidth: 3 }}
            />
          )}
        </svg>

        <div
          className="mt-4 grid grid-cols-2"
          style={{
            columnGap: `${((GX[1] - GX[0] - SIDE) / W) * 100}%`,
            paddingLeft: `${(GX[0] / W) * 100}%`,
            paddingRight: `${((W - GX[1] - SIDE) / W) * 100}%`,
          }}
        >
          <BigStat
            label={checkDone ? (anyBad ? 'check · wrong' : 'check · valid') : 'check'}
            value={fmt(checkSteps)}
            unit="steps"
            color={checkDone ? (anyBad ? GT.b : GT.opt) : GT.ink}
          />
          <BigStat
            label={solveDone ? 'solve · found' : 'solve'}
            value={fmt(solveSteps)}
            unit="steps"
            color={solveDone ? GT.opt : GT.ink}
          />
        </div>
      </div>
      <Reserve
        items={caption.map((c, i) => (
          <Caption key={i}>{c}</Caption>
        ))}
        active={capIdx}
      />
    </Specimen>
  );
}
