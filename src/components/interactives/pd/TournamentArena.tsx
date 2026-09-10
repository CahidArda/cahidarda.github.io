import { useEffect, useMemo, useRef, useState } from 'react';
import { Widget, useReducedMotion } from './shared';
import { simulateMatch } from './engine';
import { CLASSIC, STRATEGIES } from './strategies';

const ABBREV: Record<string, string> = {
  tft: 'TFT',
  alld: 'ALLD',
  allc: 'ALLC',
  grudger: 'GRUD',
  pavlov: 'PAV',
  random: 'RND',
};

const ROUNDS = 100;

interface Result {
  cells: number[][]; // per-round avg payoff to row vs col
  totals: { id: string; name: string; score: number }[];
}

function runTournament(noise: number, seed: number): Result {
  const n = STRATEGIES.length;
  const cells: number[][] = [];
  const totals = STRATEGIES.map((s) => ({ id: s.id, name: s.name, score: 0 }));
  for (let i = 0; i < n; i++) {
    cells[i] = [];
    for (let j = 0; j < n; j++) {
      const m = simulateMatch(
        STRATEGIES[i],
        STRATEGIES[j],
        ROUNDS,
        CLASSIC,
        noise,
        seed + i * 31 + j * 7,
      );
      cells[i][j] = m.scoreA / ROUNDS;
      totals[i].score += m.scoreA;
    }
  }
  totals.sort((a, b) => b.score - a.score);
  return { cells, totals };
}

// paper -> accent ramp keyed on per-round payoff (0..5)
function cellBg(v: number): string {
  const t = Math.max(0, Math.min(1, v / 5));
  return `color-mix(in srgb, var(--color-accent) ${Math.round(t * 78)}%, var(--color-paper))`;
}

export default function TournamentArena() {
  const reduced = useReducedMotion();
  const n = STRATEGIES.length;
  const [noise, setNoise] = useState(false);
  const [runId, setRunId] = useState(0);
  const [result, setResult] = useState<Result | null>(null);
  const [revealed, setRevealed] = useState(0); // cells shown so far
  const [showBoard, setShowBoard] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  const total = n * n;

  function run() {
    if (timer.current) clearInterval(timer.current);
    const res = runTournament(noise ? 0.1 : 0, 1000 + runId * 101);
    setResult(res);
    setShowBoard(false);
    if (reduced) {
      setRevealed(total);
      setShowBoard(true);
      setRunId((r) => r + 1);
      return;
    }
    setRevealed(0);
    let c = 0;
    timer.current = setInterval(
      () => {
        c += 1;
        setRevealed(c);
        if (c >= total) {
          clearInterval(timer.current);
          setTimeout(() => setShowBoard(true), 200);
          setRunId((r) => r + 1);
        }
      },
      1000 / total > 18 ? 18 : 1000 / total,
    );
  }

  useEffect(() => () => timer.current && clearInterval(timer.current), []);

  const maxScore = useMemo(
    () => (result ? Math.max(...result.totals.map((t) => t.score)) : 1),
    [result],
  );

  return (
    <Widget title="§5 · Tournament Arena">
      <p className="mb-4 text-sm text-ink-soft">Put them all in a room. Everyone plays everyone.</p>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={run}
          className="border-2 border-accent px-3.5 py-1.5 font-mono text-xs tracking-[0.12em] text-accent uppercase transition-colors hover:bg-accent hover:text-paper"
        >
          ▶ Run Tournament
        </button>
        <label className="flex cursor-pointer items-center gap-2 text-sm text-ink-soft">
          <input
            type="checkbox"
            checked={noise}
            onChange={(e) => setNoise(e.target.checked)}
            className="accent-[var(--color-accent)]"
          />
          Trembling-hand noise
        </label>
      </div>

      <div className="grid gap-5 lg:grid-cols-[auto_1fr]">
        {/* Heatmap */}
        <div>
          <div
            className="grid gap-0.5"
            style={{ gridTemplateColumns: `2.6rem repeat(${n}, minmax(0,1fr))` }}
          >
            <div />
            {STRATEGIES.map((s) => (
              <div key={s.id} className="pb-1 text-center font-mono text-[0.6rem] text-muted">
                {ABBREV[s.id]}
              </div>
            ))}
            {STRATEGIES.map((row, i) => (
              <RowCells
                key={row.id}
                label={ABBREV[row.id]}
                values={result?.cells[i] ?? []}
                baseIndex={i * n}
                revealed={revealed}
              />
            ))}
          </div>
          <p className="mt-2 font-mono text-[0.6rem] text-muted">
            row’s avg payoff vs column · darker = higher
          </p>
        </div>

        {/* Leaderboard */}
        <div className="flex flex-col gap-1.5">
          <span className="label mb-1">Leaderboard</span>
          {!result && <p className="text-sm text-muted">Run the tournament to rank the room.</p>}
          {result &&
            result.totals.map((t, rank) => (
              <div
                key={t.id}
                className={showBoard ? 'pd-fly-in' : ''}
                style={{
                  animationDelay: showBoard && !reduced ? `${rank * 70}ms` : undefined,
                  opacity: showBoard ? 1 : 0,
                }}
              >
                <div className="flex items-center gap-2">
                  <span className="w-5 text-right font-mono text-sm text-muted">{rank + 1}</span>
                  <span className="w-20 shrink-0 font-mono text-xs">{t.name}</span>
                  <div
                    className="h-4 flex-1 bg-paper"
                    style={{ border: '1px solid var(--color-line)' }}
                  >
                    <div
                      className="h-full transition-all duration-500"
                      style={{
                        width: showBoard ? `${(t.score / maxScore) * 100}%` : '0%',
                        background: rank === 0 ? 'var(--color-accent)' : 'var(--color-line-strong)',
                      }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-xs tabular-nums">{t.score}</span>
                </div>
              </div>
            ))}
          {showBoard && result && (
            <p className="mt-2 text-xs leading-snug text-ink-soft">
              The winner is{' '}
              <strong style={{ color: 'var(--color-ink)' }}>{result.totals[0].name}</strong>: nice
              (never defects first), retaliatory, and forgiving — Axelrod’s recipe. Turn on noise
              and re-run to watch unforgiving strategies slip.
            </p>
          )}
        </div>
      </div>
    </Widget>
  );
}

function RowCells({
  label,
  values,
  baseIndex,
  revealed,
}: {
  label: string;
  values: number[];
  baseIndex: number;
  revealed: number;
}) {
  return (
    <>
      <div className="flex items-center justify-end pr-1 font-mono text-[0.6rem] text-muted">
        {label}
      </div>
      {STRATEGIES.map((_, j) => {
        const idx = baseIndex + j;
        const on = idx < revealed && values.length > 0;
        const v = values[j] ?? 0;
        return (
          <div
            key={j}
            className="flex aspect-square items-center justify-center font-mono text-[0.62rem] transition-all duration-150"
            style={{
              background: on ? cellBg(v) : 'var(--color-paper)',
              border: '1px solid var(--color-line)',
              color: on && v / 5 > 0.55 ? 'var(--color-paper)' : 'var(--color-ink-soft)',
              transform: on ? 'scale(1)' : 'scale(0.85)',
            }}
          >
            {on ? v.toFixed(1) : ''}
          </div>
        );
      })}
    </>
  );
}
