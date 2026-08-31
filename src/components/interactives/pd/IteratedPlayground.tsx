import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ChoiceButton,
  MOVE_COLOR,
  MoveTape,
  Ticker,
  Widget,
  useReducedMotion,
  type Move,
} from './shared';
import { mulberry32, nextMove } from './engine';
import { CLASSIC, payoff, STRATEGIES, STRAT_BY_ID } from './strategies';

export const PLAYGROUND_ID = 'pd-playground';
/** The zoo (§4) fires this to load a strategy here and scroll into view. */
export const PLAY_EVENT = 'pd:play';

export default function IteratedPlayground() {
  const reduced = useReducedMotion();
  const [oppId, setOppId] = useState('tft');
  const [rounds, setRounds] = useState(10);
  const [mine, setMine] = useState<Move[]>([]);
  const [opp, setOpp] = useState<Move[]>([]);
  const [myScore, setMyScore] = useState(0);
  const [oppScore, setOppScore] = useState(0);
  const [lastDelta, setLastDelta] = useState<[number, number] | null>(null);
  const rngRef = useRef<() => number>(mulberry32(1));

  const opponent = STRAT_BY_ID[oppId];
  const done = mine.length >= rounds;

  const reset = useCallback((seed = (Math.random() * 1e9) | 0) => {
    rngRef.current = mulberry32(seed);
    setMine([]);
    setOpp([]);
    setMyScore(0);
    setOppScore(0);
    setLastDelta(null);
  }, []);

  // reset whenever the matchup or length changes
  useEffect(() => reset(), [oppId, rounds, reset]);

  // receive "play against this" from the strategy zoo
  useEffect(() => {
    function onPlay(e: Event) {
      const id = (e as CustomEvent<{ strategyId: string }>).detail?.strategyId;
      if (id && STRAT_BY_ID[id]) {
        setOppId(id);
        document.getElementById(PLAYGROUND_ID)?.scrollIntoView({
          behavior: reduced ? 'auto' : 'smooth',
          block: 'center',
        });
      }
    }
    window.addEventListener(PLAY_EVENT, onPlay);
    return () => window.removeEventListener(PLAY_EVENT, onPlay);
  }, [reduced]);

  function play(myMove: Move) {
    if (done) return;
    const oppMove = nextMove(opponent, opp, mine, rngRef.current);
    const dMy = payoff(myMove, oppMove, CLASSIC);
    const dOpp = payoff(oppMove, myMove, CLASSIC);
    setMine((m) => [...m, myMove]);
    setOpp((o) => [...o, oppMove]);
    setMyScore((s) => s + dMy);
    setOppScore((s) => s + dOpp);
    setLastDelta([dMy, dOpp]);
  }

  const winner =
    myScore > oppScore ? 'You lead' : myScore < oppScore ? `${opponent.name} leads` : 'Dead even';

  return (
    <Widget title="§3 · The Iterated Playground" id={PLAYGROUND_ID}>
      <p className="mb-4 text-sm text-ink-soft">
        Now you play the <strong>same</strong> person again and again. Pick an opponent and feel how
        the long game changes the maths.
      </p>

      {/* Opponent picker */}
      <div className="mb-4 flex flex-wrap gap-1.5">
        {STRATEGIES.map((s) => (
          <button
            key={s.id}
            type="button"
            onClick={() => setOppId(s.id)}
            aria-pressed={s.id === oppId}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{
              borderColor: s.id === oppId ? 'var(--color-accent)' : 'var(--color-line)',
              color: s.id === oppId ? 'var(--color-accent)' : 'var(--color-ink-soft)',
            }}
          >
            {s.name}
          </button>
        ))}
      </div>

      <p className="mb-4 text-xs text-muted">{opponent.short}</p>

      {/* Scoreboard */}
      <div className="mb-4 grid grid-cols-2 gap-3">
        <Score label="You" value={myScore} delta={lastDelta?.[0]} color={MOVE_COLOR.C} />
        <Score label={opponent.name} value={oppScore} delta={lastDelta?.[1]} color={MOVE_COLOR.D} />
      </div>

      {/* Tapes */}
      <div className="mb-4 flex flex-col gap-2 overflow-hidden">
        <MoveTape label="You" moves={mine} highlightLast />
        <MoveTape label={opponent.name} moves={opp} highlightLast />
      </div>

      {/* Controls */}
      {!done ? (
        <div className="flex gap-3">
          <ChoiceButton
            move="C"
            onClick={() => play('C')}
            sub={`Round ${mine.length + 1}/${rounds}`}
          />
          <ChoiceButton
            move="D"
            onClick={() => play('D')}
            sub={`Round ${mine.length + 1}/${rounds}`}
          />
        </div>
      ) : (
        <div className="flex flex-wrap items-center justify-between gap-3 border-2 border-line p-3">
          <span className="font-display text-lg">
            {winner} — {myScore} vs {oppScore}
          </span>
          <button
            type="button"
            onClick={() => reset()}
            className="border border-line-strong px-3 py-1.5 font-mono text-[0.7rem] tracking-[0.12em] uppercase transition-colors hover:border-accent hover:text-accent"
          >
            Rematch
          </button>
        </div>
      )}

      {/* Rounds config */}
      <label className="mt-4 flex items-center gap-3 text-sm">
        <span className="label shrink-0">Rounds</span>
        <input
          type="range"
          min={5}
          max={30}
          step={1}
          value={rounds}
          onChange={(e) => setRounds(Number(e.target.value))}
          className="min-w-0 flex-1 accent-[var(--color-accent)]"
        />
        <span className="font-mono tabular-nums">{rounds}</span>
      </label>
    </Widget>
  );
}

function Score({
  label,
  value,
  delta,
  color,
}: {
  label: string;
  value: number;
  delta?: number;
  color: string;
}) {
  return (
    <div className="flex items-center justify-between border-2 border-line px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="h-3 w-3" style={{ background: color }} />
        <span className="label">{label}</span>
      </div>
      <div className="flex items-baseline gap-1.5">
        <Ticker value={value} className="font-mono text-2xl font-semibold tabular-nums" />
        {delta !== undefined && (
          <span key={value} className="pd-fly-in font-mono text-sm text-muted">
            +{delta}
          </span>
        )}
      </div>
    </div>
  );
}
