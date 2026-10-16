import { useState } from 'react';
import {
  ChoiceButton,
  Glyph,
  MOVE_COLOR_SOFT,
  MOVE_LABEL,
  Widget,
  useReducedMotion,
  type Move,
} from './shared';
import { CLASSIC, payoff } from './strategies';

export default function PlayOnce() {
  const reduced = useReducedMotion();
  const [you, setYou] = useState<Move | null>(null);
  const [opp, setOpp] = useState<Move | null>(null);
  const [revealed, setRevealed] = useState(false);
  // Which opponent column the dominance panel is showing (null = their actual move).
  const [flip, setFlip] = useState(false);

  function choose(m: Move) {
    const o: Move = Math.random() < 0.5 ? 'C' : 'D';
    setYou(m);
    setOpp(o);
    setFlip(false);
    if (reduced) {
      setRevealed(true);
    } else {
      setRevealed(false);
      setTimeout(() => setRevealed(true), 260);
    }
  }

  function reset() {
    setYou(null);
    setOpp(null);
    setRevealed(false);
    setFlip(false);
  }

  const p = CLASSIC;
  // The opponent column the dominance panel currently shows.
  const shownOpp: Move | null = opp ? (flip ? (opp === 'C' ? 'D' : 'C') : opp) : null;

  return (
    <Widget title="§2 · Play It Once">
      {!you && (
        <>
          <p className="mb-4 text-sm text-ink-soft">You and a stranger, once, no future. Choose.</p>
          <div className="flex gap-3">
            <ChoiceButton move="C" onClick={() => choose('C')} sub="Trust them" />
            <ChoiceButton move="D" onClick={() => choose('D')} sub="Betray them" />
          </div>
        </>
      )}

      {you && (
        <div className="flex flex-col gap-5">
          {/* Reveal */}
          <div className="grid grid-cols-2 gap-3">
            <Reveal label="You" move={you} score={revealed && opp ? payoff(you, opp, p) : null} />
            <Reveal
              label="Stranger"
              move={revealed ? opp : null}
              score={revealed && opp ? payoff(opp, you, p) : null}
            />
          </div>

          {revealed && opp && (
            <>
              {/* Dominance panel */}
              <div className="border border-line p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="label">Your payoff if they play</span>
                  <button
                    type="button"
                    onClick={() => setFlip((f) => !f)}
                    className="border border-line-strong px-2.5 py-1 font-mono text-[0.7rem] tracking-[0.12em] uppercase transition-colors hover:border-accent hover:text-accent"
                  >
                    What if they’d done the opposite?
                  </button>
                </div>

                <div className="mt-3 flex items-center gap-4">
                  <div className="flex flex-col items-center gap-1">
                    <span className="label">They</span>
                    <Glyph move={shownOpp!} size={26} />
                  </div>
                  <div className="grid flex-1 gap-2" style={{ maxWidth: 320 }}>
                    {(['D', 'C'] as Move[]).map((my) => {
                      const val = payoff(my, shownOpp!, p);
                      const other = payoff(my === 'D' ? 'C' : 'D', shownOpp!, p);
                      const wins = val >= other;
                      return (
                        <div
                          key={my}
                          className={`flex items-center gap-3 border-2 px-3 py-2 transition-all ${wins && my === 'D' ? 'pd-flash' : ''}`}
                          style={{
                            borderColor:
                              wins && my === 'D' ? 'var(--color-accent)' : 'var(--color-line)',
                            background: wins && my === 'D' ? MOVE_COLOR_SOFT[my] : 'transparent',
                          }}
                        >
                          <Glyph move={my} size={22} />
                          <span className="text-sm text-ink-soft">
                            You {MOVE_LABEL[my].toLowerCase()}
                          </span>
                          <span className="ml-auto font-mono text-xl font-semibold">{val}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <p className="mt-3 text-sm text-ink-soft">
                  Whatever the stranger does — try the toggle — defecting pays you more.{' '}
                  <strong style={{ color: 'var(--color-ink)' }}>Defect dominates.</strong> So the
                  only Nash equilibrium is mutual defection (1, 1), worse for both than mutual
                  cooperation (3, 3).
                </p>
              </div>

              <button
                type="button"
                onClick={reset}
                className="self-start border border-line-strong px-3 py-1.5 font-mono text-[0.7rem] tracking-[0.12em] uppercase transition-colors hover:border-accent hover:text-accent"
              >
                Play again
              </button>
            </>
          )}
        </div>
      )}
    </Widget>
  );
}

function Reveal({
  label,
  move,
  score,
}: {
  label: string;
  move: Move | null;
  score: number | null;
}) {
  return (
    <div className="flex flex-col items-center gap-2 border border-line bg-paper py-4">
      <span className="label">{label}</span>
      {move ? (
        <span className="pd-pop">
          <Glyph move={move} size={44} />
        </span>
      ) : (
        <span
          className="inline-flex items-center justify-center font-mono text-2xl text-muted"
          style={{ width: 44, height: 44, border: '2px dashed var(--color-line-strong)' }}
        >
          ?
        </span>
      )}
      <span
        className="font-mono text-2xl font-semibold tabular-nums"
        style={{ minHeight: '1.6rem' }}
      >
        {score !== null ? <span className="pd-fly-in inline-block">+{score}</span> : ''}
      </span>
    </div>
  );
}
