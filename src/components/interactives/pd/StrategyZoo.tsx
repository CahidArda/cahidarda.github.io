import { useEffect, useMemo, useState } from 'react';
import { Glyph, Widget, useReducedMotion, type Move } from './shared';
import { mulberry32, nextMove } from './engine';
import { STRATEGIES, type Strategy } from './strategies';
import { PLAY_EVENT } from './IteratedPlayground';

// Canned opponent scripts chosen to make each rule obvious in a few cells.
const SCRIPT: Record<string, Move[]> = {
  tft: ['C', 'D', 'D', 'C', 'C'],
  alld: ['C', 'C', 'D', 'C', 'D'],
  allc: ['C', 'D', 'D', 'C', 'C'],
  grudger: ['C', 'C', 'D', 'C', 'C'],
  pavlov: ['C', 'C', 'D', 'D', 'C'],
  random: ['C', 'D', 'C', 'D', 'C'],
};

function ruleTape(s: Strategy): { opp: Move[]; me: Move[] } {
  const opp = SCRIPT[s.id] ?? ['C', 'D', 'C', 'D'];
  const rng = mulberry32(7);
  const me: Move[] = [];
  for (let i = 0; i < opp.length; i++) {
    me.push(nextMove(s, me, opp.slice(0, i), rng));
  }
  return { opp, me };
}

function MiniRule({ s }: { s: Strategy }) {
  const reduced = useReducedMotion();
  const { opp, me } = useMemo(() => ruleTape(s), [s]);
  const [step, setStep] = useState(reduced ? opp.length : 0);

  useEffect(() => {
    if (reduced) {
      setStep(opp.length);
      return;
    }
    let n = 0;
    const id = setInterval(() => {
      n = n >= opp.length ? 0 : n + 1;
      setStep(n);
    }, 520);
    return () => clearInterval(id);
  }, [opp.length, reduced]);

  return (
    <div className="flex flex-col gap-1.5 border border-line bg-paper px-2.5 py-2">
      <Row label="In" moves={opp} step={step} />
      <Row label="Out" moves={me} step={step} accent />
    </div>
  );
}

function Row({
  label,
  moves,
  step,
  accent,
}: {
  label: string;
  moves: Move[];
  step: number;
  accent?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="label" style={{ width: '2rem' }}>
        {label}
      </span>
      <div className="flex gap-1">
        {moves.map((m, i) => {
          const on = i < step;
          return on ? (
            <Glyph
              key={i}
              move={m}
              size={18}
              className={accent && i === step - 1 ? 'pd-pop' : ''}
            />
          ) : (
            <span
              key={i}
              style={{ width: 18, height: 18, border: '1px solid var(--color-line)', opacity: 0.5 }}
            />
          );
        })}
      </div>
    </div>
  );
}

export default function StrategyZoo() {
  function playAgainst(id: string) {
    window.dispatchEvent(new CustomEvent(PLAY_EVENT, { detail: { strategyId: id } }));
  }

  return (
    <Widget title="§4 · Strategy Zoo">
      <p className="mb-4 text-sm text-ink-soft">
        A handful of simple rules. Each is just a reflex.
      </p>
      <div
        className="grid gap-3"
        style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))' }}
      >
        {STRATEGIES.map((s) => (
          <div
            key={s.id}
            className="flex flex-col gap-2 border border-line-strong bg-paper-raised p-3"
          >
            <span
              className="font-display text-base leading-tight"
              style={{ color: 'var(--color-ink)' }}
            >
              {s.name}
            </span>
            <MiniRule s={s} />
            <p className="min-h-[2.5rem] text-xs leading-snug text-ink-soft">{s.short}</p>
            <button
              type="button"
              onClick={() => playAgainst(s.id)}
              className="mt-auto border border-line-strong px-2 py-1.5 font-mono text-[0.66rem] tracking-[0.1em] uppercase transition-colors hover:border-accent hover:text-accent"
            >
              Play against this →
            </button>
          </div>
        ))}
      </div>
    </Widget>
  );
}
