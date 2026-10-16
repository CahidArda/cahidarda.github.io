import { useMemo, useState } from 'react';
import { Glyph, MOVE_COLOR, MOVE_COLOR_SOFT, Widget, type Move } from './shared';
import { CLASSIC, payoff, validatePD, type Payoffs } from './strategies';

type CellKey = `${Move}${Move}`; // your move + their move
const SYMBOL: Record<CellKey, [keyof Payoffs, keyof Payoffs]> = {
  CC: ['R', 'R'],
  CD: ['S', 'T'],
  DC: ['T', 'S'],
  DD: ['P', 'P'],
};

function Slider({
  name,
  desc,
  value,
  onChange,
}: {
  name: keyof Payoffs;
  desc: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center gap-3">
      <span className="font-mono text-sm font-semibold" style={{ width: '1.2rem' }}>
        {name}
      </span>
      <input
        type="range"
        min={0}
        max={10}
        step={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="min-w-0 flex-1 accent-[var(--color-accent)]"
        aria-label={`${name} — ${desc}`}
      />
      <span className="font-mono text-sm tabular-nums" style={{ width: '1.2rem' }}>
        {value}
      </span>
    </label>
  );
}

export default function PayoffMatrixExplorer() {
  const [p, setP] = useState<Payoffs>(CLASSIC);
  const [hover, setHover] = useState<CellKey | null>(null);
  const v = useMemo(() => validatePD(p), [p]);
  const set = (k: keyof Payoffs) => (val: number) => setP((prev) => ({ ...prev, [k]: val }));

  const moves: Move[] = ['C', 'D'];

  return (
    <Widget title="§1 · Payoff Matrix Explorer">
      <p className="mb-4 text-sm text-ink-soft">
        Both players choose at the same time. Rows are <strong>your</strong> move, columns are{' '}
        <strong>theirs</strong>. Each cell shows{' '}
        <span style={{ color: 'var(--color-ink)' }}>your payoff</span> /{' '}
        <span className="text-muted">theirs</span>.
      </p>

      <div className="grid gap-5 md:grid-cols-[1fr_auto]">
        {/* Matrix */}
        <div
          className="grid select-none gap-1"
          style={{ gridTemplateColumns: 'auto 1fr 1fr', maxWidth: 360 }}
        >
          <div />
          {moves.map((m) => (
            <div key={m} className="flex flex-col items-center gap-1 pb-1">
              <span className="label">They</span>
              <Glyph move={m} size={22} />
            </div>
          ))}

          {moves.map((my) => (
            <FragmentRow key={my}>
              <div className="flex flex-col items-center justify-center gap-1 pr-1">
                <span className="label">You</span>
                <Glyph move={my} size={22} />
              </div>
              {moves.map((their) => {
                const key = `${my}${their}` as CellKey;
                const [ys, ts] = SYMBOL[key];
                const mine = payoff(my, their, p);
                const theirs = payoff(their, my, p);
                const active = hover === key;
                return (
                  <button
                    key={their}
                    type="button"
                    onMouseEnter={() => setHover(key)}
                    onMouseLeave={() => setHover(null)}
                    onFocus={() => setHover(key)}
                    onBlur={() => setHover(null)}
                    onClick={() => setHover(active ? null : key)}
                    className="flex aspect-square flex-col items-center justify-center border-2 transition-all"
                    style={{
                      borderColor: active ? 'var(--color-accent)' : 'var(--color-line)',
                      background: active
                        ? my === their
                          ? MOVE_COLOR_SOFT[my]
                          : 'var(--color-paper)'
                        : 'var(--color-paper)',
                    }}
                  >
                    <span className="flex items-baseline gap-1 font-mono text-2xl font-semibold">
                      <span style={{ color: 'var(--color-ink)' }}>{mine}</span>
                      <span className="text-muted text-base">/{theirs}</span>
                    </span>
                    <span
                      className="label mt-0.5"
                      style={{ color: active ? 'var(--color-accent)' : undefined }}
                    >
                      {ys}/{ts}
                    </span>
                  </button>
                );
              })}
            </FragmentRow>
          ))}
        </div>

        {/* Sliders + validity */}
        <div className="flex flex-col gap-2.5 md:w-56">
          <Slider
            name="T"
            desc="Temptation: defect on a cooperator"
            value={p.T}
            onChange={set('T')}
          />
          <Slider name="R" desc="Reward: both cooperate" value={p.R} onChange={set('R')} />
          <Slider name="P" desc="Punishment: both defect" value={p.P} onChange={set('P')} />
          <Slider
            name="S"
            desc="Sucker: cooperate against a defector"
            value={p.S}
            onChange={set('S')}
          />

          <div
            className="mt-2 border-2 px-3 py-2.5 transition-colors"
            style={{
              borderColor: v.ok ? MOVE_COLOR.C : 'var(--color-accent)',
              background: v.ok ? MOVE_COLOR_SOFT.C : 'transparent',
            }}
          >
            <div className="flex items-center gap-2">
              <span
                className="inline-flex h-5 w-5 items-center justify-center text-xs font-bold text-paper"
                style={{ background: v.ok ? MOVE_COLOR.C : 'var(--color-accent)' }}
              >
                {v.ok ? '✓' : '✕'}
              </span>
              <span className="font-mono text-xs font-semibold tracking-wide uppercase">
                {v.ok ? 'Valid Prisoner’s Dilemma' : 'Broken'}
              </span>
            </div>
            <p className="mt-1.5 text-xs leading-snug text-ink-soft">{v.message}</p>
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 font-mono text-xs">
              <Check ok={p.T > p.R && p.R > p.P && p.P > p.S}>
                {p.T}&gt;{p.R}&gt;{p.P}&gt;{p.S}
              </Check>
              <Check ok={2 * p.R > p.T + p.S}>
                2·{p.R}&gt;{p.T}+{p.S}
              </Check>
            </div>
          </div>
        </div>
      </div>
    </Widget>
  );
}

function Check({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return (
    <span style={{ color: ok ? MOVE_COLOR.C : 'var(--color-d)' }}>
      {ok ? '✓' : '✕'} {children}
    </span>
  );
}

// Fragment that keeps grid children flat (each row is two cells in the parent grid).
function FragmentRow({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
