import { useState } from 'react';
import { AGENTS, Widget, useTick, type Agent } from './shared';

type Mode = 'fugu' | 'trinity';

// Illustrative selection logits for the example query (a softmax over the worker pool).
const LOGITS: Record<string, number> = { gpt: 0.52, claude: 0.33, gemini: 0.15 };
const ROLES = [
  { k: 'T', name: 'Thinker', desc: 'plans & decomposes' },
  { k: 'W', name: 'Worker', desc: 'does the concrete work' },
  { k: 'V', name: 'Verifier', desc: 'accepts or asks for a revision' },
];

// The multi-turn example from the TRINITY paper (Figure 1, depreciation problem).
const TURNS = [
  {
    role: 'T',
    agent: 'gemini',
    text: 'Decompose: find the double-declining-balance rate, then the year-2 expense.',
  },
  {
    role: 'W',
    agent: 'gpt',
    text: 'Straight-line rate = 1/8; DDB rate = 2×; year-2 expense = $2,812.50.',
  },
  {
    role: 'V',
    agent: 'claude',
    text: 'Checks the figure and edge cases → ACCEPT, so the loop halts.',
  },
];

export default function TrinityDiagram() {
  const [mode, setMode] = useState<Mode>('fugu');
  const isTrinity = mode === 'trinity';
  const step = useTick(3, 1100); // pipeline cursor: 0 query · 1 backbone · 2 head · 3 pick
  const turn = useTick(TURNS.length, 1300, isTrinity);
  const top = AGENTS.reduce((a, b) => (LOGITS[b.id] > LOGITS[a.id] ? b : a));

  return (
    <Widget
      title="Fugu’s orchestrator — the TRINITY parametrization"
      kicker="it outputs a choice, not an answer"
    >
      <div className="mb-4 flex flex-wrap gap-2">
        {(['fugu', 'trinity'] as Mode[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            aria-pressed={mode === m}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{
              borderColor: mode === m ? 'var(--color-accent)' : 'var(--color-line)',
              color: mode === m ? 'var(--color-accent)' : 'var(--color-ink-soft)',
            }}
          >
            {m === 'fugu' ? 'Fugu — pick a worker' : 'TRINITY — pick a worker + role'}
          </button>
        ))}
      </div>

      {/* Forward-pass pipeline */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-stretch">
        <Stage on={step >= 0} label="1 · Query">
          <code className="block font-mono text-[0.74rem] leading-snug text-ink-soft">
            “Write me code for binary search”
          </code>
        </Stage>

        <Chevron />

        <Stage on={step >= 1} label="2 · Small LM (≈0.6B)">
          <div className="flex flex-col gap-0.5" aria-hidden>
            {[0, 1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-2 transition-all"
                style={{
                  background: step >= 1 ? 'var(--color-fugu)' : 'var(--color-line-strong)',
                  opacity: step >= 1 ? 0.4 + i * 0.16 : 0.4,
                }}
              />
            ))}
          </div>
          <span className="mt-1.5 block font-mono text-[0.6rem] text-muted">
            reads the query and turns it into an internal vector — its{' '}
            <span className="whitespace-nowrap">“hidden state” h</span>
          </span>
        </Stage>

        <Chevron />

        <Stage on={step >= 2} label="3 · Trained head">
          <span
            className="font-mono text-[0.74rem] font-semibold"
            style={{ color: 'var(--color-accent)' }}
          >
            ≈10K params
          </span>
          <span className="mt-1 block font-mono text-[0.62rem] text-muted">
            the <em>only</em> part trained — turns the hidden state into a score for every worker
            {isTrinity ? ' and every role' : ''}
          </span>
        </Stage>

        <Chevron />

        <Stage on={step >= 3} label="4 · Pick">
          <div className="flex flex-col gap-1">
            {AGENTS.map((ag) => (
              <LogitBar
                key={ag.id}
                agent={ag}
                value={LOGITS[ag.id]}
                fill={step >= 2}
                selected={step >= 3 && ag.id === top.id}
              />
            ))}
          </div>
          {isTrinity && (
            <div className="mt-2">
              <span className="font-mono text-[0.58rem] text-muted">role:</span>
              <div className="mt-0.5 flex gap-1">
                {ROLES.map((r, i) => (
                  <span
                    key={r.k}
                    className="flex-1 border py-0.5 text-center font-mono text-[0.62rem] transition-all"
                    style={{
                      borderColor:
                        step >= 3 && i === 0 ? 'var(--color-accent)' : 'var(--color-line)',
                      background: step >= 3 && i === 0 ? 'var(--color-accent)' : 'transparent',
                      color: step >= 3 && i === 0 ? 'var(--color-paper)' : 'var(--color-muted)',
                    }}
                    title={`${r.name} — ${r.desc}`}
                  >
                    {r.k}
                  </span>
                ))}
              </div>
            </div>
          )}
        </Stage>
      </div>

      <p className="mt-3 text-xs text-ink-soft">
        The orchestrator never writes the answer — its own text output is thrown away. Because it
        only needs to emit a choice, it can decide almost instantly instead of generating a full
        response, which keeps Fugu’s latency “comparable to a direct call to a frontier model.”{' '}
        {isTrinity
          ? 'TRINITY goes one step further: it gives the picked worker a role and loops over several turns.'
          : 'Fugu keeps it minimal — one worker, no roles, dispatched at once.'}
      </p>

      {/* TRINITY multi-turn role loop */}
      {isTrinity && (
        <div className="mt-3 border-t border-line pt-3">
          <span className="label">Multi-turn coordination — one worker, one role, per turn</span>
          <div className="mt-2 flex flex-col gap-1.5">
            {TURNS.map((t, i) => {
              const ag = AGENTS.find((a) => a.id === t.agent)!;
              const role = ROLES.find((r) => r.k === t.role)!;
              const on = turn >= i;
              return (
                <div
                  key={i}
                  className="flex items-start gap-2 border px-2 py-1.5 transition-all"
                  style={{
                    borderColor: on ? ag.color : 'var(--color-line)',
                    opacity: on ? 1 : 0.35,
                  }}
                >
                  <span
                    className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center font-mono text-[0.7rem] font-bold text-paper"
                    style={{ background: ag.color }}
                    title={role.name}
                  >
                    {t.role}
                  </span>
                  <span className="text-[0.72rem] leading-snug">
                    <span className="font-mono" style={{ color: ag.color }}>
                      {role.name} · {ag.name}
                    </span>
                    <span className="text-ink-soft"> — {t.text}</span>
                  </span>
                </div>
              );
            })}
          </div>
          <p className="mt-2 font-mono text-[0.62rem] text-muted">
            Halts the moment a Verifier accepts (or the turn budget runs out).
          </p>
        </div>
      )}
    </Widget>
  );
}

function Stage({
  on,
  label,
  accent,
  children,
}: {
  on: boolean;
  label: string;
  accent?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className="flex flex-1 flex-col border bg-paper p-2.5 transition-opacity"
      style={{
        borderColor: accent ? 'var(--color-accent)' : 'var(--color-line)',
        opacity: on ? 1 : 0.5,
      }}
    >
      <span className="label mb-1.5">{label}</span>
      {children}
    </div>
  );
}

function Chevron() {
  return (
    <div className="flex items-center justify-center text-muted" aria-hidden>
      <span className="hidden sm:block">→</span>
      <span className="block sm:hidden">↓</span>
    </div>
  );
}

function LogitBar({
  agent,
  value,
  fill,
  selected,
}: {
  agent: Agent;
  value: number;
  fill: boolean;
  selected: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="w-12 shrink-0 font-mono text-[0.62rem]" style={{ color: agent.color }}>
        {agent.short}
      </span>
      <div className="h-3.5 flex-1 border border-line bg-paper">
        <div
          className="h-full transition-[width] duration-500"
          style={{
            width: fill ? `${value * 100}%` : '0%',
            background: agent.color,
            outline: selected ? '2px solid var(--color-accent)' : 'none',
            outlineOffset: -2,
          }}
        />
      </div>
    </div>
  );
}
