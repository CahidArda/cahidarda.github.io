import { useState } from 'react';
import { AGENTS, Widget, useTick, type Agent } from './shared';

type Mode = 'fugu' | 'trinity';

// Illustrative selection logits for the example query (softmax over the worker pool).
const LOGITS: Record<string, number> = { gpt: 0.52, claude: 0.33, gemini: 0.15 };
const ROLES = [
  { k: 'T', name: 'Thinker', desc: 'strategises — plans & decomposes' },
  { k: 'W', name: 'Worker', desc: 'executes — concrete problem-solving' },
  { k: 'V', name: 'Verifier', desc: 'evaluates — accept or revise' },
];

// The multi-turn example from the TRINITY paper (Figure 1, depreciation problem).
const TURNS = [
  {
    role: 'T',
    agent: 'gemini',
    text: 'Decompose: compute the double-declining-balance rate, then year-2 expense.',
  },
  {
    role: 'W',
    agent: 'gpt',
    text: 'Straight-line rate = 1/8; DDB rate = 2×; year-2 expense = $2,812.50.',
  },
  { role: 'V', agent: 'claude', text: 'Checks the figure and edge cases. uₖ = ACCEPT → halt.' },
];

export default function TrinityDiagram() {
  const [mode, setMode] = useState<Mode>('fugu');
  const isTrinity = mode === 'trinity';
  const step = useTick(3, 1100); // 0 input · 1 backbone+hidden · 2 head/logits · 3 select
  const turn = useTick(TURNS.length, 1300, isTrinity);

  const top = AGENTS.reduce((a, b) => (LOGITS[b.id] > LOGITS[a.id] ? b : a));

  return (
    <Widget title="Fugu’s orchestrator — the TRINITY parametrization" kicker="logits, not text">
      <div className="mb-3 flex gap-2">
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
            {m === 'fugu' ? 'Fugu — select only' : 'TRINITY — select + role'}
          </button>
        ))}
      </div>

      {/* Forward pass */}
      <div className="grid items-stretch gap-2 sm:grid-cols-[1.1fr_1.2fr_1.4fr]">
        {/* Input + backbone */}
        <Stage on={step >= 0} label="Input">
          <code className="block font-mono text-[0.72rem] leading-snug text-ink-soft">
            “Write me code for binary search”
          </code>
          <div className="mt-2 flex flex-col gap-0.5" aria-hidden>
            {[0, 1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-2 rounded-sm transition-all"
                style={{
                  background: step >= 1 ? 'var(--color-fugu)' : 'var(--color-line-strong)',
                  opacity: step >= 1 ? 0.35 + i * 0.16 : 0.4,
                }}
              />
            ))}
          </div>
          <span className="mt-1 block font-mono text-[0.6rem] text-muted">
            ≈0.6B SLM backbone → hidden state h ∈ ℝᵈ
          </span>
        </Stage>

        {/* Two heads */}
        <Stage on={step >= 2} label="Heads (parallel)">
          <div className="flex flex-col gap-2">
            <div className="border border-line px-2 py-1.5 opacity-60">
              <span className="font-mono text-[0.66rem] text-muted">LM head → text</span>
              <span className="block font-mono text-[0.58rem] text-muted line-through">
                discarded
              </span>
            </div>
            <div
              className="border-2 px-2 py-1.5 transition-all"
              style={{ borderColor: step >= 2 ? 'var(--color-accent)' : 'var(--color-line)' }}
            >
              <span className="font-mono text-[0.66rem]" style={{ color: 'var(--color-accent)' }}>
                lightweight head
              </span>
              <span className="block font-mono text-[0.58rem] text-muted">
                ≈10K params → {isTrinity ? 'L + 3' : 'L'} logits
              </span>
            </div>
          </div>
        </Stage>

        {/* Logits + selection */}
        <Stage on={step >= 2} label="Worker logits">
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
            <div className="mt-2 flex gap-1">
              {ROLES.map((r, i) => (
                <span
                  key={r.k}
                  className="flex-1 border px-1 py-0.5 text-center font-mono text-[0.62rem] transition-all"
                  style={{
                    borderColor: step >= 3 && i === 0 ? 'var(--color-accent)' : 'var(--color-line)',
                    background: step >= 3 && i === 0 ? 'var(--color-accent)' : 'transparent',
                    color: step >= 3 && i === 0 ? 'var(--color-paper)' : 'var(--color-muted)',
                  }}
                  title={r.desc}
                >
                  {r.k}
                </span>
              ))}
            </div>
          )}
        </Stage>
      </div>

      <p className="mt-3 text-xs text-ink-soft">
        The orchestrator never writes the answer — it reads its own hidden state and emits a
        selection. “Since prompting and task execution are delegated to the selected frontier
        worker, the orchestrator only needs to produce a worker-selection decision,” which is what
        keeps Fugu’s latency “comparable to a direct call to a frontier model.”
      </p>

      {/* TRINITY multi-turn role loop */}
      {isTrinity && (
        <div className="mt-4 border-t border-line pt-3">
          <span className="label">Multi-turn coordination — one role per turn</span>
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
            Halts when a Verifier ACCEPTs, or the turn budget is spent.
          </p>
        </div>
      )}
    </Widget>
  );
}

function Stage({ on, label, children }: { on: boolean; label: string; children: React.ReactNode }) {
  return (
    <div
      className="flex flex-col border border-line bg-paper p-2.5 transition-opacity"
      style={{ opacity: on ? 1 : 0.55 }}
    >
      <span className="label mb-1.5">{label}</span>
      {children}
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
      <span
        className="w-14 shrink-0 truncate font-mono text-[0.62rem]"
        style={{ color: agent.color }}
      >
        {agent.short}
      </span>
      <div className="h-3.5 flex-1 border border-line bg-paper">
        <div
          className="h-full transition-[width] duration-500"
          style={{
            width: fill ? `${value * 100}%` : '0%',
            background: agent.color,
            outline: selected ? '2px solid var(--color-accent)' : 'none',
          }}
        />
      </div>
    </div>
  );
}
