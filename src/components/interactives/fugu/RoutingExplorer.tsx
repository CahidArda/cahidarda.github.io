import { useState } from 'react';
import { AGENTS, Widget } from './shared';

// Routing mixes illustrative of the trends reported in the technical report (§4.2,
// Figure 5): the orchestrator's deployed distribution peaks on whichever worker is
// SOTA for the domain. Values are schematic, not exact figure read-outs.
interface Domain {
  id: string;
  label: string;
  note: string;
  mix: Record<string, number>; // gpt / claude / gemini, sums to 1
}

const DOMAINS: Domain[] = [
  {
    id: 'terminal',
    label: 'Terminal Bench',
    note: 'Agentic terminal coding — GPT-5.5 is SOTA, so routing peaks on GPT.',
    mix: { gpt: 0.6, claude: 0.29, gemini: 0.11 },
  },
  {
    id: 'swe',
    label: 'SWE-Bench Pro',
    note: 'Long-horizon software engineering — Opus’ specialty leads the mix.',
    mix: { gpt: 0.34, claude: 0.55, gemini: 0.11 },
  },
  {
    id: 'gpqa',
    label: 'GPQA-Diamond',
    note: 'Graduate science — Gemini is the leading model, so Fugu focuses on Gemini.',
    mix: { gpt: 0.16, claude: 0.24, gemini: 0.6 },
  },
  {
    id: 'math',
    label: 'Math',
    note: 'Math questions are “predominantly handled by GPT-5.5”.',
    mix: { gpt: 0.62, claude: 0.16, gemini: 0.22 },
  },
  {
    id: 'chembio',
    label: 'Chemistry / Biology',
    note: 'Routed mostly to Gemini, “mirroring its SOTA scientific capabilities”.',
    mix: { gpt: 0.18, claude: 0.2, gemini: 0.62 },
  },
  {
    id: 'hle',
    label: "Humanity's Last Exam",
    note: 'Multidisciplinary by nature — a “highly balanced distribution” over all three.',
    mix: { gpt: 0.35, claude: 0.32, gemini: 0.33 },
  },
];

export default function RoutingExplorer() {
  const [domainId, setDomainId] = useState('terminal');
  const domain = DOMAINS.find((d) => d.id === domainId)!;
  const top = AGENTS.reduce((a, b) => ((domain.mix[b.id] ?? 0) > (domain.mix[a.id] ?? 0) ? b : a));

  return (
    <Widget title="Domain adaptivity" kicker="route a query, watch the mix shift">
      <p className="mb-3 text-sm text-ink-soft">
        Pick a domain. The orchestrator’s routing distribution re-weights toward whichever worker is
        strongest there — “a hallmark feature of an intelligent orchestrator.”
      </p>

      <div className="mb-4 flex flex-wrap gap-1.5">
        {DOMAINS.map((d) => (
          <button
            key={d.id}
            type="button"
            onClick={() => setDomainId(d.id)}
            aria-pressed={d.id === domainId}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{
              borderColor: d.id === domainId ? 'var(--color-accent)' : 'var(--color-line)',
              color: d.id === domainId ? 'var(--color-accent)' : 'var(--color-ink-soft)',
            }}
          >
            {d.label}
          </button>
        ))}
      </div>

      <div className="flex flex-col gap-2.5">
        {AGENTS.map((ag) => {
          const pct = Math.round((domain.mix[ag.id] ?? 0) * 100);
          return (
            <div key={ag.id} className="flex items-center gap-3">
              <span className="w-28 shrink-0 font-mono text-xs" style={{ color: ag.color }}>
                {ag.name}
              </span>
              <div className="h-6 flex-1 border border-line bg-paper">
                <div
                  className="flex h-full items-center justify-end pr-2 transition-[width] duration-500 ease-out"
                  style={{ width: `${Math.max(pct, 3)}%`, background: ag.color }}
                >
                  {pct >= 12 && (
                    <span className="font-mono text-[0.7rem] font-semibold text-paper">{pct}%</span>
                  )}
                </div>
              </div>
              {pct < 12 && <span className="w-8 font-mono text-[0.7rem] text-muted">{pct}%</span>}
            </div>
          );
        })}
      </div>

      <p className="mt-4 border-l-2 pl-3 text-sm text-ink-soft" style={{ borderColor: top.color }}>
        <strong style={{ color: 'var(--color-ink)' }}>{top.name}</strong> leads here. {domain.note}
      </p>
    </Widget>
  );
}
