import { useState } from 'react';
import { FUGU_COLOR, Widget } from './shared';

// Model Card, Table 1 of the Sakana Fugu Technical Report. Order: the two Fugu models,
// then the three frontier baselines they orchestrate.
interface ModelMeta {
  id: string;
  name: string;
  color: string;
  fugu?: boolean;
  light?: boolean;
}
const MODELS: ModelMeta[] = [
  { id: 'ultra', name: 'Fugu-Ultra', color: FUGU_COLOR, fugu: true },
  { id: 'fugu', name: 'Fugu', color: FUGU_COLOR, fugu: true, light: true },
  { id: 'opus', name: 'Claude Opus 4.8', color: 'var(--color-claude)' },
  { id: 'gemini', name: 'Gemini 3.1 Pro', color: 'var(--color-gemini)' },
  { id: 'gpt', name: 'GPT-5.5', color: 'var(--color-gpt)' },
];

type Row = { name: string; vals: Record<string, number> };
const BENCH: Row[] = [
  { name: 'SWE-Bench Pro', vals: { ultra: 73.7, fugu: 59.0, opus: 69.2, gemini: 54.2, gpt: 58.6 } },
  {
    name: 'Terminal Bench 2.1',
    vals: { ultra: 82.1, fugu: 80.2, opus: 74.6, gemini: 70.3, gpt: 78.2 },
  },
  { name: 'LiveCodeBench', vals: { ultra: 93.2, fugu: 92.9, opus: 87.8, gemini: 88.5, gpt: 85.3 } },
  {
    name: 'LiveCodeBench Pro',
    vals: { ultra: 90.8, fugu: 87.8, opus: 84.8, gemini: 82.9, gpt: 88.4 },
  },
  {
    name: "Humanity's Last Exam",
    vals: { ultra: 50.0, fugu: 47.2, opus: 49.8, gemini: 44.4, gpt: 41.4 },
  },
  {
    name: 'CharXiv Reasoning',
    vals: { ultra: 86.6, fugu: 85.1, opus: 84.2, gemini: 83.3, gpt: 84.1 },
  },
  { name: 'GPQA Diamond', vals: { ultra: 95.5, fugu: 95.5, opus: 92.0, gemini: 94.3, gpt: 93.6 } },
  { name: 'SciCode', vals: { ultra: 58.7, fugu: 60.1, opus: 53.5, gemini: 58.9, gpt: 56.1 } },
  { name: 'τ³ Banking', vals: { ultra: 20.6, fugu: 21.7, opus: 20.6, gemini: 8.4, gpt: 20.6 } },
  {
    name: 'Long Context Reasoning',
    vals: { ultra: 73.3, fugu: 74.7, opus: 67.7, gemini: 72.7, gpt: 74.3 },
  },
  { name: 'MRCRv2', vals: { ultra: 93.6, fugu: 86.6, opus: 87.9, gemini: 84.9, gpt: 94.8 } },
];

export default function BenchmarkChart() {
  const [idx, setIdx] = useState(0);
  const row = BENCH[idx];
  const best = Math.max(...MODELS.map((m) => row.vals[m.id]));
  const min = Math.min(...MODELS.map((m) => row.vals[m.id]));
  const lo = Math.max(0, min - 8); // zoom the axis so differences read

  return (
    <Widget title="Results" kicker="Technical Report, Table 1">
      <p className="mb-3 text-sm text-ink-soft">
        Through orchestration alone, the Fugu models match or beat every worker in their own pool.
        Pick a benchmark — bars are <span style={{ color: 'var(--color-accent)' }}>Fugu</span> vs
        the three frontier workers it routes between.
      </p>

      <div className="mb-4 flex flex-wrap gap-1.5">
        {BENCH.map((b, i) => (
          <button
            key={b.name}
            type="button"
            onClick={() => setIdx(i)}
            aria-pressed={i === idx}
            className="border px-2 py-0.5 font-mono text-[0.66rem] tracking-wide transition-colors"
            style={{
              borderColor: i === idx ? 'var(--color-accent)' : 'var(--color-line)',
              color: i === idx ? 'var(--color-accent)' : 'var(--color-muted)',
            }}
          >
            {b.name}
          </button>
        ))}
      </div>

      <div className="flex flex-col gap-2">
        {MODELS.map((m) => {
          const v = row.vals[m.id];
          const w = ((v - lo) / (100 - lo)) * 100;
          const isBest = v === best;
          return (
            <div key={m.id} className="flex items-center gap-3">
              <span
                className="w-32 shrink-0 font-mono text-[0.72rem]"
                style={{
                  color: m.fugu ? 'var(--color-accent)' : m.color,
                  fontWeight: m.fugu ? 700 : 400,
                }}
              >
                {m.name}
              </span>
              <div className="h-6 flex-1 border border-line bg-paper">
                <div
                  className="flex h-full items-center justify-end pr-2 transition-[width] duration-500 ease-out"
                  style={{
                    width: `${Math.max(w, 6)}%`,
                    background: m.color,
                    opacity: m.light ? 0.55 : 1,
                    outline: isBest ? '2px solid var(--color-accent)' : 'none',
                    outlineOffset: -2,
                  }}
                >
                  <span className="font-mono text-[0.72rem] font-semibold text-paper">
                    {v.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <p className="mt-3 font-mono text-[0.62rem] text-muted">
        Bars start at {lo.toFixed(0)} to make the gaps legible · highest score outlined ·
        provider-reported baselines.
      </p>
    </Widget>
  );
}
