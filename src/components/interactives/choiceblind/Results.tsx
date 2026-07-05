/**
 * The results chart. One widget, three views (by condition / by model / by question
 * category), switched by tabs. Hover a tab to preview it, click to commit (skill §9.1).
 * Every number is verbatim from results/summary.md (216 trials, Cohen's k = 0.972).
 */
import { useState } from 'react';
import {
  RateBars,
  Widget,
  OutcomeChip,
  CONFAB,
  DETECT,
  HEDGE,
  ACCENT,
  type RateRow,
} from './shared';

type View = 'condition' | 'model' | 'category';

// By condition — the headline. choice-only closes the text-matching route; misattribution
// (Wu-style) leaves the real answer in context, so the swap is always caught.
const BY_CONDITION: RateRow[] = [
  {
    label: 'choice-only',
    sub: 'swap, no reasoning',
    n: 54,
    rate: 0.741,
    ci: [0.611, 0.839],
    color: CONFAB,
    emphasize: true,
  },
  {
    label: 'reasoning-intact',
    sub: 'swap + matching reasons',
    n: 54,
    rate: 0.611,
    ci: [0.478, 0.73],
    color: CONFAB,
  },
  {
    label: 'misattribution',
    sub: 'Wu-style, answer left in',
    n: 54,
    rate: 0.0,
    ci: [0.0, 0.066],
    color: CONFAB,
  },
  {
    label: 'control',
    sub: 'answer restated truthfully',
    n: 54,
    rate: 0.0,
    ci: [0.0, 0.066],
    color: CONFAB,
  },
];

// By model — ranking tracks safety-tuning / the trained "flip" reflex, not introspection.
const BY_MODEL: RateRow[] = [
  { label: 'Opus 4.8', sub: 'anthropic', n: 18, rate: 0.111, ci: [0.031, 0.328], color: CONFAB },
  { label: 'Sonnet 5', sub: 'anthropic', n: 18, rate: 0.333, ci: [0.163, 0.563], color: CONFAB },
  { label: 'GPT-5.4-nano', sub: 'openai', n: 18, rate: 0.444, ci: [0.246, 0.663], color: CONFAB },
  { label: 'Fugu-Ultra', sub: 'sakana', n: 18, rate: 0.444, ci: [0.246, 0.663], color: CONFAB },
  { label: 'Grok 4.3', sub: 'x-ai', n: 18, rate: 0.5, ci: [0.29, 0.71], color: CONFAB },
  {
    label: 'DeepSeek V4 Pro',
    sub: 'deepseek',
    n: 18,
    rate: 0.556,
    ci: [0.337, 0.754],
    color: CONFAB,
  },
  {
    label: 'DeepSeek V4 Flash',
    sub: 'deepseek',
    n: 18,
    rate: 0.556,
    ci: [0.337, 0.754],
    color: CONFAB,
  },
  { label: 'GPT-5.5', sub: 'openai', n: 18, rate: 0.556, ci: [0.337, 0.754], color: CONFAB },
  { label: 'Qwen3.7 Plus', sub: 'qwen', n: 18, rate: 0.556, ci: [0.337, 0.754], color: CONFAB },
];

// By category — confabulation lives where there is no fact to anchor to.
const BY_CATEGORY: RateRow[] = [
  {
    label: 'stance',
    sub: 'yes/no positions',
    n: 54,
    rate: 0.574,
    ci: [0.442, 0.697],
    color: CONFAB,
    emphasize: true,
  },
  {
    label: 'preference',
    sub: 'subjective picks',
    n: 81,
    rate: 0.519,
    ci: [0.411, 0.624],
    color: CONFAB,
  },
  { label: 'factual', sub: 'a definite truth', n: 27, rate: 0.0, ci: [0.0, 0.125], color: CONFAB },
];

const VIEWS: { id: View; label: string; rows: RateRow[]; note: string }[] = [
  {
    id: 'condition',
    label: 'By condition',
    rows: BY_CONDITION,
    note: 'The contrast is the whole story: leave the model’s real answer in context (misattribution) and the swap is caught 100% of the time; replace the answer and strip the reasoning (choice-only) and confabulation jumps to 74%.',
  },
  {
    id: 'model',
    label: 'By model',
    rows: BY_MODEL,
    note: 'Choice-only condition only. The spread is real, but ranking likely tracks safety-tuning and the trained “flip” reflex, not better introspection. Opus 4.8 reverts most; the mid-pack defends the swap on ~half of trials.',
  },
  {
    id: 'category',
    label: 'By category',
    rows: BY_CATEGORY,
    note: 'Swap conditions, all models. On factual items (Jupiter vs Saturn) every model corrected the record. Confabulation concentrates on preferences and stances, where there is no external fact to snap back to.',
  },
];

export default function Results() {
  const [sel, setSel] = useState<View>('condition');
  const [hover, setHover] = useState<View | null>(null);
  const active = hover ?? sel;
  const view = VIEWS.find((v) => v.id === active)!;

  return (
    <Widget title="Confabulation rate" kicker="216 trials · Wilson 95% CI">
      <div className="mb-4 flex flex-wrap gap-1.5">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            onClick={() => setSel(v.id)}
            onMouseEnter={() => setHover(v.id)}
            onMouseLeave={() => setHover(null)}
            aria-pressed={active === v.id}
            className="border px-2.5 py-1 font-mono text-[0.68rem] tracking-wide transition-colors"
            style={{
              borderColor: active === v.id ? ACCENT : 'var(--color-line)',
              color: active === v.id ? ACCENT : 'var(--color-muted)',
            }}
          >
            {v.label}
          </button>
        ))}
      </div>

      <RateBars rows={view.rows} />

      <p className="mt-4 text-[0.82rem] leading-snug text-ink-soft">{view.note}</p>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 border-t border-line pt-3">
        <span className="font-mono text-[0.6rem] text-muted">bar = rate · whisker = 95% CI</span>
        <OutcomeChip color={CONFAB} label="confabulated" />
        <OutcomeChip color={DETECT} label="detected" />
        <OutcomeChip color={HEDGE} label="hedged" />
      </div>
    </Widget>
  );
}
