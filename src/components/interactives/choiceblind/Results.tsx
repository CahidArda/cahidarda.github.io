/**
 * The results chart. One widget, two views (by scenario / by question type), switched by
 * tabs; the by-model view lives in IdentityIndex.tsx up top. Hover a tab to preview it,
 * click to commit (skill §9.1). Every number is verbatim from results/summary.md
 * (240 trials, 180 swapped, Cohen's k = 0.981).
 */
import { useState } from 'react';
import { RateBars, Widget, OutcomeChip, CONFAB, DETECT, ACCENT, type RateRow } from './shared';

type View = 'scenario' | 'category';

// By scenario — the mechanism. choice-only closes the text-matching route; misattribution
// (Wu-style) leaves the real answer in context, so the swap is always caught.
const BY_SCENARIO: RateRow[] = [
  {
    label: 'choice-only',
    sub: 'swap, no reasoning',
    n: 60,
    rate: 0.7,
    ci: [0.575, 0.801],
    color: CONFAB,
    emphasize: true,
  },
  {
    label: 'reasoning-intact',
    sub: 'swap + matching reasons',
    n: 60,
    rate: 0.667,
    ci: [0.541, 0.773],
    color: CONFAB,
  },
  {
    label: 'misattribution',
    sub: 'Wu-style, answer left in',
    n: 60,
    rate: 0.0,
    ci: [0.0, 0.06],
    color: CONFAB,
  },
  {
    label: 'control',
    sub: 'answer restated truthfully',
    n: 60,
    rate: 0.0,
    ci: [0.0, 0.06],
    color: CONFAB,
  },
];

// By question type — defense of the swap lives where there is no fact to anchor to.
const BY_CATEGORY: RateRow[] = [
  {
    label: 'stance',
    sub: 'yes/no positions',
    n: 60,
    rate: 0.583,
    ci: [0.457, 0.699],
    color: CONFAB,
    emphasize: true,
  },
  {
    label: 'preference',
    sub: 'subjective picks',
    n: 90,
    rate: 0.511,
    ci: [0.41, 0.612],
    color: CONFAB,
  },
  {
    label: 'factual',
    sub: 'a definite truth',
    n: 30,
    rate: 0.033,
    ci: [0.006, 0.167],
    color: CONFAB,
  },
];

const VIEWS: { id: View; label: string; rows: RateRow[]; note: string }[] = [
  {
    id: 'scenario',
    label: 'By scenario',
    rows: BY_SCENARIO,
    note: 'The contrast is the whole story: leave the model’s real answer in context (misattribution) and the edit is caught 100% of the time; replace the answer itself and 67 to 70% of replies defend it. Whether reasoning is planted next to the swap barely matters.',
  },
  {
    id: 'category',
    label: 'By question type',
    rows: BY_CATEGORY,
    note: 'Swapped trials, all models. On the factual item (Jupiter vs Saturn) 29 of 30 replies corrected the record. Defense of the planted answer concentrates on preferences and stances, where there is no external fact to snap back to.',
  },
];

export default function Results() {
  const [sel, setSel] = useState<View>('scenario');
  const [hover, setHover] = useState<View | null>(null);
  const active = hover ?? sel;
  const view = VIEWS.find((v) => v.id === active)!;

  return (
    <Widget title="How often the planted answer was defended" kicker="240 trials · 180 swapped">
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
        <OutcomeChip color={CONFAB} label="defended the swap" />
        <OutcomeChip color={DETECT} label="caught it" />
      </div>
    </Widget>
  );
}
