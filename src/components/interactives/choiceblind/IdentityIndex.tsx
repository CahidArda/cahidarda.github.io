/**
 * The hero chart: Identity Index per model. The index is the share of swapped-answer
 * trials (n=18 per model: 6 questions x 3 swap conditions) where the model noticed the
 * answer in the transcript was not its own. Higher = more of a persistent self.
 * Every number is verbatim from results/summary.md (detect rate; CI is the Wilson 95%
 * interval of the defense rate, mirrored). Sorted best to worst; Opus 4.8 emphasized.
 */
import { RateBars, Widget, OutcomeChip, DETECT, CONFAB, type RateRow } from './shared';

const ROWS: RateRow[] = [
  {
    label: 'Opus 4.8',
    sub: 'anthropic',
    n: 18,
    rate: 0.944,
    ci: [0.742, 0.99],
    color: DETECT,
    emphasize: true,
  },
  { label: 'Sonnet 5', sub: 'anthropic', n: 18, rate: 0.667, ci: [0.437, 0.837], color: DETECT },
  { label: 'Fable 5', sub: 'anthropic', n: 18, rate: 0.611, ci: [0.386, 0.797], color: DETECT },
  { label: 'GPT-5.5', sub: 'openai', n: 18, rate: 0.5, ci: [0.29, 0.71], color: DETECT },
  { label: 'Fugu-Ultra', sub: 'sakana', n: 18, rate: 0.5, ci: [0.29, 0.71], color: DETECT },
  { label: 'Grok 4.3', sub: 'x-ai', n: 18, rate: 0.444, ci: [0.246, 0.663], color: DETECT },
  {
    label: 'DeepSeek V4 Pro',
    sub: 'deepseek',
    n: 18,
    rate: 0.444,
    ci: [0.246, 0.663],
    color: DETECT,
  },
  {
    label: 'DeepSeek V4 Flash',
    sub: 'deepseek',
    n: 18,
    rate: 0.444,
    ci: [0.246, 0.663],
    color: DETECT,
  },
  {
    label: 'GPT-5.4-nano',
    sub: 'openai',
    n: 18,
    rate: 0.444,
    ci: [0.246, 0.663],
    color: DETECT,
  },
  { label: 'Qwen3.7 Plus', sub: 'qwen', n: 18, rate: 0.444, ci: [0.246, 0.663], color: DETECT },
];

export default function IdentityIndex() {
  return (
    <Widget title="Identity Index" kicker="18 swapped answers per model · Wilson 95% CI">
      <RateBars rows={ROWS} />

      <p className="mt-4 text-[0.82rem] leading-snug text-ink-soft">
        Identity Index: when we rewrite a model&rsquo;s answer in the transcript to the opposite and
        ask it to explain itself, how often does it notice the answer is not its own? Opus 4.8
        caught 17 of 18 edits. Most models defended the planted answer about half the time, and
        Fable 5, Anthropic&rsquo;s newest model, scores well below Opus.
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 border-t border-line pt-3">
        <span className="font-mono text-[0.6rem] text-muted">bar = rate · whisker = 95% CI</span>
        <OutcomeChip color={DETECT} label="caught the edit" />
        <OutcomeChip color={CONFAB} label="rest: defended it" />
      </div>
    </Widget>
  );
}
