/**
 * Two versions of the same fill-in-the-blank. Left: the kind Duolingo mostly asks, where only
 * one option is a verb. Right: one that tests grammar, where every option is a form of aller.
 * The tag under each option says why it is (or is not) a plausible answer. Static HTML.
 */
import { Caption, Widget } from './shared';

type Option = { word: string; tag: string; correct?: boolean };

const EASY: Option[] = [
  { word: 'vais', tag: 'verb', correct: true },
  { word: 'chat', tag: 'noun' },
  { word: 'rouge', tag: 'adjective' },
  { word: 'demain', tag: 'adverb' },
];

const HARD: Option[] = [
  { word: 'vais', tag: 'je, present', correct: true },
  { word: 'va', tag: 'il/elle, present' },
  { word: 'vont', tag: 'ils/elles, present' },
  { word: 'allais', tag: 'je, imperfect' },
];

function Card({ kicker, options }: { kicker: string; options: Option[] }) {
  return (
    <div className="border border-line bg-paper p-4">
      <div className="label mb-3">{kicker}</div>
      <p className="mb-1 font-display text-xl text-ink">
        Je <span className="inline-block min-w-[3.5rem] border-b-2 border-accent">&nbsp;</span> au
        marché.
      </p>
      <p className="mb-4 font-mono text-[0.7rem] text-muted">I ___ to the market.</p>
      <div className="grid grid-cols-2 gap-2">
        {options.map((o) => (
          <div
            key={o.word}
            className={`border px-3 py-2 ${o.correct ? 'border-accent border-2' : 'border-line-strong'}`}
          >
            <div
              className={`font-mono text-sm ${o.correct ? 'text-accent font-semibold' : 'text-ink'}`}
            >
              {o.word}
            </div>
            <div className="font-mono text-[0.65rem] text-muted">{o.tag}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ExerciseContrast() {
  return (
    <Widget title="Same blank, two questions" kicker="illustrative">
      <div className="grid gap-4 sm:grid-cols-2">
        <Card kicker="What Duolingo mostly asks" options={EASY} />
        <Card kicker="What would test grammar" options={HARD} />
      </div>
      <Caption>
        Left: spot the only verb. Right: every option is a form of aller, so you have to know the
        conjugation.
      </Caption>
    </Widget>
  );
}
