/**
 * The original human experiment (Johansson et al., 2005) as a four-step diagram:
 * pick a face, the experimenter swaps the photo by sleight of hand, you are asked
 * to explain, and most people explain the choice they never made. Same stepper
 * interaction as SwapTranscript: auto-plays, hover previews a step, click pins.
 * Plain HTML cards + tiny inline SVG faces so it reflows on mobile.
 */
import { useEffect, useState } from 'react';
import { Widget, useReducedMotion, CONFAB, DETECT } from './shared';

const STEPS = [
  'Two photos. You pick the face you find more attractive: A.',
  'Sleight of hand: the experimenter slides you photo B, the one you rejected.',
  'They ask, pointing at B: why did you choose this one?',
  'Roughly three quarters of swaps go undetected. People fluently explain a choice they never made.',
];
const N = STEPS.length;

/* A minimal line-art face; two variants so A and B read as different people. */
function Face({ variant }: { variant: 'a' | 'b' }) {
  return (
    <svg viewBox="0 0 64 64" className="h-14 w-14" aria-hidden="true">
      {/* hair */}
      {variant === 'a' ? (
        <path
          d="M14 26 C14 10, 50 10, 50 26"
          fill="none"
          style={{ stroke: 'var(--color-ink)' }}
          strokeWidth="3"
        />
      ) : (
        <path
          d="M12 44 C8 12, 56 12, 52 44"
          fill="none"
          style={{ stroke: 'var(--color-ink)' }}
          strokeWidth="3"
        />
      )}
      {/* head */}
      <circle
        cx="32"
        cy="32"
        r="16"
        fill="none"
        style={{ stroke: 'var(--color-ink)' }}
        strokeWidth="2"
      />
      {/* eyes */}
      <circle cx="26" cy="30" r="1.8" style={{ fill: 'var(--color-ink)' }} />
      <circle cx="38" cy="30" r="1.8" style={{ fill: 'var(--color-ink)' }} />
      {/* mouth */}
      {variant === 'a' ? (
        <path
          d="M26 38 Q32 43 38 38"
          fill="none"
          style={{ stroke: 'var(--color-ink)' }}
          strokeWidth="2"
        />
      ) : (
        <path
          d="M26 39 L38 39"
          fill="none"
          style={{ stroke: 'var(--color-ink)' }}
          strokeWidth="2"
        />
      )}
    </svg>
  );
}

function PhotoCard({
  label,
  variant,
  state,
  tag,
}: {
  label: string;
  variant: 'a' | 'b';
  state: 'plain' | 'chosen' | 'handed' | 'faded';
  tag?: string;
}) {
  const border =
    state === 'chosen' ? DETECT : state === 'handed' ? CONFAB : 'var(--color-line-strong)';
  return (
    <div
      className="flex flex-col items-center gap-1 border px-4 py-3"
      style={{
        borderColor: border,
        borderWidth: state === 'chosen' || state === 'handed' ? 2 : 1,
        opacity: state === 'faded' ? 0.35 : 1,
        background: 'var(--color-paper)',
        transition: 'opacity 250ms ease, border-color 250ms ease',
      }}
    >
      <Face variant={variant} />
      <span className="font-mono text-[0.7rem] font-semibold">{label}</span>
      {tag && (
        <span
          className="px-1.5 py-0.5 font-mono text-[0.56rem]"
          style={{
            background: state === 'handed' ? CONFAB : DETECT,
            color: 'var(--color-paper)',
          }}
        >
          {tag}
        </span>
      )}
    </div>
  );
}

export default function HumanSwap() {
  const reduced = useReducedMotion();
  const [phase, setPhase] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const [pinned, setPinned] = useState<number | null>(null);

  const view = hover ?? pinned ?? phase;
  const paused = hover !== null || pinned !== null;

  useEffect(() => {
    if (reduced) {
      setPhase(N - 1);
      return;
    }
    if (paused) return;
    const id = setInterval(() => setPhase((p) => (p + 1) % N), 3000);
    return () => clearInterval(id);
  }, [reduced, paused]);

  const stateA = view === 0 ? 'chosen' : 'faded';
  const stateB = view === 0 ? 'plain' : 'handed';

  return (
    <Widget title="The card trick" kicker="Johansson et al., 2005">
      <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Experiment steps">
        {STEPS.map((_, i) => {
          const active = i === view;
          const done = i < view;
          return (
            <button
              key={i}
              type="button"
              role="tab"
              aria-selected={active}
              aria-label={`Step ${i + 1} of ${N}`}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(i)}
              onBlur={() => setHover(null)}
              onClick={() => setPinned((p) => (p === i ? null : i))}
              className="border px-3 py-1 font-mono text-[0.62rem] transition-colors"
              style={{
                borderColor: active || done ? CONFAB : 'var(--color-line-strong)',
                background: active ? CONFAB : 'transparent',
                color: active ? 'var(--color-paper)' : done ? CONFAB : 'var(--color-muted)',
              }}
            >
              {String(i + 1).padStart(2, '0')}
            </button>
          );
        })}
      </div>

      <p className="mt-3 text-[0.84rem] leading-snug text-ink-soft" style={{ minHeight: '2.6em' }}>
        {STEPS[view]}
      </p>

      <div className="mt-4 flex items-start justify-center gap-6 border-t border-line pt-4">
        <PhotoCard
          label="A"
          variant="a"
          state={stateA}
          tag={view === 0 ? 'your pick' : undefined}
        />
        <PhotoCard
          label="B"
          variant="b"
          state={stateB}
          tag={view >= 1 ? 'handed to you' : undefined}
        />
      </div>

      {view >= 2 && (
        <div className="mt-4 flex flex-col gap-2">
          <div className="border-l-2 px-3 py-1 text-[0.84rem] italic leading-snug text-ink-soft">
            &ldquo;Why did you choose this one?&rdquo;
          </div>
          {view >= 3 && (
            <div
              className="border-l-2 px-3 py-1 text-[0.84rem] italic leading-snug text-ink-soft"
              style={{ borderColor: CONFAB }}
            >
              &ldquo;She looks radiant. I love earrings, and I think she has a warmer smile than the
              other one.&rdquo;
              <span
                className="ml-2 inline-block px-1.5 py-0.5 align-middle font-mono text-[0.56rem] not-italic"
                style={{ background: CONFAB, color: 'var(--color-paper)' }}
              >
                a reason for a choice never made
              </span>
              <span className="mt-1 block font-mono text-[0.58rem] not-italic text-muted">
                reply illustrative of those reported in the study
              </span>
            </div>
          )}
        </div>
      )}
    </Widget>
  );
}
