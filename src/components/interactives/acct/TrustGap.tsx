/**
 * Trust as a growth curve: what the bot can touch rises, how often you check
 * falls, and the widening area between them is underwritten by nobody. The two
 * annotations are verbatim fragments from the Grok Bot launch post.
 */
import { useEffect, useRef, useState } from 'react';
import { BOT, LIAB, Widget, YOU, useInView, useReducedMotion } from './shared';

const ACCESS = '40,165 120,140 190,112 270,88 350,70 440,58';
const VERIFY = '40,55 120,84 190,112 270,138 350,156 440,166';
/* The uninsured gap: access from the crossing point out, verify back. */
const GAP = '190,112 270,88 350,70 440,58 440,166 350,156 270,138';

const SWEEP_MS = 2600;

export default function TrustGap() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView<HTMLDivElement>();
  const [progress, setProgress] = useState(0);
  const started = useRef(false);

  useEffect(() => {
    if (reduced) {
      setProgress(1);
      return;
    }
    if (!inView || started.current) return;
    started.current = true;
    let raf = 0;
    let t0: number | null = null;
    const tick = (t: number) => {
      if (t0 === null) t0 = t;
      const p = Math.min(1, (t - t0) / SWEEP_MS);
      setProgress(1 - (1 - p) * (1 - p)); // ease-out
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [reduced, inView]);

  return (
    <Widget title="Trust as a growth curve" kicker="annotations: the launch post" rootRef={rootRef}>
      <svg
        viewBox="0 0 460 236"
        role="img"
        aria-label="Chart: over weeks of working together, what the bot can touch rises while how often you check falls; the widening gap between the two lines is labeled: this gap is yours."
        style={{ width: '100%', height: 'auto', display: 'block' }}
      >
        <clipPath id="acct-sweep">
          <rect x="0" y="0" width={progress * 460} height="236" />
        </clipPath>

        {/* axis */}
        <line
          x1={40}
          y1={200}
          x2={440}
          y2={200}
          style={{ stroke: 'var(--color-line-strong)', strokeWidth: 1 }}
        />
        <text
          x={240}
          y={220}
          textAnchor="middle"
          style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 9 }}
        >
          weeks working together
        </text>

        <g clipPath="url(#acct-sweep)">
          <polygon points={GAP} style={{ fill: LIAB, opacity: 0.13 }} />
          <text
            x={340}
            y={116}
            textAnchor="middle"
            style={{ fill: LIAB, fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700 }}
          >
            this gap is yours
          </text>

          <polyline points={ACCESS} fill="none" style={{ stroke: BOT, strokeWidth: 2.2 }} />
          <text
            x={436}
            y={48}
            textAnchor="end"
            style={{ fill: BOT, fontFamily: 'var(--font-mono)', fontSize: 9.5, fontWeight: 600 }}
          >
            what the bot can touch
          </text>

          <polyline points={VERIFY} fill="none" style={{ stroke: YOU, strokeWidth: 2.2 }} />
          <text
            x={436}
            y={148}
            textAnchor="end"
            style={{ fill: YOU, fontFamily: 'var(--font-mono)', fontSize: 9.5, fontWeight: 600 }}
          >
            how often you check
          </text>

          {/* verbatim fragments from the launch post's customer quotes */}
          <circle cx={40} cy={55} r={3} style={{ fill: YOU }} />
          <text
            x={48}
            y={45}
            style={{ fill: 'var(--color-ink-soft)', fontFamily: 'var(--font-mono)', fontSize: 9 }}
          >
            &quot;checking in on them every 15 minutes&quot;
          </text>
          <circle cx={440} cy={166} r={3} style={{ fill: YOU }} />
          <text
            x={432}
            y={186}
            textAnchor="end"
            style={{ fill: 'var(--color-ink-soft)', fontFamily: 'var(--font-mono)', fontSize: 9 }}
          >
            &quot;fully trust it to run forever&quot;
          </text>
        </g>
      </svg>
    </Widget>
  );
}
