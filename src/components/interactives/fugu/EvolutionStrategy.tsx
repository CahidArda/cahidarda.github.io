import { useCallback, useEffect, useRef, useState } from 'react';
import { Widget, useReducedMotion } from './shared';

// A 2-D cartoon of the sep-CMA-ES loop Fugu uses to refine its orchestrator on
// end-to-end tasks: sample a population of perturbed parameter vectors around a parent,
// score each by terminal reward, recombine the best into the next parent. The real
// search is ~10K-dimensional; here it's 2-D so you can watch it converge.
const OPT: [number, number] = [1.3, -0.9]; // the (hidden) reward optimum
const LAMBDA = 24; // population size
const MU = 8; // recombined elites

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gauss = (rng: () => number) =>
  Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());

interface ES {
  m: [number, number];
  sigma: number;
  C: [number, number];
  gen: number;
  seed: number;
  pop: { x: [number, number]; f: number; elite: boolean }[];
}

const fitness = (x: [number, number]) => -((x[0] - OPT[0]) ** 2 + (x[1] - OPT[1]) ** 2);

function init(seed = 7): ES {
  return { m: [-2.4, 2.2], sigma: 1.3, C: [1, 1], gen: 0, seed, pop: [] };
}

function step(s: ES): ES {
  const rng = mulberry32(s.seed);
  const pop = Array.from({ length: LAMBDA }, () => {
    const x: [number, number] = [
      s.m[0] + s.sigma * Math.sqrt(s.C[0]) * gauss(rng),
      s.m[1] + s.sigma * Math.sqrt(s.C[1]) * gauss(rng),
    ];
    return { x, f: fitness(x), elite: false };
  });
  pop.sort((a, b) => b.f - a.f);
  for (let i = 0; i < MU; i++) pop[i].elite = true;
  const m: [number, number] = [0, 0];
  for (let i = 0; i < MU; i++) {
    m[0] += pop[i].x[0] / MU;
    m[1] += pop[i].x[1] / MU;
  }
  // diagonal covariance from elite spread + gentle step-size decay
  const C: [number, number] = [0, 0];
  for (let i = 0; i < MU; i++) {
    C[0] += (pop[i].x[0] - m[0]) ** 2 / MU;
    C[1] += (pop[i].x[1] - m[1]) ** 2 / MU;
  }
  return {
    m,
    sigma: Math.max(0.04, s.sigma * 0.86),
    C: [Math.max(0.05, C[0] + 0.02), Math.max(0.05, C[1] + 0.02)],
    gen: s.gen + 1,
    seed: (s.seed * 1664525 + 1013904223) >>> 0,
    pop,
  };
}

const MAXGEN = 22;

export default function EvolutionStrategy() {
  const reduced = useReducedMotion();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const esRef = useRef<ES>(init());
  const [es, setEs] = useState<ES>(esRef.current);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  const draw = useCallback((s: ES) => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const W = cvs.clientWidth;
    const H = cvs.clientHeight;
    if (cvs.width !== W * dpr || cvs.height !== H * dpr) {
      cvs.width = W * dpr;
      cvs.height = H * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const col = (n: string, f: string) => css.getPropertyValue(n).trim() || f;
    const toPx = (x: number, y: number): [number, number] => [((x + 4) / 8) * W, ((4 - y) / 8) * H];

    ctx.fillStyle = col('--paper', '#f4f1e9');
    ctx.fillRect(0, 0, W, H);
    // reward contours around the optimum
    const [ox, oy] = toPx(OPT[0], OPT[1]);
    for (let r = 6; r >= 1; r--) {
      ctx.beginPath();
      ctx.arc(ox, oy, (r / 6) * Math.min(W, H) * 0.42, 0, Math.PI * 2);
      ctx.strokeStyle = col('--line', '#d8d3c5');
      ctx.globalAlpha = 0.5;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // optimum marker
    ctx.fillStyle = col('--muted', '#777');
    ctx.font = '11px monospace';
    ctx.fillText('reward optimum', ox + 8, oy - 6);
    ctx.strokeStyle = col('--muted', '#777');
    ctx.beginPath();
    ctx.moveTo(ox - 5, oy);
    ctx.lineTo(ox + 5, oy);
    ctx.moveTo(ox, oy - 5);
    ctx.lineTo(ox, oy + 5);
    ctx.stroke();

    // population
    for (const p of s.pop) {
      const [px, py] = toPx(p.x[0], p.x[1]);
      ctx.beginPath();
      ctx.arc(px, py, p.elite ? 4 : 2.6, 0, Math.PI * 2);
      ctx.fillStyle = p.elite ? col('--accent', '#b14a2a') : col('--line-strong', '#b9b3a2');
      ctx.fill();
    }
    // parent
    const [mx, my] = toPx(s.m[0], s.m[1]);
    ctx.fillStyle = col('--fugu', '#b14a2a');
    ctx.beginPath();
    ctx.arc(mx, my, 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = col('--paper', '#fff');
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }, []);

  const reset = useCallback(
    (autoplay: boolean) => {
      if (timer.current) clearInterval(timer.current);
      const s = init();
      esRef.current = s;
      setEs(s);
      if (reduced) {
        let cur = s;
        for (let i = 0; i < MAXGEN; i++) cur = step(cur);
        esRef.current = cur;
        setEs(cur);
        requestAnimationFrame(() => draw(cur));
      } else {
        requestAnimationFrame(() => draw(s));
        setPlaying(autoplay);
      }
    },
    [draw, reduced],
  );

  useEffect(() => {
    reset(true);
    const onResize = () => draw(esRef.current);
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
      if (timer.current) clearInterval(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!playing) return;
    timer.current = setInterval(() => {
      const next = step(esRef.current);
      esRef.current = next;
      setEs(next);
      draw(next);
      if (next.gen >= MAXGEN) {
        if (timer.current) clearInterval(timer.current);
        setPlaying(false);
      }
    }, 420);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [playing, draw]);

  const best = es.pop.length
    ? Math.sqrt(-es.pop[0].f)
    : Math.hypot(es.m[0] - OPT[0], es.m[1] - OPT[1]);
  const done = es.gen >= MAXGEN;

  return (
    <Widget
      title="Training stage 2 — evolution with sep-CMA-ES"
      kicker="no gradients, just fitness"
    >
      <p className="mb-3 text-sm text-ink-soft">
        After supervised fine-tuning, Fugu is refined with a derivative-free evolution strategy that
        directly maximises the <strong>terminal reward</strong> of end-to-end tasks. Each dot is one
        perturbed orchestrator; <span style={{ color: 'var(--color-accent)' }}>elites</span> are
        recombined into the next parent.
      </p>

      <div className="border border-line-strong">
        <canvas ref={canvasRef} className="block h-[240px] w-full" />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => (done ? reset(true) : setPlaying((p) => !p))}
          className="border-2 border-accent px-3.5 py-1.5 font-mono text-xs tracking-[0.12em] text-accent uppercase transition-colors hover:bg-accent hover:text-paper"
        >
          {done ? '▶ Replay' : playing ? '❚❚ Pause' : '▶ Play'}
        </button>
        <span className="font-mono text-xs text-muted">
          generation {es.gen}/{MAXGEN} · distance to optimum {best.toFixed(2)}
        </span>
      </div>

      <p className="mt-3 font-mono text-[0.66rem] text-muted">
        “sep-CMA-ES… iteratively improves a central ‘parent’ policy by sampling a population of
        perturbed parameter vectors, evaluating each candidate to obtain a fitness score, and
        recombining candidates via fitness-weighted averaging.”
      </p>
    </Widget>
  );
}
