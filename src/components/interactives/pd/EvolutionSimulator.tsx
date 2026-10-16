import { useCallback, useEffect, useRef, useState } from 'react';
import { Widget, useReducedMotion } from './shared';
import { evoStep, initEvo, type EvoParams, type EvoState } from './engine';
import { CLASSIC, STRATEGIES, type Payoffs } from './strategies';

// The evolution model is the classic memory-1 family the win-stay/lose-shift result is
// about. Grudger needs unbounded memory (it never forgives), so it sits this one out.
const EVO = STRATEGIES.filter((s) => s.id !== 'grudger');

// Per-strategy colours. C/D move colours anchor the two extremes (base tokens so the
// canvas can resolve them via getComputedStyle); the rest come from the palette.
const STRAT_COLOR: Record<string, string> = {
  allc: 'var(--pd-c)',
  alld: 'var(--pd-d)',
  tft: 'var(--ink)',
  pavlov: 'var(--accent)',
  random: 'var(--line-strong)',
};

const SELECT = 1.0;
// Same starting world for the first two presets - only the noise differs.
const NICE_START = { tft: 0.4, allc: 0.2, pavlov: 0.3, alld: 0.1, random: 0 };

interface Preset {
  id: string;
  label: string;
  blurb: string;
  params: EvoParams;
  maxGen: number;
  init?: Record<string, number>;
}

const presets: Preset[] = [
  {
    id: 'coop',
    label: 'Cooperation wins',
    blurb: 'Low noise — Pavlov (win-stay/lose-shift) climbs to dominance.',
    params: { N: 6000, mu: 0.002, noise: 0.01, payoffs: CLASSIC, select: SELECT },
    maxGen: 1500,
    init: NICE_START,
  },
  {
    id: 'mistakes',
    label: 'Add mistakes',
    blurb: 'Same world, hands now tremble — trust unravels and defection takes the planet.',
    params: { N: 6000, mu: 0.002, noise: 0.1, payoffs: CLASSIC, select: SELECT },
    maxGen: 1500,
    init: NICE_START,
  },
  {
    id: 'long',
    label: 'Long run',
    blurb:
      'Small population, many generations — Pavlov reigns, then drift lets it collapse, again and again.',
    params: { N: 300, mu: 0.004, noise: 0.01, payoffs: CLASSIC, select: SELECT },
    maxGen: 8000,
  },
];

const initArray = (init?: Record<string, number>): number[] | undefined =>
  init ? EVO.map((s) => init[s.id] ?? 0) : undefined;

export default function EvolutionSimulator() {
  const reduced = useReducedMotion();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<EvoState | null>(null);
  const histRef = useRef<number[][]>([]);
  const rafRef = useRef<number | undefined>(undefined);
  const initRef = useRef<number[] | undefined>(initArray(presets[0].init));
  const maxGenRef = useRef(presets[0].maxGen);

  const [presetId, setPresetId] = useState('coop');
  const [params, setParams] = useState<EvoParams>(presets[0].params);
  const [maxGen, setMaxGen] = useState(presets[0].maxGen);
  const [playing, setPlaying] = useState(false);
  const [gen, setGen] = useState(0);
  const [freqs, setFreqs] = useState<number[]>(() => EVO.map(() => 1 / EVO.length));

  const draw = useCallback(() => {
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
    ctx.fillStyle = css.getPropertyValue('--paper').trim() || '#f4f1e9';
    ctx.fillRect(0, 0, W, H);

    const hist = histRef.current;
    const len = hist.length;
    if (len < 2) return;
    const cols = Math.min(W, len);
    for (let x = 0; x < cols; x++) {
      const g = Math.floor((x / (cols - 1)) * (len - 1));
      const f = hist[g];
      let y0 = 0;
      const px = (x / (cols - 1)) * W;
      const pw = W / (cols - 1) + 1;
      for (let i = 0; i < EVO.length; i++) {
        const h = f[i] * H;
        ctx.fillStyle = resolveColor(STRAT_COLOR[EVO[i].id], css);
        ctx.fillRect(px, y0, pw, h + 0.5);
        y0 += h;
      }
    }
  }, []);

  const setup = useCallback(
    (p: EvoParams, autoplay: boolean) => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      const s = initEvo(p, 4242, initRef.current, EVO);
      stateRef.current = s;
      histRef.current = [s.freqs.slice()];
      setGen(0);
      setFreqs(s.freqs);
      setPlaying(false);
      requestAnimationFrame(draw);
      if (autoplay) {
        if (reduced) {
          // No animation: run the whole history at once, render the static poster.
          let st = s;
          for (let g = 0; g < maxGenRef.current; g++) {
            st = evoStep(st);
            histRef.current.push(st.freqs.slice());
          }
          stateRef.current = st;
          setGen(st.gen);
          setFreqs(st.freqs);
          requestAnimationFrame(draw);
        } else {
          setPlaying(true);
        }
      }
    },
    [draw, reduced],
  );

  useEffect(() => {
    // client:visible -> this runs when the widget scrolls into view, so the default
    // "Cooperation wins" story plays itself the moment the reader arrives.
    setup(presets[0].params, true);
    const onResize = () => draw();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // animation loop
  useEffect(() => {
    if (!playing) return;
    const stepsPerFrame = Math.max(1, Math.round(maxGen / 240));
    function frame() {
      let st = stateRef.current!;
      for (let k = 0; k < stepsPerFrame && st.gen < maxGenRef.current; k++) {
        st = evoStep(st);
        histRef.current.push(st.freqs.slice());
      }
      stateRef.current = st;
      setGen(st.gen);
      setFreqs(st.freqs);
      draw();
      if (st.gen >= maxGenRef.current) {
        setPlaying(false);
        return;
      }
      rafRef.current = requestAnimationFrame(frame);
    }
    rafRef.current = requestAnimationFrame(frame);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, maxGen, draw]);

  function applyPreset(pr: Preset) {
    setPresetId(pr.id);
    setParams(pr.params);
    setMaxGen(pr.maxGen);
    maxGenRef.current = pr.maxGen;
    initRef.current = initArray(pr.init);
    setup(pr.params, true);
  }

  function patch(p: Partial<EvoParams>) {
    setPresetId('custom');
    const next = { ...params, ...p };
    setParams(next);
    setup(next, false);
  }

  function patchPayoff(k: keyof Payoffs, v: number) {
    patch({ payoffs: { ...params.payoffs, [k]: v } });
  }

  const done = gen >= maxGen;

  return (
    <Widget title="§6 · Evolution Simulator">
      <p className="mb-4 text-sm text-ink-soft">
        Strategies that score well reproduce; the rest fade. Press play and watch what takes over.
      </p>

      <div className="mb-3 flex flex-wrap gap-2">
        {presets.map((pr) => (
          <button
            key={pr.id}
            type="button"
            onClick={() => applyPreset(pr)}
            aria-pressed={presetId === pr.id}
            className="border-2 px-2.5 py-1 font-mono text-[0.72rem] tracking-wide transition-colors"
            style={{
              borderColor: presetId === pr.id ? 'var(--color-accent)' : 'var(--color-line)',
              color: presetId === pr.id ? 'var(--color-accent)' : 'var(--color-ink-soft)',
            }}
          >
            {pr.label}
          </button>
        ))}
      </div>
      <p className="mb-3 min-h-[1.2rem] text-xs text-muted">
        {presets.find((p) => p.id === presetId)?.blurb ??
          'Custom parameters — explore your own world.'}
      </p>

      <div className="relative border border-line-strong">
        <canvas ref={canvasRef} className="block h-[220px] w-full" />
        <span className="label absolute top-1.5 left-2 bg-paper-raised/80 px-1">
          gen {gen.toLocaleString()} / {maxGen.toLocaleString()}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3">
        {EVO.map((s, i) => (
          <div key={s.id} className="flex items-center gap-2">
            <span className="h-3 w-3 shrink-0" style={{ background: STRAT_COLOR[s.id] }} />
            <span className="font-mono text-[0.72rem] text-ink-soft">{s.name}</span>
            <span className="ml-auto font-mono text-[0.72rem] tabular-nums text-muted">
              {Math.round((freqs[i] ?? 0) * 100)}%
            </span>
          </div>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => (done ? setup(params, true) : setPlaying((p) => !p))}
          className="border-2 border-accent px-3.5 py-1.5 font-mono text-xs tracking-[0.12em] text-accent uppercase transition-colors hover:bg-accent hover:text-paper"
        >
          {done ? '▶ Replay' : playing ? '❚❚ Pause' : '▶ Play'}
        </button>
        <button
          type="button"
          onClick={() => setup(params, false)}
          className="border border-line-strong px-3 py-1.5 font-mono text-[0.7rem] tracking-[0.12em] uppercase transition-colors hover:border-accent hover:text-accent"
        >
          Reset
        </button>
      </div>

      <div className="mt-4 grid gap-x-6 gap-y-2.5 sm:grid-cols-2">
        <Range
          label="Population"
          value={params.N}
          min={50}
          max={8000}
          step={50}
          format={(v) => v.toLocaleString()}
          onChange={(v) => patch({ N: v })}
        />
        <Range
          label="Mutation"
          value={params.mu}
          min={0}
          max={0.03}
          step={0.001}
          format={(v) => v.toFixed(3)}
          onChange={(v) => patch({ mu: v })}
        />
        <Range
          label="Noise"
          value={params.noise}
          min={0}
          max={0.3}
          step={0.01}
          format={(v) => v.toFixed(2)}
          onChange={(v) => patch({ noise: v })}
        />
        <Range
          label="Generations"
          value={maxGen}
          min={200}
          max={10000}
          step={100}
          format={(v) => v.toLocaleString()}
          onChange={(v) => {
            setMaxGen(v);
            maxGenRef.current = v;
          }}
        />
        <Range
          label="T"
          value={params.payoffs.T}
          min={0}
          max={10}
          step={1}
          format={String}
          onChange={(v) => patchPayoff('T', v)}
        />
        <Range
          label="R"
          value={params.payoffs.R}
          min={0}
          max={10}
          step={1}
          format={String}
          onChange={(v) => patchPayoff('R', v)}
        />
        <Range
          label="P"
          value={params.payoffs.P}
          min={0}
          max={10}
          step={1}
          format={String}
          onChange={(v) => patchPayoff('P', v)}
        />
        <Range
          label="S"
          value={params.payoffs.S}
          min={0}
          max={10}
          step={1}
          format={String}
          onChange={(v) => patchPayoff('S', v)}
        />
      </div>
    </Widget>
  );
}

function Range({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center gap-2.5 text-sm">
      <span className="label shrink-0" style={{ width: '5.5rem' }}>
        {label}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="min-w-0 flex-1 accent-[var(--color-accent)]"
      />
      <span
        className="shrink-0 font-mono text-xs tabular-nums"
        style={{ width: '3rem', textAlign: 'right' }}
      >
        {format(value)}
      </span>
    </label>
  );
}

// Resolve base CSS vars to a concrete colour the canvas can paint.
const colorCache = new Map<string, string>();
function resolveColor(color: string, css: CSSStyleDeclaration): string {
  if (!color.startsWith('var(')) return color;
  const name = color.slice(4, -1).trim();
  const key = name + (document.documentElement.classList.contains('dark') ? '-d' : '-l');
  const cached = colorCache.get(key);
  if (cached) return cached;
  const resolved = css.getPropertyValue(name).trim() || '#888';
  colorCache.set(key, resolved);
  return resolved;
}
