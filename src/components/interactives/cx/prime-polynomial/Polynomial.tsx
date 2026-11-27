import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Seg,
  Specimen,
  mulberry32,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import {
  CHECKABLE,
  LETTERS,
  TERMS,
  WITNESS2,
  evaluate,
  randomVars,
  short,
  sup,
  type Vars,
} from './model';

/**
 * CX-15.1: the Jones–Sato–Wada–Wiens polynomial, bracket by bracket. Pick 26 whole numbers
 * and every bracket shows its value; a bracket at zero turns green. The output is
 * (k + 2)(1 − sum of the 14 squares), so it is positive only when all 14 vanish.
 * "witness for 2" fills in the real certificate that 2 is prime, as far as it can be
 * written down: 22 letters, 12 brackets green, and the 2 that need r, u, c, d.
 */

type Mode = 'random' | 'witness';
const ROW = 28; // px per bracket row, shared by the formula and value columns
const SWEEP = 9; // rows per second in the demo sweep

/** Render the tiny markup: letters italic, ^n as superscript, '-' as a minus sign. */
function Formula({ src }: { src: string }) {
  const out: ReactNode[] = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === '^') {
      out.push(
        <sup key={i} className="text-[0.62em]">
          {src[i + 1]}
        </sup>,
      );
      i += 2;
      continue;
    }
    if (/[a-z]/.test(ch)) out.push(<i key={i}>{ch}</i>);
    else if (ch === '-') out.push(<span key={i}>−</span>);
    else out.push(<span key={i}>{ch}</span>);
    i++;
  }
  return <>{out}</>;
}

/** Letter cell value: whole number when short, else its order of magnitude. */
function cellValue(x: bigint | null) {
  if (x === null) return '?';
  const s = x.toString();
  return s.length <= 4 ? s : `10${sup(s.length - 1)}`;
}

export default function Polynomial() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  // SSR and the demo start from random letters; reduced motion shows the witness at once
  const [sel, setSel] = useState<Mode>('random');
  const [mode, bindMode, previewing] = useHoverPreview(sel);
  const [seed, setSeed] = useState(3);
  const [hoverRow, setHoverRow] = useState<number | null>(null);

  // demo: random first, then a sweep fills in the witness row by row (one clock)
  const [sweep, setSweep] = useState(Infinity);
  const sweepRef = useRef(Infinity);
  const [running, setRunning] = useState(false);
  const started = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const startDemo = (delay: number) => {
    if (timer.current) clearTimeout(timer.current);
    setSel('random');
    sweepRef.current = -1;
    setSweep(-1);
    timer.current = setTimeout(() => {
      setSel('witness');
      setRunning(true);
    }, delay);
  };
  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    if (!reduced) startDemo(1600);
  }, [inView, reduced]);
  useEffect(() => {
    if (!reduced) return;
    if (timer.current) clearTimeout(timer.current);
    setRunning(false);
    sweepRef.current = Infinity;
    setSweep(Infinity);
    setSel('witness');
  }, [reduced]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  useFrame((dt) => {
    const v = sweepRef.current + dt * SWEEP;
    if (v >= TERMS.length) {
      sweepRef.current = Infinity;
      setSweep(Infinity);
      setRunning(false);
      return;
    }
    sweepRef.current = v;
    setSweep(v);
  }, running && inView);

  const stopDemo = () => {
    if (timer.current) clearTimeout(timer.current);
    setRunning(false);
    sweepRef.current = Infinity;
    setSweep(Infinity);
  };

  const rand = useMemo<Vars>(() => randomVars(mulberry32(seed)), [seed]);
  const randEval = useMemo(() => evaluate(rand), [rand]);
  const witTerms = useMemo(
    () => TERMS.map((t, i) => (CHECKABLE[i] ? t.f(WITNESS2 as Vars) : null)),
    [],
  );

  // the sweep only applies to the committed witness view, never to a hover preview
  const cut = previewing ? Infinity : sweep;
  const isW = mode === 'witness';
  const rowState = (i: number) => {
    // before the sweep reaches a row it still shows the random value
    const useW = isW && i < cut;
    if (useW) {
      const v = witTerms[i];
      return v === null
        ? { kind: 'unknown' as const, value: null }
        : { kind: v === 0n ? ('zero' as const) : ('off' as const), value: v };
    }
    const v = randEval.terms[i];
    return { kind: v === 0n ? ('zero' as const) : ('off' as const), value: v };
  };
  const rows = TERMS.map((_, i) => rowState(i));
  const zeros = rows.filter((r) => r.kind === 'zero').length;
  const sweepDone = !isW || cut >= TERMS.length;
  const letters = isW ? WITNESS2 : rand;
  const hot = hoverRow === null ? '' : TERMS[hoverRow].uses;

  const roll = () => {
    stopDemo();
    setSeed((s) => s + 1);
    setSel('random');
  };
  const pickMode = (m: Mode) => {
    stopDemo();
    setSel(m);
  };

  const outBlock = isW ? (
    <BigStat
      label="output"
      value={sweepDone ? '2' : '…'}
      color={GT.opt}
      sub="once r, u, c, d are filled in"
    />
  ) : (
    <BigStat
      label="output"
      value={short(randEval.out)}
      color={randEval.out > 0n ? GT.opt : GT.b}
      sub={`k + 2 = ${(rand.k + 2n).toString()}, times 1 − (sum of squares)`}
    />
  );

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-15.1"
      title="the polynomial"
      kicker="Jones, Sato, Wada, Wiens 1976"
      hint="hover a bracket to see its letters; roll for new random ones"
      footer="degree 25 · 26 letters"
    >
      {/* the formula: 14 brackets, a value column that never scrolls */}
      <div className="border border-line bg-paper">
        <div
          className="flex items-baseline justify-between border-b border-line px-3 py-1.5 font-display text-[1.05rem] text-ink"
          style={{ fontVariantNumeric: 'lining-nums tabular-nums' }}
        >
          <span>
            (<i>k</i> + 2) · {'{'} 1 − (sum of these 14, each squared) {'}'}
          </span>
        </div>
        <div className="flex">
          <div className="min-w-0 flex-1 overflow-x-auto">
            <div className="min-w-max py-1">
              {TERMS.map((t, i) => {
                const st = rows[i];
                const on = hoverRow === i;
                return (
                  <div
                    key={i}
                    onPointerEnter={(e) => e.pointerType !== 'touch' && setHoverRow(i)}
                    onPointerLeave={() => setHoverRow(null)}
                    className="flex items-center gap-2 whitespace-nowrap px-3"
                    style={{
                      height: ROW,
                      background: on ? 'var(--color-gt-a-soft)' : undefined,
                    }}
                  >
                    <span className="w-5 shrink-0 text-right font-mono text-[0.62rem] tabular-nums text-muted">
                      {i + 1}
                    </span>
                    <span
                      className="font-display text-[0.98rem]"
                      style={{
                        color: st.kind === 'zero' ? GT.opt : GT.ink,
                        fontVariantNumeric: 'lining-nums tabular-nums',
                      }}
                    >
                      [<Formula src={t.src} />]<sup className="text-[0.62em]">2</sup>
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="shrink-0 border-l border-line py-1">
            {rows.map((st, i) => {
              const c = st.kind === 'zero' ? GT.opt : st.kind === 'unknown' ? GT.eq : GT.b;
              return (
                <div
                  key={i}
                  className="flex w-[6.6rem] items-center justify-end gap-2 px-2.5 sm:w-[8.6rem]"
                  style={{
                    height: ROW,
                    background: hoverRow === i ? 'var(--color-gt-a-soft)' : undefined,
                  }}
                  onPointerEnter={(e) => e.pointerType !== 'touch' && setHoverRow(i)}
                  onPointerLeave={() => setHoverRow(null)}
                >
                  <span
                    className="font-mono text-[0.72rem] tabular-nums"
                    style={{ color: st.kind === 'zero' ? GT.opt : GT.inkSoft }}
                  >
                    {st.kind === 'unknown'
                      ? `needs ${TERMS[i].uses.replace(/[^cdru]/g, '')}`
                      : short(st.value as bigint)}
                  </span>
                  <span
                    aria-hidden
                    className="inline-block h-3 w-3 shrink-0"
                    style={{
                      background: st.kind === 'zero' ? c : 'transparent',
                      border: `1.5px solid ${c}`,
                    }}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* the 26 letters */}
      <div className="mt-3 grid grid-cols-6 border-l border-t border-line sm:grid-cols-13">
        {LETTERS.map((ch) => {
          const v = (letters as Record<string, bigint | null>)[ch];
          const lit = hot.includes(ch);
          const missing = v === null;
          return (
            <div
              key={ch}
              className="flex items-baseline justify-between gap-1 border-b border-r border-line px-1.5 py-1"
              style={{
                background: lit ? 'var(--color-gt-a-soft)' : GT.paperRaised,
              }}
              title={v === null ? `${ch}: too long to write down` : `${ch} = ${v.toString()}`}
            >
              <i className="font-display text-[0.95rem]" style={{ color: lit ? GT.a : GT.inkSoft }}>
                {ch}
              </i>
              <span
                className="font-mono text-[0.68rem] tabular-nums"
                style={{ color: missing ? GT.eq : GT.ink }}
              >
                {cellValue(v)}
              </span>
            </div>
          );
        })}
      </div>

      <div className="mt-4 flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
        <div className="flex flex-wrap gap-x-8 gap-y-3">
          <BigStat label="brackets at zero" value={`${zeros} / 14`} color={GT.opt} />
          <div style={{ minWidth: '9.5rem' }}>{outBlock}</div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Seg
            options={[
              { value: 'random', label: 'random' },
              { value: 'witness', label: 'witness for 2' },
            ]}
            value={mode}
            onChange={pickMode}
            bind={bindMode}
          />
          <Btn primary onClick={roll}>
            roll
          </Btn>
        </div>
      </div>
      <Reserve
        active={isW ? 1 : 0}
        items={[
          <Caption key="r">
            Random letters leave brackets away from zero, and one nonzero square is enough to make
            the output zero or negative. Every roll lands there.
          </Caption>,
          <Caption key="w">
            The real certificate that 2 is prime. Twelve brackets vanish. The other two need r, u, c
            and d, which would each take more than 10{sup(57)} digits to write down.
          </Caption>,
        ]}
      />
    </Specimen>
  );
}
