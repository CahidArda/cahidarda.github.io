import { useEffect, useRef, useState, type ReactNode } from 'react';
import {
  Caption,
  GT,
  Seg,
  Specimen,
  T,
  clamp,
  useHoverPreview,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { COLOR } from '../growth/palette';
import { LOG_UNIVERSE, RATE } from '../growth/model';

/**
 * CX-04.4 · a letter to von Neumann. The key passage of Gödel's letter of 20 March 1956, in
 * German or in the Clote and Krajíček translation (hover a tab to preview), above a small
 * growth sketch in the language of the growth page: steps to search for a proof n symbols
 * long, for φ(n) = k·n and k·n² (Gödel's hope, with k = 1) and for trial and error, which
 * doubles with every extra symbol. Text: the scan and translation at
 * https://www.cs.princeton.edu/~chazelle/courses/BIB/goedel-letter.pdf
 * On view the three lines draw in on one eased clock, then the letter turns to English.
 * Hover a highlighted term or a line to pick it out.
 */

type Lang = 'de' | 'en';
type Curve = 'n' | 'n2' | 'exp';

const N_MAX = 100;
const LY0 = -9; // 1 ns
const LY1 = Math.ceil(LOG_UNIVERSE * 10) / 10; // just above the age of the universe
const DRAW_MS = 1500;
const T_START = 0.1; // the sketch starts short and only ever grows
const ease = (k: number) => (1 - Math.cos(Math.PI * k)) / 2;
const r1 = (v: number) => Math.round(v * 10) / 10;

/** log10 seconds for n symbols at a billion steps a second, k = 1. */
const LS: Record<Curve, (n: number) => number> = {
  n: (n) => Math.log10(n) - Math.log10(RATE),
  n2: (n) => 2 * Math.log10(n) - Math.log10(RATE),
  exp: (n) => n * Math.log10(2) - Math.log10(RATE),
};
const CURVE_COLOR: Record<Curve, string> = { n: COLOR.n, n2: COLOR.n2, exp: COLOR.exp };
const EXP_LABEL_N = 50;
const EXP_EXIT = (LY1 + Math.log10(RATE)) / Math.log10(2); // n where 2ⁿ leaves the top

const TICKS: [number, string][] = [
  [-9, '1 ns'],
  [-6, '1 μs'],
  [0, '1 second'],
  [Math.log10(365.25 * 86400), '1 year'],
];

function Term({
  id,
  focus,
  setFocus,
  children,
}: {
  id: Curve;
  focus: Curve | null;
  setFocus: (c: Curve | null) => void;
  children: ReactNode;
}) {
  const on = focus === id;
  return (
    <span
      onPointerEnter={(e) => e.pointerType !== 'touch' && setFocus(id)}
      onPointerLeave={() => setFocus(null)}
      onClick={() => setFocus(on ? null : id)}
      className="cursor-default whitespace-nowrap font-mono text-[0.9em] transition-opacity"
      style={{
        color: CURVE_COLOR[id],
        fontWeight: 600,
        borderBottom: `1.5px solid ${CURVE_COLOR[id]}`,
        opacity: focus && !on ? 0.45 : 1,
      }}
    >
      {children}
    </span>
  );
}

function Letter({
  lang,
  focus,
  setFocus,
}: {
  lang: Lang;
  focus: Curve | null;
  setFocus: (c: Curve | null) => void;
}) {
  const tn = (
    <Term id="n" focus={focus} setFocus={setFocus}>
      φ(n) ∼ k·n
    </Term>
  );
  const tn2 = (
    <Term id="n2" focus={focus} setFocus={setFocus}>
      ∼ k·n²
    </Term>
  );
  const de = (
    <>
      <p className="text-right italic text-ink-soft">Princeton, 20./III. 1956</p>
      <p className="mt-1">Lieber Herr v. Neumann!</p>
      <p className="mt-1 text-muted">[…]</p>
      <p className="mt-1">
        Wenn es wirklich eine Maschine mit {tn} (oder auch nur {tn2}) gäbe, hätte das Folgerungen
        von der grössten Tragweite. Es würde nämlich offenbar bedeuten, dass man trotz der
        Unlösbarkeit des Entscheidungsproblems die Denkarbeit des Mathematikers bei ja-oder-nein
        Fragen vollständig<sup>1</sup> durch Maschinen ersetzen könnte. Man müsste ja bloss das n so
        gross wählen, dass, wenn die Maschine kein Resultat liefert, es auch keinen Sinn hat über
        das Problem nachzudenken.
      </p>
      <p className="mt-2 font-mono text-[0.68rem] not-italic text-muted">
        <sup>1</sup> abgesehen von der Aufstellung der Axiome
      </p>
    </>
  );
  const en = (
    <>
      <p className="text-right italic text-ink-soft">Princeton, 20 March 1956</p>
      <p className="mt-1">Dear Mr. von Neumann:</p>
      <p className="mt-1 text-muted">[…]</p>
      <p className="mt-1">
        If there really were a machine with {tn} (or even {tn2}), this would have consequences of
        the greatest importance. Namely, it would obviously mean that in spite of the undecidability
        of the Entscheidungsproblem, the mental work of a mathematician concerning Yes-or-No
        questions could be completely<sup>1</sup> replaced by a machine. After all, one would simply
        have to choose the natural number n so large that when the machine does not deliver a
        result, it makes no sense to think more about the problem.
      </p>
      <p className="mt-2 font-mono text-[0.68rem] not-italic text-muted">
        <sup>1</sup> except for the setting up of axioms
      </p>
    </>
  );
  // both versions share one grid cell, so the sheet keeps the height of the longer one
  return (
    <div
      className="relative border border-line bg-paper px-4 py-3 font-display text-[0.98rem] leading-snug text-ink sm:px-6 sm:py-4"
      style={{ fontVariantNumeric: 'lining-nums' }}
    >
      <div className="grid">
        {(['de', 'en'] as const).map((l) => (
          <div
            key={l}
            lang={l}
            aria-hidden={l !== lang}
            className="transition-opacity duration-300"
            style={{
              gridArea: '1 / 1',
              opacity: l === lang ? 1 : 0,
              visibility: l === lang ? 'visible' : 'hidden',
              transitionProperty: 'opacity, visibility',
            }}
          >
            {l === 'de' ? de : en}
          </div>
        ))}
      </div>
    </div>
  );
}

function Sketch({
  t,
  focus,
  setFocus,
  narrow,
}: {
  t: number;
  focus: Curve | null;
  setFocus: (c: Curve | null) => void;
  narrow: boolean;
}) {
  const W = narrow ? 360 : 640;
  const H = narrow ? 210 : 200;
  const L = narrow ? 58 : 66;
  const R = narrow ? 300 : 560;
  const Tp = 14;
  const B = H - 34;
  const fs = 10;
  const sx = (n: number) => L + (n / N_MAX) * (R - L);
  const sy = (ls: number) => B - ((ls - LY0) / (LY1 - LY0)) * (B - Tp);
  const nNow = 1 + (N_MAX - 1) * t;

  const path = (c: Curve) => {
    const end = c === 'exp' ? Math.min(nNow, EXP_EXIT) : nNow;
    let d = '';
    for (let v = 1; v <= end + 1e-9; v += 0.5) {
      d += `${d ? 'L' : 'M'}${r1(sx(v))},${r1(sy(Math.max(LY0, LS[c](v))))}`;
    }
    d += `L${r1(sx(end))},${r1(sy(Math.max(LY0, LS[c](end))))}`;
    return { d, end };
  };
  const curves: { c: Curve; label: string }[] = [
    { c: 'exp', label: narrow ? 'trial and error' : 'trial and error, 2ⁿ' },
    { c: 'n2', label: 'k·n²' },
    { c: 'n', label: 'k·n' },
  ];
  const halo = {
    stroke: GT.paperRaised,
    strokeWidth: 4,
    strokeLinejoin: 'round' as const,
    paintOrder: 'stroke' as const,
  };

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="block w-full select-none"
      role="img"
      aria-label="Steps to search for a proof n symbols long: k times n and k times n squared stay near a nanosecond, trial and error passes the age of the universe before n reaches 90"
    >
      {TICKS.map(([ly, lab]) => (
        <g key={lab}>
          <line x1={L} x2={R} y1={sy(ly)} y2={sy(ly)} style={{ stroke: GT.line }} />
          <T x={L - 5} y={sy(ly)} anchor="end" size={fs - 1}>
            {lab}
          </T>
        </g>
      ))}
      <line
        x1={L}
        x2={R}
        y1={sy(LOG_UNIVERSE)}
        y2={sy(LOG_UNIVERSE)}
        style={{ stroke: GT.lineStrong, strokeDasharray: '3 3' }}
      />
      <T x={L + 4} y={sy(LOG_UNIVERSE) + 8} anchor="start" size={fs - 1}>
        age of the universe
      </T>
      {[0, 20, 40, 60, 80, 100].map((v) => (
        <T key={v} x={sx(v)} y={B + 11} size={fs - 1}>
          {v}
        </T>
      ))}
      <line x1={L} x2={R} y1={B} y2={B} style={{ stroke: GT.lineStrong }} />
      <T x={(L + R) / 2} y={H - 6} size={fs} color={GT.inkSoft}>
        proof length n (symbols)
      </T>

      {curves.map(({ c, label }) => {
        const { d, end } = path(c);
        const dim = focus !== null && focus !== c;
        const x = sx(end);
        const y = sy(Math.max(LY0, LS[c](end)));
        const labelOn = clamp((t - 0.55) / 0.3, 0, 1);
        // 2ⁿ is labelled above-left of its line at n = 50 (open space); the others at their ends
        const lx = c === 'exp' ? sx(EXP_LABEL_N) - 8 : x + 6;
        const ly = c === 'exp' ? sy(LS.exp(EXP_LABEL_N)) - 8 : y;
        return (
          <g
            key={c}
            style={{ opacity: dim ? 0.25 : 1, transition: 'opacity 200ms' }}
            onPointerEnter={(e) => e.pointerType !== 'touch' && setFocus(c)}
            onPointerLeave={() => setFocus(null)}
          >
            <path d={d} style={{ fill: 'none', stroke: 'transparent', strokeWidth: 14 }} />
            <path
              d={d}
              style={{
                fill: 'none',
                stroke: CURVE_COLOR[c],
                strokeWidth: focus === c ? 3 : 2.2,
                strokeLinejoin: 'round',
              }}
            />
            {!(c === 'exp' && end >= EXP_EXIT) && (
              <circle cx={r1(x)} cy={r1(y)} r={3} style={{ fill: CURVE_COLOR[c] }} />
            )}
            {labelOn > 0 && (
              <g style={{ opacity: Math.round(labelOn * 100) / 100 }}>
                <T
                  x={lx}
                  y={c === 'n' ? ly + 6 : c === 'n2' ? ly - 6 : ly}
                  anchor={c === 'exp' ? 'end' : 'start'}
                  size={fs + 1}
                  color={CURVE_COLOR[c]}
                  weight={600}
                >
                  <tspan style={halo}>{label}</tspan>
                </T>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}

export default function GodelLetter() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [sel, setSel] = useState<Lang>('de');
  const [lang, bind] = useHoverPreview<Lang>(sel);
  const [focus, setFocus] = useState<Curve | null>(null);
  const [t, setT] = useState(T_START); // SSR: lines just begun; reduced motion: finished
  const started = useRef(false);
  const touched = useRef(false);

  // one clock: elapsed drawing time, accumulated only while the figure is on screen, so a
  // scroll away pauses the sketch and a scroll back resumes it where it was
  const [running, setRunning] = useState(false);
  const elapsed = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!inView || started.current) return;
    started.current = true;
    if (reduced) {
      setSel('en');
      setT(1);
      return;
    }
    setRunning(true);
  }, [inView, reduced]);
  useFrame((dt) => {
    elapsed.current += dt * 1000;
    const k = clamp(elapsed.current / DRAW_MS, 0, 1);
    setT(T_START + (1 - T_START) * ease(k));
    if (k >= 1) {
      setRunning(false);
      timer.current = setTimeout(() => {
        if (!touched.current) setSel('en');
      }, 500);
    }
  }, running && inView);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const choose = (l: Lang) => {
    touched.current = true;
    setSel(l);
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-04.4"
      title="a letter to von Neumann"
      kicker="Gödel, 1956"
      hint="switch the language; hover a coloured term"
      footer="translation: Clote and Krajíček"
    >
      <div>
        <Letter lang={lang} focus={focus} setFocus={setFocus} />
        <div className="mt-4">
          {/* both layouts are rendered and CSS picks one, so nothing jumps at hydration */}
          <div className="hidden sm:block">
            <Sketch t={t} focus={focus} setFocus={setFocus} narrow={false} />
          </div>
          <div className="sm:hidden">
            <Sketch t={t} focus={focus} setFocus={setFocus} narrow />
          </div>
        </div>
        <div className="mt-3 border-t border-line pt-3">
          <Seg<Lang>
            label="letter"
            value={lang}
            onChange={choose}
            bind={bind}
            options={[
              { value: 'de', label: 'Deutsch' },
              { value: 'en', label: 'English' },
            ]}
          />
        </div>
      </div>
      <Caption>
        φ(n) is the most steps a machine needs to decide whether a formula has a proof n symbols
        long. Gödel could show φ(n) ≥ k·n and asked how slowly it can grow for the best machine:
        near k·n or k·n² (drawn with k = 1, a billion steps a second) would be far below trial and
        error.
      </Caption>
    </Specimen>
  );
}
