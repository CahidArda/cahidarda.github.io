/**
 * Instruments series ("Measuring the World"): the shared design system. Every widget in the
 * landing page and the three posts (time, sky, earth) is drawn as a PLATE from an old
 * technical drawing: parchment, sepia ink, hairline dimension lines with tick arrows, one
 * red-ochre accent line, serif italic for annotations, mono for numbers. Use these pieces
 * instead of re-styling per widget, so the four pages read as one authored set.
 *
 * ── Colour roles (CSS vars in global.css, light + .dark) ─────────────────────────────────
 *   ink · soft · faint   three weights of sepia ink (`--in-ink`, `--in-ink-soft`, `--in-ink-faint`)
 *   paper · deep         parchment and its darker margin tone (`--in-paper`, `--in-paper-deep`)
 *   red                  THE accent (red ochre). One line per plate: the thing to look at.
 *   brass · sky          metal parts; sky/water (muted indigo)
 *   In SVG use `style={{ stroke: C.ink }}` (never `stroke="var(...)"`, it does not resolve).
 *
 * ── Frame & controls ──────────────────────────────────────────────────────────────────────
 *   <Plate n={3} title="The anchor escapement" kicker="drag to turn" caption={<>…</>} wide?>
 *     Root has class `fx in-plate`. `wide` adds `fx-wide` (grows into the empty right grid
 *     column on wide desktops; ordinary width elsewhere). In MDX wrap it in
 *     `<div class="fx-figure">` like every other widget. `label="Levha"` for Ottoman flavour.
 *   <Controls> row; <Segmented options value onChange preview?>; <Slider label value min max
 *     step onChange format?>; <Readout>. `preview` = the bind fn from useHoverPreview.
 *
 * ── Worked equations: <Working><Eq>…<Sym role>α</Sym>…<V role>15.45°</V>…</Eq></Working>
 *   Formula plus the live numbers, each coloured like its element in the drawing (see below).
 *
 * ── SVG drawing primitives (one coordinate system per drawing; units = viewBox units) ────
 *   <HatchDefs id? gap? angle? role?/>   pattern for sections: fill={`url(#${id})`} via style
 *   <DimLine x1 y1 x2 y2 label offset? ends='arrow'|'tick' role? size?/>
 *   <ArcScale cx cy r from to major minor? labelEvery? label? tick? inward? role? size?/>
 *       Angles in DEGREES, maths convention: 0° = +x (right), increasing COUNTER-clockwise on
 *       screen (y up). `polar(cx, cy, r, deg)` uses the same convention.
 *   <Leader x1 y1 x2 y2 label anchor? dot? role? size?/>   (x1,y1) = the thing, (x2,y2) = label
 *   <InkText x y kind='note'|'label'|'num'|'title' anchor? role? size?>text</InkText>
 *   Text gets a paper-coloured halo (paint-order stroke) so it stays legible over lines.
 *
 * ── 3D: <BlueprintCanvas> renders three.js meshes as an engraved line drawing ─────────────
 *   three is imported DYNAMICALLY inside the canvas, so a widget that only uses the SVG parts
 *   never downloads it (~180 KB gzip, its own chunk). `build(THREE, ctx)` runs ONCE when WebGL
 *   is ready (it closes over first-render props: pass live values through `params`). Add
 *   objects to `ctx.root` (the group the reader rotates, centred on the origin), call
 *   `ctx.inkify(obj, opts)` to turn meshes into paper fill + ink edges (EdgesGeometry,
 *   `threshold` deg) + a silhouette outline (`outline` px, for cylinders/tori/spheres) +
 *   optional dashed `hidden` lines, and return `update(dt, t, params)` (return true to ask for
 *   another frame). `ctx.lineMaterial(role, {dashed})` / `ctx.fillMaterial(role)` give
 *   theme-tracking materials for your own rays, threads, pointers. Rendering is on demand: only
 *   while on screen, and only when `animate` is on (and motion not reduced), the reader drags,
 *   `params` changes identity, or you call `ctx.invalidate()`. Colours re-read on theme toggle.
 *   Camera: orthographic, looking down -z, `fit` = half the view height in world units.
 *   Input: mouse drag = yaw + pitch; touch = horizontal drags only (vertical swipes scroll the
 *   page; wheel is never captured); arrow keys when focused. The canvas draws its own
 *   "drag to turn" hint, so do not repeat it in the Plate kicker. Without WebGL the `poster`
 *   stays (also what SSR and the first paint show), so give every 3D plate a real SVG poster.
 *
 *   const params = useMemo(() => ({ length }), [length]);   // live values for update()
 *   <Plate n={2} title="Pendulum and anchor" caption="Drag the slider to lengthen the rod.">
 *     <BlueprintCanvas aspect={4 / 3} fit={1.4} animate params={params}
 *       poster={<MyPosterSvg />} label="Pendulum clock movement, rotatable"
 *       build={(T, ctx) => {
 *         const bob = new T.Mesh(new T.CylinderGeometry(0.3, 0.3, 0.06, 48));
 *         bob.rotation.x = Math.PI / 2; ctx.inkify(bob, { hidden: true }); ctx.root.add(bob);
 *         return (dt, t, p) => { bob.position.x = Math.sin(t * 2) * p.length; };
 *       }} />
 *     <Controls><Slider label="length" value={length} min={0.2} max={1} step={0.01}
 *       onChange={setLength} format={(v) => `${v.toFixed(2)} m`} /></Controls>
 *   </Plate>
 *   Worked examples: ./time/PendulumModel.tsx, ./sky/SextantModel.tsx. SVG example: ./PrecisionLadder.tsx.
 *
 * ── Step-by-step plates: <StepPlate> ──────────────────────────────────────────────────────
 *   A procedure: numbered steps ("Point at the sun", "Read the arc", "Put it into the
 *   equation"), one action per step, beside a STABLE REFERENCE drawing of what is being
 *   measured (same layout every step, the current step's quantity highlighted) and a
 *   persistent EQUATION panel whose numbers fill in as steps complete. Controlled: you own
 *   `step` and every value; StepPlate draws the rail, the text, the slots and Back/Next.
 *
 *   <StepPlate
 *     n={5} title="Latitude by a noon sight" wide          // Plate props (no children)
 *     steps={[                                              // StepDef[]
 *       { id: 'noon', title: 'Wait for noon', text: <>…</>, controls: <Slider …/>,
 *         gated: true, done: atNoon, hint: 'Drag the time to noon' },
 *       { id: 'read', title: 'Read the arc', text: <>…</> },  // free step: Next always on
 *     ]}
 *     step={step} onStep={setStep}                          // controlled current step
 *     view={(i) => <Eyepiece step={i} …/>}                  // instrument view for step i
 *     reference={(i) => <EarthSection highlight={i} …/>}    // SAME layout for every i
 *     viewLabel="Through the sextant" referenceLabel="What you are measuring"
 *     equation={<Working>…<Known when={readDone} role="sky">36°43′</Known>…</Working>}
 *     settings={<Slider label="date" …/>}                   // optional plate-level controls
 *     onSkip={() => fillEverything()}                      // optional "skip to result"
 *     caption={<>…</>} />
 *
 *   - Rail: numbered buttons with states current / done / pending. A step can be visited when
 *     every GATED step before it is `done`; Next is enabled when the current step is free or
 *     done (else the `hint` shows beside it). Hovering a visitable rail step PREVIEWS it
 *     (useHoverPreview: text, view and reference switch, controls are hidden); click commits.
 *     Keyboard: Tab to the rail, ←/→/Home/End move between steps; Back/Next are buttons.
 *   - `view(i)`/`reference(i)` are called with the displayed step (committed or previewed).
 *     Both get the same column width; size SVGs with useMeasuredWidth inside them.
 *   - Equation: build it with Working/Eq/Sym/V; wrap each live number in
 *     <Known when={…} role ph?> so it reads a faint "?" (or `ph`) until its step is done.
 *   - Layout (container query on the plate body): two columns (view | reference, controls and
 *     nav under the view, equation under the reference) when the plate body is at least 700px;
 *     one column below that (each view stays at least ~340px wide) (rail compact: numbers only), order rail, text, view, controls,
 *     nav, reference, equation. No transitions under prefers-reduced-motion.
 *   - Use `wide` for procedure plates: two views need the room.
 *   Worked example: ./sky/NoonSightProcedure.tsx (geometry in ./sky/noonProcGeom.ts).
 *
 * Hooks re-exported from ../dataplat/shared: useReducedMotion, useInView, useHoverPreview.
 */
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import type { ReactNode, Ref } from 'react';
import type * as THREEns from 'three';
import { useReducedMotion, useInView, useHoverPreview } from '../dataplat/shared';

export { useReducedMotion, useInView, useHoverPreview };

/* ─────────────────────────────── colours ─────────────────────────────── */

export type Role = 'ink' | 'soft' | 'faint' | 'paper' | 'deep' | 'red' | 'brass' | 'sky';

/** CSS colour per role, for `style={{ stroke: C.ink }}` in SVG and HTML. */
export const C: Record<Role, string> = {
  ink: 'var(--in-ink)',
  soft: 'var(--in-ink-soft)',
  faint: 'var(--in-ink-faint)',
  paper: 'var(--in-paper)',
  deep: 'var(--in-paper-deep)',
  red: 'var(--in-red)',
  brass: 'var(--in-brass)',
  sky: 'var(--in-sky)',
};

/** The base literal custom property behind each role (what canvas/WebGL must read). */
const ROLE_VAR: Record<Role, string> = {
  ink: '--in-ink',
  soft: '--in-ink-soft',
  faint: '--in-ink-faint',
  paper: '--in-paper',
  deep: '--in-paper-deep',
  red: '--in-red',
  brass: '--in-brass',
  sky: '--in-sky',
};

/** Resolve every role to a literal colour string (e.g. "#3a2a1c") from an element's styles. */
export function readInk(el: Element = document.documentElement): Record<Role, string> {
  const cs = getComputedStyle(el);
  const out = {} as Record<Role, string>;
  (Object.keys(ROLE_VAR) as Role[]).forEach((r) => {
    out[r] = cs.getPropertyValue(ROLE_VAR[r]).trim() || '#000';
  });
  return out;
}

/** Calls `fn` whenever the site theme flips (the `dark` class on <html>). */
export function onThemeChange(fn: () => void): () => void {
  const obs = new MutationObserver(fn);
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  return () => obs.disconnect();
}

/** Width of an element in CSS px (ResizeObserver). Size an SVG viewBox to it so 1 unit = 1px
 *  and text stays legible at 380px instead of shrinking with a fixed viewBox. */
export function useMeasuredWidth<T extends Element = HTMLDivElement>(initial = 640) {
  const ref = useRef<T | null>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    // a hidden ancestor (display: none) reports 0: keep the last real width instead of laying
    // the drawing out at zero (which gives negative rect sizes)
    const ro = new ResizeObserver(([e]) => {
      const w = Math.round(e.contentRect.width);
      if (w > 0) setW(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/* ─────────────────────────────── geometry ─────────────────────────────── */

const RAD = Math.PI / 180;
/** Point on a circle, angle in degrees, 0° = right, counter-clockwise on screen. */
export const polar = (cx: number, cy: number, r: number, deg: number) => ({
  x: cx + r * Math.cos(deg * RAD),
  y: cy - r * Math.sin(deg * RAD),
});
/** SVG arc path between two angles (degrees, same convention as `polar`). */
export function arcPath(cx: number, cy: number, r: number, from: number, to: number) {
  const a = polar(cx, cy, r, from);
  const b = polar(cx, cy, r, to);
  const large = Math.abs(to - from) > 180 ? 1 : 0;
  const sweep = to > from ? 0 : 1; // CCW on screen = sweep 0 in SVG's y-down space
  return `M ${a.x} ${a.y} A ${r} ${r} 0 ${large} ${sweep} ${b.x} ${b.y}`;
}

/* ─────────────────────────────── the plate ─────────────────────────────── */

function RegMark({ style }: { style: React.CSSProperties }) {
  return (
    <svg className="in-reg" viewBox="0 0 14 14" style={style} aria-hidden="true">
      <circle cx="7" cy="7" r="3.2" />
      <path d="M7 0 V14 M0 7 H14" />
    </svg>
  );
}

export function Plate({
  n,
  title,
  kicker,
  caption,
  label = 'Plate',
  wide,
  id,
  rootRef,
  className,
  children,
}: {
  /** Plate number shown as "Plate n". Omit for an unnumbered plate. */
  n?: number | string;
  title: string;
  /** Short mono hint at the right of the header, e.g. "drag to turn" or a unit. */
  kicker?: string;
  /** Italic serif caption under the drawing. */
  caption?: ReactNode;
  /** "Plate" (default) or e.g. "Levha". */
  label?: string;
  /** Grow into the empty right column on wide screens (adds `.fx-wide`). */
  wide?: boolean;
  id?: string;
  rootRef?: Ref<HTMLElement>;
  className?: string;
  children: ReactNode;
}) {
  const inset = 5;
  return (
    <figure
      ref={rootRef}
      id={id}
      className={`fx in-plate${wide ? ' fx-wide' : ''}${className ? ` ${className}` : ''}`}
    >
      <RegMark style={{ top: inset, left: inset }} />
      <RegMark style={{ top: inset, right: inset }} />
      <RegMark style={{ bottom: inset, left: inset }} />
      <RegMark style={{ bottom: inset, right: inset }} />
      <figcaption className="in-plate-head">
        {n !== undefined && (
          <span className="in-plate-no">
            {label} {n}
          </span>
        )}
        <span className="in-plate-title">{title}</span>
        {kicker && <span className="in-plate-kicker">{kicker}</span>}
      </figcaption>
      {children}
      {caption && <div className="in-plate-cap">{caption}</div>}
    </figure>
  );
}

/* ─────────────────────────────── controls ─────────────────────────────── */

export function Controls({ children }: { children: ReactNode }) {
  return <div className="in-controls">{children}</div>;
}

/** Segmented buttons. Pass `preview` (the bind fn from useHoverPreview) for hover-preview. */
export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  preview,
  label,
}: {
  options: { value: T; label: string }[];
  /** The COMMITTED value (keeps aria-pressed on the chosen option while another is previewed). */
  value: T;
  onChange: (v: T) => void;
  preview?: (v: T) => { onMouseEnter: () => void; onMouseLeave: () => void };
  label?: string;
}) {
  return (
    <div className="in-seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          aria-pressed={value === o.value}
          onClick={() => onChange(o.value)}
          {...(preview ? preview(o.value) : {})}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = (v) => String(v),
  width,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  /** CSS width of the track, default 9rem. */
  width?: string;
}) {
  const id = useId();
  return (
    <span className="in-field">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        className="in-range"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={width ? { width } : undefined}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="in-readout">{format(value)}</span>
    </span>
  );
}

/** A labelled mono numeral, e.g. <Readout label="period">2.01 s</Readout>. */
export function Readout({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="in-field">
      {label} <span className="in-readout">{children}</span>
    </span>
  );
}

/* ───────────────────────── worked equations ───────────────────────── */

/**
 * Worked equations: show the formula WITH the live numbers substituted, and colour each
 * quantity with the same role as the line, angle or point it belongs to in the drawing, so
 * the reader can match them at a glance (angle α drawn in sky → α and 15.45° in sky).
 *
 *   <Working>
 *     <Eq><Sym role="red">h</Sym> = <Sym role="ink">b</Sym> · sin <Sym role="sky">α</Sym> …</Eq>
 *     <Eq><V role="red">76.0 m</V> = <V>75 m</V> · sin <V role="sky">15.45°</V> …</Eq>
 *   </Working>
 *
 * <Sym> = a symbol (serif italic), <V> = a live value (mono, tabular). Default role: ink.
 * Keep one line for the symbolic form and one for the substituted form; a third line for an
 * error range is fine. Lines wrap at narrow widths, so keep each under ~45 characters.
 */
export function Working({ children, label }: { children: ReactNode; label?: string }) {
  return (
    <div className="in-working" aria-label={label ?? 'Worked equation'}>
      {children}
    </div>
  );
}
export function Eq({ children }: { children: ReactNode }) {
  return <div className="in-eq">{children}</div>;
}
export function Sym({ children, role = 'ink' }: { children: ReactNode; role?: Role }) {
  return (
    <span className="in-sym" style={{ color: C[role] }}>
      {children}
    </span>
  );
}
export function V({ children, role = 'ink' }: { children: ReactNode; role?: Role }) {
  return (
    <span className="in-v" style={{ color: C[role] }}>
      {children}
    </span>
  );
}

/* ─────────────────────────────── SVG primitives ─────────────────────────────── */

const FONT = {
  note: { fontFamily: 'var(--font-display)', fontStyle: 'italic' as const },
  title: { fontFamily: 'var(--font-display)', fontStyle: 'normal' as const },
  label: {
    fontFamily: 'var(--font-mono)',
    letterSpacing: '0.12em',
    textTransform: 'uppercase' as const,
  },
  num: { fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' as const },
};
export type TextKind = keyof typeof FONT;

/** Text in the drawing's voice. A paper halo keeps it legible over lines. */
export function InkText({
  x,
  y,
  kind = 'note',
  anchor = 'start',
  baseline = 'alphabetic',
  role = 'ink',
  size = 13,
  rotate,
  halo = true,
  children,
}: {
  x: number;
  y: number;
  kind?: TextKind;
  anchor?: 'start' | 'middle' | 'end';
  baseline?: 'alphabetic' | 'middle' | 'hanging' | 'central';
  role?: Role;
  size?: number;
  /** Degrees, SVG convention (clockwise), about (x, y). */
  rotate?: number;
  halo?: boolean;
  children: ReactNode;
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      dominantBaseline={baseline}
      transform={rotate ? `rotate(${rotate} ${x} ${y})` : undefined}
      style={{
        ...FONT[kind],
        fontSize: kind === 'label' ? size * 0.78 : size,
        fill: C[role],
        ...(halo
          ? {
              paintOrder: 'stroke',
              stroke: C.paper,
              strokeWidth: 3,
              strokeLinejoin: 'round' as const,
            }
          : {}),
      }}
    >
      {children}
    </text>
  );
}

/** Section hatching. Use with style={{ fill: `url(#${id})` }}. Render once per <svg>. */
export function HatchDefs({
  id = 'in-hatch',
  gap = 5,
  angle = 45,
  role = 'soft',
  width = 0.7,
}: {
  id?: string;
  gap?: number;
  angle?: number;
  role?: Role;
  width?: number;
}) {
  return (
    <defs>
      <pattern
        id={id}
        width={gap}
        height={gap}
        patternUnits="userSpaceOnUse"
        patternTransform={`rotate(${angle})`}
      >
        <line x1="0" y1="0" x2="0" y2={gap} style={{ stroke: C[role], strokeWidth: width }} />
      </pattern>
    </defs>
  );
}

/** An open arrowhead at (x, y) pointing along unit vector (ux, uy), drawn as a path. */
function arrowHead(x: number, y: number, ux: number, uy: number, s: number) {
  const px = -uy;
  const py = ux;
  const bx = x - ux * s * 1.9;
  const by = y - uy * s * 1.9;
  return `M ${bx + px * s * 0.55} ${by + py * s * 0.55} L ${x} ${y} L ${bx - px * s * 0.55} ${by - py * s * 0.55}`;
}

/**
 * Dimension line between two measured points. `offset` pushes the dimension line off the
 * object (positive = to the left of the direction p1→p2 on screen) and draws extension lines.
 * `ends`: 'arrow' (filled-look open arrows) or 'tick' (architect's 45° slashes).
 */
export function DimLine({
  x1,
  y1,
  x2,
  y2,
  label,
  offset = 0,
  ends = 'arrow',
  role = 'soft',
  labelRole = 'ink',
  size = 12,
  width = 0.8,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label?: ReactNode;
  offset?: number;
  ends?: 'arrow' | 'tick';
  role?: Role;
  labelRole?: Role;
  size?: number;
  width?: number;
}) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  // screen-left normal of p1→p2 (y down): (uy, -ux)
  const nx = uy;
  const ny = -ux;
  const a = { x: x1 + nx * offset, y: y1 + ny * offset };
  const b = { x: x2 + nx * offset, y: y2 + ny * offset };
  const stroke = { stroke: C[role], strokeWidth: width, fill: 'none' };
  const s = 3.2;
  const ext = offset !== 0 ? Math.sign(offset) * 4 : 0;
  let ang = (Math.atan2(dy, dx) * 180) / Math.PI;
  if (ang > 90) ang -= 180;
  if (ang < -90) ang += 180;
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  return (
    <g>
      {offset !== 0 && (
        <path
          d={`M ${x1 + nx * Math.sign(offset) * 2} ${y1 + ny * Math.sign(offset) * 2} L ${a.x + nx * ext} ${a.y + ny * ext} M ${x2 + nx * Math.sign(offset) * 2} ${y2 + ny * Math.sign(offset) * 2} L ${b.x + nx * ext} ${b.y + ny * ext}`}
          style={{ ...stroke, strokeWidth: width * 0.8 }}
        />
      )}
      <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} style={stroke} />
      {ends === 'arrow' ? (
        <path
          d={`${arrowHead(a.x, a.y, -ux, -uy, s)} ${arrowHead(b.x, b.y, ux, uy, s)}`}
          style={stroke}
        />
      ) : (
        <path
          d={[a, b]
            .map(
              (p) =>
                `M ${p.x - (ux + nx) * s} ${p.y - (uy + ny) * s} L ${p.x + (ux + nx) * s} ${p.y + (uy + ny) * s}`,
            )
            .join(' ')}
          style={{ ...stroke, strokeWidth: width * 1.4 }}
        />
      )}
      {label !== undefined && (
        <InkText
          x={mx}
          y={my}
          kind="num"
          anchor="middle"
          baseline="central"
          size={size}
          role={labelRole}
          rotate={ang}
        >
          {label}
        </InkText>
      )}
    </g>
  );
}

/**
 * Graduated arc (quadrant limb, sextant arc, theodolite circle). Degrees, 0° = right, CCW.
 * Ticks every `minor` degrees, longer ticks every `major`, numerals every `labelEvery`
 * (default = major). `label(v)` formats a numeral; `inward` puts ticks inside the arc.
 * `value(a)` maps an angle to the number printed (e.g. a sextant prints 2× the arc angle).
 */
export function ArcScale({
  cx,
  cy,
  r,
  from,
  to,
  major = 10,
  minor = 1,
  labelEvery,
  label = (v) => `${v}`,
  value = (a) => a,
  tick = 8,
  inward = false,
  role = 'ink',
  size = 10,
  rim = true,
  width = 0.8,
}: {
  cx: number;
  cy: number;
  r: number;
  from: number;
  to: number;
  major?: number;
  minor?: number;
  labelEvery?: number;
  label?: (v: number) => string;
  value?: (angle: number) => number;
  tick?: number;
  inward?: boolean;
  role?: Role;
  size?: number;
  rim?: boolean;
  width?: number;
}) {
  const lo = Math.min(from, to);
  const hi = Math.max(from, to);
  const every = labelEvery ?? major;
  const dir = inward ? -1 : 1;
  const ticks: string[] = [];
  const nums: ReactNode[] = [];
  const eps = 1e-6;
  for (let a = lo; a <= hi + eps; a += minor) {
    const isMajor =
      Math.abs(((a - lo) / major) % 1) < eps || Math.abs(((a - lo) / major) % 1) > 1 - eps;
    const isHalf = !isMajor && major / minor >= 4 && Math.abs(((a - lo) / (major / 2)) % 1) < eps;
    const L = isMajor ? tick : isHalf ? tick * 0.7 : tick * 0.45;
    const p = polar(cx, cy, r, a);
    const q = polar(cx, cy, r + dir * L, a);
    ticks.push(`M ${p.x} ${p.y} L ${q.x} ${q.y}`);
    const onLabel =
      Math.abs(((a - lo) / every) % 1) < eps || Math.abs(((a - lo) / every) % 1) > 1 - eps;
    if (onLabel) {
      const t = polar(cx, cy, r + dir * (tick + size * 0.95), a);
      nums.push(
        <InkText
          key={a}
          x={t.x}
          y={t.y}
          kind="num"
          anchor="middle"
          baseline="central"
          size={size}
          role={role}
          rotate={90 - a}
          halo={false}
        >
          {label(Math.round(value(a) * 1000) / 1000)}
        </InkText>,
      );
    }
  }
  return (
    <g>
      {rim && (
        <path
          d={arcPath(cx, cy, r, lo, hi)}
          style={{ fill: 'none', stroke: C[role], strokeWidth: width * 1.2 }}
        />
      )}
      <path d={ticks.join(' ')} style={{ fill: 'none', stroke: C[role], strokeWidth: width }} />
      {nums}
    </g>
  );
}

/** Annotation: a small dot on the thing, a thin leader, a short shoulder, an italic label. */
export function Leader({
  x1,
  y1,
  x2,
  y2,
  label,
  anchor,
  dot = true,
  role = 'soft',
  labelRole = 'ink',
  size = 13,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: ReactNode;
  anchor?: 'start' | 'end';
  dot?: boolean;
  role?: Role;
  labelRole?: Role;
  size?: number;
}) {
  const side = anchor ?? (x2 >= x1 ? 'start' : 'end');
  const sh = side === 'start' ? 10 : -10;
  return (
    <g>
      {dot && <circle cx={x1} cy={y1} r={1.8} style={{ fill: C[role] }} />}
      <path
        d={`M ${x1} ${y1} L ${x2} ${y2} L ${x2 + sh} ${y2}`}
        style={{ fill: 'none', stroke: C[role], strokeWidth: 0.7 }}
      />
      <InkText
        x={x2 + sh + (side === 'start' ? 4 : -4)}
        y={y2}
        baseline="central"
        anchor={side}
        size={size}
        role={labelRole}
      >
        {label}
      </InkText>
    </g>
  );
}

/* ─────────────────────────────── 3D: the blueprint renderer ─────────────────────────────── */

export type THREE = typeof THREEns;

export interface InkifyOptions {
  /** Fill role for the faces, or 'none' for a wireframe-only part. Default 'paper'. */
  fill?: Role | 'none';
  /** Ink role for the visible edges. Default 'ink'. */
  edge?: Role;
  /** EdgesGeometry threshold angle in degrees: higher = fewer lines on curved parts. Default 24. */
  threshold?: number;
  /** Also draw edges hidden behind other faces, dashed and faint. Default false. */
  hidden?: boolean;
  /** Role for hidden lines. Default 'faint'. */
  hiddenEdge?: Role;
  /** Silhouette outline width in CSS px (inverted hull), for smooth parts whose outline
   *  EdgesGeometry cannot see (cylinders, spheres, tori). 0 = off. Default 1.25. */
  outline?: number;
}

export interface BlueprintCtx {
  three: THREE;
  scene: THREEns.Scene;
  camera: THREEns.OrthographicCamera;
  /** The group the reader rotates. Add your model here (centred on the origin). */
  root: THREEns.Group;
  /** Turn a mesh (and its children meshes) into paper fill + ink edges (+ silhouette, + hidden lines). */
  inkify: (obj: THREEns.Object3D, opts?: InkifyOptions) => void;
  /** A theme-tracking line material, for your own Line/LineSegments (axes, rays, threads). */
  lineMaterial: (
    role?: Role,
    opts?: { dashed?: boolean; dash?: number; gap?: number; onTop?: boolean },
  ) => THREEns.LineBasicMaterial;
  /** A theme-tracking flat material for solid fills (e.g. a red-ochre pointer). */
  fillMaterial: (role?: Role) => THREEns.MeshBasicMaterial;
  /** Request one more frame (after you changed something outside update). */
  invalidate: () => void;
  /** True when the reader prefers reduced motion; render a meaningful still. */
  reduced: boolean;
}

/** Called every rendered frame. dt, t in seconds (t frozen under reduced motion).
 *  Return true to request another frame even when `animate` is off (e.g. a settling glide). */
export type BlueprintUpdate<P> = (dt: number, t: number, params: P) => boolean | void;

export function BlueprintCanvas<P = undefined>({
  build,
  params,
  aspect = 4 / 3,
  fit = 1,
  animate = false,
  yaw: yaw0 = -0.6,
  pitch: pitch0 = 0.35,
  pitchLimit = 1.2,
  rotatable = true,
  poster,
  label,
  hint = 'drag to turn',
}: {
  /** Runs once when WebGL is ready. Build the scene under ctx.root, return per-frame update. */
  build: (three: THREE, ctx: BlueprintCtx) => BlueprintUpdate<P> | void;
  /** Values from React state the scene reads in update(); a change triggers a frame. */
  params?: P;
  /** width / height of the drawing area. */
  aspect?: number;
  /** Half-height of the view in world units (orthographic camera). */
  fit?: number;
  /** Run the clock continuously while on screen (ignored under reduced motion). */
  animate?: boolean;
  /** Initial rotation of ctx.root (radians). */
  yaw?: number;
  pitch?: number;
  /** Max |pitch| in radians. */
  pitchLimit?: number;
  rotatable?: boolean;
  /** SVG/JSX shown before WebGL is ready (SSR) and kept if WebGL is unavailable. */
  poster?: ReactNode;
  /** Accessible description of the drawing. */
  label: string;
  hint?: string | null;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const reduced = useReducedMotion();
  const [state, setState] = useState<'poster' | 'live' | 'fallback'>('poster');
  const [touched, setTouched] = useState(false);
  const paramsRef = useRef(params);
  const kickRef = useRef<() => void>(() => {});
  const optsRef = useRef({ animate, reduced });
  optsRef.current = { animate, reduced };

  // A param change from React → one frame.
  useEffect(() => {
    paramsRef.current = params;
    kickRef.current();
  }, [params]);
  useEffect(() => kickRef.current(), [animate, reduced]);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    let disposed = false;
    let cleanup = () => {};

    import('three')
      .then((T) => {
        if (disposed) return;
        let renderer: THREEns.WebGLRenderer;
        try {
          renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
        } catch {
          setState('fallback');
          return;
        }
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        const scene = new T.Scene();
        const camera = new T.OrthographicCamera(-fit * aspect, fit * aspect, fit, -fit, 0.01, 100);
        camera.position.set(0, 0, 10);
        camera.lookAt(0, 0, 0);
        const root = new T.Group();
        scene.add(root);

        // ── theme-tracking materials ──
        let ink = readInk(wrap);
        const tracked: { m: { color: THREEns.Color }; role: Role }[] = [];
        const track = <M extends { color: THREEns.Color }>(m: M, role: Role) => {
          m.color.set(ink[role]);
          tracked.push({ m, role });
          return m;
        };
        const lineMaterial: BlueprintCtx['lineMaterial'] = (role = 'ink', o = {}) => {
          const m = o.dashed
            ? new T.LineDashedMaterial({ dashSize: o.dash ?? 0.04, gapSize: o.gap ?? 0.03 })
            : new T.LineBasicMaterial();
          if (o.onTop) {
            m.depthTest = false;
          }
          return track(m, role);
        };
        const fillMaterial: BlueprintCtx['fillMaterial'] = (role = 'paper') =>
          track(
            new T.MeshBasicMaterial({
              polygonOffset: true,
              polygonOffsetFactor: 1,
              polygonOffsetUnits: 1,
            }),
            role,
          );
        // Silhouettes: an "inverted hull", back faces pushed out along their normals by a
        // width given in screen px (orthographic camera: px → world is one scale factor).
        const hulls: { u: { value: number }; px: number }[] = [];
        const pxToWorld = () => (2 * fit) / Math.max(wrap.clientWidth / aspect, 1);
        const hullMaterial = (role: Role, px: number) => {
          const m = new T.MeshBasicMaterial({ side: T.BackSide });
          const u = { value: px * pxToWorld() };
          hulls.push({ u, px });
          m.onBeforeCompile = (sh) => {
            sh.uniforms.uThick = u;
            sh.vertexShader = sh.vertexShader
              .replace('void main() {', 'uniform float uThick;\nvoid main() {')
              .replace(
                '#include <begin_vertex>',
                'vec3 transformed = position + normalize(normal) * uThick;',
              );
          };
          return track(m, role);
        };
        const inkify: BlueprintCtx['inkify'] = (obj, o = {}) => {
          const meshes: THREEns.Mesh[] = [];
          obj.traverse((c) => {
            if ((c as THREEns.Mesh).isMesh && !c.userData.inkHull) meshes.push(c as THREEns.Mesh);
          });
          for (const mesh of meshes) {
            const outline = o.outline ?? 1.25;
            if (outline > 0) {
              const hull = new T.Mesh(mesh.geometry, hullMaterial(o.edge ?? 'ink', outline));
              hull.userData.inkHull = true;
              mesh.add(hull);
            }
            if (o.fill === 'none') {
              mesh.material = new T.MeshBasicMaterial({ colorWrite: false, depthWrite: false });
            } else {
              mesh.material = fillMaterial(o.fill ?? 'paper');
            }
            const eg = new T.EdgesGeometry(mesh.geometry, o.threshold ?? 24);
            const lines = new T.LineSegments(eg, lineMaterial(o.edge ?? 'ink'));
            lines.renderOrder = 1;
            mesh.add(lines);
            if (o.hidden) {
              const hm = lineMaterial(o.hiddenEdge ?? 'faint', { dashed: true });
              hm.depthFunc = T.GreaterDepth;
              hm.depthWrite = false;
              const h = new T.LineSegments(eg, hm);
              h.computeLineDistances();
              h.renderOrder = 2; // after all fills have written depth
              mesh.add(h);
            }
          }
        };
        const recolor = () => {
          ink = readInk(wrap);
          tracked.forEach(({ m, role }) => m.color.set(ink[role]));
          kick();
        };

        // ── sizing ──
        const resize = () => {
          const w = wrap.clientWidth;
          const h = w / aspect;
          renderer.setSize(w, h, false);
          camera.left = -fit * aspect;
          camera.right = fit * aspect;
          camera.top = fit;
          camera.bottom = -fit;
          camera.updateProjectionMatrix();
          const k = pxToWorld();
          hulls.forEach((h) => (h.u.value = h.px * k));
          kick();
        };

        // ── render loop (on demand) ──
        let raf = 0;
        let last = 0;
        let t = 0;
        let inView = true;
        let dirty = true;
        let more = false;
        let yaw = yaw0;
        let pitch = pitch0;
        let vYaw = 0;
        let dragging = false;
        root.rotation.set(pitch, yaw, 0, 'XYZ');

        const ctx: BlueprintCtx = {
          three: T,
          scene,
          camera,
          root,
          inkify,
          lineMaterial,
          fillMaterial,
          invalidate: () => kick(),
          reduced: optsRef.current.reduced,
        };
        const update = build(T, ctx) || undefined;

        const frame = (now: number) => {
          raf = 0;
          const dt = last ? Math.min((now - last) / 1000, 0.1) : 0;
          last = now;
          const { animate: anim, reduced: red } = optsRef.current;
          ctx.reduced = red;
          const running = anim && !red;
          if (running) t += dt;
          // inertia after a flick
          if (!dragging && Math.abs(vYaw) > 0.0005 && !red) {
            yaw += vYaw * dt * 60;
            vYaw *= Math.pow(0.9, dt * 60);
          } else if (!dragging) vYaw = 0;
          root.rotation.set(pitch, yaw, 0, 'XYZ');
          more = update ? update(dt, t, paramsRef.current as P) === true : false;
          renderer.render(scene, camera);
          dirty = false;
          if (inView && (running || more || dragging || Math.abs(vYaw) > 0.0005)) {
            raf = requestAnimationFrame(frame);
          } else {
            last = 0;
          }
        };
        function kick() {
          dirty = true;
          if (!raf && inView && !disposed) raf = requestAnimationFrame(frame);
        }
        kickRef.current = kick;

        // ── drag to rotate (never steals vertical page scroll on touch) ──
        let px = 0;
        let py = 0;
        let pid: number | null = null;
        let decided = false;
        const down = (e: PointerEvent) => {
          if (!rotatable || pid !== null) return;
          pid = e.pointerId;
          px = e.clientX;
          py = e.clientY;
          vYaw = 0;
          // mouse/pen: rotate immediately; touch: wait to see if the drag is sideways
          decided = e.pointerType !== 'touch';
          if (decided) {
            dragging = true;
            canvas.setPointerCapture(e.pointerId);
            setTouched(true);
            kick();
          }
        };
        const move = (e: PointerEvent) => {
          if (e.pointerId !== pid) return;
          const dx = e.clientX - px;
          const dy = e.clientY - py;
          if (!decided) {
            if (Math.hypot(dx, dy) < 6) return;
            decided = true;
            if (Math.abs(dx) < Math.abs(dy)) {
              pid = null; // vertical: it is a scroll, let the page have it
              return;
            }
            dragging = true;
            canvas.setPointerCapture(e.pointerId);
            setTouched(true);
          }
          if (!dragging) return;
          px = e.clientX;
          py = e.clientY;
          const k = 3 / Math.max(canvas.clientWidth, 1);
          yaw += dx * k;
          vYaw = dx * k * 0.5;
          if (e.pointerType !== 'touch') {
            pitch = Math.max(-pitchLimit, Math.min(pitchLimit, pitch + dy * k));
          }
          kick();
        };
        const up = (e: PointerEvent) => {
          if (e.pointerId !== pid) return;
          pid = null;
          dragging = false;
          kick();
        };
        const key = (e: KeyboardEvent) => {
          if (!rotatable) return;
          const step = 0.12;
          if (e.key === 'ArrowLeft') yaw -= step;
          else if (e.key === 'ArrowRight') yaw += step;
          else if (e.key === 'ArrowUp') pitch = Math.max(-pitchLimit, pitch - step);
          else if (e.key === 'ArrowDown') pitch = Math.min(pitchLimit, pitch + step);
          else return;
          e.preventDefault();
          kick();
        };
        canvas.addEventListener('pointerdown', down);
        canvas.addEventListener('pointermove', move);
        canvas.addEventListener('pointerup', up);
        canvas.addEventListener('pointercancel', up);
        wrap.addEventListener('keydown', key);

        const ro = new ResizeObserver(resize);
        ro.observe(wrap);
        const io = new IntersectionObserver(([en]) => {
          inView = en.isIntersecting;
          if (inView && (dirty || optsRef.current.animate)) kick();
        });
        io.observe(wrap);
        const unTheme = onThemeChange(recolor);

        resize();
        setState('live');

        cleanup = () => {
          cancelAnimationFrame(raf);
          ro.disconnect();
          io.disconnect();
          unTheme();
          canvas.removeEventListener('pointerdown', down);
          canvas.removeEventListener('pointermove', move);
          canvas.removeEventListener('pointerup', up);
          canvas.removeEventListener('pointercancel', up);
          wrap.removeEventListener('keydown', key);
          kickRef.current = () => {};
          scene.traverse((o) => {
            const g = (o as THREEns.Mesh).geometry;
            if (g) g.dispose();
            const m = (o as THREEns.Mesh).material;
            if (Array.isArray(m)) m.forEach((x) => x.dispose());
            else if (m) m.dispose();
          });
          renderer.dispose();
        };
      })
      .catch(() => !disposed && setState('fallback'));

    return () => {
      disposed = true;
      cleanup();
    };
    // build/fit/aspect are treated as mount-time constants
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      ref={wrapRef}
      className="in-canvas"
      style={{ aspectRatio: String(aspect) }}
      role="img"
      aria-label={label}
      tabIndex={rotatable && state === 'live' ? 0 : -1}
    >
      <div className="in-poster" style={{ opacity: state === 'live' ? 0 : 1 }} aria-hidden="true">
        {poster ?? <PosterPlaceholder aspect={aspect} />}
      </div>
      <canvas ref={canvasRef} style={{ opacity: state === 'live' ? 1 : 0 }} />
      {rotatable && state === 'live' && hint && (
        <span className="in-hint" style={{ opacity: touched ? 0 : 1 }}>
          {hint}
        </span>
      )}
    </div>
  );
}

/** Default poster: a faint compass rose, so the plate is never an empty box. */
export function PosterPlaceholder({ aspect = 4 / 3 }: { aspect?: number }) {
  const W = 400;
  const H = W / aspect;
  const r = Math.min(W, H) * 0.32;
  return (
    <svg viewBox={`0 0 ${W} ${H}`}>
      <circle
        cx={W / 2}
        cy={H / 2}
        r={r}
        style={{ fill: 'none', stroke: C.faint, strokeWidth: 0.8 }}
      />
      <circle
        cx={W / 2}
        cy={H / 2}
        r={r * 0.62}
        style={{ fill: 'none', stroke: C.faint, strokeWidth: 0.6, strokeDasharray: '3 3' }}
      />
      <path
        d={`M ${W / 2 - r * 1.15} ${H / 2} H ${W / 2 + r * 1.15} M ${W / 2} ${H / 2 - r * 1.15} V ${H / 2 + r * 1.15}`}
        style={{ stroke: C.faint, strokeWidth: 0.6 }}
      />
    </svg>
  );
}

/** Small helper for widgets that want a stable callback identity across renders. */
export function useEvent<A extends unknown[], R>(fn: (...a: A) => R) {
  const ref = useRef(fn);
  ref.current = fn;
  return useCallback((...a: A) => ref.current(...a), []);
}

/* ─────────────────────────────── step-by-step plates ─────────────────────────────── */

export type StepDef = {
  id: string;
  /** Short imperative title for the rail, e.g. "Point at the sun". */
  title: string;
  /** One or two sentences: what to do in this step and why. */
  text: ReactNode;
  /** The step's own control(s) (sliders, buttons). Hidden while another step is previewed. */
  controls?: ReactNode;
  /** Gated: Next stays off until `done` (the reader did the action). Default: free. */
  gated?: boolean;
  /** The step's action is complete (its quantity is known). */
  done?: boolean;
  /** Shown beside a disabled Next, e.g. "Bring the sun's edge onto the horizon". */
  hint?: string;
};

/** A live value in a StepPlate equation: the value once `when`, a faint placeholder before. */
export function Known({
  when,
  role = 'ink',
  ph = '?',
  children,
}: {
  when: boolean;
  role?: Role;
  ph?: ReactNode;
  children: ReactNode;
}) {
  return when ? (
    <V role={role}>{children}</V>
  ) : (
    <span className="in-v in-unknown" style={{ color: C.faint }} aria-label="not known yet">
      {ph}
    </span>
  );
}

export function StepPlate({
  steps,
  step,
  onStep,
  view,
  reference,
  viewLabel = 'The instrument',
  referenceLabel = 'What is being measured',
  equation,
  settings,
  onSkip,
  skipLabel = 'Skip to the result',
  ...plate
}: {
  steps: StepDef[];
  /** Current (committed) step index. */
  step: number;
  onStep: (i: number) => void;
  /** Instrument view for step i (the committed or the previewed step). */
  view: (i: number) => ReactNode;
  /** The stable reference drawing; same layout for every i, highlight step i's quantity. */
  reference: (i: number) => ReactNode;
  viewLabel?: string;
  referenceLabel?: string;
  /** Persistent equation panel (Working/Eq/Sym/Known). */
  equation: ReactNode;
  /** Plate-level controls that are not a step (date, place). */
  settings?: ReactNode;
  /** Fill every step's result; StepPlate then jumps to the last step. */
  onSkip?: () => void;
  skipLabel?: string;
} & Omit<Parameters<typeof Plate>[0], 'children'>) {
  const n = steps.length;
  const cur = Math.max(0, Math.min(n - 1, step));
  // a step can be visited when every gated step before it is done
  const reach = (() => {
    let r = 0;
    while (r < n - 1 && (!steps[r].gated || steps[r].done)) r++;
    return r;
  })();
  const [visited, setVisited] = useState(cur);
  useEffect(() => setVisited((v) => Math.max(v, cur)), [cur]);
  const [shown, bind] = useHoverPreview<number>(cur);
  const show = shown <= reach ? shown : cur;
  const previewing = show !== cur;
  const s = steps[show];
  const canNext = cur < n - 1 && (!steps[cur].gated || !!steps[cur].done);
  const railRef = useRef<HTMLOListElement | null>(null);
  const go = (i: number) => onStep(Math.max(0, Math.min(reach, i)));

  const onRailKey = (e: React.KeyboardEvent, i: number) => {
    let j = i;
    if (e.key === 'ArrowRight') j = i + 1;
    else if (e.key === 'ArrowLeft') j = i - 1;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = reach;
    else return;
    e.preventDefault();
    j = Math.max(0, Math.min(reach, j));
    go(j);
    railRef.current?.querySelectorAll('button')[j]?.focus();
  };

  return (
    <Plate {...plate}>
      <div className="in-step">
        {settings && <div className="in-step-settings in-controls">{settings}</div>}
        <ol className="in-step-rail" ref={railRef} aria-label="Steps" style={{ ['--n' as string]: n }}>
          {steps.map((st, i) => {
            const state =
              i === cur
                ? 'current'
                : i <= visited && i <= reach && (!st.gated || st.done)
                  ? 'done'
                  : 'pending';
            const can = i <= reach;
            return (
              <li key={st.id} data-state={state} data-preview={previewing && i === show ? '' : undefined}>
                <button
                  type="button"
                  disabled={!can}
                  aria-current={i === cur ? 'step' : undefined}
                  aria-label={`Step ${i + 1}: ${st.title}${state === 'done' ? ' (done)' : ''}`}
                  tabIndex={i === cur ? 0 : -1}
                  onClick={() => go(i)}
                  onKeyDown={(e) => onRailKey(e, i)}
                  {...(can ? bind(i) : {})}
                >
                  <span className="in-step-no">{state === 'done' ? '✓' : i + 1}</span>
                  <span className="in-step-title">{st.title}</span>
                </button>
              </li>
            );
          })}
        </ol>
        <div className="in-step-body">
          <div className="in-step-text" aria-live="polite">
            <div className="in-step-kicker">
              Step {show + 1} of {n}
              <span className="in-step-kicker-title"> · {s.title}</span>
              {previewing && <span className="in-step-pv"> (preview)</span>}
            </div>
            <div className="in-step-say">{s.text}</div>
          </div>
          <div className="in-step-view" aria-label={viewLabel} role="group">
            <div className="in-step-label">{viewLabel}</div>
            <div className="in-step-anim" key={show}>
              {view(show)}
            </div>
          </div>
          <div className="in-step-ref" aria-label={referenceLabel} role="group">
            <div className="in-step-label">{referenceLabel}</div>
            {reference(show)}
          </div>
          <div className="in-step-ctrl">
            {previewing ? (
              <div className="in-step-pvnote">Click step {show + 1} to go there.</div>
            ) : (
              s.controls && <div className="in-controls">{s.controls}</div>
            )}
          </div>
          <div className="in-step-nav">
            <button type="button" className="in-btn" disabled={cur === 0} onClick={() => go(cur - 1)}>
              ← Back
            </button>
            {cur < n - 1 ? (
              <button
                type="button"
                className="in-btn in-btn-next"
                disabled={!canNext}
                onClick={() => go(cur + 1)}
              >
                Next →
              </button>
            ) : null}
            {!canNext && cur < n - 1 && steps[cur].hint && (
              <span className="in-step-hint">{steps[cur].hint}</span>
            )}
            {onSkip && cur < n - 1 && (
              <button
                type="button"
                className="in-btn in-btn-skip"
                onClick={() => {
                  onSkip();
                  setVisited(n - 1);
                  onStep(n - 1);
                }}
              >
                {skipLabel}
              </button>
            )}
          </div>
          <div className="in-step-eq">{equation}</div>
        </div>
      </div>
    </Plate>
  );
}
