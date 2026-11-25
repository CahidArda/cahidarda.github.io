import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Caption,
  Chip,
  GT,
  Seg,
  Slider,
  Specimen,
  T,
  clamp,
  svgPoint,
  useHoverPreview,
  useInView,
  useReducedMotion,
  useTween,
} from '../shared';
import { MAJ_1789, N_1789, race } from './data';

const KMAX = 20;
const W = 360;
const H = 282;
const X0 = 44;
const X1 = 344;
const YB = 236;
const YT = 22;
const ax = (a: number) => Math.round((X0 + (a / N_1789) * (X1 - X0)) * 100) / 100;
const ky = (k: number) => Math.round((YB - (k / KMAX) * (YB - YT)) * 100) / 100;

type Step = 'all' | 'coup' | 'plan' | 'plan4' | 'real';
const STEPS: { id: Step; label: string; a: number; k: number; text: string }[] = [
  {
    id: 'all',
    label: 'all W+A',
    a: 69,
    k: 0,
    text: 'Every elector writes Washington + Adams. 69 to 69: a tie, and the House of Representatives picks the President.',
  },
  {
    id: 'coup',
    label: '3 defect',
    a: 66,
    k: 3,
    text: 'Three electors write Adams + Clinton and leave Washington off. Adams 69, Washington 66: Adams is President.',
  },
  {
    id: 'plan',
    label: "Hamilton's plan",
    a: 61,
    k: 0,
    text: 'Hamilton\'s fix: "throw away a few votes say 7 or 8". Adams drops to 61, Washington keeps 69.',
  },
  {
    id: 'plan4',
    label: 'plan + 4',
    a: 61,
    k: 4,
    text: 'With 8 thrown away it now takes 4 defectors to force a tie (65 to 65), not 1.',
  },
  {
    id: 'real',
    label: '1789',
    a: 34,
    k: 0,
    text: 'What happened: Adams got 34. No elector left Washington off, so the margin of 35 was never tested.',
  },
];

const VERDICT = {
  washington: { text: 'Washington first', color: GT.ink },
  tie: { text: 'tie: the House decides', color: GT.warn },
  adams: { text: 'Adams first', color: GT.b },
} as const;

function Bar({ label, n, color, y }: { label: string; n: number; color: string; y: number }) {
  const bx0 = 92;
  const bx1 = 344;
  const x = (v: number) => bx0 + (v / N_1789) * (bx1 - bx0);
  return (
    <g>
      <T x={bx0 - 8} y={y + 9} anchor="end" size={11} color={color} weight={600}>
        {label}
      </T>
      <rect x={bx0} y={y} width={bx1 - bx0} height={18} style={{ fill: GT.paper, stroke: GT.line }} />
      <rect x={bx0} y={y} width={Math.max(0, x(n) - bx0)} height={18} style={{ fill: color }} />
      <T x={Math.min(x(n) + 4, bx1 - 2)} y={y + 9.5} anchor={x(n) > bx1 - 30 ? 'end' : 'start'} size={11} color={x(n) > bx1 - 30 ? GT.paper : GT.ink} weight={600}>
        {Math.round(n)}
      </T>
    </g>
  );
}

export default function SecondVotes() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [a, setA] = useState(69);
  const [k, setK] = useState(0);
  const [step, setStep] = useState<Step | null>('all');
  const [hoverStep, bindStep] = useHoverPreview<Step | null>(step);
  const touched = useRef(false);
  const playing = useRef(false);
  const dragging = useRef(false);

  // hovering a preset previews it; otherwise show the committed values
  const pv = hoverStep && hoverStep !== step ? STEPS.find((s) => s.id === hoverStep)! : null;
  const A0 = pv ? pv.a : a;
  const K0 = pv ? pv.k : k;
  const ta = useTween(A0, reduced, 7);
  const tk = useTween(K0, reduced, 7);
  const { W: wv, A: av, verdict } = race(A0, K0);
  const tw = useTween(wv, reduced, 7);
  const tav = useTween(av, reduced, 7);

  const go = (s: (typeof STEPS)[number]) => {
    setA(s.a);
    setK(s.k);
    setStep(s.id);
  };

  // auto-play the five beats once, then hand over
  useEffect(() => {
    if (!inView || playing.current || touched.current) return;
    if (reduced) {
      go(STEPS[STEPS.length - 1]);
      return;
    }
    playing.current = true;
    let i = 0;
    const id = setInterval(() => {
      if (touched.current) return clearInterval(id);
      i++;
      if (i >= STEPS.length) return clearInterval(id);
      go(STEPS[i]);
    }, 2900);
    return () => clearInterval(id);
  }, [inView, reduced]);

  const take = () => (touched.current = true);
  const setAK = (na: number, nk: number) => {
    const kk = clamp(Math.round(nk), 0, KMAX);
    const aa = clamp(Math.round(na), 0, N_1789 - kk);
    setA(aa);
    setK(kk);
    setStep(STEPS.find((s) => s.a === aa && s.k === kk)?.id ?? null);
  };
  const fromEvent = (e: React.PointerEvent) => {
    const p = svgPoint(svgRef.current!, e.clientX, e.clientY);
    setAK(((p.x - X0) / (X1 - X0)) * N_1789, ((YB - p.y) / (YB - YT)) * KMAX);
  };

  const cur = STEPS.find((s) => s.id === (pv ? pv.id : step));
  const text = cur
    ? cur.text
    : `${A0} write Washington + Adams, ${K0} write Adams + Clinton. Washington ${wv}, Adams ${av}.`;
  const v = VERDICT[verdict];

  return (
    <Specimen
      code="EX-01.1"
      title="two votes each"
      kicker="69 electors · 138 votes"
      rootRef={rootRef}
      hint="drag the dot, use the sliders, or pick a scenario"
      footer="a + 2k ≥ 69 puts Adams level or ahead"
    >
      <Seg
        options={STEPS.map((s) => ({ value: s.id, label: s.label }))}
        value={(step ?? '') as Step}
        onChange={(id) => {
          take();
          go(STEPS.find((s) => s.id === id)!);
        }}
        bind={(id) => bindStep(id)}
      />

      <div className="mt-4 grid items-start gap-4 md:grid-cols-[1.15fr_1fr]">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full touch-none select-none"
          role="img"
          aria-label={`${A0} electors write Washington and Adams, ${K0} write Adams and Clinton. ${v.text}.`}
          onPointerDown={(e) => {
            take();
            dragging.current = true;
            e.currentTarget.setPointerCapture?.(e.pointerId);
            fromEvent(e);
          }}
          onPointerMove={(e) => dragging.current && fromEvent(e)}
          onPointerUp={() => (dragging.current = false)}
          onPointerCancel={() => (dragging.current = false)}
          style={{ cursor: 'crosshair' }}
        >
          <defs>
            <pattern id="ex1-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" style={{ stroke: GT.line, strokeWidth: 2 }} />
            </pattern>
          </defs>
          {/* grid */}
          {[0, 5, 10, 15, 20].map((g) => (
            <g key={`k${g}`}>
              <line x1={X0} y1={ky(g)} x2={X1} y2={ky(g)} style={{ stroke: GT.line, strokeWidth: 0.8 }} />
              <T x={X0 - 6} y={ky(g)} anchor="end" size={10}>
                {g}
              </T>
            </g>
          ))}
          {[0, 10, 20, 30, 40, 50, 60, 69].map((g) => (
            <g key={`a${g}`}>
              <line x1={ax(g)} y1={YT} x2={ax(g)} y2={YB} style={{ stroke: GT.line, strokeWidth: 0.8 }} />
              <T x={ax(g)} y={YB + 11} size={10}>
                {g}
              </T>
            </g>
          ))}

          {/* not enough electors: a + k > 69 */}
          <polygon
            points={`${ax(69)},${ky(0)} ${ax(69)},${ky(KMAX)} ${ax(49)},${ky(KMAX)}`}
            style={{ fill: 'url(#ex1-hatch)' }}
          />
          {/* the danger zone: a + 2k ≥ 69 */}
          <polygon
            points={`${ax(69)},${ky(0)} ${ax(49)},${ky(KMAX)} ${ax(29)},${ky(KMAX)}`}
            style={{ fill: GT.bSoft }}
          />
          <line
            x1={ax(69)}
            y1={ky(0)}
            x2={ax(29)}
            y2={ky(KMAX)}
            style={{ stroke: GT.b, strokeWidth: 2 }}
          />
          <T x={ax(40)} y={ky(17.2)} size={10.5} color={GT.b} weight={600}>
            Adams level or ahead
          </T>
          <T x={ax(20)} y={ky(9)} size={10.5} color={GT.muted}>
            Washington safe
          </T>

          {/* landmarks */}
          {[
            { a: 34, k: 0, t: '1789' },
            { a: 61, k: 0, t: 'plan' },
          ].map((m) => (
            <g key={m.t}>
              <circle cx={ax(m.a)} cy={ky(m.k)} r={3} style={{ fill: GT.paperRaised, stroke: GT.inkSoft, strokeWidth: 1.2 }} />
              <T x={ax(m.a)} y={ky(m.k) - 9} size={9.5} color={GT.inkSoft}>
                {m.t}
              </T>
            </g>
          ))}

          {/* the reader's point */}
          <line x1={ax(ta)} y1={ky(tk)} x2={ax(ta)} y2={YB} style={{ stroke: GT.ink, strokeWidth: 1, strokeDasharray: '3 3' }} />
          <line x1={X0} y1={ky(tk)} x2={ax(ta)} y2={ky(tk)} style={{ stroke: GT.ink, strokeWidth: 1, strokeDasharray: '3 3' }} />
          <circle
            cx={ax(ta)}
            cy={ky(tk)}
            r={7}
            style={{ fill: verdict === 'washington' ? GT.ink : verdict === 'tie' ? GT.warn : GT.b, stroke: GT.paperRaised, strokeWidth: 2 }}
          />

          {/* axes titles */}
          <T x={(X0 + X1) / 2} y={YB + 30} size={10.5} color={GT.ink}>
            a = electors writing Washington + Adams
          </T>
          <text
            transform={`translate(12 ${(YT + YB) / 2}) rotate(-90)`}
            textAnchor="middle"
            dominantBaseline="middle"
            style={{ fill: GT.ink, fontFamily: GT.mono, fontSize: 10.5 }}
          >
            k = writing Adams + Clinton
          </text>
        </svg>

        <div>
          <div className="grid grid-cols-2 gap-3">
            <BigStat label="Washington" value={Math.round(tw)} sub={`on ${wv} of 69 ballots`} />
            <BigStat label="Adams" value={Math.round(tav)} color={GT.a} sub={`needs ${MAJ_1789} for a majority`} align="right" />
          </div>
          <svg viewBox="0 0 360 74" className="mt-3 block w-full" aria-hidden="true">
            <Bar label="Washington" n={tw} color={GT.ink} y={8} />
            <Bar label="Adams" n={tav} color={GT.a} y={36} />
            <line
              x1={92 + (MAJ_1789 / N_1789) * 252}
              y1={2}
              x2={92 + (MAJ_1789 / N_1789) * 252}
              y2={60}
              style={{ stroke: GT.muted, strokeWidth: 1, strokeDasharray: '2 2' }}
            />
            <T x={92 + (MAJ_1789 / N_1789) * 252} y={68} size={9.5}>
              majority 35
            </T>
          </svg>
          <div className="mt-2 border-t border-line pt-2">
            <div className="label">result</div>
            <div className="font-display text-2xl leading-tight" style={{ color: v.color }}>
              {v.text}
            </div>
          </div>
          <div className="mt-3 grid gap-3">
            <Slider
              label="a · Washington + Adams"
              value={a}
              min={0}
              max={N_1789 - k}
              onChange={(x) => {
                take();
                setAK(x, k);
              }}
              color={GT.a}
            />
            <Slider
              label="k · Adams + Clinton"
              value={k}
              min={0}
              max={KMAX}
              onChange={(x) => {
                take();
                setAK(a, x);
              }}
              color={GT.b}
            />
          </div>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
        <Chip color={GT.bSoft}>Adams level or ahead</Chip>
        <Chip color={GT.line}>not enough electors</Chip>
      </div>
      <Caption>{text}</Caption>
    </Specimen>
  );
}
