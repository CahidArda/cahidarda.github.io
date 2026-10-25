/**
 * Episode 4b, step by step: measuring a winding road on a chart with a map measurer.
 *   1 Zero the needle · 2 Roll the wheel along the road · 3 Read the dial · 4 Apply the
 *   chart's scale · 5 Where it fails: only as good as the map.
 * Geometry: ./mapGeom.ts. The chart is the site's plan at 1:5,000.
 */
import { useState } from 'react';
import { C, Eq, InkText, Known, Slider, StepPlate, Sym, Working, polar, useMeasuredWidth, type StepDef } from '../shared';
import { SiteRef } from './SiteRef';
import { BOX, DIAL_IN, M_PER_IN, READ_STEP, ROAD_IN, SCALE, STRAIGHT_IN, chartScene, dialScene, readOk } from './mapGeom';
import { LINE_H, ROAD_LEN } from './site';

const solid = (role: 'ink' | 'soft' | 'faint' | 'red' | 'sky' | 'brass', w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });
const path = (pts: { x: number; y: number }[]) => pts.map((p, i) => `${i ? 'L' : 'M'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');

function ChartView({ f, showWheel }: { f: number; showWheel: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = chartScene(W, f);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label="Chart of the site at 1:5,000 with the road from the minaret to the quay">
        <InkText x={14} y={16} kind="label" size={11} role="soft">{`chart, 1:${SCALE.toLocaleString('en')}`}</InkText>
        <rect x={g.shoreX} y={20} width={W - 14 - g.shoreX} height={g.H - 52} style={{ fill: C.sky, opacity: 0.12 }} />
        <line x1={g.shoreX} y1={20} x2={g.shoreX} y2={g.H - 32} style={solid('ink', 1.1)} />
        <path d={path(g.road)} style={{ ...solid('soft', 3.5), strokeLinecap: 'round' }} />
        <path d={path(g.road)} style={{ fill: 'none', stroke: C.paper, strokeWidth: 2 }} />
        {showWheel && f > 0 && <path d={path(g.done)} style={solid('red', 1.6)} />}
        <rect x={g.M.x - 4} y={g.M.y - 4} width={8} height={8} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1 }} />
        <InkText x={g.M.x} y={g.M.y - 9} anchor="middle" kind="title" size={12}>M</InkText>
        <circle cx={g.A.x} cy={g.A.y} r={3} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.1 }} />
        <InkText x={g.A.x - 6} y={g.A.y - 8} anchor="end" kind="title" size={12}>A</InkText>
        {/* scale bar */}
        <line x1={g.bar.x0} y1={g.bar.y} x2={g.bar.x1} y2={g.bar.y} style={solid('ink', 1.2)} />
        {g.bar.ticks.map((x, i) => (
          <line key={i} x1={x} y1={g.bar.y - (i % 2 ? 3 : 5)} x2={x} y2={g.bar.y + (i % 2 ? 3 : 5)} style={solid('ink', 0.9)} />
        ))}
        <InkText x={g.bar.x1 + 6} y={g.bar.y} baseline="central" kind="num" size={10} role="soft">{`2 in = ${Math.round(2 * M_PER_IN)} m`}</InkText>
        {showWheel && (
          <g transform={`translate(${g.wheel.x} ${g.wheel.y}) rotate(${(-g.dir * 180) / Math.PI})`}>
            <line x1={0} y1={0} x2={-8} y2={-26} style={solid('ink', 2.4)} />
            <circle cx={0} cy={0} r={4.5} style={{ fill: C.paper, stroke: C.red, strokeWidth: 1.6 }} />
          </g>
        )}
      </svg>
    </div>
  );
}

function DialView({ rolled, read }: { rolled: number; read: number | null }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = dialScene(W, rolled, read);
  const tip = polar(g.cx, g.cy, g.R - 12, g.needle);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`The dial reads ${rolled.toFixed(2)} inches`}>
        <InkText x={14} y={16} kind="label" size={11} role="soft">the dial: inches outside, cm inside</InkText>
        <circle cx={g.cx} cy={g.cy} r={g.R + 6} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.4 }} />
        {g.outer.map((t, i) => {
          const a = polar(g.cx, g.cy, g.R, t.a);
          const b = polar(g.cx, g.cy, g.R - (t.major ? 8 : 4), t.a);
          const l = t.label ? polar(g.cx, g.cy, g.R - 17, t.a) : null;
          return (
            <g key={i}>
              {t.major ? <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} style={solid('ink', 0.9)} /> : <circle cx={b.x} cy={b.y} r={0.9} style={{ fill: C.ink }} />}
              {l && <InkText x={l.x} y={l.y} anchor="middle" baseline="central" kind="num" size={10} halo={false}>{t.label}</InkText>}
            </g>
          );
        })}
        <circle cx={g.cx} cy={g.cy} r={g.R - 26} style={solid('faint', 0.7)} />
        {g.inner.map((t, i) => {
          const a = polar(g.cx, g.cy, g.R - 26, t.a);
          const b = polar(g.cx, g.cy, g.R - 30, t.a);
          const l = t.label ? polar(g.cx, g.cy, g.R - 38, t.a) : null;
          return (
            <g key={i}>
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} style={solid('soft', 0.5)} />
              {l && <InkText x={l.x} y={l.y} anchor="middle" baseline="central" kind="num" size={8} role="soft" halo={false}>{t.label}</InkText>}
            </g>
          );
        })}
        <line x1={g.cx} y1={g.cy} x2={tip.x} y2={tip.y} style={solid('red', 1.6)} />
        <circle cx={g.cx} cy={g.cy} r={3.5} style={{ fill: C.ink }} />
        {g.readAng !== null && (() => {
          const p = polar(g.cx, g.cy, g.R + 12, g.readAng);
          const q = polar(g.cx, g.cy, g.R + 4, g.readAng);
          return <line x1={p.x} y1={p.y} x2={q.x} y2={q.y} style={solid('sky', 2.4)} />;
        })()}
        <InkText x={W / 2} y={g.H - 6} anchor="middle" kind="num" size={12} role="red">{`needle at ${rolled.toFixed(2)} in`}</InkText>
      </svg>
    </div>
  );
}

export default function MapSteps() {
  const [step, setStep] = useState(0);
  const [zeroed, setZeroed] = useState(7.5);
  const [f, setF] = useState(0);
  const [read, setRead] = useState(3);
  const zero = zeroed === 0;
  const rolled = f * ROAD_IN;
  const atEnd = f >= 1;
  const ok = readOk(read);
  const dist = read * M_PER_IN;
  const kn = (i: number) => step >= i;
  const steps: StepDef[] = [
    {
      id: 'zero',
      title: 'Zero the needle',
      text: <>Chains measure ground. Maps let you measure the ground without walking it. Turn the little wheel backwards until the needle reads zero.</>,
      controls: <Slider label="needle before you start" value={zeroed} min={0} max={DIAL_IN / 2} step={0.5} onChange={setZeroed} format={(v) => `${v.toFixed(1)} in`} />,
      gated: true,
      done: zero,
      hint: 'Bring the needle to 0',
    },
    {
      id: 'roll',
      title: 'Roll the road',
      text: <>Put the wheel on the minaret and roll it along every bend of the road to the quay. The needle counts the turns of the wheel.</>,
      controls: <Slider label="roll along the road" value={f} min={0} max={1} step={0.005} onChange={setF} format={(v) => `${Math.round(v * 100)}%`} />,
      gated: true,
      done: atEnd,
      hint: 'Roll all the way to A',
    },
    {
      id: 'read',
      title: 'Read the dial',
      text: <>Read the outer ring in inches: long marks every inch, dots at the halves, so you can judge to a twentieth.</>,
      controls: <Slider label="your reading" value={read} min={0} max={8} step={READ_STEP} onChange={setRead} format={(v) => `${v.toFixed(2)} in`} />,
      gated: true,
      done: ok,
      hint: 'Match the needle',
    },
    {
      id: 'scale',
      title: 'Apply the scale',
      text: (
        <>
          At 1:{SCALE.toLocaleString('en')} an inch on the chart is {M_PER_IN.toFixed(0)} m. The box of the one in the
          exhibition does the same for 1:250,000: about 4 miles to the inch, 2½ km to the centimetre.
        </>
      ),
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: <>The wheel measures a drawn line. The map is only as good as whoever surveyed it, and somebody had to measure the strait first.</>,
    },
  ];
  return (
    <StepPlate
      n={7}
      wide
      title="Measuring a road on a chart"
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) => (i === 0 || i === 2 ? <DialView rolled={i === 0 ? zeroed : rolled} read={i === 2 ? read : null} /> : <ChartView f={i === 1 ? f : 1} showWheel={i >= 1} />)}
      viewLabel="The chart and the measurer"
      reference={(i) => (
        <SiteRef
          hi={i === 4 ? ['road', 'strait'] : ['road']}
          best={i >= 3 ? `known: the road is ${dist.toFixed(0)} m` : `known: M to A, ${LINE_H} m level`}
          label="The site, with the winding road highlighted on the plan"
        />
      )}
      referenceLabel="The site"
      onSkip={() => {
        setZeroed(0);
        setF(1);
        setRead(Math.round(ROAD_IN / READ_STEP) * READ_STEP);
      }}
      equation={
        <Working label="Distance from the map measurer">
          <Eq>
            <Sym role="red">D</Sym> = <Sym role="sky">reading</Sym> · <Sym role="brass">metres per inch</Sym>
          </Eq>
          <Eq>
            <Known when={kn(3) && ok} role="red">{`${dist.toFixed(0)} m`}</Known> = <Known when={kn(2) && ok} role="sky">{`${read.toFixed(2)} in`}</Known> ·{' '}
            <Known when={kn(3)} role="brass">{`${M_PER_IN.toFixed(0)} m`}</Known>
          </Eq>
          <Eq>
            road on the ground <Known when={kn(3)}>{`${ROAD_LEN.toFixed(0)} m`}</Known>; straight line {LINE_H} m = {STRAIGHT_IN.toFixed(2)} in
          </Eq>
          <Eq>
            box: 1:250,000 → <V2>{`${BOX.miPerIn.toFixed(2)} mi`}</V2> per inch, <V2>{`${BOX.kmPerCm} km`}</V2> per cm
          </Eq>
        </Working>
      }
      caption={
        <>
          After the map measurer in its yellow box, lying on a British Admiralty chart in the exhibition. Its box explains the
          outer ring (30 inches a turn, dots at half inches) and the inner one (99 cm a turn) and works the 1:250,000 example
          printed here. The chart of the site is mine, at 1:5,000.
        </>
      }
    />
  );
}

function V2({ children }: { children: string }) {
  return <span className="in-v">{children}</span>;
}
