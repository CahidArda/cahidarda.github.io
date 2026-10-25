/**
 * Episode 2, step by step: a shared unit. Measure the hilltop precinct, M to the head of the
 * water channel (100 m), with rods laid end to end.
 *   1 Choose the rod · 2 Lay whole rods · 3 Read the last rod's divisions · 4 Add it up ·
 *   5 Where it fails: a rod a little short.
 * Geometry and numbers: ./rodGeom.ts. Steps 2 to 4 use a true rod; step 5 shows the same
 * measurement with a faulty rod beside it (it never rewrites the reader's measurement).
 */
import { useState } from 'react';
import { C, Eq, InkText, Known, Segmented, Slider, StepPlate, Sym, Working, useMeasuredWidth, type StepDef } from '../shared';
import { SiteRef } from './SiteRef';
import { ROD_ERR, SPAN, UNITS, measure, rodScene, type UnitId } from './rodGeom';

const fmtMm = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)} mm`;

function RodView({ unit, err, laid, read, label }: { unit: UnitId; err: number; laid: number; read: number; label: string }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = rodScene(W, unit, err, laid, read);
  const ink = { stroke: C.ink, strokeWidth: 1 };
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={label}>
        <InkText x={g.pad} y={20} kind="label" size={11} role="soft">{`the precinct, M to the channel: ${SPAN} m`}</InkText>
        <line x1={g.pad} y1={g.oy + g.oh + 4} x2={g.endX} y2={g.oy + g.oh + 4} style={{ stroke: C.soft, strokeWidth: 0.8 }} />
        {g.blocks.map((b, i) => (
          <rect key={i} x={b.x0} y={g.oy} width={Math.max(0.5, b.x1 - b.x0)} height={g.oh} style={{ fill: b.odd ? C.deep : C.paper, stroke: C.brass, strokeWidth: 0.6 }} />
        ))}
        <line x1={g.endX} y1={g.oy - 8} x2={g.endX} y2={g.oy + g.oh + 10} style={{ stroke: C.red, strokeWidth: 1.4 }} />
        <InkText x={g.pad} y={g.oy - 8} kind="title" size={12}>M</InkText>
        <InkText x={g.endX} y={g.oy - 12} anchor="end" size={11} role="red">channel</InkText>
        <InkText x={g.pad} y={g.oy + g.oh + 22} kind="num" size={11} role="brass">{`${g.shown} rods laid`}</InkText>
        {/* zoom on the end */}
        <InkText x={g.pad} y={g.zy - 22} kind="label" size={11} role="soft">the end, enlarged</InkText>
        {g.zRods.map((r) => (
          <rect
            key={r.i}
            x={r.x0}
            y={g.zy}
            width={Math.max(0.5, r.x1 - r.x0)}
            height={g.zh}
            style={{ fill: r.extra ? C.paper : r.i % 2 ? C.deep : C.paper, stroke: r.extra ? C.ink : C.brass, strokeWidth: r.extra ? 1.1 : 0.8, strokeDasharray: r.extra ? '4 2' : undefined }}
          />
        ))}
        {g.visible > g.n &&
          g.divs.map((d, i) => (
            <g key={i}>
              <line x1={d.x} y1={g.zy} x2={d.x} y2={g.zy + (d.major ? 12 : 6)} style={{ ...ink, strokeWidth: d.major ? 1 : 0.6 }} />
              {d.label && (
                <InkText x={d.x} y={g.zy + g.zh + 14} anchor="middle" kind="num" size={10} role="soft">
                  {d.label}
                </InkText>
              )}
            </g>
          ))}
        <line x1={g.endZ} y1={g.zy - 14} x2={g.endZ} y2={g.zy + g.zh + 22} style={{ stroke: C.red, strokeWidth: 1.4 }} />
        {g.visible > g.n && (
          <>
            <path d={`M ${g.pointer} ${g.zy - 2} l -5 -9 h 10 Z`} style={{ fill: g.readOk ? C.red : C.sky }} />
            <InkText x={Math.min(g.W - 10, Math.max(g.pad + 40, g.pointer))} y={g.zy - 14} anchor="middle" kind="num" size={11} role={g.readOk ? 'red' : 'sky'}>
              {`${read.toFixed(1)} ${g.u.subShort}`}
            </InkText>
          </>
        )}
        <InkText x={W / 2} y={g.H - 8} anchor="middle" size={12} role="soft">
          {g.shown < g.n
            ? `${g.n - g.shown} whole rods still to lay`
            : g.readOk
              ? 'the last rod reads the remainder'
              : `whole rods: ${g.n}; now read the last rod at the red mark`}
        </InkText>
      </svg>
    </div>
  );
}

export default function RodSteps() {
  const [step, setStep] = useState(0);
  const [unit, setUnitRaw] = useState<UnitId>('cubit');
  const [laid, setLaid] = useState(0);
  const [read, setRead] = useState(0);
  const [err, setErr] = useState(-1);
  const m = measure(unit, 0);
  const bad = measure(unit, err);
  const setUnit = (u: UnitId) => {
    setUnitRaw(u);
    setLaid(0);
    setRead(0);
  };
  const allLaid = laid >= m.n;
  const readOk = Math.abs(read - m.subsRead) < 1e-9;
  const kn = (i: number) => step >= i;
  const steps: StepDef[] = [
    {
      id: 'unit',
      title: 'Choose a rod',
      text: (
        <>
          A level says “higher”, never “how far”. For that you need a unit everyone agrees on, cut into a rod. Egypt’s royal
          cubit was 7 palms or 28 digits; Sinan’s builders used the architect’s arşın of 24 parmak.
        </>
      ),
      controls: (
        <Segmented
          label="the rod"
          options={[
            { value: 'cubit', label: 'Royal cubit, 52.5 cm' },
            { value: 'arsin', label: 'Mimar arşını, 75.8 cm' },
          ]}
          value={unit}
          onChange={setUnit}
        />
      ),
    },
    {
      id: 'lay',
      title: 'Lay whole rods',
      text: <>Lay rods end to end along the paving from the minaret’s foot toward the head of the channel, counting as you go.</>,
      controls: <Slider label="rods laid" value={laid} min={0} max={m.n} step={1} onChange={setLaid} format={(v) => `${v}`} />,
      gated: true,
      done: allLaid,
      hint: `Lay all ${m.n} whole rods`,
    },
    {
      id: 'read',
      title: 'Read the remainder',
      text: (
        <>
          The next rod would overshoot. Lay it anyway and read its divisions ({m.u.subName}) at the red mark, to half a
          division.
        </>
      ),
      controls: <Slider label={`read the last rod (${m.u.subName})`} value={read} min={0} max={m.u.sub} step={0.5} onChange={setRead} format={(v) => v.toFixed(1)} />,
      gated: true,
      done: readOk,
      hint: 'Move the pointer onto the red mark',
    },
    {
      id: 'sum',
      title: 'Add it up',
      text: <>Whole rods times the rod, plus the remainder. Anyone with a rod of the same length gets the same number.</>,
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: (
        <>
          Copies of a rod drift. Make this one a millimetre or two off and repeat: the error grows with every rod laid. Rods
          also only measure what you can lay them on, at the scale of a building.
        </>
      ),
      controls: <Slider label="rod off by" value={err} min={ROD_ERR.MIN} max={ROD_ERR.MAX} step={ROD_ERR.STEP} onChange={setErr} format={fmtMm} />,
    },
  ];
  const view = (i: number) =>
    i === 4 ? (
      <RodView unit={unit} err={err} laid={bad.n} read={bad.subsRead} label={`A rod ${fmtMm(err)} off measures ${bad.measured.toFixed(2)} m`} />
    ) : (
      <RodView unit={unit} err={0} laid={i >= 2 ? m.n : laid} read={i >= 3 ? m.subsRead : read} label={`${laid} of ${m.n} rods laid`} />
    );
  return (
    <StepPlate
      n={2}
      wide
      title="Measuring with a cubit rod"
      steps={steps}
      step={step}
      onStep={setStep}
      view={view}
      viewLabel="Rods on the paving"
      reference={(i) => (
        <SiteRef
          hi={['line']}
          best={i >= 3 ? `known: the precinct is ${m.measured.toFixed(2)} m` : 'known: the slope of the road'}
          label="The site, with the line from the minaret's foot highlighted"
        />
      )}
      referenceLabel="The site"
      onSkip={() => {
        setLaid(m.n);
        setRead(m.subsRead);
      }}
      equation={
        <Working label="Length from rods">
          <Eq>
            <Sym role="red">L</Sym> = (<Sym role="brass">n</Sym> + <Sym role="sky">r</Sym> / {m.u.sub}) · <Sym>u</Sym>
          </Eq>
          <Eq>
            <Known when={kn(3)} role="red">{`${m.measured.toFixed(3)} m`}</Known> = (<Known when={kn(1) && allLaid} role="brass">{`${m.n}`}</Known> +{' '}
            <Known when={kn(2) && readOk} role="sky">{m.subsRead.toFixed(1)}</Known> / {m.u.sub}) · <Known when>{`${(m.u.len * 100).toFixed(m.u.id === 'cubit' ? 1 : 3)} cm`}</Known>
          </Eq>
          {step >= 4 && (
            <Eq>
              rod {fmtMm(err)}: <Known when role="red">{`${bad.measured.toFixed(3)} m`}</Known>, off by{' '}
              <Known when>{`${(bad.error * 100).toFixed(1)} cm`}</Known>
            </Eq>
          )}
        </Working>
      }
      caption={
        <>
          Rods drawn at their real length. The cubit is the Egyptian royal cubit (surviving rods about 52.5 cm); the arşın is
          the Ottoman architect’s cubit, 75.774 cm on average (Özdural). A short rod reads long: it takes more of them to cover
          the same ground.
        </>
      }
    />
  );
}
