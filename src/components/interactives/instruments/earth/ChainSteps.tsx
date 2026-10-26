/**
 * Episode 4a, step by step: Gunter's chain from the minaret's foot M down to the quay A.
 *   1 Range the line · 2 Lay whole chains, pinning each end · 3 Count the links of the last
 *   chain · 4 On the slope: lay it on the ground or step it level · 5 Where it fails.
 * Geometry: ./chainGeom.ts (section with heights ×3, the links enlarged).
 */
import { useState } from 'react';
import { C, Eq, HatchDefs, InkText, Known, Segmented, Slider, StepPlate, Sym, Working, useMeasuredWidth, type StepDef } from '../shared';
import { SiteRef } from './SiteRef';
import { CHAIN_M, LINKS, chainCount, chainScene, linksScene, type Lay } from './chainGeom';
import { LINE_H } from './site';

const solid = (role: 'ink' | 'soft' | 'faint' | 'red' | 'sky' | 'brass', w = 1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

function ChainView({ lay, laid, poles }: { lay: Lay; laid: number; poles: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = chainScene(W, lay, laid);
  const id = `chn-h-${W}`;
  const path = (pts: { x: number; y: number }[]) => pts.map((p, i) => `${i ? 'L' : 'M'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`${laid} chains laid from M toward A`}>
        <HatchDefs id={id} gap={6} role="faint" />
        <InkText x={W - 14} y={18} anchor="end" kind="label" size={11} role="soft">{`M to A, heights ×${g.VE}`}</InkText>
        <path d={`${path(g.ground)} L ${g.A.x} ${g.H - 20} L ${g.M.x} ${g.H - 20} Z`} style={{ fill: `url(#${id})`, stroke: 'none' }} />
        <path d={path(g.ground)} style={solid('ink', 1.1)} />
        {poles &&
          [g.M, g.ground[2], g.A].map((p, i) => (
            <g key={i}>
              <line x1={p.x} y1={p.y} x2={p.x} y2={p.y - 22} style={solid('red', 1.4)} />
              <path d={`M ${p.x} ${p.y - 22} l 7 3 l -7 3`} style={{ fill: C.red }} />
            </g>
          ))}
        {g.segs.map((s) => (
          <g key={s.i}>
            <path d={path(s.pts)} style={{ ...solid('brass', 2), strokeDasharray: '3 1.5' }} />
            {s.drop && <line x1={s.drop.x} y1={s.drop.y0} x2={s.drop.x} y2={s.drop.y1} style={{ ...solid('soft', 0.8), strokeDasharray: '2 2' }} />}
          </g>
        ))}
        {g.pins.slice(1).map((p, i) => (
          <line key={i} x1={p.x} y1={p.y + 3} x2={p.x} y2={p.y - 6} style={solid('ink', 1.1)} />
        ))}
        <InkText x={g.M.x + 2} y={g.M.y - 28} kind="title" size={12}>M</InkText>
        <InkText x={g.A.x - 4} y={g.A.y - 28} anchor="end" kind="title" size={12}>A</InkText>
        <InkText x={W / 2} y={g.H - 6} anchor="middle" size={12} role="soft">
          {laid === 0 ? (poles ? 'poles set out in a straight line' : 'set ranging poles on the line') : `${laid} chains pinned`}
        </InkText>
      </svg>
    </div>
  );
}

function LinksView({ lay, read }: { lay: Lay; read: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = linksScene(W, lay, read);
  const ok = read === g.links;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`The last chain, ${read} links read`}>
        <InkText x={14} y={18} kind="label" size={11} role="soft">the last chain, 100 links</InkText>
        {Array.from({ length: LINKS }, (_, i) => (
          <line key={i} x1={g.pad + i * g.kl + 0.6} y1={g.y} x2={g.pad + (i + 1) * g.kl - 0.6} y2={g.y} style={solid(i < g.linksExact ? 'brass' : 'faint', 3)} />
        ))}
        {g.tags.map((t) => (
          <g key={t.n}>
            <path d={`M ${t.x} ${g.y + 3} l -4 9 h 8 Z`} style={{ fill: C.brass }} />
            <InkText x={t.x} y={g.y + 26} anchor="middle" kind="num" size={9} role="soft">{t.n}</InkText>
          </g>
        ))}
        <line x1={g.endX} y1={g.y - 18} x2={g.endX} y2={g.y + 12} style={solid('red', 1.6)} />
        <InkText x={g.endX} y={g.y - 22} anchor="middle" size={11} role="red">A</InkText>
        <path d={`M ${g.pointer} ${g.y - 4} l -5 -9 h 10 Z`} style={{ fill: ok ? C.red : C.sky }} />
        <InkText x={W / 2} y={g.H - 8} anchor="middle" size={12} role={ok ? 'red' : 'soft'}>
          {ok ? `${g.links} links to A` : `pointer at ${read} links`}
        </InkText>
      </svg>
    </div>
  );
}

export default function ChainSteps() {
  const [step, setStep] = useState(0);
  const [laid, setLaid] = useState(0);
  const [read, setRead] = useState(50);
  const [lay, setLay] = useState<Lay>('surface');
  const c = chainCount('surface');
  const cs = chainCount('stepped');
  const cur = chainCount(lay);
  const allLaid = laid >= c.n;
  const readOk = read === c.links;
  const kn = (i: number) => step >= i;
  const steps: StepDef[] = [
    {
      id: 'range',
      title: 'Range the line',
      text: <>Rods are too slow for a field or a road. Plant ranging poles from the minaret’s foot to the quay, each hidden behind the one before, so the line runs straight.</>,
    },
    {
      id: 'lay',
      title: 'Lay the chain',
      text: <>Gunter’s chain, 1620: 66 feet in 100 links. Stretch it along the line, pin the end, drag it on, and count the pins.</>,
      controls: <Slider label="chains laid" value={laid} min={0} max={c.n} step={1} onChange={setLaid} format={(v) => `${v}`} />,
      gated: true,
      done: allLaid,
      hint: `Lay all ${c.n} whole chains`,
    },
    {
      id: 'links',
      title: 'Count the links',
      text: <>The last chain overshoots A. Count brass tags (every 10 links) and then single links to the mark.</>,
      controls: <Slider label="links to A" value={read} min={0} max={LINKS} step={1} onChange={setRead} format={(v) => `${v}`} />,
      gated: true,
      done: readOk,
      hint: 'Count to the red mark',
    },
    {
      id: 'slope',
      title: 'Step it on the slope',
      text: (
        <>
          On the road the chain lies along the slope and reads long. Surveyors held it level and dropped the low end with a
          plumb line (“stepping”). Compare the two.
        </>
      ),
      controls: (
        <Segmented
          label="how the chain lies"
          options={[
            { value: 'surface', label: 'On the ground' },
            { value: 'stepped', label: 'Stepped level' },
          ]}
          value={lay}
          onChange={setLay}
        />
      ),
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: <>The chain stops at the quay. It cannot cross 700 m of water, or climb a minaret.</>,
    },
  ];
  return (
    <StepPlate
      n={6}
      wide
      title="Chaining a line with Gunter’s chain"
      steps={steps}
      step={step}
      onStep={setStep}
      view={(i) =>
        i === 2 ? <LinksView lay="surface" read={read} /> : <ChainView lay={i >= 3 ? lay : 'surface'} laid={i === 0 ? 0 : i === 1 ? laid : (i >= 3 ? cur : c).n} poles />
      }
      viewLabel="Along the line M to A"
      reference={(i) => (
        <SiteRef
          hi={i === 4 ? ['line', 'strait'] : ['line']}
          best={i >= 3 ? `known: M to A is ${cs.measured.toFixed(1)} m level` : 'known: the slope, a gradient, a right angle'}
          label="The site, with the line from the minaret to the quay highlighted"
        />
      )}
      referenceLabel="The site"
      onSkip={() => {
        setLaid(c.n);
        setRead(c.links);
        setLay('stepped');
      }}
      equation={
        <Working label="Length from the chain">
          <Eq>
            <Sym role="red">D</Sym> = (<Sym role="brass">n</Sym> + <Sym role="sky">links</Sym> / 100) · 66 ft
          </Eq>
          <Eq>
            on the ground: <Known when={kn(2) && readOk} role="red">{`${c.measured.toFixed(1)} m`}</Known> = (<Known when={kn(1) && allLaid} role="brass">{`${c.n}`}</Known> +{' '}
            <Known when={kn(2) && readOk} role="sky">{`${c.links}`}</Known>/100) · {CHAIN_M.toFixed(2)} m
          </Eq>
          <Eq>
            stepped: <Known when={kn(3)} role="red">{`${cs.measured.toFixed(1)} m`}</Known> = (<Known when={kn(3)} role="brass">{`${cs.n}`}</Known> +{' '}
            <Known when={kn(3)} role="sky">{`${cs.links}`}</Known>/100) · {CHAIN_M.toFixed(2)} m
          </Eq>
          <Eq>
            the slope added <Known when={kn(3)}>{`${(c.measured - cs.measured).toFixed(1)} m`}</Known>; true level distance {LINE_H} m
          </Eq>
        </Working>
      }
      caption={
        <>
          Edmund Gunter’s chain (1620): 66 feet, 100 links of 7.92 inches, brass tags every 10 links; 10 square chains make an
          acre. Readings rounded to the nearest link (20 cm). Section drawn with heights ×3; the road falls 1 in 8.
        </>
      }
    />
  );
}
