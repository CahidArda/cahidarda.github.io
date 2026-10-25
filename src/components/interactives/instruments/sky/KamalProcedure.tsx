/**
 * Plate 2, procedure: latitude from Polaris with a kamal.
 *   1 Find Polaris · 2 Frame it with the board · 3 Read the cord · 4 Correct for Polaris's circle ·
 *   5 Where it fails (the cord gets short, and Polaris sets south of the equator).
 * Reference: the post's Earth section in star mode (Polaris's light nearly along the axis, its
 * height h at you equal to φ to within the star's circle). Geometry: ./kamalGeom.ts.
 */
import { useState } from 'react';
import {
  C,
  Eq,
  InkText,
  Known,
  Segmented,
  Slider,
  StepPlate,
  Sym,
  V,
  Working,
  arcPath,
  useMeasuredWidth,
  type Role,
  type StepDef,
} from '../shared';
import LatitudeReference, { type RefKey } from './LatitudeReference';
import {
  BOARD_H,
  CIRCLE_MAG,
  D_MAX,
  D_MIN,
  D_STEP,
  P_OFF,
  PORTS,
  circleScene,
  cordFor,
  curveScene,
  latOf,
  polarisAlt,
  sideScene,
  skySay,
  skyScene,
  thetaOf,
  where,
  type Port,
} from './kamalGeom';

const solid = (role: Role, w = 1.1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });
const d2 = (v: number) => `${v.toFixed(2)}°`;
const sgn2 = (v: number) => `${v < 0 ? '−' : '+'} ${Math.abs(v).toFixed(2)}°`;
const timeOfNight = (lha: number) => `${where(lha)}`;

function SkyView({ port, lha, d, board }: { port: Port; lha: number; d: number; board: boolean }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = skyScene(port, lha, d, W);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`The northern sky from ${port.name}: Polaris ${g.alt.toFixed(1)} degrees up. ${board ? skySay(g) : ''}`}>
        <rect x={0} y={0} width={W} height={g.horizonY} style={{ fill: C.deep }} />
        {g.ticks.map((t) => (
          <g key={t.a}>
            <line x1={14} y1={t.y} x2={W - 14} y2={t.y} style={{ ...solid('faint', 0.6), strokeDasharray: '2 5' }} />
            <InkText x={16} y={t.y - 4} kind="num" size={10} role="soft">{`${t.a}°`}</InkText>
          </g>
        ))}
        {/* the Plough's pointers, schematic: just a hint of where Polaris is */}
        {g.alt > 0 && (
          <>
            <circle cx={g.star.x} cy={g.star.y} r={3.6} style={{ fill: C.sky }} />
            <InkText x={g.star.x + 10} y={g.star.y - 6} size={12} role="sky">Polaris</InkText>
          </>
        )}
        {board && (
          <>
            <rect x={g.board.x0} y={g.board.y0} width={g.board.x1 - g.board.x0} height={g.board.y1 - g.board.y0} style={{ fill: C.brass, opacity: 0.32, stroke: C.brass, strokeWidth: 1.4 }} />
            {g.clipped ? (
              <InkText x={g.board.x1 + 6} y={18} size={11} role="red">↑ top edge far above</InkText>
            ) : (
              <line x1={g.board.x0} y1={g.board.y0} x2={g.board.x1} y2={g.board.y0} style={solid(g.touching ? 'sky' : 'red', 2)} />
            )}
          </>
        )}
        <rect x={0} y={g.horizonY} width={W} height={g.H - g.horizonY} style={{ fill: C.sky, opacity: 0.14 }} />
        <line x1={0} y1={g.horizonY} x2={W} y2={g.horizonY} style={solid('ink', 1.3)} />
        <InkText x={g.cx} y={g.horizonY + 16} kind="num" anchor="middle" size={11} role="ink">N</InkText>
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={board && g.touching ? 'sky' : 'soft'}>
          {board ? skySay(g) : g.alt > 0 ? `Polaris ${g.alt.toFixed(1)}° up, ${where(lha)}` : 'Polaris is below the horizon here'}
        </InkText>
      </svg>
    </div>
  );
}

function SideView({ d }: { d: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = sideScene(d, W);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Side view: a ${BOARD_H} cm board on a ${d.toFixed(1)} cm cord frames ${g.theta.toFixed(2)} degrees`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">side view, to scale</InkText>
        <line x1={g.eye.x} y1={g.eye.y} x2={g.far.x} y2={g.far.y} style={{ ...solid('sky', 0.9), strokeDasharray: '4 3' }} />
        <line x1={g.eye.x} y1={g.eye.y} x2={g.W - g.pad} y2={g.eye.y} style={{ ...solid('ink', 0.9), strokeDasharray: '4 3' }} />
        <InkText x={g.W - g.pad} y={g.eye.y + 16} anchor="end" size={11} role="soft">to the horizon</InkText>
        {/* cord with knots every 10 cm */}
        <line x1={g.eye.x} y1={g.eye.y} x2={g.boardX} y2={g.eye.y} style={solid('ink', 1.6)} />
        {g.marks.filter((m) => m.x <= g.boardX + 0.5).map((m) => (
          <circle key={m.cm} cx={m.x} cy={g.eye.y} r={2.4} style={{ fill: C.ink }} />
        ))}
        {g.marks.map((m) => (
          <InkText key={m.cm} x={m.x} y={g.eye.y + 30} kind="num" anchor="middle" size={10} role="soft">{`${m.cm}`}</InkText>
        ))}
        <InkText x={g.eye.x} y={g.eye.y + 44} size={10} role="soft">cord, cm</InkText>
        {/* the board */}
        <line x1={g.boardX} y1={g.eye.y} x2={g.boardTop.x} y2={g.boardTop.y} style={solid('brass', 4)} />
        <InkText x={g.boardX + 7} y={(g.eye.y + g.boardTop.y) / 2} baseline="central" size={13} role="ink">{`h = ${BOARD_H} cm`}</InkText>
        <line x1={g.eye.x} y1={g.eye.y + 12} x2={g.boardX} y2={g.eye.y + 12} style={solid('sky', 1.1)} />
        <InkText x={(g.eye.x + g.boardX) / 2} y={g.eye.y + 12} anchor="middle" baseline="central" size={13} role="sky">{`d = ${d.toFixed(1)} cm`}</InkText>
        <path d={arcPath(g.eye.x, g.eye.y, g.arcR, 0, g.theta)} style={solid('red', 1.5)} />
        <InkText x={g.eye.x + g.arcR + 6} y={g.eye.y - g.arcR * Math.sin((g.theta * Math.PI) / 360) - 2} size={14} role="red">θ</InkText>
        <circle cx={g.eye.x} cy={g.eye.y} r={4} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.3 }} />
        <InkText x={g.eye.x - 4} y={g.eye.y - 12} size={11} role="ink">eye (cord in the teeth)</InkText>
      </svg>
    </div>
  );
}

function CircleView({ port, lha }: { port: Port; lha: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = circleScene(port, lha, W);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Polaris on its circle round the pole, ${where(lha)}: add ${g.corr.toFixed(2)} degrees`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">{`round the pole, ${CIRCLE_MAG}× magnified`}</InkText>
        <circle cx={g.cx} cy={g.cy} r={g.r} style={{ ...solid('sky', 0.9), strokeDasharray: '4 4' }} />
        <line x1={g.cx - g.r - 18} y1={g.cy} x2={g.cx + g.r + 18} y2={g.cy} style={{ ...solid('red', 1.1) }} />
        <InkText x={g.cx + g.r + 14} y={g.cy - 8} anchor="end" size={11} role="red">pole’s height = φ</InkText>
        <circle cx={g.cx} cy={g.cy} r={3} style={{ fill: C.ink }} />
        <InkText x={g.cx - 6} y={g.cy + 16} anchor="end" size={11} role="ink">pole</InkText>
        <line x1={g.star.x} y1={g.star.y} x2={g.star.x} y2={g.cy} style={solid('brass', 1.6)} />
        <circle cx={g.star.x} cy={g.star.y} r={4} style={{ fill: C.sky }} />
        <InkText x={g.star.x + (g.star.x >= g.cx ? 8 : -8)} y={g.star.y + (g.star.y <= g.cy ? -8 : 16)} anchor={g.star.x >= g.cx ? 'start' : 'end'} size={12} role="sky">Polaris</InkText>
        <InkText x={W / 2} y={g.H - 30} anchor="middle" size={12} role="brass">{`Polaris is ${Math.abs(g.corr).toFixed(2)}° ${g.corr <= 0 ? 'above' : 'below'} the pole`}</InkText>
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={11} role="soft">{`circle radius ${P_OFF}°, once round in 23 h 56 min`}</InkText>
      </svg>
    </div>
  );
}

function CurveView({ port }: { port: Port }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = curveScene(port, W);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label="Cord length needed against latitude: long near the equator, short and crowded further north">
        <InkText x={14} y={18} kind="label" size={12} role="soft">cord length for each latitude</InkText>
        <line x1={g.x0} y1={g.y1} x2={g.x1} y2={g.y1} style={solid('ink', 1)} />
        <line x1={g.x0} y1={g.y0} x2={g.x0} y2={g.y1} style={solid('ink', 1)} />
        {g.xTicks.map((t) => (
          <InkText key={t.l} x={t.x} y={g.y1 + 16} kind="num" anchor="middle" size={10} role="soft">{`${t.l}°`}</InkText>
        ))}
        <InkText x={(g.x0 + g.x1) / 2} y={g.y1 + 32} anchor="middle" size={11} role="soft">latitude (north)</InkText>
        {g.yTicks.map((t) => (
          <InkText key={t.d} x={g.x0 - 6} y={t.y} kind="num" anchor="end" baseline="central" size={10} role="soft">{`${t.d}`}</InkText>
        ))}
        <InkText x={g.x0 + 6} y={g.y0 - 6} size={11} role="soft">cm</InkText>
        <polyline points={g.pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')} style={solid('sky', 1.6)} />
        {g.here && (
          <>
            <circle cx={g.here.x} cy={g.here.y} r={4} style={{ fill: C.red }} />
            <InkText x={g.here.x + (g.here.x > W * 0.55 ? -8 : 8)} y={g.here.y - 10} anchor={g.here.x > W * 0.55 ? 'end' : 'start'} size={12} role="red">{`${port.name}: ${cordFor(port.lat).toFixed(1)} cm`}</InkText>
          </>
        )}
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role="soft">
          {g.here ? `1° here = ${(g.here.perDeg * 10).toFixed(1)} mm of cord` : 'off the board'}
        </InkText>
      </svg>
    </div>
  );
}

const HI: RefKey[][] = [['sun'], ['h', 'horizon'], ['h'], ['phi', 'h'], ['phi']];

export default function KamalProcedure() {
  const [portId, setPortId] = useState('cal');
  const [lha, setLha] = useState(60);
  const [d, setD] = useState(30);
  const [step, setStep] = useState(0);
  const port = PORTS.find((p) => p.id === portId)!;
  const alt = polarisAlt(port.lat, lha);
  const theta = thetaOf(d);
  const touching = alt > 0 && Math.abs(theta - alt) <= 0.15;
  const phi = latOf(theta, lha);
  const corr = -P_OFF * Math.cos((lha * Math.PI) / 180);
  const kn = (i: number) => step >= i && touching;

  const steps: StepDef[] = [
    {
      id: 'find',
      title: 'Find Polaris',
      text: (
        <>
          Face north at night. Polaris barely moves: it sits {P_OFF}° from the celestial pole, the point the sky turns
          round, and the pole stands exactly your latitude above the horizon. Through the night Polaris circles it.
        </>
      ),
      controls: <Slider label="where Polaris is on its circle" value={lha} min={0} max={359} step={1} onChange={setLha} format={timeOfNight} width="11rem" />,
    },
    {
      id: 'frame',
      title: 'Frame it',
      text: (
        <>
          Hold the cord in your teeth and the board at its end, square to your eye. Put the lower edge on the horizon,
          then slide along the cord until the top edge just touches Polaris.
        </>
      ),
      controls: <Slider label="length of cord" value={d} min={D_MIN} max={D_MAX} step={D_STEP} onChange={setD} format={(v) => `${v.toFixed(2)} cm`} />,
      gated: true,
      done: touching,
      hint: alt > 0 ? 'Bring the top edge onto Polaris' : 'Polaris is below the horizon',
    },
    {
      id: 'read',
      title: 'Read the cord',
      text: (
        <>
          The navigator tied a knot for each port he knew and read the angle as knots, about a finger’s width apart. In
          degrees it is a right triangle: board height over cord length.
        </>
      ),
    },
    {
      id: 'correct',
      title: 'Correct for the circle',
      text: (
        <>
          Polaris is not quite at the pole. When it is above the pole you measured too much, below it too little.
          Subtract its height above the pole.
        </>
      ),
    },
    {
      id: 'fails',
      title: 'Where it fails',
      text: (
        <>
          Sail north and the angle grows, so the cord gets short and a degree shrinks to a few millimetres of it. Sail
          south of the equator and Polaris sets for good. And by day there is no Polaris at all.
        </>
      ),
    },
  ];

  const view = (i: number) => {
    if (i <= 1) return <SkyView port={port} lha={lha} d={d} board={i === 1} />;
    if (i === 2) return <SideView d={d} />;
    if (i === 3) return <CircleView port={port} lha={lha} />;
    return <CurveView port={port} />;
  };

  return (
    <StepPlate
      n={2}
      wide
      title="Latitude from Polaris with a kamal"
      steps={steps}
      step={step}
      onStep={setStep}
      view={view}
      viewLabel="What you see and do"
      reference={(i) => (
        <LatitudeReference
          spec={{ phi: port.lat, src: { kind: 'star', alt }, visible: alt > 0, sight: i >= 1 && i <= 2 ? theta : null }}
          hi={HI[i]}
          label={`The Earth cut along your meridian at ${port.name}: Polaris's light arrives almost parallel to the axis, so its height above your horizon is your latitude, ${port.lat} degrees.`}
        />
      )}
      referenceLabel="What you are measuring (your meridian)"
      settings={
        <Segmented
          label="home port"
          options={PORTS.map((p) => ({ value: p.id, label: p.name }))}
          value={portId}
          onChange={(id) => {
            setPortId(id);
            setStep((k) => Math.min(k, 1));
          }}
        />
      }
      onSkip={() => setD(Math.round(cordFor(alt) / D_STEP) * D_STEP)}
      equation={
        <Working label="Latitude from Polaris">
          <Eq>
            tan <Sym role="sky">h</Sym> = <Sym>board</Sym> / <Sym role="sky">d</Sym>
          </Eq>
          <Eq>
            tan <Known when={kn(2)} role="sky">{d2(theta)}</Known> = <V>{`${BOARD_H} cm`}</V> / <Known when={kn(1)} role="sky">{`${d.toFixed(2)} cm`}</Known>
          </Eq>
          <Eq>
            <Sym role="red">φ</Sym> = <Sym role="sky">h</Sym> − <Sym role="brass">p</Sym> cos <Sym role="brass">t</Sym>
          </Eq>
          <Eq>
            <Known when={kn(3)} role="red">{d2(phi)}</Known> = <Known when={kn(2)} role="sky">{d2(theta)}</Known> <Known when={kn(3)} role="brass">{sgn2(corr)}</Known>
          </Eq>
          {kn(3) && (
            <Eq>
              <span style={{ fontSize: '0.88em' }}>{`${port.name} is at ${port.lat.toFixed(2)}° N: off by ${(Math.abs(phi - port.lat) * 60).toFixed(0)}′, ${(Math.abs(phi - port.lat) * 60).toFixed(0)} nautical miles.`}</span>
            </Eq>
          )}
        </Working>
      }
      caption={
        <>
          p is Polaris’s distance from the pole ({P_OFF}° in 2026) and t where it is on its circle. The top edge counts as
          touching within 0.15°, about what a hand-held board allows. Refraction and the height of your eye are left out:
          a few arcminutes, smaller than one knot.
        </>
      }
    />
  );
}
