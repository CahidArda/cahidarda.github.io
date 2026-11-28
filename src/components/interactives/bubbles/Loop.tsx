import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Seg,
  Specimen,
  T,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../gt/shared';
import { useBoost } from '../gt/braess/useBoost';
import { useCompact } from '../gt/useCompact';
import { LOOPS, type LoopId } from './data';

/**
 * BX-02 "the money goes round". A supplier funds its own customer, the customer spends the
 * money on the supplier's hardware, and the supplier books it as revenue. One coin travels
 * the loop on one continuous clock (lap = out along the top arc, back along the bottom arc).
 * End users pay the customer along a thin third edge. In 1999 the loop ends with the
 * customer bankrupt and the revenue turning into a write-off; in 2026 it is still turning.
 * Hover a tab to preview, click to commit; restart replays.
 */

type Pt = { x: number; y: number };
type Geo = {
  W: number;
  H: number;
  NW: number;
  NH: number;
  SUP: { cx: number; cy: number };
  CUS: { cx: number; cy: number };
  END: { cx: number; cy: number };
  A0: Pt;
  A1: Pt;
  B0: Pt;
  B1: Pt;
  CT: Pt;
  CB: Pt;
};
function makeGeo(
  W: number,
  H: number,
  NW: number,
  NH: number,
  sx: number,
  cy: number,
  endY: number,
): Geo {
  const SUP = { cx: sx, cy };
  const CUS = { cx: W - sx, cy };
  const END = { cx: W - sx, cy: endY };
  return {
    W,
    H,
    NW,
    NH,
    SUP,
    CUS,
    END,
    A0: { x: SUP.cx + NW / 2, y: cy - 12 },
    A1: { x: CUS.cx - NW / 2, y: cy - 12 },
    B0: { x: CUS.cx - NW / 2, y: cy + 12 },
    B1: { x: SUP.cx + NW / 2, y: cy + 12 },
    CT: { x: W / 2, y: cy - 92 },
    CB: { x: W / 2, y: cy + 92 },
  };
}
const WIDE = makeGeo(600, 300, 150, 58, 110, 108, 252);
const NARROW = makeGeo(360, 330, 132, 54, 76, 108, 284);
const LAPS = 3;
const LAP_MS = 2300;
const BUST_MS = 1600;
const r2 = (v: number) => Math.round(v * 100) / 100;
const quad = (p0: Pt, c: Pt, p1: Pt, u: number) => ({
  x: (1 - u) * (1 - u) * p0.x + 2 * (1 - u) * u * c.x + u * u * p1.x,
  y: (1 - u) * (1 - u) * p0.y + 2 * (1 - u) * u * c.y + u * u * p1.y,
});
const easeIO = (k: number) => (1 - Math.cos(Math.PI * k)) / 2;

export default function Loop() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [wrapRef, compact] = useCompact(520);
  const G = compact ? NARROW : WIDE;
  const { W, H, NW, NH, SUP, CUS, END, A0, A1, B0, B1, CT, CB } = G;
  const [svgRef, k0] = useBoost(W);
  const k = compact ? 1 : k0;
  const TOP_D = `M${A0.x} ${A0.y} Q${CT.x} ${CT.y} ${A1.x} ${A1.y}`;
  const BOT_D = `M${B0.x} ${B0.y} Q${CB.x} ${CB.y} ${B1.x} ${B1.y}`;
  const topMid = quad(A0, CT, A1, 0.5);
  const botMid = quad(B0, CB, B1, 0.5);
  const [sel, setSel] = useState<LoopId>('1999');
  const [id, bind] = useHoverPreview(sel);
  const L = LOOPS[id];
  const total = LAPS * LAP_MS + BUST_MS;
  const [ms, setMs] = useState(0);
  const msRef = useRef(0);
  const [running, setRunning] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // poster: the finished state (also the SSR and reduced-motion view)
  const finish = () => {
    msRef.current = total;
    setMs(total);
  };
  const start = (delay: number) => {
    if (timer.current) clearTimeout(timer.current);
    if (reduced) return finish();
    msRef.current = 0;
    setMs(0);
    timer.current = setTimeout(() => setRunning(true), delay);
  };
  useEffect(() => {
    if (inView) start(600);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
    // replay whenever the active scenario changes (hover preview replays it too)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, id, reduced]);

  useFrame((dt) => {
    const n = Math.min(total, msRef.current + dt * 1000);
    msRef.current = n;
    setMs(n);
    if (n >= total) setRunning(false);
  }, running && inView);

  const lapF = Math.min(LAPS, ms / LAP_MS); // 0..3, continuous
  const laps = Math.floor(lapF);
  const inLap = lapF - laps;
  const looping = ms < LAPS * LAP_MS;
  const bustK = looping ? 0 : Math.min(1, (ms - LAPS * LAP_MS) / BUST_MS);
  const bust = L.ends === 'bust' && bustK > 0.3;
  // coin position along the lap (top arc first half, bottom arc second half)
  let coin: { x: number; y: number } | null = null;
  if (looping) {
    const u = easeIO(inLap < 0.5 ? inLap * 2 : (inLap - 0.5) * 2);
    coin = inLap < 0.5 ? quad(A0, CT, A1, u) : quad(B0, CB, B1, u);
  }
  let status: string;
  if (ms < 200) status = L.intro;
  else if (looping)
    status =
      inLap < 0.5
        ? `${L.supplier} ${L.lendVerb} ${L.customer}.`
        : `${L.customer} spends it on ${L.supplier} hardware. ${L.supplier} books it as revenue.`;
  else status = L.outro;

  const halo = {
    stroke: GT.paperRaised,
    strokeWidth: 5,
    strokeLinejoin: 'round' as const,
    paintOrder: 'stroke' as const,
  };
  const node = (
    p: { cx: number; cy: number },
    title: string,
    sub: string,
    color: string,
    dead = false,
  ) => (
    <g>
      <rect
        x={p.cx - NW / 2}
        y={p.cy - NH / 2}
        width={NW}
        height={NH}
        style={{
          fill: dead ? GT.paperRaised : GT.paperRaised,
          stroke: dead ? GT.muted : color,
          strokeWidth: 2,
          strokeDasharray: dead ? '4 3' : undefined,
          transition: 'stroke 250ms',
        }}
      />
      <T
        x={p.cx}
        y={p.cy - 8}
        size={13 * Math.min(k, 1.3)}
        color={dead ? GT.muted : color}
        weight={700}
        font="sans"
      >
        {title}
      </T>
      <T x={p.cx} y={p.cy + 12} size={9 * k} color={GT.muted}>
        {sub}
      </T>
    </g>
  );

  return (
    <Specimen
      rootRef={rootRef}
      code="BX-02"
      title="the money goes round"
      kicker="vendor financing"
      hint="hover a year to preview it, click to keep it"
      footer={L.source}
    >
      <div className="mb-3">
        <Seg
          options={[
            { value: '1999' as LoopId, label: '1999 · Lucent' },
            { value: '2026' as LoopId, label: '2026 · Nvidia' },
          ]}
          value={id}
          onChange={(v) => {
            setSel(v);
            start(300);
          }}
          bind={bind}
        />
      </div>
      <div ref={wrapRef}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full select-none"
          role="img"
          aria-label={`${L.supplier} funds ${L.customer}, which buys ${L.supplier} hardware`}
        >
          <defs>
            {[
              ['sup', L.supColor],
              ['cus', L.cusColor],
              ['muted', GT.muted],
            ].map(([key, c]) => (
              <marker
                key={key}
                id={`bx-loop-${key}`}
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="7"
                markerHeight="7"
                orient="auto"
              >
                <path d="M0 0 L10 5 L0 10 z" style={{ fill: c }} />
              </marker>
            ))}
          </defs>

          {/* the loop */}
          <path
            d={TOP_D}
            markerEnd="url(#bx-loop-sup)"
            style={{ fill: 'none', stroke: L.supColor, strokeWidth: 2 }}
          />
          <path
            d={BOT_D}
            markerEnd={bust ? undefined : 'url(#bx-loop-cus)'}
            style={{
              fill: 'none',
              stroke: bust ? GT.muted : L.cusColor,
              strokeWidth: 2,
              strokeDasharray: bust ? '5 4' : undefined,
              transition: 'stroke 250ms',
            }}
          />
          <g style={halo}>
            <T x={W / 2} y={topMid.y - 10} size={10 * k} color={L.supColor} weight={600}>
              {L.outLabel}
            </T>
            <T
              x={W / 2}
              y={botMid.y + 12}
              size={10 * k}
              color={bust ? GT.muted : L.cusColor}
              weight={600}
            >
              {L.backLabel}
            </T>
          </g>

          {/* end users: the money that is not circular */}
          <line
            x1={END.cx}
            y1={END.cy - NH / 2}
            x2={CUS.cx}
            y2={CUS.cy + NH / 2 + 3}
            markerEnd="url(#bx-loop-muted)"
            style={{ stroke: GT.muted, strokeWidth: 1.25, strokeDasharray: '3 3' }}
          />
          <g style={halo}>
            <T
              x={END.cx - 10}
              y={(END.cy + CUS.cy) / 2 + 4}
              anchor="end"
              size={9.5 * k}
              color={GT.inkSoft}
            >
              {L.endLabel}
            </T>
          </g>

          {/* the coin travels behind the boxes */}
          {coin && <circle cx={r2(coin.x)} cy={r2(coin.y)} r={8} style={{ fill: GT.ink }} />}
          {coin && (
            <T x={r2(coin.x)} y={r2(coin.y) + 0.5} size={9} color={GT.paperRaised} weight={700}>
              $
            </T>
          )}

          {node(SUP, L.supplier, L.supplierSub, L.supColor)}
          {node(CUS, L.customer, L.customerSub, L.cusColor, bust)}
          {node(END, 'End users', L.endSub, GT.inkSoft)}

          {bust && (
            <g
              transform={`rotate(-5 ${CUS.cx} ${CUS.cy + 13})`}
              style={{ opacity: Math.min(1, (bustK - 0.3) / 0.3) }}
            >
              <rect
                x={CUS.cx - 56}
                y={CUS.cy + 2}
                width={112}
                height={22}
                style={{ fill: GT.paperRaised, stroke: GT.b, strokeWidth: 2 }}
              />
              <T x={CUS.cx} y={CUS.cy + 13} size={10 * Math.min(k, 1.3)} color={GT.b} weight={700}>
                {L.bustStamp}
              </T>
            </g>
          )}
        </svg>
      </div>

      <div className="mt-4 flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-wrap items-end gap-8">
          <BigStat
            label={L.factA.label}
            value={L.factA.value}
            color={L.supColor}
            sub={L.factA.sub}
          />
          <BigStat
            label={L.factB.label}
            value={L.factB.value}
            color={bust ? GT.b : L.cusColor}
            sub={L.factB.sub}
          />
        </div>
        <Btn primary onClick={() => start(300)}>
          restart
        </Btn>
      </div>
      <div className="min-h-[3.2rem]">
        <Caption>{status}</Caption>
      </div>
    </Specimen>
  );
}
