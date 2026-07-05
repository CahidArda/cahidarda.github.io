import { ROBOT, Widget } from './shared';

// The robots-reshoring chain (Krenz, Prettner & Strulik, 2021): more robots pull previously
// offshored production back home, but it returns automated, so the new jobs and pay accrue to
// professional occupations and not to the routine workers who were displaced. A vertical flow
// so it reflows on mobile. No invented numbers: the only figure (+3.5% reshoring per robot) is
// from the paper; the outcomes are directional (one path lights up, the other stays greyed).

const VB = { w: 320, h: 266 };
const MID = 160;

// A rounded box with a main line and a smaller sub line.
function Box({
  cx,
  cy,
  w,
  main,
  sub,
  tone,
}: {
  cx: number;
  cy: number;
  w: number;
  main: string;
  sub: string;
  tone: 'chain' | 'up' | 'flat';
}) {
  const h = 44;
  const styles =
    tone === 'flat'
      ? { fill: 'none', stroke: 'var(--color-muted)', dash: '3 3' as string | undefined }
      : {
          fill: ROBOT,
          fillOpacity: tone === 'up' ? 0.18 : 0.09,
          stroke: ROBOT,
          dash: undefined,
        };
  const textColor = tone === 'flat' ? 'var(--color-muted)' : 'var(--color-ink)';
  return (
    <g>
      <rect
        x={cx - w / 2}
        y={cy - h / 2}
        width={w}
        height={h}
        rx={4}
        style={{
          fill: styles.fill,
          fillOpacity: (styles as { fillOpacity?: number }).fillOpacity ?? 1,
          stroke: styles.stroke,
          strokeWidth: 1.6,
          strokeDasharray: styles.dash,
        }}
      />
      <text
        x={cx}
        y={cy - 2}
        textAnchor="middle"
        style={{
          fill: textColor,
          fontFamily: 'var(--font-mono)',
          fontSize: 10.5,
          fontWeight: 600,
        }}
      >
        {main}
      </text>
      <text
        x={cx}
        y={cy + 11}
        textAnchor="middle"
        style={{ fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)', fontSize: 7.8 }}
      >
        {sub}
      </text>
    </g>
  );
}

function Arrow({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  // arrowhead pointing along the line direction, drawn at the (x2,y2) end
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const a = 6;
  const p1 = [x2 - a * Math.cos(ang - 0.4), y2 - a * Math.sin(ang - 0.4)];
  const p2 = [x2 - a * Math.cos(ang + 0.4), y2 - a * Math.sin(ang + 0.4)];
  return (
    <g style={{ stroke: 'var(--color-line-strong)' }}>
      <line x1={x1} y1={y1} x2={x2} y2={y2} style={{ strokeWidth: 1.6 }} />
      <path
        d={`M${x2} ${y2} L${p1[0]} ${p1[1]} L${p2[0]} ${p2[1]} Z`}
        style={{ fill: 'var(--color-line-strong)', stroke: 'none' }}
      />
    </g>
  );
}

export default function ReshoringChain() {
  return (
    <Widget title="The jobs came back, but not to people" kicker="Krenz et al. (2021)">
      <svg
        viewBox={`0 0 ${VB.w} ${VB.h}`}
        className="mx-auto w-full max-w-[340px]"
        role="img"
        aria-label="A vertical flow: more robots lead to more reshoring, which returns automated, so professional jobs and pay rise while routine jobs see no gain."
      >
        <Box
          cx={MID}
          cy={30}
          w={220}
          main="+1 robot / 1,000 workers"
          sub="automation rises"
          tone="chain"
        />
        <Arrow x1={MID} y1={52} x2={MID} y2={64} />
        <Box
          cx={MID}
          cy={86}
          w={220}
          main="production reshores"
          sub="+3.5% activity, back from abroad"
          tone="chain"
        />
        <Arrow x1={MID} y1={108} x2={MID} y2={120} />
        <Box
          cx={MID}
          cy={142}
          w={220}
          main="but it returns automated"
          sub="more machines, not more staff"
          tone="chain"
        />
        {/* fork */}
        <Arrow x1={MID} y1={164} x2={80} y2={202} />
        <Arrow x1={MID} y1={164} x2={240} y2={202} />
        <Box cx={80} cy={228} w={140} main="professional jobs" sub="& pay rise" tone="up" />
        <Box cx={240} cy={228} w={140} main="routine jobs" sub="no gain" tone="flat" />
      </svg>

      <p className="mt-3 font-mono text-[0.62rem] text-muted">
        Reshoring means production moving back home from lower-wage countries. Only the +3.5% figure
        is from the paper; the split outcome is directional.
      </p>
    </Widget>
  );
}
