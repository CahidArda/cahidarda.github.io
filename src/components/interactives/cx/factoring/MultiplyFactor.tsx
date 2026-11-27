import { useEffect, useRef, useState, type ReactNode } from 'react';
import {
  BigStat,
  GT,
  Specimen,
  T,
  clamp,
  svgPoint,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import {
  RSA2048_DIGITS,
  RSA250,
  SPEED,
  UNIVERSE_YEARS,
  YEAR,
  example,
  fmtTime,
  logMul,
  logTrial,
} from './model';

/**
 * Time to multiply two primes vs time to find them again by trial division, for numbers of
 * d digits, at a billion steps per second, on a log scale. Drag along the chart (it snaps to
 * RSA-250 and RSA-2048). On view the cursor sweeps once from 4 digits to 617.
 */
const D_MAX = 620;
const Y_LO = -34; // log10 seconds
const Y_HI = 306;
const LOG_SPEED = Math.log10(SPEED);
const LOG_UNIVERSE = Math.log10(UNIVERSE_YEARS * YEAR);
const STOPS = [RSA250.digits, RSA2048_DIGITS];
const START = 4;
const SWEEP = 3.2; // seconds

type Geo = { W: number; H: number; l: number; r: number; t: number; b: number; narrow: boolean };
const WIDE: Geo = { W: 640, H: 360, l: 16, r: 16, t: 14, b: 36, narrow: false };
const NARROW: Geo = { W: 360, H: 340, l: 8, r: 8, t: 12, b: 36, narrow: true };

const ease = (k: number) => (1 - Math.cos(Math.PI * k)) / 2;
const comma = (n: number | bigint) => n.toLocaleString('en-US');

function Time({ logSec }: { logSec: number }) {
  const f = fmtTime(logSec);
  const pow = f.v.startsWith('10^');
  return (
    <>
      {pow ? (
        <>
          10<sup className="text-[0.55em]">{f.v.slice(3)}</sup>
        </>
      ) : (
        f.v
      )}
      <span className="ml-1.5 font-mono text-sm text-muted">{f.unit}</span>
    </>
  );
}

export default function MultiplyFactor() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [d, setD] = useState(START); // SSR state: a 4-digit number; the demo sweeps from here
  const demo = useRef<{ t: number } | null>(null);
  const touched = useRef(false);
  const [, setTick] = useState(0);

  useEffect(() => {
    if (reduced) {
      if (!touched.current) setD(RSA2048_DIGITS); // static poster: the far end
      return;
    }
    if (!inView || touched.current) return;
    const id = setTimeout(() => {
      if (touched.current) return;
      touched.current = true;
      demo.current = { t: 0 };
      setTick((x) => x + 1);
    }, 700);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  useFrame(
    (dt) => {
      const g = demo.current;
      if (!g) return;
      g.t += dt;
      const k = clamp(g.t / SWEEP, 0, 1);
      setD(START + (RSA2048_DIGITS - START) * ease(k));
      if (k >= 1) {
        demo.current = null;
        setTick((x) => x + 1);
      }
    },
    inView && !reduced && demo.current !== null,
  );

  const dragging = useRef(false);
  const fromPointer = (svg: SVGSVGElement, g: Geo, clientX: number, clientY: number) => {
    const p = svgPoint(svg, clientX, clientY);
    let v = ((p.x - g.l) / (g.W - g.l - g.r)) * D_MAX;
    v = clamp(Math.round(v), 2, RSA2048_DIGITS);
    for (const s of STOPS) if (Math.abs(v - s) <= 12) v = s;
    demo.current = null;
    touched.current = true;
    setD(v);
  };

  const dr = Math.round(d); // readouts use whole digits
  const lm = logMul(dr) - LOG_SPEED;
  const lt = logTrial(dr) - LOG_SPEED;
  // the cursor and its dots move continuously, on the same smooth curves that are drawn
  const curMul = (x: number) => Math.log10(Math.max(1, (x * x) / 4));

  const chart = (g: Geo, cls: string) => {
    const X0 = g.l;
    const X1 = g.W - g.r;
    const Y0 = g.H - g.b;
    const Y1 = g.t;
    const sx = (v: number) => X0 + (v / D_MAX) * (X1 - X0);
    const sy = (ls: number) => Y0 - ((ls - Y_LO) / (Y_HI - Y_LO)) * (Y0 - Y1);
    const r2 = (v: number) => Math.round(v * 100) / 100;
    const path = (f: (x: number) => number) => {
      let s = '';
      for (let x = 2; x <= RSA2048_DIGITS; x += 1)
        s += `${s ? 'L' : 'M'}${r2(sx(x))} ${r2(sy(f(x) - LOG_SPEED))}`;
      return s;
    };
    const refs: { ls: number; label: ReactNode }[] = [
      { ls: 0, label: '1 second' },
      { ls: LOG_UNIVERSE, label: 'age of the universe' },
      {
        ls: LOG_UNIVERSE + 100,
        label: (
          <>
            10
            <tspan dy={-4} fontSize={7}>
              100
            </tspan>
            <tspan dy={4}> × the age of the universe</tspan>
          </>
        ),
      },
      {
        ls: LOG_UNIVERSE + 200,
        label: (
          <>
            10
            <tspan dy={-4} fontSize={7}>
              200
            </tspan>
            <tspan dy={4}> × the age of the universe</tspan>
          </>
        ),
      },
    ];
    const rsaY = sy(Math.log10(RSA250.coreYears * YEAR));
    const cx = sx(d);
    const fs = g.narrow ? 9 : 10;
    return (
      <svg
        viewBox={`0 0 ${g.W} ${g.H}`}
        className={`${cls} touch-none select-none`}
        style={{ cursor: 'ew-resize' }}
        role="img"
        aria-label="Log-scale chart: time to multiply stays under a millisecond; time to factor by trial division climbs past the age of the universe"
        onPointerDown={(e) => {
          dragging.current = true;
          (e.currentTarget as Element).setPointerCapture(e.pointerId);
          fromPointer(e.currentTarget, g, e.clientX, e.clientY);
        }}
        onPointerMove={(e) =>
          dragging.current && fromPointer(e.currentTarget, g, e.clientX, e.clientY)
        }
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
      >
        {/* reference times */}
        {refs.slice(0, g.narrow ? 3 : 4).map((r, i) => (
          <g key={i}>
            <line
              x1={X0}
              x2={X1}
              y1={sy(r.ls)}
              y2={sy(r.ls)}
              style={{
                stroke: GT.line,
                strokeWidth: 1,
                strokeDasharray: i < 2 ? undefined : '3 4',
              }}
            />
            <T
              x={g.narrow || i < 2 ? X1 - 2 : X0 + 2}
              y={sy(r.ls) - 6}
              anchor={g.narrow || i < 2 ? 'end' : 'start'}
              size={fs - 1}
            >
              {r.label}
            </T>
          </g>
        ))}
        {/* x axis */}
        <line x1={X0} x2={X1} y1={Y0} y2={Y0} style={{ stroke: GT.lineStrong }} />
        {[0, 100, 200, 300, 400, 500, 600].map((v) => (
          <g key={v}>
            <line x1={sx(v)} x2={sx(v)} y1={Y0} y2={Y0 + 4} style={{ stroke: GT.lineStrong }} />
            <T x={sx(v)} y={Y0 + 13} size={fs - 1}>
              {v}
            </T>
          </g>
        ))}
        <T x={X0} y={Y0 + 27} anchor="start" size={fs - 1}>
          digits in the number →
        </T>
        {STOPS.map((v) => (
          <g key={v}>
            <line
              x1={sx(v)}
              x2={sx(v)}
              y1={Y0 - 4}
              y2={Y0 + 4}
              style={{ stroke: GT.ink, strokeWidth: 1.5 }}
            />
            <T
              x={sx(v)}
              y={Y0 + 27}
              anchor={v > 400 ? 'end' : 'middle'}
              size={fs - 1}
              color={GT.ink}
            >
              {v === RSA250.digits ? 'RSA-250' : 'RSA-2048'}
            </T>
          </g>
        ))}

        {/* the two curves */}
        <path d={path(logTrial)} style={{ fill: 'none', stroke: GT.b, strokeWidth: 2.2 }} />
        <path d={path(curMul)} style={{ fill: 'none', stroke: GT.opt, strokeWidth: 2.2 }} />
        <T
          x={sx(g.narrow ? 330 : 380)}
          y={sy(logTrial(g.narrow ? 330 : 380) - LOG_SPEED) - 12}
          anchor="end"
          size={fs + 1}
          weight={600}
          color={GT.b}
        >
          factor (trial division)
        </T>
        <T
          x={sx(540)}
          y={sy(curMul(540) - LOG_SPEED) + 13}
          anchor="end"
          size={fs + 1}
          weight={600}
          color={GT.opt}
        >
          multiply
        </T>

        {/* the real record: a leader line up into the empty space under the red line */}
        <line
          x1={sx(RSA250.digits)}
          y1={rsaY}
          x2={sx(RSA250.digits)}
          y2={sy(g.narrow ? 70 : 62)}
          style={{ stroke: GT.eq, strokeWidth: 1 }}
        />
        <circle cx={sx(RSA250.digits)} cy={rsaY} r={4} style={{ fill: GT.eq }} />
        <T
          x={sx(RSA250.digits) + 6}
          y={sy(g.narrow ? 70 : 62)}
          anchor="start"
          size={fs}
          color={GT.eq}
          weight={600}
        >
          RSA-250 really was factored (2020)
        </T>
        <T
          x={sx(RSA250.digits) + 6}
          y={sy(g.narrow ? 70 : 62) + 13}
          anchor="start"
          size={fs - 1}
          color={GT.eq}
        >
          {g.narrow
            ? '2,700 core-years, cleverer method'
            : 'in 2,700 core-years, with a much cleverer method'}
        </T>

        {/* cursor */}
        <line x1={cx} x2={cx} y1={Y1} y2={Y0} style={{ stroke: GT.ink, strokeWidth: 1 }} />
        <circle
          cx={cx}
          cy={sy(logTrial(d) - LOG_SPEED)}
          r={4.5}
          style={{ fill: GT.b, stroke: GT.paperRaised, strokeWidth: 1.5 }}
        />
        <circle
          cx={cx}
          cy={sy(curMul(d) - LOG_SPEED)}
          r={4.5}
          style={{ fill: GT.opt, stroke: GT.paperRaised, strokeWidth: 1.5 }}
        />
        <rect x={cx - 17} y={Y1 - 2} width={34} height={14} style={{ fill: GT.ink }} />
        <T x={cx} y={Y1 + 5} size={fs - 1} color={GT.paper}>
          {dr}
        </T>
      </svg>
    );
  };

  // a real pair of primes for small numbers, the counts for big ones
  const ex = dr <= 16 ? example(dr) : null;
  const half = Math.ceil(dr / 2);
  const tryExp = Math.floor(logTrial(dr));
  const tryMant = Math.round(10 ** (logTrial(dr) - tryExp));

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-08.1"
      title="multiply vs factor"
      kicker="at a billion steps per second"
      hint="drag along the chart · snaps to RSA-250 and RSA-2048"
      footer="log scale"
    >
      {chart(WIDE, 'hidden w-full sm:block')}
      {chart(NARROW, 'block w-full sm:hidden')}

      <div className="mt-4 grid grid-cols-2 gap-4">
        <BigStat
          label="multiply the two primes"
          color={GT.opt}
          value={<Time logSec={lm} />}
          size="sm"
        />
        <BigStat label="find them again" color={GT.b} value={<Time logSec={lt} />} size="sm" />
      </div>
      <div className="mt-3 h-[3.9rem] overflow-hidden font-mono text-[0.7rem] leading-[1.3rem] tabular-nums text-ink-soft">
        {ex ? (
          <>
            <div className="truncate">
              {comma(ex.big)} × {comma(ex.small)} ={' '}
              <span className="text-ink">{comma(ex.product)}</span>
            </div>
            <div>multiplying: {ex.mults} digit-by-digit products</div>
            <div>
              factoring: {comma(ex.tries)} tries to reach {comma(ex.small)}
            </div>
          </>
        ) : (
          <>
            <div>
              a {dr}-digit number made of two primes of about {half} digits
            </div>
            <div>
              multiplying: {comma(Math.ceil(dr / 2) * Math.floor(dr / 2))} digit-by-digit products
            </div>
            <div>
              factoring: about {tryMant} × 10<sup>{tryExp}</sup> tries
            </div>
          </>
        )}
      </div>
    </Specimen>
  );
}
