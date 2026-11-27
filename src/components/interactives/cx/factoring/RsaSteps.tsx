import { useEffect, useRef, useState, type ReactNode } from 'react';
import {
  Caption,
  Chip,
  GT,
  Seg,
  Specimen,
  clamp,
  lerp,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { Key, Lock } from './Padlock';

/**
 * CX-08.2: one block of the RSA paper's example (§VIII), step by step.
 * The receiver picks p = 47 and q = 59, gets n = 2773 and (p - 1)(q - 1) = 2668, and a pair
 * with 17 × 157 = 2669 = 2668 + 1. It publishes (2773, 17) and keeps 47, 59, 2668 and 157.
 * The sender encodes "IT" as 0920 (blank = 00, A = 01, ..., Z = 26), locks it to
 * 920^17 mod 2773 = 948, and the receiver unlocks 948^157 mod 2773 = 920.
 * All numbers checked with Python's pow(). Three zones: sender, public, receiver's secret.
 */

// colours by role: public numbers in the accent, secrets in green (as in CX-08.3)
const PUB = GT.eq;
const SEC = GT.opt;

type Piece = { t: string; c?: string; w?: number; sup?: boolean };
const s = (t: string, c?: string, w?: number): Piece => ({ t, c, w });
const sup = (t: string, c?: string): Piece => ({ t, c, sup: true });

/** One mono line built from coloured pieces; `sup` pieces are raised and smaller. */
function Line({
  x,
  y,
  size,
  segs,
  anchor = 'start',
  color = GT.ink,
  weight = 400,
}: {
  x: number;
  y: number;
  size: number;
  segs: Piece[];
  anchor?: 'start' | 'middle' | 'end';
  color?: string;
  weight?: number;
}) {
  const up = size * 0.42;
  let raised = false;
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      dominantBaseline="middle"
      style={{
        fill: color,
        fontFamily: GT.mono,
        fontSize: size,
        fontWeight: weight,
        fontVariantNumeric: 'lining-nums tabular-nums',
        whiteSpace: 'pre',
      }}
    >
      {segs.map((g, i) => {
        const dy = g.sup && !raised ? -up : !g.sup && raised ? up : undefined;
        raised = !!g.sup;
        return (
          <tspan
            key={i}
            dy={dy}
            fontSize={g.sup ? size * 0.72 : undefined}
            style={{
              fill: g.c,
              fontWeight: g.w,
            }}
          >
            {g.t}
          </tspan>
        );
      })}
    </text>
  );
}

/* ───────────────────────────── layout ───────────────────────────── */

type Box = { x: number; y: number; w: number; h: number };
type P = { x: number; y: number };
type Geo = {
  W: number;
  H: number;
  narrow: boolean;
  fs: number; // row font size
  sender: Box;
  pub: Box;
  recv: Box;
  // rows (left edge, middle line)
  primes: P;
  n: P;
  phi: P;
  pair: P;
  pair2: P;
  unlock: P;
  unlock2: P;
  text: P;
  digits: P;
  m: P;
  lock: P;
  lock2: P;
  // public key: the two numbers land here (left edge of the number); labels sit before them
  keyN: P;
  keyE: P;
  labN: P;
  labE: P;
  keyLabel: P;
  empty: P;
  wireA: P;
  wireB: P;
  wireLabel: P & { anchor: 'start' | 'middle' | 'end' };
  eve: P;
  lockIcon: P;
  keyIcon: P;
};

const WIDE: Geo = {
  W: 640,
  H: 292,
  narrow: false,
  fs: 11,
  sender: { x: 4, y: 4, w: 172, h: 284 },
  pub: { x: 188, y: 4, w: 236, h: 284 },
  recv: { x: 436, y: 4, w: 200, h: 284 },
  primes: { x: 448, y: 50 },
  n: { x: 448, y: 78 },
  phi: { x: 448, y: 106 },
  pair: { x: 448, y: 136 },
  pair2: { x: 448, y: 154 },
  unlock: { x: 448, y: 236 },
  unlock2: { x: 448, y: 256 },
  text: { x: 16, y: 52 },
  digits: { x: 16, y: 78 },
  m: { x: 16, y: 106 },
  lock: { x: 16, y: 236 },
  lock2: { x: 16, y: 256 },
  keyN: { x: 296, y: 66 },
  keyE: { x: 296, y: 90 },
  labN: { x: 263, y: 66 },
  labE: { x: 263, y: 90 },
  keyLabel: { x: 306, y: 44 },
  empty: { x: 306, y: 70 },
  wireA: { x: 176, y: 236 },
  wireB: { x: 436, y: 236 },
  wireLabel: { x: 306, y: 258, anchor: 'middle' },
  eve: { x: 200, y: 140 },
  lockIcon: { x: 30, y: 204 },
  keyIcon: { x: 466, y: 208 },
};

const NARROW: Geo = {
  W: 360,
  H: 390,
  narrow: true,
  fs: 10.5,
  sender: { x: 4, y: 4, w: 352, h: 100 },
  pub: { x: 4, y: 112, w: 352, h: 92 },
  recv: { x: 4, y: 212, w: 352, h: 174 },
  primes: { x: 14, y: 246 },
  n: { x: 14, y: 268 },
  phi: { x: 14, y: 290 },
  pair: { x: 14, y: 314 },
  pair2: { x: 14, y: 330 },
  unlock: { x: 14, y: 356 },
  unlock2: { x: 14, y: 372 },
  text: { x: 14, y: 40 },
  digits: { x: 60, y: 40 },
  m: { x: 14, y: 62 },
  lock: { x: 14, y: 86 },
  lock2: { x: 186, y: 86 },
  keyN: { x: 45, y: 148 },
  keyE: { x: 131, y: 148 },
  labN: { x: 14, y: 148 },
  labE: { x: 100, y: 148 },
  keyLabel: { x: 14, y: 148 },
  empty: { x: 90, y: 148 },
  wireA: { x: 320, y: 104 },
  wireB: { x: 320, y: 212 },
  wireLabel: { x: 308, y: 182, anchor: 'end' },
  eve: { x: 14, y: 172 },
  lockIcon: { x: 322, y: 58 },
  keyIcon: { x: 322, y: 330 },
};

/* ───────────────────────────── steps ───────────────────────────── */

const STEPS = 9;
// seconds each step stays up while auto-playing (reading time)
const DUR = [3, 3.2, 4, 4.6, 3.8, 4.6, 5, 4.4];
const ease = (k: number) => (1 - Math.cos(Math.PI * clamp(k, 0, 1))) / 2;

const CAPTIONS: ReactNode[] = [
  <>The receiver picks two prime numbers, 47 and 59, and tells no one.</>,
  <>
    Multiplying them is easy: <b>n = 47 × 59 = 2773</b>. This number will be public.
  </>,
  <>
    From the primes the receiver also gets (47 − 1)(59 − 1) = <b>2668</b>. With only 2773, nobody
    knows a fast way to find 2668.
  </>,
  <>
    Next a matching pair. The receiver picks a secret, 157, sharing no factor with 2668, and
    divides: 2668 = 16 × 157 + 156, then 157 = 156 + 1. Working back, 1 = 17 × 157 − 2668, so{' '}
    <b>17 × 157 = 2669</b>, one more than 2668. 17 is the public partner.
  </>,
  <>
    The receiver publishes <b>2773 and 17</b> for anyone to use. 47, 59, 2668 and 157 never leave.
  </>,
  <>
    The sender turns text into numbers, two letters at a time (blank = 00, A = 01, …, Z = 26). “IT”
    is 09 20, so <b>m = 920</b>: the message as a number.
  </>,
  <>
    Lock with the public numbers: <b>c</b> = 920<sup>17</sup> mod 2773 = <b>948</b>, the locked
    number (mod 2773 keeps the remainder after dividing by 2773). Only 948 crosses the wire.
  </>,
  <>
    Unlock with the secret: 948<sup>157</sup> mod 2773 = <b>920</b>, “IT” again. 157 undoes 17
    because 17 × 157 leaves remainder 1 when divided by 2668.
  </>,
  <>
    An eavesdropper sees 2773, 17 and 948. To unlock they need 157, for 157 they need 2668, and for
    2668 they need 47 and 59: the factors of 2773.
  </>,
];

export default function RsaSteps() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [sel, setSel] = useState(0);
  const [step, bindStep, hovering] = useHoverPreview(sel);
  // seconds since the shown step began (drives the glides); the clock belongs to one step,
  // so a new step (click, auto-advance or hover preview) always starts its motion from 0
  const [clock, setClock] = useState({ k: 0, t: 0 });
  const t = reduced ? Infinity : clock.k === step ? clock.t : 0;
  const [playing, setPlaying] = useState(false);
  const started = useRef(false);

  useEffect(() => {
    if (reduced) {
      if (!started.current) setSel(STEPS - 1); // static poster: everything in place
      started.current = true;
      return;
    }
    if (!inView || started.current) return;
    const id = setTimeout(() => {
      started.current = true;
      setClock({ k: 0, t: 0 });
      setPlaying(true);
    }, 700);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  const autoOn = playing && !hovering && sel < STEPS - 1;
  useFrame(
    (dt) => {
      const nt = t + dt;
      if (autoOn && nt >= DUR[step]) {
        setSel(step + 1);
        setClock({ k: step + 1, t: 0 });
        if (step + 1 >= STEPS - 1) setPlaying(false);
      } else setClock({ k: step, t: nt });
    },
    inView && !reduced && (t < 3 || autoOn),
  );

  const choose = (v: string) => {
    started.current = true;
    setPlaying(false);
    setSel(Number(v));
    setClock({ k: Number(v), t: 0 });
  };

  // glide progress for the two moving pieces
  const pubK = step > 4 ? 1 : step === 4 ? ease((t - 0.3) / 1.3) : 0;
  const wireK = step > 6 ? 1 : step === 6 ? ease((t - 0.6) / 1.8) : 0;

  const draw = (g: Geo, cls: string) => {
    const fs = g.fs;
    const cw = fs * 0.6; // mono advance
    // opacity of a row: hidden before its step, full while it is the news (or used), dim after
    const vis = (from: number, hot: number[] = []) =>
      step < from ? 0 : step === from || hot.includes(step) ? 1 : 0.5;
    const op = (v: number) => ({ opacity: v, transition: 'opacity 300ms' });
    const head = (b: Box, label: string, color: string) => (
      <text
        x={b.x + 10}
        y={b.y + 14}
        dominantBaseline="middle"
        style={{
          fill: color,
          fontFamily: GT.mono,
          fontSize: 9,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}
      >
        {label}
      </text>
    );
    // the two published numbers start on their rows in the receiver's panel
    const fromN = { x: g.n.x + 14 * cw, y: g.n.y };
    const fromE = { x: g.pair.x + 4 * cw, y: g.pair.y };
    const nPos = { x: lerp(fromN.x, g.keyN.x, pubK), y: lerp(fromN.y, g.keyN.y, pubK) };
    const ePos = { x: lerp(fromE.x, g.keyE.x, pubK), y: lerp(fromE.y, g.keyE.y, pubK) };
    const keyFs = g.narrow ? 13 : 15;
    const tok = {
      x: lerp(g.wireA.x, g.wireB.x, wireK),
      y: lerp(g.wireA.y, g.wireB.y, wireK),
    };
    const tokOp = step === 6 ? clamp(wireK * 8, 0, 1) * clamp((1 - wireK) * 8, 0, 1) : 0;
    const eveOn = step === 8 ? 1 : 0;

    return (
      <svg
        viewBox={`0 0 ${g.W} ${g.H}`}
        className={cls}
        role="img"
        aria-label="RSA with the paper's numbers: the receiver keeps 47, 59, 2668 and 157 secret and publishes 2773 and 17; the sender locks 920 into 948; the receiver unlocks 948 back into 920"
      >
        <defs>
          <marker
            id={`rsa-arrow-${g.narrow ? 'n' : 'w'}`}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M0 0 L10 5 L0 10 z" style={{ fill: GT.lineStrong }} />
          </marker>
        </defs>

        {/* zones */}
        <rect
          x={g.sender.x}
          y={g.sender.y}
          width={g.sender.w}
          height={g.sender.h}
          style={{ fill: GT.paperRaised, stroke: GT.lineStrong, strokeWidth: 1 }}
        />
        <rect
          x={g.pub.x}
          y={g.pub.y}
          width={g.pub.w}
          height={g.pub.h}
          style={{ fill: 'none', stroke: PUB, strokeWidth: 1, strokeDasharray: '4 4' }}
        />
        <rect
          x={g.recv.x}
          y={g.recv.y}
          width={g.recv.w}
          height={g.recv.h}
          style={{ fill: GT.optSoft, fillOpacity: 0.7, stroke: SEC, strokeWidth: 1.4 }}
        />
        {head(g.sender, 'sender', GT.muted)}
        {head(g.pub, 'public: anyone can see', PUB)}
        {head(g.recv, 'receiver: secret', SEC)}

        {/* the wire, behind the token */}
        <line
          x1={g.wireA.x}
          y1={g.wireA.y}
          x2={g.wireB.x - (g.narrow ? 0 : 2)}
          y2={g.wireB.y - (g.narrow ? 2 : 0)}
          markerEnd={`url(#rsa-arrow-${g.narrow ? 'n' : 'w'})`}
          style={{ stroke: GT.lineStrong, strokeWidth: 1.2 }}
        />
        <g style={{ opacity: tokOp }}>
          <rect
            x={tok.x - 21}
            y={tok.y - 11}
            width={42}
            height={22}
            style={{ fill: GT.paperRaised, stroke: PUB, strokeWidth: 1.6 }}
          />
          <Line x={tok.x} y={tok.y + 0.5} size={12} anchor="middle" segs={[s('948', PUB, 700)]} />
        </g>
        <g style={op(step < 6 || (step === 6 && wireK < 0.5) ? 0 : vis(6, [8]))}>
          <Line
            x={g.wireLabel.x}
            y={g.wireLabel.y}
            size={fs}
            anchor={g.wireLabel.anchor}
            color={GT.muted}
            segs={
              g.narrow
                ? [s('c = '), s('948', PUB, 700)]
                : [s('on the wire: c = '), s('948', PUB, 700)]
            }
          />
        </g>

        {/* receiver's rows */}
        <g style={op(vis(0, [2, 8]))}>
          <Line
            {...g.primes}
            size={fs}
            segs={[s('p = '), s('47', SEC, 700), s('   q = '), s('59', SEC, 700)]}
          />
        </g>
        <g style={op(vis(1))}>
          <Line
            {...g.n}
            size={fs}
            segs={[s('n = '), s('47', SEC), s(' × '), s('59', SEC), s(' = '), s('2773', PUB, 700)]}
          />
        </g>
        <g style={op(vis(2, [8]))}>
          <Line
            {...g.phi}
            size={fs}
            segs={[s('('), s('47', SEC), s('−1)('), s('59', SEC), s('−1) = '), s('2668', SEC, 700)]}
          />
        </g>
        <g style={op(vis(3, [7, 8]))}>
          <Line
            {...g.pair}
            size={fs}
            segs={[s('e = '), s('17', PUB, 700), s('   d = '), s('157', SEC, 700)]}
          />
          <Line
            {...g.pair2}
            size={fs - 1}
            color={GT.inkSoft}
            segs={[
              s('17', PUB),
              s(' × '),
              s('157', SEC),
              s(' = 2669 = '),
              s('2668', SEC),
              s(' + 1'),
            ]}
          />
        </g>
        <g style={op(vis(7))}>
          <Line
            {...g.unlock}
            size={fs}
            segs={[s('948', PUB), sup('157', SEC), s(' mod '), s('2773', PUB)]}
          />
          <Line
            {...g.unlock2}
            size={fs}
            segs={[s('= '), s('920', GT.ink, 700), s(' → '), s('IT', SEC, 700)]}
          />
          <Key x={g.keyIcon.x} y={g.keyIcon.y} />
        </g>

        {/* public key: the two numbers glide out of the receiver's panel */}
        <g style={op(step < 4 ? 0.9 : 0)}>
          <Line
            x={g.empty.x}
            y={g.empty.y}
            size={fs - 1}
            anchor="middle"
            color={GT.muted}
            segs={[s('nothing published yet')]}
          />
        </g>
        {!g.narrow && (
          <g style={op(step < 4 ? 0 : clamp(pubK * 2 - 1, 0, 1))}>
            <Line
              x={g.keyLabel.x}
              y={g.keyLabel.y}
              size={fs - 1}
              anchor="middle"
              color={GT.muted}
              segs={[s('the public numbers')]}
            />
          </g>
        )}
        <g style={op(step < 4 ? 0 : vis(4, [6, 7, 8]))}>
          <g style={{ opacity: clamp(pubK * 2 - 1, 0, 1) }}>
            <Line {...g.labN} size={keyFs} segs={[s('n =')]} color={GT.inkSoft} />
            <Line {...g.labE} size={keyFs} segs={[s('e =')]} color={GT.inkSoft} />
          </g>
          <Line x={nPos.x} y={nPos.y} size={lerp(fs, keyFs, pubK)} segs={[s('2773', PUB, 700)]} />
          <Line x={ePos.x} y={ePos.y} size={lerp(fs, keyFs, pubK)} segs={[s('17', PUB, 700)]} />
        </g>

        {/* sender's rows */}
        <g style={op(vis(5))}>
          <Line
            {...g.text}
            size={g.narrow ? 14 : 18}
            weight={700}
            segs={[s('IT')]}
            color={GT.ink}
          />
          <Line
            {...g.digits}
            size={fs}
            color={GT.inkSoft}
            segs={g.narrow ? [s('I = 09, T = 20')] : [s('I = 09, T = 20')]}
          />
          <Line
            {...g.m}
            size={fs}
            segs={[s('m = '), s('920', GT.ink, 700), s('  the message', GT.muted)]}
          />
        </g>
        <g style={op(vis(6))}>
          <Line
            {...g.lock}
            size={fs}
            segs={[
              s('c = 920'),
              sup('17', PUB),
              s(' mod '),
              s('2773', PUB),
              ...(g.narrow ? [s(' = '), s('948', PUB, 700)] : []),
            ]}
          />
          {!g.narrow && (
            <Line
              {...g.lock2}
              size={fs}
              segs={[s('= '), s('948', PUB, 700), s('  locked', GT.muted)]}
            />
          )}
          <Lock x={g.lockIcon.x} y={g.lockIcon.y} />
        </g>

        {/* the eavesdropper */}
        <g style={op(eveOn)}>
          <Line
            x={g.eve.x}
            y={g.eve.y}
            size={fs - 0.5}
            color={GT.b}
            weight={600}
            segs={[s('eavesdropper sees 2773, 17, 948')]}
          />
          <Line
            x={g.eve.x}
            y={g.eve.y + (g.narrow ? 16 : 20)}
            size={fs - 0.5}
            color={GT.b}
            segs={[s('needs 157 ← 2668 ← 47 × 59')]}
          />
          {!g.narrow && (
            <Line
              x={g.eve.x}
              y={g.eve.y + 40}
              size={fs - 0.5}
              color={GT.b}
              segs={[s('= factor 2773 first')]}
            />
          )}
        </g>
      </svg>
    );
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-08.2"
      title="one locked message"
      kicker="n = 2773 = 47 × 59"
      hint="hover or tap a step"
      footer="Rivest, Shamir & Adleman 1978, §VIII"
    >
      {draw(WIDE, 'hidden w-full sm:block')}
      {draw(NARROW, 'block w-full sm:hidden')}

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
        <Chip color={SEC}>secret: only the receiver</Chip>
        <Chip color={PUB}>public: anyone</Chip>
        <span className="font-mono text-[0.68rem] text-muted">
          m: the message as a number · c: the locked number
        </span>
      </div>
      <div className="mt-3">
        <Seg
          label="step"
          options={Array.from({ length: STEPS }, (_, i) => ({
            value: String(i),
            label: String(i + 1),
          }))}
          value={String(step)}
          onChange={choose}
          bind={(v) => bindStep(Number(v))}
        />
      </div>
      <Reserve
        active={step}
        items={CAPTIONS.map((c, i) => (
          <Caption key={i}>{c}</Caption>
        ))}
      />
    </Specimen>
  );
}
