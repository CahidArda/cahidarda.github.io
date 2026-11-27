import { useEffect, useRef, useState } from 'react';
import {
  Btn,
  Chip,
  GT,
  Specimen,
  T,
  clamp,
  useFrame,
  useInView,
  useReducedMotion,
} from '../../gt/shared';

/**
 * The small example from section VIII of the RSA paper (Rivest, Shamir, Adleman 1978):
 * p = 47, q = 59, n = 2773, public exponent e = 17, private exponent d = 157.
 * Two letters per block (blank = 00, A = 01, ..., Z = 26), so ITS ALL GREEK TO ME becomes
 * 0920 1900 0112 ... and locks to 0948 2342 1084 ..., exactly as printed in the paper.
 *
 * CX-08.3: the whole message, after CX-08.2 has walked through one block. The blocks ride one
 * belt at one constant speed: through the padlock (locking needs only the public 2773 and 17)
 * and through the key (unlocking needs the secret 157, which comes from knowing 47 and 59).
 */
const P = 47;
const Q = 59;
const N = P * Q; // 2773
const E = 17;
const D = 157; // E * D = 2669 = 1 + 46 * 58

function powmod(b: number, e: number, m: number) {
  let r = 1;
  b %= m;
  while (e > 0) {
    if (e & 1) r = (r * b) % m;
    b = (b * b) % m;
    e >>= 1;
  }
  return r;
}
const code = (c: string) => (c === ' ' ? 0 : c.charCodeAt(0) - 64);
const chr = (n: number) => (n === 0 ? ' ' : String.fromCharCode(64 + n));
const clean = (s: string) =>
  s
    .toUpperCase()
    .replace(/[^A-Z ]/g, '')
    .slice(0, 20);
function blocks(msg: string) {
  const out: { m: number; c: number; text: string }[] = [];
  for (let i = 0; i < msg.length; i += 2) {
    const m = code(msg[i]) * 100 + (i + 1 < msg.length ? code(msg[i + 1]) : 0);
    out.push({ m, c: powmod(m, E, N), text: chr(Math.floor(m / 100)) + chr(m % 100) });
  }
  return out;
}
const pad4 = (n: number) => String(n).padStart(4, '0');

const SPEED = 150; // belt speed, units per second
const GAP = 62; // distance between blocks on the belt

type Geo = { W: number; H: number; x0: number; lock: number; key: number; x1: number; y: number };
const WIDE: Geo = { W: 640, H: 196, x0: 28, lock: 230, key: 420, x1: 612, y: 92 };
const NARROW: Geo = { W: 360, H: 220, x0: 6, lock: 124, key: 240, x1: 354, y: 104 };

export function Lock({ x, y }: { x: number; y: number }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      <path
        d="M-9 -12 V-19 A9 9 0 0 1 9 -19 V-12"
        style={{ fill: 'none', stroke: GT.eq, strokeWidth: 3.2 }}
      />
      <rect x={-14} y={-13} width={28} height={24} style={{ fill: GT.eq }} />
      <circle cx={0} cy={-3} r={3} style={{ fill: GT.paperRaised }} />
      <rect x={-1.2} y={-2} width={2.4} height={7} style={{ fill: GT.paperRaised }} />
    </g>
  );
}
export function Key({ x, y }: { x: number; y: number }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      <circle cx={-10} cy={0} r={8} style={{ fill: 'none', stroke: GT.opt, strokeWidth: 3.2 }} />
      <path
        d="M-2 0 H16 M10 0 V6 M15 0 V5"
        style={{ fill: 'none', stroke: GT.opt, strokeWidth: 3.2 }}
      />
    </g>
  );
}

export default function Padlock() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [msg, setMsg] = useState('ITS ALL GREEK TO ME');
  const bl = blocks(msg);
  // s = how far the belt has run (units); null before it starts (poster: everything delivered)
  const [s, setS] = useState<number | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (reduced || !inView || started.current) return;
    const id = setTimeout(() => {
      started.current = true;
      setS(0);
    }, 700);
    return () => clearTimeout(id);
  }, [inView, reduced]);

  // the belt runs until the last block has been delivered, on every layout
  const runLen = (g: Geo) => g.x1 - g.x0 + GAP * (bl.length - 1) + 20;
  const total = Math.max(runLen(WIDE), runLen(NARROW));
  useFrame(
    (dt) => setS((v) => (v === null ? v : Math.min(total, v + dt * SPEED))),
    inView && !reduced && s !== null && s < total,
  );

  const restart = (m: string) => {
    started.current = true;
    setMsg(m);
    if (!reduced) setS(-SPEED * 0.9); // empty belt for 0.9 s, then it runs again
  };

  const draw = (g: Geo, cls: string) => {
    const run = s === null ? Infinity : s;
    // block i enters at x0 when the belt has run i * GAP, leaves at x1
    const xs = bl.map((_, i) => g.x0 + run - i * GAP);
    const delivered = bl.filter((_, i) => xs[i] >= g.x1).length;
    const out = bl
      .slice(0, delivered)
      .map((b) => b.text)
      .join('')
      .trimEnd();
    const fs = g.W < 400 ? 9 : 10;
    const narrow = g.W < 400;
    return (
      <svg
        viewBox={`0 0 ${g.W} ${g.H}`}
        className={cls}
        role="img"
        aria-label="A message is locked with the public number 2773 and opened with the private key 157"
      >
        {/* the belt */}
        <line
          x1={g.x0}
          x2={g.x1}
          y1={g.y}
          y2={g.y}
          style={{ stroke: GT.lineStrong, strokeWidth: 1 }}
        />
        {/* the blocks */}
        {bl.map((b, i) => {
          const x = xs[i];
          if (x < g.x0 - 1 || x > g.x1) return null;
          const fade = clamp((x - g.x0) / 12, 0, 1) * clamp((g.x1 - x) / 12, 0, 1);
          // the number changes while the block is hidden inside a station (drawn on top)
          const stage = x < g.lock ? 0 : x < g.key ? 1 : 2;
          const color = stage === 2 ? GT.opt : stage === 1 ? GT.eq : GT.ink;
          return (
            <g key={i} style={{ opacity: fade }}>
              <rect
                x={x - 21}
                y={g.y - 11}
                width={42}
                height={22}
                style={{ fill: GT.paperRaised, stroke: color, strokeWidth: 1.4 }}
              />
              <T x={x} y={g.y + 0.5} size={12} color={color}>
                {pad4(stage === 1 ? b.c : b.m)}
              </T>
            </g>
          );
        })}

        {/* stations, drawn over the belt so a block changes out of sight */}
        <rect
          x={g.lock - 32}
          y={g.y - 16}
          width={64}
          height={32}
          style={{ fill: GT.paper, stroke: GT.eq, strokeWidth: 1.5 }}
        />
        <rect
          x={g.key - 32}
          y={g.y - 16}
          width={64}
          height={32}
          style={{ fill: GT.paper, stroke: GT.opt, strokeWidth: 1.5 }}
        />

        {/* stations: icon above, rule below */}
        <Lock x={g.lock} y={g.y - 34} />
        <Key x={g.key} y={g.y - 40} />
        <T x={g.lock} y={g.y + 28} size={fs + 1} color={GT.eq} weight={600}>
          m
          <tspan dy={-5} fontSize={fs - 1}>
            {E}
          </tspan>
          <tspan dy={5}>{` mod ${N}`}</tspan>
        </T>
        <T x={g.lock} y={g.y + 42} size={fs} color={GT.eq}>
          {narrow ? 'lock: 2773, 17' : 'lock with public 2773, 17'}
        </T>
        <T x={g.key} y={g.y + 28} size={fs + 1} color={GT.opt} weight={600}>
          c
          <tspan dy={-5} fontSize={fs - 1}>
            {D}
          </tspan>
          <tspan dy={5}>{` mod ${N}`}</tspan>
        </T>
        <T x={g.key} y={g.y + 42} size={fs} color={GT.opt}>
          {narrow ? 'unlock: 157' : 'unlock with secret 157'}
        </T>

        {/* sender and receiver */}
        <T x={g.x0} y={14} anchor="start" size={fs}>
          sender
        </T>
        <T x={g.x1} y={14} anchor="end" size={fs}>
          receiver
        </T>
        <T x={(g.lock + g.key) / 2} y={14} size={fs} color={GT.eq}>
          {narrow ? 'on the wire: c' : 'on the wire: only locked blocks c'}
        </T>
        <T x={g.x1} y={g.H - 14} anchor="end" size={13} color={GT.opt} weight={600}>
          {out || ' '}
        </T>
        <T x={g.x0} y={g.H - 14} anchor="start" size={13} color={GT.ink}>
          {msg}
        </T>
      </svg>
    );
  };

  return (
    <Specimen
      rootRef={rootRef}
      code="CX-08.3"
      title="many blocks"
      kicker="n = 2773 = 47 × 59"
      hint={
        <span className="inline-flex flex-wrap items-center gap-2">
          <input
            aria-label="message"
            value={msg}
            maxLength={20}
            onChange={(e) => restart(clean(e.target.value))}
            className="w-[11.5rem] border border-line-strong bg-paper px-2 py-1 font-mono text-[0.72rem] uppercase text-ink"
          />
          <Btn onClick={() => restart(msg)}>replay</Btn>
        </span>
      }
      footer="Rivest, Shamir & Adleman 1978, §VIII"
    >
      {draw(WIDE, 'hidden w-full sm:block')}
      {draw(NARROW, 'block w-full sm:hidden')}
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
        <Chip color={GT.ink}>m: a block of the message as a number</Chip>
        <Chip color={GT.eq}>c: the locked number</Chip>
      </div>
    </Specimen>
  );
}
