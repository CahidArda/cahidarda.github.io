/**
 * Procedure plate: latitude by a noon sight, step by step (the reference StepPlate).
 *   1 Wait for noon · 2 Bring the sun down to the horizon · 3 Read the arc · 4 Correct the
 *   reading · 5 Look up the sun's declination · 6 Put it into the equation.
 * Instrument view per step (sky panorama, eyepiece, arc and drum, corrections ladder, almanac,
 * latitude scale) beside a STABLE reference: the Earth cut along your meridian with φ, δ and h,
 * which never changes layout between steps; the step's quantity is highlighted.
 * All numbers and drawn coordinates come from ./noonProcGeom.ts (swept by noonProcGeom.sweep.ts);
 * the eyepiece reuses `sextantView`, the same optics as the 3D sextant plate.
 */
import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import {
  C,
  Eq,
  InkText,
  Known,
  Segmented,
  Slider,
  StepPlate,
  Sym,
  Working,
  arcPath,
  useMeasuredWidth,
  type Role,
  type StepDef,
} from '../shared';
import {
  DOY_PRESET,
  EYE_HEIGHT,
  PLACES,
  READ_MAX,
  READ_MIN,
  T_MAX,
  T_MIN,
  T_START,
  DRUM_STEP,
  readingOf,
  splitReading,
  almanac,
  am,
  corrScene,
  dm,
  eyeScene,
  eyeSay,
  skySay,
  resultSay,
  hhmm,
  readScene,
  reduce,
  refScene,
  resultScene,
  sight,
  skyScene,
  snapTime,
  type Sight,
} from './noonProcGeom';
import { dateLabel } from './noonGeom';

const solid = (role: Role, w = 1.1) => ({ fill: 'none', stroke: C[role], strokeWidth: w });

/* ───────────────────────────── the stable reference ───────────────────────────── */

type Hi = 'sun' | 'h' | 'horizon' | 'dec' | 'phi';
const HIGHLIGHT: Hi[][] = [['sun', 'horizon'], ['h', 'horizon'], ['h'], ['h', 'horizon'], ['dec', 'sun'], ['phi']];

function Reference({ s, step }: { s: Sight; step: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = refScene(s, W);
  const hi = new Set(HIGHLIGHT[step] ?? []);
  const o = (k: Hi) => (hi.has(k) ? 1 : 0.4);
  const wd = (k: Hi, base: number) => (hi.has(k) ? base * 1.9 : base);
  const T = (id: string, role: Role, k?: Hi, size?: number) => {
    const l = g.labels[id];
    if (!l) return null;
    return (
      <g style={{ opacity: k ? o(k) : 1 }}>
        <InkText x={l.x} y={l.y} anchor={l.anchor} baseline="central" size={size ?? l.size} role={role}>
          {l.text}
        </InkText>
      </g>
    );
  };
  return (
    <div ref={ref}>
      <svg
        viewBox={`0 0 ${W} ${g.H}`}
        width="100%"
        style={{ display: 'block' }}
        role="img"
        aria-label={`The Earth cut along your meridian: latitude ${s.place.lat.toFixed(1)}, the sun's declination ${s.dec.toFixed(1)}, its noon altitude ${s.hTrue.toFixed(1)} degrees`}
      >
        {/* sunlight: parallel rays at the declination (brass, like δ) */}
        <g style={{ opacity: o('sun') }}>
          {g.rays.map((r) => (
            <line key={r.o} x1={r.a.x} y1={r.a.y} x2={r.b.x} y2={r.b.y} style={{ ...solid('brass', 0.7), opacity: 0.6 }} />
          ))}
          <line x1={g.sunFrom.x} y1={g.sunFrom.y} x2={g.obs.x} y2={g.obs.y} style={solid('brass', wd('sun', 1.1))} />
        </g>
        {T('sun', 'soft', 'sun')}
        {/* the Earth */}
        <circle cx={g.cx} cy={g.cy} r={g.Re} style={{ ...solid('ink', 1.3), fill: C.paper }} />
        <line x1={g.segs[0].a.x} y1={g.cy} x2={g.segs[0].b.x} y2={g.cy} style={{ ...solid('soft', 0.8), strokeDasharray: '5 4' }} />
        <line x1={g.cx} y1={g.segs[1].a.y} x2={g.cx} y2={g.segs[1].b.y} style={solid('faint', 0.7)} />
        {T('equator', 'soft')}
        {T('pole', 'soft')}
        {/* centre: φ (red) and δ (brass) */}
        <line x1={g.cx} y1={g.cy} x2={g.zen.x} y2={g.zen.y} style={{ ...solid('soft', 0.8), strokeDasharray: '5 4' }} />
        <line x1={g.cx} y1={g.cy} x2={g.sub.x} y2={g.sub.y} style={{ ...solid('brass', 0.9), strokeDasharray: '3 3', opacity: o('dec') }} />
        <path d={arcPath(g.cx, g.cy, g.phiArcR, 0, g.phi)} style={{ ...solid('red', wd('phi', 1.3)), opacity: o('phi') }} />
        {T('phi', 'red', 'phi')}
        <path d={arcPath(g.cx, g.cy, g.decArcR, 0, g.dec)} style={{ ...solid('brass', wd('dec', 1.3)), opacity: o('dec') }} />
        {T('dec', 'brass', 'dec')}
        <circle cx={g.cx} cy={g.cy} r={2} style={{ fill: C.ink }} />
        {/* you: horizon, zenith, h (sky) */}
        <line x1={g.horFar.x} y1={g.horFar.y} x2={g.horSun.x} y2={g.horSun.y} style={{ ...solid('ink', wd('horizon', 1.3)), opacity: hi.has('horizon') || hi.size === 0 ? 1 : 0.7 }} />
        {T('horizon', 'soft', 'horizon')}
        <path d={arcPath(g.obs.x, g.obs.y, g.hArcR, g.hFrom, g.hTo)} style={{ ...solid('sky', wd('h', 1.4)), opacity: o('h') }} />
        {T('h', 'sky', 'h')}
        <circle cx={g.obs.x} cy={g.obs.y} r={3.5} style={{ fill: C.paper, stroke: C.ink, strokeWidth: 1.3 }} />
        {T('you', 'ink')}
        {T('zenith', 'soft')}
      </svg>
    </div>
  );
}

/* ───────────────────────────── instrument views ───────────────────────────── */

function SkyView({ s, t }: { s: Sight; t: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = skyScene(s, t, W);
  const atNoon = t === 12;
  const alts = [30, 60];
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img"
        aria-label={`The sky facing ${s.bearing === 'S' ? 'south' : 'north'} at ${hhmm(t)}: the sun at ${g.now.alt.toFixed(1)} degrees`}>
        {alts.map((a) => (
          <g key={a}>
            <line x1={g.pad} y1={g.horizonY - a * g.ky} x2={W - g.pad} y2={g.horizonY - a * g.ky} style={{ ...solid('faint', 0.6), strokeDasharray: '2 5' }} />
            <InkText x={g.pad + 2} y={g.horizonY - a * g.ky - 5} kind="num" size={10} role="soft">{`${a}°`}</InkText>
          </g>
        ))}
        {g.path.map((run, i) => (
          <polyline key={i} points={run.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')} style={{ ...solid('brass', 0.9), strokeDasharray: '4 3' }} />
        ))}
        {/* the meridian: due south (or north) */}
        <line x1={g.mid} y1={g.top - 8} x2={g.mid} y2={g.horizonY} style={{ ...solid(atNoon ? 'sky' : 'soft', atNoon ? 1.3 : 0.7), strokeDasharray: atNoon ? undefined : '3 4' }} />
        <InkText x={g.mid + 5} y={g.top - 10} size={11} role="soft">{`due ${s.bearing === 'S' ? 'south' : 'north'}`}</InkText>
        {g.now.alt > 0 && (
          <>
            <line x1={g.sun.x} y1={g.sun.y} x2={g.sun.x} y2={g.horizonY} style={{ ...solid('sky', 0.9) }} />
            <circle cx={g.sun.x} cy={g.sun.y} r={g.sunR} style={{ fill: C.brass, stroke: C.paper, strokeWidth: 1 }} />
          </>
        )}
        {/* sea and horizon */}
        <rect x={0} y={g.horizonY} width={W} height={g.H - g.horizonY} style={{ fill: C.sky, opacity: 0.14 }} />
        <line x1={0} y1={g.horizonY} x2={W} y2={g.horizonY} style={solid('ink', 1.3)} />
        {g.marks.map((m) =>
          m.x > 10 && m.x < W - 10 ? (
            <InkText key={m.t} x={m.x} y={g.horizonY + 15} kind="num" anchor="middle" size={11} role="ink">{m.t}</InkText>
          ) : null,
        )}
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={atNoon ? 'sky' : 'soft'}>
          {skySay(g, t)}
        </InkText>
      </svg>
    </div>
  );
}

function EyeView({ s, reading }: { s: Sight; reading: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = eyeScene(s, reading, W);
  const { R, cx, cy } = g;
  const sx = cx + R * 0.45;
  const say = eyeSay(g);
  const clip = `eye-${Math.round(W)}`;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img" aria-label={`Through the telescope: ${say}`}>
        <defs>
          <clipPath id={clip}>
            <circle cx={cx} cy={cy} r={R} />
          </clipPath>
        </defs>
        <circle cx={cx} cy={cy} r={R + 5} style={{ ...solid('ink', 1.2), fill: C.paper }} />
        <g clipPath={`url(#${clip})`}>
          <rect x={cx} y={cy - R} width={R} height={2 * R} style={{ fill: C.deep }} />
          <rect x={cx - R} y={g.horY} width={R} height={2 * R} style={{ fill: C.sky, opacity: 0.22 }} />
          <line x1={cx - R} y1={g.horY} x2={cx} y2={g.horY} style={solid('sky', 1.6)} />
          <line x1={cx} y1={g.horY} x2={cx + R} y2={g.horY} style={{ ...solid('sky', 0.9), strokeDasharray: '3 3' }} />
          {g.visible && <circle cx={sx} cy={g.sunY} r={g.sunR} style={{ fill: C.brass, stroke: C.paper, strokeWidth: 0.8 }} />}
        </g>
        {!g.visible && (
          <path
            d={g.gap > 0 ? `M ${sx - 7} ${cy - R + 18} L ${sx} ${cy - R + 9} L ${sx + 7} ${cy - R + 18}` : `M ${sx - 7} ${cy + R - 18} L ${sx} ${cy + R - 9} L ${sx + 7} ${cy + R - 18}`}
            style={solid('brass', 1.8)}
          />
        )}
        <line x1={cx} y1={cy - R} x2={cx} y2={cy + R} style={solid('ink', 0.8)} />
        <circle cx={cx} cy={cy} r={R} style={solid('ink', 1)} />
        <InkText x={cx - R * 0.5} y={cy - R - 12} anchor="middle" size={11} role="soft">clear glass</InkText>
        <InkText x={cx + R * 0.5} y={cy - R - 12} anchor="middle" size={11} role="soft">mirror</InkText>
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role={g.status === 'on' ? 'sky' : 'soft'}>{say}</InkText>
      </svg>
    </div>
  );
}

function ReadView({ reading }: { reading: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = readScene(reading, W);
  const { drum } = g;
  const ticks: ReactNode[] = [];
  for (let i = 0; i < 60; i++) {
    const a = 90 + drum.turn + i * 6; // division i, turned so the minutes sit under the top index
    const rad = (a * Math.PI) / 180;
    const L = i % 10 === 0 ? 12 : i % 5 === 0 ? 8 : 5;
    const p = { x: drum.cx + drum.r * Math.cos(rad), y: drum.cy - drum.r * Math.sin(rad) };
    const q = { x: drum.cx + (drum.r - L) * Math.cos(rad), y: drum.cy - (drum.r - L) * Math.sin(rad) };
    ticks.push(<line key={i} x1={p.x} y1={p.y} x2={q.x} y2={q.y} style={solid('ink', 0.8)} />);
    if (i % 10 === 0 && Math.sin(rad) > -0.2) {
      const t = { x: drum.cx + (drum.r - 22) * Math.cos(rad), y: drum.cy - (drum.r - 22) * Math.sin(rad) };
      ticks.push(<InkText key={`n${i}`} x={t.x} y={t.y} kind="num" anchor="middle" baseline="central" size={10} role="ink" halo={false}>{`${i}`}</InkText>);
    }
  }
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img"
        aria-label={`The arc reads ${g.deg} degrees and the drum ${g.min.toFixed(1)} minutes`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">the arc (whole degrees)</InkText>
        <line x1={10} y1={g.arcY} x2={W - 10} y2={g.arcY} style={solid('ink', 1.2)} />
        {g.ticks.map((t, i) =>
          t.x < 10 || t.x > W - 10 ? null : (
            <g key={i}>
              <line x1={t.x} y1={g.arcY} x2={t.x} y2={g.arcY + (t.major ? 14 : 7)} style={solid('ink', t.major ? 1 : 0.7)} />
              {t.label && <InkText x={t.x} y={g.arcY + 27} kind="num" anchor="middle" size={12} role={Number(t.label) === g.deg ? 'sky' : 'ink'}>{`${t.label}°`}</InkText>}
            </g>
          ),
        )}
        {/* the index on the arm */}
        <path d={`M ${g.index.x} ${g.arcY - 2} L ${g.index.x - 6} ${g.arcY - 14} L ${g.index.x + 6} ${g.arcY - 14} Z`} style={{ fill: C.sky }} />
        <line x1={g.index.x} y1={g.arcY - 14} x2={g.index.x} y2={g.arcY + 16} style={solid('sky', 1.4)} />
        <InkText x={14} y={drum.cy - drum.r - 10} kind="label" size={12} role="soft">the drum (minutes)</InkText>
        <circle cx={drum.cx} cy={drum.cy} r={drum.r} style={{ ...solid('ink', 1.2), fill: C.deep }} />
        {ticks}
        <path d={`M ${drum.cx} ${drum.cy - drum.r - 2} L ${drum.cx - 6} ${drum.cy - drum.r - 13} L ${drum.cx + 6} ${drum.cy - drum.r - 13} Z`} style={{ fill: C.sky }} />
        <InkText x={drum.cx} y={drum.cy} kind="num" anchor="middle" baseline="central" size={14} role="sky">
          {`${g.deg}° ${g.min.toFixed(1)}′`}
        </InkText>
      </svg>
    </div>
  );
}

function CorrView({ s, reading }: { s: Sight; reading: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = corrScene(s, reading, W);
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img"
        aria-label={`Corrections: ${g.rows.map((r) => `${r.name} ${r.value}`).join(', ')}`}>
        <InkText x={12} y={18} kind="label" size={12} role="soft">{`running total, arcminutes (grid ${g.unit}′)`}</InkText>
        {g.grid.map((t, i) => (
          <g key={i}>
            <line x1={t.x} y1={g.top - 6} x2={t.x} y2={g.top + g.rowH * 5} style={{ ...solid('faint', 0.6), strokeDasharray: t.v === 0 ? undefined : '2 4' }} />
            <InkText x={t.x} y={g.top - 12} kind="num" anchor="middle" size={9} role="soft">{t.v > 0 ? `+${t.v}` : `${t.v}`}</InkText>
          </g>
        ))}
        {g.rows.map((r, i) => {
          const isEnd = r.from === r.to;
          const dir = r.xb >= r.xa ? 1 : -1;
          const next = g.rows[i + 1];
          return (
            <g key={r.id}>
              <InkText x={r.nx} y={r.y - 6} size={12} role={r.role}>{r.name}</InkText>
              <InkText x={r.nx} y={r.y + 9} kind="num" size={11} role={r.role}>{r.value}</InkText>
              {isEnd ? (
                <path d={`M ${r.xa} ${r.y - 9} V ${r.y + 9} M ${r.xa - 5} ${r.y} H ${r.xa + 5}`} style={solid(r.role, 1.8)} />
              ) : (
                <>
                  <line x1={r.xa} y1={r.y} x2={r.xb} y2={r.y} style={solid('soft', 1.6)} />
                  {Math.abs(r.xb - r.xa) > 5 && (
                    <path d={`M ${r.xb - dir * 6} ${r.y - 4} L ${r.xb} ${r.y} L ${r.xb - dir * 6} ${r.y + 4}`} style={solid('soft', 1.4)} />
                  )}
                </>
              )}
              {next && <line x1={r.xb} y1={r.y} x2={r.xb} y2={next.y} style={{ ...solid('faint', 0.8), strokeDasharray: '2 2' }} />}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Almanac extract as an HTML table (inline styles: widget-local, no global CSS). */
function AlmanacView({ s }: { s: Sight }) {
  const a = almanac(s);
  const cell = { padding: '0.3rem 0.6rem', borderBottom: `1px dashed ${C.faint}`, textAlign: 'left' as const };
  const head = { ...cell, fontFamily: 'var(--font-mono)', fontSize: '0.6rem', letterSpacing: '0.14em', textTransform: 'uppercase' as const, color: C.soft, fontWeight: 400 };
  const num = { ...cell, fontFamily: 'var(--font-mono)', fontSize: '0.85rem', fontVariantNumeric: 'tabular-nums' as const };
  return (
    <div role="group" aria-label="Nautical almanac extract" style={{ padding: '0.8rem 0.9rem', minHeight: '17rem', background: C.paper }}>
      <div style={{ fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '1.05rem', color: C.ink, marginBottom: '0.4rem' }}>
        {`The Sun, ${dateLabel(s.doy)} 2026`}
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', margin: 0 }}>
        <thead>
          <tr>
            <th style={head}>GMT</th>
            <th style={head}>declination δ</th>
          </tr>
        </thead>
        <tbody>
          {a.rows.map((r) => (
            <tr key={r.h}>
              <td style={{ ...num, color: C.soft }}>{hhmm(r.h)}</td>
              <td style={{ ...num, color: C.soft }}>{dm(r.dec, ['N', 'S'])}</td>
            </tr>
          ))}
          <tr>
            <td style={{ ...num, color: C.ink, borderBottom: `1.5px solid ${C.brass}` }}>{`${hhmm(s.utc)} your noon`}</td>
            <td style={{ ...num, color: C.brass, borderBottom: `1.5px solid ${C.brass}` }}>{dm(a.at.dec, ['N', 'S'])}</td>
          </tr>
        </tbody>
      </table>
      <p style={{ margin: '0.6rem 0 0', fontFamily: 'var(--font-display)', fontStyle: 'italic', fontSize: '0.95rem', lineHeight: 1.4, color: C.soft }}>
        Your noon in Greenwich time is 12:00 minus your longitude ({Math.abs(s.place.lon).toFixed(2)}°{' '}
        {s.place.lon >= 0 ? 'E' : 'W'}) at 4 minutes a degree. δ moves under 1′ an hour, so a rough time is enough.
      </p>
    </div>
  );
}

function ResultView({ s, phi }: { s: Sight; phi: number }) {
  const [ref, W] = useMeasuredWidth<HTMLDivElement>(420);
  const g = resultScene(s, phi, W);
  const miss = g.miss;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${g.H}`} width="100%" style={{ display: 'block' }} role="img"
        aria-label={`Measured latitude ${dm(phi, ['N', 'S'])}, true ${dm(s.place.lat, ['N', 'S'])}`}>
        <InkText x={14} y={18} kind="label" size={12} role="soft">latitude, north up</InkText>
        <line x1={g.x} y1={g.top} x2={g.x} y2={g.bot} style={solid('ink', 1.1)} />
        {g.ticks.map((t, i) => (
          <g key={i}>
            <line x1={g.x - (t.major ? 8 : 4)} y1={t.y} x2={g.x} y2={t.y} style={solid('ink', 0.8)} />
            {t.major && <InkText x={g.x - 12} y={t.y} kind="num" anchor="end" baseline="central" size={10} role="soft">{dm(t.v)}</InkText>}
          </g>
        ))}
        <line x1={g.x} y1={g.yTrue} x2={W - 10} y2={g.yTrue} style={{ ...solid('ink', 0.9), strokeDasharray: '4 3' }} />
        <InkText x={W - 12} y={g.yTrue + (miss > 0 ? 11 : -9)} anchor="end" baseline="central" size={12} role="ink">{`${s.place.name} ${dm(s.place.lat, ['N', 'S'])}`}</InkText>
        <line x1={g.x} y1={g.yMeas} x2={W - 10} y2={g.yMeas} style={solid('red', 1.6)} />
        <circle cx={g.x} cy={g.yMeas} r={4} style={{ fill: C.red }} />
        <InkText x={W - 12} y={g.yMeas + (miss > 0 ? -9 : 11)} anchor="end" baseline="central" size={12} role="red">{`you: ${dm(phi, ['N', 'S'])}`}</InkText>
        <InkText x={W / 2} y={g.H - 10} anchor="middle" size={12} role="soft">
          {resultSay(miss)}
        </InkText>
      </svg>
    </div>
  );
}

/* ───────────────────────────── the plate ───────────────────────────── */

export default function NoonSightProcedure() {
  const [doy, setDoy] = useState(DOY_PRESET);
  const [placeId, setPlaceId] = useState('ist');
  const [t, setT] = useState(T_START);
  const [arm, setArm] = useState(0);
  const [drum, setDrum] = useState(0);
  const reading = readingOf(arm, drum);
  const setReading = (v: number) => {
    const sp = splitReading(v);
    setArm(sp.deg);
    setDrum(sp.min);
  };
  const [step, setStep] = useState(0);
  const place = PLACES.find((p) => p.id === placeId)!;
  const s = useMemo(() => sight(doy, place), [doy, place]);
  const exact = Math.round(s.hs * 600) / 600; // the true reading, to 0.1′ (the slider's step)
  const eye = eyeScene(s, reading, 400);
  const atNoon = t === 12 && s.visible;
  const aligned = eye.status === 'on';
  const r = reduce(s, reading);
  const south = s.bearing === 'S';

  // a new date or place: keep the procedure where it was if the sight had been taken
  const resettle = (fn: () => void, next: Sight) => {
    fn();
    if (aligned && next.visible) setReading(Math.round(next.hs * 600) / 600);
    else if (!next.visible) setStep(0);
    else setStep((k) => Math.min(k, 1));
  };

  const known = [true, atNoon, aligned, aligned, aligned, aligned];
  const kn = (i: number) => step >= i && known[i];
  const hDone = kn(3);
  const readDone = kn(2);
  const decDone = kn(4);
  const phiDone = kn(5);

  const steps: StepDef[] = [
    {
      id: 'noon',
      title: 'Wait for noon',
      text: s.visible ? (
        <>
          Face {south ? 'south' : 'north'} and watch the sun climb. It is highest when it crosses your meridian,
          due {south ? 'south' : 'north'}: local noon. Near the top it hardly moves, so there is time to measure.
        </>
      ) : (
        <>On {dateLabel(doy)} the noon sun in {place.name} stays under {5}° (polar night or near it). Pick another date.</>
      ),
      controls: (
        <Slider label="local sun time" value={t} min={T_MIN} max={T_MAX} step={1 / 60} onChange={(v) => setT(snapTime(v))} format={hhmm} />
      ),
      gated: true,
      done: atNoon,
      hint: s.visible ? 'Drag the time to noon (12:00)' : 'No noon sight today',
    },
    {
      id: 'align',
      title: 'Bring the sun down',
      text: (
        <>
          Look at the sea horizon through the clear glass and swing the arm until the sun’s image in the mirror
          sits with its lower edge on the horizon: the arm for whole degrees, the drum for minutes.
        </>
      ),
      controls: (
        <>
          <Slider label="swing the arm" value={arm} min={READ_MIN} max={READ_MAX} step={1} onChange={setArm} format={(v) => `${v}°`} />
          <Slider label="turn the drum" value={drum} min={0} max={59.9} step={DRUM_STEP} onChange={setDrum} format={(v) => `${v.toFixed(1)}′`} />
        </>
      ),
      gated: true,
      done: aligned,
      hint: 'Within 3′ of the horizon',
    },
    {
      id: 'read',
      title: 'Read the arc',
      text: (
        <>
          Whole degrees from the arc under the index, minutes from the drum. This is the sextant altitude,
          H<sub>s</sub> = <span style={{ color: C.sky }}>{dm(reading)}</span>.
        </>
      ),
    },
    {
      id: 'correct',
      title: 'Correct it',
      text: (
        <>
          Your eye is {EYE_HEIGHT} m above the sea, so the horizon sits a little low (dip); you measured the
          sun’s edge, not its centre (semi-diameter); the air bends its light up (refraction).
        </>
      ),
    },
    {
      id: 'dec',
      title: 'Look up δ',
      text: (
        <>
          The almanac gives the sun’s declination δ, how far {s.dec >= 0 ? 'north' : 'south'} of the equator it stands
          today, by the hour in Greenwich time.
        </>
      ),
    },
    {
      id: 'solve',
      title: 'Work out φ',
      text: south ? (
        <>The sun is due south, so the angle from your zenith to the sun is 90° − h, and your latitude is that plus δ.</>
      ) : (
        <>The sun is due north of you here, so the signs flip: φ = h − 90° + δ.</>
      ),
    },
  ];

  const view = (i: number) => {
    if (i === 0) return <SkyView s={s} t={t} />;
    if (i === 1) return <EyeView s={s} reading={reading} />;
    if (i === 2) return <ReadView reading={reading} />;
    if (i === 3) return <CorrView s={s} reading={reading} />;
    if (i === 4) return <AlmanacView s={s} />;
    return <ResultView s={s} phi={r.phi} />;
  };

  return (
    <StepPlate
      n={5}
      title="Latitude by a noon sight, step by step"
      wide
      steps={steps}
      step={step}
      onStep={setStep}
      view={view}
      viewLabel="What you see and do"
      reference={(i) => <Reference s={s} step={i} />}
      referenceLabel="What you are measuring (your meridian)"
      settings={
        <>
          <Segmented
            label="place"
            options={PLACES.map((p) => ({ value: p.id, label: p.name }))}
            value={placeId}
            onChange={(id) => {
              const p = PLACES.find((q) => q.id === id)!;
              resettle(() => setPlaceId(id), sight(doy, p));
            }}
          />
          <Slider label="date" value={doy} min={1} max={365} step={1} onChange={(d) => resettle(() => setDoy(d), sight(d, place))} format={dateLabel} />
        </>
      }
      onSkip={
        s.visible
          ? () => {
              setT(12);
              setReading(exact);
            }
          : undefined
      }
      equation={
        <Working label="Latitude from the noon sight">
          <Eq>
            <Sym role="sky">h</Sym> = <Sym>H</Sym><sub>s</sub> + dip + semi-diameter + refraction
          </Eq>
          <Eq>
            <Known when={hDone} role="sky">{dm(r.h)}</Known> = <Known when={readDone}>{dm(reading)}</Known>{' '}
            <Known when={hDone} role="soft">{am(r.dip)}</Known> <Known when={hDone} role="soft">{am(r.semi)}</Known>{' '}
            <Known when={hDone} role="soft">{am(r.refr)}</Known>
          </Eq>
          <Eq>
            {south ? (
              <>
                <Sym role="red">φ</Sym> = 90° − <Sym role="sky">h</Sym> + <Sym role="brass">δ</Sym>
              </>
            ) : (
              <>
                <Sym role="red">φ</Sym> = <Sym role="sky">h</Sym> − 90° + <Sym role="brass">δ</Sym>
              </>
            )}
          </Eq>
          <Eq>
            <Known when={phiDone} role="red">{dm(r.phi, ['N', 'S'])}</Known> ={' '}
            {south ? (
              <>
                90° − <Known when={hDone} role="sky">{dm(r.h)}</Known>
              </>
            ) : (
              <>
                <Known when={hDone} role="sky">{dm(r.h)}</Known> − 90°
              </>
            )}{' '}
            + (<Known when={decDone} role="brass">{dm(s.dec)}</Known>)
          </Eq>
          {phiDone && (
            <Eq>
              <span style={{ fontSize: '0.88em' }}>
                {`True ${place.name}: ${dm(place.lat, ['N', 'S'])}. One arcminute of latitude is one nautical mile (1.852 km).`}
              </span>
            </Eq>
          )}
        </Working>
      }
      caption={
        <>
          The left view is what the navigator sees at each step; the right one is the geometry behind it and does
          not change: the sun’s rays arrive parallel at its declination δ, you measure its height h above your
          horizon, and the angle at the Earth’s centre is your latitude φ. Declination: USNO approximate formula;
          dip 1.76′√(eye height in m) (Bowditch); refraction: Bennett’s formula at 10 °C.
        </>
      }
    />
  );
}

