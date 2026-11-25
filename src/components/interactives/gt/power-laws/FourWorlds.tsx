import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Slider,
  Specimen,
  T,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import { rng } from './model';

/**
 * Four parallel worlds with the same 48 equally good songs (after MusicLab, Salganik, Dodds
 * & Watts 2006). Each listener downloads one song: with probability p they pick uniformly,
 * otherwise they copy a random earlier download (choose in proportion to current downloads).
 */
const SONGS = 48;
const LISTENERS = 1200;
const WORLDS = 4;
const PLAY_SECONDS = 4.5;

function runWorld(p: number, seed: number): Uint8Array {
  const r = rng(seed);
  const picks = new Uint8Array(LISTENERS);
  for (let t = 0; t < LISTENERS; t++) {
    picks[t] = t === 0 || r() < p ? Math.floor(r() * SONGS) : picks[Math.floor(r() * t)];
  }
  return picks;
}

function countsAt(picks: Uint8Array, t: number) {
  const c = new Array(SONGS).fill(0);
  for (let i = 0; i < t; i++) c[picks[i]]++;
  return c;
}

const leaderOf = (c: number[]) => c.reduce((b, v, i) => (v > c[b] ? i : b), 0);

const PAD_B = 14;

function PanelSvg({
  idx,
  counts,
  yMax,
  t,
  narrow,
}: {
  idx: number;
  counts: number[];
  yMax: number;
  t: number;
  narrow: boolean;
}) {
  // phones get a smaller viewBox so the labels stay legible at half width
  const W = narrow ? 170 : 300;
  const H = narrow ? 100 : 120;
  const BAR = W / SONGS;
  const lead = leaderOf(counts);
  const share = t > 0 ? counts[lead] / t : 0;
  const plotH = H - PAD_B - 16;
  const y = (v: number) => H - PAD_B - (v / yMax) * plotH;
  const lx = lead * BAR + BAR / 2;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="block w-full"
      role="img"
      aria-label={`World ${idx + 1}: song ${lead + 1} leads with ${Math.round(share * 100)} percent`}
    >
      <line x1={0} x2={W} y1={H - PAD_B} y2={H - PAD_B} style={{ stroke: GT.lineStrong }} />
      {/* fair share line: every song equal */}
      <line
        x1={0}
        x2={W}
        y1={y(t / SONGS)}
        y2={y(t / SONGS)}
        strokeDasharray="3 3"
        style={{ stroke: GT.a, opacity: t > 0 ? 0.8 : 0 }}
      />
      {counts.map((v, i) => (
        <rect
          key={i}
          x={i * BAR + 0.6}
          y={y(v)}
          width={BAR - 1.2}
          height={Math.max(0, H - PAD_B - y(v))}
          style={{
            fill: i === lead && t > 0 ? GT.eq : GT.inkSoft,
            opacity: i === lead ? 1 : 0.55,
          }}
        />
      ))}
      {t > 0 && (
        <T
          x={Math.min(W - 14, Math.max(14, lx))}
          y={Math.max(8, y(counts[lead]) - 8)}
          size={10}
          color={GT.eq}
          weight={600}
        >
          {Math.round(share * 100)}%
        </T>
      )}
      {idx === 0 && t > 0 && (
        <text
          x={W * 0.5}
          y={y(t / SONGS) - 6}
          textAnchor="middle"
          style={{
            fill: GT.a,
            fontFamily: GT.mono,
            fontSize: narrow ? 8 : 9,
            stroke: GT.paperRaised,
            strokeWidth: 3,
            paintOrder: 'stroke',
          }}
        >
          fair share
        </text>
      )}
      <T x={2} y={H - 5} anchor="start" size={8}>
        song 1
      </T>
      <T x={W - 2} y={H - 5} anchor="end" size={8}>
        song 48
      </T>
    </svg>
  );
}

function WorldPanel(props: { idx: number; counts: number[]; yMax: number; t: number }) {
  const lead = leaderOf(props.counts);
  return (
    <div className="min-w-0 border border-line bg-paper-raised">
      <div className="flex items-baseline justify-between gap-2 border-b border-line px-2.5 py-1.5">
        <span className="font-mono text-[0.62rem] tracking-[0.1em] text-muted">
          <span className="hidden sm:inline">WORLD </span>
          {String.fromCharCode(65 + props.idx)}
        </span>
        <span className="truncate font-mono text-[0.66rem] text-ink-soft">
          hit: <span style={{ color: GT.eq }}>song {lead + 1}</span>
        </span>
      </div>
      {/* two drawings, chosen by CSS, so phones never re-layout after hydration */}
      <div className="hidden sm:block">
        <PanelSvg {...props} narrow={false} />
      </div>
      <div className="sm:hidden">
        <PanelSvg {...props} narrow={true} />
      </div>
    </div>
  );
}

export default function FourWorlds() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView<HTMLDivElement>();
  const [p, setP] = useState(0.2);
  const [history, setHistory] = useState(0);
  const [t, setT] = useState(LISTENERS);
  const tRef = useRef(LISTENERS);
  tRef.current = t;
  const [playing, setPlaying] = useState(false);
  const started = useRef(false);

  const worlds = useMemo(
    () => Array.from({ length: WORLDS }, (_, w) => runWorld(p, 1000 * (history + 1) + w * 7 + 3)),
    [p, history],
  );
  const finals = useMemo(() => worlds.map((pk) => countsAt(pk, LISTENERS)), [worlds]);
  const yMax = Math.max(...finals.map((c) => Math.max(...c))) * 1.12;
  const counts = useMemo(
    () => (t >= LISTENERS ? finals : worlds.map((pk) => countsAt(pk, Math.floor(t)))),
    [worlds, finals, t],
  );

  const replay = () => {
    if (reduced) {
      setT(LISTENERS);
      return;
    }
    setT(0);
    setPlaying(true);
  };

  // first time on screen: play the demo once
  useEffect(() => {
    if (started.current || !inView || reduced) return;
    started.current = true;
    replay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, reduced]);

  useFrame(
    (dt) => {
      const nx = Math.min(LISTENERS, tRef.current + (LISTENERS / PLAY_SECONDS) * dt);
      setT(nx);
      if (nx >= LISTENERS) setPlaying(false);
    },
    playing && inView && !reduced,
  );

  const tt = Math.floor(t);
  const leaders = counts.map(leaderOf);
  const distinct = new Set(leaders).size;
  const bigShare = tt > 0 ? Math.max(...counts.map((c) => Math.max(...c))) / tt : 0;

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-10.1"
      title="four worlds"
      kicker={`${tt} / ${LISTENERS} listeners`}
      hint="press replay history: same songs, same rules, new luck"
    >
      <div className="grid grid-cols-2 gap-2 sm:gap-3">
        {counts.map((c, i) => (
          <WorldPanel key={i} idx={i} counts={c} yMax={yMax} t={tt} />
        ))}
      </div>
      <div className="mt-4 grid items-end gap-x-8 gap-y-4 border-t border-line pt-4 sm:grid-cols-[auto_minmax(0,1fr)]">
        <div className="flex gap-8">
          <BigStat
            label="biggest hit"
            value={`${Math.round(bigShare * 100)}%`}
            color={GT.eq}
            sub={`fair share ${(100 / SONGS).toFixed(1)}%`}
          />
          <BigStat label="different #1s" value={`${distinct} of 4`} sub="same songs, new luck" />
        </div>
        <div className="grid min-w-0 items-end justify-items-start gap-3 sm:grid-cols-[minmax(0,1fr)_auto]">
          <Slider
            label="listeners who ignore the chart"
            value={p}
            min={0.05}
            max={1}
            step={0.05}
            onChange={(v) => {
              setP(v);
              setPlaying(false);
              setT(LISTENERS);
            }}
            format={(v) => `${Math.round(v * 100)}%`}
          />
          <Btn
            primary
            onClick={() => {
              setHistory((h) => h + 1);
              replay();
            }}
          >
            replay history
          </Btn>
        </div>
      </div>
      <Caption>
        48 equally good songs. Each listener picks at random or copies an earlier download, so early
        luck snowballs.
      </Caption>
    </Specimen>
  );
}
