import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  Chip,
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

const W = 300;
const H = 120;
const PAD_B = 14;
const BAR = W / SONGS;

function WorldPanel({
  idx,
  counts,
  yMax,
  t,
}: {
  idx: number;
  counts: number[];
  yMax: number;
  t: number;
}) {
  const lead = leaderOf(counts);
  const share = t > 0 ? counts[lead] / t : 0;
  const plotH = H - PAD_B - 16;
  const y = (v: number) => H - PAD_B - (v / yMax) * plotH;
  const lx = lead * BAR + BAR / 2;
  return (
    <div className="min-w-0 border border-line bg-paper-raised">
      <div className="flex items-baseline justify-between gap-2 border-b border-line px-2.5 py-1.5">
        <span className="font-mono text-[0.62rem] tracking-[0.1em] text-muted">
          WORLD {String.fromCharCode(65 + idx)}
        </span>
        <span className="font-mono text-[0.66rem] text-ink-soft">
          hit: <span style={{ color: GT.eq }}>song {lead + 1}</span>
        </span>
      </div>
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
            x={Math.min(W - 22, Math.max(22, lx))}
            y={Math.max(8, y(counts[lead]) - 8)}
            size={10}
            color={GT.eq}
            weight={600}
          >
            {Math.round(share * 100)}%
          </T>
        )}
        <T x={2} y={H - 5} anchor="start" size={8}>
          song 1
        </T>
        <T x={W - 2} y={H - 5} anchor="end" size={8}>
          song 48
        </T>
      </svg>
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
      title="four worlds, same songs"
      kicker={`${tt} / ${LISTENERS} listeners per world`}
      hint="press replay history: same songs, same rules, new luck"
      footer={<Chip color={GT.eq}>the hit in each world</Chip>}
    >
      <div className="mb-4 grid grid-cols-3 gap-3">
        <BigStat
          label="fair share"
          value={`${(100 / SONGS).toFixed(1)}%`}
          color={GT.a}
          sub="each of 48 songs"
        />
        <BigStat
          label="biggest hit"
          value={`${Math.round(bigShare * 100)}%`}
          color={GT.eq}
          sub="of one world's downloads"
        />
        <BigStat label="different #1s" value={`${distinct} of 4`} sub="worlds" />
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {counts.map((c, i) => (
          <WorldPanel key={i} idx={i} counts={c} yMax={yMax} t={tt} />
        ))}
      </div>
      <div className="mt-4 grid grid-cols-1 items-end gap-3 sm:grid-cols-[1fr_auto]">
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
        <div className="flex gap-2">
          <Btn
            primary
            onClick={() => {
              setHistory((h) => h + 1);
              replay();
            }}
          >
            replay history
          </Btn>
          <Btn onClick={replay}>play</Btn>
        </div>
      </div>
      <Caption>
        All 48 songs are equally good. A listener either picks at random or copies someone who came
        before. Copying turns small early leads into hits, and which song gets the lead is luck.
      </Caption>
    </Specimen>
  );
}
