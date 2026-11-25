import { useEffect, useRef, useState } from 'react';
import { BigStat, Caption, GT, Slider, Specimen, T, useInView, useReducedMotion } from '../shared';
import { pct, prefersReduced, useSharedT } from './sharedT';

// One resident (blue, in the middle) and the 8 cells around her. She counts the occupied
// neighbours and how many are her own kind. Click a neighbour to change it.

type Cell = 0 | 1 | 2; // empty, blue, red
const ORDER = [0, 1, 2, 3, 5, 6, 7, 8]; // grid slots of the 8 neighbours (4 is her)
// SSR / poster: outnumbered 5 to 3 and still content
const POSTER: Cell[] = [2, 1, 2, 2, 1, 2, 1, 2];
// auto-play: start among her own kind, then neighbours turn red one at a time
const DEMO: Cell[][] = [
  [1, 1, 1, 1, 1, 1, 1, 1],
  [2, 1, 1, 1, 1, 1, 1, 1],
  [2, 1, 2, 1, 1, 1, 1, 1],
  [2, 1, 2, 2, 1, 1, 1, 1],
  [2, 1, 2, 2, 1, 2, 1, 1],
  [2, 1, 2, 2, 1, 2, 1, 2],
  [2, 2, 2, 2, 1, 2, 1, 2],
  POSTER,
];

const S = 76;
const G = 6;
const fill = (c: Cell) => (c === 1 ? GT.a : c === 2 ? GT.b : 'transparent');

export default function OneResident() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [t, setT] = useSharedT();
  const [cells, setCells] = useState<Cell[]>(POSTER);
  const [demo, setDemo] = useState(-1);
  const started = useRef(false);

  useEffect(() => {
    if (reduced || prefersReduced() || started.current || !inView) return;
    started.current = true;
    setDemo(0);
  }, [reduced, inView]);

  useEffect(() => {
    if (demo < 0) return;
    setCells(DEMO[demo]);
    if (demo >= DEMO.length - 1) {
      setDemo(-1);
      return;
    }
    const id = setTimeout(() => setDemo((d) => (d < 0 ? d : d + 1)), demo === 0 ? 1100 : 900);
    return () => clearTimeout(id);
  }, [demo]);

  const cycle = (k: number) => {
    setDemo(-1);
    setCells((cs) => cs.map((c, i) => (i === k ? (((c + 1) % 3) as Cell) : c)) as Cell[]);
  };

  const alike = cells.filter((c) => c === 1).length;
  const occ = cells.filter((c) => c !== 0).length;
  const share = occ ? alike / occ : 1;
  const need = Math.ceil(t * 8 - 1e-9);
  const ok = occ === 0 || share >= t - 1e-9;

  return (
    <Specimen
      code="GT-09.1"
      title="one resident"
      kicker="NCM 4.5 · 8 neighbours"
      rootRef={rootRef}
      hint="click a neighbour to change it: blue, red, empty"
      footer={`needs ≥ ${pct(t)} alike`}
    >
      <div className="grid items-center gap-5 sm:grid-cols-[minmax(0,260px)_1fr]">
        <svg
          viewBox={`0 0 ${3 * S + 2 * G} ${3 * S + 2 * G}`}
          className="mx-auto block w-full max-w-[260px]"
          role="img"
          aria-label={`A resident with ${alike} alike neighbours out of ${occ}`}
        >
          {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((slot) => {
            const x = (slot % 3) * (S + G);
            const y = Math.floor(slot / 3) * (S + G);
            if (slot === 4) {
              return (
                <g key={slot}>
                  <rect x={x} y={y} width={S} height={S} style={{ fill: GT.a }} />
                  <rect
                    x={x - 3}
                    y={y - 3}
                    width={S + 6}
                    height={S + 6}
                    style={{ fill: 'none', stroke: GT.ink, strokeWidth: 2 }}
                  />
                  <T x={x + S / 2} y={y + S / 2} size={13} color={GT.paper} weight={600}>
                    her
                  </T>
                </g>
              );
            }
            const k = ORDER.indexOf(slot);
            const c = cells[k];
            return (
              <g
                key={slot}
                role="button"
                tabIndex={0}
                aria-label={`neighbour ${k + 1}: ${c === 1 ? 'blue' : c === 2 ? 'red' : 'empty'}`}
                onClick={() => cycle(k)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    cycle(k);
                  }
                }}
                style={{ cursor: 'pointer' }}
              >
                <rect
                  x={x + 0.5}
                  y={y + 0.5}
                  width={S - 1}
                  height={S - 1}
                  style={{
                    fill: c === 0 ? GT.paperRaised : fill(c),
                    stroke: c === 0 ? GT.lineStrong : 'none',
                    strokeDasharray: '4 4',
                    transition: 'fill 250ms',
                  }}
                />
                {c === 1 && (
                  <T x={x + S / 2} y={y + S / 2} size={20} color={GT.paper} font="display">
                    ✓
                  </T>
                )}
                {c === 0 && (
                  <T x={x + S / 2} y={y + S / 2} size={11}>
                    empty
                  </T>
                )}
              </g>
            );
          })}
        </svg>

        <div className="min-w-0">
          <BigStat
            label="she"
            value={ok ? 'stays' : 'moves'}
            size="lg"
            color={ok ? GT.opt : GT.warn}
            sub={
              <span className="tabular-nums">
                <span style={{ color: GT.a }}>{alike}</span> of {occ} occupied alike = {pct(share)}{' '}
                {ok ? '≥' : '<'} {pct(t)}
              </span>
            }
          />

          {/* the bar she needs, drawn on the same track as her share */}
          <div className="mt-5">
            <div className="relative h-3 border border-line-strong">
              <div
                className="absolute inset-y-0 left-0 transition-all duration-300"
                style={{ width: `${share * 100}%`, background: ok ? GT.opt : GT.warn }}
              />
              <div
                className="absolute -inset-y-1.5 w-0.5"
                style={{ left: `calc(${t * 100}% - 1px)`, background: GT.ink }}
              />
            </div>
            <div className="mt-1 flex justify-between font-mono text-[0.6rem] text-muted">
              <span>0%</span>
              <span>her alike share · tick = her bar</span>
              <span>100%</span>
            </div>
          </div>

          <div className="mt-4">
            <Slider
              label="her bar t"
              value={t}
              min={0}
              max={0.9}
              step={0.01}
              onChange={setT}
              format={pct}
            />
          </div>
        </div>
      </div>
      <Caption>
        Empty lots do not count. With eight full neighbours, a bar of {pct(t)} needs {need} of them
        alike.
      </Caption>
    </Specimen>
  );
}
