/**
 * CX-14.1 "the searcher that can't say yes". Wang's procedure on three tile sets: try to fill
 * a 1 x 1 square, then 2 x 2, 3 x 3, ... with matching edges; answer "no" as soon as some
 * square cannot be filled, "yes" as soon as a block that wraps around onto itself turns up
 * (it can be repeated forever). One clock grows the square ring by ring and the view zooms
 * out with it. The repeating set says yes at 4 x 4, the stuck set says no at 6 x 6, and the
 * Jeandel-Rao set keeps going: every square fills, no block ever repeats.
 */
import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  GT,
  Seg,
  Specimen,
  cssVar,
  useFrame,
  useHoverPreview,
  useInView,
  useReducedMotion,
  useThemeVersion,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { OUTCOMES, type SetId } from './sets';
import type { Tile } from './wang';

const TABS: { value: SetId; label: string }[] = [
  { value: 'repeats', label: 'repeats' },
  { value: 'stuck', label: 'gets stuck' },
  { value: 'jr', label: 'Jeandel-Rao 11' },
];
const CAPTION: Record<SetId, string> = {
  repeats:
    'Six tiles. At 4 by 4 the searcher finds a block whose right edge fits its left and whose bottom fits its top, so copies of it cover the plane. Answer: yes.',
  stuck:
    'Five tiles. Squares up to 5 by 5 fill, but no way of placing them fills 6 by 6 (the search tries 3,000 placements). If a square cannot be filled, neither can the plane. Answer: no.',
  jr: 'The 11 tiles of Jeandel and Rao (2015). Every square fills and no block ever repeats, so neither answer ever comes. The searcher runs forever.',
};

// edge colours 0..3 -> base tokens (canvas reads base names)
const EDGE_VARS = ['--gt-a', '--gt-b', '--gt-opt', '--gt-warn'];
const EDGE_CSS = [GT.a, GT.b, GT.opt, GT.warn];

// growth law: dn/dt = A + B n (brisk at the start, faster as the squares get big)
const GA = 1.6;
const GB = 0.22;
const nAt = (t: number) => (GA / GB) * (Math.exp(GB * Math.max(0, t)) - 1);
const tAt = (n: number) => Math.log(1 + (n * GB) / GA) / GB;
const tEnd = (id: SetId) => tAt(OUTCOMES[id].maxN);

const fmt = (n: number) => n.toLocaleString('en-US');

export default function Searcher() {
  const reduced = useReducedMotion();
  const theme = useThemeVersion();
  const [rootRef, inView] = useInView();
  const [sel, setSel] = useState<SetId>('repeats');
  const [tab, bindTab] = useHoverPreview(sel);
  const ends = { repeats: tEnd('repeats'), stuck: tEnd('stuck'), jr: tEnd('jr') };
  // one clock per tab; tabs not played yet show their finished poster when previewed
  const [clock, setClock] = useState<Record<SetId, number>>({
    repeats: 0,
    ...{ stuck: ends.stuck, jr: ends.jr },
  });
  const [wait, setWait] = useState(0);
  const started = useRef(false);

  useEffect(() => {
    if (reduced) setClock(ends);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced]);
  useEffect(() => {
    if (inView && !started.current && !reduced) {
      started.current = true;
      setWait(0.35);
    }
  }, [inView, reduced]);

  const running = started.current && !reduced && inView && clock[sel] < ends[sel];
  useFrame((dt) => {
    if (wait > 0) return setWait((w) => Math.max(0, w - dt));
    setClock((c) => ({ ...c, [sel]: Math.min(ends[sel], c[sel] + dt) }));
  }, running);

  const replay = (id: SetId, pause: number) => {
    if (reduced) return setClock((c) => ({ ...c, [id]: ends[id] }));
    setClock((c) => ({ ...c, [id]: 0 }));
    setWait(pause);
  };
  const choose = (id: SetId) => {
    if (id === sel) return;
    setSel(id);
    replay(id, 0.4);
  };

  const o = OUTCOMES[tab];
  const nc = Math.min(o.maxN, nAt(clock[tab]));
  const trying = Math.max(1, Math.min(o.maxN, Math.ceil(nc - 1e-9)));
  const decided = nc >= o.decideAt - 1e-9;
  const verdict = decided ? o.verdict : 'searching';

  /* ───────── canvas ───────── */
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const [cw, setCw] = useState(0);
  useEffect(() => {
    const el = boxRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const obs = new ResizeObserver(([e]) => setCw(e.contentRect.width));
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || cw <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    const W = cw;
    if (cv.width !== Math.round(W * dpr)) {
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(W * dpr);
    }
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, W);
    const col = EDGE_VARS.map(cssVar);
    const paper = cssVar('--paper-raised');
    const ink = cssVar('--ink');
    const red = cssVar('--gt-b');
    const accent = cssVar('--accent');
    const pad = 4;
    const V = Math.max(5, nc + 1);
    const s = (W - 2 * pad) / V;
    const ox = pad;
    const oy = pad;

    const drawTile = (t: Tile, x: number, y: number, a: number) => {
      ctx.globalAlpha = a;
      const cx = x + s / 2;
      const cy = y + s / 2;
      const tri = (c: number, x1: number, y1: number, x2: number, y2: number) => {
        ctx.fillStyle = col[c];
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.closePath();
        ctx.fill();
      };
      // overdraw a hair so neighbouring triangles leave no seams
      const e = 0.25;
      tri(t[0], x - e, y - e, x + s + e, y - e);
      tri(t[1], x + s + e, y - e, x + s + e, y + s + e);
      tri(t[2], x + s + e, y + s + e, x - e, y + s + e);
      tri(t[3], x - e, y + s + e, x - e, y - e);
      if (s > 9) {
        ctx.strokeStyle = paper;
        ctx.lineWidth = Math.min(1.4, s * 0.05);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + s, y + s);
        ctx.moveTo(x + s, y);
        ctx.lineTo(x, y + s);
        ctx.stroke();
        ctx.lineWidth = Math.min(2, s * 0.07);
        ctx.strokeRect(x, y, s, s);
      }
      ctx.globalAlpha = 1;
    };

    // reveal ring m (cells with max(r, c) = m) while nc runs from m to m + 1,
    // down the right column and then leftwards along the bottom row
    const rings = Math.min(o.maxN, Math.ceil(nc));
    for (let m = 0; m < rings; m++) {
      const len = 2 * m + 1;
      for (let k = 0; k < len; k++) {
        const r = k <= m ? k : m;
        const c = k <= m ? m : 2 * m - k;
        // cells appear back to back, each fading in over its share of the ring
        const a = Math.max(0, Math.min(1, (nc - m) * len - k));
        if (a <= 0) continue;
        const x = ox + c * s;
        const y = oy + r * s;
        if (r < o.patch.length && c < o.patch[0].length) {
          drawTile(o.tiles[o.patch[r][c]], x, y, a);
        } else {
          // a cell the searcher cannot fill
          ctx.globalAlpha = a;
          ctx.setLineDash([3, 3]);
          ctx.strokeStyle = red;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x + 2, y + 2, s - 4, s - 4);
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
        }
      }
    }

    if (tab === 'repeats' && o.block) {
      const b = o.block;
      const a = Math.max(0, Math.min(1, (nc - b) / 0.4));
      if (a > 0) {
        // faint seams where the copies meet, then the block itself in accent
        ctx.globalAlpha = a * 0.9;
        ctx.strokeStyle = ink;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        const reach = Math.min(o.maxN, nc);
        ctx.beginPath();
        for (let k = b; k < reach; k += b) {
          ctx.moveTo(ox + k * s, oy);
          ctx.lineTo(ox + k * s, oy + reach * s);
          ctx.moveTo(ox, oy + k * s);
          ctx.lineTo(ox + reach * s, oy + k * s);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = a;
        ctx.strokeStyle = accent;
        ctx.lineWidth = 3;
        ctx.strokeRect(ox + 1.5, oy + 1.5, b * s - 3, b * s - 3);
        ctx.globalAlpha = 1;
      }
    }
    if (tab === 'stuck') {
      const a = Math.max(0, Math.min(1, (nc - (o.decideAt - 1)) / 1));
      if (a > 0.99) {
        ctx.strokeStyle = red;
        ctx.lineWidth = 2.5;
        ctx.strokeRect(ox + 1.25, oy + 1.25, o.decideAt * s - 2.5, o.decideAt * s - 2.5);
      }
    }
  }, [cw, nc, tab, theme, o]);

  const stat =
    verdict === 'yes'
      ? { v: 'yes', c: GT.opt, sub: `a ${o.block} by ${o.block} block repeats` }
      : verdict === 'no'
        ? {
            v: 'no',
            c: GT.b,
            sub: `no ${o.decideAt} by ${o.decideAt} square: ${fmt(o.tries ?? 0)} tries`,
          }
        : { v: 'none yet', c: GT.ink, sub: `no repeat block up to ${trying} by ${trying}` };

  return (
    <Specimen
      code="CX-14.1"
      title="the searcher that can't say yes"
      kicker="Wang 1961"
      rootRef={rootRef}
      hint="pick a tile set"
      footer={<Btn onClick={() => replay(sel, 1)}>restart</Btn>}
    >
      <div className="grid items-start gap-x-8 gap-y-4 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <div ref={boxRef} className="mx-auto w-full max-w-[460px]">
          <canvas
            ref={canvasRef}
            className="block aspect-square w-full"
            role="img"
            aria-label={`Wang tiles filling a ${trying} by ${trying} square. Answer so far: ${stat.v}.`}
          />
        </div>
        <div className="grid gap-5">
          <Seg options={TABS} value={tab} onChange={choose} bind={bindTab} />
          <div className="grid grid-cols-2 gap-4">
            <BigStat
              label="square"
              value={`${trying}×${trying}`}
              sub={tab === 'stuck' && verdict === 'no' ? 'cannot be filled' : 'filled'}
            />
            <BigStat label="answer" value={stat.v} color={stat.c} sub={stat.sub} />
          </div>
          <TileRow tiles={o.tiles} />
          <Reserve
            items={TABS.map((t) => (
              <p key={t.value} className="text-[0.85rem] leading-snug text-ink-soft">
                {CAPTION[t.value]}
              </p>
            ))}
            active={TABS.findIndex((t) => t.value === tab)}
          />
        </div>
      </div>
    </Specimen>
  );
}

/** The tile set itself, as classic Wang squares (fixed 11 slots so the row never resizes). */
function TileRow({ tiles }: { tiles: Tile[] }) {
  const S = 24;
  const G = 4;
  const W = 11 * S + 10 * G;
  return (
    <div>
      <div className="label mb-1.5">the tiles · {tiles.length} · no turning</div>
      <svg viewBox={`0 0 ${W} ${S}`} className="block w-full max-w-[340px]" aria-hidden="true">
        {tiles.map((t, i) => {
          const x = i * (S + G);
          const c = { x: x + S / 2, y: S / 2 };
          const P = (a: [number, number], b: [number, number]) =>
            `M${c.x} ${c.y} L${a[0]} ${a[1]} L${b[0]} ${b[1]} Z`;
          const corners: [number, number][] = [
            [x, 0],
            [x + S, 0],
            [x + S, S],
            [x, S],
          ];
          return (
            <g key={i}>
              {[0, 1, 2, 3].map((k) => (
                <path
                  key={k}
                  d={P(corners[k], corners[(k + 1) % 4])}
                  style={{
                    fill: EDGE_CSS[t[k]],
                    stroke: GT.paperRaised,
                    strokeWidth: 1,
                    strokeLinejoin: 'round',
                  }}
                />
              ))}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
