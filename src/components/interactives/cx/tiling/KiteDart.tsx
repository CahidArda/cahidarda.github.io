/**
 * CX-14.2 "kites and darts". A Penrose tiling grown outward from a sun of five kites (the
 * patch is five deflations of that sun). Then a copy of the whole patch, drawn as outlines,
 * slides to three shifts that line up well (the best is the best of every shift tried). Matching
 * tiles turn green, the rest dim; the share
 * never reaches 100%, because no shift carries a Penrose tiling onto itself.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  GT,
  Specimen,
  cssVar,
  useFrame,
  useInView,
  useReducedMotion,
  useThemeVersion,
  useTween,
} from '../../gt/shared';
import { Reserve } from '../../gt/Reserve';
import { bestShifts, overlap, penrose, type Tile } from './penrose';

const HALF = 0.52; // half-width of the view in tiling units
const GROW = 2.2; // seconds to grow
const R_END = HALF * Math.SQRT2 + 0.06;
const HOLD = 1.1;
const GLIDE = 0.9;

const smooth = (x: number) => {
  const t = Math.max(0, Math.min(1, x));
  return t * t * (3 - 2 * t);
};

type Shift = { dx: number; dy: number; share: number };

/** Copy offset at clock u (seconds since the copy appeared): glide to each shift and hold. */
function copyAt(u: number, shifts: Shift[]) {
  let from = { dx: 0, dy: 0 };
  let t = u;
  for (let i = 0; i < shifts.length; i++) {
    const to = shifts[i];
    if (t < GLIDE) {
      const e = smooth(t / GLIDE);
      return {
        dx: from.dx + (to.dx - from.dx) * e,
        dy: from.dy + (to.dy - from.dy) * e,
        at: i,
        settle: 0,
      };
    }
    t -= GLIDE;
    if (t < HOLD || i === shifts.length - 1) {
      return { dx: to.dx, dy: to.dy, at: i, settle: Math.min(1, t / 0.25) };
    }
    t -= HOLD;
    from = to;
  }
  return { dx: 0, dy: 0, at: -1, settle: 0 };
}
const slideEnd = (n: number) => n * GLIDE + (n - 1) * HOLD + 0.3;

export default function KiteDart() {
  const reduced = useReducedMotion();
  const theme = useThemeVersion();
  const [rootRef, inView] = useInView();
  const tiles = useMemo<Tile[]>(() => penrose(5, 1), []);
  const view = useMemo(
    () => tiles.filter((t) => Math.abs(t.cx) < HALF && Math.abs(t.cy) < HALF),
    [tiles],
  );
  const kites = view.filter((t) => t.kind === 'kite').length;
  const darts = view.length - kites;
  // three of the best shifts, in increasing order of fit, in three different directions
  const shifts = useMemo<Shift[]>(() => {
    const best = bestShifts(tiles, HALF, 0.6);
    const pick: Shift[] = [];
    for (const b of best) {
      const a = Math.atan2(b.dy, b.dx);
      const apart = pick.every(
        (p) => Math.abs(Math.atan2(p.dy, p.dx) - a) > 0.5 && Math.abs(p.share - b.share) > 0.06,
      );
      if (apart) pick.push(b);
      if (pick.length === 3) break;
    }
    return pick.sort((p, q) => p.share - q.share);
  }, [tiles]);

  const [t, setT] = useState(GROW); // growth clock (SSR: grown)
  const [copy, setCopy] = useState(false);
  const [u, setU] = useState(0); // slide clock
  const started = useRef(false);
  const [wait, setWait] = useState(0);
  const END = slideEnd(3);

  useEffect(() => {
    if (reduced) {
      setT(GROW);
      setCopy(true);
      setU(END);
    }
  }, [reduced, END]);
  useEffect(() => {
    if (inView && !started.current && !reduced) {
      started.current = true;
      setT(0);
      setWait(0.3);
    }
  }, [inView, reduced]);

  const running = !reduced && inView && started.current && (t < GROW + 0.2 || (copy && u < END));
  useFrame((dt) => {
    if (wait > 0) return setWait((w) => Math.max(0, w - dt));
    if (t < GROW + 0.2) {
      const nt = t + dt;
      setT(nt);
      if (nt >= GROW + 0.2 && !copy) {
        setCopy(true);
        setU(0);
      }
      return;
    }
    if (copy) setU((x) => Math.min(END, x + dt));
  }, running);

  const restart = () => {
    setCopy(false);
    setU(0);
    if (reduced) {
      setCopy(true);
      setU(END);
      return;
    }
    setT(0);
    setWait(1);
  };
  const toggle = () => {
    if (copy) {
      setCopy(false);
      setU(0);
    } else {
      setCopy(true);
      setU(reduced ? END : 0);
      setT(GROW + 0.2);
    }
  };

  const pos = copy && shifts.length ? copyAt(u, shifts) : null;
  const target = pos && pos.at >= 0 ? shifts[pos.at] : null;
  const match = useMemo(
    () => (target ? overlap(tiles, target.dx, target.dy, HALF).match : null),
    [target, tiles],
  );
  const shareNow = pos && target ? target.share * pos.settle : 0;
  const shareShown = useTween(shareNow * 100, reduced, 12);

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
    const k = W / (2 * HALF);
    const X = (x: number) => W / 2 + x * k;
    const Y = (y: number) => W / 2 + y * k;
    const kiteFill = cssVar('--gt-a-soft');
    const dartFill = cssVar('--gt-b-soft');
    const kiteEdge = cssVar('--gt-a');
    const dartEdge = cssVar('--gt-b');
    const green = cssVar('--gt-opt');
    const ink = cssVar('--ink');
    const paper = cssVar('--paper-raised');
    const r = smooth(t / GROW) * R_END;
    const band = 0.08;

    const path = (tt: Tile, dx = 0, dy = 0) => {
      ctx.beginPath();
      tt.pts.forEach((p, i) =>
        i ? ctx.lineTo(X(p.x + dx), Y(p.y + dy)) : ctx.moveTo(X(p.x + dx), Y(p.y + dy)),
      );
      ctx.closePath();
    };
    ctx.lineJoin = 'round';
    for (const tt of tiles) {
      if (Math.abs(tt.cx) > HALF + 0.1 || Math.abs(tt.cy) > HALF + 0.1) continue;
      const d = Math.hypot(tt.cx, tt.cy);
      const a = Math.max(0, Math.min(1, (r - d) / band));
      if (a <= 0) continue;
      ctx.globalAlpha = a;
      path(tt);
      ctx.fillStyle = tt.kind === 'kite' ? kiteFill : dartFill;
      ctx.fill();
      ctx.strokeStyle = tt.kind === 'kite' ? kiteEdge : dartEdge;
      ctx.lineWidth = 1.1;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    if (pos) {
      // where the copy lines up: dim everything, then fill the matches green
      if (match && pos.settle > 0) {
        ctx.globalAlpha = 0.62 * pos.settle;
        ctx.fillStyle = paper;
        ctx.fillRect(0, 0, W, W);
        ctx.globalAlpha = 0.8 * pos.settle;
        ctx.fillStyle = green;
        tiles.forEach((tt, i) => {
          if (!match[i]) return;
          path(tt, target!.dx, target!.dy);
          ctx.fill();
        });
        ctx.globalAlpha = 1;
      }
      // the copy, outlines only
      const appear = Math.min(1, u / 0.3);
      ctx.globalAlpha = 0.6 * appear;
      ctx.strokeStyle = ink;
      ctx.lineWidth = 1;
      for (const tt of tiles) {
        const x = tt.cx + pos.dx;
        const y = tt.cy + pos.dy;
        if (Math.abs(x) > HALF + 0.1 || Math.abs(y) > HALF + 0.1) continue;
        path(tt, pos.dx, pos.dy);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }, [cw, t, u, copy, theme, tiles, pos?.dx, pos?.dy, pos?.settle, match, target]);

  const best = shifts.length ? Math.round(Math.max(...shifts.map((s) => s.share)) * 100) : 0;

  return (
    <Specimen
      code="CX-14.2"
      title="kites and darts"
      kicker="Penrose 1974"
      rootRef={rootRef}
      hint="outlines: a copy of the same tiling, shifted"
      footer={
        <span className="flex gap-2">
          <Btn onClick={toggle}>
            <Reserve items={['remove the copy', 'slide a copy']} active={copy ? 0 : 1} />
          </Btn>
          <Btn onClick={restart}>restart</Btn>
        </span>
      }
    >
      <div className="grid items-start gap-x-8 gap-y-4 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <div ref={boxRef} className="mx-auto w-full max-w-[460px]">
          <canvas
            ref={canvasRef}
            className="block aspect-square w-full"
            role="img"
            aria-label={`A Penrose tiling of ${kites} kites and ${darts} darts.${copy ? ` A shifted copy lines up with ${Math.round(shareShown)}% of the tiles.` : ''}`}
          />
        </div>
        <div className="grid gap-5">
          <div className="grid grid-cols-2 gap-4">
            <BigStat label="kites" value={kites} color={GT.a} sub="72°, 72°, 144°, 72°" />
            <BigStat label="darts" value={darts} color={GT.b} sub="72°, 36°, 216°, 36°" />
          </div>
          <BigStat
            label="copy lines up"
            value={`${Math.round(shareShown)}%`}
            color={copy && pos && pos.settle > 0 ? GT.opt : GT.ink}
            sub={`best shift found: ${best}% of tiles, never all`}
          />
          <p className="text-[0.85rem] leading-snug text-ink-soft">
            Slide the copy anywhere and big patches match, but never the whole tiling. These are not
            Wang tiles: they turn freely, and a kite and a dart may not join into the rhombus they
            were cut from (Penrose's matching rules forbid it).
          </p>
        </div>
      </div>
    </Specimen>
  );
}
