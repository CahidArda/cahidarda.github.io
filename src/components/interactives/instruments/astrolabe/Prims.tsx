/**
 * Draw plateGeom prims as SVG (posters, insets, 2D plates) or onto a canvas (the 3D model's
 * engraved faces). Same prims, same numbers, two renderers.
 */
import { C, readInk, type Role } from '../shared';
import type { Prim, UV } from './plateGeom';

/** Plate units → SVG: centre (cx, cy), k px per unit, v up. */
export type Xf = { cx: number; cy: number; k: number; mirror?: boolean };
export const toXY = (p: UV, f: Xf) => ({ x: f.cx + (f.mirror ? -1 : 1) * p.u * f.k, y: f.cy - p.v * f.k });

export function PrimsSvg({ prims, f, scaleW = 1, opacity = 1 }: { prims: Prim[]; f: Xf; scaleW?: number; opacity?: number }) {
  return (
    <g style={{ opacity }}>
      {prims.map((p, i) => {
        if (p.k === 'line') {
          const d = p.pts.map((q, j) => {
            const s = toXY(q, f);
            return `${j ? 'L' : 'M'} ${s.x.toFixed(2)} ${s.y.toFixed(2)}`;
          });
          return (
            <path
              key={i}
              d={d.join(' ')}
              style={{ fill: 'none', stroke: C[p.role], strokeWidth: p.w * scaleW, strokeDasharray: p.dash?.join(' ') }}
            />
          );
        }
        if (p.k === 'circle') {
          const s = toXY(p.c, f);
          return (
            <circle
              key={i}
              cx={s.x}
              cy={s.y}
              r={p.r * f.k}
              style={{
                fill: p.fill ? C[p.fill] : 'none',
                stroke: C[p.role],
                strokeWidth: p.w * scaleW,
                strokeDasharray: p.dash?.join(' '),
              }}
            />
          );
        }
        const s = toXY(p.at, f);
        const size = p.size * f.k;
        if (size < 6) return null; // too small to read at this scale: leave it off
        const rot = (f.mirror ? 1 : 1) * -(p.rot ?? 0);
        return (
          <text
            key={i}
            x={s.x}
            y={s.y}
            textAnchor={p.anchor}
            dominantBaseline="central"
            transform={p.rot ? `rotate(${rot} ${s.x} ${s.y})` : undefined}
            style={{ fill: C[p.role], fontFamily: 'var(--font-mono)', fontSize: size }}
          >
            {p.text}
          </text>
        );
      })}
    </g>
  );
}

/** Rasterise prims onto a square canvas covering [−half, half]² plate units (for 3D faces). */
export function drawPrims(
  ctx: CanvasRenderingContext2D,
  prims: Prim[],
  opts: { size: number; half: number; mirror?: boolean; bg?: Role | null; ink?: Record<Role, string>; lineScale?: number },
) {
  const ink = opts.ink ?? readInk();
  const S = opts.size;
  const k = S / (2 * opts.half);
  const ls = opts.lineScale ?? S / 700;
  const X = (u: number) => S / 2 + (opts.mirror ? -1 : 1) * u * k;
  const Y = (v: number) => S / 2 - v * k;
  ctx.clearRect(0, 0, S, S);
  if (opts.bg) {
    ctx.fillStyle = ink[opts.bg];
    ctx.beginPath();
    ctx.arc(S / 2, S / 2, opts.half * k, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const p of prims) {
    ctx.setLineDash(p.k !== 'text' && p.dash ? p.dash.map((d) => d * ls) : []);
    if (p.k === 'line') {
      ctx.strokeStyle = ink[p.role];
      ctx.lineWidth = p.w * ls;
      ctx.beginPath();
      p.pts.forEach((q, j) => (j ? ctx.lineTo(X(q.u), Y(q.v)) : ctx.moveTo(X(q.u), Y(q.v))));
      ctx.stroke();
    } else if (p.k === 'circle') {
      ctx.beginPath();
      ctx.arc(X(p.c.u), Y(p.c.v), p.r * k, 0, Math.PI * 2);
      if (p.fill) {
        ctx.fillStyle = ink[p.fill];
        ctx.fill();
      }
      ctx.strokeStyle = ink[p.role];
      ctx.lineWidth = p.w * ls;
      ctx.stroke();
    } else {
      ctx.save();
      ctx.translate(X(p.at.u), Y(p.at.v));
      if (p.rot) ctx.rotate((-(p.rot ?? 0) * Math.PI) / 180);
      ctx.fillStyle = ink[p.role];
      ctx.font = `${Math.round(p.size * k)}px ui-monospace, "SFMono-Regular", Menlo, monospace`;
      ctx.textAlign = p.anchor === 'middle' ? 'center' : p.anchor === 'end' ? 'right' : 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(p.text, 0, 0);
      ctx.restore();
    }
  }
  ctx.setLineDash([]);
}
