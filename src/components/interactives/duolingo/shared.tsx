/**
 * Shared primitives for the "Babies Don't Use Duolingo" figures. Reuses the data-platform
 * frame, edges and geometry helpers so the post reads as house style, plus a node with a
 * readable subtitle size and a curved "repeat" connector for the loop diagrams. All figures
 * are static (no hooks), so they render at build time with no client JS.
 */
export {
  ArrowDefs,
  Edge,
  Widget,
  bottomOf,
  leftOf,
  rightOf,
  topOf,
  type Box,
  type NState,
} from '../dataplat/shared';
import type { Box, NState } from '../dataplat/shared';

export const INK = 'var(--color-ink)';
export const ACCENT = 'var(--color-accent)';
export const MONO = 'var(--font-mono)';
export const SANS = 'var(--font-sans)';

/** Node with a title and an optional subtitle, sized for figures read at phone width. */
export function Node({
  n,
  title,
  sub,
  color = INK,
  state = 'wait',
  titleSize = 13,
  subSize = 10.5,
}: {
  n: Box;
  title: string;
  sub?: string;
  color?: string;
  state?: NState;
  titleSize?: number;
  subSize?: number;
}) {
  const active = state === 'active';
  const off = state === 'off';
  return (
    <g style={{ opacity: off ? 0.55 : 1 }}>
      <rect
        x={n.cx - n.w / 2}
        y={n.cy - n.h / 2}
        width={n.w}
        height={n.h}
        style={{
          fill: active ? color : 'var(--color-paper)',
          stroke: off ? 'var(--color-line-strong)' : color,
          strokeWidth: off ? 1.5 : 2,
          strokeDasharray: off ? '4 4' : undefined,
        }}
      />
      <text
        x={n.cx}
        y={sub ? n.cy - subSize * 0.55 : n.cy + 1}
        textAnchor="middle"
        dominantBaseline="middle"
        style={{
          fill: active ? 'var(--color-paper)' : off ? 'var(--color-muted)' : 'var(--color-ink)',
          fontFamily: MONO,
          fontSize: titleSize,
          fontWeight: 600,
        }}
      >
        {title}
      </text>
      {sub && (
        <text
          x={n.cx}
          y={n.cy + titleSize * 0.85}
          textAnchor="middle"
          dominantBaseline="middle"
          style={{
            fill: active ? 'var(--color-paper)' : 'var(--color-muted)',
            fontFamily: MONO,
            fontSize: subSize,
          }}
        >
          {sub}
        </text>
      )}
    </g>
  );
}

/** A curved connector (cubic Bezier) with the shared arrowhead, for "repeat" legs. */
export function Curve({ d, dashed = false }: { d: string; dashed?: boolean }) {
  return (
    <path
      d={d}
      markerEnd="url(#dp-arrow)"
      style={{
        fill: 'none',
        stroke: ACCENT,
        strokeWidth: 2,
        strokeDasharray: dashed ? '5 4' : undefined,
      }}
    />
  );
}

/** Small mono label inside an SVG figure. */
export function Label({
  x,
  y,
  children,
  anchor = 'middle',
  muted = true,
  size = 11,
  rotate,
}: {
  x: number;
  y: number;
  children: React.ReactNode;
  anchor?: 'start' | 'middle' | 'end';
  muted?: boolean;
  size?: number;
  rotate?: number;
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      dominantBaseline="middle"
      transform={rotate ? `rotate(${rotate} ${x} ${y})` : undefined}
      style={{
        fill: muted ? 'var(--color-muted)' : 'var(--color-ink)',
        fontFamily: MONO,
        fontSize: size,
      }}
    >
      {children}
    </text>
  );
}

/** Caption line under a figure, inside the frame. */
export function Caption({ children }: { children: React.ReactNode }) {
  return <p className="mt-3 font-mono text-[0.7rem] leading-relaxed text-muted">{children}</p>;
}
