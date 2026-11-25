/** Shared SVG geometry for directed graphs drawn with circles or boxes. */
export type Pt = { x: number; y: number };

const unit = (dx: number, dy: number) => {
  const l = Math.hypot(dx, dy) || 1;
  return { x: dx / l, y: dy / l };
};

/** Control point of an edge: bent sideways by `bend` (pairs of opposite edges bend apart). */
export function control(a: Pt, b: Pt, bend: number): Pt {
  const u = unit(b.x - a.x, b.y - a.y);
  return { x: (a.x + b.x) / 2 - u.y * bend, y: (a.y + b.y) / 2 + u.x * bend };
}

/** Point where a ray from the centre of a circle toward `to` leaves the circle. */
export function circleExit(c: Pt, r: number, to: Pt): Pt {
  const u = unit(to.x - c.x, to.y - c.y);
  return { x: c.x + u.x * r, y: c.y + u.y * r };
}

/** Point where a ray from the centre of a w x h box toward `to` leaves the box. */
export function boxExit(c: Pt, w: number, h: number, to: Pt, pad = 0): Pt {
  const dx = to.x - c.x;
  const dy = to.y - c.y;
  const sx = dx === 0 ? Infinity : (w / 2 + pad) / Math.abs(dx);
  const sy = dy === 0 ? Infinity : (h / 2 + pad) / Math.abs(dy);
  const s = Math.min(sx, sy);
  return { x: c.x + dx * s, y: c.y + dy * s };
}

/** Point on a quadratic Bezier. */
export function qAt(p0: Pt, c: Pt, p1: Pt, t: number): Pt {
  const m = 1 - t;
  return {
    x: m * m * p0.x + 2 * m * t * c.x + t * t * p1.x,
    y: m * m * p0.y + 2 * m * t * c.y + t * t * p1.y,
  };
}
