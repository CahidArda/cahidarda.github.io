/**
 * Braess network from Easley & Kleinberg ch. 8.2: 4000 drivers from A to B.
 * A->C and D->B take x/100 minutes (x = cars on that road); C->B and A->D take 45.
 * The optional shortcut C->D takes 0 minutes.
 * Routes: top = A-C-B, bot = A-D-B, zig = A-C-D-B (only when the shortcut is open).
 */
export type Route = 'top' | 'bot' | 'zig';
export const ROUTES: Route[] = ['top', 'bot', 'zig'];
export const DRIVERS = 4000;
export const BATCH = 200;
export type Flow = Record<Route, number>;

export function edgeLoads(f: Flow) {
  return { AC: f.top + f.zig, CB: f.top, AD: f.bot, DB: f.bot + f.zig, CD: f.zig };
}

export function edgeTimes(f: Flow) {
  const l = edgeLoads(f);
  return { AC: l.AC / 100, CB: 45, AD: 45, DB: l.DB / 100, CD: 0 };
}

export function routeTimes(f: Flow): Flow {
  const e = edgeTimes(f);
  return { top: e.AC + e.CB, bot: e.AD + e.DB, zig: e.AC + e.CD + e.DB };
}

export function average(f: Flow): number {
  const t = routeTimes(f);
  const n = f.top + f.bot + f.zig;
  return n ? (f.top * t.top + f.bot * t.bot + f.zig * t.zig) / n : 0;
}

export type Move = { from: Route; to: Route; before: number; after: number };

/**
 * One best-response step: the batch on the slowest used route moves to the route that
 * would be fastest for it after the move, if that strictly helps it. null = equilibrium.
 * With the shortcut closed, anyone still on the zig-zag must leave it first.
 */
export function step(f: Flow, open: boolean): { flow: Flow; move: Move } | null {
  const t = routeTimes(f);
  const allowed: Route[] = open ? ROUTES : ['top', 'bot'];
  // forced exit from a closed shortcut
  const sources: Route[] =
    !open && f.zig > 0 ? ['zig'] : allowed.filter((r) => f[r] > 0).sort((a, b) => t[b] - t[a]);
  for (const from of sources) {
    let best: { to: Route; after: number; flow: Flow } | null = null;
    for (const to of allowed) {
      if (to === from) continue;
      const n = Math.min(BATCH, f[from]);
      const g = { ...f, [from]: f[from] - n, [to]: f[to] + n } as Flow;
      const after = routeTimes(g)[to];
      if (!best || after < best.after - 1e-9) best = { to, after, flow: g };
    }
    if (best && (from === 'zig' && !open ? true : best.after < t[from] - 1e-9)) {
      return { flow: best.flow, move: { from, to: best.to, before: t[from], after: best.after } };
    }
  }
  return null;
}

export function equilibrium(f: Flow, open: boolean): Flow {
  let cur = f;
  for (let i = 0; i < 500; i++) {
    const s = step(cur, open);
    if (!s) break;
    cur = s.flow;
  }
  return cur;
}

export const START: Flow = { top: 2000, bot: 2000, zig: 0 };
