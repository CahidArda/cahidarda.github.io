/**
 * The stable reference for every earth plate: ONE composite piece of landscape (a minaret on
 * its hill, a road down to the shore, a strait about 700 m wide, the far shore), drawn the same
 * way in every episode so the reader watches the same ground get measured better.
 *
 * World frame (metres): X runs from the minaret's foot (X = 0) across the strait, west to
 * east. Y runs along the shore, north = +Y. Z is height above the water.
 *  - Section (top drawing): the straight cut M → A → C, X horizontal, Z up, heights drawn
 *    ×VE (said on the drawing). Everything else on one scale.
 *  - Plan (bottom drawing): X horizontal on the SAME scale and origin as the section (so the
 *    two line up), Y up. The plan is true to scale in both axes.
 * Screen: SVG viewBox units, y down. No React, no DOM.
 *
 * It is a composite: the real Süleymaniye minaret (76 m with its finial) stands over the
 * Golden Horn, and the Bosphorus is about 700 m wide only at its narrows. The plates say so.
 */
export type Pt = { x: number; y: number };

export const SITE = {
  MINARET_H: 76, // m, the tall Süleymaniye minarets with finials (round figure)
  HILL_Z: 40, // m, the hilltop precinct (illustrative)
  FLAT_X: 100, // m, the precinct is flat from the minaret's foot to here
  CHANNEL_FALL: 0.025, // m, then a water channel runs 100 m to the crest falling 2.5 cm: the Nîmes aqueduct's 25 cm per km
  CREST_X: 200, // m, where the road starts down
  SHORE_X: 480, // m, A: the quay at the foot of the slope
  QUAY_Z: 4.975, // m, height of the quay above the water: the road falls 35 m in 280 m, 1 in 8
  WIDTH: 700, // m, the strait at the narrows (round figure)
  FAR_Z: 4, // m, the far quay
  FAR_TOP_Z: 26, // m, the far shore at the edge of the drawing
  END_X: 1400, // m, the drawing's east edge
  START_X: -70, // m, the drawing's west edge
  B_Y: 200, // m, B: the baseline runs 200 m north from A along the near quay
  VE: 3, // vertical exaggeration of the section
} as const;
export const FAR_X = SITE.SHORE_X + SITE.WIDTH; // C: the far quay, straight across from A
export const CREST_Z = SITE.HILL_Z - SITE.CHANNEL_FALL;

/** Ground height along the cut (m). */
export function groundZ(x: number) {
  const { HILL_Z, FLAT_X, CREST_X, SHORE_X, QUAY_Z, FAR_Z, FAR_TOP_Z, END_X } = SITE;
  if (x <= FLAT_X) return HILL_Z;
  if (x <= CREST_X) return HILL_Z - ((HILL_Z - CREST_Z) * (x - FLAT_X)) / (CREST_X - FLAT_X);
  if (x <= SHORE_X) return CREST_Z - ((CREST_Z - QUAY_Z) * (x - CREST_X)) / (SHORE_X - CREST_X);
  if (x < FAR_X) return 0; // water
  return FAR_Z + ((FAR_TOP_Z - FAR_Z) * (x - FAR_X)) / (END_X - FAR_X);
}
/** The road's slope down the cut: 35 m in 280 m, 1 in 8, 7.125°. */
export const SLOPE_RAD = Math.atan((CREST_Z - SITE.QUAY_Z) / (SITE.SHORE_X - SITE.CREST_X));
/** Horizontal distance from the minaret's foot to A (what a stepped chain measures). */
export const LINE_H = SITE.SHORE_X; // 480 m
/** Distance along the ground surface M → A (what a wheel rolled straight down would read). */
export const LINE_SURFACE =
  SITE.FLAT_X +
  Math.hypot(SITE.CREST_X - SITE.FLAT_X, SITE.CHANNEL_FALL) +
  Math.hypot(SITE.SHORE_X - SITE.CREST_X, CREST_Z - SITE.QUAY_Z);

/**
 * The winding road in plan, from the minaret's foot to A: three cubic Béziers (world m).
 * It wanders south of the cut; its length is measured numerically (ROAD_LEN).
 */
export const ROAD: [Pt, Pt, Pt, Pt][] = [
  [{ x: 0, y: -6 }, { x: 60, y: -20 }, { x: 110, y: -120 }, { x: 190, y: -110 }],
  [{ x: 190, y: -110 }, { x: 270, y: -100 }, { x: 250, y: -30 }, { x: 330, y: -40 }],
  [{ x: 330, y: -40 }, { x: 420, y: -52 }, { x: 430, y: -4 }, { x: 480, y: 0 }],
];
function bez(p: [Pt, Pt, Pt, Pt], t: number): Pt {
  const u = 1 - t;
  const a = u * u * u;
  const b = 3 * u * u * t;
  const c = 3 * u * t * t;
  const d = t * t * t;
  return { x: a * p[0].x + b * p[1].x + c * p[2].x + d * p[3].x, y: a * p[0].y + b * p[1].y + c * p[2].y + d * p[3].y };
}
/** The road as a dense polyline with cumulative length (world m). */
export const ROAD_PTS: { p: Pt; s: number }[] = (() => {
  const out: { p: Pt; s: number }[] = [];
  let s = 0;
  let prev: Pt | null = null;
  for (const seg of ROAD)
    for (let i = prev ? 1 : 0; i <= 400; i++) {
      const p = bez(seg, i / 400);
      if (prev) s += Math.hypot(p.x - prev.x, p.y - prev.y);
      out.push({ p, s });
      prev = p;
    }
  return out;
})();
export const ROAD_LEN = ROAD_PTS[ROAD_PTS.length - 1].s;
/** Point on the road at fraction f (0..1) of its length, and the direction there. */
export function roadAt(f: number) {
  const target = Math.max(0, Math.min(1, f)) * ROAD_LEN;
  let lo = 0;
  let hi = ROAD_PTS.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (ROAD_PTS[m].s < target) lo = m;
    else hi = m;
  }
  const a = ROAD_PTS[lo];
  const b = ROAD_PTS[hi];
  const t = b.s === a.s ? 0 : (target - a.s) / (b.s - a.s);
  const p = { x: a.p.x + (b.p.x - a.p.x) * t, y: a.p.y + (b.p.y - a.p.y) * t };
  const dir = Math.atan2(b.p.y - a.p.y, b.p.x - a.p.x);
  return { p, dir, i: hi };
}

/* ───────────────────────── screen layout ───────────────────────── */

export const PLAN_Y0 = -150; // m, south edge of the plan
export const PLAN_Y1 = 250; // m, north edge of the plan

/** Every meaningful coordinate of the reference drawing at width W. */
export function siteScene(W: number, bY: number = SITE.B_Y) {
  const { START_X, END_X, VE, MINARET_H, HILL_Z, CREST_X, SHORE_X, QUAY_Z } = SITE;
  const padL = 8;
  const padR = 8;
  const k = (W - padL - padR) / (END_X - START_X); // px per metre, horizontal (both drawings)
  const kz = k * VE; // px per metre, section heights
  const tagH = 30; // the "best so far" line at the top
  const top = tagH + 16; // y of the finial's tip
  const sea = top + (HILL_Z + MINARET_H) * kz; // section: water level
  const secBottom = sea + 22;
  const planTop = secBottom + 26;
  const planH = (PLAN_Y1 - PLAN_Y0) * k;
  const H = Math.round(planTop + planH + 30);
  const sx = (x: number) => padL + (x - START_X) * k;
  const S = (x: number, z: number): Pt => ({ x: sx(x), y: sea - z * kz });
  const P = (x: number, y: number): Pt => ({ x: sx(x), y: planTop + (PLAN_Y1 - y) * k });
  // ground profile (section), drawn as a polyline with the water gap
  const nearGround: Pt[] = [S(START_X, HILL_Z), S(SITE.FLAT_X, HILL_Z), S(CREST_X, CREST_Z), S(SHORE_X, QUAY_Z), S(SHORE_X, 0)];
  const farGround: Pt[] = [S(FAR_X, 0), S(FAR_X, SITE.FAR_Z), S(END_X, SITE.FAR_TOP_Z)];
  const names = {
    M: S(0, HILL_Z), // minaret foot
    tip: S(0, HILL_Z + MINARET_H),
    flat: S(SITE.FLAT_X, HILL_Z),
    crest: S(CREST_X, CREST_Z),
    A: S(SHORE_X, QUAY_Z),
    C: S(FAR_X, SITE.FAR_Z),
    pM: P(0, 0),
    pA: P(SHORE_X, 0),
    pB: P(SHORE_X, bY),
    pC: P(FAR_X, 0),
  };
  const road = ROAD_PTS.filter((_, i) => i % 6 === 0 || i === ROAD_PTS.length - 1).map((r) => P(r.p.x, r.p.y));
  const nearShoreX = sx(SHORE_X);
  const farShoreX = sx(FAR_X);
  const labels = {
    section: { x: W - padR, y: top - 4, text: `section along M–A–C, heights ×${VE}`, size: 10 },
    plan: { x: W - padR, y: planTop - 8, text: 'plan, to scale', size: 10 },
    M: { x: names.M.x + 7, y: names.M.y - 6, text: 'M', size: 12 },
    A: { x: names.A.x - 4, y: names.A.y - 8, text: 'A', size: 12 },
    C: { x: names.C.x + 5, y: names.C.y - 8, text: 'C', size: 12 },
    pA: { x: names.pA.x - 6, y: names.pA.y + 12, text: 'A', size: 12 },
    pB: { x: names.pB.x - 6, y: names.pB.y + 4, text: 'B', size: 12 },
    pC: { x: names.pC.x + 6, y: names.pC.y + 12, text: 'C', size: 12 },
    pM: { x: names.pM.x + 7, y: names.pM.y + 12, text: 'M', size: 12 },
    north: { x: W - padR - 6, y: planTop + 12, text: 'N', size: 10 },
  };
  const tag = { x: padL, y: 16 };
  return {
    W,
    H,
    k,
    kz,
    top,
    sea,
    secBottom,
    planTop,
    planH,
    sx,
    S,
    P,
    nearGround,
    farGround,
    names,
    road,
    nearShoreX,
    farShoreX,
    labels,
    tag,
    pad: { l: padL, r: padR },
  };
}
export type SiteScene = ReturnType<typeof siteScene>;

/** Format a length the way the plates print it. */
export function fmtM(m: number, dp = 1) {
  if (!Number.isFinite(m)) return '∞';
  const a = Math.abs(m);
  if (a >= 10_000) return `${(m / 1000).toFixed(1)} km`;
  if (a >= 1) return `${m.toFixed(dp)} m`;
  if (a >= 0.01) return `${(m * 100).toFixed(1)} cm`;
  return `${(m * 1000).toFixed(1)} mm`;
}
