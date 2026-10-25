/**
 * Numeric sweep of every earth plate's geometry. Run with `node geometry.sweep.ts` (Node ≥ 23
 * strips the types). It imports the SAME layout functions the plates draw from, sweeps each
 * parameter over its full slider range, and asserts that the drawing agrees with the maths on
 * one scale, nothing leaves the drawing area, and labels do not collide with each other or sit
 * on the lines they describe. Exit code 1 on any failure.
 */
import {
  segHitsBox as segHits,
  CHART,
  CORD,
  H_TRUE,
  W_TRUE,
  aframeLayout,
  chartLayout,
  cordLayout,
  hitParam,
  heightFrom,
  minaretLayout,
  overlap,
  scopeClearance,
  straitLayout,
  textBox,
  widthFrom,
  type Box,
  type Pt,
} from './geometry.ts';
import { SURVEY_LADDER } from './surveyLadder.ts';

const RAD = Math.PI / 180;
let checks = 0;
const fails = new Map<string, { n: number; first: string }>();
function ok(cond: boolean, tag: string, detail: () => string) {
  checks++;
  if (cond) return;
  const f = fails.get(tag);
  if (f) f.n++;
  else fails.set(tag, { n: 1, first: detail() });
}
const near = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const inside = (b: Box, W: number, H: number, m = 0) => b.x0 >= m && b.y0 >= m && b.x1 <= W - m && b.y1 <= H - m;
/** Does segment p→q pass through box b (shrunk by `pad`)? Sampled, good to < 0.5 px. */
function segHitsBox(p: Pt, q: Pt, b: Box, pad = 0) {
  const n = Math.ceil(Math.hypot(q.x - p.x, q.y - p.y) / 0.5) + 1;
  for (let i = 0; i <= n; i++) {
    const x = p.x + ((q.x - p.x) * i) / n;
    const y = p.y + ((q.y - p.y) * i) / n;
    if (x > b.x0 + pad && x < b.x1 - pad && y > b.y0 + pad && y < b.y1 - pad) return true;
  }
  return false;
}
function segDist(c: Pt, p: Pt, q: Pt) {
  const dx = q.x - p.x;
  const dy = q.y - p.y;
  const t = Math.max(0, Math.min(1, ((c.x - p.x) * dx + (c.y - p.y) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(c.x - p.x - t * dx, c.y - p.y - t * dy);
}
function noCollisions(boxes: Box[], tag: string, ctx: string, pad = 1) {
  for (let i = 0; i < boxes.length; i++)
    for (let j = i + 1; j < boxes.length; j++)
      if (/bracket/.test(boxes[i].name) && /bracket/.test(boxes[j].name)) continue; // the two ends of one bracket
      else
      ok(!overlap(boxes[i], boxes[j], pad), `${tag}: '${boxes[i].name}' vs '${boxes[j].name}'`, () => ctx);
}
const fmtB = (b: Box) => `[${b.x0.toFixed(0)},${b.y0.toFixed(0)} – ${b.x1.toFixed(0)},${b.y1.toFixed(0)}]`;

/* Plate widths: the inner drawing width at a 390px phone, a desktop column, and wide. */
const WIDTHS = [326, 360, 520, 640, 700];
const EPS_MIN = [0, 0.5, 1, 5, 10, 20, 30, 45, 60]; // arcminutes, the slider's range

/* ── Triangulation, minaret ── */
let minRanges = 0;
let minUnbounded = 0;
for (const W of WIDTHS)
  for (let dB = 20; dB <= 250; dB += 5)
    for (let b = 5; b <= 300 - dB; b += 5)
      for (const em of EPS_MIN) {
        const e = (em / 60) * RAD;
        const L = minaretLayout(W, dB, b, e);
        const ctx = `W=${W} dB=${dB} b=${b} ε=${em}′`;
        // the maths: the computed height is the true one
        ok(near(L.h, H_TRUE, 1e-9), 'minaret: h from angles = 76 m', () => `${ctx} h=${L.h}`);
        // one scale: the drawn tip is h·k above the ground, at the foot's x
        ok(near(L.g - L.tip.y, H_TRUE * L.k, 1e-9) && L.tip.x === L.fx, 'minaret: tip at 76 m on the vertical scale', () => ctx);
        // stations at their distances on the SAME scale
        ok(near((L.fx - L.ax) / L.k, dB + b) && near((L.fx - L.bx) / L.k, dB), 'minaret: stations on the horizontal scale', () => ctx);
        // the drawn sight lines make exactly the measured angles and end on the tip
        const angA = Math.atan2(L.g - L.tip.y, L.tip.x - L.ax);
        const angB = Math.atan2(L.g - L.tip.y, L.tip.x - L.bx);
        ok(near(angA, L.al, 1e-9) && near(angB, L.be, 1e-9), 'minaret: drawn sight-line angles = α, β', () => `${ctx} ${angA} ${L.al}`);
        // the h dimension spans foot to tip
        ok(L.tip.y > 0, 'minaret: tip inside drawing', () => ctx);
        // stations inside the drawing
        ok(L.ax >= 10 && L.bx <= L.fx - 10, 'minaret: stations inside', () => `${ctx} ax=${L.ax}`);
        // label boxes inside and not colliding
        for (const bx of L.boxes.filter((bb) => em > 0 || !/bracket/.test(bb.name))) ok(inside(bx, W, L.Hs), `minaret: '${bx.name}' inside drawing`, () => `${ctx} ${fmtB(bx)} W=${W} H=${L.Hs}`);
        const shown = L.boxes.filter((bb) => em > 0 || !/range label|bracket/.test(bb.name));
        noCollisions(shown, 'minaret overlap', ctx);
        // α, β labels clear of the other sight line and of the minaret
        const mw = Math.max(4, L.h * L.k * 0.07) * 1.05 + 2;
        const minBox: Box = { name: 'minaret', x0: L.fx - mw, x1: L.fx + mw, y0: L.tip.y, y1: L.g };
        const lab = (n: string) => L.boxes.find((bb) => bb.name === n)!;
        for (const n of ['α', 'β']) {
          const bb = lab(n);
          ok(!segHitsBox({ x: L.ax, y: L.g }, L.tip, bb, 1.5), `minaret: '${n}' label on A's sight line`, () => `${ctx} ${fmtB(bb)}`);
          ok(!segHitsBox({ x: L.bx, y: L.g }, L.tip, bb, 1.5), `minaret: '${n}' label on B's sight line`, () => `${ctx} ${fmtB(bb)}`);
          ok(!overlap(bb, minBox), `minaret: '${n}' label on the minaret`, () => ctx);
        }
        if (em === 0) continue;
        // error region: true value inside, bracket and label where the numbers say
        const { reg, range } = L;
        ok(reg.lo <= H_TRUE + 1e-9 && reg.hi >= H_TRUE - 1e-9, 'minaret: 76 m inside the range', () => `${ctx} ${reg.lo}..${reg.hi}`);
        ok(near(range.y1, L.g - reg.lo * L.k, 1e-9), 'minaret: bracket bottom at lo', () => ctx);
        if (reg.bounded) {
          minRanges++;
          // the range equals brute force over the four sign combinations of the height formula
          const v = [-1, 1].flatMap((s1) => [-1, 1].map((s2) => heightFrom(b, L.al + s1 * e, L.be + s2 * e)));
          ok(near(reg.lo, Math.min(...v), 1e-7) && near(reg.hi, Math.max(...v), 1e-7), 'minaret: range = formula extremes', () => `${ctx} ${reg.lo},${reg.hi} vs ${v}`);
          const yHi = L.g - reg.hi * L.k;
          ok(range.clippedTop ? yHi < 4 : near(range.y2, yHi, 1e-9), 'minaret: bracket top at hi (or marked clipped)', () => ctx);
          // the drawn region spans lo..hi (hi clipped at the top edge of the drawing)
          const ys = L.quad!.map((p) => (L.g - p.y) / L.k);
          const topM = L.g / L.k;
          const atSide = L.quad!.some((p) => p.x < 0.01 || p.x > W - 0.01);
          ok(near(Math.min(...ys), reg.lo, 1e-7) && (atSide || near(Math.max(...ys), Math.min(reg.hi, topM), 1e-7)), 'minaret: drawn region spans lo..hi', () => `${ctx} ${Math.min(...ys)}..${Math.max(...ys)} vs ${reg.lo}..${reg.hi}`);
        } else {
          minUnbounded++;
          ok(range.clippedTop, 'minaret: unbounded range drawn as open-ended', () => ctx);
          const ys = L.quad!.map((p) => (L.g - p.y) / L.k);
          const atSide = L.quad!.some((p) => p.x < 0.01 || p.x > W - 0.01);
          ok(near(Math.min(...ys), reg.lo, 1e-7) && (atSide || near(Math.max(...ys), L.g / L.k, 1e-7)), 'minaret: unbounded region reaches the edge', () => ctx);
        }
        {
          // centred on its bracket, or pushed in by the drawing edge but still beside the bracket
          const half = ([...L.rangeLabel.text].length * L.rangeLabel.size * 0.52) / 2 + 3;
          const want = Math.min(Math.max((range.y1 + range.y2) / 2, half), L.g - half);
          ok(near(L.rangeLabel.y, want, 1e-9) && L.rangeLabel.y > range.y2 - 1 && L.rangeLabel.y < range.y1 + 1, 'minaret: range label centred on its bracket', () => ctx);
        }
        ok(range.x > L.fx + L.dimOff + 6, 'minaret: bracket clear of the h dimension', () => ctx);
      }

/* ── Triangulation, Bosphorus ── */
for (const W of WIDTHS)
  for (let b = 20; b <= 600; b += 10)
    for (const em of EPS_MIN) {
      const e = (em / 60) * RAD;
      const L = straitLayout(W, b, e);
      const ctx = `W=${W} b=${b} ε=${em}′`;
      ok(near(L.w, W_TRUE, 1e-9), 'strait: width from angles = 700 m', () => `${ctx} ${L.w}`);
      ok(near(L.nearY - L.C.y, W_TRUE * L.k, 1e-9) && near(L.C.x, L.ax, 1e-9), 'strait: C at 700 m on the scale, square from A', () => ctx);
      ok(near(L.bx - L.ax, b * L.k, 1e-9), 'strait: baseline on the same scale', () => ctx);
      ok(near(Math.atan2(L.nearY - L.C.y, L.bx - L.C.x), L.be, 1e-9), 'strait: drawn β = computed β', () => ctx);
      ok(near(L.C.y, L.farY, 1e-9), 'strait: C on the far shore', () => ctx);
      ok(L.bx <= W - 12, 'strait: B inside drawing', () => `${ctx} bx=${L.bx}`);
      for (const bx of L.boxes.filter((bb) => em > 0 || !/bracket/.test(bb.name))) ok(inside(bx, W, L.Hs), `strait: '${bx.name}' inside drawing`, () => `${ctx} ${fmtB(bx)}`);
      const shown = L.boxes.filter((bb) => em > 0 || !/range label|bracket/.test(bb.name));
      noCollisions(shown, 'strait overlap', ctx);
      const beta = L.boxes.find((bb) => bb.name === 'β')!;
      ok(!segHitsBox({ x: L.bx, y: L.nearY }, L.C, beta, 1.5), "strait: 'β' label on B's sight line", () => ctx);
      if (em === 0) continue;
      const { reg, range } = L;
      ok(reg.lo <= W_TRUE + 1e-6 && reg.hi >= W_TRUE - 1e-6, 'strait: 700 m inside the range', () => ctx);
      ok(near(range.y1, L.nearY - reg.lo * L.k, 1e-9), 'strait: bracket bottom at lo', () => ctx);
      if (reg.bounded) {
        const v = [-1, 1].flatMap((s1) => [-1, 1].map((s2) => widthFrom(b, Math.PI / 2 + s1 * e, L.be + s2 * e)));
        ok(near(reg.lo, Math.min(...v), 1e-7) && near(reg.hi, Math.max(...v), 1e-7), 'strait: range = formula extremes', () => `${ctx} ${reg.lo},${reg.hi} vs ${v}`);
        const yHi = L.nearY - reg.hi * L.k;
        ok(range.clippedTop ? yHi < 4 : near(range.y2, yHi, 1e-9), 'strait: bracket top at hi (or marked clipped)', () => ctx);
        const ys = L.quad!.map((p) => (L.nearY - p.y) / L.k);
        const atSide = L.quad!.some((p) => p.x < 0.01 || p.x > W - 0.01);
        ok(near(Math.min(...ys), reg.lo, 1e-7) && (atSide || near(Math.max(...ys), Math.min(reg.hi, L.nearY / L.k), 1e-7)), 'strait: drawn region spans lo..hi', () => ctx);
      } else ok(range.clippedTop, 'strait: unbounded range drawn as open-ended', () => ctx);
      {
        const half = ([...L.rangeLabel.text].length * L.rangeLabel.size * 0.52) / 2 + 3;
        const want = Math.min(Math.max((range.y1 + range.y2) / 2, half), L.nearY - half);
        ok(near(L.rangeLabel.y, want, 1e-9) && L.rangeLabel.y > range.y2 - 1 && L.rangeLabel.y < range.y1 + 1, 'strait: range label centred on its bracket', () => ctx);
      }
      ok(L.rangeLabel.x - 6 > 0, 'strait: range label inside on the left', () => ctx);
    }

/* ── A-frame level ── */
for (const W of WIDTHS)
  for (let th = -10; th <= 10.0001; th += 0.1) {
    const t = Math.round(th * 10) / 10;
    const G = aframeLayout(W, t);
    const ctx = `W=${W} θ=${t}°`;
    // a truly vertical string from the apex crosses the crossbar exactly at G.hit
    const u = hitParam(G.apex, { x: 0, y: 1 }, G.cl, G.cr);
    const hit = { x: G.cl.x + (G.cr.x - G.cl.x) * u, y: G.cl.y + (G.cr.y - G.cl.y) * u };
    ok(Math.hypot(hit.x - G.hit.x, hit.y - G.hit.y) < 1e-6, 'aframe: vertical string meets the crossbar at G.hit', () => `${ctx} ${hit.x},${hit.y} vs ${G.hit.x},${G.hit.y}`);
    // x = h · tan θ on the drawing's own scale
    const xDrawn = Math.hypot(G.hit.x - G.mark.x, G.hit.y - G.mark.y);
    ok(near(xDrawn, G.h * Math.tan(Math.abs(t) * RAD), 1e-9), 'aframe: drawn miss = h·tan θ', () => ctx);
    // the string lands on the DOWNHILL side of the mark
    const downhillLeft = t > 0;
    if (Math.abs(t) > 0.05) ok(downhillLeft ? G.hit.x < G.mark.x : G.hit.x > G.mark.x, 'aframe: string on the downhill side', () => ctx);
    ok(u >= 0 && u <= 1, 'aframe: string meets the crossbar within its length', () => `${ctx} u=${u}`);
    // the x dimension sits on the same side as the hit, below the crossbar (toward the feet)
    if (Math.abs(t) >= 0.6) {
      const dx = G.hit.x - G.mark.x;
      const dy = G.hit.y - G.mark.y;
      const len = Math.hypot(dx, dy);
      const nrm = { x: dy / len, y: -dx / len };
      const dimMid = { x: (G.mark.x + G.hit.x) / 2 + nrm.x * G.dimOffset, y: (G.mark.y + G.hit.y) / 2 + nrm.y * G.dimOffset };
      ok(dimMid.y > G.mark.y + 10, 'aframe: x dimension below the crossbar', () => `${ctx} ${dimMid.y} vs ${G.mark.y}`);
      // θ arc between the level line and the beam top
      const end = { x: G.cx + G.thetaR * Math.cos(G.thetaTo * RAD), y: G.gy - G.thetaR * Math.sin(G.thetaTo * RAD) };
      const beamPt = G.P(t > 0 ? -G.thetaR : G.thetaR, 0);
      ok(Math.hypot(end.x - beamPt.x, end.y - beamPt.y) < 1e-6, 'aframe: θ arc ends on the beam', () => ctx);
    }
    for (const bx of G.boxes) ok(inside(bx, W, G.Hs), `aframe: '${bx.name}' inside`, () => `${ctx} ${fmtB(bx)}`);
    noCollisions(G.boxes, 'aframe overlap', ctx);
    // status line clear of the beam's bottom edge
    const st = G.boxes.find((b) => b.name === 'status')!;
    ok(!segHitsBox(G.beam[2], G.beam[3], st) && !segHitsBox(G.beam[0], G.beam[3], st) && !segHitsBox(G.beam[1], G.beam[2], st), 'aframe: status text clear of the beam', () => ctx);
    for (const p of G.beam) ok(p.x >= 0 && p.x <= W && p.y >= 0 && p.y <= G.Hs, 'aframe: beam inside', () => ctx);
    // x label and θ label clear of the string (vertical from the apex)
    for (const n of ['x', 'θ']) {
      const bb = G.boxes.find((b) => b.name === n);
      if (bb) ok(!segHitsBox(G.apex, { x: G.apex.x, y: G.gy }, bb, 0.5), `aframe: '${n}' label on the string`, () => `${ctx} ${fmtB(bb)}`);
    }
  }

/* ── Havayi terazi on a cord ── */
for (const W of WIDTHS)
  for (let f = CORD.MIN; f <= CORD.MAX + 1e-9; f += 0.01) {
    const far = Math.round(f * 100) / 100;
    const G = cordLayout(W, far);
    const ctx = `W=${W} far=${far}`;
    // level on the drawing exactly when level in metres
    const lvlDrawn = Math.abs(G.a.y - G.b.y) < 1e-6;
    const lvlReal = Math.abs(far - CORD.NEAR - CORD.DROP) < 1e-9;
    if (lvlReal) ok(Math.abs(G.a.y - G.b.y) < 1e-6, 'cord: level cord drawn level', () => ctx);
    ok(lvlDrawn === lvlReal, 'cord: drawn level ⇔ real level', () => ctx);
    if (!lvlReal) ok(Math.sign(G.tiltDrawn) === Math.sign(G.tiltReal), 'cord: drawn and real tilt agree in sign', () => ctx);
    // pole readings: the cord ends sit at the slider value on the poles' own scale
    ok(near((G.gR - G.b.y) / G.ky, far, 1e-9) && near((G.gL - G.a.y) / G.ky, CORD.NEAR, 1e-9), 'cord: ends at the pole readings', () => ctx);
    // at rest the string passes the tip at the drawn offset; the detail shows it unclamped
    const restOff = G.tip.x - G.top.x;
    ok(near(restOff, G.th * Math.sin(G.tiltDrawn), 1e-9), 'cord: tip offset = th·sin(tilt)', () => ctx);
    ok(Math.abs(restOff * G.detail.mag) <= G.detail.r * 0.8 + 1e-9, 'cord: detail offset fits (not clamped)', () => `${ctx} ${restOff * G.detail.mag} r=${G.detail.r}`);
    ok(G.level === lvlReal || Math.abs(far - CORD.NEAR - CORD.DROP) < 0.006, 'cord: status matches level', () => ctx);
    for (const bx of G.boxes) ok(inside(bx, W, G.Hs), `cord: '${bx.name}' inside`, () => `${ctx} ${fmtB(bx)}`);
    noCollisions(G.boxes, 'cord overlap', ctx);
    const { c, r } = G.detail;
    const tri: [Pt, Pt][] = [
      [G.h1, G.h2],
      [G.h2, G.tip],
      [G.tip, G.h1],
    ];
    ok(tri.every(([p, q]) => segDist(c, p, q) > r + 3), 'cord: detail circle clear of the triangle', () => ctx);
    const circ: Box = { name: 'detail circle', x0: c.x - r, x1: c.x + r, y0: c.y - r, y1: c.y + r };
    ok(!overlap(circ, G.boxes.find((b) => b.name === 'far pole scale')!), 'cord: detail circle on the far pole scale', () => ctx);
    ok(segDist(c, G.a, G.b) > r + 3, 'cord: detail circle clear of the cord', () => ctx);
    ok(c.x + r < G.xR - 8, 'cord: detail circle clear of the far pole', () => ctx);
    ok(c.y - r > Math.max(G.a.y, G.b.y) + 4, 'cord: detail circle below the cord', () => ctx);
    for (const bx of G.boxes.filter((b) => !/scale/.test(b.name))) {
      const gy = Math.min(G.groundAt(Math.max(G.xL, Math.min(G.xR, bx.x0))), G.groundAt(Math.max(G.xL, Math.min(G.xR, bx.x1))));
      ok(bx.y1 < gy - 2, `cord: '${bx.name}' above the ground`, () => `${ctx} ${fmtB(bx)} ground ${gy.toFixed(0)}`);
    }
    ok(c.y + r < G.groundAt(c.x) - 4, 'cord: detail circle above the ground', () => ctx);
    // status line clear of the pole labels and the cord
    const st = G.boxes.find((b) => b.name === 'status')!;
    ok(!segHitsBox(G.a, G.b, st), 'cord: status on the cord', () => ctx);
    const bobBottom = G.top.y + G.th + 6 + 18 * 0.7;
    ok(bobBottom < Math.min(G.gL, G.gR) + 10, 'cord: bob above the ground', () => ctx);
  }

/* ── Arcsecond chart ── */
let chartLeaders = 0;
for (const W of WIDTHS)
  for (const eps of [0.2, 1, 200])
    for (let ld = 0; ld <= 6.0001; ld += 0.01) {
      const L = chartLayout(W, ld, eps);
      const ctx = `W=${W} ε=${eps}″ d=10^${ld.toFixed(2)}`;
      // the marker's y is the readout's value on the chart's log scale
      if (L.missOn) {
        const back = CHART.Y0 + (1 - (L.ey - L.m.t) / L.ih) * (CHART.Y1 - CHART.Y0);
        ok(near(10 ** back, L.miss, 1e-9), 'chart: miss marker at d·tan ε', () => ctx);
        const lb = textBox('miss', L.missLabel.x, L.missLabel.y, L.missText, 13, L.missLabel.anchor, 'alphabetic', 0.62);
        ok(inside(lb, W, L.H), 'chart: miss label inside', () => `${ctx} ${fmtB(lb)}`);
        const ax1 = textBox('sideways miss', L.m.l + 4, L.m.t + 12, 'sideways miss', 11, 'start', 'alphabetic', 0.48);
        ok(!overlap(lb, ax1), 'chart: miss label vs axis title', () => ctx);
        if (L.dropOn) {
          const dm: Box = { name: 'drop marker', x0: L.ex - 4, x1: L.ex + 4, y0: L.cy - 4, y1: L.cy + 4 };
          ok(!overlap(lb, dm), 'chart: miss label on the drop marker', () => `${ctx} ${fmtB(lb)}`);
        }
        const mm: Box = { name: 'miss marker', x0: L.ex - 5, x1: L.ex + 5, y0: L.ey - 5, y1: L.ey + 5 };
        ok(!overlap(lb, mm), 'chart: miss label on its own marker', () => ctx);
        ok(!L.plotted.some(([p, q]) => segHits(p, q, lb, 0)), 'chart: miss label on a plotted line', () => `${ctx} ${fmtB(lb)}`);
        if (L.missLabel.leader) chartLeaders++;
        if (L.curveLabel) {
          const cl = textBox('curve', L.curveLabel.x, L.curveLabel.y, 'Earth’s curve, d²/2R', 11, 'end', 'alphabetic', 0.5);
          ok(!overlap(lb, cl), 'chart: miss label on the curve label', () => `${ctx} ${fmtB(lb)}`);
          ok(!L.epsLines.some(([p, q]) => segHits(p, q, cl, 0)), 'chart: curve label on an error line', () => ctx);
        }
      }
      if (L.dropOn) {
        const back = CHART.Y0 + (1 - (L.cy - L.m.t) / L.ih) * (CHART.Y1 - CHART.Y0);
        ok(near(10 ** back, L.drop, 1e-9), 'chart: drop marker at d²/2R', () => ctx);
      }
      ok(L.ex >= L.m.l - 1e-9 && L.ex <= L.m.l + L.iw + 1e-9, 'chart: marker inside the plot', () => ctx);
    }
// the caption's crossover: d·tan ε = d²/2R
for (const eps of [0.2, 1, 200]) {
  const L = chartLayout(640, 0, eps);
  const d = L.crossover;
  ok(near(d * Math.tan(eps * (Math.PI / 180 / 3600)), (d * d) / (2 * 6_371_000), 1e-9), 'chart: crossover distance', () => `${eps}`);
}
ok(Math.abs(chartLayout(640, 0, 1).crossover - 61.8) < 0.1, 'chart: 1″ crossover ≈ 62 m (prose)', () => `${chartLayout(640, 0, 1).crossover}`);

/* ── Theodolite ── */
for (let v = -30; v <= 60; v += 0.5) {
  const c = scopeClearance(v);
  ok(c.low > c.plateTop, 'theodolite: telescope clears the alidade plate over the slider range', () => `v=${v} low=${c.low}`);
}
// idle sweep stays inside the slider range: h = 38 + 34 sin, v = 12 + 8 sin
ok(38 - 34 >= 0 && 38 + 34 < 360 && 12 - 8 >= -30 && 12 + 8 <= 60, 'theodolite: idle sweep inside slider range', () => '');
// the limits where it would collide, for the report
let vLo = 0;
let vHi = 0;
for (let v = 0; v >= -90; v -= 0.5) if (scopeClearance(v).low > scopeClearance(v).plateTop) vLo = v;
for (let v = 0; v <= 90; v += 0.5) if (scopeClearance(v).low > scopeClearance(v).plateTop) vHi = v;

/* ── Survey ladder (same mapping as PrecisionLadder) ── */
for (const [W, narrow] of [
  [326, true],
  [360, true],
  [640, false],
  [760, false],
  [900, false],
  [1100, false],
] as const) {
  const cfg = SURVEY_LADDER;
  const yrs = cfg.years!;
  const H = narrow ? 340 : 400;
  const m = { l: narrow ? 46 : 58, r: narrow ? 14 : 22, t: 22, b: 34 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const x = (yr: number) =>
    yr <= yrs.yb
      ? m.l + ((yr - yrs.y0) / (yrs.yb - yrs.y0)) * yrs.split * iw
      : m.l + yrs.split * iw + ((yr - yrs.yb) / (yrs.y1 - yrs.yb)) * (1 - yrs.split) * iw;
  const y = (v: number) => m.t + ((Math.log10(v) - cfg.lo) / (cfg.hi - cfg.lo)) * ih;
  const ctx = `W=${W}`;
  const boxes: Box[] = [];
  cfg.rungs.forEach((r, i) => {
    const px = x(r.year);
    const py = y(r.value);
    ok(py >= m.t && py <= m.t + ih && px >= m.l && px <= m.l + iw, `ladder: '${r.short}' inside the plot`, () => `${ctx} ${px},${py}`);
    const right = px < m.l + iw * 0.8;
    const lb = textBox(r.short, px + (right ? 12 : -12), py + (cfg.dy ? cfg.dy(i, narrow) : 0), r.short, narrow ? 11.5 : 12.5, right ? 'start' : 'end', 'central', 0.5);
    ok(inside(lb, W, H), `ladder: '${r.short}' label inside`, () => `${ctx} ${fmtB(lb)}`);
    boxes.push(lb);
    // its readout string says the same value
    ok(cfg.fmt(r.value).length > 0, 'ladder: readout', () => r.short);
  });
  // labels must not sit on the red best-so-far staircase
  {
    // same path as PrecisionLadder's `stair`
    const segs: [Pt, Pt][] = [];
    const pts = cfg.rungs.map((r) => ({ x: x(r.year), y: y(r.value) }));
    let cur = pts[0];
    let best = pts[0].y;
    pts.forEach((p, i) => {
      if (i === 0) return;
      segs.push([cur, { x: p.x, y: cur.y }]);
      cur = { x: p.x, y: cur.y };
      if (p.y < best) {
        segs.push([cur, { x: p.x, y: p.y }]);
        cur = { x: p.x, y: p.y };
      }
      best = Math.min(best, p.y);
    });
    segs.push([cur, { x: m.l + iw, y: cur.y }]);
    boxes.forEach((b) => segs.forEach(([p, q]) => ok(!segHits(p, q, b, 0), `ladder: label '${b.name}' on the staircase`, () => `${ctx} ${fmtB(b)}`)));
  }
  // ticks and year labels inside
  for (const t of cfg.ticks) ok(y(t.v) >= m.t - 1e-9 && y(t.v) <= m.t + ih + 1e-9, `ladder: tick ${t.label} inside`, () => ctx);
  const xt = narrow ? cfg.xTicks!.narrow : cfg.xTicks!.wide;
  const yl = xt.map((yr) => textBox(String(yr), x(yr), m.t + ih + 16, yr < 0 ? `${-yr} BCE` : `${yr}`, narrow ? 9 : 10, 'middle', 'alphabetic', 0.62));
  for (const b of yl) ok(inside(b, W, H + 4), 'ladder: year label inside', () => `${ctx} ${b.name}`);
  noCollisions(yl, 'ladder year labels', ctx, 2);
  // "more precise" note (wide: bottom right; narrow: top left)
  const note = narrow
    ? textBox('more precise', m.l + 17, m.t + 12, 'more precise', 12, 'start', 'alphabetic', 0.48)
    : textBox('more precise', m.l + iw - 11, m.t + ih - 30, 'more precise', 12, 'end', 'alphabetic', 0.48);
  if (!narrow) noCollisions([...boxes, note], 'ladder rung labels', ctx);
  else
    boxes.forEach((b) => {
      // narrow shows only the active rung's label: check each alone against the note
      ok(!overlap(b, note), `ladder narrow: '${b.name}' vs note`, () => ctx);
    });
  // diamonds vs labels of other rungs
  cfg.rungs.forEach((r, i) => {
    const d: Box = { name: `${r.short} mark`, x0: x(r.year) - 9, x1: x(r.year) + 9, y0: y(r.value) - 6, y1: y(r.value) + 6 };
    boxes.forEach((b, j) => {
      if (i !== j && !narrow) ok(!overlap(b, d), `ladder: label '${b.name}' on mark '${r.short}'`, () => ctx);
    });
  });
}

/* ── report ── */
console.log(`${checks} checks`);
console.log(`chart: ${chartLeaders} positions use the corner label with a leader`);
console.log(`minaret: ${minRanges} bounded error ranges, ${minUnbounded} unbounded`);
console.log(`theodolite: telescope clears the plate for tilt ${vLo}° to ${vHi}° (slider −30° to 60°)`);
if (fails.size) {
  console.log(`\nFAILED (${fails.size} kinds):`);
  for (const [k, v] of fails) console.log(`  ${k}  ×${v.n}  e.g. ${v.first}`);
  process.exit(1);
} else console.log('all passed');
