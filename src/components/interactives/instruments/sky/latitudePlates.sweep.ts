/**
 * Sweep of the latitude post's new procedure plates and their shared reference.
 *
 *   node --experimental-strip-types --import ./src/components/interactives/instruments/sky/tsreg.mjs \
 *     src/components/interactives/instruments/sky/latitudePlates.sweep.ts
 *
 *   reference  latRef in every mode (sun, star, two cities, sight line) across latitudes,
 *              declinations, Polaris positions and widths: drawn φ/δ/h = numbers, rays parallel,
 *              lines and labels inside, labels placed and ≥ 5 apart, the sight line at the reading
 *   gnomon     shadow length = 1/tan(alt); noon is shortest and due north; the penumbra brackets
 *              the tip; the ruler mark/edge agree with onEdge; θ at the true tip = 90 − h; the
 *              circumference formula; drawings to one scale and inside
 *   kamal      Polaris's altitude: exact formula vs first-order correction (error < 0.01°);
 *              board angle = atan(h/d); the board's top edge at θ on the sky's scale; touching
 *              status matches the slider; cord reachable for every port and position; drawings in
 *   staffs     cross-staff angle = 2·atan(L/2x), the slider can frame the sun with each transom,
 *              its drawing's ends subtend θ at the eye; backstaff shadow offset sign/zero, the
 *              arcs sum to h at alignment, both sliders can reach it, drawings in
 *   astrolabe  spot on the pinhole ⇔ a + ψ = h; the reading at sea = h − ψ; the average of the
 *              readings is the plotted mean; drawings in
 *   zenith     every star in its circle exactly at the true latitude; the tilt shifts the field
 *              by tilt × k; the latitude result = PHI_ASTRO; drawings in
 * Widths: stacked views ~320 (390px phone) up to side-by-side columns ~440-600.
 */
import { near, ok, report } from '../../../../../scripts/plates/check.ts';
import { latRef, labelBox, arcPts, segHitsBox as segHitsBoxK, type RefSpec } from './latRefGeom.ts';
import * as Gn from './gnomonGeom.ts';
import * as K from './kamalGeom.ts';
import * as S from './staffGeom.ts';
import * as M from './marinerGeom.ts';
import * as Z from './zenithGeom.ts';
import { sunAltAz } from './noonProcGeom.ts';

const RAD = Math.PI / 180;
const WIDTHS = [320, 340, 360, 400, 440, 520, 600];
const inBox = (p: { x: number; y: number }, W: number, H: number, m = 0) =>
  p.x >= m - 1e-6 && p.y >= m - 1e-6 && p.x <= W - m + 1e-6 && p.y <= H - m + 1e-6;
const ang = (a: { x: number; y: number }, b: { x: number; y: number }) => Math.atan2(-(b.y - a.y), b.x - a.x) / RAD;

/* ───────────────────────────── reference ───────────────────────────── */

function checkRef(spec: RefSpec, W: number, ctx: string) {
  const g = latRef(spec, W);
  const c = `${ctx} W=${W}`;
  ok(near(g.drawnPhi, spec.phi, 1e-9), 'ref scale: drawn φ', () => c);
  ok(Math.abs(ang(g.obs, g.sunFrom) - g.dir) < 1e-6, 'ref scale: ray direction', () => `${c} ${ang(g.obs, g.sunFrom)} vs ${g.dir}`);
  if (spec.src.kind === 'sun') ok(near(g.drawnDec, spec.src.dec, 1e-6), 'ref scale: sun ray at δ', () => c);
  if (g.visible && !g.two) {
    const want = spec.src.kind === 'sun' ? 90 - Math.abs(spec.phi - spec.src.dec) : spec.src.alt;
    ok(near(g.drawnH, want, 1e-6), 'ref scale: drawn h', () => `${c} ${g.drawnH} vs ${want}`);
  }
  if (spec.src.kind === 'star') ok(g.bearing === 'N', 'ref: star to the north', () => c);
  for (const r of g.rays) {
    if (Math.hypot(r.a.x - r.b.x, r.a.y - r.b.y) > 1) ok(Math.abs(ang(r.b, r.a) - g.dir) < 1e-6, 'ref scale: rays parallel', () => c);
    ok(near(Math.hypot(r.b.x - g.cx, r.b.y - g.cy), g.Re, 1e-6), 'ref: rays end on the Earth', () => c);
  }
  if (g.two) {
    // θ at the centre = angle between the two places = the angle at you between zenith and sun
    const th = spec.phi - spec.second!.phi;
    const atYou = spec.phi - g.dir;
    ok(near(th, atYou, 1e-9), 'ref two-city: θ at centre = θ at you (sun over Syene)', () => `${c} ${th} vs ${atYou}`);
    ok(Math.abs(ang(g.obs2!, g.sun2!) - g.dir) < 1e-6, 'ref two-city: Syene ray parallel', () => c);
  }
  if (spec.sight != null) {
    // the sight line rises spec.sight above the horizon on the body's side
    const s = g.sightDir!;
    const rel = g.bearing === 'S' ? s - g.horDir : g.horDir - s;
    ok(near(rel, spec.sight, 1e-9), 'ref: sight line at the reading', () => c);
    ok(inBox(g.sightEnd!, W, g.H), 'ref bounds: sight line', () => c);
  }
  for (const sg of g.segs) ok(inBox(sg.a, W, g.H) && inBox(sg.b, W, g.H), `ref bounds: line ${sg.id.replace(/\d/, '')}`, () => `${c} ${JSON.stringify(sg)}`);
  for (const pts of g.arcs) ok(pts.every((p) => inBox(p, W, g.H)), 'ref bounds: arcs', () => c);
  for (const [id, l] of Object.entries(g.labels)) {
    ok(l.placed, `ref labels: ${id} placed`, () => `${c} ${spec.src.kind}`);
    const b = labelBox(l);
    ok(b.x0 >= 0 && b.y0 >= 0 && b.x1 <= W && b.y1 <= g.H, `ref bounds: label ${id}`, () => c);
  }
  const ls = Object.values(g.labels).map((l) => ({ id: l.id, b: labelBox(l) }));
  for (let i = 0; i < ls.length; i++)
    for (let j = i + 1; j < ls.length; j++) {
      const A = ls[i].b;
      const B = ls[j].b;
      ok(A.x1 + 5 <= B.x0 || B.x1 + 5 <= A.x0 || A.y1 + 5 <= B.y0 || B.y1 + 5 <= A.y0, `ref labels: ${ls[i].id}/${ls[j].id} apart`, () => c);
    }
  ok(JSON.stringify(latRef(spec, W)) === JSON.stringify(g), 'ref world: stable', () => c);
}

// every reference the post draws, across each plate's parameters
for (const W of WIDTHS) {
  checkRef({ phi: Gn.ALEX, src: { kind: 'sun', dec: Gn.DEC }, second: { phi: Gn.SYENE, name: 'Syene' }, you: 'Alexandria' }, W, 'gnomon');
  for (const p of K.PORTS)
    for (let lha = 0; lha < 360; lha += 15) {
      const alt = K.polarisAlt(p.lat, lha);
      for (const d of [K.D_MIN, 20, 30, K.D_MAX, K.cordFor(alt)])
        checkRef({ phi: p.lat, src: { kind: 'star', alt }, visible: alt > 0, sight: K.thetaOf(d) }, W, `kamal ${p.name} lha=${lha} d=${d}`);
      checkRef({ phi: p.lat, src: { kind: 'star', alt }, visible: alt > 0 }, W, `kamal ${p.name} lha=${lha}`);
    }
  for (const x of [S.X_MIN, 30, 60, S.X_MAX]) for (const t of S.TRANSOMS) checkRef({ phi: S.IST.lat, src: { kind: 'sun', dec: S.DEC }, sight: S.csTheta(t.L, x) }, W, `cross L=${t.L} x=${x}`);
  for (let h = 10; h <= 90; h += 5) checkRef({ phi: S.IST.lat, src: { kind: 'sun', dec: S.DEC }, sight: h }, W, `backstaff h=${h}`);
  for (const amp of [0, 0.5, M.SWING_MAX]) for (let n = 1; n <= M.N_READ; n++) checkRef({ phi: M.IST.lat, src: { kind: 'sun', dec: M.DEC }, sight: M.readings(amp, n).mean }, W, `astrolabe amp=${amp} n=${n}`);
  checkRef({ phi: Z.PHI_GPS, src: { kind: 'sun', dec: Z.PHI_GPS }, srcName: 'a star overhead', visible: false }, W, 'zenith');
}

/* ───────────────────────────── gnomon ───────────────────────────── */

{
  ok(near(Gn.NOON_ALT, 90 - (Gn.ALEX - Gn.SYENE), 1e-12), 'gnomon: noon altitude', () => '');
  ok(Gn.TIP_IN < Gn.TIP && Gn.TIP < Gn.TIP_OUT, 'gnomon: penumbra brackets the tip', () => '');
  ok(near(Gn.thetaOf(Gn.TIP * 100), Gn.ALEX - Gn.SYENE, 1e-9), 'gnomon: θ at the tip = φA − φS', () => `${Gn.thetaOf(Gn.TIP * 100)}`);
  ok(Gn.TIP * 100 > Gn.S_MIN && Gn.TIP_OUT * 100 < Gn.S_MAX && Gn.TIP_IN * 100 > Gn.S_MIN, 'gnomon: ruler covers the blur', () => '');
  ok(near(Gn.circumference(7.2, 157.5).km, 250000 * 0.1575, 1e-9), 'gnomon: circumference in km', () => '');
  ok(near(Gn.circumference(7.2, 176.4).stadia, 250000, 1e-12), 'gnomon: 7.2° → 250,000 stadia', () => '');
  // noon is the shortest shadow and points due north
  const nn = Gn.tipOffset(12);
  ok(Math.abs(nn.e) < 1e-9 && nn.n > 0, 'gnomon: noon shadow due north', () => JSON.stringify(nn));
  ok(near(nn.L, Gn.TIP, 1e-9), 'gnomon: noon shadow = TIP', () => '');
  for (let t = Gn.T_MIN; t <= Gn.T_MAX + 1e-9; t += 1 / 60) {
    const o = Gn.tipOffset(t);
    ok(o.L >= Gn.TIP - 1e-12, 'gnomon: noon is the shortest', () => `t=${t}`);
    ok(near(o.L, 1 / Math.tan(sunAltAz(Gn.ALEX, Gn.DEC, t).alt * RAD), 1e-9), 'gnomon: L = g/tan(alt)', () => `t=${t}`);
    ok(o.alt > 0, 'gnomon: sun up through the slider', () => `t=${t}`);
  }
  for (const W of WIDTHS) {
    for (let t = Gn.T_MIN; t <= Gn.T_MAX + 1e-9; t += 0.25) {
      const tt = Gn.snapTime(t);
      const g = Gn.groundScene(tt, W);
      const c = `W=${W} t=${tt}`;
      ok(near(Math.hypot(g.tip.x - g.foot.x, g.tip.y - g.foot.y), g.now.L * g.k, 1e-9), 'gnomon scale: shadow length', () => c);
      for (const p of [g.tip, g.tipIn, g.tipOut, ...g.track]) ok(inBox(p, W, g.H, 2), 'gnomon bounds: ground', () => c);
      const sb = labelBox({ x: W / 2, y: g.H - 10, text: Gn.groundSay(tt), size: 12, anchor: 'middle' });
      ok(sb.x0 >= 2 && sb.x1 <= W - 2, 'gnomon bounds: status', () => `${c} ${Gn.groundSay(tt)}`);
      ok(sb.y0 > g.foot.y + 18, 'gnomon labels: status below the stick label', () => c);
      ok(!/e[+-]/.test(Gn.groundSay(tt)), 'gnomon format', () => c);
    }
    for (let s = Gn.S_MIN; s <= Gn.S_MAX + 1e-9; s += Gn.S_STEP) {
      const r = Gn.rulerScene(s, W);
      const c = `W=${W} s=${s}`;
      ok(r.xIn < r.xTip && r.xTip < r.xOut, 'gnomon ruler: blur order', () => c);
      ok(r.xMark >= r.x0 - 1e-9 && r.xMark <= r.x1 + 1e-9, 'gnomon bounds: mark', () => c);
      ok(Gn.onEdge(s) === (r.xMark >= r.xIn - 1e-6 && r.xMark <= r.xOut + 1e-6), 'gnomon honesty: edge status = drawing', () => c);
      const sb = labelBox({ x: W / 2, y: r.H - 10, text: Gn.rulerSay(s), size: 12, anchor: 'middle' });
      ok(sb.x0 >= 2 && sb.x1 <= W - 2, 'gnomon bounds: ruler status', () => `${c} ${Gn.rulerSay(s)}`);
      ok(sb.y0 > r.rulerY + 34, 'gnomon labels: status below the ruler numbers', () => c);
      const tr = Gn.triangleScene(s, W);
      ok(near(Math.atan((tr.tip.x - tr.base.x) / (tr.base.y - tr.top.y)) / RAD, tr.theta, 1e-9), 'gnomon scale: triangle angle = θ', () => c);
      for (const p of [tr.top, tr.tip, tr.rayFrom]) ok(inBox(p, W, tr.H, 2), 'gnomon bounds: triangle', () => c);
      const lb = labelBox({ x: tr.tip.x + 8, y: tr.base.y - 8, text: `s = ${s.toFixed(2)} cm`, size: 13, anchor: 'start' });
      const title = labelBox({ x: 14, y: 14, text: 'SIDE ON, TO SCALE', size: 12 * 0.78 * 1.25, anchor: 'start' });
      const sun = labelBox({ x: tr.sunLabel.x, y: tr.sunLabel.y - 4, text: 'sunlight', size: 11, anchor: 'start' });
      const gl = labelBox({ x: tr.base.x - 8, y: (tr.base.y + tr.top.y) / 2 - 4, text: 'g = 100 cm', size: 13, anchor: 'end' });
      ok(!(title.x0 < sun.x1 && sun.x0 < title.x1 && title.y0 < sun.y1 && sun.y0 < title.y1), 'gnomon labels: sunlight clear of the title', () => c);
      ok(gl.x0 >= 2, 'gnomon bounds: g label', () => c);
      ok(!segHitsBoxK(tr.rayFrom, tr.tip, sun), 'gnomon labels: sunlight off the ray', () => c);
      ok(lb.x1 <= W - 2, 'gnomon bounds: s label', () => c);
    }
    // the step-1 snap honest: the slider at a value within 3 min shows noon
    ok(Gn.snapTime(12.04) === 12 && Gn.snapTime(11.9) !== 12, 'gnomon: noon snap', () => '');
  }
  // the reachable mark: some ruler step lands in the blur
  let hit = 0;
  for (let s = Gn.S_MIN; s <= Gn.S_MAX + 1e-9; s += Gn.S_STEP) if (Gn.onEdge(+s.toFixed(4))) hit++;
  ok(hit >= 3, 'gnomon: the blur is reachable on the slider', () => `${hit}`);
  ok(Gn.onEdge(Math.round((Gn.TIP * 100) / Gn.S_STEP) * Gn.S_STEP), 'gnomon: skip lands on the edge', () => '');
}

/* ───────────────────────────── kamal ───────────────────────────── */

for (const p of K.PORTS)
  for (let lha = 0; lha < 360; lha += 5) {
    const c = `${p.name} lha=${lha}`;
    const alt = K.polarisAlt(p.lat, lha);
    ok(Math.abs(K.latOf(alt, lha) - p.lat) < 0.01, 'kamal maths: first-order correction good to 0.01°', () => `${c} ${K.latOf(alt, lha) - p.lat}`);
    ok(alt > 0, 'kamal: Polaris up at every port', () => c);
    const d = K.cordFor(alt);
    ok(d >= K.D_MIN && d <= K.D_MAX, 'kamal: cord reachable', () => `${c} d=${d}`);
    // the slider (step 0.05) can put the edge within TOUCH
    const ds = Math.round(d / K.D_STEP) * K.D_STEP;
    ok(Math.abs(K.thetaOf(ds) - alt) <= K.TOUCH, 'kamal: slider can touch', () => `${c} ${K.thetaOf(ds) - alt}`);
    for (const W of [320, 400, 520]) {
      for (const dd of [K.D_MIN, 15, 30, 45, K.D_MAX, ds]) {
        const g = K.skyScene(p, lha, dd, W);
        const cc = `${c} W=${W} d=${dd}`;
        ok(near(g.horizonY - g.star.y, alt * g.k, 1e-9), 'kamal scale: star height', () => cc);
        ok(near(g.horizonY - g.boardTopTrue, K.thetaOf(dd) * g.k, 1e-9), 'kamal scale: board top at θ', () => cc);
        ok(g.touching === (Math.abs(K.thetaOf(dd) - alt) <= K.TOUCH), 'kamal honesty: touching', () => cc);
        ok(inBox(g.star, W, g.H, 4), 'kamal bounds: star', () => cc);
        ok(g.board.y0 >= 4 - 1e-9, 'kamal bounds: board top', () => `${cc} ${g.board.y0}`);
        ok(g.clipped || near(g.board.y0, g.boardTopTrue, 1e-12), 'kamal honesty: board drawn at θ unless marked clipped', () => cc);
        if (g.clipped) ok(!g.touching, 'kamal honesty: clipped board never shows touching', () => cc);
        const sb = labelBox({ x: W / 2, y: g.H - 10, text: K.skySay(g), size: 12, anchor: 'middle' });
        ok(sb.x0 >= 2 && sb.x1 <= W - 2, 'kamal bounds: status', () => `${cc} ${K.skySay(g)}`);
        const pl = labelBox({ x: g.star.x + 10, y: g.star.y - 6, text: 'Polaris', size: 12, anchor: 'start' });
        ok(pl.x1 <= W - 2 && pl.y0 >= 0, 'kamal bounds: Polaris label', () => cc);
        const sd = K.sideScene(dd, W);
        ok(near(ang(sd.eye, sd.boardTop), K.thetaOf(dd), 1e-9), 'kamal scale: side angle = θ', () => cc);
        ok(inBox(sd.boardTop, W, sd.H, 2) && inBox(sd.far, W, sd.H, 0), 'kamal bounds: side', () => cc);
        const sl = Object.values(sd.labels).map((l) => ({ ...labelBox(l), id: l.text }));
        for (const b of sl) ok(b.x0 >= 2 && b.x1 <= W - 2 && b.y0 >= 24 && b.y1 <= sd.H - 24, 'kamal bounds: side labels', () => `${cc} ${b.id}`);
        for (let i = 0; i < sl.length; i++)
          for (let j = i + 1; j < sl.length; j++) {
            const A = sl[i];
            const B = sl[j];
            ok(A.x1 + 3 <= B.x0 || B.x1 + 3 <= A.x0 || A.y1 + 3 <= B.y0 || B.y1 + 3 <= A.y0, 'kamal labels: side labels apart', () => `${cc} ${A.id}/${B.id}`);
          }
        ok(!segHitsBoxK(sd.eye, sd.boardTop, labelBox(sd.labels.h), 0) && !segHitsBoxK(sd.eye, { x: sd.boardX, y: sd.eye.y }, labelBox(sd.labels.d), 0), 'kamal labels: side labels off the lines', () => cc);
        ok(sd.eye.y - sd.boardTop.y >= 26, 'kamal: board tall enough to read (drawn to scale)', () => cc);
      }
      const ci = K.circleScene(p, lha, W);
      ok(near((ci.cy - ci.star.y) / ci.kd, P_OFFcos(lha), 1e-9), 'kamal scale: Polaris above the pole = p cos t', () => c);
      ok(near(ci.corr, -P_OFFcos(lha), 1e-12), 'kamal: correction sign', () => c);
      ok(inBox(ci.star, W, ci.H, 10), 'kamal bounds: circle', () => c);
      const cu = K.curveScene(p, W);
      for (const q of cu.pts) ok(inBox(q, W, cu.H, 2), 'kamal bounds: curve', () => c);
      if (cu.hereLabel) {
        const lb = labelBox(cu.hereLabel);
        ok(lb.x0 >= 2 && lb.x1 <= W - 2 && lb.y1 <= cu.y1 - 2, 'kamal bounds: curve label', () => `${c} W=${W}`);
        ok(!cu.pts.some((q, i) => i > 0 && segHitsBoxK(cu.pts[i - 1], q, lb, 0)), 'kamal labels: curve label off the curve', () => `${c} W=${W}`);
      }
      const ub = labelBox(cu.unit);
      ok(!cu.yTicks.some((t) => labelBox({ x: cu.x0 - 6, y: t.y, text: `${t.d}`, size: 10, anchor: 'end' }).x1 > ub.x0), 'kamal labels: cm unit clear of ticks', () => '');
      ok(!cu.pts.some((q, i) => i > 0 && segHitsBoxK(cu.pts[i - 1], q, ub, 0)), 'kamal labels: cm unit off the curve', () => `${c} W=${W}`);
    }
  }
function P_OFFcos(lha: number) {
  return K.P_OFF * Math.cos(lha * RAD);
}

/* ───────────────────────────── cross-staff and backstaff ───────────────────────────── */

for (const t of S.TRANSOMS) {
  const x = S.csX(t.L, S.H_SUN);
  ok(x >= S.xMinFor(t.L) && x <= S.X_MAX, `cross: ${t.name} frames the sun on the staff`, () => `x=${x}`);
  const xs = Math.round(x / S.X_STEP) * S.X_STEP;
  ok(Math.abs(S.csTheta(t.L, xs) - S.H_SUN) <= S.CS_TOL, `cross: slider reaches the sun (${t.name})`, () => `${S.csTheta(t.L, xs) - S.H_SUN}`);
  for (let xx = S.xMinFor(t.L); xx <= S.X_MAX + 1e-9; xx += 0.5)
    for (const W of WIDTHS) {
      const c = `L=${t.L} x=${xx} W=${W}`;
      const sd = S.csSide(t.L, xx, W);
      const sub = Math.abs(ang(sd.eye, sd.b) - ang(sd.eye, sd.a));
      ok(near(sub, S.csTheta(t.L, xx), 1e-9), 'cross scale: transom subtends θ at the eye', () => c);
      ok(near(ang(sd.eye, sd.a), 0, 1e-9), 'cross: lower end on the horizon (level)', () => `${c} ${ang(sd.eye, sd.a)}`);
      for (const p of [sd.a, sd.b, sd.end]) ok(inBox(p, W, sd.H, 2), 'cross bounds: side', () => c);
      const lb = sd.labels.map((l) => ({ ...labelBox(l), id: l.id }));
      const lines = [[sd.eye, sd.end], [sd.a, sd.b], [sd.eye, sd.a], [sd.eye, sd.b]] as const;
      for (const l of sd.labels) ok(l.placed, `cross labels: ${l.id.replace(/\d+/, '')} placed`, () => c);
      for (const b of lb) {
        ok(b.x0 >= 2 && b.x1 <= W - 2 && b.y0 >= 2 && b.y1 <= sd.H - 26, 'cross bounds: side labels', () => `${c} ${b.id}`);
        if (b.id !== 'eye') ok(!lines.some(([p, q]) => segHitsBoxK(p, q, b, 0)), 'cross labels: side label off the lines', () => `${c} ${b.id}`);
        ok(!arcPts(sd.eye.x, sd.eye.y, sd.arcR, 0, sd.theta).some((q, i, a) => i > 0 && segHitsBoxK(a[i - 1], q, b, 0)), 'cross labels: off the h arc', () => `${c} ${b.id}`);
      }
      for (let i = 0; i < lb.length; i++)
        for (let j = i + 1; j < lb.length; j++) {
          const A = lb[i];
          const B = lb[j];
          ok(A.x1 + 3 <= B.x0 || B.x1 + 3 <= A.x0 || A.y1 + 3 <= B.y0 || B.y1 + 3 <= A.y0, 'cross labels: side labels apart', () => `${c} ${A.id}/${B.id}`);
        }
      const g = S.csSky(t.L, xx, W);
      ok(g.on === (Math.abs(S.csTheta(t.L, xx) - S.H_SUN) <= S.CS_TOL), 'cross honesty: on', () => c);
      if (!g.clipped) ok(near(g.horizonY - g.upper.y, g.theta * g.k, 1e-9), 'cross scale: upper end at θ', () => c);
      ok(g.upper.y >= 0, 'cross bounds: upper end', () => c);
      ok(near(g.horizonY - g.sun.y, S.H_SUN * g.k, 1e-9), 'cross scale: sun at h', () => c);
      const sb = labelBox({ x: W / 2, y: g.H - 10, text: S.csSay(g), size: 12, anchor: 'middle' });
      ok(sb.x0 >= 2 && sb.x1 <= W - 2, 'cross bounds: status', () => `${c} ${S.csSay(g)}`);
    }
}
ok(near(S.latFrom(S.H_SUN), S.IST.lat, 1e-9), 'staffs: exact h gives Istanbul', () => '');
for (let a60 = S.A60_MIN; a60 <= S.A60_MAX; a60 += S.A60_STEP) {
  const okp = S.bsPresetOk(a60);
  ok(okp === (S.H_SUN - a60 >= 0 && S.H_SUN - a60 <= 30), 'backstaff: preset rule', () => `${a60}`);
  if (okp) {
    const a30 = Math.round((S.H_SUN - a60) / S.A30_STEP) * S.A30_STEP;
    ok(Math.abs(a60 + a30 - S.H_SUN) <= S.BS_TOL, 'backstaff: slider reaches alignment', () => `${a60}`);
    ok(Math.abs(S.bsOffMm(a60, S.H_SUN - a60)) < 1e-9, 'backstaff: aligned ⇒ shadow on the slit', () => `${a60}`);
  }
  for (let a30 = S.A30_MIN; a30 <= S.A30_MAX + 1e-9; a30 += 0.25)
    for (const W of WIDTHS) {
      const c = `a60=${a60} a30=${a30} W=${W}`;
      const v = S.bsVane(a60, a30, W);
      ok(Math.sign(v.offMm) === Math.sign(a60 + a30 - S.H_SUN) || Math.abs(a60 + a30 - S.H_SUN) < 1e-9, 'backstaff: too big an angle ⇒ shadow above', () => c);
      ok(near(v.slitY - v.edgeY, v.offMm * v.kmm, 1e-9), 'backstaff scale: edge offset', () => c);
      ok(v.on === Math.abs(a60 + a30 - S.H_SUN) <= S.BS_TOL, 'backstaff honesty: on', () => c);
      ok(v.off === null ? Math.abs(v.edgeY - v.cy) <= (v.vane.y1 - v.vane.y0) / 2 * (6.5 / 6) + 1e-9 : true, 'backstaff: drawn edge within the vane or flagged', () => c);
      ok(!!v.off === Math.abs(v.offMm) > 6.5, 'backstaff honesty: off flag', () => c);
      const sb = labelBox({ x: W / 2, y: v.H - 10, text: S.bsSay(v), size: 12, anchor: 'middle' });
      ok(sb.x0 >= 2 && sb.x1 <= W - 2, 'backstaff bounds: status', () => `${c} ${S.bsSay(v)}`);
      const sd = S.bsSide(a60, a30, W);
      for (const p of [sd.O, sd.sightVane, sd.staffEnd, sd.shadowVane, sd.sunIn, sd.eye]) ok(inBox(p, W, sd.H, 2), 'backstaff bounds: side', () => `${c} ${JSON.stringify(p)}`);
      // the shadow ray through the vane hits O exactly when aligned
      if (Math.abs(a60 + a30 - S.H_SUN) < 1e-9) ok(near(sd.shadowHit.y, sd.O.y, 1e-6), 'backstaff scale: aligned shadow reaches the slit', () => c);
      ok(near(ang(sd.O, sd.shadowVane), a60 + a30, 1e-9), 'backstaff scale: shadow vane at a60 + a30', () => c);
      // the arcs belong to the instrument: they turn with the staff, and each carries its vane
      ok(near(sd.arc60.from, a30, 1e-12) && near(sd.arc60.to - sd.arc60.from, 60, 1e-12), 'backstaff frame: 60° arc starts at the staff', () => c);
      ok(near(sd.arc30.to, a30, 1e-12) && near(sd.arc30.to - sd.arc30.from, 30, 1e-12), 'backstaff frame: 30° arc ends at the staff', () => c);
      ok(a60 + a30 >= sd.arc60.from - 1e-9 && a60 + a30 <= sd.arc60.to + 1e-9, 'backstaff frame: shadow vane on the 60° arc', () => c);
      ok(0 >= sd.arc30.from - 1e-9 && 0 <= sd.arc30.to + 1e-9, 'backstaff frame: sight vane on the 30° arc', () => c);
      // the arcs fit the drawing
      for (const [r, a] of [[sd.r60, sd.arc60], [sd.r30, sd.arc30]] as const)
        for (const q of arcPts(sd.O.x, sd.O.y, r, a.from, a.to)) ok(inBox(q, W, sd.H, 2), 'backstaff bounds: arcs', () => `${c} ${JSON.stringify(q)}`);
    }
}

/* ───────────────────────────── mariner's astrolabe ───────────────────────────── */

for (const amp of [0, 0.05, 0.25, 0.5, 0.75, M.SWING_MAX])
  for (let n = 1; n <= M.N_READ; n++) {
    const { r, mean } = M.readings(amp, n);
    ok(r.every((v, i) => near(v, M.H_SUN - M.swingAt(amp, i), 1e-12)), 'astrolabe: reading = h − ψ', () => `${amp} ${n}`);
    ok(Math.abs(mean - M.H_SUN) <= amp + 1e-9, 'astrolabe: mean within the swing', () => `${amp} ${n}`);
    ok(r.every((v, i) => Math.abs(M.spotMm(v, M.swingAt(amp, i))) < 1e-9), 'astrolabe: each reading puts the spot on the pinhole', () => '');
    for (const W of WIDTHS) {
      const g = M.scatterScene(amp, n, W);
      for (const d of g.dots) ok(d.x >= g.x0 - 1e-9 && d.x <= g.x1 + 1e-9, 'astrolabe bounds: dots', () => `${amp} ${n} ${W}`);
      ok(near(g.xMean - g.xTrue, (mean - M.H_SUN) * 60 * g.k, 1e-9), 'astrolabe scale: mean marker', () => '');
      const lb = labelBox({ x: g.xMean, y: g.y + 48, text: `average +${Math.abs((mean - M.H_SUN) * 60).toFixed(0)}′`, size: 12, anchor: 'middle' });
      ok(lb.x0 >= 0 && lb.x1 <= W, 'astrolabe bounds: average label', () => `${amp} ${n} ${W}`);
      const sb = labelBox({ x: W / 2, y: g.H - 10, text: `each dot: one reading, off by the swing (±${(amp * 60).toFixed(0)}′)`, size: 12, anchor: 'middle' });
      ok(sb.x0 >= 0 && sb.x1 <= W, 'astrolabe bounds: scatter status', () => `W=${W}`);
    }
  }
for (let a = M.A_MIN; a <= M.A_MAX + 1e-9; a += 0.5)
  for (const psi of [-M.SWING_MAX, -1, 0, 1, M.SWING_MAX])
    for (const W of WIDTHS) {
      const c = `a=${a} ψ=${psi} W=${W}`;
      const g = M.ringScene(a, psi, W);
      ok(near(ang(g.ali.a, g.ali.b), a + psi, 1e-9), 'astrolabe scale: alidade at a + ψ', () => c);
      ok(g.on === (Math.abs(a + psi - M.H_SUN) <= M.ALI_TOL), 'astrolabe honesty: on', () => c);
      for (const p of [g.ali.a, g.ali.b, g.shackle, ...g.rays.flatMap((r) => [r.a, r.b])]) ok(inBox(p, W, g.H, 0), 'astrolabe bounds: ring', () => `${c} ${JSON.stringify(p)}`);
      ok(g.cy + g.R + 7 < g.H - 22, 'astrolabe bounds: ring above the status', () => c);
      const ll = labelBox(g.levelLabel);
      ok(ll.x0 >= 2 && ll.x1 < g.cx - g.R - 4, 'astrolabe labels: level left of the ring', () => c);
      for (const n of g.nums) {
        const nb = labelBox({ x: n.x, y: n.y, text: `${n.d}°`, size: 10, anchor: 'middle' });
        ok(nb.x0 >= 0 && nb.x1 <= W && nb.y0 >= 0, 'astrolabe bounds: scale numbers', () => `${c} ${n.d}`);
        ok(Math.hypot(n.x - g.shackle.x, n.y - g.shackle.y) > 18, 'astrolabe labels: numbers clear of the shackle', () => `${c} ${n.d}`);
      }
      const sb = labelBox({ x: W / 2, y: g.H - 10, text: M.ringSay(a, psi), size: 12, anchor: 'middle' });
      ok(sb.x0 >= 2 && sb.x1 <= W - 2, 'astrolabe bounds: status', () => `${c} ${M.ringSay(a, psi)}`);
      const pv = M.plateScene(a, psi, W);
      ok(Math.sign(pv.cy - pv.spot.y) === Math.sign(a + psi - M.H_SUN) || Math.abs(a + psi - M.H_SUN) < 1e-9, 'astrolabe: steep alidade ⇒ spot above', () => c);
      ok(M.ringSay(a, psi).includes('above') === (a + psi - M.H_SUN > M.ALI_TOL && Math.abs(M.spotMm(a, psi)) <= 6), 'astrolabe honesty: status says above when drawn above', () => c);
    }
{
  const as = Math.round(M.H_SUN / M.A_STEP) * M.A_STEP;
  ok(Math.abs(as - M.H_SUN) <= M.ALI_TOL, 'astrolabe: slider reaches the sun', () => '');
}

/* ───────────────────────────── zenith camera ───────────────────────────── */

{
  const gTrue = Math.round(Z.truthG / Z.G_STEP) * Z.G_STEP;
  ok(Z.truthG >= Z.G_MIN && Z.truthG <= Z.G_MAX, 'zenith: truth on the slider', () => `${Z.truthG}`);
  ok(Math.abs((Z.PHI_ASTRO - Z.guessLat(gTrue)) * 3600) <= Z.MATCH_TOL, 'zenith: slider reaches the match', () => '');
  ok(near((Z.PHI_ASTRO - Z.PHI_GPS) * 3600, Z.DEFLECT, 1e-6), 'zenith: deflection', () => '');
  for (const W of WIDTHS)
    for (let tilt = Z.TILT_MIN; tilt <= Z.TILT_MAX; tilt += Z.TILT_STEP)
      for (const g of [null, Z.G_MIN, gTrue, Z.G_MAX]) {
        const f = Z.fieldScene(tilt, g, W);
        const c = `W=${W} tilt=${tilt} g=${g}`;
        ok(f.stars.every((s, i) => near(f.cy - s.y, (Z.STARS[i].n - tilt) * f.k, 1e-9)), 'zenith scale: stars shift by the tilt', () => c);
        if (g === gTrue && tilt === 0) ok(f.cat.every((q, i) => Math.hypot(q.x - f.stars[i].x, q.y - f.stars[i].y) <= Z.MATCH_TOL * f.k + 1e-9), 'zenith: match puts stars in circles', () => c);
        ok(f.cy + f.size / 2 < f.H - 22 && f.cy - f.size / 2 >= 0, 'zenith bounds: field', () => c);
        ok(f.size > 200, 'zenith: field big enough', () => c);
        const b = Z.bubbleScene(tilt, W);
        ok(Math.abs(b.bubble.x - b.cx) + 14 <= b.len / 2, 'zenith bounds: bubble in the vial', () => c);
      }
  ok(Z.dms1(41.01471) === '41°00′53.0″', 'zenith format: dms1', () => Z.dms1(41.01471));
}

report();
