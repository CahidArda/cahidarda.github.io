/**
 * Sweep of the earth procedure plates' geometry (round 2). Run from the repo root:
 *   node --experimental-strip-types src/components/interactives/instruments/earth/procedures.sweep.ts
 * The older plates (A-frame, cord, minaret/strait error regions, arcsecond chart) are swept by
 * ./geometry.sweep.ts. This one covers ./site.ts (the stable reference), levelGeom, rodGeom,
 * romanGeom, chainGeom, mapGeom, theoGeom, barGeom, edmGeom and the GPS numbers, and asserts:
 *   maths    values against independent formulas or brute force
 *   scale    drawn positions = values × the drawing's one scale
 *   world    the reference does not change with any plate parameter
 *   bounds   marks and labels inside the drawing
 *   honesty  "done"/"on" flags agree with the drawn state; slider ranges reach every target
 */
import { inside, near, noCollisions, ok, range, report, textBox } from '../../../../../scripts/plates/check.ts';
import { CREST_Z, FAR_X, LINE_H, LINE_SURFACE, ROAD_LEN, ROAD_PTS, SITE, SLOPE_RAD, groundZ, roadAt, siteScene } from './site.ts';
import { AFRAME_H_CM, CORD, RAD, aframeLayout, cordLayout } from './geometry.ts';
import { LEAN, SURFACES, crossbarScene, cordLevel, missMm, seked, slopeFromMiss, wallScene } from './levelGeom.ts';
import { ROD_ERR, SPAN, UNITS, measure, rodScene, type UnitId } from './rodGeom.ts';
import {
  BED_FALL_MM,
  CHORO,
  DROP_UNC,
  GROMA,
  SIGHT_D,
  TILT_TOL,
  TRUE_DROP,
  benchScene,
  farHit,
  gromaScene,
  looksLevel,
  measuredDrop,
  onB,
  poleHidden,
  sightScene,
  tiltOf,
  windMiss,
} from './romanGeom.ts';
import { CHAIN_M, chainCount, chainEnds, chainScene, linksScene } from './chainGeom.ts';
import { M_PER_IN, READ_STEP, ROAD_IN, chartScene, dialScene, readOk } from './mapGeom.ts';
import { ALPHA as A_TRUE, BASE, BETA as B_TRUE, EL_FAR, EL_NEAR, MIN_STEP, ON_TOL, PA, PB, PC, TOWER, diff, eyepiece, heightFrom, readingTo, snap, split, toDeg, widthFrom } from './theoGeom.ts';
import { ALPHA, BASELINE, L0, T, T0, X_STAR, baseScene, compLen, compScene, plainErr, plainRead, tongueMove } from './barGeom.ts';
import { MILE_FT, REV, TEETH, hodoLayout, hodometer } from './hodoGeom.ts';
import { D_TRUE, F_COARSE, F_FINE, N_TRUE, PHASE_STEP, PHI_COARSE, PHI_FINE, coarseD, distance, halfLambda, phaseOk, phaseScene } from './edmGeom.ts';

const WIDTHS = [326, 340, 360, 420, 520, 560, 640];

/* ── site: the stable reference ── */
for (const W of WIDTHS) {
  const g = siteScene(W);
  const ctx = `W=${W}`;
  ok(g.H > 0 && g.top > 20, 'site: sane height', () => ctx);
  // one horizontal scale for section and plan
  ok(near(g.names.M.x, g.names.pM.x) && near(g.names.A.x, g.names.pA.x) && near(g.names.C.x, g.names.pC.x), 'site: section and plan share x', () => ctx);
  ok(near((g.names.C.x - g.names.A.x) / g.k, SITE.WIDTH), 'site: strait drawn 700 m', () => ctx);
  ok(near((g.names.M.y - g.names.tip.y) / g.kz, SITE.MINARET_H), 'site: minaret drawn 76 m (×VE)', () => ctx);
  ok(near((g.names.pB.y - g.names.pA.y) / -g.k, SITE.B_Y), 'site: baseline drawn 200 m', () => ctx);
  ok(near((g.names.crest.y - g.names.A.y) / -g.kz, CREST_Z - SITE.QUAY_Z), 'site: road falls 35 m (×VE)', () => ctx);
  for (const [n, p] of Object.entries(g.names)) ok(p.x >= 0 && p.x <= W && p.y >= 0 && p.y <= g.H, `site: '${n}' inside`, () => `${ctx} ${p.x},${p.y}`);
  for (const p of g.road) ok(p.x >= 0 && p.x <= W && p.y >= g.planTop - 0.5 && p.y <= g.planTop + g.planH + 0.5, 'site: road inside the plan', () => ctx);
  const L = g.labels;
  const boxes = [
    textBox('section', L.section.x, L.section.y, L.section.text, L.section.size * 0.78, 'end', 'alphabetic', 0.75),
    textBox('plan', L.plan.x, L.plan.y, L.plan.text, L.plan.size * 0.78, 'end', 'alphabetic', 0.75),
    textBox('M', L.M.x, L.M.y, 'M', 12),
    textBox('A', L.A.x, L.A.y, 'A', 12, 'end'),
    textBox('C', L.C.x, L.C.y, 'C', 12),
    textBox('pA', L.pA.x, L.pA.y, 'A', 12, 'end'),
    textBox('pB', L.pB.x, L.pB.y, 'B', 12, 'end'),
    textBox('pC', L.pC.x, L.pC.y, 'C', 12),
    textBox('pM', L.pM.x, L.pM.y, 'M', 12),
    textBox('legend', 10, g.H - 8, `M minaret (${SITE.MINARET_H} m) · A, B near quay · C far quay`, 10, 'start', 'alphabetic', 0.5),
  ];
  for (const b of boxes) ok(inside(b, W, g.H), `site: label '${b.name}' inside`, () => `${ctx} ${b.x0.toFixed(0)}..${b.x1.toFixed(0)}`);
  noCollisions(boxes, 'site: labels', ctx);
}
ok(near(SLOPE_RAD, Math.atan(1 / 8)), 'site: road is 1 in 8', () => `${SLOPE_RAD}`);
ok(near(groundZ(SITE.FLAT_X) - groundZ(SITE.CREST_X), 0.025), 'site: channel falls 2.5 cm in 100 m', () => '');
ok(LINE_SURFACE > LINE_H && ROAD_LEN > LINE_SURFACE, 'site: road > surface > level line', () => `${ROAD_LEN} ${LINE_SURFACE}`);
for (const f of range(0, 1, 0.001)) {
  const a = roadAt(f);
  ok(Number.isFinite(a.p.x) && Number.isFinite(a.dir), 'site: roadAt finite', () => `${f}`);
}
ok(near(roadAt(1).p.x, SITE.SHORE_X) && near(roadAt(0).p.x, 0), 'site: road ends at M and A', () => '');
ok(ROAD_PTS.every((r, i) => i === 0 || r.s >= ROAD_PTS[i - 1].s), 'site: road length monotone', () => '');

/* the "known" tag at the top of the reference: longest strings each plate can print */
for (const tag of [
  'plain bars, -5 °C: strait reads 700.161 m',
  'known: the slope, a gradient, a right angle',
  'known: a right angle, the slope of the road',
  'known: the channel falls -10.00 cm in 94 m',
  'rover 25 km out: ±33.0 mm across, ±40.0 mm up',
  'known: up and down · unknown: how far',
  'known: the strait 700 m (from angles)',
]) {
  const w = tag.length * 6.9;
  for (const W of WIDTHS) ok(w <= (W - 16) * 1.12, 'site: known-tag fits (squeezed at most 12%)', () => `${tag} W=${W} ${w.toFixed(0)}`);
}

/* ── plate 1: level ── */
for (const W of WIDTHS) {
  for (const lean of range(LEAN.MIN, LEAN.MAX, LEAN.STEP)) {
    const g = wallScene(W, lean);
    const ctx = `W=${W} lean=${lean}`;
    ok(near((g.lineX - g.faceFoot.x) / g.kx, g.gapFoot) && near(g.gapFoot, g.gapTop + lean), 'wall: foot gap drawn = top + lean', () => ctx);
    ok(g.plumb === (Math.abs(lean) <= LEAN.TOL), 'wall: plumb flag', () => ctx);
    ok(g.wallBack < Math.min(g.faceTop.x, g.faceFoot.x) - 4, 'wall: wall has thickness', () => ctx);
    for (const b of g.boxes) ok(inside(b, W, g.H), `wall: '${b.name}' inside`, () => `${ctx} ${b.x0}..${b.x1}`);
    noCollisions(g.boxes, 'wall: labels', ctx);
  }
  ok(range(LEAN.MIN, LEAN.MAX, LEAN.STEP).some((l) => Math.abs(l) <= LEAN.TOL), 'wall: slider reaches plumb', () => '');
  for (const s of SURFACES) {
    const g = crossbarScene(W, s.theta);
    const ctx = `W=${W} ${s.id}`;
    ok(near(Math.abs(g.sx - g.cx) / g.kmm, missMm(s.theta)), 'crossbar: drawn miss = h tan θ', () => ctx);
    ok(g.sx > g.cx - g.half && g.sx < g.cx + g.half, 'crossbar: string on the scale', () => ctx);
    ok(s.theta >= 0 || g.sx > g.cx, 'crossbar: falls right → string right of centre (downhill)', () => ctx);
    const A = aframeLayout(W, s.theta);
    ok(Math.sign(A.hit.x - A.mark.x) === Math.sign(g.sx - g.cx) || Math.abs(s.theta) < 1e-9, 'crossbar: same side as the frame drawing', () => `${ctx} ${A.hit.x - A.mark.x}`);
    for (const b of A.boxes) ok(inside(b, W, A.Hs), `A-frame on the site surfaces: '${b.name}' inside`, () => `${ctx} ${b.x0.toFixed(0)}..${b.x1.toFixed(0)} H=${A.Hs}`);
  }
  for (const far of range(CORD.MIN, CORD.MAX, 0.01)) {
    const g = cordLayout(W, far);
    ok(g.boxes.every((b) => inside(b, W, g.Hs)), 'cord: labels inside', () => `W=${W} far=${far}`);
    ok(g.a.y > 0 && g.b.y > 0 && g.tip.y < g.Hs, 'cord: cord and triangle inside', () => `W=${W} far=${far}`);
    ok(g.level === cordLevel(far), 'cord: level flag agrees', () => `${far}`);
  }
}
const road = SURFACES.find((s) => s.id === 'road')!;
ok(near(Math.tan(-road.theta * RAD), 1 / 8), 'level: road surface is 1 in 8', () => '');
ok(near(slopeFromMiss(missMm(road.theta)), Math.abs(road.theta)), 'level: arctan(x/h) inverts x', () => '');
ok(near(seked(Math.atan(1 / 8) / RAD), 56), 'level: seked of 1 in 8 is 56 palms', () => `${seked(Math.atan(1 / 8) / RAD)}`);
ok(Math.abs(missMm(road.theta) - AFRAME_H_CM * 10 / 8) < 1e-9, 'level: miss 62.5 mm', () => '');
ok(range(CORD.MIN, CORD.MAX, 0.01).some(cordLevel), 'cord: slider reaches level', () => '');
ok(near(CORD.DROP / CORD.SPAN, 1 / 8), 'cord: ground falls 1 in 8 between the poles', () => '');

/* ── plate 2: rods ── */
for (const u of ['cubit', 'arsin'] as UnitId[])
  for (const err of range(ROD_ERR.MIN, ROD_ERR.MAX, ROD_ERR.STEP)) {
    const m = measure(u, err);
    const ctx = `${u} err=${err}`;
    ok(m.n * m.real <= SPAN && (m.n + 1) * m.real > SPAN, 'rods: n whole rods fit', () => ctx);
    ok(Math.abs(m.subsRead - m.subs) <= 0.25 + 1e-9, 'rods: reading to half a division', () => ctx);
    ok(err !== 0 || Math.abs(m.error) < m.u.len / m.u.sub / 2, 'rods: true rod within half a division', () => ctx);
    ok(err === 0 || Math.sign(m.error) === -Math.sign(err) || Math.abs(m.error) < 0.01, 'rods: short rod reads long', () => `${ctx} ${m.error}`);
    // independent: measured ≈ SPAN · len / real
    ok(Math.abs(m.measured - (SPAN * m.u.len) / m.real) <= m.u.len / m.u.sub, 'rods: matches SPAN·len/real', () => ctx);
    for (const W of WIDTHS)
      for (const laid of [0, 1, Math.floor(m.n / 2), m.n]) {
        const g = rodScene(W, u, err, laid, m.subsRead);
        ok(near(g.endX - g.pad, SPAN * g.k), 'rods: end mark at 100 m', () => ctx);
        ok(g.head <= g.endX + 1e-9, 'rods: laid rods never pass the end', () => `${ctx} laid=${laid}`);
        if (laid >= m.n) ok(Math.abs(g.pointer - g.endZ) <= (g.zk * m.real) / m.u.sub / 2 + 1e-9, 'rods: correct reading points at the end mark', () => `${ctx} W=${W}`);
        for (const r of g.zRods) ok(r.x0 >= g.pad - 1e-9 && r.x1 <= W - g.pad + 1e-9, 'rods: zoom rods inside', () => `${ctx} W=${W}`);
        for (const d of g.divs) ok(d.x >= 0 && d.x <= W, 'rods: divisions inside', () => `${ctx} W=${W}`);
        const labs = g.divs.filter((d) => d.label);
        for (let q = 1; q < labs.length; q++) ok(labs[q].x - labs[q - 1].x > 2 * 10 * 0.6 + 2, 'rods: division labels apart', () => `${ctx} W=${W}`);
      }
  }
ok(near(UNITS.arsin.len / UNITS.cubit.len, 1.4433, 1e-3), 'rods: arşın ≈ 1.44 cubits', () => '');

/* ── plate 3: groma ── */
for (const psi of range(GROMA.PSI_MIN, GROMA.PSI_MAX, GROMA.PSI_STEP)) {
  ok(onB(psi) === (Math.abs(psi - 90) <= GROMA.TOL), 'groma: onB', () => `${psi}`);
  if (Math.abs(psi - 90) < 1e-9) ok(near(farHit(psi)!, 0, 1e-9), 'groma: square cross hits straight across', () => '');
  for (const W of WIDTHS) {
    const g = gromaScene(W, psi, 0, 0);
    ok(near((g.farX - g.A.x) / g.k, GROMA.WIDTH), 'groma: strait to scale', () => `W=${W}`);
    ok(near((g.A.y - g.B.y) / g.k, GROMA.B_Y), 'groma: B 200 m north', () => `W=${W}`);
    ok(g.inset.cy + g.inset.r + 14 < g.H - 14, 'groma: inset clear of the status line', () => `W=${W}`);
    ok(g.sight1.x >= 0 && g.sight1.x <= W && g.sight1.y >= 0, 'groma: sight 1 inside', () => `W=${W} ψ=${psi}`);
    const st = g.strings;
    ok(near(Math.hypot(st[0].x - st[1].x, st[0].y - st[1].y), Math.hypot(st[1].x - st[2].x, st[1].y - st[2].y)), 'groma: cross arms square', () => '');
  }
}
ok(range(GROMA.PSI_MIN, GROMA.PSI_MAX, GROMA.PSI_STEP).some(onB), 'groma: slider reaches B', () => '');
ok(range(GROMA.POLE_MIN, GROMA.POLE_MAX, GROMA.POLE_STEP).some((y) => poleHidden(90, y)), 'groma: pole slider reaches the line', () => '');
for (const psi of range(89.9, 90.1, 0.1)) {
  const y = farHit(psi)!;
  ok(y >= GROMA.POLE_MIN && y <= GROMA.POLE_MAX, 'groma: target within pole range when on B', () => `${psi} ${y}`);
  ok(range(GROMA.POLE_MIN, GROMA.POLE_MAX, GROMA.POLE_STEP).some((p) => poleHidden(psi, p)), 'groma: every on-B setting can be completed', () => `${psi}`);
}
ok(near(windMiss(5), 700 * (0.005 / 0.9), 1e-6), 'groma: wind miss formula', () => '');

/* ── plate 4: chorobates ── */
ok(near(TILT_TOL, 0.001 / 1.48), 'choro: tolerance 1 mm over the groove', () => '');
ok(near(BED_FALL_MM, 1.4, 1e-9), 'choro: bed falls 1.4 mm between legs', () => `${BED_FALL_MM}`);
for (const shim of range(CHORO.SHIM_MIN, CHORO.SHIM_MAX, CHORO.SHIM_STEP)) {
  const ctx = `shim=${shim}`;
  // independent: true fall + tilt error
  ok(near(measuredDrop(shim), TRUE_DROP - SIGHT_D * Math.tan(tiltOf(shim)), 1e-12), 'choro: drop = true − d·tan(tilt)', () => ctx);
  if (looksLevel(shim)) ok(Math.abs(measuredDrop(shim) - TRUE_DROP) <= DROP_UNC + 1e-9, 'choro: looks-level error within the band', () => ctx);
  for (const W of WIDTHS) {
    const b = benchScene(W, shim);
    ok(Math.sign(b.t) === Math.sign(tiltOf(shim)) || Math.abs(tiltOf(shim)) < 1e-12, 'choro: drawn tilt sign = tilt', () => ctx);
    ok(b.topL.x > 0 && b.topR.x < W && b.topL.y > 20 && b.topR.y > 20, 'choro: bench inside', () => `${ctx} W=${W}`);
    // water: the higher rim is the dry one
    ok(tiltOf(shim) <= 0 || b.gR.y > b.gL.y, 'choro: tilted downhill → right rim lower', () => ctx);
    for (const band of [false, true]) {
      const s = sightScene(W, shim, band);
      ok(near(s.read, measuredDrop(shim) + 1 + shim / 1000, 1e-9), 'choro: staff = drop + bench height', () => ctx);
      ok(s.hit.clipped || near(s.yOfR(s.read), s.hit.y, 1e-6), 'choro: sight meets the staff at its reading', () => `${ctx} W=${W} ${s.yOfR(s.read)} ${s.hit.y}`);
      ok(s.hit.y >= 0 && s.hit.y <= s.H, 'choro: hit inside', () => `${ctx} W=${W}`);
      ok(s.ticks.length >= 3, 'choro: staff ticks visible', () => `${ctx} W=${W}`);
    }
  }
}
ok(range(CHORO.SHIM_MIN, CHORO.SHIM_MAX, CHORO.SHIM_STEP).some(looksLevel), 'choro: slider reaches level', () => '');
ok(DROP_UNC > TRUE_DROP, 'choro: the point of step 5: tolerance error exceeds the fall', () => `${DROP_UNC} ${TRUE_DROP}`);

/* ── plate 6: chain ── */
for (const lay of ['surface', 'stepped'] as const) {
  const c = chainCount(lay);
  ok(near(c.L, lay === 'stepped' ? LINE_H : LINE_SURFACE), 'chain: length', () => lay);
  ok(Math.abs(c.measured - c.L) <= CHAIN_M / 200 + 1e-9, 'chain: within half a link', () => lay);
  const ends = chainEnds(lay, c.n);
  ok(ends[ends.length - 1] <= SITE.SHORE_X + 1e-6, 'chain: n chains stay short of A', () => `${lay} ${ends[ends.length - 1]}`);
  // independent: the surface length of each chain = CHAIN_M
  if (lay === 'surface')
    for (let i = 1; i < ends.length; i++) {
      let s = 0;
      const N = 400;
      for (let j = 0; j < N; j++) {
        const x0 = ends[i - 1] + ((ends[i] - ends[i - 1]) * j) / N;
        const x1 = ends[i - 1] + ((ends[i] - ends[i - 1]) * (j + 1)) / N;
        s += Math.hypot(x1 - x0, groundZ(x1) - groundZ(x0));
      }
      ok(Math.abs(s - CHAIN_M) < 1e-4, 'chain: each chain covers 66 ft of surface', () => `${i} ${s}`);
    }
  for (const W of WIDTHS)
    for (let laid = 0; laid <= c.n; laid++) {
      const g = chainScene(W, lay, laid);
      ok(g.pins.every((p) => p.x >= 0 && p.x <= W && p.y > 0 && p.y < g.H), 'chain: pins inside', () => `${lay} W=${W}`);
      ok(near((g.A.x - g.M.x) / g.k, SITE.SHORE_X), 'chain: M–A to scale', () => '');
    }
  for (const W of WIDTHS) {
    const g = linksScene(W, lay, c.links);
    ok(Math.abs(g.pointer - g.endX) <= g.kl / 2 + 1e-9, 'chain: link reading at the mark', () => `${lay} W=${W}`);
    ok(g.endX < W - g.pad + 1e-9, 'chain: end mark inside', () => `${lay} W=${W}`);
  }
}

/* ── plate 7: map measurer ── */
ok(near(M_PER_IN, 127, 1e-9), 'map: 1 in = 127 m at 1:5,000', () => '');
ok(range(0, 8, READ_STEP).some(readOk), 'map: reading slider reaches the needle', () => `${ROAD_IN}`);
ok(Math.abs(Math.round(ROAD_IN / READ_STEP) * READ_STEP * M_PER_IN - ROAD_LEN) <= (READ_STEP / 2) * M_PER_IN + 1e-6, 'map: read distance within half a step of the road', () => '');
for (const W of WIDTHS) {
  for (const f of range(0, 1, 0.005)) {
    const g = chartScene(W, f);
    ok(g.wheel.x >= 0 && g.wheel.x <= W && g.wheel.y >= 0 && g.wheel.y <= g.H, 'map: wheel inside', () => `W=${W} f=${f}`);
  }
  const g = chartScene(W, 1);
  ok(near(g.wheel.x, g.A.x) && near(g.wheel.y, g.A.y), 'map: wheel ends at A', () => `W=${W}`);
  ok(g.road.every((p) => p.x >= 0 && p.x <= W && p.y >= 0 && p.y <= g.H), 'map: road inside', () => `W=${W}`);
  ok(g.bar.x1 + 6 + 15 * 10 * 0.6 <= W, 'map: scale bar label inside', () => `W=${W}`);
  for (const r of [0, ROAD_IN, 7.5, 15]) {
    const d = dialScene(W, r, null);
    ok(d.cx - d.R - 6 >= 0 && d.cx + d.R + 6 <= W && d.cy + d.R + 6 < d.H - 14, 'map: dial inside', () => `W=${W}`);
    ok(near(((90 - d.needle) / 360) * 30, r), 'map: needle angle = reading', () => `${r}`);
  }
}

/* ── plates 8, 9: theodolite ── */
ok(near(A_TRUE, 90, 1e-9), 'theo: α = 90°', () => `${A_TRUE}`);
ok(near(B_TRUE, Math.atan(SITE.WIDTH / BASE) / RAD, 1e-9), 'theo: β = atan(700/200)', () => `${B_TRUE}`);
ok(near(widthFrom(BASE, A_TRUE, B_TRUE), SITE.WIDTH, 1e-9), 'theo: width = 700', () => '');
for (const k of [readingTo('A', PB), readingTo('A', PC), readingTo('B', PA), readingTo('B', PC)]) {
  const s = snap(k);
  ok(s.d >= 0 && s.d <= 359 && s.m >= 0 && s.m <= 59.9, 'theo: snap within slider ranges', () => `${k}`);
  ok(Math.abs(diff(toDeg(s.d, s.m), k)) <= ON_TOL + 1e-12, 'theo: snapped reading is on target', () => `${k} ${s.d} ${s.m}`);
}
// snapped readings give the width to within the reading resolution's effect
{
  const r = (q: number) => toDeg(snap(q).d, snap(q).m);
  const a = (((r(readingTo('A', PC)) - r(readingTo('A', PB))) % 360) + 360) % 360;
  const b = (((r(readingTo('B', PA)) - r(readingTo('B', PC))) % 360) + 360) % 360;
  ok(Math.abs(widthFrom(BASE, a, b) - 700) < 0.2, 'theo: width from slider readings ≈ 700 m', () => `${widthFrom(BASE, a, b)}`);
  const h = heightFrom(TOWER.far - TOWER.near, r(EL_FAR), r(EL_NEAR));
  ok(Math.abs(h - 76) < 0.1, 'theo: height from slider readings ≈ 76 m', () => `${h}`);
}
for (const v of range(0, 359.9, 0.05)) {
  const s = split(v);
  ok(near(toDeg(s.d, s.m), Math.round(v * 600) / 600 % 360, 1e-9) || near(v, 359.95, 1e-6), 'theo: split round-trips', () => `${v}`);
}
for (const W of WIDTHS)
  for (const dh of [-2, -0.5, -ON_TOL, 0, ON_TOL, 0.3, 0.6, 5, 179])
    for (const dv of [0, 0.3, -0.7]) {
      const g = eyepiece(W, 100, 10, [{ name: 'T', h: 100 + dh, v: 10 + dv, kind: 'tip' }]);
      const t = g.targets[0];
      ok(t.on === (Math.abs(dh) <= ON_TOL + 1e-12 && Math.abs(dv) <= ON_TOL + 1e-12), 'theo: eyepiece on-flag', () => `${dh} ${dv}`);
      ok(near(t.x - g.cx, dh * g.k, 1e-6) || !t.inside, 'theo: drawn offset = angle × scale', () => `${dh}`);
      ok(dh <= 0 || t.x > g.cx, 'theo: clockwise target appears right', () => `${dh}`);
      ok(dv <= 0 || t.y < g.cy, 'theo: higher target appears above', () => `${dv}`);
      ok(!t.inside || Math.hypot(t.x - g.cx, t.y - g.cy) <= g.R, 'theo: inside flag', () => '');
      ok(g.cx - g.R - 5 >= 0 && g.cx + g.R + 5 <= W, 'theo: eyepiece inside', () => `W=${W}`);
      ok(g.cy + g.R + 5 + 4 < g.H - 6 - 12, 'theo: status line below the eyepiece', () => `W=${W}`);
    }
ok(EL_FAR < 60 && EL_NEAR < 59 + 59.9 / 60, 'theo: elevations reachable by the sliders', () => `${EL_NEAR}`);
void MIN_STEP;

/* ── plate 11: bars ── */
ok(near(X_STAR, 2, 1e-12), 'bars: still point 2 gap widths', () => '');
for (const t of range(T.MIN, T.MAX, T.STEP)) {
  ok(near(tongueMove(X_STAR, t), 0, 1e-15), 'bars: still point does not move', () => `${t}`);
  ok(near(compLen(t), L0, 1e-12), 'bars: compensated length constant', () => `${t}`);
  ok(near(plainErr(t), BASELINE * 11.5e-6 * (t - T0)), 'bars: plain error', () => `${t}`);
  // plainErr is the first-order error; the exact shortfall differs by B·(αΔT)², under 0.1 mm
  ok(Math.abs(BASELINE - plainRead(t) - plainErr(t)) < 1e-4, 'bars: read short by the error', () => `${t}`);
  for (const W of WIDTHS) {
    for (const dx of range(0, 2.6, 0.05)) {
      const g = compScene(W, t, dx);
      for (const e of g.ends) {
        ok(e.tip.x > 0 && e.tip.x < W && e.tip.y < g.H - 20, 'bars: tongue inside', () => `W=${W} t=${t}`);
        // drawn movement of the probe = tongueMove × kx × EX, sign included
        const rest = g.cx + e.side * g.half;
        ok(near((e.probe.x - rest) * e.side, tongueMove(dx, t) * g.kx * g.EX, 1e-6), 'bars: probe drawn = tongueMove', () => `t=${t} dx=${dx}`);
      }
    }
    const b = baseScene(W, t, false);
    ok(b.readEnd > b.pad && b.readEnd < W, 'bars: read end inside', () => `W=${W} t=${t}`);
    ok(near(b.trueEnd - b.readEnd, plainErr(t) * b.k * b.EX2, 1e-3), 'bars: drawn shortfall = error × EX', () => `t=${t}`);
  }
}
ok(range(0, 2.6, 0.05).some((x) => Math.abs(x - X_STAR) < 0.05), 'bars: slider reaches the still point', () => '');
ok(ALPHA.brass / ALPHA.iron === 1.5, 'bars: 3:2', () => '');

/* ── plate 12: EDM ── */
ok(near(N_TRUE + PHI_FINE / 360, D_TRUE / halfLambda(F_FINE), 1e-12), 'edm: N + φ = D / (λ/2)', () => '');
{
  const fr = Math.round(PHI_FINE / PHASE_STEP) * PHASE_STEP;
  const cr = Math.round(PHI_COARSE / PHASE_STEP) * PHASE_STEP;
  ok(phaseOk(fr, PHI_FINE) && phaseOk(cr, PHI_COARSE), 'edm: sliders reach the phases', () => '');
  const N = Math.floor(coarseD(cr) / halfLambda(F_FINE) - fr / 360 + 0.5);
  ok(N === N_TRUE, 'edm: coarse fixes N', () => `${N}`);
  ok(Math.abs(distance(N, fr) - D_TRUE) < 0.03, 'edm: distance within 3 cm', () => `${distance(N, fr)}`);
  // every accepted pair of readings gives the right N
  for (const c of [cr - PHASE_STEP / 2, cr, cr + PHASE_STEP / 2]) for (const f of [fr - 0.25, fr, fr + 0.25]) {
    const n = Math.floor(coarseD(c) / halfLambda(F_FINE) - f / 360 + 0.5);
    ok(n === N_TRUE, 'edm: N robust within tolerance', () => `${c} ${f}`);
  }
}
ok(near(halfLambda(F_COARSE) / halfLambda(F_FINE), 100), 'edm: frequencies 100:1', () => '');
for (const W of WIDTHS)
  for (const r of range(0, 359.5, 2.5)) {
    const g = phaseScene(W, PHI_FINE, r);
    ok(g.wl > 0 && g.wy[1] + g.amp + 30 < g.H, 'edm: waves inside', () => `W=${W}`);
  }

/* ── GPS ── */
ok(near(0.008 + 1e-6 * 700, 0.0087), 'gps: 8 mm + 1 ppm at 700 m', () => '');

/* ── hodometer (reviewer) ── */
for (const W of WIDTHS) {
  const { H, wc, dc, bowl } = hodoLayout(W);
  ok(wc.x - wc.r >= 0 && dc.x + dc.r <= W && bowl.x - 34 >= 0 && bowl.x + 34 <= W, 'hodo: wheel, drum and bowl inside', () => `W=${W}`);
  const lab = textBox('bowl', bowl.x, bowl.y + 26, `bowl: ${hodometer(30_000).pebbles} pebbles`, 11, 'middle');
  ok(inside(lab, W, H), 'hodo: bowl label inside', () => `W=${W} ${JSON.stringify(lab)}`);
  ok(dc.x - dc.r > wc.x + wc.r && bowl.x - 34 > dc.x - dc.r, 'hodo: parts in order, no overlap', () => `W=${W}`);
}
for (const ft of range(0, 30_000, 50)) {
  const h = hodometer(ft);
  ok(h.pebbles === Math.floor(Math.floor(ft / REV) / TEETH), 'hodo: pebbles = whole drum turns', () => `${ft}`);
  ok(h.counted <= ft + 1e-9 || h.pebbles === 0 || h.counted - ft < MILE_FT, 'hodo: counted is whole miles', () => `${ft}`);
}

/* ── no negative widths in the full-width bands (reviewer: console showed width=-13) ── */
for (const W of [...WIDTHS, 200, 260, 300]) {
  const g = siteScene(W);
  ok(g.farShoreX - g.nearShoreX > 0 && g.nearShoreX - g.pad.l > 0 && W - g.pad.r - g.farShoreX > 0, 'site: bands have positive width', () => `W=${W}`);
  const m = chartScene(W, 0.5);
  ok(W - 14 - m.shoreX > 0, 'map: water band has positive width', () => `W=${W}`);
  const gr = gromaScene(W, 90, 0, 0);
  ok(gr.farX - gr.A.x > 0, 'groma: strait band has positive width', () => `W=${W}`);
}

void FAR_X;
report();
