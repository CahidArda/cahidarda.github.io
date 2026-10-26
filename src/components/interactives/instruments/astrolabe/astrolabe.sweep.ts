/**
 * Sweep of every astrolabe plate's geometry.
 *   node --experimental-strip-types src/components/interactives/instruments/astrolabe/astrolabe.sweep.ts
 *
 * Independent model: a plain 3D vector implementation of the sky (rotate a unit vector by the
 * latitude, project from the south pole) that shares no code with plateGeom.ts, used to check
 * the plate's circles, the rete, the solver and the dome.
 */
import { near, ok, report, textBox, overlap } from '../../../../../scripts/plates/check.ts';
import * as A from './astro.ts';
import * as P from './plateGeom.ts';
import * as S from './skyGeom.ts';
import * as F from './faceGeom.ts';
import * as T from './tellGeom.ts';
import * as J from './projGeom.ts';
import * as M from './armGeom.ts';
import * as Q from './quadGeom.ts';
import * as Z from './sapheaGeom.ts';
import { clockState } from './clockGeom.ts';

const D = Math.PI / 180;
const WIDTHS = [320, 340, 360, 400, 440, 520, 600, 700];
const DAYS = Array.from({ length: 73 }, (_, i) => 1 + i * 5);
const LATS = [21.42, 30.04, 41, 41.01];

/* ── independent model ── */
type V3 = [number, number, number];
/** Equatorial unit vector: x west (H=90°), y towards the upper meridian (H=0), z north pole. */
const eqVec = (dec: number, H: number): V3 => [Math.cos(dec * D) * Math.sin(H * D), Math.cos(dec * D) * Math.cos(H * D), Math.sin(dec * D)];
/** Local zenith in the equatorial frame at latitude φ: tilted from the pole towards the upper meridian. */
const zenithVec = (phi: number): V3 => [0, Math.cos(phi * D), Math.sin(phi * D)];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const altOf = (v: V3, phi: number) => Math.asin(dot(v, zenithVec(phi))) / D;
/** Stereographic from the south pole (0,0,−1) onto z = 0. */
const proj = (v: V3) => ({ u: v[0] / (1 + v[2]), v: v[1] / (1 + v[2]) });
/** Inverse of proj. */
const unproj = (p: { u: number; v: number }): V3 => {
  const r2 = p.u * p.u + p.v * p.v;
  return [(2 * p.u) / (1 + r2), (2 * p.v) / (1 + r2), (1 - r2) / (1 + r2)];
};

/* ── 1. projection and plate circles ── */
for (const phi of LATS) {
  for (let dec = -23; dec <= 89; dec += 4)
    for (let H = -180; H <= 180; H += 15) {
      const a = P.stereo(dec, H);
      const b = proj(eqVec(dec, H));
      ok(near(a.u, b.u, 1e-9) && near(a.v, b.v, 1e-9), 'stereo = independent projection', () => `${dec} ${H}`);
      const back = P.unstereo(a);
      ok(near(back.dec, dec, 1e-7), 'unstereo inverts', () => `${dec} ${H}`);
    }
  for (const h of [...P.ALMS, -6, -18]) {
    for (const run of P.almucantarRuns(h, phi))
      for (const p of run) {
        ok(Math.hypot(p.u, p.v) <= P.RC + 1e-6, 'almucantar inside plate', () => `${phi} ${h}`);
        ok(near(altOf(unproj(p), phi), h, 1e-6), 'almucantar point has altitude h (independent)', () => `${phi} ${h} ${altOf(unproj(p), phi)}`);
      }
  }
  for (const Az of P.AZIS)
    for (const run of P.azimuthRuns(Az, phi))
      for (const p of run) {
        const v = unproj(p);
        ok(altOf(v, phi) >= -1e-6, 'azimuth line above horizon', () => `${phi} ${Az}`);
        // azimuth from south, + west, computed independently in the local frame
        const z = zenithVec(phi);
        const southDir: V3 = [0, Math.sin(phi * D), -Math.cos(phi * D)]; // horizontal, towards the upper meridian
        const westDir: V3 = [1, 0, 0];
        const az = Math.atan2(dot(v, westDir), dot(v, southDir)) / D;
        const h = altOf(v, phi);
        if (h < 89.5) ok(near(A.wrap180(az - Az), 0, 1e-5) || Math.abs(A.wrap180(az - Az)) < 1e-4, 'azimuth point has azimuth A (independent)', () => `${phi} ${Az} got ${az}`);
        void z;
      }
  const z = P.zenithOf(phi);
  ok(near(altOf(unproj(z), phi), 90, 1e-7), 'zenith mark is the zenith', () => `${phi}`);
}
ok(near(P.RCAN, 0.6564, 1e-3) && near(P.RC, 1.5235, 1e-3), 'tropic radii (dossier 0.656, 1.524)');
const hz41 = P.almucantar(0, 41);
ok(near(hz41.cv, 1.150, 1e-3) && near(hz41.r, 1.524, 1e-3), 'horizon at 41° (dossier 1.150, 1.524)');
for (const [h, cv, r] of [[10, 0.91, 1.187], [30, 0.653, 0.749], [50, 0.531, 0.452], [80, 0.46, 0.106]] as const) {
  const a = P.almucantar(h, 41);
  ok(Math.abs(a.cv - cv) < 1e-3 && Math.abs(a.r - r) < 1e-3, 'almucantar table (dossier)', () => `${h}`);
}

/* ── 2. rete: ecliptic ring, sun point, stars ── */
for (let lst = 0; lst < 360; lst += 7.5) {
  const ec = P.eclipticCircle(lst);
  for (let lam = 0; lam < 360; lam += 10) {
    const p = P.eclipticPoint(lam, lst);
    ok(near(Math.hypot(p.u - ec.c.u, p.v - ec.c.v), ec.r, 1e-9), 'ecliptic point on the ecliptic ring', () => `${lst} ${lam}`);
    // independent: rotate the ecliptic vector into equatorial, then by the sky turn
    const e = 23.44 * D;
    const x0 = Math.cos(lam * D);
    const y0 = Math.sin(lam * D) * Math.cos(e);
    const z0 = Math.sin(lam * D) * Math.sin(e);
    const ra = Math.atan2(y0, x0) / D;
    const dec = Math.asin(z0) / D;
    const q = proj(eqVec(dec, lst - ra));
    ok(near(p.u, q.u, 1e-9) && near(p.v, q.v, 1e-9), 'sun on rete = independent', () => `${lst} ${lam}`);
  }
  for (const s of A.STARS) {
    const p = P.starPoint(s, lst);
    ok(Math.hypot(p.u, p.v) <= P.RC, 'star pointer on the rete', () => s.id);
  }
}

/* ── 3. sun, solver, time ── */
ok(Math.abs(A.sunLambda(80)) < 1.5 || Math.abs(A.sunLambda(80) - 360) < 1.5, 'equinox λ≈0 on 21 Mar');
ok(Math.abs(A.sunLambda(172) - 90) < 1.5, 'solstice λ≈90 on 21 Jun');
ok(Math.abs(A.eotMinutes(42) + 14.2) < 0.6 && Math.abs(A.eotMinutes(307) - 16.4) < 0.6, 'equation of time extremes (−14.2, +16.4 min)');
const ex = A.hourAngle(30, 0, 41)!;
ok(near(ex, 48.51, 1e-3) && A.hm(A.timeOf(-ex)) === '8:46', 'dossier example 8:46', () => `${ex}`);
for (const [d, len] of [[23.44, 14.95], [0, 12], [-23.44, 9.05]] as const) ok(Math.abs((2 * A.halfDay(d, 41)) / 15 - len) < 0.01, 'day length at 41°', () => `${d}`);

for (const doy of DAYS)
  for (const m of A.MOMENTS) {
    for (const phi of LATS) {
      const sun = A.sunAt(doy, m.t, phi);
      ok(sun.h > 1, 'every preset moment has the sun up', () => `${doy} ${m.id} ${phi} h=${sun.h}`);
      // independent altitude
      ok(near(altOf(eqVec(sun.dec, sun.H), phi), sun.h, 1e-9), 'altitude = independent', () => `${doy} ${m.id}`);
    }
    const sun = A.sunAt(doy, m.t);
    const side = m.t < 12 ? 'am' : 'pm';
    // the reader's sight snaps to 0.1°; solve from that reading
    for (const err of [-0.3, 0, 0.3]) {
      const a = T.snapAli(sun.h + err, sun.h);
      ok(T.sighted(a, sun.h), 'alidade within tolerance is sighted');
      ok(Math.abs(a - sun.h) <= 0.05 + 1e-9, 'snapped alidade within 0.05° of the sun', () => `${a} ${sun.h}`);
    }
    const hMeas = T.round1(sun.h);
    const sol = P.solveRete(sun.lam, hMeas, side, 41)!;
    ok(!!sol, 'solver finds the rete turn', () => `${doy} ${m.id}`);
    const st = T.reteState(sun.lam, sol.lst, hMeas, side);
    ok(st.on && near(st.hPlate, hMeas, 1e-7), 'solved rete puts the sun on the circle', () => `${doy} ${m.id}`);
    // the sun's point drawn on the plate sits on the drawn almucantar (independent check)
    const pt = P.eclipticPoint(sun.lam, sol.lst);
    const al = P.almucantar(hMeas, 41);
    ok(near(Math.hypot(pt.u, pt.v - al.cv), al.r, 1e-7), 'sun point on drawn almucantar', () => `${doy} ${m.id}`);
    ok(near(altOf(unproj(pt), 41), hMeas, 1e-6), 'sun point altitude (independent)');
    ok(side === 'am' ? pt.u < 0 : pt.u > 0, 'morning on the east (left)', () => `${doy} ${m.id} ${pt.u}`);
    const s = T.solution(sun.lam, hMeas, side);
    ok(s.t !== null && Math.abs(s.t! - m.t) < 2.5 / 60, 'measured time = true time within 2.5 min (0.1° reading)', () => `${doy} ${m.id} ${s.t} vs ${m.t}`);
    ok(near(s.H!, sol.H, 1e-9), 'equation H = rete H');
    // rule: snapping to the sun's hour gives the time in the equation; the limb point matches
    const Hs = P.sunHourOnRete(sun.lam, sol.lst);
    const rule = T.snapRule(Hs + 0.4, Hs);
    ok(rule === Hs, 'rule snaps onto the sun');
    ok(A.hm(12 + rule / 15) === A.hm(s.t!), 'rule reading = equation', () => `${doy} ${m.id}`);
    for (const W of [340, 520, 700]) {
      const f = F.frontScene(W, { phi: 41, lst: sol.lst, sunLam: sun.lam, ruleH: rule });
      // the sun dot lies on the rule's red line
      const r = f.rule!;
      const cross = (r.b.x - r.a.x) * (f.sun!.y - r.a.y) - (r.b.y - r.a.y) * (f.sun!.x - r.a.x);
      const len = Math.hypot(r.b.x - r.a.x, r.b.y - r.a.y);
      ok(Math.abs(cross / len) < 1e-6, 'sun dot on the drawn rule', () => `${W} ${cross / len}`);
      ok(f.sun!.x >= 0 && f.sun!.x <= W && f.sun!.y >= 0 && f.sun!.y <= f.Hd, 'sun dot inside the face view', () => `${W}`);
      // zoomed reading view keeps the sun and the limb mark on screen
      const sp = P.eclipticPoint(sun.lam, sol.lst);
      const la = (P.limbAngle(rule) * Math.PI) / 180;
      const z = F.frontScene(W, { phi: 41, lst: sol.lst, sunLam: sun.lam, ruleH: rule, zoom: 1.6, focus: { u: sp.u * 0.6 + P.RC * 0.4 * Math.cos(la), v: sp.v * 0.6 + P.RC * 0.4 * Math.sin(la) } });
      for (const q of [z.sun!, z.rule!.limb]) ok(q.x >= 4 && q.x <= W - 4 && q.y >= 4 && q.y <= z.Hd - 4, 'zoomed reading: sun and limb mark visible', () => `${doy} ${m.id} ${W} ${JSON.stringify(q)}`);
    }
    // clock time is plausible: within 64 ± 17 min of sun time
    const c = T.clockOf(s.t!, doy) - s.t!;
    ok(c * 60 > 64 - 17 && c * 60 < 64 + 17, 'clock correction range');
  }

/* ── 4. alidade and back scales ── */
for (let a = 0; a <= 90; a += 0.5) {
  for (const W of [340, 520, 700]) {
    const g = F.backScene(W, { ali: a });
    // the edge passes through the pin and through the altitude scale at a
    const psi = P.backPsi(a);
    const r = (P.BACK.altIn + P.BACK.altOut) / 2;
    const want = { x: g.f.cx + r * Math.cos(psi * D) * g.f.k, y: g.f.cy - r * Math.sin(psi * D) * g.f.k };
    ok(near(g.altMark.x, want.x, 1e-9) && near(g.altMark.y, want.y, 1e-9), 'altitude mark at a on the scale');
    // reading the angle back from the drawing gives a
    const back = 180 - Math.atan2(-(g.altMark.y - g.f.cy), g.altMark.x - g.f.cx) / D;
    ok(near(A.wrap180(back - a), 0, 1e-6), 'drawn alidade angle = setting', () => `${a} ${back}`);
  }
  const s = P.sight(a, 30);
  ok(s.on === Math.abs(a - 30) <= P.SIGHT_TOL, 'sight status = tolerance');
  // the ray from the top hole lands on the bottom vane's line, `miss` from the hole
  const n = { u: Math.cos(P.backPsi(a) * D), v: Math.sin(P.backPsi(a) * D) };
  const off = (s.hit.u - s.tail.u) * -n.v + (s.hit.v - s.tail.v) * n.u;
  ok(Math.abs(Math.abs(off) - Math.abs(s.miss)) < 1e-9, 'ray miss = drawn miss', () => `${a} ${off} ${s.miss}`);
}
// date → calendar ring → zodiac: same angle
for (const doy of DAYS) {
  const a = F.aliForDate(doy);
  ok(near(A.wrap180(P.backPsi(a) - (180 - A.sunLambda(doy))), 0, 1e-9), 'calendar alidade on λ(date)');
}

/* ── 5. stable reference (dome) ── */
for (const W of WIDTHS) {
  const H0 = S.viewH(W);
  for (const doy of DAYS)
    for (const m of A.MOMENTS) {
      const sun = A.sunAt(doy, m.t);
      const g = S.dome(W, { dec: sun.dec, sunH: sun.H, alm: sun.h, showHourArc: true });
      ok(g.H === H0 && g.W === W, 'dome size fixed per width');
      const g2 = S.dome(W, { dec: 0, sunH: null });
      ok(g.cx === g2.cx && g.cy === g2.cy && g.R === g2.R && near(g.N.x, g2.N.x) && near(g.pole.y, g2.pole.y), 'dome camera does not depend on the step/sun');
      ok(g.E.x < g.cx && g.Wp.x > g.cx, 'east on the left');
      const all = [...g.horizon, ...g.meridian, ...g.day, ...g.alm, ...g.hArc, ...g.hourArc].flatMap((r) => r.pts);
      ok(all.every((p) => p.x >= 0 && p.x <= W && p.y >= 0 && p.y <= g.H), 'dome lines inside', () => `${W}`);
      // the sun dot lies on its altitude: independent from the projection of the local vector
      ok(near(g.sun!.h, sun.h, 1e-9), 'dome sun altitude = sun');
      const foot = g.hFoot!;
      const top = g.hArc[g.hArc.length - 1].pts.at(-1)!;
      ok(near(top.x, g.sun!.x, 1e-6) && near(top.y, g.sun!.y, 1e-6), 'altitude arc ends at the sun');
      void foot;
      ok(g.labels.every((l) => l.placed), 'dome labels found a free spot', () => `${W} ${doy} ${m.id}`);
      // labels inside and not overlapping
      const boxes = g.labels.map((l) => ({ ...S.labelBox(l), name: l.id }));
      ok(boxes.every((b) => b.x0 >= 0 && b.x1 <= W && b.y0 >= 0 && b.y1 <= g.H), 'dome labels inside', () => `${W} ${JSON.stringify(boxes.filter((b) => b.x0 < 0 || b.x1 > W || b.y0 < 0 || b.y1 > g.H))}`);
      for (let i = 0; i < boxes.length; i++)
        for (let j = i + 1; j < boxes.length; j++)
          ok(!overlap(boxes[i] as never, boxes[j] as never, 1), 'dome labels do not collide', () => `${W} ${doy} ${m.id} ${boxes[i].name}/${boxes[j].name}`);
    }
}

/* ── 6. armillary ── */
for (const W of [320, 400, 520, 700])
  for (const doy of DAYS)
    for (let axis = 0; axis <= 90; axis += 15)
      for (let H = M.ARM_H_MIN; H <= M.ARM_H_MAX; H += 30) {
        const g = M.armScene(W, { axis, lam: A.sunLambda(doy), H });
        ok(near(g.alt, altOf(eqVec(A.eclDec(A.sunLambda(doy)), H), axis), 1e-9), 'armillary altitude (independent)');
        const pts = [...g.meridian, ...g.horizon, ...g.equator, ...g.ecliptic].flatMap((r) => r.pts);
        ok(pts.every((p) => p.x >= 0 && p.x <= W && p.y >= 0 && p.y <= g.H - 20), 'armillary inside (above the note)', () => `${W} ${axis}`);
        ok(g.ticks.every((t) => t.lab.x > 6 && t.lab.x < W - 6 && t.lab.y > 6 && t.lab.y < g.H - 24), 'armillary hour labels inside');
        ok(g.stand.foot.y <= g.H - 12 - 12 && g.stand.top.y < g.stand.foot.y, 'armillary stand above the note', () => `${W} ${axis}`);
      }
for (const doy of DAYS)
  for (const m of A.MOMENTS) {
    const sun = A.sunAt(doy, m.t);
    const g = M.armScene(400, { axis: 41, lam: sun.lam, H: sun.H });
    ok(near(g.alt, sun.h, 1e-9), 'armillary at the true hour shows the true altitude');
    ok(A.hm(M.armReading(sun.H)) === A.hm(m.t), 'armillary hour reading');
    // reachable with the 0.5° slider within tolerance
    const Hs = Math.round(sun.H * 2) / 2;
    ok(Math.abs(M.armScene(400, { axis: 41, lam: sun.lam, H: Hs }).alt - sun.h) <= M.ARM_TOL, 'armillary target reachable at slider step');
  }

/* ── 7. projection plate ── */
for (const W of WIDTHS)
  for (let dec = J.DEC_MIN; dec <= J.DEC_MAX; dec += 1)
    for (const h of [10, 20, 30, 40, 50, 60, 70, 80, 0]) {
      for (const mode of ['star', 'alm'] as const) {
        const g = J.projScene(W, { mode, dec, h, phi: 41 });
        for (const p of g.pts) {
          // the ray from the south pole through the point really crosses the plate line at p.cross
          const t = (g.sc.y - g.south.y) / (p.on.y - g.south.y);
          const x = g.south.x + (p.on.x - g.south.x) * t;
          ok(near(x, p.cross.x, 1e-6), 'side ray hits the plate at the marked point', () => `${W} ${dec}`);
          // and that crossing is drawn on the plan at the same distance from the pole
          // the drawn ray reaches both the point and its crossing (ray end is the farther one)
          const dEye = (q: { x: number; y: number }) => Math.hypot(q.x - g.south.x, q.y - g.south.y);
          ok(dEye(p.rayEnd) >= dEye(p.on) - 1e-9 && dEye(p.rayEnd) >= dEye(p.cross) - 1e-9, 'drawn ray reaches the point and the plate', () => `${W} ${mode} ${dec} ${h}`);
          ok(near((p.cross.x - g.sc.x) / g.ks, (g.pc.y - p.planPt.y) / g.kp, 1e-9), 'side crossing = plan distance');
          ok(p.cross.x >= g.sideBox.x0 && p.cross.x <= g.sideBox.x0 + g.sideBox.w, 'crossing inside its panel', () => `${W} ${mode} ${dec} ${h}`);
          if (Math.abs(p.x) <= P.RC) ok(p.planPt.y >= g.planBox.y0 && p.planPt.y <= g.planBox.y0 + g.planBox.h, 'plan point inside its panel');
        }
        if (mode === 'star') ok(near(g.circle!.rr, A.tand((90 - dec) / 2), 1e-12), 'star circle radius');
        else {
          // the plan circle passes through both projected meridian points
          for (const p of g.pts) ok(near(Math.hypot(p.planPt.x - g.circle!.c.x, p.planPt.y - g.circle!.c.y), g.circle!.r, 1e-6), 'almucantar circle through its two points', () => `${h}`);
        }
        ok(g.S(0, -1).y <= g.sideBox.y0 + g.sideBox.h + 1e-6 && g.S(0, 1).y >= g.sideBox.y0, 'side sphere inside its panel', () => `${W}`);
        ok(g.pc.y + g.tropics.cap <= g.planBox.y0 + g.planBox.h + 1e-6 && g.pc.y - g.tropics.cap >= g.planBox.y0 + 14, 'plan plate inside its panel', () => `${W}`);
      }
    }

/* ── 8. other latitudes (wrong plate) ── */
for (const doy of DAYS)
  for (const m of A.MOMENTS)
    for (const p of Z.PLACES) {
      const side = m.t < 12 ? 'am' : 'pm';
      const sun = A.sunAt(doy, m.t, p.lat);
      const h = T.round1(sun.h);
      const right = Z.plateReading(p.lat, h, sun.dec, side)!;
      ok(!!right && Math.abs(right.t - m.t) < 3 / 60, 'right plate gives the right time', () => `${p.id} ${doy} ${m.id}`);
      const wrong = Z.plateReading(41, h, sun.dec, side);
      if (p.id === 'ist') ok(!!wrong && Math.abs(wrong.t - right.t) < 1 / 60, 'Istanbul plate in Istanbul agrees');
    }
/* saphea: every latitude's horizon is a diameter; the day line meets it at the true sunset */
for (const phi of [0, 10, 21.42, 30.04, 41, 60])
  for (let dec = -23; dec <= 23; dec += 1) {
    const hz = Z.horizonDiameter(phi);
    for (let H = 1; H < 180; H += 7) {
      // a point on the horizon at azimuth: sample the horizon great circle via altitude 0
      const v = eqVec(dec, H);
      if (Math.abs(altOf(v, phi)) < 1e-3) continue;
    }
    for (let t = 0; t < 360; t += 5) {
      // horizon great circle points (independent): perpendicular to the zenith vector
      const z = zenithVec(phi);
      const u1: V3 = [1, 0, 0];
      const u2: V3 = [0, -z[2], z[1]];
      const p: V3 = [Math.cos(t * D) * u1[0] + Math.sin(t * D) * u2[0], Math.cos(t * D) * u1[1] + Math.sin(t * D) * u2[1], Math.cos(t * D) * u1[2] + Math.sin(t * D) * u2[2]];
      if (p[0] < -0.999) continue;
      const q = { x: p[1] / (1 + p[0]), y: p[2] / (1 + p[0]) };
      const cross = hz.a.x * q.y - hz.a.y * q.x;
      ok(Math.abs(cross) < 1e-9, 'saphea horizon is a diameter (independent)', () => `${phi} ${t}`);
    }
    const ss = Z.sunsetOn(phi, dec);
    const pt = ss.pt;
    const cross = hz.a.x * pt.y - hz.a.y * pt.x;
    ok(Math.abs(cross) < 1e-9, 'saphea sunset point on the horizon', () => `${phi} ${dec}`);
    ok(Math.abs(altOf(eqVec(dec, ss.H0), phi)) < 1e-9, 'sunset altitude 0');
  }

/* ── 9. quadrant ── */
for (const doy of DAYS)
  for (const m of A.MOMENTS) {
    const sun = A.sunAt(doy, m.t);
    const absH = Math.abs(sun.H);
    ok(absH <= Q.QUAD_MAX, 'preset moments fit on the quarter', () => `${doy} ${m.id} ${absH}`);
    const b = Q.bead(sun.dec, absH);
    ok(b.u >= 0 && b.v >= 0 && Math.hypot(b.u, b.v) <= P.RC, 'bead on the quarter');
    ok(near(altOf(unproj(b), 41), sun.h, 1e-7), 'bead sits on the sun altitude circle (independent)', () => `${doy} ${m.id}`);
    // the bead is where the rete would put the sun (mirrored to the west half)
    const sol = P.solveRete(sun.lam, sun.h, 'pm', 41)!;
    const rp = P.eclipticPoint(sun.lam, sol.lst);
    ok(near(rp.u, b.u, 1e-7) && near(rp.v, b.v, 1e-7), 'bead = rete sun point');
    const quarterPts = Q.quarterRuns().flatMap((q) => q.runs.flat());
    ok(quarterPts.every((p) => p.u >= -1e-9 && p.v >= -1e-9), 'quarter runs in the quarter');
  }

/* ── 10. clock dial ── */
for (let days = 0; days <= 365; days += 5) {
  const gain = days * 1440 * (A.SIDEREAL_RATE - 1);
  ok(Math.abs(gain / days - 3.943) < 0.01 || days === 0, 'sidereal gain 3.94 min/day');
}
for (let days = 0; days <= 365; days++) {
  const c = clockState(days);
  // the printed rete turn is the drawn one and stays within the equation of time (a few
  // degrees) of the mean gain: never wraps to 0° at a full year
  ok(Math.abs(c.turned - c.gain / 4) < 6, 'clock: printed rete turn ≈ gain', () => `${days} ${c.turned} ${c.gain / 4}`);
  ok(Math.abs(((c.turned % 360) + 360) % 360 - A.wrap360(c.lst - clockState(0).lst)) < 1e-6 || Math.abs(((c.turned % 360) + 360) % 360 - A.wrap360(c.lst - clockState(0).lst)) > 359.99, 'clock: printed turn = drawn turn', () => `${days}`);
}

/* ── 11. formatting ── */
for (let v = -200; v <= 200; v += 0.37) {
  for (const s of [A.deg1(v), A.deg0(v), A.num3(v / 100), A.hm(v / 10), A.dmin(v / 60)]) ok(!/e[+-]|NaN|undefined/.test(s), 'formatting', () => s);
  ok(A.deg1(-0.01) === '0.0°' && A.dmin(0) === '0 min', 'no negative zero');
}
void textBox;
report();
