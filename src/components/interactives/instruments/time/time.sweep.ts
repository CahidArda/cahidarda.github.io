/**
 * Sweep of every plate in the time post (see .claude/skills/verify-interactives/SKILL.md).
 *
 *   node --experimental-strip-types src/components/interactives/instruments/time/time.sweep.ts
 *
 * Per plate: maths against an independent formula, drawn = computed, bounds, label collisions,
 * honesty (readouts = drawn state; one input never rewrites another), and the deadbeat escapement
 * frame by frame (contact without interpenetration or air gap, no recoil, one tooth per period).
 */
import { near, ok, report, range, overlap } from '../../../../../scripts/plates/check.ts';
import { PHI, altitudeAt, declination, halfDay, hourAngle, noonAltitude } from './astro.ts';
import { fmtDist, fmtSec, hm, sci, sig } from './fmt.ts';
import { DEVICES, GAUGE_HI, GAUGE_LO, REF_H, lblBox, refScene, type RefInput } from './timeRefGeom.ts';
import * as SD from './sundialGeom.ts';
import * as WC from './waterGeom.ts';
import * as Q from './quadrantGeom.ts';
import * as VG from './vergeGeom.ts';
import * as DB from './deadbeat.ts';
import * as PG from './pendulumGeom.ts';
import { escScene } from './escScene.ts';
import * as SEA from './seaGeom.ts';
import * as QZ from './quartzGeom.ts';
import * as AT from './atomicGeom.ts';
import * as GPS from './gpsGeom.ts';

const WIDTHS = [300, 326, 360, 400, 440, 520, 600];
const D = Math.PI / 180;
const fmtOk = (s: string) => !/e[+-]?\d/.test(s) && !/NaN|Infinity|undefined/.test(s);

/* ───────────── formatting ───────────── */
for (const v of [0, 1e-15, 3e-12, 9e-9, 3.8e-5, 0.0616, 0.17, 1, 4.3, 59.6, 60, 148, 3599.6, 3600, 7200, 86400]) {
  ok(fmtOk(fmtSec(v)), 'fmt: fmtSec', () => `${v} → ${fmtSec(v)}`);
  ok(fmtOk(fmtDist(v * 1000)), 'fmt: fmtDist', () => `${v} → ${fmtDist(v * 1000)}`);
  ok(fmtOk(sig(v || 1)), 'fmt: sig', () => `${v}`);
  if (v) ok(fmtOk(sci(v)) || /10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]/.test(sci(v)), 'fmt: sci', () => sci(v));
}
for (let h = -1; h <= 25; h += 0.01) ok(/^\d{1,2}:\d\d$/.test(hm(h)) && !hm(h).startsWith('24'), 'fmt: hm', () => `${h} → ${hm(h)}`);

/* ───────────── the shared reference ───────────── */
function checkRef(inp: RefInput, ctx: string) {
  for (const W of WIDTHS) {
    const g = refScene(inp, W);
    const boxes = g.labels.map(lblBox);
    for (const b of boxes) ok(b.x0 >= 0 && b.x1 <= W && b.y0 >= 0 && b.y1 <= REF_H, `ref: label '${b.name}' inside`, () => `${ctx} W=${W} ${JSON.stringify(b)}`);
    for (let i = 0; i < boxes.length; i++)
      for (let j = i + 1; j < boxes.length; j++)
        ok(!overlap(boxes[i], boxes[j], 0.5), `ref: '${boxes[i].name}' vs '${boxes[j].name}'`, () => `${ctx} W=${W}`);
    ok(g.sun.x >= g.x0 - 1e-9 && g.sun.x <= g.x1 + 1e-9, 'ref: sun on the time axis', () => ctx);
    // the sun's height is drawn at the true altitude (world object)
    const alt = altitudeAt(inp.t, inp.dec);
    if (alt > 0) ok(near(g.horizonY - g.sun.y, alt * ((g.horizonY - g.skyTop) / 75), 1e-9), 'ref: sun drawn at its altitude', () => ctx);
    // true and device ticks at their times
    ok(near(g.trueX, g.x0 + ((((inp.t % 24) + 24) % 24) / 24) * (g.x1 - g.x0), 1e-9), 'ref: true tick at true time', () => ctx);
    if (inp.read !== null) ok(near(g.readX!, g.x0 + ((((inp.read % 24) + 24) % 24) / 24) * (g.x1 - g.x0), 1e-9), 'ref: device tick at reading', () => ctx);
    // honesty: gap text agrees with |read − t| when both are on the same day
    const gap = g.labels.find((l) => l.id === 'gap0');
    if (inp.gapSec != null && inp.gapSec > 0) ok(!!gap && gap.text.includes(fmtSec(inp.gapSec)), 'ref: gap text', () => `${ctx} ${gap?.text}`);
    for (const l of g.labels) ok(fmtOk(l.text), 'ref: label format', () => l.text);
    for (const b of g.bands) ok(b.a >= g.x0 - 2.01 && b.b <= g.x1 + 2.01 && b.b > b.a, 'ref: band inside gauge', () => `${ctx} ${JSON.stringify(b)}`);
    if (inp.rate) ok(g.rateX! >= g.x0 && g.rateX! <= g.x1, 'ref: rate mark inside', () => ctx);
  }
}
for (const d of DEVICES) ok(d.lo <= d.hi && d.lo >= GAUGE_LO && d.hi <= GAUGE_HI * 1.0001, 'ref: device range on the gauge', () => d.short);

/* ───────────── 1 sundial ───────────── */
{
  // independent model: sun direction from alt/az, shadow = −cot(alt) along the azimuth
  const altAz = (Hdeg: number, dec: number) => {
    const s = SD.sunVec(Hdeg, dec);
    return { alt: Math.asin(s.U) / D, az: Math.atan2(s.E, s.N) / D };
  };
  for (let day = 1; day <= 365; day += 4) {
    const dd = SD.dayOf(day);
    ok(near(dd.hourMin, ((2 * halfDay(declination(day))) / 15 / 12) * 60, 1e-9), 'sundial: hour length = 2H₀/15/12', () => `${day}`);
    ok(dd.hourMin > 44 && dd.hourMin < 76, 'sundial: 45..75 min at 41°', () => `${day} ${dd.hourMin}`);
    for (let t = SD.T_MIN; t <= SD.T_MAX + 1e-9; t += 1 / 12) {
      for (const mode of ['unequal', 'equal'] as const) {
        const s = SD.sundial(mode, day, t);
        const H = 15 * (t - 12);
        const aa = altAz(H, dd.dec);
        ok(near(aa.alt, altitudeAt(t, dd.dec), 1e-7), 'sundial: altitude agrees with astro', () => `${day} ${t}`);
        ok(s.up === (aa.alt > 0 && t > dd.rise && t < dd.set), 'sundial: shadow only when the sun is up', () => `${day} ${t} ${mode}`);
        if (s.tipW && mode === 'unequal') {
          const len = Math.hypot(s.tipW.x, s.tipW.y);
          ok(near(len, 1 / Math.tan(aa.alt * D), 1e-7), 'sundial: shadow length = cot(alt)', () => `${day} ${t}`);
          // shadow points away from the sun
          const dir = Math.atan2(-s.tipW.x, -s.tipW.y) / D;
          ok(Math.abs(((dir - aa.az + 540) % 360) - 180) < 1e-5 || len < 1e-9, 'sundial: shadow points away from the sun', () => `${day} ${t}`);
        }
        if (mode === 'equal' && s.tipW) {
          // the style's shadow lies on the equal-hour line for this time
          const dir = SD.equalDir(t);
          const cr = s.tipW.x * dir.y - s.tipW.y * dir.x;
          ok(Math.abs(cr) < 1e-7, 'sundial: style shadow on its hour line', () => `${day} ${t} cr=${cr}`);
          ok(s.read === t, 'sundial: equal dial reads true time', () => `${day} ${t}`);
        }
        if (mode === 'unequal' && s.read !== null) {
          ok(!s.tipOff && SD.onPlate(s.tipW!), 'sundial: reading only from a tip on the plate', () => `${day} ${t}`);
          ok(near(s.read, 6 + SD.timeToTemporal(t, day), 1e-9), 'sundial: read = 6 + temporal hours', () => `${day} ${t}`);
        }
        if (s.shadowEnd) ok(SD.onPlate(s.shadowEnd, -1e-9), 'sundial: drawn shadow end on the plate', () => `${day} ${t}`);
        ok(fmtOk(s.say), 'sundial: status format', () => s.say);
      }
    }
    // the unequal hour lines really pass through the tip at temporal hours
    for (const k of [2, 4, 6, 8, 10]) {
      const tk = SD.temporalToTime(k, day);
      const tip = SD.shadowOf(SD.PIN_TOP, 15 * (tk - 12), dd.dec);
      if (tip && SD.onPlate(tip)) {
        const s = SD.sundial('unequal', day, tk);
        ok(Math.abs(s.k - k) < 1e-9, 'sundial: tip on hour line k at temporal hour k', () => `${day} ${k}`);
      }
    }
  }
  for (const W of WIDTHS) {
    for (const mode of ['unequal', 'equal'] as const) {
      const f = SD.dialFrame(W);
      const labs = SD.hourLabels(mode, W);
      const boxes = labs.map(SD.dlBox);
      for (const b of boxes) ok(b.x0 >= f.x0 && b.x1 <= f.x1 && b.y0 >= f.y0 && b.y1 <= f.y1, `sundial: hour numeral inside (${mode})`, () => `W=${W} ${b.name}`);
      for (let i = 0; i < boxes.length; i++)
        for (let j = i + 1; j < boxes.length; j++) ok(!overlap(boxes[i], boxes[j], 0), `sundial: hour numerals collide (${mode})`, () => `W=${W} ${boxes[i].name} ${boxes[j].name}`);
      for (const l of SD.hourLines(mode)) for (const r of l.runs) for (const p of r) ok(SD.onPlate(p, -1e-9), 'sundial: hour line on plate', () => `${l.key}`);
    }
  }
  for (const day of [1, 80, 172, 266, 355])
    for (const t of [5, 7.5, 9.25, 12, 15, 17.9, 20])
      for (const mode of ['unequal', 'equal'] as const) {
        const s = SD.sundial(mode, day, t);
        checkRef({ ep: 0, t, dec: s.dec, dateText: '21 Jun', read: s.read, gapSec: s.read === null ? null : Math.round(Math.abs(s.read - t) * 3600), gapNote: mode === 'equal' ? undefined : 'an hour of daylight is not a clock hour', noRead: 'no shadow', hi: 'clock' }, `sundial ${day} ${t} ${mode}`);
      }
}

/* ───────────── 2 water clock ───────────── */
{
  for (const v of ['cyl', 'slope', 'head'] as const)
    for (const clog of range(0, 0.3, 0.05)) {
      let prev = v === 'head' ? -1 : 2;
      for (const t of range(0, WC.NIGHT, 0.25)) {
        const w = WC.waterClock(v, t, clog);
        ok(w.h >= -1e-12 && w.h <= 1 + 1e-12, 'water: level in 0..1', () => `${v} ${t}`);
        ok(v === 'head' ? w.h >= prev - 1e-12 : w.h <= prev + 1e-12, 'water: level moves one way', () => `${v} ${t}`);
        prev = w.h;
        if (v === 'cyl' && clog === 0) ok(near(w.h, (1 - t / WC.NIGHT) ** 2, 1e-6), 'water: straight bowl h = (1 − t/T)²', () => `${t}`);
        if (v === 'head') ok(near(w.h, Math.min(1, (t * (1 - clog)) / WC.NIGHT), 1e-12), 'water: constant head is linear', () => `${t}`);
        // independent: integrate dh/dt = −k√h/A(h) numerically for the slope vessel
        ok(near(w.errSec, (w.elapsed - t) * 3600, 1e-9), 'water: error = reading − truth', () => `${v} ${t}`);
      }
      if (clog === 0) ok(v === 'head' ? WC.worstError(v) < 1 : WC.worstError(v) > 600, 'water: only constant head keeps time', () => `${v} ${WC.worstError(v)}`);
    }
  // Euler check of the sloping bowl
  {
    const k = (2 + (4 * (WC.TOP_OVER_BOTTOM - 1)) / 3 + (2 * (WC.TOP_OVER_BOTTOM - 1) ** 2) / 5) / WC.NIGHT;
    let h = 1;
    const dt = 1e-4;
    for (let t = 0; t < 6; t += dt) h -= (dt * k * Math.sqrt(Math.max(h, 0))) / WC.radius('slope', h) ** 2;
    ok(Math.abs(h - WC.level('slope', 6)) < 2e-3, 'water: sloping bowl vs Euler integration', () => `${h} ${WC.level('slope', 6)}`);
  }
  ok(WC.worstError('slope') < WC.worstError('cyl'), 'water: sloping walls beat straight ones');
  for (const W of WIDTHS)
    for (const v of ['cyl', 'slope', 'head'] as const) {
      const f = WC.vesselFrame(W, v);
      ok(f.cx + f.rAt(1) + 40 < f.c.x0, 'water: vessel and its scale left of the chart', () => `W=${W} ${v}`);
      ok(f.c.x1 <= W && f.c.x0 < f.c.x1 - 60, 'water: chart inside', () => `W=${W}`);
      for (const p of WC.chartCurve(v, W)) ok(p.x >= f.c.x0 - 1e-9 && p.x <= f.c.x1 + 1e-9 && p.y >= f.c.y0 - 1e-9 && p.y <= f.c.y1 + 1e-9, 'water: curve inside chart', () => `W=${W} ${v}`);
      ok(f.cx - f.rAt(1) >= 0, 'water: vessel inside', () => `W=${W} ${v}`);
    }
  for (const v of ['cyl', 'slope', 'head'] as const)
    for (const t of [0, 3, 6, 9, 11.75, 12]) {
      const w = WC.waterClock(v, t);
      checkRef({ ep: 1, t: (WC.SUNSET + t) % 24, dec: 0, dateText: 'at the equinox', read: w.empty ? null : w.read % 24, gapSec: w.empty ? null : Math.round(Math.abs(w.errSec)), gapNote: w.errSec > 0 ? 'running ahead' : 'running behind', noRead: 'empty bowl', hi: 'clock' }, `water ${v} ${t}`);
    }
}

/* ───────────── 3 sine quadrant ───────────── */
{
  for (let day = 1; day <= 365; day += 2)
    for (let t = Q.T_MIN; t <= Q.T_MAX + 1e-9; t += 0.25) {
      const tq = Q.quadrant(day, t, 30); // altitude independent of date/time
      ok(tq.hMeas === 30, 'quadrant: date/time never rewrite the measured altitude', () => `${day} ${t}`);
      const hTrue = altitudeAt(t, declination(day));
      if (hTrue <= 0) continue;
      const h = Math.round(hTrue * 10) / 10; // the slider's 0.1° step
      const q = Q.quadrant(day, t, h);
      ok(q.onSun, 'quadrant: a slider value within 0.05° counts as on the sun', () => `${day} ${t}`);
      ok(!q.tooHigh, 'quadrant: the true sun is never too high', () => `${day} ${t}`);
      // independent: hour angle from astro.hourAngle
      const H2 = hourAngle(h, q.dec);
      ok(q.valid && H2 !== null && near(q.Hdeg, H2, 1e-9), 'quadrant: H matches astro', () => `${day} ${t}`);
      // board construction: bead at the divisor on the dividing thread sits over the difference
      for (const W of [326, 440, 600]) {
        const s = Q.quadScene(q, 4, W);
        const g = s.g;
        ok(near(s.sinX - g.AX, 60 * q.sinh * g.U, 1e-9), 'quadrant: 60 sin h on the top scale', () => `${day} ${t}`);
        ok(near(s.thread.b.x - g.AX, 60 * Math.sin(h * D) * g.U, 1e-9), 'quadrant: thread rim point straight above its sine', () => `${day} ${t}`);
        ok(near(s.beadA.x - g.AX, 60 * Math.abs(q.a) * g.U, 1e-9), 'quadrant: date bead reads sin φ |sin δ|', () => `${day} ${t}`);
        ok(near(s.beadB.y - g.AY, 60 * q.b * g.U, 1e-9), 'quadrant: divisor bead reads cos φ cos δ on the left scale', () => `${day} ${t}`);
        ok(near(s.divBead!.x, s.diffX, 1e-6), 'quadrant: dividing bead stands over the difference', () => `${day} ${t} ${s.divBead!.x} ${s.diffX}`);
        const rimTheta = Math.atan2(s.divEnd!.x - g.AX, s.divEnd!.y - g.AY) / D;
        const Hb = q.diff >= 0 ? 90 - rimTheta : 90 + rimTheta;
        ok(near(Hb, q.Hdeg, 1e-6), 'quadrant: rim reading gives H', () => `${day} ${t}`);
        for (const p of [s.thread.b, s.beadA, s.beadB, s.divEnd!, s.divBead!]) ok(Math.hypot(p.x - g.AX, p.y - g.AY) <= g.R + 1e-6, 'quadrant: marks inside the quarter', () => `${day} ${t}`);
        ok(s.diffX >= g.AX - 1e-9 && s.diffX <= g.AX + g.R + 1e-9, 'quadrant: difference on the top scale', () => `${day} ${t}`);
        ok(g.H <= 600 && g.AX - g.BAND >= 0 && g.AX + g.R + 30 <= W, 'quadrant: board inside the view', () => `W=${W}`);
      }
      // reading = true time, to the slider's resolution: 0.05° of altitude at most a few minutes
      ok(Math.abs(q.errSec!) < 8 * 60, 'quadrant: reading near the true time', () => `${day} ${t} ${q.errSec}`);
    }
  // impossible altitudes are flagged, not moved
  for (let day = 1; day <= 365; day += 7)
    for (const h of range(0, 90, 0.5)) {
      const q = Q.quadrant(day, 9, h);
      ok(q.hMeas === h, 'quadrant: altitude kept as entered', () => `${day} ${h}`);
      ok(q.tooHigh === h > noonAltitude(declination(day)), 'quadrant: too-high flag', () => `${day} ${h}`);
      if (q.tooHigh) ok(!q.valid && q.read === null && Q.sightSay(q).includes('never gets this high'), 'quadrant: too-high gives no hour', () => `${day} ${h}`);
      ok(fmtOk(Q.sightSay(q)), 'quadrant: status format', () => Q.sightSay(q));
    }
}

/* ───────────── 4 verge and foliot ───────────── */
{
  for (const r of range(VG.R_MIN, VG.R_MAX, VG.R_STEP))
    for (const drive of range(VG.DRIVE_MIN, VG.DRIVE_MAX, 0.01)) {
      const T = VG.period(r, drive);
      ok(near(T / VG.period(VG.R_SET, 1), Math.sqrt(VG.inertia(r) / VG.inertia(VG.R_SET) / drive), 1e-9), 'verge: T ∝ √(I/τ)', () => `${r} ${drive}`);
      // independent: simulate θ'' = −(τ/I) sign θ from the extreme and time a full swing
      if (Math.abs(r - 0.5) < 1e-9 && Math.abs(drive - 1) < 1e-9) {
        const acc = (32 * VG.A_SWING) / (T * T);
        let th = VG.A_SWING;
        let om = 0;
        let t = 0;
        const dt = 1e-5;
        let crossings = 0;
        let last = th;
        while (crossings < 2) {
          om -= acc * Math.sign(th) * dt;
          th += om * dt;
          t += dt;
          if (Math.sign(th) !== Math.sign(last)) crossings++;
          last = th;
        }
        ok(Math.abs(t - T / 2 - T / 4) < 1e-3 * T + 1e-4, 'verge: simulated swing matches T', () => `${t} ${T}`);
      }
      // the animation's angle stays within the swing, and the crown only moves forward
      let prevC = -1;
      for (let k = 0; k <= 40; k++) {
        const tt = (k / 40) * 2 * T;
        const a = VG.foliotAngle(tt, T);
        if (Math.abs(r - 0.62) < 1e-9 || Math.abs(r - 0.2) < 1e-9) {
          ok(Math.abs(a) <= VG.A_SWING + 1e-9, 'verge: foliot within its swing', () => `${r} ${tt}`);
          const c = VG.crownTeeth(tt, T);
          ok(c >= prevC - 1e-9, 'verge: crown wheel never runs back', () => `${r} ${tt}`);
          prevC = c;
        }
      }
    }
  ok(near(VG.crownTeeth(2 * VG.T_SET, VG.T_SET) - VG.crownTeeth(0, VG.T_SET), 2, 1e-9), 'verge: one tooth per swing');
  ok(range(VG.R_MIN, VG.R_MAX, VG.R_STEP).some((r) => Math.abs(VG.errPerDay(r, 1)) <= VG.KEEP_TOL), 'verge: the slider can reach "keeps time"');
  ok(Math.abs(VG.errPerDay(VG.R_SET, VG.DRIVE_MIN)) > 3600 && Math.abs(VG.errPerDay(VG.R_SET, VG.DRIVE_MAX)) > 3600, 'verge: drive changes rate by hours a day');
  // continuity of the foliot angle (no jumps between arcs)
  for (const T of [1.07, 2, 3.01]) for (let k = 0; k < 400; k++) ok(Math.abs(VG.foliotAngle((k + 1) * T / 400, T) - VG.foliotAngle((k * T) / 400, T)) < 0.05, 'verge: smooth swing', () => `${T} ${k}`);
  for (const W of WIDTHS) for (const r of [VG.R_MIN, VG.R_MAX]) {
    const g = VG.foliotScene(VG.A_SWING, r, W);
    for (const p of [...g.ends, ...g.weights]) ok(p.x >= 0 && p.x <= W && p.y >= 0 && p.y <= g.H, 'verge: foliot inside', () => `W=${W}`);
  }
}

/* ───────────── 5 pendulum and deadbeat ───────────── */
{
  for (const L of range(PG.L_MIN, PG.L_MAX, 0.01)) {
    ok(near(PG.period(L), 2 * Math.PI * Math.sqrt(L / 9.80665), 1e-12), 'pendulum: T formula');
    ok(fmtOk(fmtSec(Math.abs(PG.errPerDay(L)))), 'pendulum: error format');
  }
  ok(near(PG.period(PG.SECONDS_L), 2, 1e-12) && Math.abs(PG.SECONDS_L - 0.9936) < 1e-4, 'pendulum: seconds pendulum 0.994 m');
  ok(Math.abs(PG.errPerDay(PG.SECONDS_L, 1) + 43.5) < 0.5, 'pendulum: 1 mm loses ~43 s a day', () => `${PG.errPerDay(PG.SECONDS_L, 1)}`);
  ok(PG.snapL(PG.SECONDS_L + 0.004) === PG.SECONDS_L && PG.isSeconds(PG.snapL(0.99)), 'pendulum: slider snaps to seconds');
  // the static geometry
  const P = DB.PIVOT;
  for (const k of [0, 1] as const) {
    const g = DB.REGIONS[k];
    ok(near(Math.hypot(g.A[0], g.A[1]), 1 - DB.DA, 1e-9), 'deadbeat: locking corner depth', () => `${k}`);
    ok(near(Math.hypot(g.B[0], g.B[1]), 1 - DB.DB, 1e-9), 'deadbeat: impulse corner depth', () => `${k}`);
    const rhoA = Math.hypot(g.A[0] - P[0], g.A[1] - P[1]);
    ok(near(rhoA, k === 0 ? g.rho2 : g.rho1, 1e-12), 'deadbeat: locking face is the arc about the pivot', () => `${k}`);
    // the polygon's locking arc points are all at that radius about P
    for (let i = 0; i <= DB.CORNER_A; i++) ok(near(Math.hypot(DB.PALLETS[k][i][0] - P[0], DB.PALLETS[k][i][1] - P[1]), rhoA, 1e-9), 'deadbeat: locking arc concentric with the pivot', () => `${k} ${i}`);
  }
  ok(near(DB.PIVOT_Y, Math.SQRT2, 1e-12) && near(DB.SPAN / DB.PITCH, 7.5, 1e-12), 'deadbeat: pallets span 7½ teeth');
  // locked wheel: while the tooth rests on a locking face the anchor can swing on with no recoil
  // and the wheel cannot move forward
  // frame-by-frame run at several fps and periods
  for (const T of [PG.period(PG.L_MIN), 2, PG.period(PG.L_MAX)])
    for (const fps of [20, 30, 60, 144]) {
      let s = DB.initialState();
      const w0 = s.w;
      let maxGap = 0;
      let recoil = 0;
      let back = 0;
      let locks = 0;
      let impulses = 0;
      let lastFace = '';
      let lockedMoves = 0;
      const frames = Math.round(fps * T * 6);
      for (let f = 0; f < frames; f++) {
        const n = DB.stepMovement(s, 1 / fps, T);
        recoil += n.recoil;
        if (n.w < s.w - 1e-12) back++;
        const a = DB.AMP * Math.sin(n.phase);
        const c = DB.contactInfo(n.w, a);
        ok(!DB.collides(n.w, a), 'deadbeat: no interpenetration in any frame', () => `T=${T} fps=${fps} f=${f}`);
        maxGap = Math.max(maxGap, c.gap);
        // a locked wheel moves 0; allow an invisible 1e-4 rad (0.006°) for the frame where the
        // tooth crosses from the locking corner onto the impulse face
        if (lastFace === 'lock' && c.face === 'lock' && n.w - s.w > 1e-4) lockedMoves++;
        if (c.face !== lastFace) (c.face === 'lock' ? locks++ : impulses++);
        lastFace = c.face;
        s = { phase: n.phase, w: n.w };
      }
      ok(recoil === 0, 'deadbeat: no recoil', () => `T=${T} fps=${fps} recoil=${recoil}`);
      ok(back === 0, 'deadbeat: wheel never turns backward', () => `T=${T} fps=${fps}`);
      ok(maxGap < 1e-5, 'deadbeat: no air gap at contact (every frame within 1e-5 of the radius)', () => `T=${T} fps=${fps} gap=${maxGap}`);
      ok(near((s.w - w0) / DB.PITCH, 6, 0.02), 'deadbeat: one tooth per period', () => `T=${T} fps=${fps} teeth=${(s.w - w0) / DB.PITCH}`);
      ok(lockedMoves === 0, 'deadbeat: wheel dead still while locked', () => `T=${T} fps=${fps} ${lockedMoves}`);
      ok(locks >= 11 && impulses >= 11, 'deadbeat: lock and impulse each twice a period', () => `T=${T} fps=${fps} ${locks} ${impulses}`);
    }
  // the close-up and main views stay inside their drawing
  for (const W of WIDTHS)
    for (const L of [PG.L_MIN, 0.5, PG.SECONDS_L, PG.L_MAX]) {
      let s = DB.initialState();
      for (let f = 0; f < 40; f++) {
        const n = DB.stepMovement(s, 0.05, PG.period(L));
        s = { phase: n.phase, w: n.w };
        const a = DB.AMP * Math.sin(s.phase);
        const g = escScene(s.w, a, L, W);
        for (const p of [g.pivot, g.centre, { x: g.bob.x, y: g.bob.y + g.bobR }]) ok(p.x >= 0 && p.x <= W && p.y >= 0 && p.y <= g.H - 20, 'escapement view: inside', () => `W=${W} L=${L}`);
        for (const q of g.wheel) {
          const p = g.P(q);
          ok(p.x >= 0 && p.x <= W && p.y >= 0, 'escapement view: wheel inside', () => `W=${W}`);
        }
        const z = g.zoom!;
        ok(z.cx - z.r >= 0 && z.cx + z.r <= W && z.cy + z.r + 16 <= g.H - 12, 'escapement view: close-up inside', () => `W=${W} L=${L} ${JSON.stringify({ cx: z.cx, cy: z.cy, r: z.r, H: g.H })}`);
        // contact point is centred in the close-up
        const cp = z.f(g.contact.point);
        ok(near(cp.x, z.cx, 1e-9) && near(cp.y, z.cy, 1e-9), 'escapement view: close-up centred on the contact', () => `W=${W}`);
      }
    }
  // tilt (out of beat): small tilts keep going, large ones stop it
  ok(DB.teethPerPeriod(0.5 * D) > 0.99 && DB.teethPerPeriod(-0.5 * D) > 0.99, 'deadbeat: in beat within 0.5°');
}

/* ───────────── 6 at sea ───────────── */
{
  ok(Math.abs(SEA.gravityRate(4.9, 48.8) + 128) < 1.5, 'sea: Paris→Cayenne gravity ≈ −128 s/day (Richer saw −148)', () => `${SEA.gravityRate(4.9, 48.8)}`);
  ok(Math.abs(SEA.heaveRate(2) + 85.5) < 1, 'sea: 2 m swell at 8 s ≈ −85 s/day', () => `${SEA.heaveRate(2)}`);
  ok(near(SEA.errorAfter({ clock: 'balance', heave: 3, dest: 18, heat: true }, SEA.H4_DAYS), -5, 1e-9), 'sea: H4 5 s slow after 81 d 5 h');
  ok(near(SEA.arcminOf(-5), 1.25, 1e-12), 'sea: 5 s = 1.25′');
  ok(near(SEA.kmOf(1, 0) * 1000, 463.8, 1e-9), 'sea: 1 s at the equator = 463.8 m');
  ok(SEA.errorAfter({ clock: 'pendulum', heave: 0, dest: SEA.PORTSMOUTH, heat: false }, 90) === 0, 'sea: on land the pendulum is perfect');
  for (const heave of range(0, SEA.HEAVE_MAX, 0.5))
    for (const dest of [SEA.JAMAICA, 30, SEA.PORTSMOUTH])
      for (const heat of [false, true])
        for (const clock of ['pendulum', 'balance'] as const) {
          const v = { clock, heave, dest, heat };
          for (const W of WIDTHS) {
            const f = SEA.chartFrame(W);
            for (const p of SEA.errorCurve(v, W, SEA.DAYS_MAX)) ok(p.x >= f.m.l - 1e-9 && p.x <= f.m.l + f.iw + 1e-9 && p.y >= f.m.t - 1e-9 && p.y <= f.m.t + f.ih + 1e-9, 'sea: curve inside chart', () => `W=${W}`);
            // the drawn height is the computed error (log scale)
            const d = 42;
            const a = SEA.arcminOf(SEA.errorAfter(v, d));
            if (a >= SEA.Y_MIN && a <= SEA.Y_MAX) {
              const y = f.y(a);
              const back = 10 ** (Math.log10(SEA.Y_MIN) + ((f.m.t + f.ih - y) / f.ih) * (Math.log10(SEA.Y_MAX) - Math.log10(SEA.Y_MIN)));
              ok(near(back, a, 1e-9), 'sea: chart height = error', () => `W=${W}`);
            }
            // prize lines' labels inside
            ok(f.y(30) + 11 <= f.m.t + f.ih && f.y(60) - 14 >= f.m.t, 'sea: prize labels inside chart', () => `W=${W}`);
            ok(f.y(60) - 4 < f.y(30) + 11 - 10 - 1, 'sea: prize labels apart (above 1°, below 30′)', () => `W=${W}`);
          }
          for (const days of [0, 1, 42, 81.2, 90]) {
            const e = SEA.errorAfter(v, days);
            ok(Number.isFinite(e), 'sea: finite error');
            const p = SEA.prizeFor(e);
            ok(p === null ? SEA.arcminOf(e) > 60 : SEA.arcminOf(e) <= p.arcmin, 'sea: prize text agrees', () => `${e}`);
            if (clock === 'balance') ok(near(e, SEA.H4_RATE * days, 1e-9), 'sea: balance has no g or heave term');
            checkRef({ ep: clock === 'balance' ? 5 : 4, t: 12, dec: 0, dateText: `at sea, day ${Math.round(days)}`, read: 12 + e / 3600, gapSec: Math.abs(e), metresPerSec: SEA.M_PER_S_EQ * Math.cos((SEA.latOnDay(v, days) * Math.PI) / 180), distWord: 'of longitude', rate: Math.abs(SEA.rateOn(v, Math.min(days, 41)).total) || null, hi: 'clock' }, `sea ${clock} ${heave} ${dest} ${days}`);
          }
        }
}

/* ───────────── 7 quartz ───────────── */
{
  ok(QZ.stageFreq(QZ.STAGES) === 1 && QZ.F0 === 2 ** 15, 'quartz: 15 halvings of 32 768 Hz make 1 Hz');
  for (let T = QZ.T_MIN; T <= QZ.T_MAX; T++) {
    ok(QZ.errPerDay(T) <= 0, 'quartz: off the turnover it only runs slow', () => `${T}`);
    ok(near(QZ.errPerDay(T), (86400 * QZ.hzShift(T)) / QZ.F0, 1e-12), 'quartz: error = 86 400 Δf/f');
  }
  ok(QZ.errPerDay(QZ.TURNOVER) === 0, 'quartz: exact at 25 °C');
  for (const W of WIDTHS) {
    const ch = QZ.chainLayout(W);
    for (const b of ch.items) {
      ok(b.x >= 0 && b.x + b.w <= W, 'quartz: chain box inside', () => `W=${W}`);
      ok(QZ.hzText(b.f).length * (W < 560 ? 9.5 : 10) * 0.6 <= b.w - 2, 'quartz: frequency fits its box', () => `W=${W} ${QZ.hzText(b.f)} w=${b.w}`);
    }
  }
}

/* ───────────── 8 atomic ───────────── */
{
  ok(AT.flipFraction(0) === 1 && AT.flipFraction(AT.PEAK_TOL) >= 0.9, 'atomic: peak and tolerance');
  ok(AT.grouped(AT.F_CS) === '9 192 631 770', 'atomic: cycle count text');
  for (const e of AT.ERAS) ok(e.perDay > 0 && fmtOk(fmtSec(e.perDay)) && fmtOk(fmtSec(e.perDay * 365.25)), 'atomic: era formatting', () => e.id);
  ok(near(AT.ERAS[0].perDay * 300 * 365.25, 1, 1e-12), 'atomic: 1955 = 1 s in 300 years');
  for (const W of WIDTHS) {
    const f = AT.curveFrame(W);
    for (const d of range(AT.DET_MIN, AT.DET_MAX, 50)) ok(f.x(d) >= f.m.l - 1e-9 && f.x(d) <= f.m.l + f.iw + 1e-9, 'atomic: marker inside', () => `W=${W}`);
    ok(f.iw > 200, 'atomic: curve wide enough for its three axis labels', () => `W=${W}`);
  }
}

/* ───────────── 9 GPS ───────────── */
{
  for (const b of range(-GPS.B_MAX, GPS.B_MAX, 1e-7))
    for (const bHat of [-GPS.B_MAX, 0, b, GPS.B_MAX])
      for (const relHours of [0, 1, 12, 24]) {
        const s = { n: 3, b, bHat, relHours };
        const f = GPS.fix(s);
        // independent: each line's distance from the origin = its δ
        f.d.forEach((d, i) => {
          const seg = GPS.clipLine(GPS.SATS[i].u, d, GPS.halfWidth(s));
          ok(seg !== null, 'gps: every range line is in view', () => `${b} ${bHat} ${relHours}`);
          if (seg) {
            const u = GPS.SATS[i].u;
            for (const p of seg) ok(near(u.x * p.x + u.y * p.y, -d, 1e-6), 'gps: line at its range offset', () => `${i}`);
            for (const p of seg) ok(Math.abs(p.x) <= GPS.halfWidth(s) + 1e-6 && Math.abs(p.y) <= GPS.halfWidth(s) + 1e-6, 'gps: clipped to view');
          }
        });
        if (relHours === 0) {
          // a common clock error moves all three lines together: they meet only when it is solved
          const r = Math.abs(b - bHat) * GPS.C;
          ok((f.size <= GPS.MEET_TOL) === (r <= GPS.MEET_TOL * 0.5) || Math.abs(r - GPS.MEET_TOL * 0.5) < 2, 'gps: meet only when the clock is solved', () => `${b} ${bHat} size=${f.size} r=${r}`);
        }
        for (const p of f.pts) ok(Math.abs(p.x) <= GPS.halfWidth(s) && Math.abs(p.y) <= GPS.halfWidth(s), 'gps: fix inside the view', () => `${b} ${bHat} ${relHours}`);
      }
  // flat approximation: sag over the view is under a pixel at the widest view
  const hwMax = GPS.halfWidth({ n: 3, b: GPS.B_MAX, bHat: -GPS.B_MAX, relHours: 24 });
  ok((hwMax * hwMax) / (2 * GPS.ORBIT_KM * 1000) < hwMax / 300, 'gps: range sphere is flat at this scale', () => `${hwMax}`);
  ok(Math.abs(GPS.C * GPS.REL_PER_DAY - 11570) < 50, 'gps: 38.6 µs ≈ 11.6 km');
  // B_STEP lands exactly on b (the slider can solve)
  ok(range(-GPS.B_MAX, GPS.B_MAX, GPS.B_STEP).some((v) => Math.abs(v - 1e-6) * GPS.C <= GPS.MEET_TOL), 'gps: slider can reach the solution');
}

report();
