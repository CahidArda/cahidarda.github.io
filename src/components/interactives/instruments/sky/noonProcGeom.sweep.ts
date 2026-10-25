/**
 * Sweep of the noon-sight procedure plate (NoonSightProcedure.tsx) through ./noonProcGeom.ts.
 *
 *   node --experimental-strip-types --import ./src/components/interactives/instruments/sky/tsreg.mjs \
 *     src/components/interactives/instruments/sky/noonProcGeom.sweep.ts
 *
 * (tsreg.mjs lets sextantView.ts's extensionless import resolve, as Vite does.)
 * Sweeps every day of 2026 × latitudes 70° S to 70° N (every 1°, plus the four places) × view
 * widths 280..600 and asserts:
 *   maths    an exact reading reduces back to the true latitude; an error of x′ in the reading
 *            gives about x′ in φ; δ and h agree with an independent alt/az computation at noon
 *   scale    reference drawing: the drawn φ, δ and h angles equal the numbers (one scale)
 *   world    the reference does not depend on the step (it is a function of the sight only)
 *   bounds   every line end, arc and label inside the drawing
 *   labels   every label found a free spot (no label/label, label/line or label/arc overlap)
 *   honesty  the eyepiece status agrees with the reading; the drum + arm show the reading;
 *            the correction staircase ends at h; the result marker sits at φ on the scale
 */
import { near, ok, report } from '../../../../../scripts/plates/check.ts';
import {
  MIN_ALT,
  ON_TOL,
  PLACES,
  READ_MAX,
  T_MAX,
  T_MIN,
  arcPts,
  corrScene,
  dm,
  eyeScene,
  eyeSay,
  skySay,
  resultSay,
  labelBox,
  readScene,
  reduce,
  refScene,
  resultScene,
  sight,
  skyScene,
  snapTime,
  splitReading,
  sunAltAz,
  type Place,
} from './noonProcGeom.ts';

// View widths: stacked (plate body < 700px) the views take the whole body, ~320 on a 390px
// phone up to 699; side by side (≥ 700) each column is ≥ ~340 and ~440 on a wide desktop plate.
const WIDTHS = [320, 340, 360, 400, 440, 520, 600, 700];
const LATS = Array.from({ length: 141 }, (_, i) => i - 70);
const places: Place[] = [
  ...PLACES,
  ...LATS.map((lat) => ({ id: `L${lat}`, name: `lat ${lat}`, lat, lon: 0 })),
];
const DAYS = Array.from({ length: 365 }, (_, i) => i + 1);
const inBox = (p: { x: number; y: number }, W: number, H: number, m = 0) =>
  p.x >= m - 1e-9 && p.y >= m - 1e-9 && p.x <= W - m + 1e-9 && p.y <= H - m + 1e-9;

let sights = 0;
let invisible = 0;
for (const pl of places) {
  const fullDays = pl.id.startsWith('L') && pl.lat % 5 !== 0 ? DAYS.filter((d) => d % 15 === 1) : DAYS;
  for (const doy of fullDays) {
    const s = sight(doy, pl);
    const ctx = `${pl.name} day ${doy}`;
    sights++;
    if (!s.visible) {
      invisible++;
      ok(s.hTrue < MIN_ALT, 'visible flag', () => ctx);
      continue;
    }
    // ── maths ──
    const r = reduce(s, s.hs);
    ok(near(r.phi, pl.lat, 1e-6), 'maths: exact reading → true φ', () => `${ctx} φ=${r.phi}`);
    ok(near(r.h, s.hTrue, 1e-6), 'maths: corrections undo the reading', () => ctx);
    ok(s.hs <= READ_MAX && s.hs >= 0, 'slider: reading reachable', () => `${ctx} hs=${s.hs}`);
    const noon = sunAltAz(pl.lat, s.dec, 12);
    ok(near(noon.alt, s.hTrue, 1e-6), 'maths: h = 90 − |φ − δ| (alt/az at noon)', () => `${ctx} ${noon.alt} vs ${s.hTrue}`);
    ok(Math.abs(Math.abs(noon.az) - (s.bearing === 'S' ? 0 : 180)) < 1e-6 || s.hTrue > 89.9, 'maths: noon sun on the meridian, right side', () => `${ctx} az=${noon.az}`);
    for (const e of [-5, -1, 1, 5]) {
      const ri = reduce(s, s.hs + e / 60);
      const dphi = (ri.phi - pl.lat) * 60;
      const want = s.bearing === 'S' ? -e : e;
      ok(Math.abs(dphi - want) < 0.15 * Math.abs(e) + 0.02, "maths: x′ error in reading → ≈x′ in φ", () => `${ctx} e=${e} dφ=${dphi}`);
    }
    // the corrections are the textbook sizes (h_s on a 3 m eye: dip −3.0′, SD ≈ 16′, refraction ≤ 10′ at 5°)
    ok(near(r.dip * 60, -3.05, 0.01), 'maths: dip 1.76√3', () => `${ctx} ${r.dip * 60}`);
    ok(r.semi * 60 > 15.7 && r.semi * 60 < 16.4, 'maths: semi-diameter 15.8′..16.3′', () => `${ctx} ${r.semi * 60}`);
    ok(r.refr * 60 <= 0 && r.refr * 60 > -10.5, 'maths: refraction 0..10′', () => `${ctx} ${r.refr * 60}`);
    ok(!/e[+-]/.test(dm(r.phi) + dm(s.dec) + dm(s.hs)), 'format: no exponent', () => ctx);

    // eyepiece honesty: aligned exactly at hs, off by more than the tolerance elsewhere
    const e0 = eyeScene(s, s.hs, 360);
    ok(e0.status === 'on' && Math.abs(e0.limb) < 1e-9, 'eyepiece: on at the exact reading', () => `${ctx} limb=${e0.limb}`);
    for (const e of [-ON_TOL * 1.05, ON_TOL * 1.05, -1, 1, 3]) {
      const ei = eyeScene(s, s.hs + e, 360);
      ok(ei.status === (e < 0 ? 'above' : 'below'), 'eyepiece: status sign', () => `${ctx} e=${e} ${ei.status}`);
      ok(near(ei.limb, -e, 1e-9), 'eyepiece: limb = −(reading error)', () => `${ctx}`);
    }
    // the drum reading: arm + drum show the exact reading (to the drum's 0.1′)
    const sp = splitReading(s.hs);
    ok(Math.abs(sp.deg + sp.min / 60 - s.hs) <= 0.05 / 60 + 1e-9 && sp.min < 60, 'drum: split = reading', () => ctx);

    const doWidths = pl.id.startsWith('L') ? (pl.lat % 5 === 0 && doy % 5 === 1 ? WIDTHS : []) : doy % 3 === 0 ? WIDTHS : [];
    for (const W of doWidths) {
      const c = `${ctx} W=${W}`;
      // ── reference ──
      const g = refScene(s, W);
      ok(near(g.drawnPhi, pl.lat, 1e-9), 'scale: drawn φ = φ', () => c);
      ok(near(g.drawnDec, s.dec, 1e-6), 'scale: drawn sun ray at δ', () => `${c} ${g.drawnDec} vs ${s.dec}`);
      ok(near(g.drawnH, s.hTrue, 1e-9), 'scale: drawn h arc = h', () => `${c} ${g.drawnH} vs ${s.hTrue}`);
      // horizon ⟂ the radius to you
      const hx = g.horSun.x - g.obs.x;
      const hy = g.horSun.y - g.obs.y;
      const rx = g.obs.x - g.cx;
      const ry = g.obs.y - g.cy;
      ok(Math.abs(hx * rx + hy * ry) < 1e-6 * Math.hypot(hx, hy) * Math.hypot(rx, ry), 'scale: horizon tangent', () => c);
      // all rays parallel at δ
      for (const ray of g.rays) {
        const a = Math.atan2(-(ray.a.y - ray.b.y), ray.a.x - ray.b.x) * (180 / Math.PI);
        ok(Math.abs(a - s.dec) < 1e-6 || Math.hypot(ray.a.x - ray.b.x, ray.a.y - ray.b.y) < 1, 'scale: rays parallel', () => c);
        ok(near(Math.hypot(ray.b.x - g.cx, ray.b.y - g.cy), g.Re, 1e-6), 'scale: rays end on the Earth', () => c);
      }
      for (const sg of g.segs) {
        ok(inBox(sg.a, W, g.H) && inBox(sg.b, W, g.H), `bounds: ref line ${sg.id.replace(/\d/, '')}`, () => `${c} ${JSON.stringify(sg)}`);
      }
      ok(inBox({ x: g.cx - g.Re, y: g.cy - g.Re }, W, g.H, 2) && inBox({ x: g.cx + g.Re, y: g.cy + g.Re }, W, g.H, 2), 'bounds: Earth', () => c);
      for (const pts of [arcPts(g.cx, g.cy, g.phiArcR, 0, pl.lat), arcPts(g.obs.x, g.obs.y, g.hArcR, g.hFrom, g.hTo)])
        ok(pts.every((p) => inBox(p, W, g.H)), 'bounds: arcs', () => c);
      for (const [id, l] of Object.entries(g.labels)) {
        ok(l.placed, `labels: ${id} found a free spot`, () => `${c} at ${l.x.toFixed(0)},${l.y.toFixed(0)}`);
        const b = labelBox(l);
        ok(b.x0 >= 0 && b.y0 >= 0 && b.x1 <= W && b.y1 <= g.H, `bounds: label ${id}`, () => c);
      }
      // the reference is a function of the sight and width only: same object for every step
      ok(JSON.stringify(refScene(s, W)) === JSON.stringify(g), 'world: reference stable', () => c);

      // ── sky (step 1) ──
      for (const t of [T_MIN, 10, 11.5, 11.96, 12, 12.04, 13, T_MAX]) {
        const tt = snapTime(t);
        const k = skyScene(s, tt, W);
        if (k.now.alt > 0) {
          ok(inBox(k.sun, W, k.H), 'bounds: sky sun', () => `${c} t=${tt} ${JSON.stringify(k.sun)}`);
          ok(near(k.horizonY - k.sun.y, k.now.alt * k.ky, 1e-9), 'scale: sky sun height = alt', () => c);
        }
        for (const run of k.path) ok(run.every((p) => inBox(p, W, k.H)), 'bounds: sun path', () => c);
        if (tt === 12) {
          ok(near(k.sun.x, k.mid, 1e-6), 'sky: noon sun on the meridian line', () => `${c} ${k.sun.x} vs ${k.mid}`);
          ok(near(k.now.alt, s.hTrue, 1e-9), 'sky: noon alt = h', () => c);
          ok(Math.abs(k.rate) < 0.05, 'sky: not climbing at noon', () => `${c} ${k.rate}`);
        }
        ok(k.sunNoon.y <= k.sun.y + 1e-6, 'sky: noon is highest', () => `${c} t=${tt}`);
        const sb = labelBox({ x: W / 2, y: k.H - 10, text: skySay(k, tt), size: 12, anchor: 'middle' });
        ok(sb.x0 >= 2 && sb.x1 <= W - 2, 'bounds: sky status line', () => `${c} ${skySay(k, tt)}`);
        ok(sb.y0 > k.horizonY + 18, 'labels: sky status clear of the compass letters', () => c);
      }
      ok(skyScene(s, 12 - 0.03, W).now.alt <= s.hTrue, 'sky: snap only near the top', () => c);

      // ── eyepiece (step 2) ──
      for (const e of [-3, -0.3, 0, 0.3, 3]) {
        const ey = eyeScene(s, s.hs + e, W);
        ok(near(ey.cy - ey.sunY, ey.gap * ey.k, 1e-9), 'scale: eyepiece sun at gap × k', () => c);
        ok(near(ey.horY, ey.cy, 1e-9), 'eyepiece: horizon on the axis (untilted)', () => c);
        ok(ey.cy + ey.R + 5 < ey.H - 18 && ey.cy - ey.R - 5 > 18, 'bounds: eyepiece + its labels', () => c);
        ok(ey.visible === Math.abs(ey.gap) <= 2.5 + s.sd, 'eyepiece: visible = within the field', () => c);
        const eb = labelBox({ x: W / 2, y: ey.H - 10, text: eyeSay(ey), size: 12, anchor: 'middle' });
        ok(eb.x0 >= 2 && eb.x1 <= W - 2, 'bounds: eyepiece status line', () => `${c} ${eyeSay(ey)}`);
        ok(eb.y0 > ey.cy + ey.R + 5, 'labels: eyepiece status below the eyepiece', () => c);
      }
      // ── arc and drum (step 3) ──
      for (const rd of [0, 0.5, s.hs, 45.999, READ_MAX - 0.0001]) {
        const rs = readScene(rd, W);
        ok(inBox(rs.index, W, rs.H, 10), 'bounds: arc index', () => `${c} r=${rd}`);
        ok(near(rs.deg + rs.min / 60, rd, 1e-9), 'read: deg + min = reading', () => c);
        ok(rs.drum.cy + rs.drum.r < rs.H - 4 && rs.drum.cy - rs.drum.r - 22 > rs.arcY + 30, 'bounds: drum clear of the arc', () => c);
      }
      // ── corrections (step 4) ──
      for (const e of [0, -0.5, 0.5]) {
        const cs = corrScene(s, s.hs + e, W);
        ok(near(cs.tot[cs.tot.length - 1] / 60, cs.r.corr, 1e-9), 'corr: total = h − Hs', () => c);
        for (const rw of cs.rows) {
          ok(rw.xa >= cs.x0 - 1e-9 && rw.xb <= cs.x1 + 1e-9 && rw.xb >= cs.x0 - 1e-9, 'bounds: corr bars', () => `${c} ${rw.id}`);
          ok(near(rw.xb - rw.xa, (rw.to - rw.from) * cs.k, 1e-9), 'scale: corr bar = arcminutes × k', () => c);
          ok(rw.y + 12 < cs.H, 'bounds: corr rows', () => c);
          for (const [t, dy, sz] of [[rw.name, -6, 12], [rw.value, 9, 11]] as const) {
            const b = labelBox({ x: rw.nx, y: rw.y + dy, text: t, size: sz, anchor: 'start' });
            ok(b.x1 <= cs.x0 - 4, 'labels: corr names left of the bars', () => `${c} ${t} ${b.x1} vs ${cs.x0}`);
          }
        }
        ok(cs.rows[4].xa === cs.rows[3].xb, 'corr: h at the end of the last bar', () => c);
      }
      // ── result (step 6) ──
      for (const e of [0, 1, -1, 10, -30]) {
        const rr = reduce(s, s.hs + e / 60);
        const rs = resultScene(s, rr.phi, W);
        ok(rs.yMeas >= rs.top && rs.yMeas <= rs.bot, 'bounds: result marker on the scale', () => `${c} e=${e}`);
        ok(near(rs.yTrue - rs.yMeas, (rr.phi - pl.lat) * 60 * rs.k, 1e-9), 'scale: result offset = miss', () => c);
        const rb = labelBox({ x: W / 2, y: rs.H - 10, text: resultSay(rs.miss), size: 12, anchor: 'middle' });
        ok(rb.x0 >= 2 && rb.x1 <= W - 2 && rb.y0 > rs.bot, 'bounds: result status line', () => `${c} ${resultSay(rs.miss)}`);
        // the two line labels sit on opposite sides and must not overlap
        const tl = labelBox({ x: W - 12, y: rs.yTrue + (rs.miss > 0 ? 11 : -9), text: `${pl.name} 00°00.0′ N`, size: 12, anchor: 'end' });
        const ml = labelBox({ x: W - 12, y: rs.yMeas + (rs.miss > 0 ? -9 : 11), text: 'you: 00°00.0′ N', size: 12, anchor: 'end' });
        ok(!(tl.y0 < ml.y1 && ml.y0 < tl.y1) || Math.abs(rs.miss) < 0.05, 'labels: result labels apart', () => `${c} e=${e}`);
        ok(tl.y0 > 24 && ml.y0 > 24 && tl.y1 < rs.H - 22 && ml.y1 < rs.H - 22, 'bounds: result labels', () => `${c} e=${e}`);
      }
    }
  }
}
report([`${sights} sights, ${invisible} too low to measure (polar night or near)`]);
