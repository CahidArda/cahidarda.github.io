/**
 * Screen layout for the flat face views (Faces.tsx), pure so the sweep can check that what is
 * drawn agrees with the numbers: the sun's dot sits on the ecliptic at the drawn rete turn, the
 * rule's red edge hits the limb at the hour it reports, the alidade's edge hits the altitude
 * scale at its setting and the zodiac / calendar rings at the date.
 * Plate units → px: x = cx + u·k, y = cy − v·k (both faces drawn as you look at them).
 */
import { RC, RM, BACK, backPsi, eclipticPoint, limbAngle, type UV } from './plateGeom.ts';
import { cosd, sind, sunLambda } from './astro.ts';

export type XY = { x: number; y: number };
export const faceH = (W: number) => Math.round(Math.max(270, Math.min(400, W * 0.9)));
const NOTE = 26; // room under the drawing for the one-line note

function frame(W: number, zoom: number, focus?: UV) {
  const H = faceH(W);
  const Hd = H - NOTE;
  const k0 = (Math.min(W, Hd) / 2 - 6) / RM;
  const k = k0 * zoom;
  const fu = focus?.u ?? 0;
  const fv = focus?.v ?? 0;
  let cx = W / 2 - fu * k;
  let cy = Hd / 2 + fv * k;
  if (zoom > 1) {
    // never pan past the instrument's edge
    cx = Math.max(W - RM * k, Math.min(RM * k, cx));
    cy = Math.max(Hd - RM * k, Math.min(RM * k, cy));
  } else {
    cx = W / 2;
    cy = Hd / 2;
  }
  return { H, Hd, f: { cx, cy, k } };
}
const px = (p: UV, f: { cx: number; cy: number; k: number }): XY => ({ x: f.cx + p.u * f.k, y: f.cy - p.v * f.k });

export function frontScene(W: number, o: { phi: number; lst: number; sunLam: number | null; ruleH: number | null; zoom?: number; focus?: UV }) {
  const { H, Hd, f } = frame(W, o.zoom ?? 1, o.focus);
  const sun = o.sunLam === null ? null : px(eclipticPoint(o.sunLam, o.lst), f);
  let rule: { a: XY; b: XY; limb: XY; H: number } | null = null;
  if (o.ruleH !== null) {
    const a = limbAngle(o.ruleH);
    const tipR = RC * 1.09;
    rule = {
      a: px({ u: -0.06 * cosd(a), v: -0.06 * sind(a) }, f),
      b: px({ u: tipR * cosd(a), v: tipR * sind(a) }, f),
      limb: px({ u: RC * 1.015 * cosd(a), v: RC * 1.015 * sind(a) }, f),
      H: o.ruleH,
    };
  }
  return { W, H, Hd, f, sun, rule, RCpx: RC * f.k, RMpx: RM * f.k };
}

export function backScene(W: number, o: { ali: number; doy?: number; zoom?: number; focus?: UV }) {
  const { H, Hd, f } = frame(W, o.zoom ?? 1, o.focus);
  const psi = backPsi(o.ali);
  const dir = { u: cosd(psi), v: sind(psi) };
  const nrm = { u: -dir.v, v: dir.u };
  const L = RM * 0.97;
  const hw = 0.045;
  // the bar's edge (fiducial) runs exactly through the pin along ψ; the bar lies on one side
  const bar = [
    { u: -dir.u * L, v: -dir.v * L },
    { u: dir.u * L, v: dir.v * L },
    { u: dir.u * L + nrm.u * hw * 2, v: dir.v * L + nrm.v * hw * 2 },
    { u: -dir.u * L + nrm.u * hw * 2, v: -dir.v * L + nrm.v * hw * 2 },
  ].map((p) => px(p, f));
  const edge = { a: px({ u: -dir.u * L, v: -dir.v * L }, f), b: px({ u: dir.u * L, v: dir.v * L }, f) };
  const at = (r: number) => px({ u: r * dir.u, v: r * dir.v }, f);
  const altMark = at((BACK.altIn + BACK.altOut) / 2);
  // on the back the date rings are read with the SAME edge; the zodiac/calendar marks are where
  // the edge crosses those rings (the plate sets ali so that ψ = 180° − λ)
  const zodMark = at((BACK.zodIn + BACK.zodOut) / 2);
  const calMark = at((BACK.calIn + BACK.calOut) / 2);
  return { W, H, Hd, f, bar, edge, altMark, zodMark, calMark, psi, RMpx: RM * f.k };
}

/** The alidade setting that lays the edge over date `doy` on the calendar ring: a = λ(date). */
export const aliForDate = (doy: number) => {
  const lam = sunLambda(doy);
  return lam > 180 ? lam - 360 : lam; // any value with ψ = 180° − λ (mod 360)
};
