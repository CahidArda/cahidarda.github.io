/**
 * ONE geometry function for the 3D sextant plate: the 3D ray paths, the eyepiece inset, the
 * readouts and the worked equation all read from `sextantView(reading, tilt)`.
 *
 * World: the frame plane is vertical and contains the sun, x towards the eye (away from the
 * sun's azimuth), y up, units = arc radius, origin = the index-arm pivot (it does not move).
 * The sun and the sea horizon are fixed directions in the world. The instrument (frame, horizon
 * mirror, telescope, and the arm at its reading) tips by `tilt` degrees in its own plane about
 * the pivot, as it does on a moving deck.
 *
 * Physics (checked against an independent 3D model in the plate's test): after two reflections
 * the sun's light is turned by twice the angle between the mirrors, whatever the tilt. So the
 * sun's image and the horizon both sit `tilt` degrees off the telescope axis together, and the
 * gap between them is (sun's altitude − reading), constant over the tilt.
 */
import { EYE, H, NH, TEL_FROM, optics, reflect, rot, type V2 } from './sextantGeom.ts';

/** The noon sun over Istanbul on 25 October 2026 (USNO approximate solar coordinates). */
export const SUN_ALT = 36.82;
/** The sun's angular radius that day, degrees (16.1′). */
export const SUN_SD = 16.1 / 60;
/** Half the eyepiece's field of view, degrees. */
export const FIELD_R = 2.5;
/** The images "touch" (status: on the horizon) when the gap is within this, degrees (3′). */
export const ON_TOL = 0.05;
/** How far the deck tips the instrument in its own plane, degrees, and the period, seconds. */
export const TILT_AMP = 1.6;
export const TILT_PERIOD = 5.5;
export const tiltAt = (t: number) => TILT_AMP * Math.sin((2 * Math.PI * t) / TILT_PERIOD);
/** Half the length of the horizon mirror in the frame plane, and the objective's radius. */
export const MIRROR_HALF = 0.07;
export const APERTURE = 0.058;
/** Where the sun is drawn, distance from the pivot, and where the horizon line starts (x). */
export const SUN_DIST = 1.08;
export const HORIZON_X0 = -1.45;

const RAD = Math.PI / 180;
const dot = (a: V2, b: V2) => a.x * b.x + a.y * b.y;
const add = (a: V2, b: V2, k = 1): V2 => ({ x: a.x + b.x * k, y: a.y + b.y * k });
const sub = (a: V2, b: V2): V2 => ({ x: a.x - b.x, y: a.y - b.y });

export type SextantView = {
  /** Direction towards the sun, and where the sun disc is drawn. */
  toSun: V2;
  sun: V2;
  /** Red path, world points: sun → index mirror → horizon mirror → objective → eye when the
   *  image is in the field; it stops at the objective (field stop) or runs on past a mirror it misses. */
  red: V2[];
  /** Blue dashed path: the sea horizon's light, level in the world, through the clear half
   *  at the same point of the mirror as the sun's light, to the eye. */
  blue: V2[];
  /** The reflected ray reached the horizon mirror, and then the telescope's objective. */
  hitsMirror: boolean;
  entersTelescope: boolean;
  /** Elevations in the eyepiece field, degrees above its centre (the telescope axis). */
  horizonEl: number;
  sunEl: number | null;
  /** Sun's image above the horizon, degrees. Exact for any tilt: (altitude − reading). */
  gap: number;
  /** The sun's image is inside the eyepiece's field of view (it moves out as the deck tilts). */
  inField: boolean;
  /** Sun's image vs horizon, from `gap` alone (so the same at every tilt, like the eyepiece). */
  status: 'on' | 'above' | 'below';
  /** Mirror turned (θ) and the instrument's pieces in the world. */
  theta: number;
  pivot: V2;
  mirrorH: V2;
  objective: V2;
  eye: V2;
  axis: V2;
};

/** Elevation (deg) in the field of a ray TRAVELLING along d, for a telescope whose axis points `axis` (towards the eye). */
function fieldEl(d: V2, axis: V2, up: V2) {
  const src = { x: -d.x, y: -d.y };
  const look = { x: -axis.x, y: -axis.y };
  return Math.atan2(dot(src, up), dot(src, look)) / RAD;
}

export function sextantView(reading: number, tilt = 0, sunAlt = SUN_ALT): SextantView {
  const toSun: V2 = { x: -Math.cos(sunAlt * RAD), y: Math.sin(sunAlt * RAD) };
  const sun = { x: toSun.x * SUN_DIST, y: toSun.y * SUN_DIST };
  const pivot: V2 = { x: 0, y: 0 };
  const { theta, nI } = optics(reading);
  const nIw = rot(nI, tilt);
  const nHw = rot(NH, tilt);
  const Hw = rot(H, tilt);
  const axis = rot({ x: 1, y: 0 }, tilt);
  const up = rot({ x: 0, y: 1 }, tilt);
  const objective = rot({ x: TEL_FROM, y: H.y }, tilt);
  const eye = rot(EYE, tilt);
  const horizonDir: V2 = { x: 1, y: 0 }; // light from the sea horizon travels towards +x
  const horizonEl = fieldEl(horizonDir, axis, up);

  // sun → pivot (index mirror) → reflect
  const d1 = { x: -toSun.x, y: -toSun.y };
  const d2 = reflect(d1, nIw);
  // meet the horizon mirror's line: pivot + a·d2 = Hw + b·t
  const t = { x: -nHw.y, y: nHw.x };
  const det = d2.x * -t.y - d2.y * -t.x;
  let a = -1;
  let b = 0;
  if (Math.abs(det) > 1e-12) {
    a = (Hw.x * -t.y - Hw.y * -t.x) / det;
    b = (d2.x * Hw.y - d2.y * Hw.x) / det;
  }
  const hitsMirror = a > 0 && Math.abs(b) <= MIRROR_HALF;
  const red: V2[] = [sun, pivot];
  let sunEl: number | null = null;
  let entersTelescope = false;
  // where a ray from p along d crosses the objective's plane, and how far off the axis
  const toPlane = (p: V2, d: V2, at: V2) => {
    const den = dot(d, axis);
    if (den <= 1e-9) return null;
    const s = dot(sub(at, p), axis) / den;
    if (s <= 0) return null;
    const E = add(p, d, s);
    return { E, off: dot(sub(E, at), up) };
  };
  const toObjective = (p: V2, d: V2) => toPlane(p, d, objective);
  if (hitsMirror) {
    const P = add(pivot, d2, a);
    const d3 = reflect(d2, nHw);
    red.push(P);
    const o = toObjective(P, d3);
    if (o && Math.abs(o.off) <= APERTURE) {
      entersTelescope = true;
      red.push(o.E);
      sunEl = fieldEl(d3, axis, up);
      if (Math.abs(sunEl) <= FIELD_R) red.push(toPlane(o.E, d3, eye)!.E);
    } else {
      red.push(add(P, d3, 0.5));
    }
  } else {
    red.push(add(pivot, d2, 0.95));
  }
  // the horizon ray drawn through the same point of the mirror as the sun's (clear half above it)
  const bq = hitsMirror ? b : 0;
  const Q = add(Hw, t, bq);
  const oq = toObjective(Q, horizonDir);
  const blue: V2[] = [{ x: HORIZON_X0, y: Q.y }, Q];
  if (oq) blue.push(oq.E, toPlane(oq.E, horizonDir, eye)!.E);
  else blue.push(add(Q, horizonDir, 0.2));

  // Two reflections turn the light by 2θ = reading, so the image sits (altitude − reading)
  // above the horizon at every tilt (when it enters, sunEl − horizonEl is exactly this).
  const gap = sunAlt - reading;
  const inField = sunEl !== null && Math.abs(sunEl) <= FIELD_R;
  const status: SextantView['status'] =
    Math.abs(gap) <= ON_TOL ? 'on' : gap > 0 ? 'above' : 'below';

  return {
    toSun,
    sun,
    red,
    blue,
    hitsMirror,
    entersTelescope,
    horizonEl,
    sunEl,
    gap,
    inField,
    status,
    theta,
    pivot,
    mirrorH: Hw,
    objective,
    eye,
    axis,
  };
}
