/**
 * Sextant geometry shared by the 3D model and its SVG poster. Everything is in the plane of
 * the frame, units = arc radius, origin = the index-arm pivot, y up, angles in degrees
 * (0° = +x, counter-clockwise). The instrument is held upright: arc at the bottom, the
 * telescope looking left at the horizon through the horizon mirror.
 *
 * Optics (exact, no small-angle step): the horizon mirror H is fixed and sends light coming
 * from the index mirror along +x into the telescope. At reading 0 the index mirror is
 * parallel to it. Turning the arm (and the mirror on it) clockwise by θ swings the line of
 * sight that reaches the eye up by 2θ, so a 60° arc is engraved 0 to 120.
 */
export const R = 1; // arc radius
export const ARC_FROM = -5; // engraved range, in READING degrees (off-arc part below 0)
export const ARC_TO = 125;

/** Arm direction (deg) for a given arc reading. Reading 0 = right end of the arc. */
export const armAngle = (reading: number) => -60 - reading / 2;

export type V2 = { x: number; y: number };
const RAD = Math.PI / 180;
export const pol = (r: number, deg: number): V2 => ({
  x: r * Math.cos(deg * RAD),
  y: r * Math.sin(deg * RAD),
});
const norm = (v: V2): V2 => {
  const l = Math.hypot(v.x, v.y) || 1;
  return { x: v.x / l, y: v.y / l };
};
export const reflect = (d: V2, n: V2): V2 => {
  const k = 2 * (d.x * n.x + d.y * n.y);
  return { x: d.x - k * n.x, y: d.y - k * n.y };
};
export const rot = (v: V2, deg: number): V2 => {
  const c = Math.cos(deg * RAD);
  const s = Math.sin(deg * RAD);
  return { x: v.x * c - v.y * s, y: v.x * s + v.y * c };
};

/** Horizon mirror centre: on the left frame bar, a little under halfway down. */
export const H: V2 = { x: -0.22, y: -0.4 };
/** Unit vector from the index mirror (pivot) to the horizon mirror. */
export const U = norm(H);
/** Horizon-mirror normal: reflects U into +x (towards the telescope). */
export const NH = norm({ x: 1 - U.x, y: -U.y });
/** Angle of the horizon mirror's normal, degrees. */
export const NH_DEG = Math.atan2(NH.y, NH.x) / RAD;

/** Telescope along y = H.y, from just right of the horizon mirror to the eyepiece. */
export const TEL_FROM = H.x + 0.2;
export const TEL_TO = H.x + 0.86;
export const EYE: V2 = { x: TEL_TO, y: H.y };

/**
 * For an arc reading, the index mirror normal and the direction TOWARDS the sun whose
 * doubly reflected image sits on the horizon in the telescope.
 */
export function optics(reading: number) {
  const theta = reading / 2; // mirror turned clockwise by θ
  const nI = rot(NH, -theta);
  // trace backwards from the eye: -x hits H, goes to the pivot, reflects off the index mirror
  const back1 = reflect({ x: -1, y: 0 }, NH); // H → pivot
  const toSun = reflect(back1, nI); // pivot → sun
  return { theta, nI, toSun: norm(toSun), altitude: (Math.atan2(toSun.y, -toSun.x) / RAD) };
}
