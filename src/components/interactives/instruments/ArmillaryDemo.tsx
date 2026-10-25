/**
 * A small armillary sphere: proof plate for BlueprintCanvas. Rings are square-section tori
 * (four corner edges = an engraved band), the polar axis is a rod, the ecliptic ring is the
 * red-ochre line. Slowly turns on its axis; drag to look around.
 */
import { BlueprintCanvas, C, Plate } from './shared';

const TILT = (23.44 * Math.PI) / 180;
const LAT = (41 * Math.PI) / 180; // Istanbul

function Poster() {
  // A static engraving of the same object, seen from the start angle (for SSR / no WebGL).
  const W = 400;
  const H = 300;
  const cx = W / 2;
  const cy = H / 2 - 6;
  const R = 104;
  const s = { fill: 'none', stroke: C.ink, strokeWidth: 1 };
  return (
    <svg viewBox={`0 0 ${W} ${H}`}>
      <circle cx={cx} cy={cy} r={R} style={s} />
      <circle cx={cx} cy={cy} r={R - 5} style={s} />
      <ellipse cx={cx} cy={cy} rx={R} ry={R * 0.32} style={s} transform={`rotate(-41 ${cx} ${cy})`} />
      <ellipse
        cx={cx}
        cy={cy}
        rx={R}
        ry={R * 0.3}
        style={{ ...s, stroke: C.red }}
        transform={`rotate(-17 ${cx} ${cy})`}
      />
      <ellipse cx={cx} cy={cy} rx={R + 8} ry={(R + 8) * 0.26} style={s} />
      <line x1={cx - 92} y1={cy + 107} x2={cx + 92} y2={cy - 107} style={s} transform={`rotate(-8 ${cx} ${cy})`} />
      <circle cx={cx} cy={cy} r={11} style={s} />
      <path d={`M ${cx} ${cy + R} V ${H - 14} M ${cx - 40} ${H - 14} H ${cx + 40}`} style={s} />
    </svg>
  );
}

export default function ArmillaryDemo() {
  return (
    <Plate
      n={2}
      id="plate-armillary"
      title="An armillary sphere, for the road"
      caption={
        <>
          The sky as nested rings: horizon, meridian, celestial equator, and the ecliptic in
          red, tilted 23.44° to the equator. The polar axis is set for Istanbul, 41° above the
          horizon.
        </>
      }
    >
      <BlueprintCanvas
        aspect={4 / 3}
        fit={1.62}
        animate
        yaw={-0.5}
        pitch={0.32}
        label="Rotatable line drawing of an armillary sphere"
        poster={<Poster />}
        build={(T, ctx) => {
          const band = (r: number, w = 0.045) => {
            const g = new T.TorusGeometry(r, w, 4, 128);
            return new T.Mesh(g);
          };
          ctx.root.position.y = 0.1;
          // Fixed frame: horizon ring (flat) and meridian ring (vertical, N-S)
          const horizon = band(1.18, 0.05);
          horizon.rotation.x = Math.PI / 2;
          ctx.inkify(horizon, { outline: 1 });
          ctx.root.add(horizon);

          const meridian = band(1.06);
          ctx.inkify(meridian, { outline: 1 });
          ctx.root.add(meridian);

          // Stand
          const post = new T.Mesh(new T.CylinderGeometry(0.035, 0.05, 0.42, 24));
          post.position.y = -1.27;
          ctx.inkify(post);
          ctx.root.add(post);
          const foot = new T.Mesh(new T.CylinderGeometry(0.42, 0.48, 0.07, 64));
          foot.position.y = -1.5;
          ctx.inkify(foot, { threshold: 40 });
          ctx.root.add(foot);

          // The sky: tilted by latitude, spins about the polar axis
          const pole = new T.Group();
          pole.rotation.z = Math.PI / 2 - LAT; // axis from vertical to 41° above horizon
          ctx.root.add(pole);
          const sky = new T.Group();
          pole.add(sky);

          const axis = new T.Mesh(new T.CylinderGeometry(0.018, 0.018, 2.5, 12));
          ctx.inkify(axis, { threshold: 60, outline: 0.6 });
          pole.add(axis);

          const equator = band(0.98, 0.04);
          equator.rotation.x = Math.PI / 2;
          ctx.inkify(equator, { outline: 1 });
          sky.add(equator);

          const colure = band(0.98, 0.035);
          colure.rotation.y = Math.PI / 2;
          ctx.inkify(colure, { outline: 1 });
          sky.add(colure);

          const ecliptic = band(0.98, 0.05);
          ecliptic.rotation.x = Math.PI / 2 + TILT;
          ctx.inkify(ecliptic, { edge: 'red', hidden: true, outline: 1.2 });
          sky.add(ecliptic);

          const earth = new T.Mesh(new T.SphereGeometry(0.13, 32, 16));
          ctx.inkify(earth, { threshold: 80, outline: 1.2 });
          sky.add(earth);

          return (_dt, t) => {
            sky.rotation.y = t * 0.25;
          };
        }}
      />
    </Plate>
  );
}
