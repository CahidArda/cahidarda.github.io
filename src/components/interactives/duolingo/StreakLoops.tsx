/**
 * The streak loop versus the learning loop. One viewBox, edges on box edges, the "repeat"
 * legs drawn as curves back to the top node. Static.
 */
import {
  ACCENT,
  ArrowDefs,
  Caption,
  Curve,
  Edge,
  INK,
  Label,
  Node,
  Widget,
  bottomOf,
  leftOf,
  rightOf,
  topOf,
  type Box,
} from './shared';

const W = 190;
const H = 44;
const col = (cx: number): Box[] => [70, 140, 210].map((cy) => ({ cx, cy, w: W, h: H }));
const S = col(150);
const L = col(410);

export default function StreakLoops() {
  const s0 = leftOf(S[0]);
  const s2 = leftOf(S[2]);
  const l0 = rightOf(L[0]);
  const l2 = rightOf(L[2]);
  return (
    <Widget title="Two loops" kicker="both repeat forever">
      <svg
        viewBox="0 0 560 250"
        className="block h-auto w-full"
        role="img"
        aria-label="The streak loop: obvious options, you get it right, streak plus one, repeat. The learning loop: plausible options, you get it wrong, you work out why, repeat."
      >
        <ArrowDefs />
        <Label x={150} y={24} muted={false} size={13}>
          THE STREAK LOOP
        </Label>
        <Label x={410} y={24} muted={false} size={13}>
          THE LEARNING LOOP
        </Label>

        <Edge from={bottomOf(S[0])} to={topOf(S[1])} on />
        <Edge from={bottomOf(S[1])} to={topOf(S[2])} on />
        <Edge from={bottomOf(L[0])} to={topOf(L[1])} on />
        <Edge from={bottomOf(L[1])} to={topOf(L[2])} on />
        <Curve d={`M${s2.x},${s2.y} C${s2.x - 38},${s2.y} ${s0.x - 38},${s0.y} ${s0.x},${s0.y}`} dashed />
        <Curve d={`M${l2.x},${l2.y} C${l2.x + 38},${l2.y} ${l0.x + 38},${l0.y} ${l0.x},${l0.y}`} />
        <Label x={20} y={140} rotate={-90}>
          repeat
        </Label>
        <Label x={541} y={140} rotate={90}>
          you learn
        </Label>

        <Node n={S[0]} title="Obvious options" color={INK} />
        <Node n={S[1]} title="You get it right" color={INK} />
        <Node n={S[2]} title="Streak +1" color={INK} state="active" />
        <Node n={L[0]} title="Plausible options" color={ACCENT} />
        <Node n={L[1]} title="You get it wrong" color={ACCENT} />
        <Node n={L[2]} title="You work out why" color={ACCENT} state="active" />
      </svg>
      <Caption>Only one of them teaches you French.</Caption>
    </Widget>
  );
}
