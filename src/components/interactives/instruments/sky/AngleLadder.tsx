/**
 * Plate: the angle precision ladder, from the gnomon to Gaia. A thin wrapper so the MDX island
 * gets no function props (the ladder config carries formatters, which cannot be serialised).
 */
import PrecisionLadder, { ANGLE_LADDER } from '../PrecisionLadder';

export default function AngleLadder() {
  return (
    <PrecisionLadder
      n={6}
      config={ANGLE_LADDER}
      title="How precisely we could measure angles, from the gnomon to Gaia"
      caption={
        <>
          Each rung is the best error in angle an instrument could hold; the red line is the best so
          far. Hollow marks are order-of-magnitude estimates. Hover or tap a rung for the figure and
          its source. The years before 1500 are squeezed into the dashed stretch on the left.
        </>
      }
    />
  );
}
