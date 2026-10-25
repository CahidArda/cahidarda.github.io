/** Plate 1 of the time post: the precision ladder for clocks (wraps the shared ladder; the
 *  config holds functions, which Astro cannot pass as island props from MDX). */
import PrecisionLadder from '../PrecisionLadder';
import { CLOCK_LADDER } from './timeLadder';

export default function ClockLadder() {
  return (
    <PrecisionLadder
      n={1}
      config={CLOCK_LADDER}
      title="How clock accuracy improved, 1500 BCE to today"
      caption={
        <>
          Each mark is the best clock of its time, placed at how many seconds it gained or lost in
          a day; higher is better, and each gridline is ten times better than the one below. The
          red line is the best so far. Hollow marks are order-of-magnitude estimates. The 3,000
          years before 1500 are squeezed into the dashed stretch on the left. Hover or tap a mark
          for its source.
        </>
      }
    />
  );
}
