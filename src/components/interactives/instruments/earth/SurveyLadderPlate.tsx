/**
 * Earth: the precision ladder for measured length. A thin wrapper so the config (which holds
 * functions) stays on the client side of the island boundary.
 */
import PrecisionLadder from '../PrecisionLadder';
import { SURVEY_LADDER } from './surveyLadder';

export default function SurveyLadderPlate({ n }: { n: number }) {
  return (
    <PrecisionLadder
      n={n}
      config={SURVEY_LADDER}
      title="How precisely we could measure distance, from the pyramids to GPS"
      caption={
        <>
          Error per kilometre of length measured, on a log scale; the red line is the best so
          far, hollow marks would be order-of-magnitude estimates. The pyramid builders levelled
          a 325 m pavement to under 5 cm per kilometre, while Snellius’s triangle chain, 4,000
          years later, was off by 35 m per km. Short, careful levelling was good early; long
          distances caught up only when the angles did. The years before 1500 are squeezed into
          the dashed stretch.
        </>
      }
    />
  );
}
