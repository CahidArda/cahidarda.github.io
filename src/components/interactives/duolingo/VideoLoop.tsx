/**
 * How watching the same creators built vocabulary: watch, miss a word, look it up, hear it
 * again in the next video, repeat. A four-node ring in one viewBox, edges on box edges. Static.
 */
import {
  ACCENT,
  ArrowDefs,
  Caption,
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

const W = 200;
const H = 54;
const N: Record<'watch' | 'miss' | 'look' | 'again', Box> = {
  watch: { cx: 130, cy: 55, w: W, h: H },
  miss: { cx: 430, cy: 55, w: W, h: H },
  look: { cx: 430, cy: 190, w: W, h: H },
  again: { cx: 130, cy: 190, w: W, h: H },
};

export default function VideoLoop() {
  return (
    <Widget title="How the vocabulary grew" kicker="English, by accident">
      <svg
        viewBox="0 0 560 250"
        className="block h-auto w-full"
        role="img"
        aria-label="A loop: watch a video you like, miss a word, look it up, hear the same word again in the next video, repeat."
      >
        <ArrowDefs />
        <Edge from={rightOf(N.watch)} to={leftOf(N.miss)} on />
        <Edge from={bottomOf(N.miss)} to={topOf(N.look)} on />
        <Edge from={leftOf(N.look)} to={rightOf(N.again)} on />
        <Edge from={topOf(N.again)} to={bottomOf(N.watch)} on />

        <Label x={280} y={114}>
          same creators,
        </Label>
        <Label x={280} y={130}>
          same words, again
        </Label>

        <Node n={N.watch} title="Watch" sub="a video I want to see" color={INK} />
        <Node n={N.miss} title="Miss a word" sub="didn't understand it" color={INK} />
        <Node n={N.look} title="Look it up" sub="one word at a time" color={INK} />
        <Node
          n={N.again}
          title="Hear it again"
          sub="in the next video"
          color={ACCENT}
          state="active"
        />
      </svg>
      <Caption>Duolingo's repetition, except every repeat is something I wanted to watch.</Caption>
    </Widget>
  );
}
