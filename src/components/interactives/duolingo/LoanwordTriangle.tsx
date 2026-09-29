/**
 * French words that travelled into Turkish and English. Arrows run the direction of borrowing;
 * the dashed Turkish-English edge is what a speaker of both gets for free. Static.
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
  leftOf,
  rightOf,
  topOf,
  type Box,
} from './shared';

const N: Record<'fr' | 'tr' | 'en', Box> = {
  fr: { cx: 280, cy: 48, w: 140, h: 44 },
  tr: { cx: 110, cy: 200, w: 140, h: 44 },
  en: { cx: 450, cy: 200, w: 140, h: 44 },
};
const frBottom = N.fr.cy + N.fr.h / 2;

export default function LoanwordTriangle() {
  return (
    <Widget title="Where the French went" kicker="arrows follow the borrowing">
      <svg
        viewBox="0 0 560 250"
        className="block h-auto w-full"
        role="img"
        aria-label="French lent words to Turkish (şimendifer, abajur, randevu) and to English (rendezvous, chauffeur, coiffure). A speaker of Turkish and English already shares those roots."
      >
        <ArrowDefs />
        <Edge from={{ x: N.fr.cx - 40, y: frBottom }} to={topOf(N.tr)} on />
        <Edge from={{ x: N.fr.cx + 40, y: frBottom }} to={topOf(N.en)} on />
        <Edge from={rightOf(N.tr)} to={leftOf(N.en)} />

        <Label x={175} y={112} anchor="end" muted={false}>
          şimendifer
        </Label>
        <Label x={175} y={128} anchor="end" muted={false}>
          abajur · randevu
        </Label>
        <Label x={385} y={112} anchor="start" muted={false}>
          rendezvous
        </Label>
        <Label x={385} y={128} anchor="start" muted={false}>
          chauffeur · coiffure
        </Label>
        <Label x={280} y={188}>
          shared roots
        </Label>
        <Label x={280} y={232}>
          vocabulary you already own
        </Label>

        <Node n={N.fr} title="French" color={ACCENT} state="active" titleSize={15} />
        <Node n={N.tr} title="Turkish" color={INK} titleSize={15} />
        <Node n={N.en} title="English" color={INK} titleSize={15} />
      </svg>
      <Caption>If you speak Turkish and English, you already know a surprising amount of French.</Caption>
    </Widget>
  );
}
