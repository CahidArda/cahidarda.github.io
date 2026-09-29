/**
 * Wireframe of the app: the video on the left, synced French lines with translations on the
 * right, a hover gloss on one word, and the pipeline behind it along the bottom. Static.
 */
import {
  ACCENT,
  ArrowDefs,
  Caption,
  Edge,
  INK,
  Label,
  MONO,
  Node,
  SANS,
  Widget,
  leftOf,
  rightOf,
  type Box,
} from './shared';

const PIPE: { n: Box; title: string; sub: string }[] = [
  { title: 'Song or URL', sub: 'what I want to watch' },
  { title: 'Lyrics', sub: 'or the transcript' },
  { title: 'Translate', sub: 'AI, line by line' },
  { title: 'Synced view', sub: 'play and read' },
].map((p, i) => ({ ...p, n: { cx: 82 + i * 158, cy: 292, w: 136, h: 50 } }));

const line = (y: number, fr: string, en: string, dim: boolean) => (
  <g>
    <text x={352} y={y} style={{ fill: dim ? 'var(--color-muted)' : 'var(--color-ink)', fontFamily: SANS, fontSize: 13 }}>
      {fr}
    </text>
    <text x={352} y={y + 17} style={{ fill: 'var(--color-muted)', fontFamily: MONO, fontSize: 10.5, opacity: 0.8 }}>
      {en}
    </text>
  </g>
);

export default function AppLayout() {
  return (
    <Widget title="The app" kicker="work in progress">
      <svg
        viewBox="0 0 640 330"
        className="block h-auto w-full"
        role="img"
        aria-label="App layout: a YouTube player on the left, synced French lines with English translations on the right, and a hover gloss on one word. Below, the pipeline: song or URL, lyrics or transcript, AI translation, synced view."
      >
        <ArrowDefs />
        <rect x={10} y={10} width={620} height={222} style={{ fill: 'var(--color-paper)', stroke: 'var(--color-line-strong)', strokeWidth: 1.5 }} />

        {/* video */}
        <rect x={24} y={24} width={304} height={194} style={{ fill: 'var(--color-paper-raised)', stroke: 'var(--color-line)', strokeWidth: 1 }} />
        <path d="M160,98 L160,144 L198,121 z" style={{ fill: 'var(--color-muted)' }} />
        <Label x={176} y={196}>
          YOUTUBE PLAYER
        </Label>

        {/* synced lines */}
        {line(46, 'Il fait beau aujourd’hui', 'The weather is nice today', true)}
        <rect x={342} y={78} width={276} height={50} style={{ fill: ACCENT, fillOpacity: 0.1, stroke: 'none' }} />
        <rect x={342} y={78} width={3} height={50} style={{ fill: ACCENT }} />
        <text x={352} y={98} style={{ fill: 'var(--color-ink)', fontFamily: SANS, fontSize: 15, fontWeight: 600 }}>
          On se{' '}
          <tspan style={{ fill: ACCENT, textDecoration: 'underline' }}>retrouve</tspan> au café
        </text>
        <text x={352} y={117} style={{ fill: 'var(--color-ink-soft)', fontFamily: MONO, fontSize: 10.5 }}>
          We’ll meet up at the café
        </text>

        {/* hover gloss */}
        <path d="M436,134 L444,126 L452,134 z" style={{ fill: INK }} />
        <rect x={400} y={134} width={180} height={44} style={{ fill: INK }} />
        <text x={490} y={151} textAnchor="middle" style={{ fill: 'var(--color-paper)', fontFamily: MONO, fontSize: 12, fontWeight: 600 }}>
          retrouver
        </text>
        <text x={490} y={168} textAnchor="middle" style={{ fill: 'var(--color-paper)', fontFamily: MONO, fontSize: 10.5 }}>
          to meet (up) again
        </text>

        {line(200, 'Tu viens avec nous ?', 'Are you coming with us?', true)}

        {/* pipeline */}
        {PIPE.slice(1).map((p, i) => (
          <Edge key={p.title} from={rightOf(PIPE[i].n)} to={leftOf(p.n)} on />
        ))}
        {PIPE.map((p, i) => (
          <Node
            key={p.title}
            n={p.n}
            title={p.title}
            sub={p.sub}
            color={i === PIPE.length - 1 ? ACCENT : INK}
            state={i === PIPE.length - 1 ? 'active' : 'wait'}
            titleSize={12}
            subSize={9.5}
          />
        ))}
      </svg>
      <Caption>Watch what you would watch anyway. The translation sits next to it, one hover away.</Caption>
    </Widget>
  );
}
