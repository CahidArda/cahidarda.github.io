import { useEffect, useRef, useState } from 'react';
import {
  BigStat,
  Btn,
  Caption,
  GT,
  Seg,
  Specimen,
  SvgArrows,
  T,
  useFrame,
  useInView,
  useReducedMotion,
} from '../shared';
import { useBoost } from '../braess/useBoost';
import { boxExit, control, qAt, type Pt } from './graph';
import { outLists, update, type Edge } from './model';

/* Five pages drawn as tanks of "rank". Each step, every tank empties itself equally
   down its out-links. In the trap, D and E link only to each other. */
const NAMES = 'ABCDE'.split('');
const POS: Pt[] = [
  { x: 78, y: 150 }, // A
  { x: 222, y: 62 }, // B
  { x: 222, y: 238 }, // C
  { x: 372, y: 62 }, // D
  { x: 462, y: 214 }, // E
];
const id = (c: string) => NAMES.indexOf(c);
const parse = (s: string): Edge[] => s.split(' ').map((p) => [id(p[0]), id(p[1])]);
const WEB: Record<'open' | 'trap', Edge[]> = {
  open: parse('AB AC BC BD CA DA DE EA'),
  trap: parse('AB AC BC BD CA DE ED'),
};
const N = 5;
const TW = 60;
const TH = 72;
const VW = 540;
const VH = 300;
const STEP_MS = 750;
const S_SCALED = 0.85;
const FULL = 0.5; // rank that fills a tank to the brim
const PLATE = 20; // label strip at the top of each tank

type Web = 'open' | 'trap';
type Rule = 'basic' | 'scaled';
const uniform = () => new Array(N).fill(1 / N);
const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

export default function RankFlow() {
  const reduced = useReducedMotion();
  const [rootRef, inView] = useInView();
  const [svgRef, k] = useBoost(VW);
  const [web, setWeb] = useState<Web>('open');
  const [rule, setRule] = useState<Rule>('basic');
  const [r, setR] = useState<number[]>(uniform);
  const [prev, setPrev] = useState<number[]>(uniform);
  const [stepN, setStepN] = useState(0);
  const [phase, setPhase] = useState(1); // 0..1 progress of the current step's animation
  const [playing, setPlaying] = useState(false);
  const touched = useRef(false);
  const script = useRef<{ web: Web; rule: Rule; steps: number }[]>([]);

  const edges = WEB[web];
  const out = outLists(N, edges);
  const s = rule === 'scaled' ? S_SCALED : 1;

  const advance = () => {
    setPrev(r);
    setR(update(r, out, s));
    setStepN((n) => n + 1);
    setPhase(reduced ? 1 : 0);
  };

  // animate the pour
  useFrame(
    (dt) => {
      setPhase((p) => Math.min(1, p + (dt * 1000) / (STEP_MS * 0.85)));
    },
    phase < 1 && !reduced,
  );

  // play: one step every STEP_MS; the auto demo follows a short script
  useEffect(() => {
    if (!playing || !inView) return;
    const t = setTimeout(() => {
      if (!touched.current && script.current.length) {
        const cur = script.current[0];
        if (cur.web !== web || cur.rule !== rule) {
          setWeb(cur.web);
          setRule(cur.rule);
          return;
        }
        cur.steps--;
        if (cur.steps <= 0) script.current.shift();
        if (!script.current.length) setPlaying(false);
      } else if (Math.max(...update(r, out, s).map((v, i) => Math.abs(v - r[i]))) < 5e-4) {
        // the levels have stopped changing: rest until the reader changes something
        setPlaying(false);
        return;
      }
      advance();
    }, STEP_MS);
    return () => clearTimeout(t);
  }, [playing, inView, r, web, rule]);

  useEffect(() => {
    if (!inView || touched.current || stepN > 0) return;
    if (reduced) {
      // poster: the trap after the basic rule has run for a while
      let x = uniform();
      const o = outLists(N, WEB.trap);
      for (let i = 0; i < 25; i++) x = update(x, o, 1);
      setWeb('trap');
      setR(x);
      setPrev(x);
      setStepN(25);
      touched.current = true;
      return;
    }
    script.current = [
      { web: 'open', rule: 'basic', steps: 4 },
      { web: 'trap', rule: 'basic', steps: 12 },
      { web: 'trap', rule: 'scaled', steps: 8 },
    ];
    const t = setTimeout(() => setPlaying(true), 900);
    return () => clearTimeout(t);
  }, [inView, reduced]);

  const restartTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (restartTimer.current) clearTimeout(restartTimer.current);
    },
    [],
  );
  const touch = () => {
    touched.current = true;
    script.current = [];
    if (restartTimer.current) clearTimeout(restartTimer.current);
    restartTimer.current = null;
  };
  // switching the web or the rule keeps the current levels and lets them flow on
  const pickWeb = (v: Web) => {
    touch();
    setWeb(v);
    setPlaying(true);
  };
  const pickRule = (v: Rule) => {
    touch();
    setRule(v);
    setPlaying(true);
  };
  // restart: every page back to 1/5, then after a beat the rank starts to flow
  const restart = () => {
    touch();
    setPlaying(false);
    setR(uniform());
    setPrev(uniform());
    setStepN(0);
    setPhase(1);
    restartTimer.current = setTimeout(() => {
      restartTimer.current = null;
      setPlaying(true);
    }, 1000);
  };

  const e = ease(phase);
  const shown = r.map((v, i) => prev[i] + (v - prev[i]) * e);
  const trapShare = shown[3] + shown[4];
  const total = shown.reduce((a, b) => a + b, 0);

  // what flowed along each edge during the last step (from the previous ranks)
  const prevOut = out;
  const flowOn = (a: number) => (s * prev[a]) / prevOut[a].length;

  let caption: string;
  if (stepN === 0) caption = 'Every page starts with an equal share of rank: 1/5 each.';
  else if (web === 'trap' && rule === 'basic')
    caption =
      'D and E only link to each other. Rank flows in through B and never flows out, so the trap slowly takes all of it.';
  else if (web === 'trap')
    caption = `Scaled rule: each page passes on only ${S_SCALED} of its rank and the other ${(1 - S_SCALED).toFixed(2)} is shared by all pages, so A, B and C keep some rank.`;
  else
    caption =
      'Each page splits its rank equally among its out-links. A few steps and the levels stop changing.';

  return (
    <Specimen
      rootRef={rootRef}
      code="GT-06.2"
      title="rank flows like water"
      kicker={`step ${stepN}`}
      hint="switch the web or the rule: the rank flows on from where it is"
      footer="Easley & Kleinberg, sec. 14.3"
    >
      <svg
        ref={svgRef}
        viewBox={`0 0 ${VW} ${VH}`}
        className="block w-full"
        role="img"
        aria-label="Five pages drawn as tanks, with rank flowing along their links"
      >
        <SvgArrows />
        {/* trap zone */}
        {web === 'trap' && (
          <rect
            x={POS[3].x - TW / 2 - 18}
            y={POS[3].y - TH / 2 - 16}
            width={POS[4].x - POS[3].x + TW + 36}
            height={POS[4].y - POS[3].y + TH + 40}
            style={{ fill: 'none', stroke: GT.warn, strokeDasharray: '4 4' }}
          />
        )}
        {edges.map(([a, b], i) => {
          const pair = out[b].includes(a);
          const c = control(POS[a], POS[b], pair ? 18 : 0);
          const p0 = boxExit(POS[a], TW, TH, c, 3);
          const p1 = boxExit(POS[b], TW, TH, c, 5);
          const amt = stepN > 0 ? flowOn(a) : 0;
          const moving = stepN > 0 && phase < 1 && !reduced;
          const drop = qAt(p0, c, p1, Math.min(1, phase * 1.15));
          return (
            <g key={`${web}-${i}`}>
              <path
                d={`M${p0.x} ${p0.y} Q${c.x} ${c.y} ${p1.x} ${p1.y}`}
                markerEnd="url(#gt-arrow-muted)"
                style={{ fill: 'none', stroke: GT.muted, strokeWidth: 1.2 }}
              />
              {moving && amt > 0.002 && (
                <circle
                  cx={drop.x}
                  cy={drop.y}
                  r={2 + 12 * Math.sqrt(amt)}
                  style={{ fill: GT.a, opacity: 0.85 }}
                />
              )}
            </g>
          );
        })}
        {/* rain: the (1 - s)/n every page gets under the scaled rule */}
        {rule === 'scaled' &&
          stepN > 0 &&
          phase < 1 &&
          !reduced &&
          POS.map((p, i) => {
            const moving = stepN > 0 && phase < 1 && !reduced;
            const y = moving ? p.y - TH / 2 - 18 + 16 * phase : p.y - TH / 2 - 10;
            return (
              <path
                key={`rain${i}`}
                d={`M${p.x} ${y - 6} L${p.x + 3.5} ${y} A3.5 3.5 0 1 1 ${p.x - 3.5} ${y} Z`}
                style={{ fill: GT.alt, opacity: moving ? 1 - phase * 0.6 : 0.8 }}
              />
            );
          })}
        {POS.map((p, i) => {
          const lvl = Math.min(1, shown[i] / FULL);
          const inTrap = web === 'trap' && i >= 3;
          return (
            <g key={i}>
              <rect
                x={p.x - TW / 2}
                y={p.y - TH / 2}
                width={TW}
                height={TH}
                style={{
                  fill: GT.paperRaised,
                  stroke: inTrap ? GT.warn : GT.ink,
                  strokeWidth: 1.5,
                }}
              />
              <rect
                x={p.x - TW / 2 + 1}
                y={p.y + TH / 2 - 1 - (TH - PLATE - 2) * lvl}
                width={TW - 2}
                height={(TH - PLATE - 2) * lvl}
                style={{ fill: GT.aSoft }}
              />
              <line
                x1={p.x - TW / 2 + 1}
                x2={p.x + TW / 2 - 1}
                y1={p.y + TH / 2 - 1 - (TH - PLATE - 2) * lvl}
                y2={p.y + TH / 2 - 1 - (TH - PLATE - 2) * lvl}
                style={{ stroke: GT.a, strokeWidth: 2 }}
              />
              {/* label plate: name and current rank, above the water */}
              <line
                x1={p.x - TW / 2}
                x2={p.x + TW / 2}
                y1={p.y - TH / 2 + PLATE}
                y2={p.y - TH / 2 + PLATE}
                style={{ stroke: GT.line }}
              />
              <T
                x={p.x - TW / 2 + 6}
                y={p.y - TH / 2 + PLATE / 2 + 1}
                anchor="start"
                size={14}
                font="display"
                color={GT.ink}
              >
                {NAMES[i]}
              </T>
              <T
                x={p.x + TW / 2 - 5}
                y={p.y - TH / 2 + PLATE / 2 + 1}
                anchor="end"
                size={10 * Math.min(k, 1.3)}
                color={GT.ink}
                weight={700}
              >
                {shown[i].toFixed(2)}
              </T>
            </g>
          );
        })}
        {web === 'trap' && (
          <T
            x={POS[4].x + TW / 2 + 12}
            y={POS[4].y + TH / 2 + 24 + 10 * k}
            anchor="end"
            size={9 * k}
            color={GT.warn}
          >
            spider trap
          </T>
        )}
      </svg>

      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-line pt-4">
        <Seg
          options={[
            { value: 'open', label: 'open web' },
            { value: 'trap', label: 'spider trap' },
          ]}
          value={web}
          onChange={pickWeb}
        />
        <Seg
          options={[
            { value: 'basic', label: 'basic, s = 1' },
            { value: 'scaled', label: `scaled, s = ${S_SCALED}` },
          ]}
          value={rule}
          onChange={pickRule}
        />
        <Btn primary onClick={restart}>
          restart
        </Btn>
      </div>

      <div className="mt-4">
        <BigStat
          size="lg"
          label="rank held by D and E"
          value={`${Math.round((trapShare / total) * 100)}`}
          unit="%"
          color={web === 'trap' && trapShare > 0.5 ? GT.warn : GT.ink}
          sub={
            rule === 'scaled'
              ? `every page also gets (1 - s)/5 = ${((1 - S_SCALED) / N).toFixed(2)} each step`
              : `total rank stays ${total.toFixed(2)}: it only moves`
          }
        />
      </div>
      <div className="min-h-[2.8rem]">
        <Caption>{caption}</Caption>
      </div>
    </Specimen>
  );
}
