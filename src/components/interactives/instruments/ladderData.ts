/**
 * Data for the precision ladder on the series landing. Each rung is one instrument and the
 * best error it could hold. `oom: true` marks an honest order-of-magnitude estimate (no single
 * sourced figure exists); everything else is taken from the cited source, with the arithmetic
 * spelled out in `figure`.
 *
 * time: error in SECONDS PER DAY.  angle: error in ARCSECONDS.
 */
export interface Rung {
  name: string;
  /** Short label drawn on the chart. */
  short: string;
  /** Display date. */
  era: string;
  /** Year for the x axis (negative = BCE). */
  year: number;
  /** Error in base units (s/day or arcsec). */
  value: number;
  /** How the number was arrived at, in words. */
  figure: string;
  oom?: boolean;
  src: string;
  srcLabel: string;
}

const DAY = 86400;
const YEAR = 365.25;

const WIKI_PENDULUM = 'https://en.wikipedia.org/wiki/Pendulum_clock';
const HOG = 'https://arxiv.org/pdf/1707.01020';

export const TIME: Rung[] = [
  {
    name: 'Shadow clock and water clock',
    short: 'sundial, clepsydra',
    era: 'c. 1500 BCE',
    year: -1500,
    value: 900,
    figure:
      'Order of magnitude, a quarter of an hour. The sun is half a degree wide so the shadow edge is soft, water flow changes with level and temperature, and the hours themselves were often unequal.',
    oom: true,
    src: 'https://en.wikipedia.org/wiki/Water_clock',
    srcLabel: 'Wikipedia: Water clock',
  },
  {
    name: 'Verge-and-foliot clock',
    short: 'verge',
    era: 'c. 1300',
    year: 1300,
    value: 15 * 60,
    figure: '"accurate to about 15 minutes per day" before the pendulum.',
    src: WIKI_PENDULUM,
    srcLabel: 'Wikipedia: Pendulum clock',
  },
  {
    name: 'Pendulum clock (Huygens)',
    short: 'pendulum',
    era: '1656',
    year: 1656,
    value: 15,
    figure: 'The pendulum "increased their accuracy to 15 seconds per day".',
    src: WIKI_PENDULUM,
    srcLabel: 'Wikipedia: Pendulum clock',
  },
  {
    name: "Harrison's H4 chronometer",
    short: 'H4',
    era: '1761',
    year: 1761,
    value: 5 / 61,
    figure: '5 seconds slow after the roughly two-month voyage to Jamaica: about 0.08 s a day.',
    src: 'https://www.lindahall.org/about/news/scientist-of-the-day/john-harrison',
    srcLabel: 'Linda Hall Library: John Harrison',
  },
  {
    name: 'Shortt-Synchronome free-pendulum clock',
    short: 'Shortt',
    era: '1920s',
    year: 1925,
    value: 1 / YEAR,
    figure: '"better than one second per year": under 3 ms a day.',
    src: WIKI_PENDULUM,
    srcLabel: 'Wikipedia: Pendulum clock',
  },
  {
    name: 'Quartz clock',
    short: 'quartz',
    era: '1930s',
    year: 1935,
    value: 30 / (300 * YEAR),
    figure:
      'The 1955 caesium clock was "about 30 times better" than 1930s quartz clocks: about 0.3 ms a day.',
    src: 'https://collection.sciencemuseumgroup.org.uk/objects/co855/caesium-atomic-clock-1955',
    srcLabel: 'Science Museum: caesium atomic clock, 1955',
  },
  {
    name: 'Caesium clock (Essen and Parry)',
    short: 'caesium',
    era: '1955',
    year: 1955,
    value: 1 / (300 * YEAR),
    figure: 'About one second in 300 years: roughly 9 µs a day.',
    src: 'https://collection.sciencemuseumgroup.org.uk/objects/co855/caesium-atomic-clock-1955',
    srcLabel: 'Science Museum: caesium atomic clock, 1955',
  },
  {
    name: 'NIST-F2 caesium fountain',
    short: 'NIST-F2',
    era: '2014',
    year: 2014,
    value: 1 / (300e6 * YEAR),
    figure:
      'Designed to "not gain or lose a second in at least 300 million years" (δf/f < 1e-16): about 9 ps a day.',
    src: 'https://en.wikipedia.org/wiki/NIST-F2',
    srcLabel: 'Wikipedia: NIST-F2',
  },
  {
    name: 'Strontium optical lattice clock (JILA)',
    short: 'optical',
    era: '2024',
    year: 2024,
    value: 8.1e-19 * DAY,
    figure: 'Total systematic uncertainty 8.1 × 10⁻¹⁹: times 86,400 s, about 0.07 ps a day.',
    src: 'https://arxiv.org/abs/2403.10664',
    srcLabel: 'Aeppli et al. 2024, arXiv:2403.10664',
  },
];

export const ANGLE: Rung[] = [
  {
    name: 'Gnomon (shadow stick)',
    short: 'gnomon',
    era: 'c. 240 BCE',
    year: -240,
    value: 30 * 60,
    figure:
      'Order of magnitude. The sun is about half a degree across, so the shadow tip is blurred by roughly that much.',
    oom: true,
    src: 'https://en.wikipedia.org/wiki/Gnomon',
    srcLabel: 'Wikipedia: Gnomon',
  },
  {
    name: "Ptolemy's star catalogue",
    short: 'Ptolemy',
    era: 'c. 150',
    year: 150,
    value: 25 * 60,
    figure: 'Median error of star positions about 25′ (Høg 2017, Table 1).',
    src: HOG,
    srcLabel: 'Høg 2017, Astrometric accuracy during the past 2000 years',
  },
  {
    name: "Ulugh Beg's catalogue (Samarkand)",
    short: 'Ulugh Beg',
    era: '1437',
    year: 1437,
    value: 20 * 60,
    figure: 'About 20′ (Høg 2017, Table 1).',
    src: HOG,
    srcLabel: 'Høg 2017',
  },
  {
    name: "Tycho Brahe's mural quadrant",
    short: 'Tycho',
    era: 'c. 1600',
    year: 1600,
    value: 2 * 60,
    figure: 'About 2′, a tenfold jump in one lifetime (Høg 2017, Table 1).',
    src: HOG,
    srcLabel: 'Høg 2017',
  },
  {
    name: 'Marine sextant (vernier reading)',
    short: 'sextant',
    era: '18th c.',
    year: 1760,
    value: 6,
    figure:
      'A vernier "reads to 0.1 minute" (6″). That is reading resolution; on a rolling deck the real error is larger.',
    src: 'https://en.wikipedia.org/wiki/Sextant',
    srcLabel: 'Wikipedia: Sextant',
  },
  {
    name: "Ramsden's great theodolite",
    short: 'theodolite',
    era: '1787',
    year: 1787,
    value: 1,
    figure: 'Horizontal angles "within one arc second".',
    src: 'https://en.wikipedia.org/wiki/Ramsden_surveying_instruments',
    srcLabel: 'Wikipedia: Ramsden surveying instruments',
  },
  {
    name: "Argelander's meridian-circle catalogue",
    short: 'meridian circle',
    era: '1856',
    year: 1856,
    value: 0.9,
    figure: 'About 0.9″ (Høg 2017).',
    src: HOG,
    srcLabel: 'Høg 2017',
  },
  {
    name: 'Hipparcos satellite',
    short: 'Hipparcos',
    era: '1997',
    year: 1997,
    value: 0.001,
    figure: 'Median 1 milliarcsecond for 120,000 stars (Høg 2017).',
    src: HOG,
    srcLabel: 'Høg 2017',
  },
  {
    name: 'Gaia satellite (bright stars)',
    short: 'Gaia',
    era: '2013 to 2025',
    year: 2020,
    value: 10e-6,
    figure:
      'Expected median about 10 microarcseconds for stars brighter than magnitude 14 (Høg 2017).',
    src: HOG,
    srcLabel: 'Høg 2017',
  },
];

/** Human tick marks for the log axis. */
export const TIME_TICKS: { v: number; label: string }[] = [
  { v: 3600, label: '1 hour' },
  { v: 60, label: '1 min' },
  { v: 1, label: '1 s' },
  { v: 1e-3, label: '1 ms' },
  { v: 1e-6, label: '1 µs' },
  { v: 1e-9, label: '1 ns' },
  { v: 1e-12, label: '1 ps' },
];
export const ANGLE_TICKS: { v: number; label: string }[] = [
  { v: 3600, label: '1°' },
  { v: 60, label: '1′' },
  { v: 1, label: '1″' },
  { v: 1e-3, label: '1 mas' },
  { v: 1e-6, label: '1 µas' },
];

/** Compact human formatting of a value in base units. */
export function fmtTime(s: number): string {
  if (s >= 60) return `${+(s / 60).toPrecision(2)} min`;
  if (s >= 1) return `${+s.toPrecision(2)} s`;
  if (s >= 1e-3) return `${+(s * 1e3).toPrecision(2)} ms`;
  if (s >= 1e-6) return `${+(s * 1e6).toPrecision(2)} µs`;
  if (s >= 1e-9) return `${+(s * 1e9).toPrecision(2)} ns`;
  return `${+(s * 1e12).toPrecision(2)} ps`;
}
export function fmtAngle(a: number): string {
  if (a >= 3600) return `${+(a / 3600).toPrecision(2)}°`;
  if (a >= 60) return `${+(a / 60).toPrecision(2)}′`;
  if (a >= 1) return `${+a.toPrecision(2)}″`;
  if (a >= 1e-3) return `${+(a * 1e3).toPrecision(2)} mas`;
  return `${+(a * 1e6).toPrecision(2)} µas`;
}
