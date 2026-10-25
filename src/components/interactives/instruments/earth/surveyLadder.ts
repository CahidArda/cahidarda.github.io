/**
 * The earth post's precision ladder. One quantity throughout: the error per kilometre of
 * length measured (a relative error, written as metres per km so it reads as a distance).
 * Figures from the research dossier's precision table; the arithmetic is in each `figure`.
 */
import type { LadderConfig } from '../PrecisionLadder';
import type { Rung } from '../ladderData';

const COLE = 'https://home.hiwaay.net/~jalison/cole2.html';
const STIROS =
  'https://fig.net/resources/proceedings/fig_proceedings/fig2008/papers/hs02/hs02_03_stiros_2837.pdf';

export const SURVEY: Rung[] = [
  {
    name: 'Great Pyramid, side lengths',
    short: 'pyramid sides',
    era: 'c. 2560 BCE',
    year: -2560,
    value: 0.2 / 0.23,
    figure: 'The four sides differ by up to about 20 cm on 230 m: about 87 cm per km.',
    src: COLE,
    srcLabel: 'Cole 1925, survey of the Great Pyramid',
  },
  {
    name: 'Great Pyramid, level of the pavement',
    short: 'pyramid level',
    era: 'c. 2560 BCE',
    year: -2560,
    value: 0.015 / 0.325,
    figure:
      'The pavement falls about 15 mm from one corner to the opposite one, some 325 m: under 5 cm per km.',
    src: COLE,
    srcLabel: 'Cole 1925, survey of the Great Pyramid',
  },
  {
    name: 'Tunnel of Eupalinos, Samos',
    short: 'Eupalinos',
    era: '6th century BCE',
    year: -530,
    value: 0.6 / 1.036,
    figure: 'Two crews dug from both ends and met about 60 cm apart over 1,036 m: about 58 cm per km.',
    src: STIROS,
    srcLabel: 'Stiros, FIG 2008',
  },
  {
    name: 'Roman levelling',
    short: 'Roman levelling',
    era: '1st century CE',
    year: 50,
    value: 0.25,
    figure: 'Ancient levelling error of about 25 cm per √km, so about 25 cm over one kilometre.',
    src: STIROS,
    srcLabel: 'Stiros, FIG 2008',
  },
  {
    name: 'Snellius, triangulation of Holland',
    short: 'Snellius',
    era: '1615',
    year: 1615,
    value: 35,
    figure: 'His 33-triangle chain came out about 3.5% short: 35 m per km.',
    src: 'https://en.wikipedia.org/wiki/Triangulation_%28surveying%29',
    srcLabel: 'Wikipedia: Triangulation (surveying)',
  },
  {
    name: "Ramsden's great theodolite",
    short: 'Ramsden',
    era: 'c. 1787',
    year: 1787,
    value: 1000 * Math.tan(Math.PI / 180 / 3600),
    figure: 'Horizontal angles within 1 arcsecond. A 1″ error puts a sight line 4.8 mm off at 1 km.',
    src: 'https://en.wikipedia.org/wiki/Ramsden_surveying_instruments',
    srcLabel: 'Wikipedia: Ramsden surveying instruments',
  },
  {
    name: 'Great Trigonometrical Survey: Peak XV',
    short: 'Everest',
    era: '1856',
    year: 1856,
    value: (30 / 29_032) * 1000,
    figure:
      '29,002 ft against today’s 29,032 ft, from stations over 100 miles away: 0.1%, about 1 m per km.',
    src: 'https://www.gpsworld.com/peak-xv-the-framework-that-measured-mount-everest/',
    srcLabel: 'GPS World: Peak XV',
  },
  {
    name: 'Total station (Leica TS13)',
    short: 'total station',
    era: 'today',
    year: 2024,
    value: 0.002,
    figure: 'Distance to a prism within 1 mm + 1 ppm: 2 mm at 1 km.',
    src: 'https://cm.leica-geosystems.com/-/media/files/leicageosystems/products/datasheets/leica-ts13-lr-ds.ashx',
    srcLabel: 'Leica TS13 datasheet',
  },
  {
    name: 'RTK satellite positioning (Trimble SPS855)',
    short: 'RTK GNSS',
    era: 'today',
    year: 2024,
    value: 0.009,
    figure:
      'Horizontal position within 8 mm + 1 ppm: 9 mm at 1 km. Worse than the total station, but it needs no line of sight.',
    src: 'https://spaces.awi.de/download/attachments/183566393/Trimble_SPS855_specification.pdf?api=v2',
    srcLabel: 'Trimble SPS855 specification',
  },
];

export function fmtPerKm(m: number): string {
  if (m >= 1) return `${+m.toPrecision(2)} m`;
  if (m >= 0.01) return `${+(m * 100).toPrecision(2)} cm`;
  return `${+(m * 1000).toPrecision(2)} mm`;
}

export const SURVEY_LADDER: LadderConfig = {
  rungs: SURVEY,
  ticks: [
    { v: 100, label: '100 m' },
    { v: 10, label: '10 m' },
    { v: 1, label: '1 m' },
    { v: 0.1, label: '10 cm' },
    { v: 0.01, label: '1 cm' },
    { v: 0.001, label: '1 mm' },
  ],
  lo: -3,
  hi: 2,
  fmt: fmtPerKm,
  unit: 'error per kilometre measured',
  per: 'per km',
  start: 4,
  years: { y0: -2700, yb: 1500, y1: 2040, split: 0.38 },
  xTicks: {
    wide: [-2500, -1000, 500, 1500, 1700, 1800, 1900, 2000],
    narrow: [-2500, -500, 1500, 1800, 2000],
  },
  // the best-so-far line runs right from 'pyramid level' and 'Ramsden': lift their labels off it
  dy: (i) => (i === 1 || i === 5 ? -10 : 0),
};
