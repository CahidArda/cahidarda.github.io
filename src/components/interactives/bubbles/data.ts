import { GT } from '../gt/shared';

/**
 * One colour per boom, used in every figure of the post:
 *   railways  GT.warn (amber)   radio/aviation GT.alt (violet)
 *   telecom   GT.a    (blue)    AI chips       GT.eq  (site accent)
 */
export const BOOM = {
  rail: { color: GT.warn, soft: 'color-mix(in srgb, var(--color-gt-warn) 22%, transparent)' },
  radio: { color: GT.alt, soft: 'color-mix(in srgb, var(--color-gt-alt) 22%, transparent)' },
  fiber: { color: GT.a, soft: GT.aSoft },
  ai: { color: GT.eq, soft: 'color-mix(in srgb, var(--color-accent) 22%, transparent)' },
} as const;

export type Asset = {
  id: string;
  label: string;
  built: string;
  short: string;
  life: number; // useful years after the crash (> 30 means "still in use")
  lifeLabel: string;
  color: string;
  soft: string;
};

export const ASSETS: Asset[] = [
  {
    id: 'rail',
    label: 'Railway line',
    built: 'built in the 1840s mania',
    short: 'railway lines',
    life: 999,
    lifeLabel: 'many still carry trains, 180 years on',
    ...BOOM.rail,
  },
  {
    id: 'fiber',
    label: 'Fiber-optic cable',
    built: 'laid in the 1990s telecom boom',
    short: 'fiber',
    life: 25,
    lifeLabel: '25+ years, most of it lit later',
    ...BOOM.fiber,
  },
  {
    id: 'plane',
    label: 'Airliner',
    built: 'ordered in an airline boom',
    short: 'airliners',
    life: 27,
    lifeLabel: 'about 25 to 30 years',
    ...BOOM.radio,
  },
  {
    id: 'gpu',
    label: 'AI data-center GPU',
    built: 'bought in the 2020s AI boom',
    short: 'the GPUs',
    life: 6,
    lifeLabel: 'depreciated over 5 to 6 years',
    ...BOOM.ai,
  },
];

/* ───────────── BX-02 the loop (vendor financing) ───────────── */

export type LoopId = '1999' | '2026';
export type LoopCase = {
  supplier: string;
  supplierSub: string;
  customer: string;
  customerSub: string;
  endSub: string;
  supColor: string;
  cusColor: string;
  lendVerb: string;
  outLabel: string;
  backLabel: string;
  endLabel: string;
  perLap: number; // $B per lap: the documented total split over three laps
  lapSub: string;
  ends: 'bust' | 'open';
  bustStamp: string;
  bustSub: string;
  intro: string;
  outro: string;
  source: string;
};

export const LOOPS: Record<LoopId, LoopCase> = {
  '1999': {
    supplier: 'Lucent',
    supplierSub: 'telecom equipment',
    customer: 'Winstar',
    customerSub: 'startup carrier',
    endSub: 'phone and data subscribers',
    supColor: GT.a,
    cusColor: GT.alt,
    lendVerb: 'lends money to',
    outLabel: 'loans',
    backLabel: 'equipment orders',
    endLabel: 'subscriptions',
    perLap: 2 / 3,
    lapSub: 'TODO',
    ends: 'bust',
    bustStamp: 'BANKRUPT 2001',
    bustSub: 'TODO',
    intro: 'TODO',
    outro: 'TODO',
    source: 'TODO',
  },
  '2026': {
    supplier: 'Nvidia',
    supplierSub: 'AI chips',
    customer: 'OpenAI',
    customerSub: 'AI lab',
    endSub: 'ChatGPT and API customers',
    supColor: GT.eq,
    cusColor: GT.opt,
    lendVerb: 'invests in',
    outLabel: 'investment',
    backLabel: 'chip purchases',
    endLabel: 'subscriptions',
    perLap: 100 / 3,
    lapSub: 'TODO',
    ends: 'open',
    bustStamp: '',
    bustSub: '',
    intro: 'TODO',
    outro: 'TODO',
    source: 'TODO',
  },
};
