import { GT } from '../gt/shared';

/**
 * One colour per boom, used in every figure of the post:
 *   railways  GT.alt  (violet)  aviation       GT.warn (amber)
 *   telecom   GT.a    (blue)    AI chips       GT.eq  (site accent)
 */
export const BOOM = {
  rail: { color: GT.alt, soft: 'color-mix(in srgb, var(--color-gt-alt) 22%, transparent)' },
  air: { color: GT.warn, soft: 'color-mix(in srgb, var(--color-gt-warn) 22%, transparent)' },
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
    lifeLabel: 'much of it still part of the network',
    ...BOOM.rail,
  },
  {
    id: 'fiber',
    label: 'Fiber-optic cable',
    built: 'laid in the 1990s telecom boom',
    short: 'fiber cables',
    life: 999,
    lifeLabel: 'expected to last 25 to 40 years',
    ...BOOM.fiber,
  },
  {
    id: 'plane',
    label: 'Airliner',
    built: 'ordered in an airline boom',
    short: 'airliners',
    life: 25,
    lifeLabel: 'depreciated over 25 to 30 years',
    ...BOOM.air,
  },
  {
    id: 'gpu',
    label: 'AI data-center GPU',
    built: 'bought in the 2020s AI boom',
    short: 'GPUs',
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
  factA: { label: string; value: string; sub: string };
  factB: { label: string; value: string; sub: string };
  ends: 'bust' | 'open';
  bustStamp: string;
  intro: string;
  outro: string;
  source: string;
};

export const LOOPS: Record<LoopId, LoopCase> = {
  '1999': {
    supplier: 'Lucent',
    supplierSub: 'network gear',
    customer: 'Winstar',
    customerSub: 'startup carrier',
    endSub: 'subscribers',
    supColor: GT.a,
    cusColor: GT.opt,
    lendVerb: 'lends money to',
    outLabel: 'loans',
    backLabel: 'equipment orders',
    endLabel: 'subscriptions',
    factA: { label: 'credit Lucent promised customers', value: '$8.4B', sub: 'end of 1999, 10-Q' },
    factB: { label: 'loans to Winstar', value: 'written off', sub: 'fully reserved, April 2001' },
    ends: 'bust',
    bustStamp: 'BANKRUPT 2001',
    intro: 'Lucent sells network gear to new carriers and lends them the money to buy it.',
    outro: 'Winstar filed for bankruptcy in April 2001, and Lucent wrote off its loans.',
    source: 'Lucent 10-Q and 8-K',
  },
  '2026': {
    supplier: 'Nvidia',
    supplierSub: 'AI chips',
    customer: 'OpenAI',
    customerSub: 'AI lab',
    endSub: 'ChatGPT users',
    supColor: GT.eq,
    cusColor: GT.opt,
    lendVerb: 'invests in',
    outLabel: 'investment, lease backstop',
    backLabel: 'chip purchases',
    endLabel: 'subscriptions',
    factA: {
      label: 'Nvidia investment in OpenAI',
      value: '$30B',
      sub: 'February 2026, per Reuters',
    },
    factB: {
      label: 'lease backstop for OpenAI',
      value: 'up to $105B',
      sub: 'Nvidia 10-Q, July 2026',
    },
    ends: 'open',
    bustStamp: '',
    intro: 'Nvidia sells chips to an AI lab that is also one of its investments.',
    outro: 'As of November 2026 the loop is still turning.',
    source: 'Nvidia 10-Q; Reuters',
  },
};

/* ───────────── BX-04 Boeing's order book ───────────── */

/**
 * Boeing commercial airplanes per year. `net` = net orders after cancellations and
 * conversions, and from 2018 also after the ASC 606 backlog adjustment (Boeing's own basis
 * for its negative 2019 and 2020 totals and of its Q4 results releases). Sources are listed
 * per year in the article's "Sources and data" section.
 */
export const BOEING: { year: number; net: number; delivered: number; note: string }[] = [
  {
    year: 2015,
    net: 768,
    delivered: 762,
    note: '2015: a record 762 deliveries and 768 new orders. The book keeps growing.',
  },
  { year: 2016, net: 668, delivered: 748, note: '2016: orders and deliveries roughly in balance.' },
  {
    year: 2017,
    net: 912,
    delivered: 763,
    note: '2017: 912 net orders, worth $134.8 billion at list prices.',
  },
  {
    year: 2018,
    net: 893,
    delivered: 806,
    note: '2018: record deliveries, and 893 more planes ordered.',
  },
  {
    year: 2019,
    net: -87,
    delivered: 380,
    note: '2019: the 737 MAX is grounded. After cancellations and an accounting adjustment, net orders fall below zero.',
  },
  {
    year: 2020,
    net: -1026,
    delivered: 157,
    note: '2020: buyers cancel 641 MAX jets and Boeing drops 523 more it no longer expects to deliver. The book shrinks by 1,026 planes.',
  },
  { year: 2021, net: 535, delivered: 340, note: '2021: orders come back as airlines recover.' },
  {
    year: 2022,
    net: 808,
    delivered: 480,
    note: '2022: 808 net orders, with deliveries climbing back to 480.',
  },
  {
    year: 2023,
    net: 1576,
    delivered: 528,
    note: '2023: 1,576 net orders, three times the planes delivered.',
  },
  {
    year: 2024,
    net: 279,
    delivered: 348,
    note: '2024: deliveries fall to 348, and new orders fall below them.',
  },
  {
    year: 2025,
    net: 1173,
    delivered: 600,
    note: '2025: the backlog passes 6,100 airplanes, about ten years of production at this pace.',
  },
];
