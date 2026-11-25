/**
 * Election data for "Gaming the Ballot: 1789 and 1800".
 * 1789 and 1800 tallies: National Archives, Office of the Federal Register,
 * https://www.archives.gov/electoral-college/1789 and /1800 (state-by-state tables).
 * House ballot 36: A New Nation Votes (Tufts / AAS), https://elections.lib.tufts.edu/catalog/gx41mj27q
 * Ballots 1 to 35 (8 Jefferson, 6 Burr, 2 divided each time): Britannica, "United States
 * presidential election of 1800".
 */

/* ───────────── 1789: the 69 electors and their second votes ───────────── */

export type Rec =
  | 'Adams'
  | 'Jay'
  | 'Harrison'
  | 'Rutledge'
  | 'Hancock'
  | 'Clinton'
  | 'Huntington'
  | 'Milton'
  | 'Armstrong'
  | 'Lincoln'
  | 'Telfair';

/** Recipients in order of votes (NARA totals). */
export const RECIPIENTS: { id: Rec; name: string; votes: number }[] = [
  { id: 'Adams', name: 'John Adams', votes: 34 },
  { id: 'Jay', name: 'John Jay', votes: 9 },
  { id: 'Harrison', name: 'Robert H. Harrison', votes: 6 },
  { id: 'Rutledge', name: 'John Rutledge', votes: 6 },
  { id: 'Hancock', name: 'John Hancock', votes: 4 },
  { id: 'Clinton', name: 'George Clinton', votes: 3 },
  { id: 'Huntington', name: 'Samuel Huntington', votes: 2 },
  { id: 'Milton', name: 'John Milton', votes: 2 },
  { id: 'Armstrong', name: 'James Armstrong', votes: 1 },
  { id: 'Lincoln', name: 'Benjamin Lincoln', votes: 1 },
  { id: 'Telfair', name: 'Edward Telfair', votes: 1 },
];

/** Every voting elector wrote Washington plus one other name (Washington got 69 of 69). */
export const STATES_1789: {
  id: string;
  name: string;
  second: Partial<Record<Rec, number>>;
  note: string;
}[] = [
  { id: 'NH', name: 'New Hampshire', second: { Adams: 5 }, note: 'All 5: Washington + Adams.' },
  { id: 'MA', name: 'Massachusetts', second: { Adams: 10 }, note: 'All 10: Washington + Adams.' },
  {
    id: 'CT',
    name: 'Connecticut',
    second: { Adams: 5, Huntington: 2 },
    note: 'Hamilton asked Connecticut to throw away two. Two wrote Samuel Huntington, the governor.',
  },
  {
    id: 'NJ',
    name: 'New Jersey',
    second: { Adams: 1, Jay: 5 },
    note: 'Hamilton asked for two throwaways. Five of six wrote John Jay.',
  },
  {
    id: 'PA',
    name: 'Pennsylvania',
    second: { Adams: 8, Hancock: 2 },
    note: 'Hamilton suggested "three or four". Two wrote John Hancock.',
  },
  { id: 'DE', name: 'Delaware', second: { Jay: 3 }, note: 'All 3 wrote John Jay.' },
  {
    id: 'MD',
    name: 'Maryland',
    second: { Harrison: 6 },
    note: 'All 6 who voted wrote Robert H. Harrison, a Maryland judge. Two of its 8 electors did not vote.',
  },
  {
    id: 'VA',
    name: 'Virginia',
    second: { Adams: 5, Clinton: 3, Jay: 1, Hancock: 1 },
    note: 'The only Clinton votes, Washington still on every ballot.',
  },
  {
    id: 'SC',
    name: 'South Carolina',
    second: { Rutledge: 6, Hancock: 1 },
    note: 'Six wrote John Rutledge of South Carolina.',
  },
  {
    id: 'GA',
    name: 'Georgia',
    second: { Milton: 2, Armstrong: 1, Lincoln: 1, Telfair: 1 },
    note: 'Madison: Georgia\'s second votes were "all thrown away".',
  },
];

export type Elector = { i: number; state: number; rec: Rec; inState: number; inRec: number };

/** One record per voting elector, ordered by state then recipient. */
export function electors(): Elector[] {
  const out: Elector[] = [];
  const recCount: Record<string, number> = {};
  STATES_1789.forEach((s, si) => {
    let k = 0;
    for (const r of RECIPIENTS) {
      const n = s.second[r.id] ?? 0;
      for (let j = 0; j < n; j++) {
        const inRec = recCount[r.id] ?? 0;
        recCount[r.id] = inRec + 1;
        out.push({ i: out.length, state: si, rec: r.id, inState: k++, inRec });
      }
    }
  });
  return out;
}

/* ───────────── 1789: the arithmetic of the second vote ───────────── */

export const N_1789 = 69;
export const MAJ_1789 = 35;

/**
 * a = Federalist electors who write Washington + Adams,
 * k = electors who write Adams + someone else (leaving Washington off).
 * Every other elector writes Washington + a third name.
 */
export function race(a: number, k: number) {
  const W = N_1789 - k;
  const A = a + k;
  const verdict: 'washington' | 'tie' | 'adams' = A > W ? 'adams' : A === W ? 'tie' : 'washington';
  return { W, A, verdict };
}

/* ───────────── 1800 ───────────── */

export const TALLY_1800 = [
  { name: 'Jefferson', votes: 73, side: 'R' as const },
  { name: 'Burr', votes: 73, side: 'R' as const },
  { name: 'Adams', votes: 65, side: 'F' as const },
  { name: 'Pinckney', votes: 64, side: 'F' as const },
  { name: 'Jay', votes: 1, side: 'F' as const },
];
export const N_1800 = 138;
export const MAJ_1800 = 70;

export type HouseVote = 'J' | 'B' | 'div' | 'blank';

/** House of Representatives, February 1801: one vote per state delegation, 9 of 16 to win. */
export const HOUSE_1801: { id: string; first35: HouseVote; last: HouseVote }[] = [
  { id: 'GA', first35: 'J', last: 'J' },
  { id: 'KY', first35: 'J', last: 'J' },
  { id: 'NJ', first35: 'J', last: 'J' },
  { id: 'NY', first35: 'J', last: 'J' },
  { id: 'NC', first35: 'J', last: 'J' },
  { id: 'PA', first35: 'J', last: 'J' },
  { id: 'TN', first35: 'J', last: 'J' },
  { id: 'VA', first35: 'J', last: 'J' },
  { id: 'MD', first35: 'div', last: 'J' },
  { id: 'VT', first35: 'div', last: 'J' },
  { id: 'DE', first35: 'B', last: 'blank' },
  { id: 'SC', first35: 'B', last: 'blank' },
  { id: 'CT', first35: 'B', last: 'B' },
  { id: 'MA', first35: 'B', last: 'B' },
  { id: 'NH', first35: 'B', last: 'B' },
  { id: 'RI', first35: 'B', last: 'B' },
];
export const BALLOTS = 36;
export const STATES_NEEDED = 9;

export const houseVote = (row: number, ballot: number): HouseVote =>
  ballot >= BALLOTS ? HOUSE_1801[row].last : HOUSE_1801[row].first35;

export function houseCount(ballot: number) {
  let J = 0;
  let B = 0;
  HOUSE_1801.forEach((_, r) => {
    const v = houseVote(r, ballot);
    if (v === 'J') J++;
    if (v === 'B') B++;
  });
  return { J, B };
}
