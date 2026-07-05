# robots-vs-ai — source verification log

How each source cited in the post (and the outline brief) was retrieved and checked.
"Full text" = I extracted the actual document text (PDF via `pypdf`, or full HTML page) and
matched quotes/numbers word-for-word. "Secondary" = confirmed via an authoritative summary or
abstract, not the primary full text.

Retrieval methods used:
- **WebFetch** — fetches a URL and reads it. Works for HTML pages; often fails on PDFs (returns
  raw/encoded stream) and on paywalled hosts (303 redirect to auth) or bot-blocked hosts (403).
- **WebFetch → local pypdf** — when WebFetch saved the PDF binary to disk, I ran `pypdf` locally
  to extract the text, then grepped it. This is how most PDFs got read.
- **curl + pypdf** — downloaded the PDF directly, then extracted with `pypdf`.
- **WebSearch** — used to confirm figures and to locate open-access mirrors of paywalled papers.
- **Local file paste** — for three sources I couldn't fetch at all (paywall / 403), the user pasted
  the full text into the `*.md` files in this folder; I then verified against that. See the
  "Full text added later" section below.

---

## Load-bearing sources — FULL TEXT obtained

| Source | Used for | How I got the full text | Status |
|---|---|---|---|
| **Acemoglu & Restrepo, "Robots and Jobs" (published JPE 128(6), 2020)** | The 0.2pp/0.42% quote; six-workers-local / 3.3-aggregate; −0.39/−0.77 local; "no positive effect on masters/doctoral"; Detroit | `curl` the [MIT shapingwork PDF](https://shapingwork.mit.edu/wp-content/uploads/2023/10/Robots-and-Jobs-Evidence-from-US-Labor-Markets.p.pdf) → `pypdf` (57pp) | ✅ verbatim |
| **Acemoglu & Restrepo, NBER WP 23285 (2017)** | Cross-check — caught that the post's quote used the WP's *range* phrasing, not the JPE's final numbers | WebFetch saved the [NBER PDF](https://www.nber.org/system/files/working_papers/w23285/w23285.pdf) → `pypdf` (91pp) | ✅ verbatim |
| **Dauth, Findeisen, Südekum & Wößner, "German Robots" (IAB DP 30/2017)** | "no evidence that robots cause total job losses… offset by service-sector jobs" | `curl` [doku.iab.de/discussionpapers/2017/dp3017.pdf](https://doku.iab.de/discussionpapers/2017/dp3017.pdf) → `pypdf` (63pp). Found via WebSearch after the post's original link turned out wrong (see notes) | ✅ verbatim |
| **Lábaj, Oleš & Procházka, "Impact of robots and AI on labor and skill demand: evidence from the UK" (Eurasian Business Review, 2025)** | The whole thesis: robots strongest among HS dropouts + decline monotonically w/ education; AI increases monotonically across income percentiles; robots hit the middle of the income distribution | Springer link 303-redirects to auth. WebFetch of the [German National Library copy](https://d-nb.info/1380033667/34) saved the full PDF → `pypdf` (49pp). Confirmed the claims appear in the **body**, not just the abstract | ✅ verbatim (full text) |
| **Schaal, AI-exposure-index paper (arXiv:2510.13369, 2025)** | "false analogies"; robots tangible/rivalrous vs AI intangible/non-rivalrous | WebFetch saved the [arXiv PDF](https://arxiv.org/pdf/2510.13369) → `pypdf` (35pp) | ✅ verbatim |
| **Autor, "Applying AI to Rebuild Middle Class Jobs" (NBER WP 32140, 2024)** | The "AI may reverse polarization" hypothesis; the added quote "extend the relevance, reach, and value of human expertise" | WebFetch saved the [NBER PDF](https://www.nber.org/system/files/working_papers/w32140/w32140.pdf) → `pypdf` (22pp) | ✅ verbatim |
| **Graetz & Michaels, "Robots at Work" (CEP DP 1335 / REStat 2018)** | Robots raised productivity + wages while reducing low-skill hours | WebFetch saved the [LSE CEP PDF](http://cep.lse.ac.uk/pubs/download/dp1335.pdf) → `pypdf` (56pp) | ✅ verbatim |
| **Jacobson, LaLonde & Sullivan (1993)** | Scarring: ~25% loss still present ~5 years out (their actual horizon) | `curl` the [Chicago Fed Economic Perspectives PDF](https://www.chicagofed.org/-/media/publications/economic-perspectives/1993/ep-nov-dec1993-part1-jacobson-pdf.pdf) (same authors' companion article) → `pypdf` (22pp) | ✅ verbatim |
| **Bipartisan Policy Center, "What Happens When Jobs Disappear" (2025)** | TAA ~$50k / dissipates / wraparound supports; IMF unemployment-benefit cushion | WebFetch of the [explainer page](https://bipartisanpolicy.org/explainer/what-happens-when-jobs-disappear-a-guide-to-displaced-worker-programs-in-the-u-s/) (HTML) | ✅ verbatim |
| **Petrova, Schubert, Taska & Yildirim, "Automation, Career Values, and Political Preferences" (NBER WP 32655, 2024)** | Career value −$3.9K (2004–08) / −$2.48K (2008–16); Trump-2016 vote share; housing + schooling investment | User pasted the full paper → [`yildirim-automation-career-values.md`](./yildirim-automation-career-values.md). Originally confirmed via the [Knowledge@Wharton write-up](https://knowledge.wharton.upenn.edu/article/robots-are-taking-over-low-skilled-jobs-and-changing-votes/) | ✅ verbatim (full text, local file) |

---

## Full text added later — pasted into local files in this folder

I couldn't pull these directly (paywall / bot-block), so the full text was pasted into the local
markdown files below. All three now check out against their **primary** text.

| Source | Used for | Local file | What the full text confirmed |
|---|---|---|---|
| **Couch & Placzek (2010), AER 100(1): 572–89** | Scarring short curve: ~33% immediate, ~15% at 6 years | [`couch-placzek-2010.md`](./couch-placzek-2010.md) | "immediately following job loss… earnings reductions… range from 32 percent to 33 percent… Six years later… 13 percent to 15 percent." Also confirms JLS's PA numbers: ">40 percent" immediate, "25 percent" sustained. ✅ |
| **BLS Worker Displacement, Jan 2024 (USDL-24-1777)** | 65.7% reemployed; 62% equal-or-higher pay; ~34% not reemployed; mfg 17% | [`bls-displaced-workers-jan-2024.md`](./bls-displaced-workers-jan-2024.md) | "65.7 percent… were reemployed"; "62 percent had earnings that were as much or greater"; "Seventeen percent… lost a job in manufacturing" (427,000). Not-reemployed = 16.1% unemployed + 18.2% NILF = 34.3%. **mfg 17% now verified.** ✅ |
| **Petrova, Schubert, Taska & Yildirim (NBER WP 32655, 2024)** | Career-value + voting numbers | [`yildirim-automation-career-values.md`](./yildirim-automation-career-values.md) | "One additional robot per 1000 workers decreased the average local market career value by $3.9K between 2004 and 2008 and by $2.48K between 2008 and 2016"; Trump 2016 vote share; investment in schooling/housing. ✅ |

## Still verified via SECONDARY only (primary full text not obtained)

| Source | Used for | Why not full text | Confirmed via |
|---|---|---|---|
| **von Wachter, Song & Manchester (2009)** | Scarring long horizon: ~30% initial, still ~20% down at 15–20 years (EarningsScar long curve) | Didn't pull the primary PDF | WebSearch: "declines of 30 percent or more… earnings 20 percent lower… even 15 to 20 years after displacement" — matches the widget curve exactly. (Also independently corroborated inside `couch-placzek-2010.md`, which reports JLS's own 25%-sustained PA figure.) |

---

## Narrative / historical sources (qualitative, not re-fetched)

These back the Section 4 timeline and color, not any headline number. They match well-established
history (Levinson, *The Box*, 2006; the 1964 longshore strike; ILWU M&M 1960; Luddites). Left as
cited; not independently pulled:

- Luddites; UMW continuous miner (1950s); ILWU Mechanization & Modernization Agreement (1960);
  East Coast longshore strike (1964); UAW automation-for-security; Teamsters/UPS 2023; ILA 2024.
- EPI "zombie robot" argument; St. Louis Fed (2024); The Century Foundation (2022);
  PMC "Far-Reaching Impact of Job Loss" (linked in-post for the well-being/family claims).

---

## One retrieval note worth keeping

The post originally linked the German study as `docs.iza.org/dp12306.pdf`. When I extracted that
PDF locally, page 1 read **"IZA DP No. 12306 — The Gender Promotion Gap: Evidence from Central
Banking."** The brief had confused the CEPR discussion-paper number (DP12306 = German Robots) with
the IZA number (DP12306 = a different paper). The correct German Robots paper is **IAB DP 30/2017**
(`doku.iab.de/discussionpapers/2017/dp3017.pdf`), which I verified by extracting its title page and
abstract. Lesson: always extract page 1 and confirm the title before trusting a numeric-ID link.
