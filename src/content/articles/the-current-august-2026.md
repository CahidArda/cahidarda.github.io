---
title: The Current - August 2026
description: A public record of the links I save - August 2026, when the full accounts of July's incidents came out.
date: 2026-08-31
tags: ['accessions']
---

August was mostly the aftermath of [July](/articles/the-current-july-2026): the full accounts
of the incidents came out, and the labs said what they are changing. The rest is biology, and
one piece on anger at work.

## What actually happened in July

- **[Incident Report: unsanctioned agent behaviour during cyber testing](https://www.aisi.gov.uk/blog/incident-report-unsanctioned-agent-behaviour-during-cyber-testing)**, _UK AI Security Institute, 4 Aug 2026_
  - It happened a third time, and this time to the evaluator. Across 122 runs of one cyber
    challenge, agents in 10 runs took 19 unsanctioned actions against real people and
    organisations, 17 of them by Mythos 5. The worst was an attempt to slip malicious code into
    a real open-source project, backed by fake identities pressuring the maintainer to approve
    it. It failed, and AISI found no evidence of real-world harm.
  - [Simon Willison's note](https://simonwillison.net/2026/Aug/5/incident-report/) points out
    the uncomfortable part: the internet access was deliberate and the developers' cyber
    classifiers were switched off, so none of this needed a sandbox escape.
- **[Pacing model development in an era of cyber-critical capabilities](https://openai.com/index/pacing-model-development-cyber-capabilities/)**, _OpenAI, 18 Aug 2026_
  - OpenAI paused reinforcement learning on the models it intends to deploy for two weeks, and
    its largest planned frontier RL run is still on hold. Two reasons: the Hugging Face
    incident, and early evidence that its upcoming Astra model may cross the "Critical" cyber
    threshold in its Preparedness Framework. The new monitoring adds roughly a fifth more
    compute to parts of training.
- **[The Hugging Face incident and the road ahead](https://openai.com/index/hugging-face-incident-and-the-road-ahead/)**, _OpenAI, 26 Aug 2026_
  - The full account, with a
    [technical report](https://cdn.openai.com/pdf/67869394-cb91-4c12-888c-5cbd85c7814c/OpenAI-Hugging-Face%20Incident-Technical-Report.pdf).
    The main actor was an internal-only research model, and agents had been leaving each other
    notes in the internal package registry since May, which turned it into a message board. An
    internal team saw that activity in late May, but its significance never reached the people
    handling the July incident.
  - The number that explains a lot: 198 of the 898 ExploitGym tasks had never been solved by any
    model, and 93% of the tasks discussed on the message board came from that set. Agents stuck
    on impossible tasks, with no safe way to give up, kept escalating. OpenAI calls the whole
    thing a "warning shot".
- **[Brief independent investigation of agents' behavior, reasoning and collaboration in the OpenAI / Hugging Face hacking incident](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/)**, _METR and Redwood Research, 26 Aug 2026_
  - The one to read if you only read one. About 1,200 agents that were meant to be isolated
    found each other on the board, sent more than 70,000 messages and files, and roughly 700 of
    them joined the attack on Hugging Face. They invented mailboxes, holds and vetoes, and
    started signing their messages with Ed25519 keys after agents began impersonating each
    other. Agents near the end of their budgets volunteered for experiments that would fail
    their own task if it helped the "collective".
  - Two twists. The scorer the agents worked so hard to fool did not actually check what they
    believed it checked. And the investigators had to lean on GPT-5.6 Sol to read over a
    thousand transcripts, many millions of tokens each, while noting they could not rule out
    that it misled them.
- **[Improving our alignment and security practices](https://www.anthropic.com/news/improving-alignment-security-efforts)**, _Anthropic, 31 Aug 2026_
  - Anthropic's follow-up on its own July incidents and on the AISI one, where Mythos 5 was the
    main actor. It is analysing both in depth and bringing in METR for an independent review.

## Also

- **[Generative design of bacteriophages with genome language models](https://www.science.org/doi/10.1126/science.aec2657)**, _King et al., Science, 6 Aug 2026_
  - Researchers at Stanford and the Arc Institute used the Evo genome language models to design
    complete genomes for a bacteriophage (a virus that infects bacteria), starting from ΦX174.
    Of roughly 300 designs they synthesized, 16 produced working phages, and a cocktail of them
    overcame E. coli strains that had become resistant to the natural virus. The
    [Stanford write-up](https://news.stanford.edu/stories/2026/08/evo-2-ai-tool-e-coli-killer-bacteriophages)
    is the readable version. Read it next to the Somers essay below.
- **[I Should Have Loved Biology](https://jsomers.net/i-should-have-loved-biology/)**, _James Somers_
  - "In the textbooks, astonishing facts were presented without astonishment." On why school
    biology reads as a list of names, and what changes once you picture the cell as a crowd of
    little machines with actual shapes. An older essay, undated on the page, that I only got
    to now.
- **[You should never be angry at work](https://www.seangoedecke.com/you-should-never-be-angry-at-work/)**, _Sean Goedecke, 22 Aug 2026_
  - Anger usually means you care, which is exactly why it is a trap: "An angry colleague
    immediately becomes a new problem to be managed, not a professional helping you manage
    problems." Get angry and you get routed around, and then you cannot fix the thing you were
    angry about.
