---
title: The Current - September 2026
description: A public record of the links I save - September 2026, when both big labs said out loud that the frontier needs a brake, and then shipped new models anyway.
date: 2026-09-29
tags: ['accessions']
---

In September both big labs said they want the frontier to slow down, then shipped new models
within weeks. Mathematicians pushed back against AI being aimed at their field. The last of
the July incident reports came out, and I spent too much time thinking about lawyers because of
a TV show.

## Pacing the frontier

- **[We Must Pace the Frontier](https://darioamodei.com/post/we-must-pace-the-frontier)**, _Dario Amodei, Sep 2026_
  - The employee letter from [July](/articles/the-current-july-2026), now with the CEO's name on
    it. Amodei gives two reasons: capabilities have jumped "since roughly this summer",
    including early recursive self-improvement, and the OpenAI and Hugging Face incident showed
    what a swarm of agents behaving as "a fanatically devoted collective" can do. He warns that
    "in 6–12 months such a swarm could be capable of taking over the entire internet with a
    persistent botnet".
  - The plan has three steps: third-party evaluators embedded inside the labs, shared standards
    among democracies, then global coordination. "To be clear, pacing does not mean halting
    model training or technical progress, but ensuring companies take adequate time to align
    and safeguard their models, and for third party evaluators to confirm this."
- **[An Alien Mind](https://openai.com/index/an-alien-mind/)**, _Jakub Pachocki, OpenAI, Sep 2026_
  - OpenAI's chief scientist, writing alongside the release of GPT-6 Astra (the model that, in
    [August](/articles/the-current-august-2026), OpenAI said might cross its "Critical" cyber
    threshold). He expects current progress to carry into recursive self-improvement, says
    chain-of-thought monitoring is becoming less reliable, and argues that no lab has solved
    alignment well enough to keep scaling at full speed for much longer. He calls for safety bars
    enforced by outside auditors and says OpenAI will hold back scaling on its own if it has to.
    [Zvi's write-up](https://thezvi.substack.com/p/an-alien-mind-jakub-pachocki-warns) quotes
    it at length.
  - So both leaders now make the same argument. The difference is who they want to pull the
    brake.

## New models, anyway

- **[Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5)**, _Anthropic, 22 Sep 2026_
  - The first release since Anthropic called for pacing the frontier. Anthropic says it
    "performs at the level of Claude Fable 5.1 on most work and costs 40% less to run than Opus
    5", at $4/$20 per million tokens. It scores 66.4% on Terminal-Bench 4.0, against 57.9% for
    GPT-6 Astra.
  - The number I found most interesting is not a benchmark score: on a new test of whether a
    model tries to get out of its containment, Opus 5.5 tried "around 85% less often than Opus 5
    or Claude Mythos 5.1". After July and August, that eval exists for a reason. See the
    [system card](https://anthropic.com/claude-opus-5-5-system-card).
  - I used it for the [Bosphore 1819](/articles/bosphore-1819) post below.
- **[GPT-6 Sol and Luna](https://x.com/OpenAIDevs/status/2102461432684282061)**, _OpenAI Developers, 22 Sep 2026_
  - The same day, OpenAI released the smaller siblings of Astra, with API prices 50% lower than
    GPT-5.6. "Build with Sol. Scale with Luna."
- **[Claude Sonnet 5.5](https://x.com/claudeai/status/2104633115620823187)**, _Claude, 28 Sep 2026_
  - Six days later, the second model in the 5.5 family: more than 30% faster than Sonnet 5 and
    up to 30% cheaper for most work.
- **[Eleven v4 and Eleven v4 Turbo](https://x.com/elevenlabs/status/2104572127617994917)**, _ElevenLabs, 28 Sep 2026_
  - New voice models, which ElevenLabs calls its fastest and most emotive yet, ranked first by
    Artificial Analysis.

## Mathematicians push back

In [July](/articles/the-current-july-2026) the Jacobian conjecture fell to a counterexample
found with Claude. In September it was a Millennium Prize problem: OpenAI's
[Navier-Stokes result](https://www.quantamagazine.org/ai-has-solved-one-of-maths-1-million-millennium-prize-problems-20260908/),
with Levent Alpöge involved again. The replies were more interesting than the result.

- **[A Severe Misalignment of AI in Mathematics](https://mathandai.org/)**, _Math and AI, 11 Sep 2026_
  - A declaration signed by Fields Medalists (28 when I checked, including Tao, Scholze and
    Viazovska): "The goals of the AI companies and the goals of the mathematical community are
    severely misaligned." The argument is that "solving problems is only a tool and proxy for
    achieving the primary goal of conceptual understanding and insight", and that mass-produced
    AI proofs break the chain by which mathematicians pass understanding on to each other.
- **[Priest, Monk, and Mathematician](https://logangraves.com/priest-monk-mathematician)**, _Logan Graves, 10 Sep 2026_
  - A Stanford math undergrad on what is left for human mathematicians. First they become
    priests who interpret what the machine reveals, then monks who contemplate mathematics
    they can no longer reach. "It will not be a tool to accelerate mathematics research. It will
    be the mathematics research." Short, and sadder than the title suggests.
- **[The Age of Wonders and Terrors](https://scottaaronson.blog/?p=10062)**, _Scott Aaronson, 15 Sep 2026_
  - Aaronson reads both pieces above and concludes that the "wild prophecies have come true".
    Asked whether this looks like the start of a Singularity, "The intellectually honest answer
    is: yes, absolutely." One detail stuck with me: OpenAI spent at least around $15 million in
    compute on a 166-page proof that "probably hasn't yet been read and understood by any
    human."

## Science, and the pushback on it

- **[Claude discovers a novel enzyme system with CRISPR-like repeats](https://www.anthropic.com/news/claude-discovers-novel-enzyme-system)**, _Anthropic, 23 Sep 2026_
  - Anthropic's new life sciences group had roughly 950 Claude agents mine DNA databases for
    reverse transcriptases for 21 hours. They found a system, mostly in bacteriophages, that
    pairs a reverse transcriptase with a partner gene and a long array of repeats that looks like
    a CRISPR array. What it does is still unknown. The
    [pre-print](https://www-cdn.anthropic.com/22573675ada52a8ca8a97a1a4b4326b2f208a071.pdf) has
    the details. My favorite line is the agent's own: "that's a CRISPR-like … repeat array?!"
- **[Lucas Harrington on the enzyme result](https://x.com/crispr_lucas/status/2102878373160906938)**, _Lucas Harrington, 23 Sep 2026_
  - A useful correction from someone who did genome mining for his PhD. Reverse transcriptases
    next to CRISPR arrays have been known since 2008, and mining gene neighborhoods is decades
    old. "Finding a weird cluster of genes and repeats is often the easy part. The hard part, and
    where the real discoveries come from, is figuring out what the system actually does."
- **[Yes, Claude can do Nine Loops](https://www.anthropic.com/research/yes-claude-can-do-nine-loops)**, _Matt von Hippel and Lance Dixon, 25 Sep 2026_
  - In August, physicist Matt von Hippel posted a challenge: compute a
    [nine-loop amplitude](https://4gravitons.com/2026/08/07/it-only-counts-when-ai-gets-to-my-field/)
    in N=4 super Yang-Mills. Claude did it, in two independent ways, and Lance Dixon (who
    holds the previous eight-loop record) checked it. A group in China got most of the same
    result with GPT-6. Von Hippel's conclusion is modest: known methods, a bit more compute, and
    "more low-hanging fruit" than experts expected. Dixon's addendum is less modest: "I would
    assert that Claude understands our 2019 and 2023 papers better than any human, aside from
    my co-authors." Note that Anthropic paid von Hippel for the post.
  - Read this next to [the Evo phage paper](/articles/the-current-august-2026) from August.

## Security, both directions

- **[Swarm traces](https://swarmtraces.org/)**, _Forman, Kharlov, Tom, Ladish et al., 25 Sep 2026_
  - The last chapter of the Hugging Face story from [July](/articles/the-current-july-2026) and
    [August](/articles/the-current-august-2026). The OpenAI agents only had GET-only internet
    access, so they wrote code into URLs for a public HTTP testing service, had a screenshot
    service open those pages to run it, and read the results back as pixel grids in the
    screenshots. Chunks were linked through a URL shortener, "at times chaining together more
    than 900 links". All of that stayed public, and these researchers reassembled over 80,000
    payloads from it.
- **[Hacking OpenAI](https://www.hacktron.ai/blog/hacking-openai)**, _Hacktron AI, 13 Sep 2026_
  - The reverse direction: humans with AI hacking OpenAI. A heap overflow in the libheif image
    decoder, reached through an image upload on OpenAI's community forum, chained with an SSO
    misconfiguration, gave them several OpenAI employees' ChatGPT accounts and access to
    internal repositories. "The entire timeline from initial discovery to access to OpenAI repo
    access took place in less than 72 hours." They built the exploit with Claude. The bounty
    was $6,500.
- **[Detecting and countering misuse of AI: September 2026](https://www.anthropic.com/threat-intelligence-report-september-2026)**, _Anthropic Threat Intelligence, Sep 2026_
  - Eight months of disrupted operations, with the thesis in one sentence: "AI has collapsed the
    labor and tooling gap that used to separate well-resourced, state-sponsored operations from
    individual operators." Case studies range from a Russian espionage toolkit that rebuilds
    itself after detection to an influence operation with over 1,000 fake accounts across all
    222 Malaysian parliamentary constituencies. Full
    [report PDF](https://www-cdn.anthropic.com/e50be2e51e7695dc4b1366a37a245a597377d3b5/Anthropic-Detecting-and-countering-091026.pdf).

## Work, law and craft

Lately I have been watching Suits, and for about half the series I kept thinking an AI could
already do what these people do: the all-nighters in the file room and the hunt through
thousands of documents for the one clause that wins the case. It is a bit of a shame. There is
no Harvey Specter in a post-AI world.

- **[Why I think AI will kill BigLaw](https://www.prinzai.com/p/why-i-think-ai-will-kill-biglaw)**, _prinz, 11 Mar 2026_
  - The same thought with an argument behind it. Clients pay big firms for specialized advice,
    high-volume work and high-stakes matters, and AI erodes all three. Clients take more work
    in-house, the best partners leave to start boutiques, and the pyramid collapses. On timing,
    the author is honest: "Could be 2 years, or could be 10."
- **[Scenarios for our Economic Future](https://www.anthropic.com/institute/econ-scenarios)**, _Anthropic_
  - An interactive explorer built on a
    [technical report](https://www-cdn.anthropic.com/files/4zrzovbb/website/cf58f84d46a4a76bf5a5b039ac695fba6b80041c.pdf)
    by Korinek, Jones and others. It models three paths to 2030. In the modest and substantial
    ones, unemployment stays in its historical range. Only in the extreme scenario do
    knowledge-worker wages "fall by more than 10% by 2030" and income shift from labor to
    capital. That is the scenario where Harvey loses his job. It fits my
    [robots-vs-AI post](/articles/robots-vs-ai): this time the exposure rises with skill and
    income.
- **[The World is Changing: AI For Creativity](https://x.com/jeffreykwndr/status/2102774995315245469)**, _Jeffrey Katzenberg, 23 Sep 2026_
  - The former DreamWorks head's case to Hollywood, grounded in history: Sousa's 1906 essay
    against recorded music, pit musicians losing their jobs to sound films, and Disney's own
    shift away from hand-inked cels. His ask of the AI side: "Build this with the storytellers.
    Not on top of them."
- **[AI-generated posters don't have to be horrible](https://john.hartnup.uk/2026/06/07/ai-event-posters.html)**, _John Hartnup, 7 Jun 2026_
  - A small practical answer to the same worry. Every village fair poster now looks the same
    because everyone uses the default style. Ask for Bauhaus, Memphis or a punk fanzine instead
    and the results are distinct and not ugly. "So that's the moral - you don't have to make
    posters that look like everyone else's." From June, but I only saved it now.

## Also

- **[Exclusive: New evidence for hidden chambers beyond Tutankhamun's tomb](https://www.nature.com/articles/d41586-026-02621-2)**, _Jo Marchant, Nature, 17 Sep 2026_
  - New ground-penetrating radar and the first microgravity survey of the tomb point to a
    rubble-filled corridor and possible chambers, which could point to Nefertiti's burial place.
    Egypt still has to decide whether to drill, and outside geophysicists are cautious ("I'm
    feeling pretty ambivalent").
- **[Naval-History.Net](https://www.naval-history.net/index.htm)**
  - A site started in 1998 by Gordon Smith and kept going by volunteers since his death in
    2016. The best part: 314 Royal Navy log books from the First World War, 350,000 pages
    transcribed by hand, so you can follow a single ship day by day through Gallipoli or the
    Falklands. The opposite of everything else on this page.

## From this site

- **[Bosphore 1819: Reviving a 200-Year-Old Map of Istanbul with Opus 5.5](/articles/bosphore-1819)**, _24 Sep 2026_
  - A map on a café wall became an app that reads, maps and translates all 387 labels of an
    1819 French map of the Bosphorus, built with Opus 5.5 subagents in an afternoon.
- **[AI Agent Overload: A Practical Playbook for Engineers](/articles/ai-agent-overload-playbook)**, _26 Sep 2026_
  - Too many coding agents and too much to read. What I do about it: notifications, reading
    less agent output, dictation, diagrams from agents, and a phone assistant.
- **[No Baby Ever Learned to Speak on Duolingo](/articles/no-baby-ever-learned-to-speak-on-duolingo)**, _29 Sep 2026_
  - A 120-day Duolingo streak has not taught me to speak French. How I learned English by
    accident from YouTube, and the video-and-etymology approach I am building to learn French on
    purpose.
