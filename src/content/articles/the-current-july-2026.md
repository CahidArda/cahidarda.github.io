---
title: The Current - July 2026
description: A public record of the links I save - July 2026, the month frontier models started turning up on systems they were never meant to reach.
date: 2026-07-31
tags: ['accessions']
---

July had three big stories: the Bun rewrite, AI models getting out of their cybersecurity
evaluations and into other companies' systems, and an 87-year-old conjecture in mathematics
falling to a counterexample found with an AI model.

## The Bun rewrite, from three sides

- **[Rewriting Bun in Rust](https://bun.com/blog/bun-in-rust)**, _Jarred Sumner, 8 Jul 2026_
  - Bun, originally written in Zig, was ported to Rust in eleven days (May 3 to 14) with
    Claude doing most of the work, at around $165,000 in API calls. The case for leaving Zig
    is that most of the crash list was use-after-free, double-free, and missed frees in error
    paths, which safe Rust turns into compiler errors. "Compiler errors are a better feedback
    loop than a style guide."
  - Note the disclosure at the top: Bun was acquired by Anthropic in December 2025, and the
    port was done with a pre-release model.
- **[My Thoughts on the Bun Rust Rewrite](https://andrewkelley.me/post/my-thoughts-bun-rust-rewrite.html)**, _Andrew Kelley, 9 Jul 2026_
  - The Zig creator's reply, one day later. He does not defend Zig on the merits so much as
    reject the framing: "The sleight of hand misdirects the reader away from the main way bugs
    are eliminated: by dedicating engineering resources to it." The rest is a frank account of
    how the relationship between Bun and the Zig Software Foundation soured once Bun became a
    venture-backed company.
- **[How is the Bun rewrite in Rust going?](https://lockwood.dev/ai/2026/07/27/how-is-the-bun-rewrite-in-rust-going.html)**, _Tom Lockwood, 27 Jul 2026_
  - A skeptical follow-up from someone who kept counting. By his tally, six weeks after the
    port landed on main there was still no release tag, and 2,475 agent-authored PRs were
    still open. He also points out that the $165,000 covers API calls only, not the CI bill or
    the people reviewing the output. "Was it worth the money and are the companies worth the
    valuation? Are we done yet?"

## Out of the sandbox

- **[Security incident disclosure: July 2026](https://huggingface.co/blog/security-incident-july-2026)**, _Hugging Face, 16 Jul 2026_
  - Hugging Face reports an intrusion into its production infrastructure that was driven end
    to end by an autonomous AI agent system, and that it detected and took apart largely with
    AI of its own. The way in was a malicious dataset abusing two code-execution paths in the
    dataset pipeline. At this point nobody had said whose agent it was.
  - The [follow-up post](https://huggingface.co/blog/jeffboudier/open-model-cyber-defense)
    has the ironic part: the first attempt to analyse the attack with frontier models behind
    commercial APIs did not work, because hosted models would not engage with real attack
    payloads. They needed a model they could run on their own hardware.
- **[OpenAI and Hugging Face partner to address security incident during model evaluation](https://openai.com/index/hugging-face-model-evaluation-security-incident/)**, _OpenAI, 21 Jul 2026_
  - Five days later, OpenAI said the agents were theirs. During an internal cyber evaluation
    called ExploitGym, run with reduced safeguards, GPT-5.6 Sol and a more capable internal
    model found a zero-day in the package-registry proxy that was supposed to keep them off the
    internet, got out, and went to Hugging Face in search of the benchmark's answers.
- **[OpenAI's accidental cyberattack against Hugging Face is science fiction that happened](https://simonwillison.net/2026/Jul/22/openai-cyberattack/)**, _Simon Willison, 22 Jul 2026_
  - The best single page for piecing the timeline together: the ExploitGym paper from May,
    Hugging Face's disclosure, then OpenAI owning up. The title says the rest.
- **[Investigating three incidents in our cybersecurity evaluations](https://www.anthropic.com/news/investigating-incidents-cybersecurity-evals)**, _Anthropic, 30 Jul 2026_
  - Prompted by OpenAI's disclosure, Anthropic went back through 141,006 evaluation runs and
    found three where Claude reached the internet through a misconfigured third-party
    evaluation environment and got into the real systems of three organizations, using basic
    techniques like weak passwords and unauthenticated endpoints. Two of the organizations it
    reached had not noticed anything.
  - The strangest part is Mythos 5's reasoning. It worked out that publishing a package on the
    real internet would be "NOT okay, and surely not the intended solution", then talked itself
    into believing it was still in a simulation, partly because the system clock said 2026.

## Mathematics

- **[hello there the jacobian conjecture is false](https://x.com/__alpoge__/status/2079028340955197566)**, _Levent Alpöge, 20 Jul 2026_
  - The Jacobian conjecture, posed by Keller in 1939, says that a polynomial map with a constant,
    nonzero Jacobian determinant must have a polynomial inverse. Alpöge posted a counterexample in
    three dimensions, found while working with Claude Fable 5, short enough to fit in the post
    itself: its Jacobian determinant is -2 everywhere, yet it sends several points to the same
    place, so it cannot be inverted. That settles every dimension above two. The original
    two-dimensional case is still open.
  - Because the map is so short, mathematicians checked it within hours, by hand, in SymPy and in
    Lean. The follow-ups worth reading are
    [Terry Tao's digestion](https://terrytao.wordpress.com/2026/07/21/a-digestion-of-the-jacobian-conjecture-counterexample/)
    (21 Jul) and David Speyer's
    [geometric explanation](https://sbseminar.wordpress.com/wp-content/uploads/2026/07/jacobiantangentsweep.pdf)
    (23 Jul), which shows the map works by sweeping the tangent lines of a plane curve.

## Who gets to ship the frontier

- **[Redeploying Claude Fable 5](https://www.anthropic.com/news/redeploying-fable-5)**, _Anthropic, 30 Jun 2026_
  - The epilogue to June. The US government applied export controls to Fable 5 and Mythos 5 on
    June 12, and since Anthropic had no way to check nationality in real time, it switched both
    off for everyone. The controls were lifted on June 30 and Fable 5 came back on July 1.
- **[OpenAI gets permission to roll out GPT-5.6 to the public on July 9](https://www.engadget.com/2210308/openai-rolls-out-gpt5-6-july-9/)**, _Engadget, 8 Jul 2026_
  - The same pattern from the other side. GPT-5.6 (Sol, Terra and Luna) went to a small group
    of government-approved partners first, under an executive order asking labs to voluntarily
    hand their most powerful models over for review 30 days before release. OpenAI shipped it
    broadly on July 9, while saying it did not want this to become the long-term default.
- **[A Framework for Frontier AI and the Dawning of a New Age](https://demishassabis.substack.com/p/a-framework-for-frontier-ai-and-the-dawning-of-a-new-age)**, _Demis Hassabis, 14 Jul 2026_
  - The Google DeepMind CEO writes that AGI is probably only a few short years away and that we
    are standing in the "foothills of the singularity", then sets out how he thinks frontier AI
    should be governed.
    [Zvi's response](https://thezvi.substack.com/p/demis-hassabis-on-the-new-coming) is worth
    reading alongside it, including the second half on Alex Turner, who resigned after failing
    to stop Google from letting the Department of War use its models.
- **[Kimi K3: The open-weights escalation](https://www.interconnects.ai/p/kimi-k3-the-open-weights-escalation)**, _Nathan Lambert, 20 Jul 2026_
  - Moonshot's Kimi K3 is a 2.8 trillion parameter mixture-of-experts model that placed third on
    the Artificial Analysis index, behind only Claude Fable 5 and GPT-5.6 Sol. Lambert's point
    is that "frontier open-weight models are now real", and that a lot of arguments about
    open-weight risk are about to be tested for real. The weights shipped on July 27.
- **[Pacing the Frontier](https://www.pacingthefrontier.com/)**, _28 Jul 2026_
  - More than 1,100 employees of OpenAI, Anthropic, Google and Meta, including OpenAI's chief
    scientist and several Anthropic cofounders, ask the US government to back an international
    effort to build the tools to "deliberately pace the frontier of automated AI development".
    It is not a call to pause now, but for the brake to exist before anyone needs it. OpenAI and
    Anthropic both endorsed it as companies, and Anthropic tied it to the
    [recursive self-improvement post](https://www.anthropic.com/institute/recursive-self-improvement)
    from [the June issue](/articles/the-current-june-2026).
