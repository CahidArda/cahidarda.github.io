---
title: The Current - July & August 2026
description: A public record of the links I save - some of what I read in July and August.
date: 2026-08-31
tags: ['accessions']
---

I skipped July, so this one covers two months. It is shorter than
[June](/articles/the-current-june-2026), and most of it is a single story told from
three sides: the Bun rewrite.

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

## Also

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
