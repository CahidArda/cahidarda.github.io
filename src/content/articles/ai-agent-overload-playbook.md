---
title: 'AI Agent Overload: A Practical Playbook for Engineers'
description: 'Too many coding agents, too many PRs, Slack at midnight. A practical playbook for engineers: agent notifications, reading less output, AI-assisted PR review, Slack VIPs, switching off, and sleep.'
date: 2026-09-26
tags: ['blog']
---

AI agents were supposed to reduce my workload. Instead, the work changed shape. I type less code,
but I now run a handful of agents in parallel, check whether each one is finished, read long
transcripts, review a stream of large pull requests, and answer Slack from people in other time
zones. By the evening I am tired in a way that has nothing to do with writing code.

I sat down and listed everything that bothered me, then researched fixes for each item. This post
is the part of that list that should apply to most engineers working with AI agents. Tool
details are current as of September 2026.

## The loop behind agent fatigue

Most of the problems feed one another:

1. Agents make it cheap to start work, so more work gets started.
2. Someone has to read, review, and merge it. That person is usually the most senior engineer.
3. The days get longer, and at home there is no natural stopping point.
4. You show up tired to the meetings where your work is judged, then work late to compensate.
5. You never recover, so you have no energy to fix the process that caused it.

You don't need to fix every link. Each section below breaks the loop at a different point, and
the cheapest ones are at the top.

## Stop polling your agents: set up notifications

Checking whether an agent is done is what drained me the most. Every major surface can notify you
now. You just have to turn it on.

**Claude Code (terminal).** Run `/config` and set the notification channel. Native desktop
notifications only work in iTerm2, Ghostty, and Kitty. Other terminals get a bell
([terminal config docs](https://code.claude.com/docs/en/terminal-config)). For a Mac notification
in any terminal, add a `Notification` hook to `~/.claude/settings.json`
([hooks guide](https://code.claude.com/docs/en/hooks-guide)):

```json
{
  "hooks": {
    "Notification": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "osascript -e 'display notification \"Claude needs you\" with title \"Claude Code\" sound name \"Glass\"'"
          }
        ]
      }
    ]
  }
}
```

A `Stop` hook works the same way if you also want "done" alerts. Be aware that `Stop` fires at the
end of every turn, which gets noisy.

**Phone push.** Install the Claude app with the same account, then in `/config` enable "Push when
actions required" (and optionally "Push when Claude decides"). Push only works for sessions with
Remote Control on, so also enable "Remote Control for all sessions"
([mobile docs](https://code.claude.com/docs/en/mobile)).

**claude.ai and the desktop app.** Allow notifications for claude.ai in your browser, and for the
Claude app in macOS or iOS notification settings.

**Scheduled and cloud runs.** Scheduled tasks in the Claude apps push to your phone when they
finish ([scheduled tasks](https://support.claude.com/en/articles/13854387-schedule-recurring-tasks-in-claude-cowork)).
Claude Code routines have no built-in completion alert, so make "post to a webhook" the routine's
last step ([routines docs](https://code.claude.com/docs/en/routines)).

The rule that kept this sane for me: always notify when an agent needs me, and only notify "done"
for long tasks. Per-tool pings bring the checking habit right back.

## Read less agent output

The second drain is reading. A few settings cut it down a lot:

- **Concise output.** Set `"outputStyle": "Concise"` in `~/.claude/settings.json`. The answer
  comes first, without narration or recap ([output styles](https://code.claude.com/docs/en/output-styles)).
- **A fixed report format.** Write a custom output style whose first line is always `Result:` and
  the second `Needs you:`. You can triage a session from two lines.
- **Subagents don't inherit output styles.** Add "report in 3 lines, result first" to your
  subagent prompts and agent definitions.
- **One view for many sessions.** `claude --bg` starts background sessions, and `claude agents`
  lists them grouped by working, needs input, and done, each with a one-line summary
  ([agent view](https://code.claude.com/docs/en/agent-view)). The Claude phone app lists cloud
  sessions and Remote Control sessions together.
- **Hear it instead.** A `Stop` hook can summarize the last message into one sentence and pass it
  to macOS `say`.
- **Cap work in progress.** If you review every result, your reading speed is the limit, not the
  number of agents. Three at a time is plenty.

## Type less, even outdoors

Dictation is a big relief for your hands, but it's awkward in a café. Whisper-mode dictation apps
(Wispr Flow, superwhisper, Willow) now pick up whispered speech. Claude Code has `/voice` for
push-to-talk dictation. The biggest win, though, is turning prompts you repeat into skills or
slash commands (`/ship`, `/fix-ci`, `/review-pr`), so one word replaces a paragraph.

## Connect GitHub, Linear and Slack without leaving tokens on disk

I avoided connecting tools to my agents because I didn't want long-lived tokens sitting in config
files. Remote MCP servers with OAuth fix most of that. Claude Code stores the tokens in the macOS
Keychain ([MCP docs](https://code.claude.com/docs/en/mcp)):

```sh
claude mcp add --scope user --transport http linear https://mcp.linear.app/mcp
claude mcp add --scope user --transport http slack https://mcp.slack.com/mcp
```

GitHub is the exception. Its hosted MCP server doesn't support OAuth in Claude Code
([issue #3433](https://github.com/anthropics/claude-code/issues/3433)), so use a fine-grained
token scoped to a few repositories, read-only where possible, with a short expiry. Connectors you
add on claude.ai also show up in Claude Code when you're signed in with the same account.

## Let agents show you diagrams, not just text

Text is a poor format for architecture, data, and UI. Here is what works today:

- **Claude Code in the terminal** doesn't render images or Mermaid inline. An image a tool returns
  stays folded inside the tool call.
- **Artifacts** open in your browser on first publish and update in place. They're the most
  reliable way for an agent to say "look at this".
- **claude.ai chat** draws interactive charts and diagrams inline
  ([custom visuals](https://support.claude.com/en/articles/13979539-custom-visuals-in-chat-and-cowork))
  and renders [MCP Apps](https://modelcontextprotocol.io/extensions/apps/overview) in the conversation.
- **Build it yourself.** A small MCP server with a `show_user(kind, src, caption)` tool can push
  images, Mermaid, and video to a local viewer page over a WebSocket. The agent chooses to share
  something, which is exactly what's missing when images are hidden in tool results. If you use the
  [herdr](https://herdr.dev) terminal multiplexer, its socket API can draw images into a pane
  (Kitty-protocol terminals only).

## PR review when everyone ships with agents

When every teammate has agents, review becomes the bottleneck. Deep-reviewing fifty open PRs isn't
a plan. Tiering them is:

| Tier   | What falls in it                                      | Review                                  |
| ------ | ----------------------------------------------------- | --------------------------------------- |
| Low    | Docs, copy, tests, UI tweaks, work behind a flag      | AI review + green CI, the author merges |
| Medium | A feature inside one product area                     | AI review + a 10-minute human skim      |
| High   | Billing, auth, data deletion, migrations, CI, infra   | Full human review (CODEOWNERS)          |

A few habits make AI review trustworthy enough for the low tier:

- **Keep a REVIEW.md with a "missed before" log.** Every time AI review misses something, add a
  rule. The reviewer doesn't remember your repository's history unless you write it down.
- **Run the review on PR open**, for example with
  [claude-code-action](https://code.claude.com/docs/en/github-actions), so it's done before a human looks.
- **Cap PR size.** Warn in CI above roughly 400 changed lines unless the PR is labeled.
- **Make the author's agent write the description:** what, why, how it was tested, the tier, and
  where to look first.
- **Spread review.** Teammates review each other's low and medium PRs. The lead takes the high tier.

## Leading a team where everyone is their own PM

With agents, a small team often splits into one person per product. That works for the products,
but nobody owns the shared middle: small fixes, upgrades, cross-cutting bugs, reviews. It
quietly falls to whoever is called the lead.

- **Make the middle visible.** Put every small fix in one tracker project and count it weekly.
  The count is also your argument for more help.
- **Keep an on-call rotation, but change the job.** The person on rotation doesn't fix things by
  hand. They triage the queue, dispatch an agent per item, and review those PRs. The point of the
  rotation is ownership, not typing.
- **Delegate by risk, not by trust.** Low-risk work can go to someone you don't fully trust yet.
  That's how the trust gets built. Ask for a five-line plan before any code.
- **Give over-builders a complexity budget.** One concern per PR, extra ideas as follow-up issues.
  Put the rule in each repository's CLAUDE.md so their agents follow it too.

## Reporting up when your work isn't flashy

A lot of agent-era work is maintenance, and nobody wants to hear about maintenance in a meeting.
Write a short update the day before instead: what shipped, one number that moved, one experiment
for next week, and one line of counts for the maintenance ("14 fixes, 22 reviews"). The meeting
becomes a discussion, and you're no longer judged on how you present at the end of a long day.

## Slack: notify only for selected people and channels

If your colleagues are in another time zone, it's tempting to check Slack in bed "in case something
important comes in". Slack can let only the important messages through:

1. **Default to mentions and DMs.** Preferences, then Notifications. Turn off thread replies and
   huddles, and remove unused keywords.
2. **Add VIPs and allow them through.** Preferences, then VIP. Check "Always allow notifications
   from VIPs". People from Slack Connect organizations count. Paid plans only
   ([VIPs](https://slack.com/help/articles/34963579361683-Add-contacts-as-VIPs)).
3. **Set a notification schedule,** for example weekdays 10:00 to 19:00. Outside it only VIPs get
   through ([pause notifications](https://slack.com/help/articles/214908388-Pause-your-Slack-notifications)).
4. **Set chosen channels to "All new posts"** and mute the rest
   ([per-channel settings](https://slack.com/help/articles/360056534254-Manage-notifications-for-specific-channels-and-direct-messages)).
5. **Turn off the app icon badge.** The red number is what pulls you in.

Slack has no strict "DMs only from these people" setting. The schedule plus VIP list is the
closest you get.

## Switching off when you work from home

Without a commute, work fades into the evening. What works is a physical act, not a decision:

- **A shutdown ritual** at a fixed time. Write down open loops and tomorrow's first task, then
  close Slack. Writing a plan for unfinished tasks reduces intrusive thoughts about them
  ([Masicampo and Baumeister, 2011](https://doi.org/10.1037/a0024192)).
- **A fake commute.** A ten-minute walk at the end of the day marks the switch from work to home
  ([HBS research on commutes as role transitions](https://www.hbs.edu/faculty/Pages/item.aspx?num=59121)).
- **The phone charges outside the bedroom.** Use a cheap alarm clock.
- **A separate macOS user for work,** with Slack and the work browser only there. Log out at
  shutdown.
- **Say the rule out loud:** "I'm off Slack after 19:00. DM me if it's urgent." Research on
  psychological detachment finds that recovery depends on actually disconnecting after hours
  ([Sonnentag, 2012](https://journals.sagepub.com/doi/abs/10.1177/0963721411434979)).

## Sleep: fix regularity before length

A six-hour average that swings between eight-hour and four-hour nights is mostly an irregularity
problem. Sleep regularity predicts health outcomes on its own, separate from total sleep
([Windred et al., 2023](https://doi.org/10.1093/sleep/zsad253)).

- **Fix one wake time, seven days a week.** Bedtime is that time minus eight hours in bed.
- **Use the iPhone sleep schedule** (Health, then Sleep) with a 45-minute wind-down. It turns on
  Sleep Focus, which also silences Slack apart from VIPs.
- **Get morning light:** ten minutes outdoors before late morning. The walk to start work counts.
- **The four-hour nights** are often revenge bedtime procrastination: staying up to reclaim free
  time the day didn't give you. The real fix is the evening stopping time, not willpower at
  midnight.

## Reading more books (and reading in bed without a lamp)

If you finish books you love but struggle to start new ones, the issue is starting, not
discipline. Put a book where the phone used to be, drop books that haven't grabbed you after 50
pages, and keep a queue of three so the next book is never a decision. Keep a calmer book for
bed. A page-turner at midnight costs sleep.

No bedside lamp or socket? You don't have to read sitting up:

- **An e-reader with a front light** (like a Kindle Paperwhite) lights itself, so any position
  works. Use warm light and dark mode at night.
- **A neck reading light** shines onto the page from just below your chin. It's rechargeable and
  works for paper books.
- **A clip-on book light** moves with the book.
- **A rechargeable gooseneck lamp** clamped or stuck magnetically to a nearby shelf, bent forward
  so the light lands in front of you instead of behind your head.

Whatever you choose, keep it warm (2700K or lower) and dim.

## Lunch without ordering in

Ordering lunch is easy to fall into when you work from home. A stocked fridge and one 30-minute
prep session a week make a 15-minute lunch easier than ordering:

- **Protein:** boneless chicken thighs (hard to overcook), thin-cut chicken breast, a couple of
  steaks, eggs. Freeze single portions.
- **Ready carbs:** a weekly pot of rice or bulgur (keeps four days), flatbread, baby potatoes.
- **No-cook sides:** salad leaves, cherry tomatoes, cucumber, yogurt, feta, olives, hummus.
- **Frozen vegetables:** green beans, broccoli, peas.
- **One tool:** an instant-read thermometer. Steak at 52 to 54 °C for medium-rare, chicken at
  74 °C or more.

Weekly prep: marinate four chicken portions (yogurt, olive oil, garlic, paprika, salt) and cook
the grains. Then lunch is chicken in the pan for 12 minutes with a grain and yogurt, or a steak for
two to three minutes a side with a leafy salad.

## A personal assistant on your phone

Once your agents can reach your tools, the same setup can help outside work:

- **Calendar and email.** The Google Workspace connectors in claude.ai work on the phone and in
  Claude Code. Note that the Calendar connector can write, with approval for each action
  ([Google connectors](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors)).
  For strictly read-only access, use Google Calendar's
  [secret iCal address](https://support.google.com/calendar/answer/37648) or your own MCP server
  that only requests `calendar.readonly`.
- **A morning brief.** A daily scheduled task that summarizes today's calendar and last night's
  sleep, capped at eight lines, pushed to your phone.
- **Apple Watch data.** Claude's native Apple Health integration is US-only at the time of writing
  ([iOS apps](https://support.claude.com/en/articles/11869619-use-claude-with-ios-apps)). Elsewhere,
  the Health Auto Export app can send workouts and metrics as JSON to your own endpoint
  ([REST export](https://help.healthyapps.dev/en/health-auto-export/automations/rest-api/)), and a
  small read-only MCP server can expose them to your assistant.
- **A training coach.** After each ride, a job can compare the workout's heart-rate trace with
  your goal and save the next session. An iOS Shortcut with the "Apple Watch Workout" trigger can
  read the plan to you when you start the next one
  ([Shortcuts triggers](https://support.apple.com/en-ca/guide/shortcuts/apd932ff833f/ios)). One
  catch if the goal is VO2max: Apple only estimates it from outdoor walks, runs, and hikes, not
  cycling ([Apple](https://support.apple.com/en-us/108790)).

Pick one hub for this, whether that's claude.ai, Grok Bot, or something you build yourself.
Two assistants mean two memories and two sets of notifications, which is the problem this post
started with.

## Where to start this week

If you only do a few things:

1. Turn on phone push for "needs input" and Remote Control for all sessions. (10 minutes)
2. Set Slack VIPs, a notification schedule, and turn off the badge. (15 minutes)
3. Switch to concise output and a `Result:` / `Needs you:` report format. (10 minutes)
4. Choose a fixed wake time and move the phone charger out of the bedroom. (tonight)
5. Spend one hour sorting open PRs into low, medium, and high. (1 hour)
6. Write your weekly update the day before the meeting. (20 minutes a week)

None of these are about working harder. They get rid of the checking, reading, and worrying that
AI agents added on top of the actual work.
