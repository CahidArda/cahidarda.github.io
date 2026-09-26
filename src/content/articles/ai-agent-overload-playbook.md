---
title: 'AI Agent Overload: A Practical Playbook for Engineers'
description: 'Too many coding agents and too much to read. A practical playbook for engineers: Claude Code notifications, reading less agent output, dictation, MCP without tokens on disk, diagrams from agents, Slack VIPs, and a phone assistant.'
date: 2026-09-26
tags: ['blog']
---

AI agents were supposed to reduce my workload. Instead, the work changed shape. I type less code,
but I now run a handful of agents in parallel, check whether each one is finished, read long
transcripts, review a stream of large pull requests, and answer Slack from people in other time
zones. By the evening I am tired in a way that has nothing to do with writing code.

I sat down and listed everything that bothered me, then researched fixes for each item. This post
is the part of that list about tooling, which should apply to most engineers working with AI agents. Tool
details are current as of September 2026.

## Where the fatigue comes from

Most of it isn't the agents themselves. It's the overhead around them:

1. **Polling.** Switching back to each session to see whether it's done.
2. **Reading.** Long transcripts, when you only needed the result.
3. **Input.** Typing or dictating the same instructions over and over.
4. **Plumbing.** Tools your agents can't reach, so you copy things in by hand.
5. **Interruptions.** Messages that don't need you right now.

Each section below removes one of these, and the cheapest fixes are at the top.

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

## A personal assistant on your phone

Once your agents can reach your tools, the same setup can help outside work:

- **Calendar and email.** The Google Workspace connectors in claude.ai work on the phone and in
  Claude Code. Note that the Calendar connector can write, with approval for each action
  ([Google connectors](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors)).
  For strictly read-only access, use Google Calendar's
  [secret iCal address](https://support.google.com/calendar/answer/37648) or your own MCP server
  that only requests `calendar.readonly`.
- **A morning brief.** A daily scheduled task that summarizes today's calendar and anything that
  needs you, capped at eight lines, pushed to your phone.
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

## Conclusion

Agents made the work faster. The tiring part is the overhead around them: checking, reading,
retyping, and being reachable all the time. Most of that comes down to settings you can change in
an afternoon. Turn on notifications, ask for shorter answers, connect your tools, and let only
the right people reach you. The attention you get back is worth more than another agent running
in parallel.
