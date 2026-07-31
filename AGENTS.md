# Repository Guide

## Runtime and package manager

- This is an Astro site. `package.json` pins `pnpm@10.33.4`, but the available
  pnpm may fail while trying to self-switch to that exact version.
- Set `npm_config_manage_package_manager_versions=false` for **every** pnpm
  command. Passing `--config.manage-package-manager-versions=false` does not
  help because pnpm attempts the switch before parsing command-line flags.
- Node.js must satisfy the declared `>=22.12.0` engine requirement.

## Common commands

```sh
npm_config_manage_package_manager_versions=false pnpm install --frozen-lockfile
npm_config_manage_package_manager_versions=false pnpm dev --port 3000 --host
npm_config_manage_package_manager_versions=false pnpm build
npm_config_manage_package_manager_versions=false pnpm check
npm_config_manage_package_manager_versions=false pnpm format
```

- There is no test script. Use the targeted command above to validate a change.
- `pnpm check` can report existing hints: deprecated Zod `.url()` usage in
  `src/content.config.ts` and `src/lib/github.ts`, plus unused variables in
  `ClusterMap.tsx` and `OnboardFanout.tsx`. Do not fix them incidentally.

## Development server

- Use `--host`; Astro otherwise binds only to localhost, which makes it
  unreachable through external browser tooling.
- Confirm the server with:

  ```sh
  curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/
  ```

- Before restarting, ensure an old Astro process is actually stopped. A second
  server started on port 3000 can fail to bind while stale content continues to
  be served. If needed, inspect with `ps aux | grep astro` and stop the old
  process before launching the new one.
- For HTML text checks, use `curl -H 'Accept-Encoding: identity'` so compressed
  response bytes do not interfere with `grep`.
- An initial-request React "Invalid hook call" warning has been observed in
  this Astro/React combination. It is noise when the page still returns 200 and
  renders correctly.

## Site layout

- `src/pages/` contains routes, including the landing page, articles,
  experience page, and 404 page.
- `src/components/` contains Astro and React UI; `ArticleCard.astro` renders
  project cards on `/articles`.
- `src/data/profile.ts` drives profile/sidebar content and the landing lede.
- `src/content/articles/` holds MDX articles.
- `src/lib/entries.ts` merges local articles, external posts, and GitHub
  repositories into the articles list.
- `src/lib/github.ts` fetches project repositories. Build-time fetch failures
  intentionally resolve to an empty project list rather than failing the build.

## GitHub project descriptions

- GitHub API descriptions can be `null`. Project-card descriptions must be
  checked in the `ArticleCard.astro` output on `/articles`, not in the command
  palette: palette repository subtitles are the literal `GitHub` label.
- The intended fallback design is a `DESCRIPTION_OVERRIDES: Record<string,
  string>` in `src/lib/github.ts`, keyed by `owner/repo`. In `toProject()`,
  derive the full name as follows:

  ```ts
  const fullName = r.full_name ?? `${r.owner?.login ?? GITHUB_USER}/${r.name}`;
  description: r.description ?? DESCRIPTION_OVERRIDES[fullName] ?? undefined;
  ```

  This ensures a real GitHub description always wins.
- This checkout currently has no `DESCRIPTION_OVERRIDES` map. Verify the
  implementation before relying on or extending that fallback behavior.
- When verifying a description override on a branch or main, inspect the
  actual file directly, for example:

  ```sh
  git fetch origin
  git show origin/main:src/lib/github.ts | grep DESCRIPTION_OVERRIDES
  ```

  A no-match exits with status 1, so do not put follow-up checks behind `&&` if
  they must still run.
