import { z } from 'astro/zod';

/**
 * Build-time GitHub repo fetch. Unauthenticated is fine for builds (60 req/hr);
 * if a GITHUB_TOKEN is present we use it for a higher rate limit. Every failure
 * mode (offline, rate-limited, unexpected shape) resolves to an empty list so a
 * build never breaks on this - the Projects section simply hides.
 */

const GITHUB_USER = 'CahidArda';

// Repos owned by others worth featuring (e.g. work I did at Upstash). Pulled in
// individually alongside my own repos.
const EXTRA_REPOS = [
  'upstash/redis-analytics',
  'upstash/workflow-js',
  'upstash/degree-guru',
  'upstash/box',
  'upstash/workflow-agents-js',
  'upstash/agent-analytics',
  'upstash/eve-example',
];

// Lenient: many repos lack a description, homepage, or language.
const RepoSchema = z.object({
  name: z.string(),
  full_name: z.string().optional(),
  owner: z.object({ login: z.string() }).optional(),
  html_url: z.string().url(),
  description: z.string().nullable().optional(),
  stargazers_count: z.number().optional(),
  language: z.string().nullable().optional(),
  fork: z.boolean().optional(),
  archived: z.boolean().optional(),
  created_at: z.string(),
});

type Repo = z.infer<typeof RepoSchema>;

export interface Project {
  name: string;
  description?: string;
  url: string;
  stars: number;
  language?: string;
  createdAt: Date;
}

// Fetch once per build and reuse - the layout, index, and palette all ask for repos.
let cache: Promise<Project[]> | null = null;

function buildHeaders(): Record<string, string> {
  const token = process.env.GITHUB_TOKEN ?? import.meta.env.GITHUB_TOKEN;
  const headers: Record<string, string> = {
    Accept: 'application/vnd.github+json',
    'User-Agent': `${GITHUB_USER}.github.io-build`,
    'X-GitHub-Api-Version': '2022-11-28',
  };
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

// Repos not owned by me show "owner/name" so the attribution is clear.
function toProject(r: Repo): Project {
  const ownedByMe = !r.owner || r.owner.login === GITHUB_USER;
  return {
    name: ownedByMe ? r.name : (r.full_name ?? r.name),
    description: r.description ?? undefined,
    url: r.html_url,
    stars: r.stargazers_count ?? 0,
    language: r.language ?? undefined,
    createdAt: new Date(r.created_at),
  };
}

async function fetchOwnRepos(headers: Record<string, string>): Promise<Repo[]> {
  try {
    const res = await fetch(
      `https://api.github.com/users/${GITHUB_USER}/repos?sort=created&direction=desc&per_page=100`,
      { headers },
    );
    if (!res.ok) {
      console.warn(`[github] repos fetch failed: ${res.status} ${res.statusText}`);
      return [];
    }
    const parsed = z.array(RepoSchema).safeParse(await res.json());
    if (!parsed.success) {
      console.warn('[github] unexpected repos payload shape');
      return [];
    }
    return parsed.data.filter((r) => !r.fork && !r.archived);
  } catch (err) {
    console.warn('[github] repos fetch error:', err instanceof Error ? err.message : err);
    return [];
  }
}

async function fetchRepo(full: string, headers: Record<string, string>): Promise<Repo | null> {
  try {
    const res = await fetch(`https://api.github.com/repos/${full}`, { headers });
    if (!res.ok) {
      console.warn(`[github] ${full} fetch failed: ${res.status}`);
      return null;
    }
    const parsed = RepoSchema.safeParse(await res.json());
    return parsed.success ? parsed.data : null;
  } catch (err) {
    console.warn(`[github] ${full} fetch error:`, err instanceof Error ? err.message : err);
    return null;
  }
}

async function fetchAllProjects(): Promise<Project[]> {
  const headers = buildHeaders();
  const [own, extra] = await Promise.all([
    fetchOwnRepos(headers),
    Promise.all(EXTRA_REPOS.map((full) => fetchRepo(full, headers))),
  ]);

  return [...own, ...extra.filter((r): r is Repo => r !== null)]
    .map(toProject)
    .sort((a, b) => b.createdAt.valueOf() - a.createdAt.valueOf());
}

export async function getProjects(limit = 6): Promise<Project[]> {
  cache ??= fetchAllProjects();
  return (await cache).slice(0, limit);
}

export const githubProfileUrl = `https://github.com/${GITHUB_USER}`;
