import type { TagKey } from './tags';

/**
 * Posts published on external sites (e.g. the Upstash blog). These show up in the
 * content list as link-out cards - no local page. To add one, append a single entry
 * below; no new file needed. (`tags` defaults to ['blog'].)
 */
export interface ExternalPost {
  title: string;
  description: string;
  /** ISO date, e.g. '2025-10-05'. */
  date: string;
  url: string;
  source: string;
  tags?: TagKey[];
}

export const externalPosts: ExternalPost[] = [
  {
    title: 'Fast, Cost-Effective MCPs with Redis',
    description: 'How to back a Model Context Protocol server with Redis for low-latency, cheap tool calls.',
    date: '2025-10-05',
    url: 'https://upstash.com/blog/mcp-with-redis',
    source: 'Upstash',
  },
  {
    title: 'Storing Time Series Data in Redis',
    description: 'Patterns for modeling, writing, and querying time-series data on Redis without a dedicated TSDB.',
    date: '2025-09-03',
    url: 'https://upstash.com/blog/redis-timeseries',
    source: 'Upstash',
  },
  {
    title: 'Caching Drizzle Queries with Upstash Redis',
    description: 'Add a Redis cache layer to Drizzle ORM queries for faster reads in serverless apps.',
    date: '2025-07-10',
    url: 'https://upstash.com/blog/drizzle-integration',
    source: 'Upstash',
  },
  {
    title: 'Introducing Workflow Agents',
    description: 'Durable, long-running AI agents built on Upstash Workflow.',
    date: '2025-02-17',
    url: 'https://upstash.com/blog/workflow-agents',
    source: 'Upstash',
  },
  {
    title: 'Four Ways to Reduce Your Vercel Serverless Costs',
    description: 'Practical tactics to cut Vercel function spend without rewriting your app.',
    date: '2024-10-07',
    url: 'https://upstash.com/blog/vercel-cost',
    source: 'Upstash',
  },
  {
    title: 'Upstash Ratelimit in LangChain',
    description: 'Rate-limit LangChain apps and agents using Upstash Ratelimit.',
    date: '2024-06-10',
    url: 'https://upstash.com/blog/ratelimit-langchain',
    source: 'Upstash',
  },
  {
    title: 'DegreeGuru: Build a RAG Chatbot',
    description: 'Building a retrieval-augmented chatbot over university data with Upstash Vector.',
    date: '2024-03-05',
    url: 'https://upstash.com/blog/degree-guru',
    source: 'Upstash',
  },
];
