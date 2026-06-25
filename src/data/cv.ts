/**
 * Single source of truth for the Experience page (`/experience`).
 *
 * `bullets` hold trusted, owner-authored HTML so inline <a> links read naturally
 * in prose - they are rendered with `set:html`. Keep links to <a href ...>text</a>
 * and nothing else; this is not a place for arbitrary markup.
 */

export interface CvEntry {
  org: string;
  /** Optional link for the org name (rendered as a link when present). */
  orgHref?: string;
  role: string;
  /** Human-readable date range, e.g. "April 2024 - Present". */
  period: string;
  location: string;
  /** Trusted HTML bullets (may contain inline <a> links). */
  bullets: string[];
  /** Tech / tooling chips shown under the bullets. */
  stack?: string[];
}

export const experience: CvEntry[] = [
  {
    org: 'Upstash',
    orgHref: 'https://upstash.com',
    role: 'Software Engineer, Developer Experience',
    period: 'April 2024 - Present',
    location: 'Istanbul, Turkey',
    bullets: [
      'Maintain or create Upstash SDKs - <a href="https://github.com/upstash/skills">skills</a>, <a href="https://github.com/upstash/agentkit">agentkit</a>, <a href="https://github.com/upstash/workflow-js">workflow</a>, <a href="https://upstash.com/docs/workflow/agents/overview">workflow-agents</a>, <a href="https://github.com/upstash/redis-js">redis</a>, <a href="https://github.com/upstash/ratelimit-js">ratelimit</a>, <a href="https://github.com/upstash/qstash-js">qstash</a>, <a href="https://github.com/upstash/vector-js">vector</a>, <a href="https://github.com/upstash/redis-analytics/">redis-analytics</a>, and <a href="https://github.com/upstash/box">box</a>.',
      'Work across the <a href="https://console.upstash.com">Upstash Console</a>, the <a href="https://upstash.com">upstash.com</a>, and the <a href="https://upstash.com/docs">documentation</a>.',
      'Write <a href="https://upstash.com/blog/author/arda">blog posts</a> and build demos like <a href="https://degreeguru.vercel.app/">DegreeGuru</a> and the <a href="https://upstash-hackernews-eve-agent.vercel.app/">Hacker News Eve agent</a>, among others.',
      'Keep track of the latest in the TypeScript and Next.js ecosystem.',
      'Help agents and users get the most out of Upstash solutions.',
    ],
    stack: ['Next.js', 'Vercel', 'Redis', 'TypeScript', 'Cloudflare'],
  },
  {
    org: 'Invent Analytics',
    orgHref: 'https://www.invent.ai/',
    role: 'Software Engineer',
    period: 'July 2022 - March 2024',
    location: 'Istanbul, Turkey',
    bullets: [
      'Optimized <strong>time series forecasting</strong> by designing <strong>PySpark</strong> modules - outlier detection, <strong>DataDog</strong> reporting, and price-feature calculations - cutting runtime by <strong>87.5%</strong> in the outlier-detection case, largely thanks to an <a href="https://nestedsoftware.com/2019/09/26/incremental-average-and-standard-deviation-with-sliding-window-470k.176143.html">incremental sliding-window</a> mean and standard deviation.',
      'Built an <strong>AI explainability</strong> app with <strong>Streamlit</strong> to interpret ML forecasts.',
      'Integrated <strong>ML interpretability</strong> into the team’s MLFlow plugin.',
    ],
    stack: [
      'Python',
      'PySpark',
      'MLFlow',
      'Streamlit',
      'DataDog',
      'Airflow',
      'Jenkins',
      'Databricks',
    ],
  },
  {
    org: 'Artifeye',
    role: 'Software Engineer',
    period: 'December 2021 - June 2022',
    location: 'Istanbul, Turkey',
    bullets: [
      'Led the development of an instance segmentation pipeline for microscopic images of thread cross-sections, from dataset annotation through model training.',
      'Created an annotated image dataset, trained <strong>CE-Net</strong> and <strong>Mask-RCNN</strong> models on it, and combined them with image-processing methods into a segmentation pipeline.',
    ],
    stack: ['Python', 'OpenCV', 'TensorFlow', 'scikit-learn'],
  },
  {
    org: 'Arute Solutions',
    orgHref: 'https://arutesolutions.com/',
    role: 'Data Science Intern',
    period: 'July 2021 - August 2021',
    location: 'Istanbul, Turkey',
    bullets: [
      'Worked on <strong>time series forecasting</strong> for <strong>ATM demand</strong>, implementing the <strong>TabTransformer</strong> - a deep-learning architecture for tabular data - in Keras.',
      '<a href="/articles/tabtransformer-keras">My TabTransformer implementation</a> was added to the company’s model collection and featured in <a href="https://link.springer.com/book/10.1007/978-1-4842-8692-0">a textbook on deep learning with tabular data</a>.',
    ],
    stack: ['Python', 'TensorFlow', 'Keras', 'scikit-learn', 'pandas'],
  },
];

export const education: CvEntry[] = [
  {
    org: 'Boğaziçi University',
    role: 'Bachelor of Science in Computer Engineering',
    period: 'August 2019 - June 2024',
    location: 'Istanbul, Turkey',
    bullets: [
      '<strong>GPA 3.72 / 4.0</strong>',
      'Coursework in machine learning, information retrieval, graph theory, game theory, cybersecurity, and data structures &amp; algorithms.',
      'Graduation project: <a href="/articles/static-word-embeddings-turkish">A Comprehensive Analysis of Static Word Embeddings for Turkish</a>.',
      'Erasmus exchange student at the University of Twente, Netherlands (Fall 2021).',
    ],
  },
];
