export type SocialIcon = 'email' | 'x' | 'linkedin' | 'github' | 'orcid' | 'wikipedia' | 'rss';

export interface Social {
  label: string;
  href: string;
  icon: SocialIcon;
}

/** Seeded from the previous site's author profile. */
export const socials: Social[] = [
  { label: 'X', href: 'https://x.com/cahidarda', icon: 'x' },
  { label: 'LinkedIn', href: 'https://www.linkedin.com/in/cahid-arda/', icon: 'linkedin' },
  { label: 'GitHub', href: 'https://github.com/CahidArda', icon: 'github' },
  { label: 'Wikipedia', href: 'https://en.wikipedia.org/wiki/User:VonArda', icon: 'wikipedia' },
  { label: 'RSS', href: '/rss.xml', icon: 'rss' },
  // Kept last so Email sits second to last in the sidebar row (the theme toggle comes after).
  { label: 'Email', href: 'mailto:cahidardaooz@gmail.com', icon: 'email' },
];
