import { spawnSync } from 'node:child_process';

// Tracked 2026-09-11: image-size has no patched release (every version
// <= 2.0.2 is in range) and @docusaurus/mdx-loader requires ^2.0.2 on the
// newest Docusaurus. Both are denial-of-service parsers (ICNS, JXL, HEIF)
// that this static build only ever runs over the repository's own images.
// Retire both entries with the first image-size release above 2.0.2.
const TRACKED_ADVISORIES = new Set([
  'https://github.com/advisories/GHSA-w3rx-r6r6-pgpr',
  'https://github.com/advisories/GHSA-5p2g-fcmc-qvqq',
]);

const result = spawnSync('npm', ['audit', '--omit=dev', '--json'], {
  cwd: process.cwd(),
  encoding: 'utf8',
});

if (!result.stdout) {
  process.stderr.write(result.stderr || 'npm audit produced no JSON output\n');
  process.exit(result.status || 1);
}

const report = JSON.parse(result.stdout);
const vulnerabilities = report.vulnerabilities || {};

function reachesTrackedAdvisory(name, seen = new Set()) {
  if (seen.has(name)) {
    return false;
  }
  seen.add(name);

  const vulnerability = vulnerabilities[name];
  if (!vulnerability) {
    return false;
  }

  for (const via of vulnerability.via || []) {
    if (typeof via === 'string') {
      if (reachesTrackedAdvisory(via, seen)) {
        return true;
      }
      continue;
    }
    if (via.url && TRACKED_ADVISORIES.has(via.url)) {
      return true;
    }
  }

  return false;
}

const untracked = Object.keys(vulnerabilities).filter(
  name => !reachesTrackedAdvisory(name),
);

if (untracked.length > 0) {
  process.stderr.write(`Untracked docs-site npm vulnerabilities: ${untracked.join(', ')}\n`);
  process.exit(1);
}

const count = Object.keys(vulnerabilities).length;
if (count > 0) {
  process.stdout.write(`Tracked docs-site npm vulnerabilities: ${count}\n`);
  process.exit(0);
}

process.stdout.write('No docs-site npm vulnerabilities found\n');
