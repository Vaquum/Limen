import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), '..', '..');
const siteRoot = path.resolve(repoRoot, 'docs-site');
const outRoot = path.resolve(siteRoot, '.generated', 'docs');

const repoBlobBaseUrl = 'https://github.com/Vaquum/Limen/blob/main';
const repoTreeBaseUrl = 'https://github.com/Vaquum/Limen/tree/main';

const sectionCategories = [
  {
    dir: 'overview',
    label: 'Overview',
    position: 1,
    slug: '/overview',
    description: 'What Limen is, how it fits together, and where to start.',
  },
  {
    dir: 'guides',
    label: 'Guides',
    position: 2,
    slug: '/guides',
    description: 'End-to-end workflows for running research in Limen.',
  },
  {
    dir: 'reference',
    label: 'Reference',
    position: 3,
    slug: '/reference',
    description: 'Detailed interface and library reference for Limen.',
  },
  {
    dir: 'developer',
    label: 'Developer',
    position: 4,
    slug: '/developer',
    description: 'Contributor, release, and documentation maintenance guides.',
  },
  {
    dir: 'packages',
    label: 'Packages',
    position: 5,
    slug: '/packages',
    description: 'Module ownership, boundaries, and canonical entry points.',
  },
];

const docs = [
  {
    source: 'README.md',
    dest: 'index.md',
    slug: '/',
    title: 'Limen',
    sidebarLabel: 'Home',
  },
  {
    source: 'docs/README.md',
    dest: 'overview/docs-hub.md',
    slug: '/overview/docs-hub',
    sidebarPosition: 1,
  },
  {
    source: 'docs/Historical-Data.md',
    dest: 'guides/historical-data.md',
    slug: '/guides/historical-data',
    sidebarPosition: 1,
  },
  {
    source: 'docs/Data-Bars.md',
    dest: 'guides/data-bars.md',
    slug: '/guides/data-bars',
    sidebarPosition: 2,
  },
  {
    source: 'docs/Single-File-Decoder.md',
    dest: 'guides/single-file-decoder.md',
    slug: '/guides/single-file-decoder',
    sidebarPosition: 3,
  },
  {
    source: 'docs/Experiment-Manifest.md',
    dest: 'guides/experiment-manifest.md',
    slug: '/guides/experiment-manifest',
    sidebarPosition: 4,
  },
  {
    source: 'docs/Universal-Experiment-Loop.md',
    dest: 'guides/universal-experiment-loop.md',
    slug: '/guides/universal-experiment-loop',
    sidebarPosition: 5,
  },
  {
    source: 'docs/Log.md',
    dest: 'guides/log.md',
    slug: '/guides/log',
    sidebarPosition: 6,
  },
  {
    source: 'docs/Benchmark.md',
    dest: 'guides/benchmark.md',
    slug: '/guides/benchmark',
    sidebarPosition: 7,
  },
  {
    source: 'docs/Backtest.md',
    dest: 'guides/backtest.md',
    slug: '/guides/backtest',
    sidebarPosition: 8,
  },
  {
    source: 'docs/Trainer.md',
    dest: 'guides/trainer.md',
    slug: '/guides/trainer',
    sidebarPosition: 9,
  },
  {
    source: 'docs/Regime-Diversified-Opinion-Pools.md',
    dest: 'guides/regime-diversified-opinion-pools.md',
    slug: '/guides/regime-diversified-opinion-pools',
    sidebarPosition: 10,
  },
  {
    source: 'docs/Conserved-Flux-Renormalization.md',
    dest: 'guides/conserved-flux-renormalization.md',
    slug: '/guides/conserved-flux-renormalization',
    sidebarPosition: 11,
  },
  {
    source: 'docs/Indicators.md',
    dest: 'reference/indicators.md',
    slug: '/reference/indicators',
    sidebarPosition: 1,
  },
  {
    source: 'docs/Features.md',
    dest: 'reference/features.md',
    slug: '/reference/features',
    sidebarPosition: 2,
  },
  {
    source: 'docs/Transforms.md',
    dest: 'reference/transforms.md',
    slug: '/reference/transforms',
    sidebarPosition: 3,
  },
  {
    source: 'docs/Scalers.md',
    dest: 'reference/scalers.md',
    slug: '/reference/scalers',
    sidebarPosition: 4,
  },
  {
    source: 'docs/Standard-Metrics-Library.md',
    dest: 'reference/standard-metrics-library.md',
    slug: '/reference/standard-metrics-library',
    sidebarPosition: 5,
  },
  {
    source: 'docs/Developer/README.md',
    dest: 'developer/developer-home.md',
    slug: '/developer/home',
    sidebarPosition: 1,
    sidebarLabel: 'Developer Home',
  },
  {
    source: 'docs/Developer/Documentation-System.md',
    dest: 'developer/documentation-system.md',
    slug: '/developer/documentation-system',
    sidebarPosition: 2,
  },
  {
    source: 'docs/Developer/Writing-Docstrings.md',
    dest: 'developer/writing-docstrings.md',
    slug: '/developer/writing-docstrings',
    sidebarPosition: 3,
  },
  {
    source: 'docs/Developer/Contributing-Foundational-SFDs.md',
    dest: 'developer/contributing-foundational-sfds.md',
    slug: '/developer/contributing-foundational-sfds',
    sidebarPosition: 4,
  },
  {
    source: 'docs/Developer/Making-Release.md',
    dest: 'developer/making-release.md',
    slug: '/developer/making-release',
    sidebarPosition: 5,
  },
  {
    source: 'docs/Semantic-Versioning.md',
    dest: 'developer/semantic-versioning.md',
    slug: '/developer/semantic-versioning',
    sidebarPosition: 6,
  },
  {
    source: 'limen/data/README.md',
    dest: 'packages/data.md',
    slug: '/packages/data',
    sidebarPosition: 1,
    sidebarLabel: 'Data',
  },
  {
    source: 'limen/experiment/README.md',
    dest: 'packages/experiment.md',
    slug: '/packages/experiment',
    sidebarPosition: 2,
    sidebarLabel: 'Experiment',
  },
  {
    source: 'limen/sfd/README.md',
    dest: 'packages/sfd.md',
    slug: '/packages/sfd',
    sidebarPosition: 3,
    sidebarLabel: 'SFD',
  },
  {
    source: 'limen/indicators/README.md',
    dest: 'packages/indicators.md',
    slug: '/packages/indicators',
    sidebarPosition: 4,
    sidebarLabel: 'Indicators',
  },
  {
    source: 'limen/features/README.md',
    dest: 'packages/features.md',
    slug: '/packages/features',
    sidebarPosition: 5,
    sidebarLabel: 'Features',
  },
  {
    source: 'limen/transforms/README.md',
    dest: 'packages/transforms.md',
    slug: '/packages/transforms',
    sidebarPosition: 6,
    sidebarLabel: 'Transforms',
  },
  {
    source: 'limen/scalers/README.md',
    dest: 'packages/scalers.md',
    slug: '/packages/scalers',
    sidebarPosition: 7,
    sidebarLabel: 'Scalers',
  },
  {
    source: 'limen/metrics/README.md',
    dest: 'packages/metrics.md',
    slug: '/packages/metrics',
    sidebarPosition: 8,
    sidebarLabel: 'Metrics',
  },
  {
    source: 'limen/log/README.md',
    dest: 'packages/log.md',
    slug: '/packages/log',
    sidebarPosition: 9,
    sidebarLabel: 'Log',
  },
  {
    source: 'limen/cohort/README.md',
    dest: 'packages/cohort.md',
    slug: '/packages/cohort',
    sidebarPosition: 10,
    sidebarLabel: 'Cohort',
  },
  {
    source: 'limen/backtest/README.md',
    dest: 'packages/backtest.md',
    slug: '/packages/backtest',
    sidebarPosition: 11,
    sidebarLabel: 'Backtest',
  },
  {
    source: 'limen/trading/README.md',
    dest: 'packages/trading.md',
    slug: '/packages/trading',
    sidebarPosition: 12,
    sidebarLabel: 'Trading',
  },
  {
    source: 'limen/utils/README.md',
    dest: 'packages/utils.md',
    slug: '/packages/utils',
    sidebarPosition: 13,
    sidebarLabel: 'Utils',
  },
];

const mappingBySource = new Map(
  docs.map((doc) => [normalizePath(doc.source), doc])
);

function normalizePath(value) {
  return value.split(path.sep).join('/');
}

async function ensureDir(dir) {
  await fs.mkdir(dir, {recursive: true});
}

async function writeJson(filePath, value) {
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function buildFrontMatter(doc) {
  const lines = ['---'];

  if (doc.slug) {
    lines.push(`slug: ${doc.slug}`);
  }

  if (doc.title) {
    lines.push(`title: ${doc.title}`);
  }

  if (typeof doc.sidebarPosition === 'number') {
    lines.push(`sidebar_position: ${doc.sidebarPosition}`);
  }

  if (doc.sidebarLabel) {
    lines.push(`sidebar_label: ${doc.sidebarLabel}`);
  }

  if (doc.dest === 'index.md') {
    lines.push('pagination_next: null');
    lines.push('pagination_prev: null');
  }

  lines.push(`custom_edit_url: ${repoBlobBaseUrl}/${doc.source}`);
  lines.push('---', '');
  return lines.join('\n');
}

function resolveDocLink(fromSource, target) {
  if (!target || target.startsWith('http://') || target.startsWith('https://') || target.startsWith('mailto:') || target.startsWith('#')) {
    return target;
  }

  const [targetPath, targetHash] = target.split('#');

  if (!targetPath) {
    return target;
  }

  if (targetPath.startsWith('/')) {
    return target;
  }

  const resolvedSource = normalizePath(path.posix.normalize(path.posix.join(path.posix.dirname(normalizePath(fromSource)), targetPath)));
  let targetDoc = mappingBySource.get(resolvedSource);

  if (!targetDoc && !path.posix.extname(resolvedSource)) {
    targetDoc = mappingBySource.get(normalizePath(path.posix.join(resolvedSource, 'README.md')));
  }

  if (!targetDoc) {
    const repoFsPath = path.resolve(repoRoot, resolvedSource);

    if (fsSync.existsSync(repoFsPath)) {
      const repoUrlBase = fsSync.statSync(repoFsPath).isDirectory()
        ? repoTreeBaseUrl
        : repoBlobBaseUrl;
      return targetHash
        ? `${repoUrlBase}/${resolvedSource}#${targetHash}`
        : `${repoUrlBase}/${resolvedSource}`;
    }

    return target;
  }

  const currentDoc = mappingBySource.get(normalizePath(fromSource));
  const fromDest = currentDoc ? normalizePath(currentDoc.dest) : '';
  const toDest = normalizePath(targetDoc.dest);
  let relative = normalizePath(path.posix.relative(path.posix.dirname(fromDest), toDest));

  if (!relative) {
    relative = path.posix.basename(toDest);
  }

  return targetHash ? `${relative}#${targetHash}` : relative;
}

function rewriteLinks(content, fromSource) {
  return content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, target) => {
    const rewritten = resolveDocLink(fromSource, target.trim());
    return `[${label}](${rewritten})`;
  });
}

function normalizeForMdx(content) {
  return content
    .replace(/<p align="center">/g, '<div align="center">')
    .replace(/<\/p>/g, '</div>')
    .replace(/<br>/g, '<br />')
    .replace(/<hr>/g, '<hr />')
    .replace(/<img([^>]*?)(?<!\/)>/g, '<img$1 />');
}

function normalizeReferencePlaceholders(content, doc) {
  if (!doc.dest.startsWith('reference/')) {
    return content;
  }

  let out = '';
  let index = 0;
  let inFence = false;
  let inInlineCode = false;

  while (index < content.length) {
    if (content.startsWith('```', index)) {
      inFence = !inFence;
      out += '```';
      index += 3;
      continue;
    }

    if (!inFence && content[index] === '`') {
      inInlineCode = !inInlineCode;
      out += '`';
      index += 1;
      continue;
    }

    if (!inFence && !inInlineCode) {
      const match = content.slice(index).match(/^\{([a-zA-Z0-9_:+.-]+)\}/);
      if (match) {
        out += `(${match[1]})`;
        index += match[0].length;
        continue;
      }
    }

    out += content[index];
    index += 1;
  }

  return out;
}

async function copyDoc(doc) {
  const sourcePath = path.resolve(repoRoot, doc.source);
  const destPath = path.resolve(outRoot, doc.dest);
  const raw = await fs.readFile(sourcePath, 'utf8');
  const rewritten = normalizeReferencePlaceholders(
    normalizeForMdx(rewriteLinks(raw, doc.source)),
    doc
  );
  const output = `${buildFrontMatter(doc)}${rewritten}`;

  await ensureDir(path.dirname(destPath));
  await fs.writeFile(destPath, output);
}

async function writeCategoryFiles() {
  for (const category of sectionCategories) {
    const categoryPath = path.resolve(outRoot, category.dir, '_category_.json');
    await writeJson(categoryPath, {
      label: category.label,
      position: category.position,
      collapsible: true,
      collapsed: false,
      link: {
        type: 'generated-index',
        slug: category.slug,
        title: category.label,
        description: category.description,
      },
    });
  }
}

async function main() {
  await fs.rm(outRoot, {recursive: true, force: true});
  await ensureDir(outRoot);
  await writeCategoryFiles();

  for (const doc of docs) {
    await copyDoc(doc);
  }
}

await main();
